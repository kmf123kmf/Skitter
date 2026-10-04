"""The matching pipeline: regions + tile library + settings -> a mosaic.

See the package docstring for the steps. `Matcher` keeps candidate sets and
search indexes between runs (keyed by library version and settings), so
changing reuse rules or tint reuses them; changing the crop settings or the
library rebuilds only what depends on them.

Good-enough search: before searching every region of a shape, a sample of
regions is also searched exactly (every candidate). If the approximate
results cost noticeably more (relative regret above REGRET_LIMIT), search
effort is raised and the sample checked again. After assignment, adaptive
passes search harder only for the regions the quality check finds worst,
keeping a pass only if the overall score improves and it places no more
regions against the reuse rules (the score doesn't see repetition).

The result remembers each region's best few candidates (candidates.py) for
manual picks. Pins (manual picks kept from an earlier run) are placed
first and never moved; everything else is matched around them.

Previews: `run(preview=...)` is handed a MatchPreview at each point where
the run has something to show (see PreviewStage), copied so it stays valid
while the run goes on. Previews only read the run's state: results are the
same with or without them. `run(detail=...)` reports progress within the
long steps (building candidates and indexes, searching, widening), and
cancelling stops those too.

Widening: after the first assignment, regions whose candidates the reuse
rules used up are searched again. Such regions usually share their favorite
photo (a large plain area: every region wants the same few photos), so they
are grouped by it. Each group is searched once, from its average target,
skipping photos at their use limit, as deep as the group needs (about
WIDEN_SLACK times the photos it must use, counting each photo's crops, up to
WIDEN_MAX candidates). Each member
gets an interleaved slice of that list (members m apart share none), reranked
for its own target, so a plain area spreads over many photos instead of
exhausting the same ones again. Adaptive passes search their worst regions
again the same way.

Searches run in batches of regions sized to take about SEARCH_SECONDS each,
in a shuffled order so previews fill in all over the mosaic. Each region's
search is independent, so batching never changes results. Small batches cost
throughput, more inside the app than in a plain script (measured at search
effort 1024, 300,000 photos: 512 regions +47%, 1,024 +16%, 2,048 +9%,
4,096 none), hence MIN_BATCH.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import faiss
import numpy as np

from skitter.core.color import oklab_to_srgb
from skitter.core.matching.assign import Assignment
from skitter.core.matching.candidates import KEEP, Candidates, Pins
from skitter.core.matching.index import (
    CandidateSet,
    SearchIndex,
    build_candidates,
    exact_best,
    region_weights,
    rerank,
)
from skitter.core.matching.quality import QualityReport, evaluate, target_raster
from skitter.core.matching.raster import rasterize
from skitter.core.matching.settings import MatchSettings
from skitter.core.matching.targets import target_descriptors
from skitter.core.slicing import RegionSet, SliceContext
from skitter.core.tiles.crops import aspect_classes
from skitter.core.tiles.descriptors import MEAN, cell_colors
from skitter.core.tiles.library import TileLibrary

REGRET_LIMIT = 0.03  # acceptable mean extra cost of approximate search, relative
REGRET_SAMPLES = 128
MAX_EFFORT = 1024
TINT_REBUILD = 0.15  # rebuild an index when the tint moves this far from its reference
WORST_SHARE = 0.1  # share of regions re-searched in each adaptive pass
GAMUT_DE = 0.06  # a region's color is "missing" from the library beyond this ΔE (OKLab)
SEARCH_SECONDS = 3.0  # aim for search batches about this long (progress and previews)
MIN_BATCH, FIRST_BATCH, MAX_BATCH = 2048, 2048, 65_536  # regions per search batch
WIDEN_SLACK = 2.0  # widening: photos searched per photo a group of regions needs
WIDEN_MAX = 4096  # widening: deepest search for one group
UNLIMITED_USES = 8  # widening: photos a group may reuse this often when uses are unlimited

Progress = Callable[[str, float | None], None]


class PreviewStage(Enum):
    """What a preview's tiles are, in the order a run reaches them."""

    SKETCH = "Target colors (no tiles yet)"
    BEST = "Best tiles, before reuse rules"  # each region's own best: repeats unlimited
    ASSIGNED = "Assigned, some regions waiting"  # reuse rules left these without a tile
    # (waiting regions show their favorite, the BEST tile, until widening replaces it)
    WIDENING = "Widening: tentative tiles"  # waiting regions: cheapest tile allowed so far
    COMPLETE = "Every region has a tile"
    REFINING = "Refining: swapping tiles"
    ADAPTIVE = "Adaptive pass: worst regions"  # searched again, harder


class MatchCancelled(Exception):
    pass


@dataclass(frozen=True)
class RegretStats:
    samples: int
    mean: float  # mean extra cost of the approximate best over the exact best
    p95: float
    relative: float  # mean regret / median exact cost
    effort: int  # search effort (nprobe) finally used
    exact: bool  # the index searched exactly (small library)


@dataclass
class MatchResult:
    regions: RegionSet
    tile: np.ndarray  # (R,) library slot, -1 for regions without a tile (hidden)
    rect: np.ndarray  # (R, 4) crop window, fractions of the tile image
    mirrored: np.ndarray  # (R,) bool
    cost: np.ndarray  # (R,) matching cost (inf: none)
    tile_mean: np.ndarray  # (R, 3) OKLab average of the chosen crop
    tint_target: np.ndarray  # (R, 3) OKLab color each tile is tinted toward
    tint: float
    quality: QualityReport
    regret: list[RegretStats]
    stats: dict = field(default_factory=dict)
    # For manual picks (None in results made without them):
    candidates: Candidates | None = None
    choice: np.ndarray | None = None  # (R,) int32 shown candidate, index into its class's set
    manual: np.ndarray | None = None  # (R,) bool: picked by hand
    settings: MatchSettings | None = None  # the settings matched with

    @property
    def manual_count(self) -> int:
        return 0 if self.manual is None else int(self.manual.sum())

    def tinted_mean(self) -> np.ndarray:
        return self.tile_mean + self.tint * (self.tint_target - self.tile_mean)

    def tint_offset(self) -> np.ndarray:
        """(R, 3) sRGB shift (in [-1, 1]) that tints each tile's average as chosen."""
        return (oklab_to_srgb(self.tinted_mean()) - oklab_to_srgb(self.tile_mean)).astype(
            np.float32
        )


@dataclass(frozen=True)
class MatchPreview:
    """A look at a run in progress."""

    stage: PreviewStage
    regions: RegionSet
    target: np.ndarray  # (R, 3) sRGB (0..1) each region aims for; nan: hidden, no tile needed
    result: MatchResult | None = None  # tiles so far (tile -1: none yet; quality None)


class Matcher:
    def __init__(self, library: TileLibrary):
        self.library = library
        self._candidates: dict[tuple, CandidateSet] = {}
        self._indexes: dict[tuple, SearchIndex] = {}

    def candidates(
        self, aspect: float, settings: MatchSettings, progress=lambda message, fraction: None
    ) -> tuple[tuple, CandidateSet]:
        key = (self.library.version, round(float(aspect), 4), settings.crops, settings.mirror)
        if key not in self._candidates:
            cands = build_candidates(self.library, aspect, settings.crops, settings.mirror,
                                     progress)  # fmt: skip
            self._candidates = {k: v for k, v in self._candidates.items() if k[0] == key[0]}
            self._candidates[key] = cands
        return key, self._candidates[key]

    def index(
        self, key: tuple, cands: CandidateSet, settings: MatchSettings,
        progress=lambda message, fraction: None,
    ) -> SearchIndex:  # fmt: skip
        weights = settings.weights()
        full = (key, tuple(np.round(weights, 6)))
        index = self._indexes.get(full)
        if index is None or abs(index.tint_ref - settings.tint_value) > TINT_REBUILD:
            index = SearchIndex(cands, weights, settings.tint_value, progress=progress)
            self._indexes = {k: v for k, v in self._indexes.items() if k[0][0] == key[0]}
            self._indexes[full] = index
        return index

    def run(
        self,
        regions: RegionSet,
        ctx: SliceContext,
        settings: MatchSettings,
        progress: Progress = lambda message, fraction: None,
        cancelled: Callable[[], bool] = lambda: False,
        pins: Pins | None = None,
        preview: Callable[[MatchPreview], None] | None = None,
        detail: Progress | None = None,
    ) -> MatchResult:
        timings: dict[str, float] = {}
        clock = [time.perf_counter()]

        def step(message: str, fraction: float | None = None, timing: str | None = None):
            if cancelled():
                raise MatchCancelled()
            now = time.perf_counter()
            if timing:
                timings[timing] = timings.get(timing, 0.0) + now - clock[0]
            clock[0] = now
            progress(message, fraction)

        def sub_step(message: str, fraction: float) -> None:  # within a long step
            if cancelled():
                raise MatchCancelled()
            if detail is not None:
                detail(message, fraction)

        def counter(message: str, total: int):
            """A found(rows) callback reporting how many of total regions are done."""
            done = [0]

            def found(rows) -> None:
                done[0] += len(rows)
                sub_step(f"{message}: {done[0]:,} of {total:,} regions", done[0] / max(total, 1))

            return found

        n = len(regions)
        tint = settings.tint_value
        k = settings.candidates
        kw = 2 * k  # room for wider searches
        step("Analyzing regions", 0.0)
        targets = target_descriptors(regions, ctx, rasterize(regions))
        need = np.flatnonzero(targets.needed)
        target_rgb = np.full((n, 3), np.nan, np.float32)
        target_rgb[need] = np.clip(oklab_to_srgb(targets.desc[need][:, MEAN]), 0, 1)

        def show(stage: PreviewStage, ref=None, cost=None, tint_target=None) -> None:
            if preview is None:
                return
            result = None
            if ref is not None:
                tint_target = targets.desc[:, MEAN] if tint_target is None else tint_target
                result = self._snapshot(regions, ref, cost, tint_target, shapes, cls, tint)
            preview(MatchPreview(stage, regions, target_rgb, result))

        def show_assigned(stage: PreviewStage) -> None:
            if preview is not None:
                show(stage, self._chosen(assign, cand_ref), assign.chosen_costs(),
                     assign.tint_target)  # fmt: skip

        show(PreviewStage.SKETCH)
        weights = region_weights(settings.weights(), targets.mask, tint)
        classes, cls = (
            aspect_classes(regions.size[:, 0] / regions.size[:, 1])
            if n
            else (np.zeros(0), np.zeros(0, np.int64))
        )
        step("Preparing tiles", 0.05, "regions")

        cand_ref = np.full((n, kw), -1, np.int64)  # candidate index within its class
        costs = np.full((n, kw), np.inf, np.float32)
        shapes: dict[int, tuple[CandidateSet, SearchIndex, np.ndarray]] = {}
        regret: list[RegretStats] = []
        efforts: dict[int, int] = {}
        searched = [0]  # regions searched so far
        for c, aspect in enumerate(classes):
            regs = need[cls[need] == c]
            if not len(regs):
                continue
            step(f"Indexing tiles for {aspect:.3g}:1 regions", 0.05 + 0.25 * c / len(classes))
            key, cands = self.candidates(aspect, settings, sub_step)
            if not len(cands):
                continue
            index = self.index(key, cands, settings, sub_step)
            shapes[c] = (cands, index, regs)
            step("Checking search accuracy", None, "index")
            effort, stats = self._calibrate(cands, index, regs, targets.desc, weights, settings,
                                            sub_step)  # fmt: skip
            regret.append(stats)
            efforts[c] = effort
            step("Searching", 0.3 + 0.3 * c / len(classes), "calibrate")

            def found(rows, ids, cs, regs=regs):
                cand_ref[rows, :k], costs[rows, :k] = ids, cs
                searched[0] += len(rows)
                sub_step(f"Searching: {searched[0]:,} of {len(need):,} regions",
                         searched[0] / max(len(need), 1))  # fmt: skip
                show(PreviewStage.BEST, np.where(np.isfinite(costs[:, 0]), cand_ref[:, 0], -1),
                     costs[:, 0])  # fmt: skip

            self._search(cands, index, regs, targets.desc, weights, settings, k, effort, found)
        step("Assigning tiles", 0.6, "search")

        if not shapes:
            raise ValueError("no tiles fit these regions (empty library?)")
        pinned, pin_ref, pin_cost = self._pin(pins, shapes, cls, need, targets, weights, settings,
                                              cand_ref, costs)  # fmt: skip
        # Each region's favorite: previews keep showing it until the region gets a tile.
        favorite = np.where(np.isfinite(costs[:, 0]), cand_ref[:, 0], -1)
        favorite_cost = costs[:, 0].copy()
        show(PreviewStage.BEST, favorite, favorite_cost)
        free = need[~np.isin(need, pinned)]
        tiles = np.full((n, kw), -1, np.int64)
        cand_mean = np.zeros((n, kw, 3), np.float32)
        for cands, _, regs in shapes.values():
            ref = cand_ref[regs]
            safe = np.maximum(ref, 0)
            tiles[regs] = np.where(ref >= 0, cands.tile[safe], -1)
            cand_mean[regs] = cands.desc[safe][..., MEAN].astype(np.float32)
        tiles[~np.isfinite(costs)] = -1

        area = regions.area() * targets.visible
        importance = area / max(float(area[need].mean()) if len(need) else 1.0, 1e-12)
        tile_w, tile_h = ctx.tile_size
        spacing = settings.min_spacing * math.sqrt(tile_w * tile_h)
        assign = Assignment(
            tiles, costs, regions.center, len(self.library.status), settings.max_uses, spacing
        )
        assign.force(pinned)  # first, whatever the rules say
        order = self._priority(free, costs, importance)
        if settings.error_diffusion:
            reading = self._reading_order(regions, free)
            self._diffuse(assign, regions, reading, cand_mean, targets, settings)
        else:
            assign.greedy(order)
        unplaced = need[assign.choice[need] < 0]
        if len(unplaced):  # widen the search for regions every candidate of which is taken
            # Placed tiles, and the waiting regions' favorites until widening finds better.
            tentative = self._chosen(assign, cand_ref)
            tentative_cost = assign.chosen_costs()
            waiting = tentative < 0
            tentative[waiting], tentative_cost[waiting] = favorite[waiting], favorite_cost[waiting]
            show(PreviewStage.ASSIGNED, tentative, tentative_cost, assign.tint_target)
            step("Widening search", 0.65, "assign")
            widened = counter("Widening search", len(unplaced))

            def widening(rows):
                widened(rows)
                if preview is None:
                    return
                j = assign.first_allowed(rows)  # else the best: it would be forced
                j = np.where(j >= 0, j, 0)
                found = np.isfinite(assign.costs[rows, j])  # else keep the favorite
                tentative[rows[found]] = cand_ref[rows[found], j[found]]
                tentative_cost[rows[found]] = assign.costs[rows[found], j[found]]
                show(PreviewStage.WIDENING, tentative, tentative_cost, assign.tint_target)

            self._widen(shapes, cls, unplaced, targets, weights, settings, kw, efforts, assign,
                        cand_ref, cand_mean, widening)  # fmt: skip
            assign.greedy(self._priority(unplaced, assign.costs, importance))
        rest = need[assign.choice[need] < 0]
        assign.force(self._priority(rest, assign.costs, importance))
        forced = np.zeros(n, bool)  # placed against the rules (none to spare)
        forced[rest] = assign.choice[rest] >= 0
        show_assigned(PreviewStage.COMPLETE)

        if not settings.error_diffusion:
            deadline = time.perf_counter() + settings.refine_seconds
            total = float(np.nansum(np.where(np.isfinite(assign.chosen_costs()),
                                             assign.chosen_costs(), 0) * importance))  # fmt: skip
            while time.perf_counter() < deadline:
                step("Refining", 0.7, "assign")
                gain = assign.refine_pass(order, importance)
                show_assigned(PreviewStage.REFINING)
                if gain <= 1e-6 * max(total, 1e-12):
                    break

        step("Checking quality", 0.8, "refine")

        def quality(target=None):
            return self._quality(regions, ctx, assign, shapes, cls, cand_ref, targets, tint,
                                 target)  # fmt: skip

        report, target = quality()
        best = (report.score, assign.choice.copy(), assign.tiles.copy(), assign.costs.copy(),
                cand_ref.copy(), cand_mean.copy(), forced.copy())  # fmt: skip
        for round_ in range(settings.adaptive_rounds):
            step(f"Adaptive pass {round_ + 1}", 0.8 + 0.15 * round_ / settings.adaptive_rounds)
            err = np.where(np.isnan(report.region_error), -np.inf, report.region_error)[free]
            count = max(1, int(WORST_SHARE * len(free)))
            worst = free[np.argsort(-err * importance[free])[:count]]
            self._widen(shapes, cls, worst, targets, weights, settings, kw, efforts, assign,
                        cand_ref, cand_mean, counter("Searching worst regions again", len(worst)),
                        boost=4 * (round_ + 1))  # fmt: skip
            assign.unplace(worst)
            assign.greedy(self._priority(worst, assign.costs, importance))
            rest = worst[assign.choice[worst] < 0]
            assign.force(self._priority(rest, assign.costs, importance))
            forced[worst] = False
            forced[rest] = assign.choice[rest] >= 0
            if not settings.error_diffusion:
                assign.refine_pass(order, importance)
            show_assigned(PreviewStage.ADAPTIVE)
            report, target = quality(target)
            # Kept only if better and within the rules as much as before (the score
            # doesn't see reuse, so it could trade rule breaks for a sliver of quality).
            if report.score < best[0] - 1e-3 and forced.sum() <= best[6].sum():
                improvement = (best[0] - report.score) / max(best[0], 1e-9)
                best = (report.score, assign.choice.copy(), assign.tiles.copy(),
                        assign.costs.copy(), cand_ref.copy(), cand_mean.copy(),
                        forced.copy())  # fmt: skip
                if improvement < 0.005:
                    break
            else:
                break
        if not np.array_equal(best[1], assign.choice):  # the last pass made things worse
            _, choice, tiles, costs, cand_ref[:], cand_mean[:], forced = best
            tint_target = assign.tint_target
            assign = Assignment(tiles, costs, regions.center, len(self.library.status),
                                settings.max_uses, spacing)  # fmt: skip
            assign.tint_target = tint_target
            assign.set_choices(choice)
            report, target = quality(target)
        step("Done", 1.0, "quality")

        result = self._result(regions, assign, shapes, cls, cand_ref, targets, tint, report, regret)
        result.settings = settings.copy()
        result.candidates = self._remember(classes, shapes, cls, cand_ref, assign, pinned, pin_ref,
                                           pin_cost)  # fmt: skip
        result.choice = self._chosen(assign, cand_ref).astype(np.int32)
        result.manual = np.zeros(n, bool)
        result.manual[pinned] = True
        over, close = assign.check()
        chosen = result.tile[result.tile >= 0]
        result.stats.update(
            regions=n,
            placed=int(len(chosen)),
            hidden=int(n - len(need)),
            unique_tiles=int(len(np.unique(chosen))),
            most_uses=int(np.bincount(chosen).max()) if len(chosen) else 0,
            rule_violations=int(over + close),
            forced=int(forced.sum()),
            manual=int(len(pinned)),
            candidates=int(sum(len(c) for c, _, _ in shapes.values())),
            gamut_gap=self._gamut_gap(shapes, targets, need, area),
            timings=timings,
        )
        return result

    # Steps

    def _pin(self, pins, shapes, cls, need, targets, weights, settings, cand_ref, costs):
        """Make each pinned region's only candidate its pinned crop.

        Crops the class's candidate set lacks are added to it. Returns the
        pinned regions and their searched (ranked) lists, still offered for
        picking later.
        """
        empty = np.zeros(0, np.int64)
        if pins is None or not len(pins):
            return empty, np.zeros((0, KEEP), np.int64), np.zeros((0, KEEP), np.float32)
        keep = np.isin(pins.region, need) & np.isin(cls[pins.region], list(shapes))
        pins = pins.subset(keep)
        listed = cand_ref[pins.region, :KEEP].copy(), costs[pins.region, :KEEP].copy()
        for c, (cands, index, regs) in list(shapes.items()):
            mine = np.flatnonzero(cls[pins.region] == c)
            if not len(mine):
                continue
            rows = pins.region[mine]
            cands, ref = cands.with_crops(self.library, pins.tile[mine], pins.rect[mine],
                                          pins.mirrored[mine])  # fmt: skip
            shapes[c] = (cands, index, regs)
            _, cost = rerank(cands, ref[:, None], targets.desc[rows], weights[rows],
                             settings.crop_penalty)  # fmt: skip
            cand_ref[rows], costs[rows] = -1, np.inf
            cand_ref[rows, 0], costs[rows, 0] = ref, cost[:, 0]
        return pins.region.astype(np.int64), *listed

    def _remember(self, classes, shapes, cls, cand_ref, assign, pinned, pin_ref, pin_cost):
        """The best KEEP candidates of each region, for manual picks."""
        n = len(cand_ref)
        ref = cand_ref[:, :KEEP].astype(np.int32)
        cost = assign.costs[:, :KEEP].astype(np.float32)
        if ref.shape[1] < KEEP:
            pad = ((0, 0), (0, KEEP - ref.shape[1]))
            ref = np.pad(ref, pad, constant_values=-1)
            cost = np.pad(cost, pad, constant_values=np.inf)
        cost[ref < 0] = np.inf
        auto = self._chosen(assign, cand_ref).astype(np.int32)
        auto_cost = assign.chosen_costs().astype(np.float32)
        if len(pinned):
            ref[pinned], cost[pinned] = pin_ref, pin_cost
            auto[pinned], auto_cost[pinned] = ref[pinned, 0], cost[pinned, 0]
        sets = tuple(shapes[c][0] if c in shapes else None for c in range(len(classes)))
        return Candidates(sets, cls.astype(np.int16) if n else np.zeros(0, np.int16), ref, cost,
                          auto, auto_cost)  # fmt: skip

    @staticmethod
    def _priority(regs, costs, importance) -> np.ndarray:
        """Large, visible regions first, and those that lose most from a second choice."""
        if not len(regs):
            return regs
        best = costs[regs, 0]
        second = np.where(np.isfinite(costs[regs, 1]), costs[regs, 1], best)
        finite = best[np.isfinite(best)]
        scale = max(float(np.median(finite)) if len(finite) else 1.0, 1e-9)
        gap = np.where(np.isfinite(best), second - np.where(np.isfinite(best), best, 0), 0)
        key = importance[regs] * (1 + gap / scale)
        return regs[np.argsort(-np.nan_to_num(key, nan=0.0), kind="stable")]

    @staticmethod
    def _reading_order(regions: RegionSet, regs) -> np.ndarray:
        cell = float(np.median(regions.size[regs].min(axis=1))) if len(regs) else 1.0
        row = np.floor(regions.center[regs, 1] / cell)
        return regs[np.lexsort((regions.center[regs, 0], row))]

    def _diffuse(self, assign, regions, order, cand_mean, targets, settings) -> int:
        cell = float(np.median(regions.size[order].min(axis=1)))
        pos = np.floor(regions.center / cell).astype(np.int64)
        pos -= pos[order].min(axis=0)
        pos = np.clip(pos, 0, None)
        shape = (int(pos[:, 1].max()) + 1, int(pos[:, 0].max()) + 1)
        mean_weight = settings.color_weight * targets.mask[:, 0]
        return assign.greedy_diffused(
            order, cand_mean, targets.desc[:, MEAN], mean_weight, settings.tint_value,
            pos[:, ::-1], shape,
        )  # fmt: skip

    def _calibrate(self, cands, index, regs, desc, weights, settings, report=None):
        """Raise search effort until sampled approximate results are close to exact."""
        rng = np.random.default_rng(0)
        sample = rng.choice(regs, min(len(regs), REGRET_SAMPLES), replace=False)
        _, exact = exact_best(cands, desc[sample], weights[sample], settings.crop_penalty)
        effort = start = settings.search_effort
        while True:
            if report is not None and not index.exact:
                share = math.log(effort / start) / max(math.log(MAX_EFFORT / start), 1e-9)
                report(f"Trying search effort {effort:,} (at most {MAX_EFFORT:,})",
                       min(max(share, 0.0), 1.0))  # fmt: skip
            ids = index.search(desc[sample], settings.candidates, effort)
            _, cs = rerank(cands, ids, desc[sample], weights[sample], settings.crop_penalty)
            approx = np.where(np.isfinite(cs[:, 0]), cs[:, 0], exact)
            gap = np.maximum(approx - exact, 0)
            scale = max(float(np.median(exact)), 1e-9)
            relative = float(gap.mean() / scale)
            if index.exact or relative <= REGRET_LIMIT or effort >= MAX_EFFORT:
                return effort, RegretStats(
                    len(sample), float(gap.mean()), float(np.percentile(gap, 95)), relative,
                    effort, index.exact,
                )  # fmt: skip
            effort = min(MAX_EFFORT, effort * 4)

    @staticmethod
    def _batches(count: int):
        """Slices of range(count) sized to take about SEARCH_SECONDS each (measured on the
        work the caller does between them)."""
        lo, batch = 0, FIRST_BATCH
        while lo < count:
            hi = min(count, lo + batch)
            start = time.perf_counter()
            yield slice(lo, hi)
            per_item = (time.perf_counter() - start) / (hi - lo)
            batch = int(np.clip(SEARCH_SECONDS / max(per_item, 1e-9), MIN_BATCH, MAX_BATCH))
            lo = hi

    @classmethod
    def _search(cls, cands, index, regs, desc, weights, settings, k, effort, found=None):
        """(len(regs), k) candidates and costs of each region, best first (padded with -1
        and inf). Searched in batches (see the module doc), shuffled; after each one,
        found(rows, ids, cs) gets its regions and their results."""
        ids = np.full((len(regs), k), -1, np.int64)
        cs = np.full((len(regs), k), np.inf, np.float32)
        kk = min(k, len(cands))
        order = np.random.default_rng(len(regs)).permutation(len(regs))
        for part in cls._batches(len(order)):
            pos = order[part]
            r = regs[pos]
            ids[pos, :kk], cs[pos, :kk] = rerank(
                cands, index.search(desc[r], kk, effort), desc[r], weights[r], settings.crop_penalty
            )
            if found is not None:
                found(r, ids[pos], cs[pos])
        return ids, cs

    @staticmethod
    def _replacer(cands, assign, cand_ref, cand_mean, found=None):
        """replace(rows, ids, cs): give regions new candidate lists, then call found(rows)."""

        def replace(rows, ids, cs):
            cand_ref[rows], assign.costs[rows] = ids, cs
            safe = np.maximum(ids, 0)
            assign.tiles[rows] = np.where((ids >= 0) & np.isfinite(cs), cands.tile[safe], -1)
            cand_mean[rows] = cands.desc[safe][..., MEAN].astype(np.float32)
            if found is not None:
                found(rows)

        return replace

    def _research(self, shapes, cls, regs, targets, weights, settings, k, efforts, boost,
                  assign, cand_ref, cand_mean, found=None):  # fmt: skip
        """Search regions again, wider and harder, replacing their candidate lists.
        found(rows): after each batch, with its regions (their lists already replaced)."""
        for c, (cands, index, _) in shapes.items():
            r = regs[cls[regs] == c]
            if not len(r):
                continue
            assign.unplace(r[assign.choice[r] >= 0])
            effort = min(MAX_EFFORT, efforts[c] * boost)
            replace = self._replacer(cands, assign, cand_ref, cand_mean, found)
            self._search(cands, index, r, targets.desc, weights, settings, k, effort, replace)

    def _widen(self, shapes, cls, regs, targets, weights, settings, k, efforts, assign,
               cand_ref, cand_mean, found=None, boost=4):  # fmt: skip
        """New candidate lists for regions (unplaced first), searched boost times harder
        and grouped by their favorite photo (see the module doc). found(rows): after
        each batch, with its regions."""
        per_photo = settings.max_uses if settings.max_uses > 0 else UNLIMITED_USES
        rng = np.random.default_rng(0)
        for c, (cands, index, _) in shapes.items():
            r = regs[cls[regs] == c]
            if not len(r):
                continue
            assign.unplace(r[assign.choice[r] >= 0])
            effort = min(MAX_EFFORT, efforts[c] * boost)
            exclude = None
            if settings.max_uses > 0:  # photos at their limit can't take any of these
                full = np.flatnonzero(assign.uses >= settings.max_uses)
                exclude = np.flatnonzero(np.isin(cands.tile, full)) if len(full) else None
            # Groups: regions whose favorite is the same photo (or alone, without one).
            best = cand_ref[r, 0]
            key = np.where(best >= 0, cands.tile[np.maximum(best, 0)], -1 - np.arange(len(r)))
            _, group, size = np.unique(key, return_inverse=True, return_counts=True)
            order = np.argsort(group, kind="stable")
            members = np.split(r[order], np.cumsum(size)[:-1])
            query = np.zeros((len(size), targets.desc.shape[1]), np.float64)
            np.add.at(query, group, targets.desc[r])
            query = (query / size[:, None]).astype(np.float32)
            crops = len(cands) / max(len(np.unique(cands.tile)), 1)  # candidates per photo
            need = np.ceil(WIDEN_SLACK * size / per_photo * crops)
            depth = np.clip(need, k, WIDEN_MAX).astype(int)
            depth = np.minimum(k * 2 ** np.ceil(np.log2(depth / k)).astype(int), WIDEN_MAX)
            replace = self._replacer(cands, assign, cand_ref, cand_mean, found)
            for d in np.unique(depth):
                groups = np.flatnonzero(depth == d)
                groups = groups[rng.permutation(len(groups))]  # previews fill in all over
                for part in self._batches(len(groups)):
                    batch = groups[part]
                    found_ids = index.search(query[batch], int(d), effort, exclude)
                    rows, lists = [], []
                    for g, ids in zip(batch, found_ids, strict=True):
                        ids = ids[ids >= 0]
                        # Slices by photo (a photo's crops together), in turn down the
                        # ranking: no two slices share a photo.
                        slices = max(1, -(-len(ids) // k))
                        _, first, photo = np.unique(cands.tile[ids], return_index=True,
                                                    return_inverse=True)  # fmt: skip
                        rank = np.empty(len(first), np.int64)
                        rank[np.argsort(first, kind="stable")] = np.arange(len(first))
                        part_of = rank[photo] % slices
                        who = members[g][rng.permutation(len(members[g]))]
                        for i, row in enumerate(who):
                            mine = ids[part_of == i % slices][:k]
                            lists.append(np.pad(mine, (0, k - len(mine)), constant_values=-1))
                            rows.append(row)
                    rows = np.asarray(rows, np.int64)
                    ids, cs = rerank(cands, np.asarray(lists, np.int64), targets.desc[rows],
                                     weights[rows], settings.crop_penalty)  # fmt: skip
                    replace(rows, ids, cs)

    @staticmethod
    def _chosen(assign, cand_ref) -> np.ndarray:
        """Per region: index of the chosen candidate within its class (-1: none)."""
        ref = np.full(len(assign.choice), -1, np.int64)
        rows = np.flatnonzero(assign.choice >= 0)
        ref[rows] = cand_ref[rows, assign.choice[rows]]
        return ref

    def _quality(self, regions, ctx, assign, shapes, cls, cand_ref, targets, tint, target):
        ref = self._chosen(assign, cand_ref)
        cells = np.zeros((len(regions), 4, 4, 3), np.float32)
        placed = ref >= 0
        tint_target = (
            assign.tint_target if assign.tint_target is not None else targets.desc[:, MEAN]
        )
        for c, (cands, _, _) in shapes.items():
            rows = np.flatnonzero(placed & (cls == c))
            if not len(rows):
                continue
            desc = cands.desc[ref[rows]].astype(np.float32)
            mean = desc[:, MEAN]
            shift = tint * (tint_target[rows] - mean)
            cells[rows] = cell_colors(desc) + shift[:, None, None, :]
        if target is None:
            report = evaluate(regions, ctx, cells, placed)
            ny, nx = report.extra["raster_shape"]
            return report, target_raster(ctx, nx, ny)
        return evaluate(regions, ctx, cells, placed, target), target

    def _result(self, regions, assign, shapes, cls, cand_ref, targets, tint, report, regret):
        tint_target = (
            assign.tint_target if assign.tint_target is not None else targets.desc[:, MEAN]
        )
        return self._snapshot(regions, self._chosen(assign, cand_ref), assign.chosen_costs(),
                              tint_target, shapes, cls, tint, report, regret)  # fmt: skip

    @staticmethod
    def _snapshot(regions, ref, cost, tint_target, shapes, cls, tint, report=None, regret=None):
        """A result showing candidate ref[r] in each region r (-1: none), every array
        copied (the run may go on changing its own)."""
        n = len(regions)
        tile = np.full(n, -1, np.int64)
        rect = np.zeros((n, 4), np.float32)
        mirrored = np.zeros(n, bool)
        mean = np.zeros((n, 3), np.float32)
        for c, (cands, _, _) in shapes.items():
            rows = np.flatnonzero((ref >= 0) & (cls == c))
            i = ref[rows]
            tile[rows], rect[rows], mirrored[rows] = cands.tile[i], cands.rect[i], cands.mirrored[i]
            mean[rows] = cands.desc[i][:, MEAN].astype(np.float32)
        return MatchResult(
            regions=regions, tile=tile, rect=rect, mirrored=mirrored,
            cost=np.array(cost, np.float32), tile_mean=mean,
            tint_target=np.array(tint_target, np.float32), tint=tint, quality=report,
            regret=[] if regret is None else regret,
        )  # fmt: skip

    @staticmethod
    def _gamut_gap(shapes, targets, need, area) -> float:
        """Share of the visible area whose average color no tile comes close to."""
        means = np.concatenate([c.desc[:, MEAN] for c, _, _ in shapes.values()]).astype(np.float32)
        if len(means) > 200_000:
            means = means[np.random.default_rng(0).choice(len(means), 200_000, replace=False)]
        index = faiss.IndexFlatL2(3)
        index.add(np.ascontiguousarray(means))
        d2, _ = index.search(np.ascontiguousarray(targets.desc[need, MEAN]), 1)
        missing = np.sqrt(np.maximum(d2[:, 0], 0)) > GAMUT_DE
        total = float(area[need].sum())
        return float(area[need][missing].sum() / total) if total > 0 else 0.0

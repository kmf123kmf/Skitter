"""One matching run, phase by phase (Matcher.run drives it; see matcher.py).

`MatchRun` holds a run's state (targets, candidates, the assignment) and
does one phase per method: analyze, search, pin, assign (with widening),
refine, adapt, result. `Reporter` is how a run talks to the outside:
progress steps (each checks for cancelling), progress within a step, and
previews of the run so far.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import faiss
import numpy as np

from skitter.core.color import oklab_to_srgb
from skitter.core.matching.assign import Assignment
from skitter.core.matching.candidates import KEEP, Candidates, Pins
from skitter.core.matching.index import (
    CandidateSet,
    SearchIndex,
    exact_best,
    region_weights,
    rerank,
)
from skitter.core.matching.quality import evaluate, target_raster
from skitter.core.matching.raster import rasterize
from skitter.core.matching.result import (
    MatchCancelled,
    MatchPreview,
    MatchResult,
    PreviewStage,
    Progress,
    RegretStats,
)
from skitter.core.matching.settings import MatchSettings
from skitter.core.matching.targets import (
    FULL_DETAIL,
    detail_levels,
    level_weights,
    target_descriptors,
)
from skitter.core.slicing import RegionSet, SliceContext
from skitter.core.tiles.crops import aspect_classes
from skitter.core.tiles.descriptors import MEAN, cell_colors

REGRET_LIMIT = 0.03  # acceptable mean extra cost of approximate search, relative
REGRET_SAMPLES = 128
REGRET_GAIN = 0.75  # more effort is kept only if it cuts regret to this share or less
WORST_SHARE = 0.1  # share of regions re-searched in each adaptive pass
GAMUT_DE = 0.06  # a region's color is "missing" from the library beyond this ΔE (OKLab)
SEARCH_SECONDS = 3.0  # aim for search batches about this long (progress and previews)
MIN_BATCH, FIRST_BATCH, MAX_BATCH = 2048, 2048, 65_536  # regions per search batch
WIDEN_SLACK = 2.0  # widening: photos searched per photo a group of regions needs
WIDEN_MAX = 4096  # widening: deepest search for one group
UNLIMITED_USES = 8  # widening: photos a group may reuse this often when uses are unlimited


class Reporter:
    """Progress, cancelling and previews for one run."""

    def __init__(
        self,
        progress: Progress,
        cancelled: Callable[[], bool],
        detail: Progress | None,
        preview: Callable[[MatchPreview], None] | None,
    ):
        self._progress, self._cancelled = progress, cancelled
        self._detail, self._preview = detail, preview
        self.timings: dict[str, float] = {}
        self._clock = time.perf_counter()

    @property
    def previewing(self) -> bool:
        return self._preview is not None

    def step(self, message: str, fraction: float | None = None, timing: str | None = None):
        """A new step; the time since the last one is added to timings[timing]."""
        self.check()
        now = time.perf_counter()
        if timing:
            self.timings[timing] = self.timings.get(timing, 0.0) + now - self._clock
        self._clock = now
        self._progress(message, fraction)

    def detail(self, message: str, fraction: float) -> None:
        """Progress within a long step."""
        self.check()
        if self._detail is not None:
            self._detail(message, fraction)

    def counter(self, message: str, total: int) -> Callable:
        """A found(rows) callback reporting how many of total regions are done."""
        done = [0]

        def found(rows) -> None:
            done[0] += len(rows)
            self.detail(f"{message}: {done[0]:,} of {total:,} regions", done[0] / max(total, 1))

        return found

    def preview(self, item: MatchPreview) -> None:
        if self._preview is not None:
            self._preview(item)

    def check(self) -> None:
        if self._cancelled():
            raise MatchCancelled()


def search_batches(count: int):
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


def batched_search(cands, index, regs, desc, weights, settings, k, effort, found=None):
    """(len(regs), k) candidates and costs of each region, best first (padded with -1
    and inf). Searched in batches (see matcher.py), shuffled; after each one,
    found(rows, ids, cs) gets its regions and their results."""
    ids = np.full((len(regs), k), -1, np.int64)
    cs = np.full((len(regs), k), np.inf, np.float32)
    kk = min(k, len(cands))
    order = np.random.default_rng(len(regs)).permutation(len(regs))
    for part in search_batches(len(order)):
        pos = order[part]
        r = regs[pos]
        ids[pos, :kk], cs[pos, :kk] = rerank(
            cands, index.search(desc[r], kk, effort), desc[r], weights[r], settings.crop_penalty
        )
        if found is not None:
            found(r, ids[pos], cs[pos])
    return ids, cs


@dataclass(frozen=True)
class Search:
    """How one shape class's regions of one detail level are searched."""

    index: SearchIndex
    effort: int  # calibrated search effort (IVF cells per query)


class MatchRun:
    """One run of the pipeline; call the phases in order (see Matcher.run)."""

    def __init__(self, matcher, regions: RegionSet, ctx: SliceContext, settings: MatchSettings,
                 pins: Pins | None, report: Reporter):  # fmt: skip
        self.matcher, self.library = matcher, matcher.library
        self.regions, self.ctx, self.settings, self.pins = regions, ctx, settings, pins
        self.report = report
        self.n = len(regions)
        self.tint = settings.tint_value
        self.k = settings.candidates
        self.kw = 2 * self.k  # room for wider searches
        # Candidates per shape class: c -> (candidate set, regions of that class)
        self.shapes: dict[int, tuple[CandidateSet, np.ndarray]] = {}
        # Searches per shape class and detail level (targets.detail_levels): (c, level) -> Search
        self.searches: dict[tuple[int, int], Search] = {}
        self.regret: list[RegretStats] = []

    # Previews

    def show(self, stage: PreviewStage, ref=None, cost=None, tint_target=None) -> None:
        """Preview candidate ref[r] in each region r (-1: none; no ref: target colors)."""
        if not self.report.previewing:
            return
        result = None
        if ref is not None:
            tint_target = self.targets.desc[:, MEAN] if tint_target is None else tint_target
            result = self._snapshot(ref, cost, tint_target)
        self.report.preview(MatchPreview(stage, self.regions, self.target_rgb, result))

    def show_assigned(self, stage: PreviewStage) -> None:
        if self.report.previewing:
            assign = self.assignment
            self.show(stage, self._chosen(), assign.chosen_costs(), assign.tint_target)

    # Phases

    def analyze(self) -> None:
        """Describe each region of the image; group the regions by shape."""
        report, n, regions = self.report, self.n, self.regions
        report.step("Analyzing regions", 0.0)
        self.targets = targets = target_descriptors(regions, self.ctx, rasterize(regions))
        self.need = np.flatnonzero(targets.needed)
        self.target_rgb = np.full((n, 3), np.nan, np.float32)
        self.target_rgb[self.need] = np.clip(oklab_to_srgb(targets.desc[self.need][:, MEAN]), 0, 1)
        self.show(PreviewStage.SKETCH)
        self.weights = region_weights(self.settings.weights(), targets.mask, self.tint)
        self.level = detail_levels(targets.mask)
        self.classes, self.cls = (
            aspect_classes(regions.size[:, 0] / regions.size[:, 1])
            if n
            else (np.zeros(0), np.zeros(0, np.int64))
        )
        report.step("Preparing tiles", 0.05, "regions")

    def search(self) -> None:
        """Each region's best candidates, per shape class and detail level: index,
        calibrate, search."""
        report, settings, need, cls, k = self.report, self.settings, self.need, self.cls, self.k
        self.cand_ref = np.full((self.n, self.kw), -1, np.int64)  # candidate index in its class
        self.costs = np.full((self.n, self.kw), np.inf, np.float32)
        searched = [0]

        def found(rows, ids, cs):
            self.cand_ref[rows, :k], self.costs[rows, :k] = ids, cs
            searched[0] += len(rows)
            report.detail(f"Searching: {searched[0]:,} of {len(need):,} regions",
                          searched[0] / max(len(need), 1))  # fmt: skip
            if report.previewing:
                best = np.where(np.isfinite(self.costs[:, 0]), self.cand_ref[:, 0], -1)
                self.show(PreviewStage.BEST, best, self.costs[:, 0])

        for c, aspect in enumerate(self.classes):
            regs = need[cls[need] == c]
            if not len(regs):
                continue
            progress = 0.05 + 0.25 * c / len(self.classes)
            report.step(f"Indexing tiles for {aspect:.3g}:1 regions", progress)
            key, cands = self.matcher.candidates(aspect, settings, report.detail)
            if not len(cands):
                continue
            self.shapes[c] = (cands, regs)
            for level in np.unique(self.level[regs])[::-1]:  # full detail first
                mine = regs[self.level[regs] == level]
                if level != FULL_DETAIL:
                    report.step(f"Indexing tiles for {aspect:.3g}:1 regions (less detail)",
                                progress)  # fmt: skip
                weights = level_weights(settings.weights(), int(level))
                index = self.matcher.index(key, cands, settings, report.detail, weights)
                report.step("Checking search accuracy", None, "index")
                effort, stats = self._calibrate(cands, index, mine)
                self.regret.append(stats)
                self.searches[c, int(level)] = Search(index, effort)
                report.step("Searching", 0.3 + 0.3 * c / len(self.classes), "calibrate")
                batched_search(cands, index, mine, self.targets.desc, self.weights, settings, k,
                               effort, found)  # fmt: skip
        report.step("Assigning tiles", 0.6, "search")
        if not self.shapes:
            raise ValueError("no tiles fit these regions (empty library?)")

    def pin(self) -> None:
        """Make each pinned region's only candidate its pinned crop.

        Crops the class's candidate set lacks are added to it. The pinned
        regions' searched (ranked) lists are kept, still offered for picking.
        """
        self.pinned = np.zeros(0, np.int64)
        self.pin_ref = np.zeros((0, KEEP), np.int64)
        self.pin_cost = np.zeros((0, KEEP), np.float32)
        pins, cls, shapes = self.pins, self.cls, self.shapes
        if pins is not None and len(pins):
            keep = np.isin(pins.region, self.need) & np.isin(cls[pins.region], list(shapes))
            pins = pins.subset(keep)
            self.pin_ref = self.cand_ref[pins.region, :KEEP].copy()
            self.pin_cost = self.costs[pins.region, :KEEP].copy()
            for c, (cands, regs) in list(shapes.items()):
                mine = np.flatnonzero(cls[pins.region] == c)
                if not len(mine):
                    continue
                rows = pins.region[mine]
                cands, ref = cands.with_crops(self.library, pins.tile[mine], pins.rect[mine],
                                              pins.mirrored[mine])  # fmt: skip
                shapes[c] = (cands, regs)
                _, cost = rerank(cands, ref[:, None], self.targets.desc[rows], self.weights[rows],
                                 self.settings.crop_penalty)  # fmt: skip
                self.cand_ref[rows], self.costs[rows] = -1, np.inf
                self.cand_ref[rows, 0], self.costs[rows, 0] = ref, cost[:, 0]
            self.pinned = pins.region.astype(np.int64)
        # Each region's favorite: previews keep showing it until the region gets a tile.
        self.favorite = np.where(np.isfinite(self.costs[:, 0]), self.cand_ref[:, 0], -1)
        self.favorite_cost = self.costs[:, 0].copy()
        self.show(PreviewStage.BEST, self.favorite, self.favorite_cost)

    def assign(self) -> None:
        """Choose a candidate per region under the reuse rules; widen the search for the
        regions they leave without one, and force those still left."""
        settings, regions, need, n = self.settings, self.regions, self.need, self.n
        self.free = free = need[~np.isin(need, self.pinned)]
        tiles = np.full((n, self.kw), -1, np.int64)
        self.cand_mean = np.zeros((n, self.kw, 3), np.float32)
        for cands, regs in self.shapes.values():
            ref = self.cand_ref[regs]
            safe = np.maximum(ref, 0)
            tiles[regs] = np.where(ref >= 0, cands.tile[safe], -1)
            self.cand_mean[regs] = cands.desc[safe][..., MEAN].astype(np.float32)
        tiles[~np.isfinite(self.costs)] = -1

        self.area = regions.area() * self.targets.visible
        mean_area = float(self.area[need].mean()) if len(need) else 1.0
        self.importance = self.area / max(mean_area, 1e-12)
        tile_w, tile_h = self.ctx.tile_size
        self.spacing = settings.min_spacing * math.sqrt(tile_w * tile_h)
        self.assignment = assign = Assignment(tiles, self.costs, regions.center,
                                              len(self.library.status), settings.max_uses,
                                              self.spacing)  # fmt: skip
        assign.force(self.pinned)  # first, whatever the rules say
        self.order = self._priority(free, self.costs)
        if settings.error_diffusion:
            self._diffuse(self._reading_order(free))
        else:
            assign.greedy(self.order)
        unplaced = need[assign.choice[need] < 0]
        if len(unplaced):
            self._widen_waiting(unplaced)
        rest = need[assign.choice[need] < 0]
        assign.force(self._priority(rest, assign.costs))
        self.forced = np.zeros(n, bool)  # placed against the rules (none to spare)
        self.forced[rest] = assign.choice[rest] >= 0
        self.show_assigned(PreviewStage.COMPLETE)

    def _widen_waiting(self, unplaced) -> None:
        """Search again for regions every candidate of which the rules have taken."""
        assign, report = self.assignment, self.report
        # Placed tiles, and the waiting regions' favorites until widening finds better.
        tentative = self._chosen()
        tentative_cost = assign.chosen_costs()
        waiting = tentative < 0
        tentative[waiting] = self.favorite[waiting]
        tentative_cost[waiting] = self.favorite_cost[waiting]
        self.show(PreviewStage.ASSIGNED, tentative, tentative_cost, assign.tint_target)
        report.step("Widening search", 0.65, "assign")
        widened = report.counter("Widening search", len(unplaced))

        def found(rows):
            widened(rows)
            if not report.previewing:
                return
            j = assign.first_allowed(rows)  # else the best: it would be forced
            j = np.where(j >= 0, j, 0)
            ok = np.isfinite(assign.costs[rows, j])  # else keep the favorite
            tentative[rows[ok]] = self.cand_ref[rows[ok], j[ok]]
            tentative_cost[rows[ok]] = assign.costs[rows[ok], j[ok]]
            self.show(PreviewStage.WIDENING, tentative, tentative_cost, assign.tint_target)

        self._widen(unplaced, found)
        assign.greedy(self._priority(unplaced, assign.costs))

    def refine(self) -> None:
        """Move and swap tiles while that lowers the total cost (and time allows)."""
        assign, settings = self.assignment, self.settings
        if not settings.error_diffusion:
            deadline = time.perf_counter() + settings.refine_seconds
            chosen = assign.chosen_costs()
            total = float(np.nansum(np.where(np.isfinite(chosen), chosen, 0) * self.importance))
            while time.perf_counter() < deadline:
                self.report.step("Refining", 0.7, "assign")
                gain = assign.refine_pass(self.order, self.importance)
                self.show_assigned(PreviewStage.REFINING)
                if gain <= 1e-6 * max(total, 1e-12):
                    break
        self.report.step("Checking quality", 0.8, "refine")

    def adapt(self) -> None:
        """Adaptive passes: search harder for the worst regions, keeping a pass only if
        the score improves and no more regions break the rules (see matcher.py)."""
        settings, free, report = self.settings, self.free, self.report
        self.quality, self._raster = self._quality()
        best = self._keep()
        for round_ in range(settings.adaptive_rounds):
            report.step(f"Adaptive pass {round_ + 1}",
                        0.8 + 0.15 * round_ / settings.adaptive_rounds)  # fmt: skip
            assign, forced = self.assignment, self.forced
            error = self.quality.region_error
            err = np.where(np.isnan(error), -np.inf, error)[free]
            count = max(1, int(WORST_SHARE * len(free)))
            worst = free[np.argsort(-err * self.importance[free])[:count]]
            self._widen(worst, report.counter("Searching worst regions again", len(worst)),
                        boost=4 * (round_ + 1))  # fmt: skip
            assign.unplace(worst)
            assign.greedy(self._priority(worst, assign.costs))
            rest = worst[assign.choice[worst] < 0]
            assign.force(self._priority(rest, assign.costs))
            forced[worst] = False
            forced[rest] = assign.choice[rest] >= 0
            if not settings.error_diffusion:
                assign.refine_pass(self.order, self.importance)
            self.show_assigned(PreviewStage.ADAPTIVE)
            self.quality, _ = self._quality(self._raster)
            # Kept only if better and within the rules as much as before (the score
            # doesn't see reuse, so it could trade rule breaks for a sliver of quality).
            if self.quality.score < best["score"] - 1e-3 and forced.sum() <= best["forced"].sum():
                improvement = (best["score"] - self.quality.score) / max(best["score"], 1e-9)
                best = self._keep()
                if improvement < 0.005:
                    break
            else:
                break
        if not np.array_equal(best["choice"], self.assignment.choice):  # the last pass was worse
            self._restore(best)
            self.quality, _ = self._quality(self._raster)
        report.step("Done", 1.0, "quality")

    def result(self) -> MatchResult:
        assign, n = self.assignment, self.n
        tint_target = (assign.tint_target if assign.tint_target is not None
                       else self.targets.desc[:, MEAN])  # fmt: skip
        result = self._snapshot(self._chosen(), assign.chosen_costs(), tint_target, self.quality,
                                self.regret)  # fmt: skip
        result.settings = self.settings.copy()
        result.candidates = self._remember()
        result.choice = self._chosen().astype(np.int32)
        result.manual = np.zeros(n, bool)
        result.manual[self.pinned] = True
        over, close = assign.check()
        chosen = result.tile[result.tile >= 0]
        result.stats.update(
            regions=n,
            placed=int(len(chosen)),
            hidden=int(n - len(self.need)),
            unique_tiles=int(len(np.unique(chosen))),
            most_uses=int(np.bincount(chosen).max()) if len(chosen) else 0,
            rule_violations=int(over + close),
            forced=int(self.forced.sum()),
            manual=int(len(self.pinned)),
            candidates=int(sum(len(c) for c, _ in self.shapes.values())),
            gamut_gap=self._gamut_gap(),
            timings=self.report.timings,
        )
        return result

    # Searching

    def _calibrate(self, cands, index, regs):
        """(effort, RegretStats): search effort for these regions, raised (x4) while
        sampled approximate results cost noticeably more than exact ones, as long as
        that clearly helps, up to every cell of the index."""
        settings, desc, weights = self.settings, self.targets.desc, self.weights
        rng = np.random.default_rng(0)
        sample = rng.choice(regs, min(len(regs), REGRET_SAMPLES), replace=False)
        _, exact = exact_best(cands, desc[sample], weights[sample], settings.crop_penalty)
        effort = start = max(1, min(settings.search_effort, index.groups))
        kept = None
        while True:
            if not index.exact:
                share = math.log(effort / start) / max(math.log(index.groups / start), 1e-9)
                self.report.detail(f"Trying search effort {effort:,} (at most {index.groups:,})",
                                   min(max(share, 0.0), 1.0))  # fmt: skip
            ids = index.search(desc[sample], settings.candidates, effort)
            _, cs = rerank(cands, ids, desc[sample], weights[sample], settings.crop_penalty)
            approx = np.where(np.isfinite(cs[:, 0]), cs[:, 0], exact)
            gap = np.maximum(approx - exact, 0)
            scale = max(float(np.median(exact)), 1e-9)
            extra = 100 * (np.sqrt(np.maximum(approx, 0)) - np.sqrt(np.maximum(exact, 0)))
            stats = RegretStats(
                len(sample), float(gap.mean()), float(np.percentile(gap, 95)),
                float(gap.mean() / scale), effort, index.exact, float(np.maximum(extra, 0).mean()),
            )  # fmt: skip
            if kept is not None and stats.relative > REGRET_GAIN * kept.relative:
                return kept.effort, kept  # more effort barely helps: keep the cheaper search
            kept = stats
            if index.exact or stats.relative <= REGRET_LIMIT or effort >= index.groups:
                return effort, stats
            effort = min(index.groups, effort * 4)

    def _widen(self, regs, found=None, boost=4) -> None:
        """New candidate lists for regions (unplaced first), searched boost times harder
        and grouped by their favorite photo (see matcher.py). found(rows): after each
        batch, with its regions."""
        settings, assign, cls, k = self.settings, self.assignment, self.cls, self.kw
        per_photo = settings.max_uses if settings.max_uses > 0 else UNLIMITED_USES
        rng = np.random.default_rng(0)
        for (c, level), search in self.searches.items():
            cands, index = self.shapes[c][0], search.index
            r = regs[(cls[regs] == c) & (self.level[regs] == level)]
            if not len(r):
                continue
            assign.unplace(r[assign.choice[r] >= 0])
            effort = min(index.groups, search.effort * boost)
            exclude = None
            if settings.max_uses > 0:  # photos at their limit can't take any of these
                full = np.flatnonzero(assign.uses >= settings.max_uses)
                exclude = np.flatnonzero(np.isin(cands.tile, full)) if len(full) else None
            # Groups: regions whose favorite is the same photo (or alone, without one).
            best = self.cand_ref[r, 0]
            key = np.where(best >= 0, cands.tile[np.maximum(best, 0)], -1 - np.arange(len(r)))
            _, group, size = np.unique(key, return_inverse=True, return_counts=True)
            order = np.argsort(group, kind="stable")
            members = np.split(r[order], np.cumsum(size)[:-1])
            query = np.zeros((len(size), self.targets.desc.shape[1]), np.float64)
            np.add.at(query, group, self.targets.desc[r])
            query = (query / size[:, None]).astype(np.float32)
            crops = len(cands) / max(len(np.unique(cands.tile)), 1)  # candidates per photo
            need = np.ceil(WIDEN_SLACK * size / per_photo * crops)
            depth = np.clip(need, k, WIDEN_MAX).astype(int)
            depth = np.minimum(k * 2 ** np.ceil(np.log2(depth / k)).astype(int), WIDEN_MAX)
            for d in np.unique(depth):
                groups = np.flatnonzero(depth == d)
                groups = groups[rng.permutation(len(groups))]  # previews fill in all over
                for part in search_batches(len(groups)):
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
                    ids, cs = rerank(cands, np.asarray(lists, np.int64), self.targets.desc[rows],
                                     self.weights[rows], settings.crop_penalty)  # fmt: skip
                    self._replace(cands, rows, ids, cs)
                    if found is not None:
                        found(rows)

    def _replace(self, cands, rows, ids, cs) -> None:
        """Give regions new candidate lists."""
        assign = self.assignment
        self.cand_ref[rows], assign.costs[rows] = ids, cs
        safe = np.maximum(ids, 0)
        assign.tiles[rows] = np.where((ids >= 0) & np.isfinite(cs), cands.tile[safe], -1)
        self.cand_mean[rows] = cands.desc[safe][..., MEAN].astype(np.float32)

    # Assigning

    def _priority(self, regs, costs) -> np.ndarray:
        """Large, visible regions first, and those that lose most from a second choice."""
        if not len(regs):
            return regs
        best = costs[regs, 0]
        second = np.where(np.isfinite(costs[regs, 1]), costs[regs, 1], best)
        finite = best[np.isfinite(best)]
        scale = max(float(np.median(finite)) if len(finite) else 1.0, 1e-9)
        gap = np.where(np.isfinite(best), second - np.where(np.isfinite(best), best, 0), 0)
        key = self.importance[regs] * (1 + gap / scale)
        return regs[np.argsort(-np.nan_to_num(key, nan=0.0), kind="stable")]

    def _reading_order(self, regs) -> np.ndarray:
        regions = self.regions
        cell = float(np.median(regions.size[regs].min(axis=1))) if len(regs) else 1.0
        row = np.floor(regions.center[regs, 1] / cell)
        return regs[np.lexsort((regions.center[regs, 0], row))]

    def _diffuse(self, order) -> int:
        regions, targets, settings = self.regions, self.targets, self.settings
        cell = float(np.median(regions.size[order].min(axis=1)))
        pos = np.floor(regions.center / cell).astype(np.int64)
        pos -= pos[order].min(axis=0)
        pos = np.clip(pos, 0, None)
        shape = (int(pos[:, 1].max()) + 1, int(pos[:, 0].max()) + 1)
        mean_weight = settings.color_weight * targets.mask[:, 0]
        return self.assignment.greedy_diffused(
            order, self.cand_mean, targets.desc[:, MEAN], mean_weight, settings.tint_value,
            pos[:, ::-1], shape,
        )  # fmt: skip

    def _chosen(self) -> np.ndarray:
        """Per region: index of the chosen candidate within its class (-1: none)."""
        choice = self.assignment.choice
        ref = np.full(len(choice), -1, np.int64)
        rows = np.flatnonzero(choice >= 0)
        ref[rows] = self.cand_ref[rows, choice[rows]]
        return ref

    def _keep(self) -> dict:
        """The state an adaptive pass may need to go back to."""
        assign = self.assignment
        return dict(score=self.quality.score, choice=assign.choice.copy(),
                    tiles=assign.tiles.copy(), costs=assign.costs.copy(),
                    cand_ref=self.cand_ref.copy(), cand_mean=self.cand_mean.copy(),
                    forced=self.forced.copy())  # fmt: skip

    def _restore(self, kept: dict) -> None:
        settings, old = self.settings, self.assignment
        self.cand_ref[:], self.cand_mean[:] = kept["cand_ref"], kept["cand_mean"]
        self.forced = kept["forced"]
        assign = Assignment(kept["tiles"], kept["costs"], self.regions.center,
                            len(self.library.status), settings.max_uses, self.spacing)  # fmt: skip
        assign.tint_target = old.tint_target
        assign.set_choices(kept["choice"])
        self.assignment = assign

    # Results

    def _quality(self, raster=None):
        """(QualityReport, the target raster to reuse) of the current assignment."""
        regions, targets, assign = self.regions, self.targets, self.assignment
        ref = self._chosen()
        cells = np.zeros((len(regions), 4, 4, 3), np.float32)
        placed = ref >= 0
        tint_target = (assign.tint_target if assign.tint_target is not None
                       else targets.desc[:, MEAN])  # fmt: skip
        for c, (cands, _) in self.shapes.items():
            rows = np.flatnonzero(placed & (self.cls == c))
            if not len(rows):
                continue
            desc = cands.desc[ref[rows]].astype(np.float32)
            mean = desc[:, MEAN]
            shift = self.tint * (tint_target[rows] - mean)
            cells[rows] = cell_colors(desc) + shift[:, None, None, :]
        if raster is None:
            report = evaluate(regions, self.ctx, cells, placed)
            ny, nx = report.extra["raster_shape"]
            return report, target_raster(self.ctx, nx, ny)
        return evaluate(regions, self.ctx, cells, placed, raster), raster

    def _snapshot(self, ref, cost, tint_target, report=None, regret=None) -> MatchResult:
        """A result showing candidate ref[r] in each region r (-1: none), every array
        copied (the run may go on changing its own)."""
        n = self.n
        tile = np.full(n, -1, np.int64)
        rect = np.zeros((n, 4), np.float32)
        mirrored = np.zeros(n, bool)
        mean = np.zeros((n, 3), np.float32)
        for c, (cands, _) in self.shapes.items():
            rows = np.flatnonzero((ref >= 0) & (self.cls == c))
            i = ref[rows]
            tile[rows], rect[rows], mirrored[rows] = cands.tile[i], cands.rect[i], cands.mirrored[i]
            mean[rows] = cands.desc[i][:, MEAN].astype(np.float32)
        return MatchResult(
            regions=self.regions, tile=tile, rect=rect, mirrored=mirrored,
            cost=np.array(cost, np.float32), tile_mean=mean,
            tint_target=np.array(tint_target, np.float32), tint=self.tint, quality=report,
            regret=[] if regret is None else regret,
        )  # fmt: skip

    def _remember(self) -> Candidates:
        """The best KEEP candidates of each region, for manual picks."""
        assign, n, pinned = self.assignment, self.n, self.pinned
        ref = self.cand_ref[:, :KEEP].astype(np.int32)
        cost = assign.costs[:, :KEEP].astype(np.float32)
        if ref.shape[1] < KEEP:
            pad = ((0, 0), (0, KEEP - ref.shape[1]))
            ref = np.pad(ref, pad, constant_values=-1)
            cost = np.pad(cost, pad, constant_values=np.inf)
        cost[ref < 0] = np.inf
        auto = self._chosen().astype(np.int32)
        auto_cost = assign.chosen_costs().astype(np.float32)
        if len(pinned):
            ref[pinned], cost[pinned] = self.pin_ref, self.pin_cost
            auto[pinned], auto_cost[pinned] = ref[pinned, 0], cost[pinned, 0]
        sets = tuple(self.shapes[c][0] if c in self.shapes else None
                     for c in range(len(self.classes)))  # fmt: skip
        return Candidates(sets, self.cls.astype(np.int16) if n else np.zeros(0, np.int16), ref,
                          cost, auto, auto_cost)  # fmt: skip

    def _gamut_gap(self) -> float:
        """Share of the visible area whose average color no tile comes close to."""
        need, area = self.need, self.area
        means = np.concatenate([c.desc[:, MEAN] for c, _ in self.shapes.values()])
        means = means.astype(np.float32)
        if len(means) > 200_000:
            means = means[np.random.default_rng(0).choice(len(means), 200_000, replace=False)]
        index = faiss.IndexFlatL2(3)
        index.add(np.ascontiguousarray(means))
        d2, _ = index.search(np.ascontiguousarray(self.targets.desc[need, MEAN]), 1)
        missing = np.sqrt(np.maximum(d2[:, 0], 0)) > GAMUT_DE
        total = float(area[need].sum())
        return float(area[need][missing].sum() / total) if total > 0 else 0.0

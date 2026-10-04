"""Manual picks: changing which remembered candidate a region shows.

A pick never changes the matcher's own choice (`Candidates.auto`), so every
pick can be reverted. Picks may break the reuse rules (the user's choice
wins); `reuse` reports how a candidate stands against them so the picker
can warn.

Changing a few regions rescores only around them (quality.rescore); many
changes at once (reverting everything) evaluate the whole mosaic again.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree

from skitter.core.matching.candidates import KEEP, Pins
from skitter.core.matching.index import CROP_PENALTY_SCALE, exact_top, region_weights
from skitter.core.matching.matcher import MatchResult
from skitter.core.matching.quality import evaluate, rescore
from skitter.core.matching.raster import RasterGrid
from skitter.core.matching.targets import target_descriptors
from skitter.core.slicing import SliceContext
from skitter.core.tiles.descriptors import MEAN, cell_colors

LOCAL_LIMIT = 48  # more changed regions than this: evaluate the whole mosaic again


def editable(result: MatchResult | None, region: int) -> bool:
    """Whether a region's tile can be picked by hand (it shows and has candidates)."""
    return (
        result is not None
        and result.candidates is not None
        and 0 <= region < len(result.tile)
        and result.tile[region] >= 0
        and result.candidates.set_of(region) is not None
    )


def cells_of(result: MatchResult, regions, choice=None) -> np.ndarray:
    """(n, 4, 4, 3) OKLab cells the given regions show (tinted), for scoring."""
    regions = np.asarray(regions, dtype=np.int64)
    choice = result.choice if choice is None else choice
    desc = result.candidates.desc(regions, choice[regions])
    shift = result.tint * (result.tint_target[regions] - desc[:, MEAN])
    return cell_colors(desc) + shift[:, None, None, :]


def region_target(result: MatchResult, ctx: SliceContext, region: int):
    """(descriptor, cost weights) of one region, as matching computed them."""
    regions = result.regions
    grid = RasterGrid.of(regions)  # matching's visibility raster
    raster = grid.paint(regions, *grid.pixels(*regions[region : region + 1].bounds()[0]))
    targets = target_descriptors(regions, ctx, raster, indices=[region])
    settings = result.settings
    weights = region_weights(settings.weights(), targets.mask, settings.tint_value)
    return targets.desc[0], weights[0]


def candidate_cost(result: MatchResult, ctx: SliceContext, region: int, ref: int) -> float:
    """Exact matching cost of one candidate for a region."""
    refs, costs = result.candidates.ranked(region)
    hit = np.flatnonzero(refs == ref)
    if len(hit):
        return float(costs[hit[0]])
    s = result.candidates.set_of(region)
    desc, weights = region_target(result, ctx, region)
    x = s.desc[ref].astype(np.float32)
    penalty = result.settings.crop_penalty * CROP_PENALTY_SCALE * (1.0 - s.retained[ref])
    return float(((x - desc) ** 2 * weights).sum() + penalty)


def find_more(result: MatchResult, ctx: SliceContext, region: int, count: int = KEEP) -> int:
    """Search every candidate exactly for one region and remember the next `count`
    best it doesn't list yet. Returns how many were added."""
    cands = result.candidates
    s = cands.set_of(region)
    listed, _ = cands.ranked(region)
    desc, weights = region_target(result, ctx, region)
    ids, costs = exact_top(s, desc, weights, result.settings.crop_penalty, len(listed) + count)
    new = ~np.isin(ids, listed)
    ids, costs = ids[new][:count], costs[new][:count]
    added = len(ids)
    if added:
        old = cands.more.get(region)
        if old is not None:
            ids, costs = np.concatenate([old[0], ids]), np.concatenate([old[1], costs])
        cands.more[region] = (ids.astype(np.int32), costs.astype(np.float32))
    return added


def apply_choices(
    result: MatchResult, ctx: SliceContext, changes: dict[int, tuple[int, bool]]
) -> MatchResult:
    """A new result in which region r shows candidate ref (manual: picked by hand),
    for each r: (ref, manual) in changes. The given result is left as it was."""
    changes = {int(r): (int(ref), bool(m)) for r, (ref, m) in changes.items()}
    changes = {r: c for r, c in changes.items()
               if (result.choice[r], result.manual[r]) != c}  # fmt: skip
    if not changes:
        return result
    cands = result.candidates
    tile, rect, mirrored = result.tile.copy(), result.rect.copy(), result.mirrored.copy()
    cost, tile_mean = result.cost.copy(), result.tile_mean.copy()
    choice, manual = result.choice.copy(), result.manual.copy()
    for r, (ref, picked) in changes.items():
        t, rc, m, mean = cands.crops(r, [ref])
        tile[r], rect[r], mirrored[r], tile_mean[r] = t[0], rc[0], m[0], mean[0]
        cost[r] = (
            cands.auto_cost[r] if ref == cands.auto[r] else candidate_cost(result, ctx, r, ref)
        )
        choice[r], manual[r] = ref, picked
    placed = tile >= 0
    changed = np.fromiter(changes, np.int64)
    moved = changed[choice[changed] != result.choice[changed]]
    if len(moved) <= LOCAL_LIMIT:
        quality = rescore(
            result.quality, result.regions, ctx, placed,
            lambda ids: cells_of(result, ids), lambda ids: cells_of(result, ids, choice), moved,
        )  # fmt: skip
    else:
        all_regions = np.arange(len(tile))
        quality = evaluate(result.regions, ctx, cells_of(result, all_regions, choice), placed)
    edited = replace(
        result, tile=tile, rect=rect, mirrored=mirrored, cost=cost, tile_mean=tile_mean,
        quality=quality, stats=dict(result.stats), choice=choice, manual=manual,
    )  # fmt: skip
    edited.stats.update(usage_stats(edited, ctx))
    return edited


def pick(result: MatchResult, ctx: SliceContext, region: int, ref: int) -> MatchResult:
    """Show candidate ref in a region, picked by hand."""
    return apply_choices(result, ctx, {region: (ref, True)})


def revert(result: MatchResult, ctx: SliceContext, regions=None) -> MatchResult:
    """Give regions (default: every manual pick) back the matcher's choice."""
    if regions is None:
        regions = np.flatnonzero(result.manual)
    auto = result.candidates.auto
    return apply_choices(result, ctx, {int(r): (int(auto[r]), False) for r in regions})


def pins(result: MatchResult) -> Pins:
    """The manual picks, to keep when matching again."""
    rows = np.flatnonzero(result.manual) if result.manual is not None else np.zeros(0, np.int64)
    return Pins(rows, result.tile[rows].copy(), result.rect[rows].copy(),
                result.mirrored[rows].copy())  # fmt: skip


def _spacing(result: MatchResult, ctx: SliceContext) -> float:
    """The repeat spacing rule in mosaic units (0: off)."""
    return result.settings.min_spacing * math.sqrt(ctx.tile_size[0] * ctx.tile_size[1])


def usage_stats(result: MatchResult, ctx: SliceContext) -> dict:
    """Stats that depend on which tiles are used (see MatchResult.stats)."""
    chosen = result.tile[result.tile >= 0]
    counts = np.bincount(chosen) if len(chosen) else np.zeros(0, np.int64)
    settings = result.settings
    over = int(np.maximum(counts - settings.max_uses, 0).sum()) if settings.max_uses else 0
    close = 0
    spacing = _spacing(result, ctx)
    if spacing > 0:
        centers = result.regions.center
        for t in np.flatnonzero(counts > 1):
            tree = cKDTree(centers[result.tile == t])
            close += len(tree.query_pairs(spacing * (1 - 1e-9)))
    return {
        "placed": int(len(chosen)),
        "unique_tiles": int(np.count_nonzero(counts)),
        "most_uses": int(counts.max()) if len(counts) else 0,
        "rule_violations": over + close,
        "manual": result.manual_count,
    }


@dataclass(frozen=True)
class Reuse:
    """How candidate tiles of one region stand against the reuse rules."""

    uses: np.ndarray  # (n,) uses of each tile elsewhere in the mosaic
    nearest: np.ndarray  # (n,) distance to the nearest other use, base tiles (inf: none)
    max_uses: int
    spacing: float  # base tiles (0: off)

    @property
    def over(self) -> np.ndarray:
        """Picking it would use the tile more often than allowed."""
        return (self.uses >= self.max_uses) if self.max_uses else np.zeros(len(self.uses), bool)

    @property
    def close(self) -> np.ndarray:
        """Picking it would put two uses closer than the spacing."""
        return self.nearest < self.spacing

    @property
    def breaks(self) -> np.ndarray:
        return self.over | self.close


def reuse(result: MatchResult, ctx: SliceContext, region: int, slots) -> Reuse:
    """Uses elsewhere and the nearest other use of each tile (slot), for one region."""
    slots = np.asarray(slots, dtype=np.int64)
    others = result.tile.copy()
    others[region] = -1
    unit = math.sqrt(ctx.tile_size[0] * ctx.tile_size[1])
    center = result.regions.center
    uses = np.zeros(len(slots), np.int64)
    nearest = np.full(len(slots), np.inf)
    for i, s in enumerate(slots):
        where = np.flatnonzero(others == s)
        uses[i] = len(where)
        if len(where):
            nearest[i] = np.sqrt(((center[where] - center[region]) ** 2).sum(axis=1)).min() / unit
    return Reuse(uses, nearest, result.settings.max_uses, result.settings.min_spacing)

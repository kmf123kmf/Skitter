"""Manual picks: remembered candidates, picking, reverting, rescoring, and pins."""

import numpy as np
import pytest
from PIL import Image

from skitter.core.matching.candidates import KEEP
from skitter.core.matching.edit import (
    LOCAL_LIMIT,
    apply_choices,
    cells_of,
    editable,
    find_more,
    pick,
    pins,
    reuse,
    revert,
    usage_stats,
)
from skitter.core.matching.index import CandidateSet, exact_top, rerank
from skitter.core.matching.matcher import Matcher
from skitter.core.matching.quality import evaluate
from skitter.core.matching.settings import MatchSettings
from skitter.core.slicing import MosaicLayout, RegionSet, SliceContext
from skitter.core.slicing.operations import GridSlicer
from skitter.core.tiles.descriptors import DIM
from skitter.core.tiles.library import TileLibrary


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """Solid-color tiles of a few shapes, plus a few with structure (some mirror-sensitive)."""
    root = tmp_path_factory.mktemp("tiles")
    rng = np.random.default_rng(5)
    for i, color in enumerate(rng.integers(0, 256, (90, 3))):
        size = [(60, 60), (90, 60), (60, 90), (120, 60)][i % 4]
        Image.new("RGB", size, tuple(int(c) for c in color)).save(root / f"solid{i}.png")
    for i in range(6):
        split = np.zeros((60, 90, 3), np.uint8)
        split[:, : 20 + 10 * i] = (255, 200 - 30 * i, 40 * i)
        Image.fromarray(split).save(root / f"split{i}.png")
    lib = TileLibrary(tmp_path_factory.mktemp("cache"))
    lib.set_roots([root])
    lib.update(workers=0)
    yield lib
    lib.close()


def make_ctx(columns=12):
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (9, 12, 3), dtype=np.uint8)
    image = np.repeat(np.repeat(image, 12, 0), 12, 1)  # 144 x 108 px of colored blocks
    return SliceContext(image, MosaicLayout(columns=columns), tile_width=12)


@pytest.fixture(scope="module")
def matched(library):
    ctx = make_ctx()
    regions = GridSlicer().apply(ctx.canvas(), ctx)
    settings = MatchSettings(tint="subtle", max_uses=3, min_spacing=2.0, refine_seconds=0.2,
                             mirror=True)  # fmt: skip
    return ctx, Matcher(library).run(regions, ctx, settings)


def full_score(result, ctx):
    everything = np.arange(len(result.tile))
    return evaluate(result.regions, ctx, cells_of(result, everything), result.tile >= 0)


# Remembered candidates


def test_result_remembers_ranked_candidates(matched):
    ctx, result = matched
    cands = result.candidates
    n = len(result.tile)
    assert cands.ref.shape == (n, KEEP) and cands.ref.dtype == np.int32
    assert cands.nbytes < n * (KEEP * 8 + 16)  # about 8 bytes per remembered candidate
    assert not result.manual.any() and result.manual_count == 0
    for r in range(n):
        refs, costs = cands.ranked(r)
        assert len(refs) == KEEP and np.all(np.diff(costs) >= 0)
        assert result.choice[r] == cands.auto[r]
        slot, rect, mirrored, mean = cands.crops(r, [result.choice[r]])
        assert slot[0] == result.tile[r] and np.allclose(rect[0], result.rect[r])
        assert mirrored[0] == result.mirrored[r] and np.allclose(mean[0], result.tile_mean[r])
    # The list's costs are the matcher's own: the chosen one's cost appears in it.
    listed = cands.ref == result.choice[:, None]
    rows = np.flatnonzero(listed.any(axis=1))
    assert len(rows) > n // 2
    assert np.allclose(cands.cost[listed][: len(rows)], result.cost[rows], rtol=1e-5)


def test_results_without_candidates_are_not_editable(matched):
    ctx, result = matched
    assert editable(result, 0)
    assert not editable(None, 0) and not editable(result, -1) and not editable(result, 10**6)


# Picking and reverting


def test_pick_shows_the_candidate_and_rescores_like_a_full_evaluation(matched):
    ctx, result = matched
    before = result.tile.copy()
    edited = result
    for r, rank in ((0, 5), (17, 1), (60, 12), (107, 31)):
        refs, _ = result.candidates.ranked(r)
        edited = pick(edited, ctx, r, int(refs[rank]))
        slot, rect, mirrored, _ = result.candidates.crops(r, [refs[rank]])
        assert edited.tile[r] == slot[0] and np.allclose(edited.rect[r], rect[0])
        assert edited.mirrored[r] == mirrored[0] and edited.manual[r]
    assert np.array_equal(result.tile, before) and not result.manual.any()  # untouched
    assert edited.manual_count == 4 and edited.stats["manual"] == 4

    full = full_score(edited, ctx)
    assert edited.quality.score == pytest.approx(full.score, abs=1e-4)
    assert edited.quality.ssim == pytest.approx(full.ssim, abs=1e-5)
    for scale, value in full.by_scale.items():
        assert edited.quality.by_scale[scale] == pytest.approx(value, abs=1e-4)
    assert np.allclose(edited.quality.region_error, full.region_error, atol=1e-3, equal_nan=True)

    picked_cost = result.candidates.ranked(0)[1][5]
    assert edited.cost[0] == pytest.approx(picked_cost)


def test_revert_restores_the_matchers_choice(matched):
    ctx, result = matched
    refs, _ = result.candidates.ranked(3)
    other = int(result.candidates.ranked(9)[0][2])
    edited = pick(pick(result, ctx, 3, int(refs[4])), ctx, 9, other)
    one = revert(edited, ctx, [3])
    assert one.tile[3] == result.tile[3] and not one.manual[3] and one.manual[9]
    back = revert(edited, ctx)
    assert np.array_equal(back.tile, result.tile) and not back.manual.any()
    assert np.allclose(back.rect, result.rect) and np.allclose(back.cost, result.cost)
    assert back.quality.score == pytest.approx(result.quality.score, abs=1e-4)
    assert apply_choices(back, ctx, {}) is back  # nothing to change


def test_many_changes_evaluate_the_whole_mosaic(matched):
    ctx, result = matched
    changes = {r: (int(result.candidates.ranked(r)[0][1]), True)
               for r in range(LOCAL_LIMIT + 5)}  # fmt: skip
    edited = apply_choices(result, ctx, changes)
    assert edited.manual_count == LOCAL_LIMIT + 5
    assert edited.quality.score == pytest.approx(full_score(edited, ctx).score, abs=1e-6)


def test_find_more_adds_the_next_best_crops(matched):
    ctx, result = matched
    cands = result.candidates
    listed, costs = cands.ranked(40)
    added = find_more(result, ctx, 40, count=10)
    try:
        assert added == 10
        refs, all_costs = cands.ranked(40)
        assert len(refs) == len(listed) + 10 and len(set(refs.tolist())) == len(refs)
        more, more_costs = refs[len(listed) :], all_costs[len(listed) :]
        assert np.all(np.diff(more_costs) >= 0)
        # They are what an exact search ranks next, and picking one works.
        edited = pick(result, ctx, 40, int(more[0]))
        assert edited.cost[40] == pytest.approx(more_costs[0], rel=1e-4)
    finally:
        cands.more.pop(40, None)


def test_reuse_reports_rule_breaks(matched):
    ctx, result = matched
    r = 50
    neighbor_tile = result.tile[51]  # used right next door
    status = reuse(result, ctx, r, [neighbor_tile, -5])
    assert status.uses[0] >= 1 and status.nearest[0] == pytest.approx(1.0)
    assert status.close[0] and status.breaks[0]
    assert status.uses[1] == 0 and not status.breaks[1]  # unused
    # Breaking a rule on purpose counts as a rule break.
    s = result.candidates.set_of(r)
    where = np.flatnonzero(s.tile == neighbor_tile)
    if len(where):
        edited = pick(result, ctx, r, int(where[0]))
        assert usage_stats(edited, ctx)["rule_violations"] > result.stats["rule_violations"]


# Pins


def test_pins_are_kept_when_matching_again(library, matched):
    ctx, result = matched
    picks = {}
    for r in (2, 30, 31, 77):  # 30 and 31 get the same tile: a rule break the user wants
        picks[r] = int(result.candidates.ranked(r)[0][6])
    picks[31] = int(np.flatnonzero(result.candidates.set_of(31).tile == result.candidates.crops(
        30, [picks[30]])[0][0])[0])  # fmt: skip
    edited = apply_choices(result, ctx, {r: (ref, True) for r, ref in picks.items()})
    kept = pins(edited)
    assert sorted(kept.region.tolist()) == [2, 30, 31, 77]

    # Matching again, even with other crop settings (no mirroring): the picks stay.
    settings = edited.settings.copy()
    settings.mirror = False
    again = Matcher(library).run(edited.regions, ctx, settings, pins=kept)
    rows = kept.region
    assert np.array_equal(again.tile[rows], edited.tile[rows])
    assert np.allclose(again.rect[rows], edited.rect[rows])
    assert np.array_equal(again.mirrored[rows], edited.mirrored[rows])
    assert again.manual[rows].all() and again.manual_count == 4
    assert again.stats["manual"] == 4
    # They can still be changed: their ranked lists are remembered, and revert works.
    assert (again.candidates.ref[rows, 0] >= 0).all()
    back = revert(again, ctx)
    assert not back.manual.any()
    # Everything else respects the rules around the pins.
    others = np.setdiff1d(np.arange(len(again.tile)), rows)
    counts = np.bincount(again.tile[others])
    assert counts.max() <= settings.max_uses


# Candidate sets


def test_candidate_sets_find_and_add_crops(library):
    m = 5
    rng = np.random.default_rng(0)
    cands = CandidateSet(
        1.0, np.array([3, 1, 2, 1, 3]), rng.uniform(0, 1, (m, 4)).astype(np.float32),
        np.array([False, False, False, True, True]), np.ones(m, np.float32),
        rng.normal(0, 0.1, (m, DIM)).astype(np.float16),
    )  # fmt: skip
    found = cands.find([1, 1, 3, 2], cands.rect[[1, 3, 0, 0]], [False, True, False, False])
    assert found.tolist() == [1, 3, 0, -1]
    slot = int(library.ids[0])
    bigger, index = cands.with_crops(library, [slot, 3], [[0, 0, 1, 1], cands.rect[4]],
                                     [True, True])  # fmt: skip
    assert index.tolist() == [5, 4] and len(bigger) == 6 and len(cands) == 5
    assert bigger.tile[5] == slot and bigger.mirrored[5]


def test_exact_top_matches_a_full_rerank():
    rng = np.random.default_rng(3)
    m = 3000
    cands = CandidateSet(
        1.0, np.arange(m), np.tile([0, 0, 1, 1], (m, 1)).astype(np.float32), np.zeros(m, bool),
        rng.uniform(0.5, 1, m).astype(np.float32), rng.normal(0, 0.1, (m, DIM)).astype(np.float16),
    )  # fmt: skip
    q = rng.normal(0, 0.1, DIM).astype(np.float32)
    w = rng.uniform(0.5, 1.5, DIM).astype(np.float32)
    ids, costs = exact_top(cands, q, w, 0.3, 20, chunk=700)
    every, every_cost = rerank(cands, np.arange(m)[None], q[None], w[None], 0.3)
    assert np.array_equal(ids, every[0, :20])
    assert np.allclose(costs, every_cost[0, :20], rtol=1e-5)


def test_rescore_handles_overlapping_rotated_regions():
    """Photo Pile-like stacks: local rescoring equals a full evaluation."""
    from skitter.core.matching.quality import rescore

    rng = np.random.default_rng(4)
    image = np.repeat(np.repeat(rng.integers(0, 256, (10, 15, 3), dtype=np.uint8), 10, 0), 10, 1)
    ctx = SliceContext(image, MosaicLayout(columns=15), tile_width=10)
    n = 400
    regions = RegionSet.from_arrays(
        rng.uniform(0, [ctx.width, ctx.height], (n, 2)), rng.uniform(40, 160, (n, 2)),
        rng.uniform(-0.6, 0.6, n), rng.uniform(0, 1, n),
    )  # fmt: skip
    cells = rng.uniform(-0.1, 0.1, (n, 4, 4, 3)).astype(np.float32)
    cells[..., 0] += 0.6
    placed = rng.uniform(size=n) > 0.05
    base = evaluate(regions, ctx, cells, placed)
    changed = np.array([0, 5, 6, n - 1, int(regions.stacking_order()[0])])
    new = cells.copy()
    new[changed] = rng.uniform(0, 1, (len(changed), 4, 4, 3))
    local = rescore(base, regions, ctx, placed, lambda ids: cells[ids].copy(),
                    lambda ids: new[ids].copy(), changed)  # fmt: skip
    full = evaluate(regions, ctx, new, placed)
    assert local.score == pytest.approx(full.score, abs=1e-4)
    assert local.ssim == pytest.approx(full.ssim, abs=1e-5)
    assert np.allclose(local.region_error, full.region_error, atol=1e-3, equal_nan=True)

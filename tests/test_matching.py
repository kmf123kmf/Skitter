"""Candidate search, constrained assignment, quality scoring, and the full matcher."""

import math
from types import SimpleNamespace

import faiss
import numpy as np
import pytest
from PIL import Image

from skitter.core.color import rgb8_to_oklab
from skitter.core.matching.assign import Assignment
from skitter.core.matching.index import (
    EXACT_LIMIT,
    CandidateSet,
    SearchIndex,
    build_candidates,
    exact_best,
    region_weights,
    rerank,
    train_ivf,
)
from skitter.core.matching.matcher import MatchCancelled, Matcher, PreviewStage
from skitter.core.matching.quality import evaluate, target_raster
from skitter.core.matching.settings import MatchSettings
from skitter.core.slicing import MosaicLayout, RegionSet, SliceContext, mask_regions
from skitter.core.slicing.operations import GridSlicer
from skitter.core.tiles.descriptors import DIM, MEAN, dim_weights
from skitter.core.tiles.library import TileLibrary

# Candidates and search


def random_candidates(m, seed=0):
    rng = np.random.default_rng(seed)
    desc = rng.normal(0, 0.1, (m, DIM)).astype(np.float16)
    return CandidateSet(
        1.0, np.arange(m), np.tile([0, 0, 1, 1], (m, 1)).astype(np.float32),
        np.zeros(m, bool), np.ones(m, np.float32), desc,
    )  # fmt: skip


def test_build_candidates_crops_and_mirrors():
    thumbs = np.zeros((3, 32, 32, 3), np.uint8)
    thumbs[:, :, :16] = 255  # left half white
    lib = SimpleNamespace(
        ids=np.array([0, 2]), width=np.array([300, 0, 100]), height=np.array([100, 0, 100]),
        thumbs=thumbs, thumb_size=np.array([[32, 11], [0, 0], [32, 32]]), status=None,
    )  # fmt: skip
    cands = build_candidates(lib, aspect=1.0, crops=3, mirrored=True)
    assert cands.tile.tolist() == [0, 0, 0, 2] * 2
    assert cands.mirrored.tolist() == [False] * 4 + [True] * 4
    # Mirroring swaps the white side.
    assert cands.desc[3, 15] > 0 and cands.desc[7, 15] < 0


def test_rerank_is_exact_and_sorted():
    cands = random_candidates(500)
    rng = np.random.default_rng(1)
    desc = rng.normal(0, 0.1, (20, DIM)).astype(np.float32)
    mask = rng.random((20, DIM)).astype(np.float32)
    w = region_weights(dim_weights(), mask, tint=0.3)
    ids = np.tile(np.arange(30), (20, 1))
    ids[:, -1] = -1
    out_ids, costs = rerank(cands, ids, desc, w, crop_penalty=0.0)
    x = cands.desc[out_ids[0, 0]].astype(np.float32)
    assert costs[0, 0] == pytest.approx((w[0] * (x - desc[0]) ** 2).sum(), rel=1e-4)
    assert (np.diff(costs, axis=1)[:, :-2] >= 0).all() and np.isinf(costs[:, -1]).all()
    # Tint shrinks the average-color term by (1 - t)^2.
    assert w[0, MEAN][0] == pytest.approx(dim_weights()[0] * mask[0, 0] * 0.49, rel=1e-5)


def test_exact_best_matches_brute_force():
    cands = random_candidates(3000)
    rng = np.random.default_rng(2)
    desc = rng.normal(0, 0.1, (16, DIM)).astype(np.float32)
    w = region_weights(dim_weights(), np.ones((16, DIM), np.float32), 0.0)
    best, cost = exact_best(cands, desc, w, 0.0, chunk=1000)
    x = cands.desc.astype(np.float32)
    brute = ((x[None] - desc[:, None]) ** 2 * w[:, None]).sum(-1)
    np.testing.assert_array_equal(best, brute.argmin(axis=1))
    np.testing.assert_allclose(cost, brute.min(axis=1), rtol=1e-3, atol=1e-6)


def test_approximate_index_finds_near_neighbors():
    cands = random_candidates(EXACT_LIMIT + 10_000)
    weights = dim_weights()
    index = SearchIndex(cands, weights, tint_ref=0.0)
    assert not index.exact
    queries = cands.desc[:200].astype(np.float32) + 0.001
    found = index.search(queries, 5, nprobe=32)
    assert (found[:, 0] == np.arange(200)).mean() > 0.9


# Assignment


def grid_centers(n, step=10.0):
    side = math.ceil(math.sqrt(n))
    return np.stack([np.arange(n) % side, np.arange(n) // side], axis=1) * step


def test_stepwise_training_gives_faiss_own_index():
    rng = np.random.default_rng(4)
    centers = rng.normal(0, 1, (300, DIM)).astype(np.float32)
    x = (centers[rng.integers(0, 300, 20_000)] + rng.normal(0, 0.3, (20_000, DIM))).astype(
        np.float32
    )
    own = faiss.index_factory(DIM, "IVF128,SQ8")
    own.train(x)
    stepwise = faiss.index_factory(DIM, "IVF128,SQ8")
    reports = []
    train_ivf(stepwise, x, lambda message, fraction: reports.append((message, fraction)))

    quantizers = [faiss.extract_index_ivf(i).quantizer for i in (own, stepwise)]
    np.testing.assert_array_equal(*(q.reconstruct_n(0, 128) for q in quantizers))
    for index in (own, stepwise):
        faiss.extract_index_ivf(index).nprobe = 8
        index.add(x)
    np.testing.assert_array_equal(own.search(x[:500], 10)[1], stepwise.search(x[:500], 10)[1])
    # One report per k-means round, then the finish.
    rounds = faiss.extract_index_ivf(own).cp.niter
    assert len(reports) == rounds + 1 and reports[0] == (
        f"Grouping similar tiles: round 1 of {rounds}",
        0.0,
    )
    assert [f for _, f in reports] == sorted(f for _, f in reports) and reports[-1][1] == 1.0


def test_index_building_reports_progress_and_can_stop(monkeypatch):
    import skitter.core.matching.index as index_module

    monkeypatch.setattr(index_module, "EXACT_LIMIT", 1000)  # IVF without a huge set
    cands = random_candidates(6000)
    weights = dim_weights()
    reports = []
    SearchIndex(cands, weights, 0.0, chunk=2048,
                progress=lambda message, fraction: reports.append((message, fraction)))  # fmt: skip
    messages = [m for m, _ in reports]
    assert any(m.startswith("Grouping similar tiles") for m in messages)
    assert [m for m in messages if m.startswith("Adding")] == [
        f"Adding tiles to the index: {lo:,} of 6,000" for lo in (0, 2048, 4096)
    ]
    assert all(0.0 <= f <= 1.0 for _, f in reports)

    class Stop(Exception):
        pass

    def stop(message, fraction):
        if message.startswith("Grouping similar tiles: round 2"):
            raise Stop

    with pytest.raises(Stop):
        SearchIndex(cands, weights, 0.0, progress=stop)


def test_matcher_reports_indexing_and_cancels_within_it(library):
    ctx = flat_ctx(target_image(), columns=8, tile=20)
    regions = GridSlicer().apply(ctx.canvas(), ctx)
    settings = MatchSettings(adaptive_rounds=0, refine_seconds=0.1)
    details = []
    Matcher(library).run(regions, ctx, settings, detail=lambda *report: details.append(report))
    assert details[0][0].startswith("Describing tile crops: 0 of")
    assert any(m.startswith("Adding tiles to the index") for m, _ in details)

    # Cancelling while indexing stops the run there, and leaves nothing half built.
    matcher, calls = Matcher(library), []
    with pytest.raises(MatchCancelled):
        matcher.run(regions, ctx, settings, cancelled=lambda: len(calls) >= 1,
                    detail=lambda message, fraction: calls.append(message))  # fmt: skip
    assert calls and not matcher._candidates and not matcher._indexes


@pytest.mark.parametrize("exact", [True, False])
def test_batched_search_matches_one_big_batch(monkeypatch, exact):
    import skitter.core.matching.index as index_module
    import skitter.core.matching.run as run_module
    from skitter.core.matching.run import batched_search

    if not exact:
        monkeypatch.setattr(index_module, "EXACT_LIMIT", 1000)
    cands = random_candidates(4000)
    index = SearchIndex(cands, dim_weights(), 0.0)
    assert index.exact == exact
    rng = np.random.default_rng(5)
    desc = rng.normal(0, 0.1, (1500, DIM)).astype(np.float32)
    weights = region_weights(dim_weights(), np.ones((1500, DIM), np.float32), 0.0)
    regs = rng.permutation(3000)[:1500]  # region ids into the (larger) arrays below
    big_desc = np.zeros((3000, DIM), np.float32)
    big_desc[regs] = desc
    big_w = np.zeros((3000, DIM), np.float32)
    big_w[regs] = weights
    settings = MatchSettings()

    one = batched_search(cands, index, regs, big_desc, big_w, settings, 20, 8)  # one batch
    monkeypatch.setattr(run_module, "FIRST_BATCH", 37)  # many small ones
    monkeypatch.setattr(run_module, "MIN_BATCH", 37)
    monkeypatch.setattr(run_module, "MAX_BATCH", 37)
    batches = []
    many = batched_search(cands, index, regs, big_desc, big_w, settings, 20, 8,
                           lambda rows, ids, cs: batches.append((rows, ids, cs)))  # fmt: skip
    np.testing.assert_array_equal(many[0], one[0])
    np.testing.assert_array_equal(many[1], one[1])
    assert len(batches) == -(-1500 // 37)
    rows = np.concatenate([b[0] for b in batches])
    assert sorted(rows) == sorted(regs) and not np.array_equal(rows, regs)  # shuffled
    position = {r: i for i, r in enumerate(regs)}
    for rows, ids, cs in batches:  # each batch hands over its own regions' results
        at = [position[r] for r in rows]
        np.testing.assert_array_equal(ids, one[0][at])
        np.testing.assert_array_equal(cs, one[1][at])


def test_first_allowed_is_what_greedy_would_pick():
    n, k = 60, 10
    rng = np.random.default_rng(1)
    tiles = rng.integers(0, 30, (n, k))
    costs = np.sort(rng.random((n, k)), axis=1).astype(np.float32)
    costs[5, 3:] = np.inf  # few candidates
    costs[59] = np.inf  # none at all: nothing allowed
    a = Assignment(tiles, costs, grid_centers(n), n_tiles=30, max_uses=2, spacing=25.0)
    a.greedy(np.arange(30))  # some placed; the rest wait
    waiting = np.flatnonzero(a.choice < 0)
    before = a.choice.copy(), a.uses.copy()
    pick = a.first_allowed(waiting)
    np.testing.assert_array_equal(a.choice, before[0])  # nothing placed
    np.testing.assert_array_equal(a.uses, before[1])
    for r, j in zip(waiting, pick, strict=True):  # each alone, greedy picks the same
        trial = Assignment(tiles, costs, grid_centers(n), n_tiles=30, max_uses=2, spacing=25.0)
        trial.set_choices(before[0])
        trial.greedy([r])
        assert trial.choice[r] == j
    assert (pick >= 0).any() and (pick < 0).any()


@pytest.mark.parametrize("exact", [True, False])
def test_search_skips_excluded_candidates(monkeypatch, exact):
    import skitter.core.matching.index as index_module

    if not exact:
        monkeypatch.setattr(index_module, "EXACT_LIMIT", 1000)
    cands = random_candidates(3000)
    index = SearchIndex(cands, dim_weights(), 0.0)
    queries = cands.desc[:50].astype(np.float32)
    plain = index.search(queries, 30, 16)
    exclude = np.unique(plain[:, :10])  # everyone's favorites
    found = index.search(queries, 30, 16, exclude)
    assert (found >= 0).all() and not np.isin(found, exclude).any()


def test_forced_regions_spread_over_the_least_used_photos():
    n, k = 40, 5
    tiles = np.tile(np.arange(k), (n, 1))  # everyone wants photo 0, then 1, ...
    costs = np.tile(np.arange(k, dtype=np.float32), (n, 1))
    a = Assignment(tiles, costs, grid_centers(n), n_tiles=k, max_uses=1, spacing=0.0)
    a.greedy(np.arange(n))  # 5 placed, 35 waiting
    assert a.force(np.arange(n)) == n - k
    np.testing.assert_array_equal(np.bincount(a.chosen_tiles(), minlength=k), [8] * k)
    # Equally used: the photo farthest from its other uses wins.
    centers = np.array([[0.0, 0.0], [100.0, 0.0], [10.0, 0.0]])
    b = Assignment(np.array([[0, 1]] * 3), np.zeros((3, 2), np.float32), centers, n_tiles=2,
                   max_uses=1)  # fmt: skip
    b.greedy([0, 1])  # 0 takes photo 0 at x=0, 1 takes photo 1 at x=100
    b.force([2])  # at x=10: photo 1 is farther from its use
    assert b.chosen_tiles()[2] == 1


def test_plain_areas_keep_the_reuse_rules(library):
    # Every region wants the same few photos; there are enough photos for all of them
    # within the rules, if the search looks past each region's favorites.
    image = np.full((150, 200, 3), 235, np.uint8)
    ctx = flat_ctx(image, columns=20, tile=10)
    regions = GridSlicer().apply(ctx.canvas(), ctx)
    assert len(regions) == 300 and len(library) * 3 >= len(regions)
    settings = MatchSettings(max_uses=3, min_spacing=0.0, refine_seconds=5.0, adaptive_rounds=1)
    result = Matcher(library).run(regions, ctx, settings)
    assert result.stats["rule_violations"] == 0 and result.stats["forced"] == 0
    assert np.bincount(result.tile).max() <= 3
    # Too few photos even so: the breaks spread instead of piling onto one photo.
    strict = Matcher(library).run(regions, ctx, MatchSettings(max_uses=1, refine_seconds=5.0,
                                                              adaptive_rounds=1))  # fmt: skip
    assert strict.stats["forced"] == len(regions) - len(library)
    assert np.bincount(strict.tile).max() <= 3  # 300 regions over 122 photos


def test_reuse_rules_hold():
    n, k = 100, 30
    rng = np.random.default_rng(0)
    tiles = np.tile(np.arange(k), (n, 1))  # everyone prefers tile 0, then 1, ...
    costs = np.sort(rng.random((n, k)), axis=1).astype(np.float32)
    a = Assignment(tiles, costs, grid_centers(n), n_tiles=k, max_uses=4, spacing=25.0)
    assert a.greedy(np.arange(n)) == 0
    assert a.check() == (0, 0)
    counts = np.bincount(a.chosen_tiles(), minlength=k)
    assert counts.max() <= 4
    before = a.chosen_costs().sum()
    a.refine_pass(np.arange(n), np.ones(n))
    assert a.check() == (0, 0) and a.chosen_costs().sum() <= before + 1e-6


def test_refinement_moves_a_blocking_region():
    # Both want tile 0; region 1 loses much more without it but was placed second.
    tiles = np.array([[0, 1], [0, 2]])
    costs = np.array([[0.0, 0.1], [0.0, 1.0]], np.float32)
    a = Assignment(tiles, costs, [[0, 0], [100, 0]], n_tiles=3, max_uses=1)
    a.greedy([0, 1])
    assert a.chosen_tiles().tolist() == [0, 2]
    saved = a.refine_pass([0, 1], np.ones(2))
    assert a.chosen_tiles().tolist() == [1, 0] and saved == pytest.approx(0.9)


def test_unplaceable_regions_can_be_forced():
    tiles = np.array([[0], [0]])
    a = Assignment(tiles, np.zeros((2, 1)), [[0, 0], [1, 0]], n_tiles=1, max_uses=1)
    assert a.greedy([0, 1]) == 1
    assert a.force([0, 1]) == 1 and a.check() == (1, 0)


def test_error_diffusion_mixes_tiles_to_keep_average_color():
    # Two tiles, black and white; every region is mid gray. Plain greedy gives
    # every region the same (slightly closer) tile; diffusion alternates.
    n = 64
    gray = rgb8_to_oklab([128, 128, 128])
    black, white = rgb8_to_oklab([0, 0, 0]), rgb8_to_oklab([255, 255, 255])
    means = np.stack([black, white])
    tiles = np.tile([0, 1], (n, 1))
    costs = ((means - gray) ** 2).sum(-1)[None].repeat(n, 0).astype(np.float32)
    order = np.argsort(costs[0])
    tiles, costs = tiles[:, order], costs[:, order]
    centers = grid_centers(n, 1.0)
    plain = Assignment(tiles, costs, centers, 2)
    plain.greedy(np.arange(n))
    assert len(set(plain.chosen_tiles())) == 1

    diffused = Assignment(tiles, costs, centers, 2)
    cell = centers.astype(int)[:, ::-1]
    diffused.greedy_diffused(
        np.arange(n), means[order][None].repeat(n, 0), np.tile(gray, (n, 1)), np.ones(n),
        0.0, cell, (8, 8),
    )  # fmt: skip
    share_white = (diffused.chosen_tiles() == 1).mean()
    shown = np.where(diffused.chosen_tiles()[:, None] == 1, white, black).mean(0)
    assert 0.25 < share_white < 0.75
    assert abs(shown[0] - gray[0]) < abs(black[0] - gray[0]) / 2


# Quality


def flat_ctx(image, columns=8, tile=10):
    return SliceContext(image, MosaicLayout(columns=columns), tile_width=tile)


def test_quality_rewards_faithful_mosaics():
    image = np.zeros((80, 80, 3), np.uint8)
    image[:, 40:] = (220, 60, 30)
    ctx = flat_ctx(image)
    regions = GridSlicer().apply(ctx.canvas(), ctx)  # 8 x 8 tiles of 10 px
    lab = target_raster(ctx, 8, 8)  # one sample per tile
    cells = np.repeat(np.repeat(lab.reshape(64, 1, 1, 3), 4, 1), 4, 2)
    placed = np.ones(64, bool)
    good = evaluate(regions, ctx, cells, placed)
    wrong = cells.copy()
    wrong[5] = rgb8_to_oklab([0, 255, 0])
    bad = evaluate(regions, ctx, wrong, placed)
    assert good.score < 1.0 and bad.score > good.score + 0.3  # one bad tile in 64
    assert bad.region_error[5] > 20
    assert good.ssim > 0.95 and good.covered == 1.0
    assert np.nanargmax(bad.region_error) == 5
    missing = evaluate(regions, ctx, cells, np.arange(64) != 3)
    assert missing.covered < 1.0


# End to end


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """A library of solid-color tiles of a few shapes, plus split black/white tiles."""
    root = tmp_path_factory.mktemp("tiles")
    rng = np.random.default_rng(3)
    for i, color in enumerate(rng.integers(0, 256, (120, 3))):
        size = [(60, 60), (90, 60), (60, 90), (120, 60)][i % 4]
        Image.new("RGB", size, tuple(int(c) for c in color)).save(root / f"solid{i}.png")
    split = np.zeros((60, 60, 3), np.uint8)
    split[:, :30] = 255
    Image.fromarray(split).save(root / "split.png")
    Image.fromarray(split[:, ::-1].copy()).save(root / "split_r.png")
    lib = TileLibrary(tmp_path_factory.mktemp("cache"))
    lib.set_roots([root])
    lib.update(workers=0)
    yield lib
    lib.close()


def target_image():
    image = np.zeros((120, 160, 3), np.uint8)
    image[:60, :80] = (230, 40, 40)
    image[:60, 80:] = (40, 200, 60)
    image[60:, :80] = (30, 40, 220)
    image[60:, 80:] = (240, 230, 50)
    return image


def test_matcher_end_to_end(library):
    ctx = flat_ctx(target_image(), columns=16, tile=20)  # 16 x 12 grid of 20 px tiles
    regions = GridSlicer().apply(ctx.canvas(), ctx)
    settings = MatchSettings(tint="none", max_uses=4, min_spacing=2.0, refine_seconds=1.0)
    result = Matcher(library).run(regions, ctx, settings)

    assert (result.tile >= 0).all() and set(result.tile) <= set(library.ids)
    assert result.stats["rule_violations"] == 0 and result.stats["most_uses"] <= 4
    assert result.regret and result.regret[0].exact  # small library: exact search
    # Each quadrant (48 regions, each tile used at most 4 times, so at least 12
    # tiles) gets close to the best the library allows: its 12 nearest colors.
    cx, cy = regions.center.T / [[ctx.width / 2], [ctx.height / 2]]  # mosaic px
    quad = (cy >= 1).astype(int) * 2 + (cx >= 1)
    colors = rgb8_to_oklab(library.thumbs[library.ids, 0, 0])
    for q, color in enumerate([(230, 40, 40), (40, 200, 60), (30, 40, 220), (240, 230, 50)]):
        target = rgb8_to_oklab(color)
        bound = np.sort(np.linalg.norm(colors - target, axis=1))[:12].mean()
        de = np.linalg.norm(result.tile_mean[quad == q] - target, axis=1).mean()
        assert de <= 1.3 * bound + 0.01, (q, de, bound)
    assert result.quality.covered == 1.0

    # Tinting toward each region's color can only bring the proxy closer.
    tinted = Matcher(library).run(regions, ctx, MatchSettings(tint="custom", tint_strength=0.5,
                                                              max_uses=4, min_spacing=2.0,
                                                              refine_seconds=1.0))  # fmt: skip
    assert tinted.quality.score < result.quality.score
    shift = tinted.tinted_mean() - tinted.tile_mean
    assert shift.shape == (len(regions), 3) and np.abs(shift).max() > 0


def test_matcher_on_a_transparent_source_matches_only_the_picture(library):
    # 16 x 12 tiles of 20 mosaic units (2 per source pixel). Hidden pixels hold
    # magenta; the picture starts at x = 75 px (150 units), inside the column of
    # tiles at 140..160, which overhangs the mask's edge and must match what shows.
    image = np.zeros((120, 160, 4), np.uint8)
    image[..., :3] = target_image()
    image[:, :75, :3] = (255, 0, 255)
    image[:, 75:, 3] = 255
    ctx = flat_ctx(image, columns=16, tile=20)
    regions = mask_regions(GridSlicer().apply(ctx.canvas(), ctx), ctx)
    assert len(regions) == 9 * 12  # columns from 140 units on
    settings = MatchSettings(tint="none", max_uses=8, refine_seconds=0.5)
    result = Matcher(library).run(regions, ctx, settings)
    assert (result.tile >= 0).all()
    edge = regions.center[:, 0] < 160
    top = regions.center[:, 1] < ctx.height / 2
    magenta = rgb8_to_oklab((255, 0, 255))
    for rows, color in ((edge & top, (230, 40, 40)), (edge & ~top, (30, 40, 220))):
        got = result.tile_mean[rows]
        near = np.linalg.norm(got - rgb8_to_oklab(color), axis=1)
        assert np.all(near < np.linalg.norm(got - magenta, axis=1))


def test_matcher_previews_the_run_without_changing_it(library):
    ctx = flat_ctx(target_image(), columns=16, tile=20)
    regions = GridSlicer().apply(ctx.canvas(), ctx)
    # Fewer allowed uses than regions need: the first assignment leaves regions waiting.
    settings = MatchSettings(tint="custom", tint_strength=0.3, max_uses=1, min_spacing=2.0,
                             refine_seconds=5.0, adaptive_rounds=2)  # fmt: skip
    plain = Matcher(library).run(regions, ctx, settings)
    previews = []
    result = Matcher(library).run(regions, ctx, settings, preview=previews.append)

    for name in ("tile", "rect", "mirrored", "cost", "tint_target"):  # same with or without
        np.testing.assert_array_equal(getattr(result, name), getattr(plain, name))
    stages = [p.stage for p in previews]
    runs = [s for i, s in enumerate(stages) if i == 0 or s != stages[i - 1]]  # repeats folded
    assert runs[:5] == [PreviewStage.SKETCH, PreviewStage.BEST, PreviewStage.ASSIGNED,
                        PreviewStage.WIDENING, PreviewStage.COMPLETE]  # fmt: skip
    assert set(runs[5:]) <= {PreviewStage.REFINING, PreviewStage.ADAPTIVE}
    assert PreviewStage.REFINING in stages and PreviewStage.ADAPTIVE in stages
    rank = list(PreviewStage)
    assert [rank.index(s) for s in stages] == sorted(rank.index(s) for s in stages)

    def first(stage):
        return next(p for p in previews if p.stage is stage)

    sketch, assigned, complete = (first(s) for s in (PreviewStage.SKETCH, PreviewStage.ASSIGNED,
                                                     PreviewStage.COMPLETE))  # fmt: skip
    best = [p for p in previews if p.stage is PreviewStage.BEST][-1]
    assert sketch.result is None and sketch.regions is regions
    right, low = (regions.center / [ctx.width / 2, ctx.height / 2]).T >= 1  # quadrant
    expected = target_image()[np.where(low, 90, 30), np.where(right, 120, 40)]
    np.testing.assert_allclose(sketch.target * 255, expected, atol=2)  # flat quadrants
    assert (best.result.tile >= 0).all()  # each region's own best: repeats allowed
    assert np.bincount(best.result.tile).max() > 1
    # Every region keeps showing a tile: regions the rules left waiting show their
    # favorite (the best tile) until widening replaces it.
    assert (assigned.result.tile >= 0).all() and (complete.result.tile >= 0).all()
    waiting = assigned.result.tile == best.result.tile
    assert waiting.any() and np.bincount(assigned.result.tile).max() > 1  # favorites repeat
    placed = ~waiting
    assert np.bincount(assigned.result.tile[placed]).max() == 1  # placed within the rules
    widenings = [p for p in previews if p.stage is PreviewStage.WIDENING]
    for frame in widenings:  # placed tiles stay; waiting ones change only to tentative tiles
        np.testing.assert_array_equal(frame.result.tile[placed], assigned.result.tile[placed])
        assert (frame.result.tile >= 0).all()
    assert (widenings[-1].result.tile[waiting] != best.result.tile[waiting]).any()
    assert complete.result.quality is None
    assert complete.result.tint == result.tint  # tiles show tinted, as they will
    # Snapshots are copies: nothing the run did afterwards shows in them.
    arrays = [getattr(p.result, name) for p in previews[1:] for name in ("tile", "cost")]
    arrays += [result.tile, result.cost]
    assert not any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[:i])


def test_matcher_skips_hidden_regions_and_uses_structure(library):
    image = np.zeros((60, 60, 3), np.uint8)
    image[:, :30] = 255
    ctx = flat_ctx(image, columns=1, tile=60)
    regions = RegionSet.from_rects([0, 0], [0, 0], 60, 60, z=[0, 1])  # one hides the other
    result = Matcher(library).run(regions, ctx, MatchSettings(tint="none", adaptive_rounds=0))
    assert result.tile[0] == -1 and result.stats["hidden"] == 1
    path = library.paths([result.tile[1]])[0]
    assert path.endswith("split.png") and not result.mirrored[1]


def test_matcher_settings_round_trip():
    s = MatchSettings(tint="custom", tint_strength=0.4)
    assert s.tint_value == 0.4 and MatchSettings().tint_value == 0.2
    assert s.copy().key() == s.key()

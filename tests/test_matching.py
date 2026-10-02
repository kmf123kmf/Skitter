"""Candidate search, constrained assignment, quality scoring, and the full matcher."""

import math
from types import SimpleNamespace

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
)
from skitter.core.matching.matcher import Matcher
from skitter.core.matching.quality import evaluate, target_raster
from skitter.core.matching.settings import MatchSettings
from skitter.core.slicing import MosaicLayout, RegionSet, SliceContext
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
    return SliceContext(image, MosaicLayout(tile_width=tile, columns=columns))


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
    offset = tinted.tint_offset()
    assert offset.shape == (len(regions), 3) and np.abs(offset).max() > 0


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

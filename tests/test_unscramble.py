"""Unscramble (core/animation/unscramble.py) and moving tiles by hand (moves.py)."""

import numpy as np
import pytest
from test_animation import pile_scene
from test_show import tiled_scene

from skitter.core.animation.moves import swap_plan
from skitter.core.animation.unscramble import (
    UnscrambleChoreography,
    batch_sizes,
    batch_times,
)

# The swap plan


def replay(places, moves):
    """Each batch of swaps applied in time order: the place of every tile after each."""
    now = places.copy()
    states = []
    for s in np.unique(moves.start):
        at = moves.start == s
        now[moves.tile[at]] = moves.target[at]
        states.append(now.copy())
    return states


def batched(n: int, batches: int) -> np.ndarray:
    """(n,) times: n tiles going home in `batches` batches one second apart."""
    return np.floor(np.arange(n) * batches / n)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("batches", [1, 5, 59])
def test_swap_plan_unscrambles_with_every_batch_putting_tiles_home(seed, batches):
    rng = np.random.default_rng(seed)
    n = 60
    places, moves = swap_plan(n, rng.permutation(n), batched(n, batches), rng)
    assert sorted(places.tolist()) == list(range(n))  # one tile per place
    home = places == np.arange(n)
    for state in replay(places, moves):
        assert sorted(state.tolist()) == list(range(n))  # still one tile per place
        now_home = state == np.arange(n)
        assert now_home.sum() > home.sum() and (now_home | ~home).all()  # more home, none left
        home = now_home
    assert home.all()
    # Every swap takes at least one tile home.
    for s in np.unique(moves.start):
        at = moves.start == s
        goes_home = moves.target[at] == moves.tile[at]
        assert goes_home.sum() >= at.sum() // 2


@pytest.mark.parametrize("centers", [False, True])
def test_swap_plan_tiles_go_home_in_order(centers):
    rng = np.random.default_rng(0)
    n = 80
    order = rng.permutation(n)
    times = batched(n, 8)
    grid = np.stack(np.meshgrid(np.arange(10.0), np.arange(8.0)), -1).reshape(-1, 2)
    _, moves = swap_plan(n, order, times, rng, centers=grid if centers else None,
                         reach=2.0)  # fmt: skip
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    home = moves.target == moves.tile  # moves taking a tile home: in its own batch
    np.testing.assert_array_equal(moves.start[home], times[rank[moves.tile[home]]])


def test_swap_plan_never_moves_a_tile_twice_at_once():
    rng = np.random.default_rng(3)
    n = 400
    grid = np.stack(np.meshgrid(np.arange(20.0), np.arange(20.0)), -1).reshape(-1, 2)
    for reach in (1.5, 100.0):
        _, moves = swap_plan(n, rng.permutation(n), batched(n, 4), rng, centers=grid,
                             reach=reach)  # fmt: skip
        for s in np.unique(moves.start):
            tiles = moves.tile[moves.start == s]
            assert len(np.unique(tiles)) == len(tiles)


# Batches


def test_batches_fill_the_duration_with_pauses_between():
    times = batch_times(500, duration=10.0, travel=0.8, pause=0.5)
    starts, sizes = np.unique(times, return_counts=True)
    assert starts[0] == 0 and starts[-1] == pytest.approx(10.0 - 0.8)  # ends at the end
    assert (np.diff(starts) >= 0.8 + 0.5 - 1e-9).all()  # a pause after every batch
    assert len(starts) == 8 and sizes.sum() == 500
    assert sizes.max() - sizes.min() <= 1  # no wind-down: alike


def test_batches_shrink_over_the_wind_down_to_a_single_swap():
    times = batch_times(500, duration=20.0, travel=0.8, pause=0.5, wind_down=6.0)
    _, sizes = np.unique(times, return_counts=True)
    assert sizes.sum() == 500 and sizes[-1] == 2  # one swap of two tiles, last
    assert (np.diff(sizes) <= 1).all()  # never growing
    assert sizes[0] == sizes[len(sizes) // 2]  # alike before the wind-down
    few = batch_sizes(5, batches=8, tail=3)
    assert few.sum() == 5 and few[-1] >= 2 and few.max() <= 3


# Unscramble


def test_unscramble_starts_scrambled_and_whole():
    scene = tiled_scene(24, 16)
    timeline = UnscrambleChoreography().timeline(scene)
    start = timeline.frame(0.0)
    assert start.at_rest().all() and not start.heights().any()
    # Every place holds exactly one tile, nearly all of them the wrong one.
    place = [int(np.argmin(np.hypot(*(scene.center - c).T))) for c in start.center]
    assert sorted(place) == list(range(len(scene)))
    assert (np.array(place) != np.arange(len(scene))).mean() > 0.9
    np.testing.assert_allclose(start.size, scene.size)  # tiles keep their own size


def test_unscramble_tiles_start_near_home():
    scene = tiled_scene(40, 30)

    def distances(reach):
        timeline = UnscrambleChoreography(reach=reach).timeline(scene)
        m = timeline.moves
        assert (timeline.places != np.arange(len(scene))).mean() > 0.9  # nearly all scrambled
        return np.hypot(*(scene.center[m.source] - scene.center[m.target]).T) / 10.0

    near, default, anywhere = distances(1.5), distances(6.0), distances(200.0)
    assert np.mean(default <= 6.0 + 1e-9) > 0.95  # within reach, nearly always
    # The last few still out of place may lie further apart than a small reach allows.
    assert np.median(near) < np.median(default) < np.median(anywhere)
    assert np.median(anywhere) > 10


def test_unscramble_rests_between_batches():
    scene = tiled_scene(24, 16)
    timeline = UnscrambleChoreography(pause=1.0).timeline(scene)
    starts = np.unique(timeline.moves.start)
    assert len(starts) > 3
    for s in starts[:-1]:
        still = timeline.frame(s + timeline.travel + 0.5)  # in the pause after the batch
        after = timeline.frame(s + timeline.travel + 0.9)
        np.testing.assert_allclose(still.center, after.center)
        mid = timeline.frame(s + timeline.travel / 2)
        assert (np.hypot(*(mid.center - still.center).T) > 1e-6).sum() >= 2  # moving


def test_unscramble_with_nothing_scrambled_is_the_mosaic_throughout():
    scene = tiled_scene(12, 8)
    timeline = UnscrambleChoreography(scrambled=0.0).timeline(scene)
    for t in np.linspace(0, timeline.duration, 11):
        np.testing.assert_allclose(timeline.frame(t).center, scene.center)


@pytest.mark.parametrize("order", ["center", "reading"])
def test_unscramble_fixes_tiles_in_its_order(order):
    scene = tiled_scene(24, 16)
    timeline = UnscrambleChoreography(order=order).timeline(scene)
    m = timeline.moves
    home = m.target == m.tile
    last_home = np.zeros(len(scene))
    np.maximum.at(last_home, m.tile[home], m.start[home])
    key = scene.distance if order == "center" else scene.reading_order
    moved = last_home > 0
    assert np.corrcoef(last_home[moved], key[moved])[0, 1] > 0.9


def test_unscramble_partners_pass_side_by_side_one_over_the_other():
    scene = tiled_scene(12, 8)
    timeline = UnscrambleChoreography(wind_down=0.0).timeline(scene)
    m = timeline.moves
    s = m.start[0]
    pair = np.flatnonzero(m.start == s)[:2]
    frame = timeline.frame(s + timeline.travel / 2)
    assert frame.at_rest().all() and not frame.heights().any()  # sliding on the table
    a, b = m.tile[pair]
    # Side by side, not through each other: their centers are apart mid-swap.
    gap = np.hypot(*(frame.center[a] - frame.center[b]))
    way = scene.center[m.target[pair[0]]] - scene.center[m.source[pair[0]]]
    assert gap > 0.2 * np.hypot(*way)
    np.testing.assert_allclose(frame.rotation, scene.rotation)  # no tilt


def test_unscramble_settings_of_earlier_versions_are_left_out_quietly():
    problems = []
    old = UnscrambleChoreography.from_values(
        {"motion": "carry", "lift": 2.0, "tilt": 9.0, "stack_height": 0.3, "explode": 2.0,
         "pile_spread": 30.0, "depth_shading": 20.0, "spread": 0.2, "order": "center"},
        problems,
    )  # fmt: skip
    assert not problems and old.spread == 0.2 and old.order == "center"
    # Saved before Pile spread had its name, "spread" was the pile's: left out too.
    older = UnscrambleChoreography.from_values({"stack_height": 0.3, "spread": 40.0}, problems)
    assert not problems and older.spread == UnscrambleChoreography().spread
    new = UnscrambleChoreography(spread=0.2)
    assert UnscrambleChoreography.from_values(new.values()).values() == new.values()


def test_unscramble_spread_loosens_its_order():
    scene = tiled_scene(24, 16)

    def fit(spread):
        m = UnscrambleChoreography(order="reading", spread=spread).timeline(scene).moves
        home = m.target == m.tile
        last_home = np.zeros(len(scene))
        np.maximum.at(last_home, m.tile[home], m.start[home])
        moved = last_home > 0
        return np.corrcoef(last_home[moved], scene.reading_order[moved])[0, 1]

    assert fit(0.0) > fit(0.05) > fit(0.5)


# Levels: tiles overlapping on the way (different sizes, a pile)


def split_grid():
    """A grid split by detail: big tiles and quarters of them."""
    from test_animation import make_scene

    from skitter.core.slicing import RegionSet

    xs, ys, ws = [], [], []
    for by in range(0, 120, 20):
        for bx in range(0, 160, 20):
            if (bx + by) // 20 % 3 == 0:
                xs.append(bx), ys.append(by), ws.append(20)
            else:
                for dy in (0, 10):
                    for dx in (0, 10):
                        xs.append(bx + dx), ys.append(by + dy), ws.append(10)
    w = np.array(ws, float)
    regions = RegionSet.from_rects(np.array(xs, float), np.array(ys, float), w, w)
    return make_scene(regions, np.arange(len(regions)) % 7)


@pytest.mark.parametrize("scene_of", [split_grid, pile_scene, lambda: tiled_scene(12, 8)])
@pytest.mark.parametrize("settings", [{}, {"order": "center", "reach": 30.0, "travel": 0.4}])
def test_unscramble_keeps_every_tile_at_its_level(scene_of, settings):
    """Photos in spots not their own hang over their neighbors (different sizes, a
    pile): every tile keeps one level throughout, sliding on the table, so none ever
    pops over or under another, and a pile ends in its own order."""
    from skitter.core.animation import TileFrame

    scene = scene_of()
    timeline = UnscrambleChoreography(duration=6.0, **settings).timeline(scene)
    order = timeline.frame(0.0).draw_order()
    rank = np.empty(len(scene), np.int64)
    rank[order] = np.arange(len(scene))
    lower, upper = scene.overlaps.T
    assert (rank[lower] < rank[upper]).all()  # a pile's levels keep its own order
    assert (timeline.places != np.arange(len(scene))).mean() > 0.9  # scrambled
    moved = False
    for t in np.linspace(0, timeline.duration, 121)[:-1]:
        frame = timeline.frame(t)
        np.testing.assert_array_equal(frame.draw_order(), order)
        assert not frame.heights().any() and frame.at_rest().all()  # on the table
        np.testing.assert_allclose(frame.size, scene.size)  # each photo its own size
        moved |= not np.allclose(frame.center, timeline.frame(0.0).center)
    assert moved
    final = timeline.frame(timeline.duration)
    np.testing.assert_allclose(final.center, TileFrame.final(scene).center)

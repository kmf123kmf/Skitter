"""Unscramble (core/animation/unscramble.py) and moving tiles by hand (moves.py)."""

import numpy as np
import pytest
from test_animation import overlapping, pile_scene
from test_show import tiled_scene

from skitter.core.animation.look import AnimationLook
from skitter.core.animation.moves import LayeredTimeline, layers, swap_plan
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
    places, moves = swap_plan(n, rng.permutation(n), batched(n, batches), 0.5, rng)
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
    _, moves = swap_plan(n, order, times, 0.5, rng, centers=grid if centers else None,
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
        _, moves = swap_plan(n, rng.permutation(n), batched(n, 4), 0.5, rng, centers=grid,
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
        frame = timeline.frame(s + timeline.travel + 0.5)  # in the pause after the batch
        assert frame.at_rest().all() and not frame.heights().any()
        assert (~timeline.frame(s + timeline.travel / 2).at_rest()).sum() >= 2  # moving


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


def test_unscramble_partners_pass_one_over_the_other():
    scene = tiled_scene(12, 8)
    timeline = UnscrambleChoreography(motion="carry", wind_down=0.0).timeline(scene)
    m = timeline.moves
    s = m.start[0]
    pair = np.flatnonzero(m.start == s)[:2]
    frame = timeline.frame(s + timeline.travel / 2)
    a, b = m.tile[pair]
    assert frame.heights()[a] > 0 and frame.heights()[b] > 0
    assert frame.heights()[a] != pytest.approx(frame.heights()[b])
    # Side by side, not through each other: their centers are apart mid-swap.
    gap = np.hypot(*(frame.center[a] - frame.center[b]))
    way = scene.center[m.target[pair[0]]] - scene.center[m.source[pair[0]]]
    assert gap > 0.2 * np.hypot(*way)


def test_unscramble_slides_stay_low_and_carries_stay_below_the_camera():
    scene = tiled_scene(12, 8)
    slide = UnscrambleChoreography(motion="slide").timeline(scene)
    tops = [slide.frame(t).heights().max() for t in np.linspace(0, slide.duration, 41)]
    assert 0 < max(tops) <= 0.01 * scene.tile_size[0] + 1e-9  # a hover, no visible shadow
    look = AnimationLook(camera_height=0.5)
    carry = UnscrambleChoreography(lift=10.0).timeline(scene, look)
    extent = max(np.ptp(scene.corners[..., 0]), np.ptp(scene.corners[..., 1]))
    tops = [carry.frame(t).heights().max() for t in np.linspace(0, carry.duration, 41)]
    assert max(tops) <= 0.25 * 0.5 * extent + 1e-9


def test_unscramble_carried_tiles_tilt_and_land_straight():
    scene = tiled_scene(12, 8)
    timeline = UnscrambleChoreography(tilt=20.0, wind_down=0.0).timeline(scene)
    m = timeline.moves
    k = int(np.argmax(np.abs(m.tilt)))
    mid = timeline.frame(m.start[k] + timeline.travel / 2)
    assert abs(mid.rotation[m.tile[k]] - scene.rotation[m.tile[k]]) > 0.05
    landed = timeline.frame(m.start[k] + timeline.travel + 1e-6)
    assert landed.rotation[m.tile[k]] == pytest.approx(scene.rotation[m.tile[k]])


# Piles: exploded into layers


def test_layers_put_every_tile_above_the_tiles_under_it():
    scene = pile_scene()
    layer = layers(scene)
    lower, upper = scene.overlaps.T
    assert (layer[upper] > layer[lower]).all()  # so one layer's tiles never overlap
    on_table = np.setdiff1d(np.arange(len(scene)), upper)
    assert (layer[on_table] == 0).all() and layer.max() > 2
    assert not layers(tiled_scene(6, 4)).any()  # a grid is one layer


def test_unscramble_explodes_a_pile_and_settles_it():
    scene = pile_scene()
    layer = layers(scene)
    timeline = UnscrambleChoreography(pile_spread=20.0, pause=1.0).timeline(scene)
    assert isinstance(timeline, LayeredTimeline)
    start = timeline.frame(0.0)  # a flat, scrambled pile
    assert start.at_rest().all() and not start.heights().any()
    assert (timeline.places != np.arange(len(scene))).mean() > 0.9
    np.testing.assert_allclose(start.center, scene.center[timeline.places])
    # Apart (before the first swaps): layers evenly spaced, the higher spread further out.
    apart = timeline.frame(timeline.explode)
    place = timeline.places
    np.testing.assert_allclose(apart.heights(), (layer[place] + 1) * timeline.gap)
    middle = np.array([(scene.bounds[0] + scene.bounds[2]) / 2,
                       (scene.bounds[1] + scene.bounds[3]) / 2])  # fmt: skip
    out = np.hypot(*(apart.center - middle).T) / np.maximum(
        np.hypot(*(scene.center[place] - middle).T), 1e-9
    )
    top = layer[place] == layer.max()
    np.testing.assert_allclose(out[top], 1.2)
    np.testing.assert_allclose(out[layer[place] == 0], 1.0)
    # A photo takes on the turn of the spot it lies in.
    np.testing.assert_allclose(apart.rotation, scene.rotation[place])
    assert timeline.frame(timeline.duration - 1e-9).heights().max() < 1e-6 * timeline.gap


def test_unscramble_swaps_a_pile_within_its_layers():
    scene = pile_scene()
    layer = layers(scene)
    m = UnscrambleChoreography().timeline(scene).moves
    assert len(m) and (layer[m.source] == layer[m.target]).all()
    # The tile going home travels above its layer, its partner below it.
    level = layer[m.source] + 1.0
    assert ((m.lane > level) & (m.lane < level + 1))[~m.low].all()
    assert ((m.lane < level) & (m.lane > level - 1))[m.low].all()


@pytest.mark.parametrize("settings", [{}, {"order": "center"}, {"reach": 30.0, "travel": 0.4}])
def test_unscramble_pile_tiles_never_pass_through_each_other(settings):
    """Flat tiles at different heights never touch: a collision is two overlapping tiles
    swapping which is higher between two moments close together."""
    from scipy.spatial import cKDTree

    scene = pile_scene(seed=5)
    timeline = UnscrambleChoreography(**settings).timeline(scene)
    reach = float(np.hypot(*scene.size.T).max())
    times = np.linspace(0, timeline.duration, 1500)[1:-1]
    before = timeline.frame(times[0])
    for t in times[1:]:
        now = timeline.frame(t)
        i, j = cKDTree(now.center).query_pairs(reach, output_type="ndarray").T
        h0, h1 = before.heights(), now.heights()
        crossed = np.sign(h0[i] - h0[j]) * np.sign(h1[i] - h1[j]) < 0
        i, j = i[crossed], j[crossed]
        hit = overlapping(now.center[i], now.size[i], now.rotation[i], now.center[j],
                          now.size[j], now.rotation[j])  # fmt: skip
        assert not hit.any(), f"tiles pass through each other at {t:.3f} s"
        before = now


def test_unscramble_depth_shading_darkens_lower_layers():
    scene = pile_scene()
    layer = layers(scene)
    timeline = UnscrambleChoreography(depth_shading=40.0).timeline(scene)
    apart = timeline.frame(timeline.explode)
    shade = apart.tint[:, 3]
    deep = layer[timeline.places]
    assert shade[deep == 0].mean() > shade[deep == deep.max()].mean() >= 0
    assert (apart.tint[:, :3] == 0).all() and shade.max() <= 0.4
    plain = UnscrambleChoreography(depth_shading=0.0).timeline(scene)
    assert not plain.frame(plain.explode).tint.any()


def test_unscramble_settings_saved_before_pile_spread_keep_it():
    problems = []
    old = UnscrambleChoreography.from_values({"spread": 40.0, "order": "center"}, problems)
    assert not problems
    assert old.pile_spread == 40.0 and old.spread == 0.05 and old.order == "center"
    new = UnscrambleChoreography(pile_spread=30.0, spread=0.2)
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

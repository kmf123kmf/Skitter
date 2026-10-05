"""The mosaic scene and the animation framework (choreographies and timelines)."""

import math

import numpy as np
import pytest

from skitter.core.animation import (
    Choreography,
    FlightTimeline,
    TileFrame,
    TossTimeline,
    choreography_types,
    get_choreography,
    landing_times,
)
from skitter.core.animation.assemble import ORDERS, AssembleChoreography
from skitter.core.animation.deal import DealChoreography, DealTimeline
from skitter.core.matching.matcher import MatchResult
from skitter.core.scene import MosaicScene
from skitter.core.slicing import MosaicLayout, RegionSet, SliceContext


def make_scene(regions, tile, shift=None, error=None) -> MosaicScene:
    n = len(regions)
    mean = np.tile(np.float32([0.6, 0.0, 0.0]), (n, 1))
    mean[:, 0] = np.linspace(0.2, 0.9, n)
    quality = None
    if error is not None:
        from types import SimpleNamespace

        quality = SimpleNamespace(region_error=np.asarray(error, np.float32))
    result = MatchResult(
        regions=regions, tile=np.asarray(tile),
        rect=np.tile([0, 0, 1, 1], (n, 1)).astype(np.float32),
        mirrored=np.arange(n) % 2 == 1, cost=np.zeros(n, np.float32), tile_mean=mean,
        tint_target=mean if shift is None else mean + np.asarray(shift, np.float32),
        tint=1.0, quality=quality, regret=[],
    )  # fmt: skip
    ctx = SliceContext(np.zeros((40, 60, 3), np.uint8), MosaicLayout(columns=6), tile_width=10)
    return MosaicScene.from_result(result, ctx)


def grid_scene(columns=6, rows=4, seed=0) -> MosaicScene:
    regions = RegionSet.grid(60, 40, columns, rows)
    rng = np.random.default_rng(seed)
    return make_scene(regions, rng.integers(0, 5, columns * rows))


# Scene


def test_scene_lists_placed_tiles_bottom_to_top():
    regions = RegionSet.from_rects([0, 10, 20, 30], 0, 10, 10, z=[3, 1, 2, 0])
    scene = make_scene(regions, [7, 8, -1, 9], shift=[[0.05, 0, 0]] * 4, error=[1, 2, 3, 4])
    assert scene.region.tolist() == [3, 1, 0]  # by z, without the unplaced region 2
    assert scene.slot.tolist() == [9, 8, 7]
    assert scene.mirrored.tolist() == [True, True, False]
    np.testing.assert_allclose(scene.center[:, 0], [35, 15, 5])
    np.testing.assert_allclose(scene.error, [4, 2, 1])
    np.testing.assert_allclose(scene.tint_shift, [[0.05, 0, 0]] * 3, atol=1e-6)
    assert scene.bounds == (0.0, 0.0, 40.0, 10.0)
    assert scene.canvas == (60.0, 40.0) and scene.tile_size == (10.0, 10.0)


def test_scene_ordering_attributes():
    scene = grid_scene()
    # Reading order: rows top to bottom, each left to right (the grid's own order).
    assert scene.reading_order.tolist() == list(range(len(scene)))
    assert scene.distance.min() >= 0 and scene.distance.max() <= 1
    np.testing.assert_allclose(scene.position[0], [5 / 60, 5 / 40])
    for slot in np.unique(scene.slot):
        uses = scene.use_index[scene.slot == slot]
        assert sorted(uses.tolist()) == list(range(len(uses)))
    assert scene.lightness.shape == (len(scene),)


# Timelines


def test_registry_lists_assemble():
    assert AssembleChoreography in choreography_types()
    assert get_choreography("assemble") is AssembleChoreography
    with pytest.raises(KeyError):
        get_choreography("nope")


def assert_frame_equal(a: TileFrame, b: TileFrame):
    np.testing.assert_allclose(a.center, b.center, atol=1e-9)
    np.testing.assert_allclose(a.size, b.size, atol=1e-9)
    np.testing.assert_allclose(a.rotation, b.rotation, atol=1e-9)
    np.testing.assert_allclose(a.alpha, b.alpha, atol=1e-9)
    np.testing.assert_allclose(a.tint, b.tint, atol=1e-9)
    np.testing.assert_array_equal(a.draw_order(), b.draw_order())


# Every registered choreography with its defaults, plus Assemble's other motion.
CONFIGS = [(cls, {}) for cls in choreography_types()] + [
    (AssembleChoreography, {"motion": "glide"}),
    (AssembleChoreography, {"flips": 2}),
    (AssembleChoreography, {"motion": "glide", "flips": 1}),
    (AssembleChoreography, {"motion": "drop"}),
    (AssembleChoreography, {"motion": "drop", "slant": 0.0, "flips": 1}),
    (AssembleChoreography, {"order": "random"}),
    (AssembleChoreography, {"order": "random", "motion": "drop"}),
    (AssembleChoreography, {"motion": "drop", "bounce": (0.1, 0.7), "wobble": (4.0, 10.0)}),
    (DealChoreography, {"face_down": True, "spin": (1.0, 2.0)}),
    (DealChoreography, {"decks": 3, "order": "random"}),
    (DealChoreography, {"decks": 4, "arrangement": "side", "order": "far"}),
    (DealChoreography, {"position": "center", "decks": 2, "neatness": 0.0}),
    (DealChoreography, {"position": "top_right", "offset": -0.2, "order": "reading"}),
    (DealChoreography, {"position": "orbit", "face_down": True}),
    (DealChoreography, {"position": "orbit", "decks": 3, "orbit_turns": 2.5, "order": "far"}),
    (DealChoreography, {"position": "orbit", "radius": 0.2, "orbit_direction": "counter"}),
    (DealChoreography, {"position": "orbit", "decks": 2, "order": "random", "orbit_turns": 0}),
]
CONFIG_IDS = [
    f"{cls.id}{'-' + '-'.join(map(str, kw.values())) if kw else ''}" for cls, kw in CONFIGS
]


@pytest.mark.parametrize(("cls", "settings"), CONFIGS, ids=CONFIG_IDS)
def test_every_choreography_ends_on_the_finished_mosaic(cls: type[Choreography], settings):
    scene = grid_scene()
    timeline = cls(**settings).timeline(scene)
    assert timeline.duration > 0
    final = TileFrame.final(scene)
    assert_frame_equal(timeline.frame(timeline.duration), final)
    last = timeline.frame(timeline.duration)
    assert last.at_rest().all() and not last.heights().any()  # all down: no shadows
    assert np.all(last.facings() == 1)  # face up
    assert_frame_equal(timeline.frame(timeline.duration + 5), final)  # clamped
    start = timeline.frame(0.0)
    assert len(start) == len(scene)
    for t in np.linspace(0, timeline.duration, 7):
        order = timeline.frame(t).draw_order()
        assert sorted(order.tolist()) == list(range(len(scene)))  # a permutation
    # Pure function of time: the same moment twice is the same frame.
    assert_frame_equal(timeline.frame(1.234), timeline.frame(1.234))


@pytest.mark.parametrize("order", [value for value, _ in ORDERS])
def test_assemble_orders_and_settings(order):
    scene = grid_scene()
    timeline = AssembleChoreography(order=order, duration=5, travel=1, spread=0).timeline(scene)
    assert timeline.duration == pytest.approx(5)
    start = timeline.frame(0)
    assert np.all(start.alpha == 0)  # nothing visible yet
    assert np.all(np.hypot(*(start.center - (30, 20)).T) > 30)  # all start outside


def test_assemble_lands_center_first_and_is_seeded():
    scene = grid_scene()
    make = AssembleChoreography(order="center", duration=4, travel=1, spread=0)
    timeline = make.timeline(scene)
    first = np.argmin(timeline.delay)
    assert scene.distance[first] == scene.distance.min()
    same = make.timeline(scene)
    assert_frame_equal(timeline.frame(2), same.frame(2))
    other = AssembleChoreography(order="center", duration=4, travel=1, seed=2).timeline(scene)
    assert not np.allclose(other.frame(0).center, timeline.frame(0).center)


def test_flight_draws_landed_tiles_below_tiles_in_the_air():
    scene = grid_scene(columns=2, rows=1)
    start = TileFrame.final(scene).replace(center=scene.center + 100, alpha=np.zeros(2))
    timeline = FlightTimeline(scene, start, delay=[0, 2], travel=1)
    mid = timeline.frame(1.5)  # tile 0 landed, tile 1 waiting
    assert mid.draw_order().tolist() == [0, 1]
    np.testing.assert_allclose(mid.center[0], scene.center[0])
    assert mid.alpha[1] == 0
    timeline = FlightTimeline(scene, start, delay=[2, 0], travel=1)
    assert timeline.frame(1.5).draw_order().tolist() == [1, 0]  # tile 1 landed: below 0
    assert timeline.frame(3).order is None  # all landed: stacking order


def test_empty_scene_has_an_empty_timeline():
    scene = make_scene(RegionSet.from_rects([0], 0, 10, 10), [-1])
    assert len(scene) == 0
    timeline = AssembleChoreography().timeline(scene)
    assert timeline.duration == 0 and len(timeline.frame(0)) == 0


# Overlaps and stacking


def pile_scene(seed=3) -> MosaicScene:
    from skitter.core.slicing.operations import PileSlicer

    ctx = SliceContext(np.zeros((40, 60, 3), np.uint8), MosaicLayout(columns=6), tile_width=10)
    regions = PileSlicer(seed=seed).apply(ctx.canvas(), ctx)
    return make_scene(regions, np.arange(len(regions)) % 7)


def brute_force_overlaps(scene: MosaicScene) -> set[tuple[int, int]]:
    """Every pair tested with the separating axis theorem (slow, obviously complete)."""
    found = set()
    for i in range(len(scene)):
        for j in range(i + 1, len(scene)):
            axes = []
            for r in (scene.rotation[i], scene.rotation[j]):
                axes += [np.array([np.cos(r), np.sin(r)]), np.array([-np.sin(r), np.cos(r)])]
            gap = False
            for axis in axes:
                proj = [scene.corners[k] @ axis for k in (i, j)]
                if proj[0].max() <= proj[1].min() + 1e-9 or proj[1].max() <= proj[0].min() + 1e-9:
                    gap = True
            if not gap:
                found.add((i, j))
    return found


def test_overlaps_ignore_touching_grid_cells():
    assert len(grid_scene().overlaps) == 0


def test_overlaps_handle_rotation():
    # Two diamonds whose bounding boxes overlap but whose shapes don't, and two that do.
    regions = RegionSet.from_arrays([[10, 10], [24.5, 10], [40, 10], [48, 10]], (10, 10),
                                    rotation=np.pi / 4)  # fmt: skip
    scene = make_scene(regions, [0, 1, 2, 3])
    assert scene.overlaps.tolist() == [[2, 3]]


def test_overlaps_match_brute_force_on_a_pile():
    scene = pile_scene()
    assert len(scene) > 20
    found = {tuple(p) for p in scene.overlaps.tolist()}
    assert found == brute_force_overlaps(scene)
    assert all(lower < upper for lower, upper in found)


def test_landing_order_follows_preference_but_lands_bottom_first():
    from skitter.core.animation import landing_order

    regions = RegionSet.from_rects([0, 5, 20, 40], 0, 10, 10)  # 0 and 1 overlap
    scene = make_scene(regions, [0, 1, 2, 3])
    order = landing_order(scene, [5.0, 1.0, 3.0, 0.5])
    assert order.tolist() == [2, 3, 1, 0]  # 3, 2, then 0 before the 1 lying on it
    ties = [0, 0, 0, 0]
    assert landing_order(scene, ties, gap=0).tolist() == [0, 1, 2, 3]  # ties: stacking order
    # By default a tile waits a few landings after the one it lies on, while others can go.
    assert landing_order(scene, ties).tolist() == [0, 3, 1, 2]
    assert landing_order(scene, ties, gap=1).tolist() == [0, 2, 1, 3]


def big_pile(seed=1, columns=20) -> MosaicScene:
    from skitter.core.slicing.operations import PileSlicer

    ctx = SliceContext(np.zeros((400, 600, 3), np.uint8), MosaicLayout(columns=columns),
                       tile_width=10)  # fmt: skip
    regions = PileSlicer(seed=seed).apply(ctx.canvas(), ctx)
    return make_scene(regions, np.arange(len(regions)) % 9)


def nearby_soon(scene, rank, window: int, near: float = 1.5) -> float:
    """Share of landings with another one within `window` landings and `near` tiles."""
    centers = scene.center[np.argsort(rank)]
    tile = np.sqrt(np.prod(scene.size, axis=1)).mean()
    hit = np.zeros(len(rank), bool)
    for w in range(1, window + 1):
        close = np.hypot(*(centers[w:] - centers[:-w]).T) < near * tile
        hit[w:] |= close
        hit[:-w] |= close
    return float(hit.mean())


def landing_ranks(timeline) -> np.ndarray:
    rank = np.empty(len(timeline.delay), np.int64)
    rank[np.argsort(timeline.delay, kind="stable")] = np.arange(len(rank))
    return rank


@pytest.mark.parametrize("order", [o for o, _ in ORDERS])
def test_every_order_on_a_pile_is_either_random_like_chance_or_directed(order):
    """Random orders land close together in space and time just as often as chance has
    it (the bottom-first rule must not clump them); directed ones follow their order.
    A new ordering lands in one of the two groups (assemble.RANDOM_ORDERS)."""
    from skitter.core.animation.assemble import RANDOM_ORDERS

    if order in RANDOM_ORDERS:
        for seed in (1, 2):
            scene = big_pile(seed)
            n = len(scene)
            assert len(scene.overlaps) > 2 * n
            ranks = [landing_ranks(AssembleChoreography(order=order, seed=s).timeline(scene))
                     for s in (seed, seed + 10)]  # fmt: skip
            shuffled = [np.random.default_rng(s).permutation(n) for s in range(4)]
            for share in (0.017, 0.05, 0.1):  # about 0.1, 0.3 and 0.6 s of an 8 s Assemble
                window = max(1, round(share * n))
                got = np.mean([nearby_soon(scene, r, window) for r in ranks])
                chance = np.mean([nearby_soon(scene, r, window) for r in shuffled])
                assert abs(got - chance) < 0.05, (share, got, chance)
    else:
        scene = big_pile()
        choreography = AssembleChoreography(order=order, spread=0.0)
        key = choreography._order_key(scene, np.random.default_rng(0))
        rank = landing_ranks(choreography.timeline(scene))
        assert np.corrcoef(rank, np.argsort(np.argsort(key)))[0, 1] > 0.8


def test_random_landing_order_is_valid_reproducible_and_uniform():
    from skitter.core.animation import landing_order, random_landing_order

    scene = big_pile()
    n = len(scene)
    rank = random_landing_order(scene, 3)
    lower, upper = scene.overlaps.T
    assert sorted(rank.tolist()) == list(range(n)) and np.all(rank[lower] < rank[upper])
    assert np.array_equal(rank, random_landing_order(scene, 3))
    assert not np.array_equal(rank, random_landing_order(scene, 4))
    # Random preferences through landing_order are not a random order: tiles held back
    # land right beside the ones they lie on (why random orders don't use it).
    biased = landing_order(scene, np.random.default_rng(3).random(n))
    window = max(1, n // 60)
    chance = np.mean([nearby_soon(scene, np.random.default_rng(s).permutation(n), window)
                      for s in range(4)])  # fmt: skip
    assert nearby_soon(scene, biased, window) > chance + 0.2
    assert abs(nearby_soon(scene, rank, window) - chance) < 0.06
    # Without overlaps it is a plain random permutation.
    grid = grid_scene(columns=20, rows=15)
    plain = random_landing_order(grid, 1)
    assert abs(np.corrcoef(plain, np.arange(len(grid)))[0, 1]) < 0.2


def test_pile_landings_spread_over_the_mosaic():
    """Consecutive landings are about as far apart as without the overlap rule: the
    sequence doesn't climb one stack after another."""
    from skitter.core.animation import landing_order
    from skitter.core.slicing.operations import PileSlicer

    ctx = SliceContext(np.zeros((400, 600, 3), np.uint8), MosaicLayout(columns=16), tile_width=10)
    regions = PileSlicer().apply(ctx.canvas(), ctx)
    scene = make_scene(regions, np.arange(len(regions)) % 9)
    rng = np.random.default_rng(0)
    preference = scene.distance + rng.uniform(-0.15, 0.15, len(scene))  # center outward

    def median_step(rank):
        centers = scene.center[np.argsort(rank)]
        return np.median(np.hypot(*np.diff(centers, axis=0).T))

    ideal = np.argsort(np.argsort(preference, kind="stable"), kind="stable")
    clumped = median_step(landing_order(scene, preference, gap=0))
    spread = median_step(landing_order(scene, preference))
    assert clumped < 0.4 * median_step(ideal)  # what the overlap rule alone does
    assert spread > 0.75 * median_step(ideal)
    rank = landing_order(scene, preference)
    assert np.corrcoef(rank, ideal)[0, 1] > 0.8  # the preferred order is kept


def test_landing_order_on_a_pile_is_complete_and_respects_overlaps():
    from skitter.core.animation import landing_order

    scene = pile_scene()
    order = landing_order(scene, np.random.default_rng(0).random(len(scene)))
    assert sorted(order.tolist()) == list(range(len(scene)))
    lower, upper = scene.overlaps.T
    assert np.all(order[lower] < order[upper])


def test_assemble_fits_short_durations():
    scene = grid_scene()
    for duration in (0.5, 1.0, 1.6):  # shorter than flight time + settling
        timeline = AssembleChoreography(duration=duration, travel=1.5).timeline(scene)
        assert timeline.duration == pytest.approx(duration)


@pytest.mark.parametrize("make", [grid_scene, pile_scene])
@pytest.mark.parametrize("motion", ["toss", "glide"])
def test_assemble_lands_tiles_at_an_even_pace(make, motion):
    scene = make()
    make = AssembleChoreography(duration=8, travel=1.5, motion=motion, wind_down=0)
    timeline = make.timeline(scene)
    settle = getattr(timeline, "settle", np.zeros(len(scene)))  # Toss: bounces and wobble
    assert timeline.duration == pytest.approx(8)  # the last tile comes to rest at the end
    impact = timeline.delay + timeline.travel
    assert (impact + settle).max() == pytest.approx(8)
    landing = np.sort(impact)
    assert landing[0] == pytest.approx(1.5)
    np.testing.assert_allclose(np.diff(landing), np.diff(landing).mean(), atol=1e-9)  # even


@pytest.mark.parametrize("make", [grid_scene, pile_scene])
@pytest.mark.parametrize("motion", ["toss", "glide"])
def test_assemble_winds_down_to_the_last_gap(make, motion):
    scene = make()
    choreography = AssembleChoreography(
        duration=8, travel=1.5, motion=motion, wind_down=0.3, last_gap=0.8
    )
    timeline = choreography.timeline(scene)
    settle = getattr(timeline, "settle", np.zeros(len(scene)))
    impact = timeline.delay + timeline.travel
    assert (impact + settle).max() == pytest.approx(8)  # still fits the duration
    gaps = np.diff(np.sort(impact))
    assert gaps[-1] == pytest.approx(0.8, rel=0.15)  # about, as the window is fitted
    assert np.all(np.diff(gaps[-5:]) > 0)  # slowing down to the end
    assert gaps[0] < gaps[-1] / 2


def test_landing_times_pace():
    times = landing_times(2000, 6.0, 1.2, 0.25)
    gaps = np.diff(times)
    assert times[0] == 0 and times[-1] == 6.0
    assert np.all(gaps >= 0)
    assert gaps[-1] == pytest.approx(0.25)  # exactly the last gap
    steady = times < 6.0 - 1.2
    np.testing.assert_allclose(gaps[steady[1:]], gaps[0], rtol=1e-6)  # even until the wind-down
    assert np.all(np.diff(gaps[~steady[:-1]]) > -1e-12)  # then slowing down


@pytest.mark.parametrize(
    ("n", "wind_down", "last_gap"),
    [(500, 0.0, 0.25), (500, 1.2, 0.0), (10, 1.2, 0.25), (1, 1.2, 0.25), (0, 1.2, 0.25)],
)
def test_landing_times_even_without_a_wind_down(n, wind_down, last_gap):
    # No wind-down, no gap, or landings already further apart than the last gap.
    expected = np.linspace(0.0, 6.0, n) if n > 1 else np.zeros(n)
    np.testing.assert_allclose(landing_times(n, 6.0, wind_down, last_gap), expected)


def test_landing_times_last_gap_is_at_most_half_the_wind_down():
    gaps = np.diff(landing_times(1000, 6.0, 0.4, 3.0))
    assert gaps[-1] == pytest.approx(0.2)


@pytest.mark.parametrize(("cls", "settings"), CONFIGS, ids=CONFIG_IDS)
def test_every_choreography_keeps_overlapping_tiles_in_stacking_order(cls, settings):
    # If a lower tile ever drew above a tile that covers it, it would have to pop
    # underneath later (at the latest on the last frame). The one exception is a
    # lower tile higher in the air (nearer the camera): it rightly covers the other,
    # and their order changes only as their heights cross, continuously.
    scene = pile_scene()
    lower, upper = scene.overlaps.T
    assert len(lower)
    timeline = cls(**settings).timeline(scene)
    for t in np.linspace(0, timeline.duration, 97):
        frame = timeline.frame(t)
        rank = np.empty(len(scene), np.int64)
        rank[frame.draw_order()] = np.arange(len(scene))
        visible = (frame.alpha[lower] > 0) & (frame.alpha[upper] > 0)
        height = frame.heights()
        nearer = height[lower] > height[upper]
        wrong = visible & (rank[upper] < rank[lower]) & ~nearer
        assert not wrong.any(), f"{wrong.sum()} lower tiles drawn on top at {t:.2f} s"


def test_assemble_tiles_are_opaque_in_flight():
    scene = grid_scene()
    timeline = AssembleChoreography().timeline(scene)
    for t in np.linspace(0, timeline.duration, 41):
        alpha = timeline.frame(t).alpha
        assert set(np.unique(alpha)) <= {0.0, 1.0}  # hidden until it sets off, then opaque


# Toss: gravity, bounces, settling


def single_toss(**kw):
    from skitter.core.animation import TossTimeline

    scene = grid_scene(columns=1, rows=1)
    settings = dict(delay=[0.0], travel=2.0, apex=10.0, bounce=0.5, wobble=0.2) | kw
    start = scene.center + [[-100.0, 0.0]]
    return scene, TossTimeline(scene, start, scene.rotation + 3.0, **settings)


def test_toss_falls_under_constant_gravity():
    scene, timeline = single_toss(bounce=0.0, wobble=0.0)
    times = np.linspace(0, 2.0, 41)
    height = np.array([timeline.frame(t).heights()[0] for t in times])
    assert height[0] == 0 and height[20] == pytest.approx(10.0) and height[-1] == 0  # apex mid-way
    accel = np.diff(height, 2) / (times[1] - times[0]) ** 2
    np.testing.assert_allclose(accel, -timeline.gravity, rtol=1e-6)  # parabola: constant g
    assert timeline.gravity == pytest.approx(8 * 10.0 / 2.0**2)
    # Sideways at constant speed, and spinning at a constant rate until impact.
    x = np.array([timeline.frame(t).center[0, 0] for t in times])
    np.testing.assert_allclose(np.diff(x), np.diff(x)[0])
    turn = np.array([timeline.frame(t).rotation[0] for t in times])
    np.testing.assert_allclose(np.diff(turn), np.diff(turn)[0])


def test_toss_bounces_lower_each_time_then_rests():
    scene, timeline = single_toss(wobble=0.0)
    speeds = timeline.bounce_speed[0]
    speeds = speeds[speeds > 0]
    assert len(speeds) >= 2
    np.testing.assert_allclose(speeds[1:] / speeds[:-1], 0.5)  # each keeps half the speed
    peaks = [speeds[k] ** 2 / (2 * timeline.gravity) for k in range(len(speeds))]
    assert peaks[0] == pytest.approx(10.0 * 0.25)  # heights shrink by bounce²
    # Sample the first bounce: up to its peak, back to the table.
    start = 2.0 + timeline.bounce_start[0, 0]
    mid = start + timeline.bounce_time[0, 0] / 2
    assert timeline.frame(mid).heights()[0] == pytest.approx(peaks[0])
    assert not timeline.frame(mid).at_rest()[0]  # still moving: draws above tiles at rest
    end = timeline.frame(2.0 + timeline.settle[0])
    assert end.heights()[0] == 0 and end.at_rest()[0]
    assert timeline.duration == pytest.approx(2.0 + timeline.settle[0])


def stacked(motion, order, bounce=(0.1, 0.7), wobble=(4.0, 10.0)):
    scene = pile_scene()
    choreography = AssembleChoreography(motion=motion, order=order, bounce=bounce, wobble=wobble)
    return scene, choreography.timeline(scene)


@pytest.mark.parametrize("motion", ["toss", "drop"])
@pytest.mark.parametrize("order", ["random", "center"])
def test_covered_tiles_stop_hopping_first_and_rocking_soon_after(motion, order):
    """A tile something lands on is flat on the table by then (else, seen from the
    camera, it would grow out from under it) and still within COVER_DAMP after it;
    rocking may go on until then (as if pressed down)."""
    scene, timeline = stacked(motion, order)
    final = TileFrame.final(scene)
    lower, upper = scene.overlaps.T
    impact = timeline.delay + timeline.travel
    damp = TossTimeline.COVER_DAMP
    rocked_under = 0
    for t in np.linspace(0, timeline.duration, 900):
        frame = timeline.frame(t)
        covered = (impact[upper] <= t) & (impact[lower] <= t)
        assert np.all(frame.heights()[lower[covered]] == 0)
        turned = np.abs(frame.rotation[lower] - final.rotation[lower]) > 1e-9
        assert not np.any(turned & (impact[upper] + damp <= t))
        rocked_under += int((turned & covered).sum())
    assert rocked_under > 0  # rocking goes on a little after a tile lands on it
    # Nothing moved: impacts are where the plan put them, and the end is the mosaic.
    assert timeline.frame(timeline.duration) is timeline.final


@pytest.mark.parametrize("motion", ["toss", "drop"])
def test_stacked_settling_has_no_jumps(motion):
    from skitter.core.animation import TossTimeline as Toss

    scene, timeline = stacked(motion, "random")
    free = Toss(scene, timeline.start_center, timeline.start_rotation, timeline.delay,
                timeline.travel, timeline.apex, 0.0, timeline.wobble, timeline.flips,
                timeline.start_height)  # fmt: skip
    dt = 2e-3
    fastest_fall = math.sqrt(2 * timeline.gravity * timeline.apex) * dt
    steps = {"height": 0.0, "turn": 0.0, "free turn": 0.0}
    times = np.arange(0, timeline.duration, dt)
    previous = timeline.frame(0.0), free.frame(0.0)
    for t in times[1:]:
        now = timeline.frame(t), free.frame(t)
        moved = {
            "height": np.abs(now[0].heights() - previous[0].heights()).max(),
            "turn": np.abs(now[0].rotation - previous[0].rotation).max(),
            "free turn": np.abs(now[1].rotation - previous[1].rotation).max(),
        }
        steps = {k: max(v, moved[k]) for k, v in steps.items()}
        previous = now
    assert steps["height"] <= fastest_fall * 1.01  # never faster than a fall: no jumps
    assert steps["turn"] <= steps["free turn"] * 1.05  # no faster than uncovered tiles turn


def test_each_tile_bounces_with_its_own_springiness():
    scene, timeline = stacked("toss", "random", bounce=(0.2, 0.7))
    speeds = timeline.bounce_speed
    hopping = (speeds[:, 0] > 0) & np.isinf(timeline.cover)  # free tiles that bounce
    impact_speed = math.sqrt(2 * timeline.gravity * timeline.apex)
    ratio = speeds[hopping, 0] / impact_speed  # each keeps its own share of the speed
    assert len(ratio) > 5 and ratio.min() >= 0.2 - 1e-9 and ratio.max() <= 0.7 + 1e-9
    assert ratio.std() > 0.05
    two = hopping & (speeds[:, 1] > 0)
    np.testing.assert_allclose(speeds[two, 1] / speeds[two, 0], ratio[two[hopping]])
    for duration in (4.0, 8.0):  # each tile's own settling fits; the last rests at the end
        timeline = AssembleChoreography(duration=duration, bounce=(0.1, 0.7)).timeline(scene)
        assert timeline.duration == pytest.approx(duration)


def test_a_tile_landing_on_a_bouncing_one_stays_on_top():
    from skitter.core.animation import TossTimeline

    regions = RegionSet.from_rects([0, 5], 0, 10, 10)  # 1 lies on 0
    scene = make_scene(regions, [0, 1])
    timeline = TossTimeline(scene, scene.center, scene.rotation, [0.0, 0.1], 1.0, apex=20.0,
                            bounce=0.6)  # fmt: skip
    checked = 0
    for t in np.linspace(1.1, 1.1 + timeline.settle.max(), 60):  # 1 has landed; 0 still bouncing
        frame = timeline.frame(t)
        rank = np.argsort(frame.draw_order())
        if frame.heights()[0] > frame.heights()[1]:
            checked += 1
        assert rank[1] > rank[0]
    assert checked  # 0 bounced higher than 1 lay, and still drew below it


def test_toss_settles_with_a_dying_wobble():
    scene, timeline = single_toss(bounce=0.0, wobble=0.2)
    assert timeline.settle[0] == pytest.approx(0.3 * 2.0)
    after = [timeline.frame(2.0 + s).rotation[0] - scene.rotation[0]
             for s in np.linspace(0.01, timeline.settle[0], 30)]  # fmt: skip
    assert max(np.abs(after)) <= 0.2 and abs(after[-1]) < 1e-9
    assert np.abs(after[:10]).max() > np.abs(after[-10:]).max()  # dies away


def test_table_camera_perspective_and_shadows():
    from skitter.core.animation.look import AnimationLook, TableCamera, scene_extent

    scene = grid_scene(columns=2, rows=1)
    look = AnimationLook(camera_height=2.0, light_direction=0.0, shadow_strength=0.5)
    camera = TableCamera.for_scene(scene, look)
    assert camera.height == pytest.approx(2.0 * scene_extent(scene))
    final = TileFrame.final(scene)
    flat = camera.project(final)  # everything at rest: unchanged, no shadows
    np.testing.assert_allclose(flat.center, final.center)
    assert len(flat.shadow) == 0 and flat.ground.tolist() == [0, 1]
    lifted = final.replace(height=np.array([0.0, camera.height / 2]),
                           rest=np.array([True, False]), order=np.array([0, 1]))  # fmt: skip
    seen = camera.project(lifted)
    np.testing.assert_allclose(seen.size[1], final.size[1] * 2)  # half way up: twice as big
    away = seen.center[1] - camera.point
    np.testing.assert_allclose(away, (final.center[1] - camera.point) * 2)
    assert seen.ground.tolist() == [0] and seen.air.tolist() == [1] and seen.shadow.tolist() == [1]
    assert seen.shadow_center[0, 1] > final.center[1, 1]  # light from the top: shadow below
    assert 0 < seen.shadow_alpha[0] < 0.5 and seen.shadow_blur[0] > 0


# Drop


def test_drop_falls_from_rest_at_the_camera():
    from skitter.core.animation import TossTimeline

    scene = grid_scene(columns=1, rows=1)
    timeline = TossTimeline(scene, scene.center + 5.0, scene.rotation, 0.0, 2.0, apex=50.0,
                            start_height=50.0)  # fmt: skip
    assert timeline.gravity == pytest.approx(2 * 50.0 / 2.0**2) and timeline.speed == 0
    times = np.linspace(0, 2.0, 41)
    height = np.array([timeline.frame(t).heights()[0] for t in times])
    np.testing.assert_allclose(height, 50.0 * (1 - (times / 2.0) ** 2), atol=1e-9)
    # Slow near the camera, fast near the table: half the fall time covers a quarter.
    assert height[20] == pytest.approx(37.5)


def test_drop_starts_at_the_camera_whatever_its_height():
    from skitter.core.animation.look import AnimationLook, TableCamera, scene_extent

    scene = pile_scene()
    for camera_height in (1.0, 3.0):
        look = AnimationLook(camera_height=camera_height)
        timeline = AssembleChoreography(motion="drop", spread=0).timeline(scene, look)
        camera = TableCamera.for_scene(scene, look)
        first = int(np.argmin(timeline.delay))
        frame = timeline.frame(timeline.delay[first] + 1e-6)
        assert frame.heights()[first] == pytest.approx(camera_height * scene_extent(scene))
        seen = camera.project(frame)
        assert seen.alpha[first] == 0  # at the camera: not seen


def test_drop_slant_starts_tiles_beside_their_spots_outward():
    import math

    from skitter.core.animation.look import NEAR, clear_of_axis

    scene = pile_scene()
    final = TileFrame.final(scene)
    x0, y0, x1, y1 = scene.bounds
    middle = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    extent = max(x1 - x0, y1 - y0)
    # Straight down, except tiles near the camera's axis: those start just far enough
    # out to come into view beside the mosaic.
    clear = clear_of_axis(scene)
    least = 1.05 * clear / (1 - math.sqrt(1 - NEAR))  # sideways way it takes at most
    straight = AssembleChoreography(motion="drop", slant=0.0).timeline(scene)
    moved = np.hypot(*(straight.start_center - final.center).T)
    off_axis = np.hypot(*(final.center - middle).T) > 1.1 * clear
    assert off_axis.any() and np.all(moved[off_axis] < 1e-9)
    assert np.all(moved <= least + 1e-9)
    slanted = AssembleChoreography(motion="drop", slant=0.5).timeline(scene)
    away = slanted.start_center - final.center
    reach = np.hypot(*away.T)
    assert np.all((reach >= 0.4 * extent - 1e-9) & (reach <= np.maximum(0.6 * extent, least)))
    out = final.center - middle
    far = np.hypot(*out.T) > 0.05 * extent
    cos = (away[far] * out[far]).sum(axis=1) / (reach[far] * np.hypot(*out[far].T))
    assert np.all(cos > np.cos(0.36))  # outward from the middle, within about 20 degrees


@pytest.mark.parametrize("slant", [0.0, 0.25])
def test_drops_come_into_view_off_the_picture_and_opaque(slant):
    """No pop-in and no see-through tiles: each dropped tile becomes visible (at the
    near plane) entirely beside the mosaic, fully opaque, and slides in from there."""
    from skitter.core.animation.look import AnimationLook, TableCamera

    scene = pile_scene()
    look = AnimationLook()
    timeline = AssembleChoreography(motion="drop", slant=slant).timeline(scene, look)
    camera = TableCamera.for_scene(scene, look)
    x0, y0, x1, y1 = scene.bounds
    seen_before = np.zeros(len(scene), bool)
    appeared = looming = 0
    for t in np.linspace(0, timeline.duration, 1500):
        frame = timeline.frame(t)
        seen = camera.project(frame)
        assert np.isin(seen.alpha, (0.0, 1.0)).all()  # never see-through
        new = np.flatnonzero((seen.alpha > 0) & ~seen_before)
        half = np.hypot(*seen.size[new].T) / 2  # reach of each rotated tile
        cx, cy = seen.center[new].T
        gap_x = np.maximum.reduce([x0 - cx, np.zeros_like(cx), cx - x1])
        gap_y = np.maximum.reduce([y0 - cy, np.zeros_like(cy), cy - y1])
        outside = np.hypot(gap_x, gap_y) > half  # the tile's circle misses the mosaic
        assert outside.all(), f"{(~outside).sum()} tiles popped into view at {t:.2f} s"
        appeared += len(new)
        seen_before |= seen.alpha > 0
        looming += np.count_nonzero((seen.alpha > 0) & (camera.scale(frame.heights()) > 3))
    assert appeared == len(scene) and looming > 0  # all came in, looking big for a while


def test_camera_hides_tiles_at_the_lens():
    from skitter.core.animation.look import NEAR, AnimationLook, TableCamera

    scene = grid_scene(columns=4, rows=1)
    camera = TableCamera.for_scene(scene, AnimationLook())
    H = camera.height
    h = np.array([0.0, 0.5 * H, NEAR * H, 1.01 * NEAR * H, 1.2 * H])
    frame = TileFrame.final(scene).replace(height=h[:4], rest=np.zeros(4, bool))
    seen = camera.project(frame)
    assert seen.alpha.tolist() == [1.0, 1.0, 1.0, 0.0]  # opaque below the near plane
    scale = camera.scale(h)  # capped at the near plane, never infinite
    assert np.isfinite(scale).all() and scale[-1] == pytest.approx(1 / (1 - NEAR))


def test_throws_never_reach_the_camera():
    from skitter.core.animation.assemble import TOSS_LIMIT
    from skitter.core.animation.look import AnimationLook, scene_extent

    scene = pile_scene()
    look = AnimationLook(camera_height=0.5)
    timeline = AssembleChoreography(arc=5.0).timeline(scene, look)
    top = max(timeline.frame(t).heights().max() for t in np.linspace(0, timeline.duration, 300))
    assert top <= TOSS_LIMIT * 0.5 * scene_extent(scene) + 1e-9


def test_airborne_tiles_draw_nearest_the_camera_on_top():
    scene = grid_scene(columns=6, rows=1)
    timeline = AssembleChoreography(motion="drop", spread=0, duration=3, travel=2).timeline(scene)
    checked = 0
    for t in np.linspace(0.0, timeline.duration, 60):
        frame = timeline.frame(t)
        falling = (timeline.delay < t) & (timeline.delay + timeline.travel > t)
        order = frame.draw_order()
        air = order[falling[order]]  # still falling (touched-down tiles keep stacking order)
        if len(air) >= 2:
            checked += 1
            assert np.all(np.diff(frame.heights()[air]) >= 0)
    assert checked


# Ranges


def test_range_param_validates_and_draws():
    from skitter.core.slicing import Configurable, RangeParam

    class Thing(Configurable):
        turns = RangeParam((0.0, 1.0), min=0.0, max=5.0)
        count = RangeParam((1, 3), min=0, max=8, whole=True)

    thing = Thing()
    assert thing.turns == (0.0, 1.0) and thing.count == (1, 3)
    thing.turns = 2  # one number: a fixed value
    assert thing.turns == (2.0, 2.0)
    for bad in ((3.0, 1.0), (0.0, 9.0), "x", (1, 2, 3)):
        with pytest.raises(ValueError):
            thing.turns = bad
    with pytest.raises(ValueError):
        thing.count = (1.5, 2)
    rng = np.random.default_rng(0)
    counts = Thing.count.draw((1, 3), rng, 3000)
    assert set(counts.tolist()) == {1, 2, 3}  # both ends included, each likely
    assert np.all(np.abs(np.bincount(counts)[1:] / 3000 - 1 / 3) < 0.04)
    turns = Thing.turns.draw((0.5, 2.0), rng, 1000)
    assert turns.min() >= 0.5 and turns.max() <= 2.0
    assert Thing(turns=(1.0, 1.0)).key() != Thing().key()


@pytest.mark.parametrize("motion", ["toss", "drop", "glide"])
def test_spin_and_flips_vary_per_tile_within_their_ranges(motion):
    scene = pile_scene()
    final = TileFrame.final(scene)
    timeline = AssembleChoreography(motion=motion, spin=(0.5, 2.0), flips=(1, 3)).timeline(scene)
    start = timeline.start_rotation if motion != "glide" else timeline.start.rotation
    turns = np.abs(start - final.rotation) / (2 * np.pi)
    assert turns.min() >= 0.5 - 1e-9 and turns.max() <= 2.0 + 1e-9 and turns.std() > 0.2
    assert np.any(start > final.rotation) and np.any(start < final.rotation)  # either way
    flips = np.abs(timeline.flips.turns)
    assert set(flips.tolist()) <= {1, 2, 3} and len(set(flips.tolist())) == 3
    fixed = AssembleChoreography(motion=motion, spin=(1.0, 1.0)).timeline(scene)
    start = fixed.start_rotation if motion != "glide" else fixed.start.rotation
    np.testing.assert_allclose(np.abs(start - final.rotation), 2 * np.pi)  # all the same


@pytest.mark.parametrize("motion", ["toss", "drop"])
def test_settle_wobble_varies_per_tile_within_its_range(motion):
    scene = pile_scene()
    final = TileFrame.final(scene)
    timeline = AssembleChoreography(motion=motion, wobble=(2.0, 10.0), bounce=0.0).timeline(scene)
    amount = np.degrees(np.abs(timeline.wobble))
    assert amount.min() >= 2.0 - 1e-9 and amount.max() <= 10.0 + 1e-9 and amount.std() > 1.0
    assert np.any(timeline.wobble > 0) and np.any(timeline.wobble < 0)  # rocking either way
    # Each tile rocks within its own amount after landing, then lies still.
    landed = timeline.delay + timeline.travel
    swings = np.zeros(len(scene))
    for i in range(len(scene)):
        for s in np.linspace(0.01, timeline.settle[i], 25):
            rotation = timeline.frame(landed[i] + s).rotation[i]
            swings[i] = max(swings[i], abs(rotation - final.rotation[i]))
    assert np.all(swings <= np.abs(timeline.wobble) + 1e-9)
    assert np.corrcoef(swings, np.abs(timeline.wobble))[0, 1] > 0.9
    same = AssembleChoreography(motion=motion, wobble=(5.0, 5.0)).timeline(scene)
    np.testing.assert_allclose(np.degrees(np.abs(same.wobble)), 5.0)


def test_range_editor_keeps_its_ends_in_order(qapp):
    from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

    from skitter.core.slicing import Configurable, RangeParam
    from skitter.ui.widgets.param_form import ParamForm

    class Thing(Configurable):
        spin = RangeParam((0.0, 1.0), "Spin", min=0.0, max=20.0, suffix=" turns")
        flips = RangeParam((0, 0), "Flips", min=0, max=8, whole=True)

    thing = Thing()
    form = ParamForm()
    form.set_target(thing)
    edits = []
    form.changed.connect(edits.append)
    low, high = form.editor("spin").widget.findChildren(QDoubleSpinBox)
    assert high.suffix() == " turns" and low.suffix() == ""
    high.setValue(3.0)
    assert thing.spin == (0.0, 3.0) and edits == ["spin"]
    low.setValue(5.0)  # above the high end: the high end follows
    assert thing.spin == (5.0, 5.0) and high.value() == 5.0
    high.setValue(2.0)  # below the low end: the low end follows
    assert thing.spin == (2.0, 2.0)
    flips_low, flips_high = form.editor("flips").widget.findChildren(QSpinBox)
    flips_high.setValue(4)
    assert thing.flips == (0, 4) and isinstance(thing.flips[1], int)
    thing.spin = (1.0, 1.5)  # edited elsewhere: refresh shows it without reporting an edit
    count = len(edits)
    form.refresh()
    assert (low.value(), high.value()) == (1.0, 1.5) and len(edits) == count


# Flips


def test_flips_turn_whole_times_and_land_face_up():
    from skitter.core.animation import Flips

    flips = Flips(turns=np.array([1.0, -2.0]), axis=np.array([0, 1]))
    size = np.array([[10.0, 6.0], [10.0, 6.0]])
    s, facing = flips.apply(size, np.array([0.5, 0.125]))
    assert facing[0] == pytest.approx(-1.0)  # half way through one turn: back up
    assert s[0].tolist() == pytest.approx([10.0, 6.0])  # flat again (back up), about x
    s, facing = flips.apply(size, np.array([0.25, 0.25]))
    assert s[0, 1] == pytest.approx(6.0 * Flips.THINNEST) and s[0, 0] == 10.0  # edge-on: height
    assert s[1, 0] == pytest.approx(10.0) and s[1, 1] == 6.0  # 2 turns: a full turn by now
    s, facing = flips.apply(size, np.array([1.0, 1.0]))
    assert np.allclose(facing, 1.0) and np.allclose(s, size)  # landed: face up, full size


@pytest.mark.parametrize("motion", ["toss", "glide"])
def test_flipping_tiles_show_their_backs_but_land_face_up(motion):
    scene = pile_scene()
    timeline = AssembleChoreography(flips=3, motion=motion, duration=4.0).timeline(scene)
    backs = [np.sum(timeline.frame(t).facings() < 0) for t in np.linspace(0, 4.0, 41)]
    assert max(backs) > 0  # some tiles show their backs on the way
    # At the moment each tile lands it is face up, unmirrored, at its full size.
    impact = timeline.delay + timeline.travel
    for i in np.argsort(impact)[:: max(1, len(scene) // 10)]:
        frame = timeline.frame(float(impact[i]))
        assert frame.facings()[i] == pytest.approx(1.0)
        np.testing.assert_allclose(frame.size[i], scene.size[i])


def test_backs_show_the_back_color_shaded():
    from skitter.core.animation.look import FLIP_SHADE, flip_tint

    scene = grid_scene(columns=3, rows=1)
    frame = TileFrame.final(scene).replace(facing=np.array([1.0, 0.0, -1.0]))
    tint = flip_tint(frame, (0.9, 0.8, 0.7))
    assert tint[0, 3] == 0  # face up and flat: the photo as it is
    assert tint[1, 3] == pytest.approx(FLIP_SHADE) and tint[1, :3].tolist() == [0, 0, 0]  # edge-on
    np.testing.assert_allclose(tint[2], [0.9, 0.8, 0.7, 1.0])  # back up and flat: the back color


# Deal


@pytest.mark.parametrize("make", [grid_scene, pile_scene])
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"decks": 3},
        {"order": "random", "face_down": True},
        {"position": "orbit"},
        {"position": "orbit", "decks": 4, "orbit_turns": 3, "order": "center"},
    ],
)
def test_deal_fits_the_duration_and_starts_at_once(make, settings):
    scene = make()
    timeline = DealChoreography(duration=6, slide=1.2, **settings).timeline(scene)
    assert isinstance(timeline, DealTimeline)
    assert timeline.duration == pytest.approx(6)
    assert timeline.depart.min() == pytest.approx(0, abs=1e-6)  # dealing starts at once
    assert np.all(timeline.depart >= 0)
    assert timeline.slide.max() == pytest.approx(1.2, rel=1e-3)  # the farthest one
    land = timeline.depart + timeline.slide
    lower, upper = scene.overlaps.T
    assert np.all(land[lower] < land[upper])  # bottom first


def test_deal_starts_with_every_tile_in_its_nearest_deck():
    scene = grid_scene()
    choreography = DealChoreography(decks=4, neatness=1.0)
    spots = choreography.deck_spots(scene)
    timeline = choreography.timeline(scene)
    start = timeline.frame(0.0)
    nearest = np.argmin(np.linalg.norm(scene.center[:, None] - spots[None], axis=2), axis=1)
    np.testing.assert_allclose(start.center, spots[nearest], atol=1e-9)
    assert not start.at_rest().any()
    # Each deck draws bottom card first, and its top card is the first of it to leave.
    order = start.draw_order()
    for d in range(4):
        mine = [i for i in order if nearest[i] == d]
        assert np.all(np.diff(timeline.depart[mine]) <= 1e-12)


def test_deal_deck_positions():
    scene = grid_scene()  # bounds 0..60 x 0..40, extent 60
    spot = DealChoreography(position="bottom", offset=0.5).deck_spots(scene)
    np.testing.assert_allclose(spot, [[30, 40 + 30]])
    spot = DealChoreography(position="top_left", offset=0.0).deck_spots(scene)
    np.testing.assert_allclose(spot, [[0, 0]], atol=1e-9)
    around = DealChoreography(position="left", offset=0.0, decks=2).deck_spots(scene)
    np.testing.assert_allclose(around, [[0, 20], [60, 20]], atol=1e-9)
    side = DealChoreography(position="bottom", offset=0.0, decks=3, arrangement="side")
    np.testing.assert_allclose(side.deck_spots(scene), [[10, 40], [30, 40], [50, 40]])
    middle = DealChoreography(position="center", decks=1).deck_spots(scene)
    np.testing.assert_allclose(middle, [[30, 20]])
    # Side by side needs an edge: a corner deals around the mosaic.
    corner = DealChoreography(position="top_left", decks=2, arrangement="side", offset=0.0)
    np.testing.assert_allclose(corner.deck_spots(scene), [[0, 0], [60, 40]], atol=1e-9)
    assert not DealChoreography.arrangement.is_available(corner, "side")


def test_deal_nearest_first_lands_near_tiles_first():
    scene = grid_scene(12, 8)
    choreography = DealChoreography(order="near", spread=0.0)
    timeline = choreography.timeline(scene)
    reach = np.linalg.norm(scene.center - choreography.deck_spots(scene)[0], axis=1)
    land = timeline.depart + timeline.slide
    assert np.corrcoef(np.argsort(np.argsort(land)), np.argsort(np.argsort(reach)))[0, 1] > 0.95


def test_deal_face_down_cards_turn_face_up_on_arrival():
    scene = grid_scene()
    timeline = DealChoreography(face_down=True).timeline(scene)
    assert np.all(timeline.frame(0.0).facings() == -1)  # backs up in the deck
    for t in np.linspace(0, timeline.duration, 50):
        frame = timeline.frame(t)
        landed = frame.at_rest()
        assert np.all(frame.facings()[landed] == 1)
        assert np.all(frame.heights()[landed] == 0)
        # Turned over as it leaves the deck, as a dealer does: face up for the rest.
        turned = t >= timeline.depart + np.minimum(0.3, 0.4 * timeline.slide)
        assert np.all(frame.facings()[turned] == pytest.approx(1.0))
        waiting = t <= timeline.depart
        assert np.all(frame.facings()[waiting] == -1)


def test_deal_cards_stay_on_top_of_their_deck_until_clear():
    # A card leaving never draws below a card still in its deck while over it.
    scene = grid_scene(12, 8)
    timeline = DealChoreography(order="far", spread=0.0).timeline(scene)
    diagonal = np.hypot(*scene.size.T)
    for t in np.linspace(0, timeline.duration, 200):
        frame = timeline.frame(t)
        moving = (t > timeline.depart) & ~frame.at_rest()
        waiting = t <= timeline.depart
        if not moving.any() or not waiting.any():
            continue
        rank = np.empty(len(scene), np.int64)
        rank[frame.draw_order()] = np.arange(len(scene))
        near = np.linalg.norm(frame.center - timeline.deck_center, axis=1) < 0.99 * diagonal
        top = rank[waiting].max()
        assert np.all(rank[moving & near] > top)


@pytest.mark.parametrize("spin", [(0.0, 0.0), (0.3, 0.7), (1.2, 3.4)])
def test_deal_tiles_arrive_at_their_rotation_without_a_jump(spin):
    scene = grid_scene()
    timeline = DealChoreography(spin=spin, neatness=0.3, deck_angle=40).timeline(scene)
    land = timeline.depart + timeline.slide
    before = timeline.frame(float(land.max()) - 1e-6)
    moving = np.flatnonzero(land == land.max())
    off = (before.rotation[moving] - scene.rotation[moving] + math.pi) % math.tau - math.pi
    np.testing.assert_allclose(off, 0.0, atol=1e-4)  # already turned to its final angle
    # The extra spin is close to what each tile drew, in its own direction.
    turned = np.abs(timeline.end_rotation - timeline.deck_rotation) / math.tau
    assert np.all(turned <= spin[1] + 0.5 + 1e-9)
    assert np.all(turned >= max(spin[0] - 0.5, 0.0) - 1e-9)


def test_deal_orbiting_decks_circle_the_middle_evenly_spaced():
    scene = grid_scene()  # middle (30, 20), extent 60
    choreography = DealChoreography(position="orbit", radius=1.0, decks=3, start_angle=90)
    path = choreography.deck_path(scene)
    np.testing.assert_allclose(path.spots[0], [90, 20], atol=1e-9)  # 90°: right of the middle
    for t in (0.0, 1.3, 5.0):
        center, rotation = path.at(np.arange(3), t)
        np.testing.assert_allclose(np.hypot(*(center - [30, 20]).T), 60.0)
        angles = np.sort(np.arctan2(center[:, 0] - 30, -(center[:, 1] - 20)) % math.tau)
        np.testing.assert_allclose(np.diff(angles), math.tau / 3)  # evenly spaced
    # One turn over the duration, clockwise: a quarter of the way in, at the bottom.
    center, rotation = path.at(0, choreography.duration / 4)
    np.testing.assert_allclose(center, [30, 80], atol=1e-9)
    assert rotation == pytest.approx(math.pi)  # turned with it: bottom toward the middle
    counter = DealChoreography(position="orbit", radius=1.0, start_angle=90,
                               orbit_direction="counter").deck_path(scene)  # fmt: skip
    np.testing.assert_allclose(counter.at(0, choreography.duration / 4)[0], [30, -40], atol=1e-9)


def test_deal_tiles_ride_their_orbiting_deck_until_they_leave():
    scene = grid_scene()
    timeline = DealChoreography(position="orbit", decks=2, neatness=1.0).timeline(scene)
    path = timeline.path
    for t in np.linspace(0, timeline.duration, 30):
        frame = timeline.frame(t)
        waiting = np.flatnonzero(t < timeline.depart)
        center, rotation = path.at(timeline.deck[waiting], t)
        np.testing.assert_allclose(frame.center[waiting], center, atol=1e-9)
        np.testing.assert_allclose(frame.rotation[waiting], rotation, atol=1e-9)
    # Leaving is seamless: just before and just after, in the same place.
    for i in range(0, len(scene), 5):
        before = timeline.frame(timeline.depart[i] - 1e-7)
        after = timeline.frame(timeline.depart[i] + 1e-7)
        np.testing.assert_allclose(before.center[i], after.center[i], atol=1e-4)


def test_deal_orbiting_decks_deal_what_they_pass():
    # Nearest first, one deck, one turn: tiles land in the order the deck sweeps by.
    scene = grid_scene(16, 12)
    choreography = DealChoreography(position="orbit", spread=0.0, radius=0.6)
    timeline = choreography.timeline(scene)
    out = scene.center - timeline.path.middle
    angle = (np.arctan2(out[:, 0], -out[:, 1]) - math.radians(180)) % math.tau
    land = timeline.depart + timeline.slide
    assert np.corrcoef(np.argsort(np.argsort(land)), np.argsort(np.argsort(angle)))[0, 1] > 0.9
    # And each leaves from the deck nearest it at the time, while it is close by.
    spots = timeline.path.all_at(timeline.depart)
    gaps = np.linalg.norm(scene.center[:, None] - spots, axis=2)
    np.testing.assert_array_equal(timeline.deck, np.argmin(gaps, axis=1))

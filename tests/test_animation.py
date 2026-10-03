"""The mosaic scene and the animation framework (choreographies and timelines)."""

import numpy as np
import pytest

from skitter.core.animation import (
    Choreography,
    FlightTimeline,
    TileFrame,
    choreography_types,
    get_choreography,
)
from skitter.core.animation.assemble import ORDERS, AssembleChoreography
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
    assert landing_order(scene, [0, 0, 0, 0]).tolist() == [0, 1, 2, 3]  # ties: stacking order


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
    timeline = AssembleChoreography(duration=8, travel=1.5, motion=motion).timeline(scene)
    settle = getattr(timeline, "settle", 0.0)  # Toss: bounces and wobble after impact
    assert timeline.duration == pytest.approx(8)  # the last tile comes to rest at the end
    landing = np.sort(timeline.delay + timeline.travel)
    assert landing[0] == pytest.approx(1.5) and landing[-1] == pytest.approx(8 - settle)
    np.testing.assert_allclose(np.diff(landing), (6.5 - settle) / (len(scene) - 1))


@pytest.mark.parametrize(("cls", "settings"), CONFIGS, ids=CONFIG_IDS)
def test_every_choreography_keeps_overlapping_tiles_in_stacking_order(cls, settings):
    # If a lower tile ever drew above a tile that covers it, it would have to pop
    # underneath later (at the latest on the last frame).
    scene = pile_scene()
    lower, upper = scene.overlaps.T
    assert len(lower)
    timeline = cls(**settings).timeline(scene)
    for t in np.linspace(0, timeline.duration, 97):
        frame = timeline.frame(t)
        rank = np.empty(len(scene), np.int64)
        rank[frame.draw_order()] = np.arange(len(scene))
        visible = (frame.alpha[lower] > 0) & (frame.alpha[upper] > 0)
        wrong = visible & (rank[upper] < rank[lower])
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
    speeds = timeline.bounce_speed
    assert len(speeds) >= 2
    np.testing.assert_allclose(speeds[1:] / speeds[:-1], 0.5)  # each keeps half the speed
    peaks = [speeds[k] ** 2 / (2 * timeline.gravity) for k in range(len(speeds))]
    assert peaks[0] == pytest.approx(10.0 * 0.25)  # heights shrink by bounce²
    # Sample the first bounce: up to its peak, back to the table.
    start = 2.0 + timeline.bounce_start[0]
    mid = start + timeline.bounce_time[0] / 2
    assert timeline.frame(mid).heights()[0] == pytest.approx(peaks[0])
    assert not timeline.frame(mid).at_rest()[0]  # still moving: draws above tiles at rest
    end = timeline.frame(2.0 + timeline.settle)
    assert end.heights()[0] == 0 and end.at_rest()[0]
    assert timeline.duration == pytest.approx(2.0 + timeline.settle)


def test_toss_settles_with_a_dying_wobble():
    scene, timeline = single_toss(bounce=0.0, wobble=0.2)
    assert timeline.settle == pytest.approx(0.3 * 2.0)
    after = [timeline.frame(2.0 + s).rotation[0] - scene.rotation[0]
             for s in np.linspace(0.01, timeline.settle, 30)]  # fmt: skip
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

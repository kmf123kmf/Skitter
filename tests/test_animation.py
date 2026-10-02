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
    np.testing.assert_allclose(scene.tint_offset, scene.result.tint_offset()[scene.region])
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


@pytest.mark.parametrize("cls", choreography_types())
def test_every_choreography_ends_on_the_finished_mosaic(cls: type[Choreography]):
    scene = grid_scene()
    timeline = cls().timeline(scene)
    assert timeline.duration > 0
    final = TileFrame.final(scene)
    assert_frame_equal(timeline.frame(timeline.duration), final)
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

"""Clear choreographies (core/animation/clear.py)."""

import math

import numpy as np
import pytest
from test_animation import grid_scene, pile_scene
from test_show import tiled_scene

from skitter.core.animation.base import TileFrame
from skitter.core.animation.clear import BOB, PEEL, BlowAwayChoreography, _blown_time, half_extent
from skitter.core.animation.compass import COMPASS, heading
from skitter.core.animation.look import AnimationLook, TableCamera


def same(a: TileFrame, b: TileFrame) -> bool:
    return all(np.allclose(getattr(a, f), getattr(b, f))
               for f in ("center", "size", "rotation", "alpha"))  # fmt: skip


def test_blown_time_inverts_the_drag_model():
    k, wind = 3.0, 200.0
    for distance in (1e-6, 0.5, 10.0, 400.0, 1e5):
        tau = _blown_time(k, distance, wind)
        blown = wind * (tau - (1.0 - math.exp(-k * tau)) / k)
        assert blown == pytest.approx(distance, rel=1e-6)
    assert _blown_time(k, 0.0, wind) == 0.0


@pytest.mark.parametrize("scene", [grid_scene(), pile_scene()], ids=["grid", "pile"])
@pytest.mark.parametrize("distance", [0.5, 3.0])
def test_blow_away_starts_at_the_mosaic_and_is_gone_exactly_at_the_duration(scene, distance):
    timeline = BlowAwayChoreography(duration=5.0, distance=distance).timeline(scene)
    assert timeline.duration == 5.0
    assert timeline.frame(0.0) is timeline.final
    assert timeline.gone.max() == pytest.approx(5.0, rel=1e-6)
    assert (timeline.gone <= 5.0 + 1e-9).all()
    assert timeline.frame(5.0).alpha.max() == 0.0
    # A tile is gone only once all of it is `distance` picture widths past the right
    # edge, as the camera sees it, and every tile shown is short of that.
    camera = TableCamera.for_scene(scene, AnimationLook())
    x0, _, x1, _ = scene.bounds
    out = x1 + distance * (x1 - x0)
    for t in np.linspace(0.0, 5.0, 41):
        frame = timeline.frame(t)
        hidden = frame.alpha == 0
        seen = camera.project(frame)
        near_edge = seen.center[:, 0] - np.hypot(*seen.size.T) / 2
        assert (near_edge[hidden] >= out - 1e-6).all()
        reach = np.hypot(*scene.size.T) / 2  # its own size (tumbling squashes frame.size)
        assert (frame.center[~hidden, 0] - reach[~hidden] < out).all()


def test_flying_farther_takes_a_stronger_wind():
    scene = grid_scene()
    near = BlowAwayChoreography(distance=0.5).timeline(scene)
    far = BlowAwayChoreography(distance=4.0).timeline(scene)
    assert far.wind > near.wind
    tile = scene.tile_size[0]
    note = next(n.text for n in far.notes if n.setting == "distance")
    assert note.startswith(f"{4.0 * 60 / tile:.0f} tiles")  # the grid is 60 wide


@pytest.mark.parametrize("direction", list(COMPASS))
def test_the_gust_reaches_the_tiles_its_way(direction):
    scene = tiled_scene(24, 16)
    timeline = BlowAwayChoreography(direction=direction, raggedness=0.0).timeline(scene)
    along = scene.center @ heading(COMPASS[direction])
    assert np.corrcoef(timeline.release, along)[0, 1] > 0.99
    assert timeline.release.min() == 0.0
    assert timeline.release.max() == pytest.approx(0.6 * 4.0)  # Sweep: 60% of the duration


def test_without_a_sweep_every_tile_sets_off_at_once():
    timeline = BlowAwayChoreography(sweep=0.0).timeline(grid_scene())
    assert (timeline.release == 0.0).all()
    assert not BlowAwayChoreography.raggedness.is_active(BlowAwayChoreography(sweep=0.0))


def test_tiles_blow_downwind():
    scene = tiled_scene()
    timeline = BlowAwayChoreography(direction="down", gusts=0.0, skid_jitter=0.0).timeline(scene)
    frame = timeline.frame(1.5)
    moved = frame.center - scene.center
    going = ~frame.at_rest()
    assert going.any()
    assert (moved[going, 1] > 0).all()  # down the picture
    np.testing.assert_allclose(moved[:, 0], 0.0, atol=1e-9)  # nothing drifts sideways


def test_a_pile_blows_away_top_first():
    scene = pile_scene()
    choreography = BlowAwayChoreography(direction="right", raggedness=0.0)
    timeline = choreography.timeline(scene)
    lower, upper = scene.overlaps.T
    assert len(lower)
    assert (timeline.release[lower] >= timeline.release[upper]).all()
    # When a covered tile sets off, every tile lying on it is a share of the way clear.
    direction = timeline.direction
    half = half_extent(scene.size, scene.rotation, direction)
    reach = np.hypot(*scene.size.T) / 2
    for i, j in zip(lower, upper, strict=True):
        frame = timeline.frame(timeline.release[i])
        moved = (frame.center[j] - scene.center[j]) @ direction
        overlap = half[i] + reach[j] - (scene.center[j] - scene.center[i]) @ direction
        assert moved >= PEEL * overlap - 1e-6
    assert any(n.setting == "sweep" and "wait" in n.text for n in timeline.notes)


def test_a_grid_has_nothing_waiting():
    timeline = BlowAwayChoreography().timeline(grid_scene())
    assert not any("wait" in n.text for n in timeline.notes)


def test_tiles_skid_rattling_then_lift():
    scene = tiled_scene(8, 6)
    timeline = BlowAwayChoreography(sweep=0.0, skid=50.0, skid_jitter=10.0, spin=(0.0, 0.0),
                                    gusts=0.0, pickup=1.0).timeline(scene)  # fmt: skip
    early = timeline.frame(0.1)  # a share of the wind's speed well below every skid
    assert (~early.at_rest()).all() and (early.heights() == 0).all()
    turn = early.rotation - scene.rotation
    assert np.abs(turn).max() > 0  # rattling
    assert np.abs(turn).max() <= math.radians(10.0) + 1e-9
    late = timeline.frame(3.0)
    assert (late.heights() > 0).all()  # all lifted by then

    smooth = BlowAwayChoreography(sweep=0.0, skid=50.0, skid_jitter=0.0, spin=(0.0, 0.0),
                                  gusts=0.0).timeline(scene)  # fmt: skip
    np.testing.assert_allclose(smooth.frame(0.1).rotation, scene.rotation)


def test_without_a_skid_tiles_lift_at_once():
    scene = tiled_scene(8, 6)
    timeline = BlowAwayChoreography(sweep=0.0, skid=0.0).timeline(scene)
    assert (timeline.frame(0.05).heights() > 0).all()
    assert not BlowAwayChoreography.skid_jitter.is_active(BlowAwayChoreography(skid=0.0))


def test_tiles_move_smoothly():
    scene = pile_scene()
    timeline = BlowAwayChoreography().timeline(scene)
    tile = scene.tile_size[0]
    times = np.linspace(0.0, timeline.duration, 801)
    dt = times[1] - times[0]
    wind = timeline.wind
    last = timeline.frame(0.0)
    for t in times[1:]:
        frame = timeline.frame(t)
        shown = (frame.alpha > 0) & (last.alpha > 0)
        step = np.hypot(*(frame.center - last.center)[shown].T)
        assert (step <= wind * dt + 0.5 * tile).all()  # never faster than the wind, give or take
        assert (np.abs(frame.heights() - last.heights())[shown] < 0.5 * tile).all()
        last = frame


def test_tiles_at_rest_draw_below_moving_ones_in_stacking_order():
    scene = pile_scene()
    timeline = BlowAwayChoreography().timeline(scene)
    frame = timeline.frame(1.0)
    order = frame.draw_order()
    rest = frame.at_rest()[order]
    assert rest.any() and (~rest).any()
    assert not rest[np.argmax(~rest) :].any()  # all at rest first
    assert (np.diff(order[rest]) > 0).all()  # in stacking order
    assert (np.diff(frame.heights()[order[~rest]]) >= 0).all()  # then low to high


def test_tumbling_shows_backs_and_none_keeps_tiles_face_up():
    scene = tiled_scene(8, 6)
    tumbling = BlowAwayChoreography(tumble=(2.0, 2.0)).timeline(scene)
    assert any((tumbling.frame(t).facings() < 0).any() for t in np.linspace(0, 4, 41))
    flat = BlowAwayChoreography(tumble=(0.0, 0.0)).timeline(scene)
    assert all(flat.frame(t).facing is None for t in np.linspace(0, 4, 11))


def test_lift_is_kept_below_the_camera():
    scene = tiled_scene()
    look = AnimationLook(camera_height=0.5)
    timeline = BlowAwayChoreography(lift=50.0).timeline(scene, look)
    camera = TableCamera.for_scene(scene, look)
    top = max(timeline.frame(t).heights().max() for t in np.linspace(0, 4, 41))
    assert top <= 0.25 * (1.0 + BOB * 0.4) * camera.height + 1e-9  # Gusts 40% bob it a little
    assert any(n.setting == "lift" and n.warning for n in timeline.notes)
    assert not any(n.setting == "lift" for n in BlowAwayChoreography().timeline(scene).notes)


def test_same_seed_same_wind_and_settings_round_trip():
    scene = pile_scene()
    a = BlowAwayChoreography(seed=4).timeline(scene)
    b = BlowAwayChoreography(seed=4).timeline(scene)
    assert same(a.frame(1.7), b.frame(1.7))
    choreography = BlowAwayChoreography(direction="custom", angle=30.0, tumble=(0.0, 1.0))
    assert BlowAwayChoreography.from_values(choreography.values()).values() == choreography.values()


def test_a_longer_duration_blows_more_gently():
    scene = grid_scene()
    quick = BlowAwayChoreography(duration=2.0).timeline(scene)
    slow = BlowAwayChoreography(duration=8.0).timeline(scene)
    assert slow.wind < quick.wind
    assert any(n.setting == "duration" for n in quick.notes)

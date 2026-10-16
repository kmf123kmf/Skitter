"""Clear choreographies (core/animation/clear.py)."""

import math

import numpy as np
import pytest
from test_animation import grid_scene, pile_scene
from test_show import tiled_scene

from skitter.core.animation.base import TileFrame
from skitter.core.animation.clear import (
    BOB,
    PEEL,
    BlowAwayChoreography,
    DrainChoreography,
    _blown_time,
    half_extent,
    spin_up_clock,
    spin_up_time,
)
from skitter.core.animation.compass import COMPASS, clock_angle, heading
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
    timeline = BlowAwayChoreography(direction="down", gusts=0.0, skid_jitter=0.0,
                                    spread=0.0).timeline(scene)  # fmt: skip
    frame = timeline.frame(1.5)
    moved = frame.center - scene.center
    going = ~frame.at_rest()
    assert going.any()
    assert (moved[going, 1] > 0).all()  # down the picture
    np.testing.assert_allclose(moved[:, 0], 0.0, atol=1e-9)  # nothing drifts sideways


def test_tiles_fan_out_within_the_spread():
    scene = tiled_scene()
    choreography = BlowAwayChoreography(direction="down", gusts=0.0, skid_jitter=0.0, spread=60.0)
    timeline = choreography.timeline(scene)
    frame = timeline.frame(2.0)
    moved = frame.center - scene.center
    going = moved[:, 1] > 1.0
    assert going.sum() > 100
    turn = np.degrees(np.arctan2(moved[going, 0], moved[going, 1]))
    assert np.abs(turn).max() <= 30.0 + 1e-6  # within the cone, either side
    assert (turn < -10).any() and (turn > 10).any()
    assert np.abs(turn).mean() < 15.0  # most near the middle
    # Flying at an angle, a tile still goes only once it is the distance down the wind.
    gone = BlowAwayChoreography(direction="down", spread=90.0, distance=0.5).timeline(scene)
    _, _, _, y1 = scene.bounds
    out = y1 + 0.5 * (y1 - scene.bounds[1])
    reach = np.hypot(*scene.size.T) / 2
    for t in np.linspace(0.0, gone.duration, 21):
        frame = gone.frame(t)
        hidden = frame.alpha == 0
        assert (frame.center[hidden, 1] - reach[hidden] >= out - 1e-6).all()
        assert (frame.center[~hidden, 1] - reach[~hidden] < out).all()


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


# Drain


def test_spin_up_time_inverts_the_clock():
    for tau in (0.0, 0.5, 3.0):
        for t in (0.01, 0.7, 5.0):
            assert spin_up_time(float(spin_up_clock(t, tau)), tau) == pytest.approx(t, abs=1e-9)


@pytest.mark.parametrize("scene", [grid_scene(), pile_scene()], ids=["grid", "pile"])
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"spin_up": 0.0, "falloff": 0.0, "turbulence": 100.0},
        {"falloff": 3.0, "center_x": 0.0, "center_y": 100.0},
    ],
)
def test_drain_starts_at_the_mosaic_and_is_gone_exactly_at_the_duration(scene, settings):
    timeline = DrainChoreography(duration=5.0, **settings).timeline(scene)
    assert timeline.duration == 5.0
    assert same(timeline.frame(0.0), TileFrame.final(scene))
    last = (timeline.u0 / timeline.rate).max()  # the flow's time when the last tile goes
    assert last == pytest.approx(spin_up_clock(5.0, timeline.tau))
    assert (timeline.frame(4.99).alpha > 0).any()
    assert (timeline.frame(5.0).alpha == 0).all()


def test_drain_turns_the_farthest_tile_exactly_and_either_way():
    scene = tiled_scene()
    clockwise = DrainChoreography(turns=2.0, turbulence=0.0).timeline(scene)
    circled = clockwise.angle(np.zeros(len(scene)))
    assert circled.max() == pytest.approx(2.0 * math.tau)
    assert (circled >= 0).all() and circled.argmax() == clockwise.u0.argmax()
    counter = DrainChoreography(turns=2.0, turbulence=0.0, swirl="counter").timeline(scene)
    assert np.allclose(counter.angle(np.zeros(len(scene))), -circled)
    # Clockwise on the picture: each tile's clock angle about the drain grows at first.
    frame = clockwise.frame(0.3)
    outside = np.hypot(*(frame.center - clockwise.drain).T) > 4.0 * scene.tile_size[0]
    before, after = (clock_angle(f.center[outside] - clockwise.drain)
                     for f in (TileFrame.final(scene), frame))  # fmt: skip
    turned = (after - before) % math.tau
    assert outside.sum() > len(scene) / 2 and ((turned > 0) & (turned < math.pi)).all()


def test_drain_tiles_only_head_inward_without_turbulence():
    scene = tiled_scene()
    timeline = DrainChoreography(turbulence=0.0).timeline(scene)
    last = np.hypot(*(scene.center - timeline.drain).T)
    for t in np.linspace(0.05, timeline.duration, 40):
        frame = timeline.frame(t)
        distance = np.hypot(*(frame.center - timeline.drain).T)
        shown = frame.alpha > 0
        assert (distance[shown] <= last[shown] + 1e-9).all()
        last = distance


def test_drain_tiles_sink_only_in_the_core():
    scene = tiled_scene()
    tile = scene.tile_size[0]
    timeline = DrainChoreography(turbulence=0.0, core=3.0, shrink=80.0, darken=50.0,
                                 fade=40.0).timeline(scene)  # fmt: skip
    for t in (0.5, 1.5, 3.0):
        frame = timeline.frame(t)
        distance = np.hypot(*(frame.center - timeline.drain).T)
        shown = frame.alpha > 0
        outside = shown & (distance >= 3.0 * tile)
        assert np.allclose(frame.size[outside], scene.size[outside])
        assert np.allclose(frame.tint[outside, 3], 0.0) and np.allclose(frame.alpha[outside], 1)
        sinking = shown & (distance < 2.0 * tile)
        assert sinking.any()
        assert (frame.size[sinking, 0] < scene.size[sinking, 0]).all()
        assert (frame.tint[sinking, 3] > 0).all() and (frame.alpha[sinking] < 1).all()
        # Sinking tiles draw under the rest (a grid has no piles to keep in order).
        rank = np.empty(len(scene), np.int64)
        rank[frame.draw_order()] = np.arange(len(scene))
        assert rank[sinking].max() < rank[outside].min()


def test_drain_spin_carries_tiles_round_or_keeps_their_heading():
    scene = tiled_scene()
    carried = DrainChoreography(turbulence=0.0, spin=100.0).timeline(scene)
    frame = carried.frame(1.0)
    u = carried.u0 - carried.rate * float(spin_up_clock(1.0, carried.tau))
    shown = frame.alpha > 0
    turned = carried.angle(np.maximum(u, 0.0))
    assert np.allclose((frame.rotation - scene.rotation)[shown], turned[shown])
    still = DrainChoreography(turbulence=0.0, spin=0.0).timeline(scene)
    assert np.allclose(still.frame(1.0).rotation, scene.rotation)


def test_drain_falloff_paces_the_middle_against_the_edges():
    scene = tiled_scene()

    def gone_halfway(falloff):
        timeline = DrainChoreography(falloff=falloff, turbulence=0.0, spin_up=0.0).timeline(scene)
        return np.count_nonzero(timeline.frame(timeline.duration / 2).alpha == 0)

    assert gone_halfway(0.0) < gone_halfway(1.0) < gone_halfway(2.5)


def test_drain_spin_up_eases_the_start():
    scene = tiled_scene()
    eased = DrainChoreography(spin_up=2.0, turbulence=0.0).timeline(scene)
    sudden = DrainChoreography(spin_up=0.0, turbulence=0.0).timeline(scene)

    def moved(timeline, t):
        return np.abs(timeline.frame(t).center - scene.center).max()

    assert moved(eased, 0.1) < 0.2 * moved(sudden, 0.1)
    assert not any(n.setting == "spin_up" for n in eased.notes)
    slow = DrainChoreography(spin_up=4.0, duration=5.0).timeline(scene)
    assert any(n.setting == "spin_up" for n in slow.notes)


def test_drain_notes_and_a_core_past_every_tile():
    scene = tiled_scene()
    timeline = DrainChoreography().timeline(scene)
    assert {n.setting for n in timeline.notes} == {"duration", "turns"}
    wide = DrainChoreography(core=50.0).timeline(scene)
    assert any(n.setting == "core" and not n.warning for n in wide.notes)
    frame = wide.frame(0.5)
    assert (frame.size[frame.alpha > 0, 0] < scene.size[frame.alpha > 0, 0]).all()


def test_drain_same_seed_same_flow_and_settings_round_trip():
    scene = pile_scene()
    a = DrainChoreography(seed=4).timeline(scene)
    b = DrainChoreography(seed=4).timeline(scene)
    assert same(a.frame(1.7), b.frame(1.7))
    assert not same(a.frame(1.7), DrainChoreography(seed=5).timeline(scene).frame(1.7))
    choreography = DrainChoreography(swirl="counter", center_x=20.0, falloff=2.0, spin=150.0)
    assert DrainChoreography.from_values(choreography.values()).values() == choreography.values()

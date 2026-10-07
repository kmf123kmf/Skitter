"""Camera keyframes (core/animation/keyframes.py) and the video clock (video.py)."""

import cmath
import math
from types import SimpleNamespace

import numpy as np
import pytest
from test_video import gpu, render_setup  # noqa: F401 (fixture, helper)

from skitter.core.animation.camera import Shot, home_shot
from skitter.core.animation.keyframes import CameraKey, CameraTrack, KeyTime
from skitter.core.animation.video import ClockedTimeline, VideoClock, VideoSettings, plan_video

CLOCK = VideoClock(hold_start=1.0, duration=8.0, hold_end=2.0)  # 11 s of video
BASE = (0.0, 0.0, 100.0, 80.0)
HOME = (50.0, 40.0)


def key(t, center=HOME, zoom=1.0, rotation=0.0, **kwargs) -> CameraKey:
    return CameraKey(KeyTime.at(t, CLOCK), Shot(center, zoom, rotation), **kwargs)


def track(*keys, stretch=True) -> CameraTrack:
    return CameraTrack(tuple(keys), stretch)


def frame_position(shot: Shot, point) -> complex:
    """Where a table point sits in the frame (relative to its middle, in frame units)."""
    return (complex(*point) - complex(*shot.center)) * cmath.exp(-1j * shot.rotation) * shot.zoom


def velocity(path, t, dt=1e-4) -> np.ndarray:
    a, b = path.shot(t - dt), path.shot(t + dt)
    return np.array([b.center[0] - a.center[0], b.center[1] - a.center[1],
                     math.log(b.zoom / a.zoom), b.rotation - a.rotation]) / (2 * dt)  # fmt: skip


# Key times


def test_key_times_anchor_to_their_part_of_the_video():
    assert KeyTime.at(0.5, CLOCK) == KeyTime("lead", 0.5)
    assert KeyTime.at(5.0, CLOCK) == KeyTime("body", 0.5)  # half way through the animation
    assert KeyTime.at(10.0, CLOCK) == KeyTime("tail", 1.0)
    assert KeyTime.at(5.0, CLOCK, stretch=False) == KeyTime("body", 4.0)
    longer = VideoClock(hold_start=2.0, duration=16.0, hold_end=2.0)
    assert KeyTime("lead", 0.5).seconds(longer) == 0.5  # seconds from the start
    assert KeyTime("body", 0.5).seconds(longer) == 10.0  # still half way through
    assert KeyTime("body", 4.0).seconds(longer, stretch=False) == 6.0  # 4 s in, pinned
    assert KeyTime("tail", 1.0).seconds(longer) == 19.0  # a second after the end
    shorter = VideoClock(hold_start=0.2, duration=2.0, hold_end=0.5)
    assert KeyTime("lead", 0.5).seconds(shorter) == 0.2  # clamped into its part
    assert KeyTime("body", 4.0).seconds(shorter, stretch=False) == 2.2
    assert KeyTime("tail", 1.0).seconds(shorter) == pytest.approx(2.7)


# Paths


def test_no_keys_show_the_framing_and_holds_surround_the_keys():
    assert CameraTrack().path(CLOCK, BASE).shot(3.0) == home_shot(BASE)
    path = track(key(2.0, zoom=3.0), key(6.0)).path(CLOCK, BASE)
    assert path.shot(0.0) == path.shot(2.0) == Shot(HOME, 3.0)
    assert path.shot(6.0) == path.shot(11.0) == Shot(HOME)
    assert path.max_zoom == pytest.approx(3.0)


def test_the_camera_eases_to_rest_at_keys_that_stop_and_flows_through_the_others():
    keys = (key(1.0, (20.0, 20.0), 3.0), key(5.0, (60.0, 30.0), 2.0, 0.4, stop=False),
            key(9.0, (50.0, 60.0), 1.0))  # fmt: skip
    path = track(*keys).path(CLOCK, BASE)
    for k, t in zip(keys, (1.0, 5.0, 9.0), strict=True):  # it passes through every key
        shot = path.shot(t)
        assert shot.center == pytest.approx(k.shot.center) and shot.zoom == pytest.approx(
            k.shot.zoom
        )
    assert np.abs(velocity(path, 1.0 + 1e-3)).max() < 0.05  # just leaving a stop: at rest
    assert np.abs(velocity(path, 9.0 - 1e-3)).max() < 0.05  # just arriving: at rest
    before, after = velocity(path, 5.0 - 2e-3), velocity(path, 5.0 + 2e-3)
    assert np.abs(after).max() > 1.0  # moving through the middle key
    np.testing.assert_allclose(before, after, rtol=0.05, atol=0.05)  # without a jolt


def test_keys_about_one_point_never_drift():
    point = (30.0, 25.0)

    def about(zoom, rotation):  # shots that see `point` at the same place in the frame
        anchor = complex(10.0, -5.0)  # where point sits in the frame
        center = complex(*point) - anchor * cmath.exp(1j * rotation) / zoom
        return Shot((center.real, center.imag), zoom, rotation)

    keys = [
        CameraKey(KeyTime.at(1.0, CLOCK), about(5.0, 0.3)),
        CameraKey(KeyTime.at(4.0, CLOCK), about(2.0, -0.2), stop=False),  # flows through
        CameraKey(KeyTime.at(9.0, CLOCK), about(1.0, 0.0)),
    ]
    path = CameraTrack(tuple(keys)).path(CLOCK, BASE)
    seen = [frame_position(path.shot(t), point) for t in np.linspace(1.0, 9.0, 33)]
    np.testing.assert_allclose(seen, [seen[0]] * len(seen), atol=1e-6)


def test_linear_moves_steadily_and_hold_cuts():
    steady = track(key(1.0, (20.0, 40.0), 2.0, motion="linear"),
                   key(5.0, (80.0, 40.0), 2.0)).path(CLOCK, BASE)  # fmt: skip
    xs = [steady.shot(t).center[0] for t in (1.0, 2.0, 3.0, 4.0, 5.0)]
    np.testing.assert_allclose(np.diff(xs), [15.0] * 4)  # the same distance every second
    held = track(key(1.0, zoom=3.0, motion="hold"), key(5.0)).path(CLOCK, BASE)
    assert held.shot(4.99).zoom == 3.0 and held.shot(5.0).zoom == 1.0


def test_turns_stay_unwrapped_for_spins():
    spin = track(key(1.0, motion="linear"), key(9.0, rotation=4 * math.pi)).path(CLOCK, BASE)
    assert spin.shot(5.0).rotation == pytest.approx(2 * math.pi)  # one turn of two, not none


# Editing and files


def test_tracks_edit_into_new_tracks_and_round_trip():
    t = track(key(2.0), key(6.0, zoom=2.0))
    replaced = t.with_key(key(6.0, zoom=4.0), CLOCK)  # same moment: replaces
    assert len(replaced.keys) == 2 and replaced.keys[1].shot.zoom == 4.0
    added = t.with_key(key(4.0, zoom=3.0), CLOCK)
    assert [k.shot.zoom for k in added.keys] == [1.0, 3.0, 2.0]  # kept in time order
    assert len(added.without(0).keys) == 2 and len(t.keys) == 2  # t itself unchanged
    pinned = added.restretched(False, CLOCK)
    np.testing.assert_allclose(pinned.times(CLOCK), added.times(CLOCK))
    assert not pinned.stretch and pinned.keys[1].time == KeyTime("body", 3.0)

    again = CameraTrack.from_dict(added.to_dict())
    assert again == added
    problems = []
    data = added.to_dict()
    data["keys"][1]["zoom"] = "close"
    data["keys"][2]["motion"] = "warp"
    loaded = CameraTrack.from_dict(data, problems)
    assert len(loaded.keys) == 2 and loaded.keys[1].motion == "smooth" and problems


# The video clock


def test_the_video_clock_and_its_moments():
    assert CLOCK.total == 11.0 and CLOCK.animation_end == 9.0
    np.testing.assert_allclose(CLOCK.animation_time([0.0, 1.0, 5.0, 10.0]), [0.0, 0.0, 4.0, 8.0])
    frames = SimpleNamespace(frame=lambda t: t)
    clocked = ClockedTimeline(frames, CLOCK)
    assert clocked.duration == 11.0 and clocked.frame(5.0) == 4.0 and clocked.frame(10.5) == 8.0

    scene = SimpleNamespace(bounds=(0.0, 0.0, 100.0, 80.0), canvas=(100.0, 80.0))
    settings = VideoSettings(frame_rate="30", hold_start=1.0, hold_end=2.0, motion_blur=8)
    plan = plan_video(scene, 8.0, settings, "#000000")
    assert plan.video_clock == CLOCK
    video, tiles = plan.video_moments(300), plan.moments(300)  # 10 s: in the end hold
    assert video.min() > 9.9 and np.all(tiles == 8.0)  # the camera's clock runs on
    np.testing.assert_allclose(plan.moments(75), CLOCK.animation_time(plan.video_moments(75)))


def test_the_camera_moves_over_the_finished_mosaic_in_the_end_hold(tmp_path, gpu):  # noqa: F811
    from skitter.ui.render.video_renderer import VideoRenderer

    scene, _, plan, timeline, pages, base = render_setup(tmp_path, motion_blur=0, hold_end=1.0)
    clock = plan.video_clock
    center = home_shot(plan.view).center
    push = CameraTrack((CameraKey(KeyTime("tail", 0.0), Shot(center)),
                        CameraKey(KeyTime("tail", 1.0), Shot(center, 2.0))))  # fmt: skip
    path = push.path(clock, plan.view)
    renderer = VideoRenderer(plan, pages, base, (0, 0, 0), supersampling=1)
    try:
        middle = plan.frames - 1 - round(0.5 * float(plan.fps))  # half way through the hold
        still = [renderer.render(timeline, k) for k in (middle, plan.frames - 1)]
        moving = [renderer.render(timeline, k, path) for k in (middle, plan.frames - 1)]
    finally:
        renderer.release()
    assert np.array_equal(still[0], still[1])  # the finished mosaic, unmoving
    assert not np.array_equal(moving[0], moving[1])  # the camera still pushes in

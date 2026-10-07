"""Camera moves (core/animation/camera.py): shots over the animation's time."""

import cmath
import math
from types import SimpleNamespace

import numpy as np
import pytest
from test_video import gpu, render_setup  # noqa: F401 (fixture, helper)

from skitter.core.animation.camera import (
    FollowMove,
    PanMove,
    PullBackMove,
    PushInMove,
    RotateMove,
    SampledPath,
    Segment,
    SegmentPath,
    Shot,
    StaticMove,
    StillPath,
    camera_move_types,
    get_camera_move,
    home_shot,
)

BASE = (-10.0, 0.0, 120.0, 80.0)  # the video's framing (zoom 1) around a 100 x 80 mosaic
HOME = (50.0, 40.0)


class FakeScene:
    """Just what camera moves read of a scene."""

    def __init__(self, width=100.0, height=80.0, centers=None, size=4.0):
        self.bounds = (0.0, 0.0, width, height)
        self.canvas = (width, height)
        self.center = np.zeros((0, 2)) if centers is None else np.asarray(centers, float)
        self.size = np.full((len(self.center), 2), size)

    def __len__(self):
        return len(self.center)


def scene(**kwargs):
    return FakeScene(**kwargs)


def timeline(duration=10.0):
    return SimpleNamespace(duration=duration)


def frame_position(shot: Shot, point) -> np.ndarray:
    """Where a table point sits in the frame (0..1 across and down), turned with it."""
    w, h = shot.size(BASE)
    d = complex(*point) - complex(*shot.center)
    d *= cmath.exp(-1j * shot.rotation)  # into the frame's own axes
    return np.array([d.real / w + 0.5, d.imag / h + 0.5])


def test_moves_are_registered_with_static_first():
    ids = [cls.id for cls in camera_move_types()]
    assert ids[0] == "static"
    assert {"pull_back", "push_in", "pan", "rotate", "follow"} <= set(ids)
    assert get_camera_move("pull_back") is PullBackMove
    with pytest.raises(KeyError):
        get_camera_move("dolly_zoom")


def test_a_shot_is_a_turned_part_of_the_video_view():
    assert home_shot(BASE) == Shot(HOME)
    assert Shot((20.0, 20.0), 4.0).size(BASE) == (30.0, 20.0)
    corners = Shot((0.0, 0.0), 1.0, math.pi / 2).corners((0, 0, 4.0, 2.0))
    np.testing.assert_allclose(corners[0], (1.0, -2.0), atol=1e-12)  # top left, turned 90°


def test_static_holds_the_video_framing():
    path = StaticMove().path(scene(), timeline(), BASE)
    assert isinstance(path, StillPath) and path.max_zoom == 1.0
    assert path.shot(0.0) == home_shot(BASE) == path.shot(99.0)


def test_pull_back_starts_close_on_the_focus_and_ends_on_the_whole_mosaic():
    move = PullBackMove(focus_x=40.0, focus_y=50.0, zoom=4.0, timing=(10.0, 80.0))
    path = move.path(scene(), timeline(10.0), BASE)
    assert path.max_zoom == 4.0
    start = path.shot(0.0)
    assert start.zoom == pytest.approx(4.0) and start.center == pytest.approx((40.0, 40.0))
    assert path.shot(1.0) == start  # still until 10% of the animation
    end = path.shot(8.0)
    assert end.zoom == pytest.approx(1.0) and end.center == pytest.approx(HOME)
    assert path.shot(10.0) == end  # and still after 80%
    zooms = [path.shot(t).zoom for t in np.linspace(1.0, 8.0, 30)]
    assert all(a >= b for a, b in zip(zooms, zooms[1:], strict=False))  # only ever widens


def test_segments_scale_and_turn_about_one_fixed_point():
    for start, end in [
        (Shot((20.0, 30.0), 6.0), Shot(HOME)),  # a pure zoom
        (Shot(HOME, 1.5, -0.6), Shot(HOME)),  # turning while pulling back
        (Shot((10.0, 60.0), 2.0, 0.3), Shot((70.0, 20.0), 1.2, -0.2)),  # everything at once
    ]:
        segment = Segment(0.0, 10.0, start, end)
        fixed = segment.fixed_point
        assert fixed is not None
        seen = [frame_position(segment.shot(t), (fixed.real, fixed.imag))
                for t in np.linspace(0, 10, 9)]  # fmt: skip
        np.testing.assert_allclose(seen, [seen[0]] * len(seen), atol=1e-9)  # it never moves
        assert segment.shot(10.0).center == pytest.approx(end.center)
        assert segment.shot(10.0).rotation == pytest.approx(end.rotation)
    pan = Segment(0.0, 10.0, Shot((10.0, 40.0), 2.0), Shot((90.0, 40.0), 2.0), "linear")
    assert pan.fixed_point is None and pan.shot(5.0).center == pytest.approx((50.0, 40.0))


def test_close_ups_near_an_edge_stay_on_the_mosaic():
    path = PullBackMove(focus_x=0.0, focus_y=100.0, zoom=4.0).path(scene(), timeline(), BASE)
    shot = path.shot(0.0)
    w, h = shot.size(BASE)
    assert shot.center[0] - w / 2 == pytest.approx(0.0)
    assert shot.center[1] + h / 2 == pytest.approx(80.0)  # the corner, no background
    # A close-up wider than the mosaic on an axis centers on that axis.
    wide = PullBackMove(focus_x=0.0, zoom=1.1).path(scene(), timeline(), BASE)
    assert wide.shot(0.0).center[0] == pytest.approx(50.0)


def test_push_in_reverses_pull_back_and_ends_close():
    pull = PullBackMove(focus_x=70.0, zoom=3.0, timing=(0.0, 100.0)).path(scene(), timeline(), BASE)
    push = PushInMove(focus_x=70.0, zoom=3.0, timing=(0.0, 100.0)).path(scene(), timeline(), BASE)
    for t in np.linspace(0, 10, 7):
        a, b = pull.shot(t), push.shot(10.0 - t)
        assert a.zoom == pytest.approx(b.zoom) and a.center == pytest.approx(b.center)
    assert push.shot(10.0).zoom == pytest.approx(3.0)


def test_timing_follows_the_duration_and_easing_shapes_the_move():
    move = PullBackMove(zoom=4.0, timing=(0.0, 50.0))
    short, long = move.path(scene(), timeline(4.0), BASE), move.path(scene(), timeline(40.0), BASE)
    assert short.shot(2.0).zoom == pytest.approx(1.0) == long.shot(20.0).zoom
    assert short.shot(1.0).zoom == pytest.approx(long.shot(10.0).zoom)
    linear = PullBackMove(zoom=4.0, ease="linear").path(scene(), timeline(), BASE)
    assert linear.shot(4.0).zoom == pytest.approx(2.0)  # geometric: halfway is 2x, not 2.5x
    gentle = PullBackMove(zoom=4.0, ease="in").path(scene(), timeline(), BASE)
    assert gentle.shot(4.0).zoom > linear.shot(4.0).zoom  # a gentle start has moved less


def test_no_zoom_is_still():
    path = PushInMove(zoom=1.0).path(scene(), timeline(), BASE)
    assert isinstance(path, StillPath) and path.shot(5.0) == home_shot(BASE)


def test_pan_glides_close_up_then_pulls_back():
    move = PanMove(start="left", finish="right", zoom=2.0, timing=(0.0, 80.0))
    path = move.path(scene(), timeline(10.0), BASE)
    assert isinstance(path, SegmentPath) and path.max_zoom == 2.0
    first, mid = path.shot(0.0), path.shot(6.0)  # the pan takes 75% of 0..8 s
    w, _ = first.size(BASE)
    assert first.zoom == mid.zoom == 2.0
    assert first.center[0] == pytest.approx(w / 2)  # at the left edge, still on the mosaic
    assert mid.center[0] == pytest.approx(100.0 - w / 2)  # at the right edge
    assert path.shot(8.0).zoom == pytest.approx(1.0)  # then the whole mosaic
    stay = PanMove(start="top_left", finish="bottom_right", ending="stay").path(
        scene(), timeline(), BASE
    )
    assert stay.shot(10.0).zoom == pytest.approx(2.5)


def test_rotate_turns_upright_onto_the_whole_mosaic():
    path = RotateMove(angle=30.0, zoom=1.5, timing=(0.0, 100.0)).path(scene(), timeline(), BASE)
    start, end = path.shot(0.0), path.shot(10.0)
    assert start.rotation == pytest.approx(-math.radians(30))  # the frame turns back
    assert start.zoom == 1.5 and start.center == pytest.approx(HOME)
    assert end == home_shot(BASE)
    rotations = [path.shot(t).rotation for t in np.linspace(0, 10, 11)]
    assert all(a <= b for a, b in zip(rotations, rotations[1:], strict=False))
    assert isinstance(RotateMove(angle=0.0, zoom=1.0).path(scene(), timeline(), BASE), StillPath)


class BuildCorners:
    """Tiles landing in the top-left quarter at 2.5 s, the bottom-right one at 7.5 s
    (the rest lie still)."""

    def __init__(self, scene_, duration=10.0):
        self.duration = duration
        self.scene = scene_
        x, y = scene_.center.T
        self.land = np.where((x < 50) & (y < 40), 2.5, np.where((x > 50) & (y > 40), 7.5, -9.0))

    def frame(self, t):
        flying = np.abs(t - self.land) < 2.0
        center = np.where(flying[:, None], self.scene.center + 30.0, self.scene.center)
        return SimpleNamespace(alpha=np.ones(len(flying)), rest=~flying, center=center)


def test_follow_watches_where_the_mosaic_is_being_built():
    xs, ys = np.meshgrid(np.arange(5, 100, 10.0), np.arange(5, 80, 10.0))
    s = scene(centers=np.stack([xs.ravel(), ys.ravel()], axis=1))
    path = FollowMove(closest=4.0, room=1.0, smoothness=0.0, ending=20.0).path(
        s, BuildCorners(s), BASE
    )
    assert isinstance(path, SampledPath) and 1.0 < path.max_zoom <= 4.0
    left, right, end = path.shot(2.5), path.shot(7.5), path.shot(10.0)
    assert left.center[0] < 40 < 60 < right.center[0]  # it went where the tiles were landing
    assert left.center[1] < 30 < 50 < right.center[1]
    assert left.zoom > 1.5 and right.zoom > 1.5
    assert end.zoom == pytest.approx(1.0) and end.center == pytest.approx(HOME)
    calm = FollowMove(closest=4.0, room=1.0, smoothness=30.0).path(s, BuildCorners(s), BASE)
    assert abs(calm.shot(2.5).center[0] - 50) < abs(left.center[0] - 50)  # smoothed toward

    nothing = SimpleNamespace(duration=5.0, frame=lambda t: SimpleNamespace(
        alpha=np.ones(len(s)), rest=np.ones(len(s), bool), center=s.center))  # fmt: skip
    assert isinstance(FollowMove().path(s, nothing, BASE), StillPath)


# Rendering


def test_the_turned_2d_camera_maps_screen_and_world_both_ways():
    from skitter.ui.render.camera import Camera2D

    cam = Camera2D()
    cam.center, cam.zoom, cam.viewport = np.array([10.0, 20.0]), 2.0, np.array([100.0, 50.0])
    cam.rotation = 0.7
    world = cam.screen_to_world(80.0, 5.0)
    np.testing.assert_allclose(cam.world_to_screen(*world), (80.0, 5.0))
    np.testing.assert_allclose(cam.screen_to_world(50.0, 25.0), (10.0, 20.0))
    before = cam.screen_to_world(30.0, 40.0)
    cam.pan_pixels(7.0, -3.0)  # content follows the drag
    np.testing.assert_allclose(cam.screen_to_world(37.0, 37.0), before)


def test_the_video_renders_each_frame_through_its_shot(tmp_path, gpu):  # noqa: F811
    from skitter.ui.render.video_renderer import VideoRenderer

    scene_, _, plan, timeline_, pages, base = render_setup(tmp_path, motion_blur=0)
    # Push in on the top-left tile (a quarter of the 2 x 2 mosaic) to 2x: it fills the frame.
    path = PushInMove(focus_x=25.0, focus_y=25.0, zoom=2.0, timing=(0.0, 100.0)).path(
        scene_, timeline_, plan.view
    )
    # Turned a half turn, the finished mosaic shows upside down.
    upside_down = StillPath(Shot(home_shot(plan.view).center, 1.0, math.pi))
    renderer = VideoRenderer(plan, pages, base, (0, 0, 0), supersampling=1)
    try:
        last = plan.frames - 1
        whole = renderer.render(timeline_, last)
        close = renderer.render(timeline_, last, path)
        turned = renderer.render(timeline_, last, upside_down)
    finally:
        renderer.release()
    quarter = whole[: plan.height // 2, : plan.width // 2, :3].astype(float)
    assert close.shape == whole.shape
    np.testing.assert_allclose(close[..., :3].mean(axis=(0, 1)), quarter.mean(axis=(0, 1)),
                               atol=4)  # fmt: skip
    assert not np.array_equal(close, whole)
    flipped = whole[::-1, ::-1, :3].astype(int)
    assert np.median(np.abs(turned[..., :3].astype(int) - flipped)) <= 1

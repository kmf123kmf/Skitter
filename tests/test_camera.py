"""Camera moves (core/animation/camera.py): shots over the animation's time."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_video import gpu, render_setup  # noqa: F401 (fixture, helper)

from skitter.core.animation.camera import (
    PullBackMove,
    PushInMove,
    Shot,
    StaticMove,
    StillPath,
    ZoomPath,
    camera_move_types,
    get_camera_move,
)

BASE = (-10.0, 0.0, 120.0, 80.0)  # the video's framing (zoom 1) around a 100 x 80 mosaic
HOME = (50.0, 40.0)


def scene(width=100.0, height=80.0):
    """Just what camera moves read of a scene: its bounds and canvas."""
    return SimpleNamespace(bounds=(0.0, 0.0, width, height), canvas=(width, height))


def timeline(duration=10.0):
    return SimpleNamespace(duration=duration)


def frame_position(shot: Shot, point) -> np.ndarray:
    """Where a table point sits in the frame (0..1 across and down)."""
    x, y, w, h = shot.view(BASE)
    return (np.asarray(point) - (x, y)) / (w, h)


def test_moves_are_registered_with_static_first():
    ids = [cls.id for cls in camera_move_types()]
    assert ids[0] == "static" and {"pull_back", "push_in"} <= set(ids)
    assert get_camera_move("pull_back") is PullBackMove
    with pytest.raises(KeyError):
        get_camera_move("dolly_zoom")


def test_a_shot_frames_part_of_the_video_view():
    assert Shot(HOME).view(BASE) == BASE
    assert Shot((20.0, 20.0), 4.0).view(BASE) == (5.0, 10.0, 30.0, 20.0)


def test_static_holds_the_video_framing():
    path = StaticMove().path(scene(), timeline(), BASE)
    assert isinstance(path, StillPath) and path.max_zoom == 1.0
    assert path.shot(0.0).view(BASE) == BASE == path.shot(99.0).view(BASE)


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


def test_zoom_moves_are_pure_zooms_about_one_table_point():
    path = PullBackMove(focus_x=20.0, focus_y=30.0, zoom=6.0).path(scene(), timeline(), BASE)
    assert isinstance(path, ZoomPath)
    anchor = path.anchor
    seen = [frame_position(path.shot(t), anchor) for t in np.linspace(0, 10, 12)]
    np.testing.assert_allclose(seen, [seen[0]] * len(seen), atol=1e-9)  # it never moves


def test_close_ups_near_an_edge_stay_on_the_mosaic():
    path = PullBackMove(focus_x=0.0, focus_y=100.0, zoom=4.0).path(scene(), timeline(), BASE)
    x, y, w, h = path.shot(0.0).view(BASE)
    assert x == pytest.approx(0.0) and y + h == pytest.approx(80.0)  # the corner, no background
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
    assert isinstance(path, StillPath) and path.shot(5.0).view(BASE) == BASE


# Rendering


def test_the_video_renders_each_frame_through_its_shot(tmp_path, gpu):  # noqa: F811
    from skitter.ui.render.video_renderer import VideoRenderer

    scene_, _, plan, timeline_, pages, base = render_setup(tmp_path, motion_blur=0)
    # Push in on the top-left tile (a quarter of the 2 x 2 mosaic) to 2x: it fills the frame.
    path = PushInMove(focus_x=25.0, focus_y=25.0, zoom=2.0, timing=(0.0, 100.0)).path(
        scene_, timeline_, plan.view
    )
    renderer = VideoRenderer(plan, pages, base, (0, 0, 0), supersampling=1)
    try:
        last = plan.frames - 1
        whole = renderer.render(timeline_, last)
        close = renderer.render(timeline_, last, path)
    finally:
        renderer.release()
    quarter = whole[: plan.height // 2, : plan.width // 2, :3].astype(float)
    assert close.shape == whole.shape
    np.testing.assert_allclose(close[..., :3].mean(axis=(0, 1)), quarter.mean(axis=(0, 1)),
                               atol=4)  # fmt: skip
    assert not np.array_equal(close, whole)

"""Camera shots (camera.py) and the turned 2D camera that renders them."""

import math

import numpy as np
from test_video import gpu, render_setup  # noqa: F401 (fixture, helper)

from skitter.core.animation.camera import Shot, StillPath, home_shot
from skitter.core.animation.keyframes import CameraKey, CameraTrack, KeyTime
from skitter.core.animation.video import content_rect

BASE = (-10.0, 0.0, 120.0, 80.0)  # the video's framing (zoom 1) around a 100 x 80 mosaic
HOME = (50.0, 40.0)


def test_a_shot_is_a_turned_part_of_the_video_view():
    assert home_shot(BASE) == Shot(HOME)
    assert Shot((20.0, 20.0), 4.0).size(BASE) == (30.0, 20.0)
    corners = Shot((0.0, 0.0), 1.0, math.pi / 2).corners((0, 0, 4.0, 2.0))
    np.testing.assert_allclose(corners[0], (1.0, -2.0), atol=1e-12)  # top left, turned 90°
    assert StillPath(Shot(HOME, 2.0)).shot(5.0).zoom == 2.0


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
    x0, y0, x1, y1 = content_rect(scene_)
    corner = ((3 * x0 + x1) / 4, (3 * y0 + y1) / 4)
    keys = [CameraKey(KeyTime("body", 0.0), home_shot(plan.view)),
            CameraKey(KeyTime("body", 1.0), Shot(corner, 2.0))]  # fmt: skip
    path = CameraTrack.of(keys).path(plan.video_clock, plan.view)
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
    np.testing.assert_allclose(close[..., :3].mean(axis=(0, 1)), quarter.mean(axis=(0, 1)),
                               atol=4)  # fmt: skip
    assert not np.array_equal(close, whole)
    flipped = whole[::-1, ::-1, :3].astype(int)
    assert np.median(np.abs(turned[..., :3].astype(int) - flipped)) <= 1

"""Camera shots (camera.py) and ready-made moves written as keys (camera_moves.py)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from test_video import gpu, render_setup  # noqa: F401 (fixture, helper)

from skitter.core.animation.camera import Shot, StillPath, home_shot, keep_on_mosaic
from skitter.core.animation.camera_moves import FollowMove, camera_move_types, get_camera_move
from skitter.core.animation.keyframes import CameraKey, CameraTrack, KeyTime
from skitter.core.animation.video import VideoClock

BASE = (-10.0, 0.0, 120.0, 80.0)  # the video's framing (zoom 1) around a 100 x 80 mosaic
HOME = (50.0, 40.0)
CLOCK = VideoClock(0.0, 10.0, 0.0)  # keys' times are shares of this 10 s animation


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


def played(move, s=None, t=None):
    """A move's keys as the camera plays them over CLOCK."""
    keys = move.keys(s or scene(), t or timeline(), BASE)
    return keys, CameraTrack.of(keys).path(CLOCK, BASE)


def test_moves_are_registered():
    ids = [cls.id for cls in camera_move_types()]
    assert ids == ["follow"]
    assert get_camera_move("follow") is FollowMove
    for gone in ("static", "pull_back", "pan"):  # no keys: still; simple moves: keyed by hand
        with pytest.raises(KeyError):
            get_camera_move(gone)


def test_a_shot_is_a_turned_part_of_the_video_view():
    assert home_shot(BASE) == Shot(HOME)
    assert Shot((20.0, 20.0), 4.0).size(BASE) == (30.0, 20.0)
    corners = Shot((0.0, 0.0), 1.0, math.pi / 2).corners((0, 0, 4.0, 2.0))
    np.testing.assert_allclose(corners[0], (1.0, -2.0), atol=1e-12)  # top left, turned 90°
    assert StillPath(Shot(HOME, 2.0)).shot(5.0).zoom == 2.0


def test_shots_near_an_edge_stay_on_the_mosaic():
    s = scene()
    center = keep_on_mosaic(s, BASE, (0.0, 80.0), 4.0)
    w, h = Shot(center, 4.0).size(BASE)
    assert center[0] - w / 2 == pytest.approx(0.0)
    assert center[1] + h / 2 == pytest.approx(80.0)  # the corner, no background
    assert keep_on_mosaic(s, BASE, (0.0, 40.0), 1.1)[0] == pytest.approx(50.0)  # wider: centered


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


def test_follow_writes_keys_that_watch_where_the_mosaic_is_built():
    xs, ys = np.meshgrid(np.arange(5, 100, 10.0), np.arange(5, 80, 10.0))
    s = scene(centers=np.stack([xs.ravel(), ys.ravel()], axis=1))
    move = FollowMove(count=11, closest=4.0, room=1.0, smoothness=0.0, ending=20.0)
    keys, path = played(move, s, BuildCorners(s))
    assert len(keys) == 11 and keys[0].stop and keys[-1].stop
    assert not any(k.stop for k in keys[1:-1])  # it flows from key to key
    left, right = path.shot(2.5), path.shot(7.5)
    assert left.center[0] < 40 < 60 < right.center[0]  # where the tiles were landing
    assert left.center[1] < 30 < 50 < right.center[1]
    assert left.zoom > 1.5 and right.zoom > 1.5
    assert keys[-1].shot.zoom == pytest.approx(1.0) and keys[-1].shot.center == pytest.approx(HOME)
    nothing = SimpleNamespace(duration=5.0, frame=lambda t: SimpleNamespace(
        alpha=np.ones(len(s)), rest=np.ones(len(s), bool), center=s.center))  # fmt: skip
    assert FollowMove().keys(s, nothing, BASE) == []


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
    corner = keep_on_mosaic(scene_, plan.view, (0.0, 0.0), 2.0)
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

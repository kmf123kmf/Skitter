"""GL-free parts of the renderer: camera math and sprite data."""

import numpy as np

from skitter.ui.render.camera import Camera2D, clamp_center
from skitter.ui.render.sprites import INSTANCE_DTYPE, SpriteLayer, make_instances


def make_camera():
    camera = Camera2D()
    camera.viewport[:] = (800, 600)
    return camera


def test_screen_world_roundtrip():
    camera = make_camera()
    camera.center[:] = (100, 50)
    camera.zoom = 2.5
    world = camera.screen_to_world(123, 456)
    np.testing.assert_allclose(camera.world_to_screen(*world), [123, 456])


def test_zoom_keeps_anchor_fixed():
    camera = make_camera()
    before = camera.screen_to_world(200, 150)
    camera.zoom_at(200, 150, 3.0)
    np.testing.assert_allclose(camera.screen_to_world(200, 150), before)
    assert camera.zoom == 3.0


def test_fit_frames_rect():
    camera = make_camera()
    camera.fit(0, 0, 1600, 600, margin=1.0)
    assert camera.zoom == 0.5
    np.testing.assert_allclose(camera.world_to_screen(0, 0), [0, 150])
    np.testing.assert_allclose(camera.world_to_screen(1600, 600), [800, 450])


def test_pan_moves_content_with_drag():
    camera = make_camera()
    camera.zoom = 2.0
    before = camera.world_to_screen(10, 10)
    camera.pan_pixels(30, -20)
    np.testing.assert_allclose(camera.world_to_screen(10, 10), before + [30, -20])


def test_clamp_centers_small_content_and_limits_large():
    viewport = np.array([800.0, 600.0])
    # 1000x200 world rect at zoom 1: wider than the view, shorter than it.
    center = clamp_center((-500, 999), 1.0, viewport, (0, 0, 1000, 200))
    np.testing.assert_allclose(center, [400, 100])  # left edge aligned; vertically centered
    center = clamp_center((5000, 0), 1.0, viewport, (0, 0, 1000, 200))
    np.testing.assert_allclose(center, [600, 100])  # right edge aligned


def test_fit_ignores_min_zoom():
    camera = make_camera()
    camera.min_zoom = 1.0
    camera.fit(0, 0, 8000, 6000, margin=1.0)
    assert camera.zoom == 0.1


def test_instance_layout_is_packed():
    assert INSTANCE_DTYPE.itemsize == 18 * 4
    instances = make_instances(3)
    assert (instances["alpha"] == 1).all()
    assert (instances["uv"] == (0, 0, 1, 1)).all() and (instances["offset"] == 0).all()


def test_sprite_layer_promotes_single_rgb_image():
    layer = SpriteLayer(np.zeros((5, 7, 3), np.uint8), make_instances(1))
    assert layer.textures.shape == (1, 5, 7, 4)
    assert (layer.textures[..., 3] == 255).all()

"""GL-free parts of the renderer: camera math and sprite data."""

import numpy as np

from skitter.ui.render.camera import Camera2D
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


def test_instance_layout_is_packed():
    assert INSTANCE_DTYPE.itemsize == 11 * 4
    assert (make_instances(3)["alpha"] == 1).all()


def test_sprite_layer_promotes_single_rgb_image():
    layer = SpriteLayer(np.zeros((5, 7, 3), np.uint8), make_instances(1))
    assert layer.textures.shape == (1, 5, 7, 4)
    assert (layer.textures[..., 3] == 255).all()

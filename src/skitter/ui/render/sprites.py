"""Instanced sprite rendering: many textured quads in a single draw call.

A SpriteLayer is plain CPU-side data: a stack of same-sized textures and a
structured numpy array with one row per sprite. Animations edit the instance
array directly and call mark_dirty(); the SpriteRenderer uploads changes on
the next frame.
"""

from importlib.resources import files

import moderngl
import numpy as np

from skitter.ui.render.camera import Camera2D

INSTANCE_DTYPE = np.dtype(
    [
        ("pos", "f4", 2),  # world-space center
        ("size", "f4", 2),  # world-space width, height
        ("rotation", "f4"),  # radians
        ("alpha", "f4"),
        ("layer", "f4"),  # index into the layer's textures
        ("tint", "f4", 4),  # rgb in [0, 1], a = tint strength
    ]
)
_INSTANCE_FORMAT = "2f 2f 1f 1f 1f 4f /i"
_INSTANCE_ATTRIBUTES = ("in_pos", "in_size", "in_rotation", "in_alpha", "in_layer", "in_tint")


def make_instances(count: int) -> np.ndarray:
    """Zeroed instance array with full opacity."""
    instances = np.zeros(count, INSTANCE_DTYPE)
    instances["alpha"] = 1.0
    return instances


def to_rgba(textures: np.ndarray) -> np.ndarray:
    """Convert (N, H, W, 3|4) uint8 textures to contiguous RGBA."""
    if textures.shape[-1] == 3:
        alpha = np.full((*textures.shape[:-1], 1), 255, dtype=np.uint8)
        textures = np.concatenate([textures, alpha], axis=-1)
    return np.ascontiguousarray(textures, dtype=np.uint8)


class SpriteLayer:
    def __init__(self, textures: np.ndarray, instances: np.ndarray):
        """textures: (H, W, C) for a single texture or (N, H, W, C) for a stack."""
        textures = np.asarray(textures)
        if textures.ndim == 3:
            textures = textures[None]
        self.textures = to_rgba(textures)
        self.instances = instances
        self.visible = True
        self.dirty = True

    def mark_dirty(self) -> None:
        """Flag the instance array for upload on the next frame."""
        self.dirty = True


def _load_shader(name: str) -> str:
    return files("skitter").joinpath(f"resources/shaders/{name}").read_text()


class _GpuLayer:
    def __init__(self, ctx: moderngl.Context, program: moderngl.Program, quad, layer):
        count, h, w, _ = layer.textures.shape
        max_layers = ctx.info["GL_MAX_ARRAY_TEXTURE_LAYERS"]
        if count > max_layers:
            raise ValueError(f"{count} textures exceeds the GPU limit of {max_layers} per layer")
        self.texture = ctx.texture_array((w, h, count), 4, layer.textures.tobytes())
        self.texture.repeat_x = self.texture.repeat_y = False
        self.texture.build_mipmaps()
        self.texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.texture.anisotropy = 8.0

        self.ctx, self.program, self.quad = ctx, program, quad
        self.capacity = 0
        self.vbo = None
        self.vao = None

    def write_instances(self, instances: np.ndarray) -> None:
        if len(instances) > self.capacity:
            self._release_buffers()
            self.capacity = max(len(instances), 2 * self.capacity)
            nbytes = self.capacity * INSTANCE_DTYPE.itemsize
            self.vbo = self.ctx.buffer(reserve=nbytes, dynamic=True)
            self.vao = self.ctx.vertex_array(
                self.program,
                [
                    (self.quad, "2f", "in_corner"),
                    (self.vbo, _INSTANCE_FORMAT, *_INSTANCE_ATTRIBUTES),
                ],
            )
        if len(instances):
            self.vbo.write(np.ascontiguousarray(instances, dtype=INSTANCE_DTYPE).tobytes())

    def render(self, count: int) -> None:
        if self.vao is None or count == 0:
            return
        self.texture.use(0)
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=count)

    def _release_buffers(self) -> None:
        if self.vao is not None:
            self.vao.release()
            self.vbo.release()

    def release(self) -> None:
        self._release_buffers()
        self.texture.release()


class SpriteRenderer:
    """Owns the shader program and per-layer GPU resources.

    All methods must be called with the OpenGL context current.
    """

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program = ctx.program(
            vertex_shader=_load_shader("sprite.vert.glsl"),
            fragment_shader=_load_shader("sprite.frag.glsl"),
        )
        self.program["u_textures"] = 0
        corners = np.array([-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5], dtype="f4")
        self.quad = ctx.buffer(corners.tobytes())
        self._layers: dict[SpriteLayer, _GpuLayer] = {}

    def set_camera(self, camera: Camera2D) -> None:
        self.program["u_center"] = tuple(camera.center)
        self.program["u_zoom"] = camera.zoom
        self.program["u_viewport"] = tuple(camera.viewport)

    def render(self, layer: SpriteLayer) -> None:
        gpu = self._layers.get(layer)
        if gpu is None:
            gpu = self._layers[layer] = _GpuLayer(self.ctx, self.program, self.quad, layer)
        if layer.dirty:
            gpu.write_instances(layer.instances)
            layer.dirty = False
        gpu.render(len(layer.instances))

    def release(self, layer: SpriteLayer) -> None:
        gpu = self._layers.pop(layer, None)
        if gpu is not None:
            gpu.release()

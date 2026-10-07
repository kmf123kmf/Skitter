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
        ("uv", "f4", 4),  # texture rect (u0, v0, u1, v1) the sprite shows, e.g. an atlas cell
        ("offset", "f4", 3),  # OKLab shift of every texel (matching's tint), before `tint`
        ("blur", "f4"),  # > 0: a soft shadow, edges blurred this far (world units), no texture
    ]
)
_INSTANCE_FORMAT = "2f 2f 1f 1f 1f 4f 4f 3f 1f /i"
_INSTANCE_ATTRIBUTES = (
    "in_pos", "in_size", "in_rotation", "in_alpha", "in_layer", "in_tint", "in_uv", "in_offset",
    "in_blur",
)  # fmt: skip


def make_instances(count: int) -> np.ndarray:
    """Zeroed instance array with full opacity, each showing its whole texture."""
    instances = np.zeros(count, INSTANCE_DTYPE)
    instances["alpha"] = 1.0
    instances["uv"] = (0.0, 0.0, 1.0, 1.0)
    return instances


def to_rgba(textures: np.ndarray) -> np.ndarray:
    """Convert (N, H, W, 3|4) uint8 textures to contiguous RGBA."""
    if textures.shape[-1] == 3:
        alpha = np.full((*textures.shape[:-1], 1), 255, dtype=np.uint8)
        textures = np.concatenate([textures, alpha], axis=-1)
    return np.ascontiguousarray(textures, dtype=np.uint8)


class SpriteLayer:
    """A batch of sprites drawn in one call, plus how to draw them.

    textures: (H, W, C) for a single texture, (N, H, W, C) for a stack, or
    None for plain white (color sprites using each instance's tint).

    Region-marker style (outline_px > 0): each sprite is drawn as a line of
    outline_px screen pixels in its tint color with a dark band of edge_px
    inside it. The interior is the tint at fill_alpha, or, with
    project_texture, the texture shown at the sprite's world position
    (texture_size gives the world size it spans). Sprites draw in instance
    order, so later ones cover earlier ones.

    line_alpha scales the opacity of the line and its dark band.

    texture_from: use another layer's GPU texture instead of uploading this
    layer's own (for example, an overlay sampling the image layer). That
    layer must be drawn earlier in the same canvas.
    """

    def __init__(
        self,
        textures: np.ndarray | None,
        instances: np.ndarray,
        *,
        outline_px: float = 0.0,
        edge_px: float = 0.0,
        fill_alpha: float = 0.0,
        line_alpha: float = 1.0,
        project_texture: bool = False,
        texture_size: tuple[float, float] = (1.0, 1.0),
        texture_from: "SpriteLayer | None" = None,
    ):
        if textures is None:
            textures = np.full((1, 1, 1, 4), 255, dtype=np.uint8)
        textures = np.asarray(textures)
        if textures.ndim == 3:
            textures = textures[None]
        self.textures = to_rgba(textures)
        self.instances = instances
        self.outline_px = outline_px
        self.edge_px = edge_px
        self.fill_alpha = fill_alpha
        self.line_alpha = line_alpha
        self.project_texture = project_texture
        self.texture_size = texture_size
        self.texture_from = texture_from
        self.visible = True
        self.dirty = True
        self.patches: list[tuple[int, int, int, int, int]] = []  # texels to upload again

    def mark_dirty(self) -> None:
        """Flag the instance array for upload on the next frame."""
        self.dirty = True

    def patch_texture(self, rects, source: np.ndarray | None = None) -> None:
        """Texels (page, x, y, w, h) of the textures changed: upload them on the next frame.

        source: the array they were written to, if not this layer's own textures
        (layers made from RGB pages hold an RGBA copy).
        """
        for page, x, y, w, h in rects:
            if source is not None and source is not self.textures:
                self.textures[page, y : y + h, x : x + w, : source.shape[-1]] = source[
                    page, y : y + h, x : x + w
                ]
            self.patches.append((int(page), int(x), int(y), int(w), int(h)))


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
        self.nearest_mag = False

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

    def write_patches(self, textures: np.ndarray, patches) -> None:
        for page, x, y, w, h in patches:
            data = np.ascontiguousarray(textures[page, y : y + h, x : x + w])
            self.texture.write(data.tobytes(), viewport=(x, y, page, w, h, 1))
        self.texture.build_mipmaps()

    def set_nearest_mag(self, nearest: bool) -> None:
        if nearest != self.nearest_mag:
            mag = moderngl.NEAREST if nearest else moderngl.LINEAR
            self.texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, mag)
            self.nearest_mag = nearest

    def render(self, count: int, texture: moderngl.TextureArray) -> None:
        if self.vao is None or count == 0:
            return
        texture.use(0)
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
        # Output linear light instead of sRGB (offscreen rendering into float buffers).
        self.linear_output = False

    def set_camera(self, camera: Camera2D) -> None:
        self.program["u_center"] = tuple(camera.center)
        self.program["u_zoom"] = camera.zoom
        self.program["u_rotation"] = float(camera.rotation)
        self.program["u_viewport"] = tuple(camera.viewport)

    def render(self, layer: SpriteLayer, nearest_mag: bool = False) -> None:
        """Draw a layer; nearest_mag shows magnified texels as sharp squares."""
        gpu = self._layers.get(layer)
        if gpu is None:
            gpu = self._layers[layer] = _GpuLayer(self.ctx, self.program, self.quad, layer)
            layer.patches.clear()  # just uploaded whole
        if layer.patches:
            gpu.write_patches(layer.textures, layer.patches)
            layer.patches.clear()
        if layer.dirty:
            gpu.write_instances(layer.instances)
            layer.dirty = False
        texture_gpu = gpu
        if layer.texture_from is not None:
            texture_gpu = self._layers.get(layer.texture_from)
            if texture_gpu is None:  # source not uploaded (not drawn on this canvas)
                return
        texture_gpu.set_nearest_mag(nearest_mag)

        program = self.program
        program["u_outline_px"] = float(layer.outline_px)
        program["u_edge_px"] = float(layer.edge_px)
        program["u_fill_alpha"] = float(layer.fill_alpha)
        program["u_line_alpha"] = float(layer.line_alpha)
        program["u_project"] = int(layer.project_texture)
        program["u_texture_size"] = tuple(float(v) for v in layer.texture_size)
        program["u_linear"] = int(self.linear_output)
        gpu.render(len(layer.instances), texture_gpu.texture)

    def release(self, layer: SpriteLayer) -> None:
        gpu = self._layers.pop(layer, None)
        if gpu is not None:
            gpu.release()

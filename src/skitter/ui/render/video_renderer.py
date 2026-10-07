"""Offscreen GPU rendering of animation frames for video export.

Uses its own OpenGL context (moderngl standalone), so it runs on a worker
thread without touching the on-screen canvases, and draws tiles with the
same sprite shader as the preview, so the video shows what the Animate tab
shows. Quality steps on top:

- Each output pixel is the average of supersampling² pixels rendered with
  8 x multisampling: up to 128 coverage samples, each owned by exactly one
  tile, so edges are smooth and neighboring tiles never let the background
  through their seams.
- Blending happens in linear light, premultiplied, in 16-bit float buffers;
  frames are converted to sRGB only at the end, laid over the background
  (or keeping straight alpha for transparent output).
- Motion blur averages several moments of the timeline within each frame.
- Large outputs render in sections of at most RENDER_TILE pixels per side
  (supersampled), so GPU memory stays bounded whatever the output size.
"""

import moderngl
import numpy as np

from skitter.core.animation import Timeline
from skitter.core.animation.camera import CameraPath, Shot, home_shot
from skitter.core.animation.look import TableCamera
from skitter.core.animation.video import VideoPlan
from skitter.ui.render.camera import Camera2D
from skitter.ui.render.player import frame_layers
from skitter.ui.render.sprites import SpriteLayer, SpriteRenderer, make_instances

RENDER_TILE = 2048  # largest render pass, supersampled pixels per side
MSAA = 8

_FULLSCREEN = """
#version 330
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);  // one triangle covering the view
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_DOWNSAMPLE = """
#version 330
uniform sampler2D u_source;  // a resolved render pass, premultiplied linear
uniform int u_factor;        // supersampling per side
uniform ivec2 u_origin;      // this pass's corner in the output, pixels
uniform float u_weight;      // share of the frame (motion blur moments)
out vec4 f_color;
void main() {
    ivec2 base = (ivec2(gl_FragCoord.xy) - u_origin) * u_factor;
    vec4 sum = vec4(0.0);
    for (int j = 0; j < u_factor; j++)
        for (int i = 0; i < u_factor; i++)
            sum += texelFetch(u_source, base + ivec2(i, j), 0);
    f_color = sum * (u_weight / float(u_factor * u_factor));
}
"""

_FINISH = """
#version 330
uniform sampler2D u_accum;   // premultiplied linear
uniform int u_alpha;         // 1: keep transparency (straight alpha out)
uniform vec3 u_background;   // linear
out vec4 f_color;
vec3 linear_to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
void main() {
    vec4 c = texelFetch(u_accum, ivec2(gl_FragCoord.xy), 0);
    float a = clamp(c.a, 0.0, 1.0);
    if (u_alpha == 1) {
        f_color = vec4(a > 0.0 ? linear_to_srgb(c.rgb / a) : vec3(0.0), a);
    } else {
        f_color = vec4(linear_to_srgb(c.rgb + u_background * (1.0 - a)), 1.0);
    }
}
"""


def srgb_to_linear(rgb) -> tuple[float, float, float]:
    c = np.asarray(rgb, dtype=np.float64)
    return tuple(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


class GpuUnavailable(RuntimeError):
    pass


class VideoRenderer:
    """Renders frames of a timeline at a plan's size and view.

    Create, use and release it on one thread (OpenGL contexts are per thread).
    """

    def __init__(
        self,
        plan: VideoPlan,
        pages: np.ndarray,
        base: np.ndarray,
        background: tuple[float, float, float] | None,
        supersampling: int,
        camera: TableCamera | None = None,
    ):
        try:
            self.ctx = moderngl.create_standalone_context(require=330)
        except Exception as exc:
            raise GpuUnavailable(f"Could not start the GPU renderer: {exc}") from exc
        ctx = self.ctx
        self.plan = plan
        self.base = base
        self.factor = int(supersampling)
        self.background = srgb_to_linear(background or (0.0, 0.0, 0.0))
        width, height = plan.width, plan.height
        max_side = min(RENDER_TILE, ctx.info["GL_MAX_TEXTURE_SIZE"])
        self.tile = max(16, max_side // self.factor)  # output pixels per pass, per side
        side = self.tile * self.factor
        try:
            self.sprites = SpriteRenderer(ctx)
            self.sprites.linear_output = True
            # Tiles at rest, the shadows of tiles in the air, then the tiles in the air.
            self.layer = SpriteLayer(pages, base[:0].copy())
            self.shadow_layer = SpriteLayer(None, make_instances(0))
            self.air_layer = SpriteLayer(None, base[:0].copy(), texture_from=self.layer)
            self.table_camera = camera  # looks down at the table: perspective, shadows
            samples = min(MSAA, ctx.info["GL_MAX_SAMPLES"])
            self.msaa = ctx.framebuffer(
                ctx.renderbuffer((side, side), 4, samples=samples, dtype="f2")
            )
            self.resolved_tex = ctx.texture((side, side), 4, dtype="f2")
            self.resolved = ctx.framebuffer(self.resolved_tex)
            self.accum_tex = ctx.texture((width, height), 4, dtype="f4")
            self.accum = ctx.framebuffer(self.accum_tex)
            self.out = ctx.framebuffer(ctx.texture((width, height), 4, dtype="f1"))
            self.downsample = ctx.program(vertex_shader=_FULLSCREEN, fragment_shader=_DOWNSAMPLE)
            self.finish = ctx.program(vertex_shader=_FULLSCREEN, fragment_shader=_FINISH)
            self.downsample_vao = ctx.vertex_array(self.downsample, [])
            self.finish_vao = ctx.vertex_array(self.finish, [])
        except Exception:
            self.release()
            raise
        for tex in (self.resolved_tex, self.accum_tex):
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.camera = Camera2D(min_zoom=1e-9, max_zoom=1e9)
        self.passes = [
            (x, y, min(self.tile, width - x), min(self.tile, height - y))
            for y in range(0, height, self.tile)
            for x in range(0, width, self.tile)
        ]

    def render(self, timeline: Timeline, k: int, path: CameraPath | None = None) -> np.ndarray:
        """Frame k as (height, width, 4) uint8 RGBA (straight alpha; opaque unless plan.alpha).

        path: the camera on the video clock (keyframes.video_camera_path); each moment
        blended into the frame is seen through its own shot, so camera motion blurs
        too. None: the plan's view throughout.
        """
        ctx, plan = self.ctx, self.plan
        moments = plan.video_moments(k)  # video time: the camera's clock
        tiles = plan.moments(k)  # the same moments in animation time: the tiles'
        self.accum.use()
        self.accum.clear(0.0, 0.0, 0.0, 0.0)
        weight = 1.0 / len(moments)
        for t, a in zip(moments, tiles, strict=True):
            layers = frame_layers(self.base, timeline.frame(float(a)), self.table_camera)
            for layer, instances in zip(self._layers(), layers, strict=True):
                layer.instances = instances
                layer.mark_dirty()
            shot = home_shot(plan.view) if path is None else path.shot(float(t))
            for section in self.passes:
                self._render_pass(section, weight, shot)
        # Lay over the background (or keep alpha) and convert to sRGB.
        self.out.use()
        ctx.viewport = (0, 0, plan.width, plan.height)
        ctx.disable(moderngl.BLEND)
        self.accum_tex.use(0)
        self.finish["u_accum"] = 0
        self.finish["u_alpha"] = int(plan.alpha)
        self.finish["u_background"] = self.background
        self.finish_vao.render(moderngl.TRIANGLES, vertices=3)
        data = self.out.read(components=4, alignment=1)
        frame = np.frombuffer(data, np.uint8).reshape(plan.height, plan.width, 4)
        return np.ascontiguousarray(frame[::-1])  # OpenGL rows run bottom to top

    def _render_pass(self, section, weight: float, shot: Shot) -> None:
        ctx, plan, s = self.ctx, self.plan, self.factor
        x, y, w, h = section
        scale = plan.width / shot.size(plan.view)[0]  # output pixels per mosaic unit
        cam = self.camera
        cam.rotation = shot.rotation
        cam.zoom = scale * s
        # The section's middle, from the frame's middle, along the (turned) frame.
        offset = np.array([x + w / 2 - plan.width / 2, y + h / 2 - plan.height / 2]) / scale
        cam.center = np.asarray(shot.center) + cam.turn(offset)
        cam.viewport = np.array([w * s, h * s], dtype=float)

        self.msaa.use()
        ctx.viewport = (0, 0, w * s, h * s)
        self.msaa.clear(0.0, 0.0, 0.0, 0.0)
        ctx.enable(moderngl.BLEND)
        # Straight-alpha sprites accumulate premultiplied color and coverage.
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
                          moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)  # fmt: skip
        self.sprites.set_camera(cam)
        for layer in self._layers():
            self.sprites.render(layer)
        ctx.copy_framebuffer(self.resolved, self.msaa)

        # Average the supersampled pixels into this section of the frame.
        top = plan.height - y - h  # OpenGL rows run bottom to top
        self.accum.use()
        ctx.viewport = (x, top, w, h)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        self.resolved_tex.use(0)
        self.downsample["u_source"] = 0
        self.downsample["u_factor"] = s
        self.downsample["u_origin"] = (x, top)
        self.downsample["u_weight"] = weight
        self.downsample_vao.render(moderngl.TRIANGLES, vertices=3)

    def _layers(self) -> tuple[SpriteLayer, SpriteLayer, SpriteLayer]:
        return self.layer, self.shadow_layer, self.air_layer

    def release(self) -> None:
        try:
            self.ctx.release()
        except Exception:
            pass

"""GPU canvas: draws sprite layers with moderngl inside a Qt OpenGL widget."""

import time
from collections.abc import Callable

import moderngl
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from skitter.core.easing import Easing, ease_out_cubic, lerp
from skitter.ui.render.camera import Camera2D, WorldRect, clamp_center
from skitter.ui.render.sprites import SpriteLayer, SpriteRenderer

# Called every frame with seconds elapsed since the animation started.
# Return False once finished to stop being called.
Animation = Callable[[float], bool]

VIEW_ANIMATION_S = 0.2
WHEEL_ZOOM_STEP = 1.15
CHECKER_PX = 8  # checkerboard square size, logical pixels (transparent backgrounds)

_CHECKER_VERTEX = """
#version 330
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
_CHECKER_FRAGMENT = """
#version 330
uniform float u_square;
out vec4 f_color;
void main() {
    vec2 cell = floor(gl_FragCoord.xy / u_square);
    float light = mod(cell.x + cell.y, 2.0);
    f_color = vec4(vec3(mix(0.36, 0.46, light)), 1.0);
}
"""


def configure_opengl() -> None:
    """Request an OpenGL 3.3 core context with vsync and MSAA.

    Must be called before the QApplication is created.
    """
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSwapInterval(1)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)


class MosaicCanvas(QOpenGLWidget):
    """Zoomable, pannable view of sprite layers.

    Layers draw in the order they were added. While any animation is active
    the canvas redraws every frame (vsync-paced); otherwise it redraws only
    when something changes.

    Navigation: set `bounds` to the content's world rect. `zoom_to_fit()`
    frames it and enters fit mode, which refits whenever the widget resizes;
    any other zoom or pan leaves fit mode. With `clamp_to_bounds`, the view
    cannot scroll past the content's edges.
    """

    fps_changed = Signal(float)  # measured each second while animating; 0.0 when idle
    view_changed = Signal()  # camera moved or zoomed, or the widget resized
    cursor_moved = Signal(float, float)  # world coordinates under the mouse
    cursor_left = Signal()
    double_clicked = Signal(float, float)  # screen coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = Camera2D()
        self.background = (0.12, 0.12, 0.12)
        self.checkerboard = False  # draw a checkerboard instead (shows transparency)
        self._checker = None
        self.layers: list[SpriteLayer] = []
        self.ctx: moderngl.Context | None = None
        self.renderer: SpriteRenderer | None = None

        self.bounds: WorldRect | None = None
        self.clamp_to_bounds = False
        self.fit_mode = False
        self.fit_margin = 0.95
        # Show texels as sharp squares once each covers this many device pixels.
        self.pixelate_above: float | None = None

        self._animations: list[tuple[Animation, float | None]] = []
        self._released: list[SpriteLayer] = []
        self._view_token: object | None = None
        self._fbo: moderngl.Framebuffer | None = None
        self._drag_pos: QPointF | None = None
        self._fps_frames = 0
        self._fps_start = time.perf_counter()

        self.setMouseTracking(True)
        self.frameSwapped.connect(self._on_frame_swapped)

    # Layers and animations

    def add_layer(self, layer: SpriteLayer, index: int | None = None) -> SpriteLayer:
        """Add a layer on top, or at index in the draw order (0 draws first)."""
        if index is None:
            self.layers.append(layer)
        else:
            self.layers.insert(index, layer)
        self.update()
        return layer

    def remove_layer(self, layer: SpriteLayer) -> None:
        self.layers.remove(layer)
        self._released.append(layer)  # GPU resources freed on next paint
        self.update()

    def clear_layers(self) -> None:
        for layer in list(self.layers):
            self.remove_layer(layer)

    def add_animation(self, animation: Animation) -> None:
        self._animations.append((animation, None))
        self.update()

    def clear_animations(self) -> None:
        self._animations.clear()

    # View

    def device_zoom(self) -> float:
        """Device pixels per world unit (1.0 shows images at actual size)."""
        return self.camera.zoom * self.devicePixelRatioF()

    def fit_to(self, x: float, y: float, w: float, h: float, animate: bool = False) -> None:
        self.bounds = (x, y, w, h)
        self.zoom_to_fit(animate)

    def zoom_to_fit(
        self,
        animate: bool = False,
        duration: float = VIEW_ANIMATION_S,
        easing: Easing = ease_out_cubic,
    ) -> None:
        """Frame `bounds` and enter fit mode. Deferred until the widget has a size."""
        self.fit_mode = True
        if self.bounds is None or not self._has_viewport():
            self.update()
            return
        center, zoom = self.camera.fit_params(*self.bounds, margin=self.fit_margin)
        self._apply_view(center, zoom, animate, duration, easing)

    def set_view(
        self,
        center,
        zoom: float,
        animate: bool = False,
        duration: float = VIEW_ANIMATION_S,
        easing: Easing = ease_out_cubic,
    ) -> None:
        """Move the camera, leaving fit mode if the view actually changes."""
        cam = self.camera
        zoom = float(np.clip(zoom, cam.min_zoom, cam.max_zoom))
        center = self._clamped(center, zoom)
        if self._view_token is None and zoom == cam.zoom and np.allclose(center, cam.center):
            return
        self.fit_mode = False
        self._apply_view(center, zoom, animate, duration, easing)

    def zoom_at(self, x: float, y: float, factor: float, animate: bool = False) -> None:
        """Zoom by factor about screen point (x, y)."""
        center, zoom = self.camera.zoom_at_params(x, y, factor)
        self.set_view(center, zoom, animate)

    def report_cursor(self, pos: QPointF) -> None:
        """Emit cursor_moved for a screen position (used by overlay widgets)."""
        x, y = self.camera.screen_to_world(pos.x(), pos.y())
        self.cursor_moved.emit(float(x), float(y))

    def _apply_view(self, center, zoom, animate, duration, easing) -> None:
        cam = self.camera
        center = self._clamped(center, zoom)
        if not (animate and self.isVisible()):
            self._view_token = None
            cam.center, cam.zoom = center, zoom
            self.view_changed.emit()
            self.update()
            return

        token = self._view_token = object()
        start_center, start_zoom = cam.center.copy(), cam.zoom

        def step(t: float) -> bool:
            if self._view_token is not token:
                return False
            e = float(easing(min(t / duration, 1.0)))
            cam.zoom = start_zoom * (zoom / start_zoom) ** e  # geometric: even zoom speed
            cam.center = lerp(start_center, center, e)
            self.view_changed.emit()
            if t >= duration:
                self._view_token = None
                return False
            return True

        self.add_animation(step)

    def _clamped(self, center, zoom: float) -> np.ndarray:
        center = np.asarray(center, dtype=float)
        if self.clamp_to_bounds and self.bounds is not None:
            return clamp_center(center, zoom, self.camera.viewport, self.bounds)
        return center

    def _has_viewport(self) -> bool:
        return self.isVisible() and self.width() > 0 and self.height() > 0

    def _sync_viewport(self) -> None:
        cam = self.camera
        cam.viewport[:] = (max(self.width(), 1), max(self.height(), 1))
        if self.fit_mode and self.bounds is not None:
            self._view_token = None
            cam.center, cam.zoom = cam.fit_params(*self.bounds, margin=self.fit_margin)
        else:
            cam.center = self._clamped(cam.center, cam.zoom)
        self.view_changed.emit()
        self.update()

    def resizeEvent(self, event) -> None:
        # Before the base class: it recreates the framebuffer and paints right
        # away, which must already use the new viewport. (A single resize, like
        # snapping the window, otherwise left a stretched frame on screen.)
        self._sync_viewport()
        super().resizeEvent(event)

    def showEvent(self, event) -> None:
        self._sync_viewport()
        super().showEvent(event)

    # OpenGL

    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context()
        self.renderer = SpriteRenderer(self.ctx)

    def resizeGL(self, w: int, h: int) -> None:
        self._fbo = None  # Qt recreates its framebuffer on resize

    def paintGL(self) -> None:
        # Render only. Animations advance in _on_frame_swapped, outside Qt's
        # paint pass, because they may show/hide or resize other widgets.
        if self._fbo is None:
            self._fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
        self._fbo.use()
        ratio = self.devicePixelRatioF()
        self.ctx.viewport = (0, 0, round(self.width() * ratio), round(self.height() * ratio))
        self.ctx.clear(*self.background, 1.0)
        if self.checkerboard:
            if self._checker is None:
                program = self.ctx.program(
                    vertex_shader=_CHECKER_VERTEX, fragment_shader=_CHECKER_FRAGMENT
                )
                self._checker = (program, self.ctx.vertex_array(program, []))
            program, vao = self._checker
            program["u_square"] = CHECKER_PX * ratio
            self.ctx.disable(moderngl.BLEND)
            vao.render(moderngl.TRIANGLES, vertices=3)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        for layer in self._released:
            self.renderer.release(layer)
        self._released.clear()

        nearest = self.pixelate_above is not None and self.device_zoom() >= self.pixelate_above
        self.renderer.set_camera(self.camera)
        for layer in self.layers:
            if layer.visible:
                self.renderer.render(layer, nearest)

    def _run_animations(self, now: float) -> None:
        running, self._animations = self._animations, []
        alive = []
        for animation, start in running:
            start = now if start is None else start
            if animation(now - start):
                alive.append((animation, start))
        # Keep animations added by the callbacks themselves.
        self._animations = alive + self._animations

    def _on_frame_swapped(self) -> None:
        now = time.perf_counter()
        self._count_frame(now)
        if self._animations:
            self._run_animations(now)
            self.update()  # draw the new state, including a just-finished final frame

    def _count_frame(self, now: float) -> None:
        if not self._animations:
            if self._fps_frames:
                self.fps_changed.emit(0.0)
            self._fps_frames, self._fps_start = 0, now
            return
        self._fps_frames += 1
        elapsed = now - self._fps_start
        if elapsed >= 1.0:
            self.fps_changed.emit(self._fps_frames / elapsed)
            self._fps_frames, self._fps_start = 0, now

    # Input

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.report_cursor(event.position())
        if self._drag_pos is not None:
            delta = event.position() - self._drag_pos
            self._drag_pos = event.position()
            cam = self.camera
            self.set_view(cam.center - np.array([delta.x(), delta.y()]) / cam.zoom, cam.zoom)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_pos = None
        self.unsetCursor()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self.double_clicked.emit(pos.x(), pos.y())

    def wheelEvent(self, event: QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120
        if steps:
            pos = event.position()
            self.zoom_at(pos.x(), pos.y(), WHEEL_ZOOM_STEP**steps)

    def leaveEvent(self, event) -> None:
        self.cursor_left.emit()

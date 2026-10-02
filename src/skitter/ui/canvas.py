"""GPU canvas: draws sprite layers with moderngl inside a Qt OpenGL widget."""

import time
from collections.abc import Callable

import moderngl
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from skitter.core.easing import ease_out_cubic
from skitter.ui.render.camera import Camera2D
from skitter.ui.render.sprites import SpriteLayer, SpriteRenderer, make_instances

# Called every frame with seconds elapsed since the animation started.
# Return False once finished to stop being called.
Animation = Callable[[float], bool]


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
    """

    fps_changed = Signal(float)  # measured each second while animating; 0.0 when idle

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = Camera2D()
        self.background = (0.12, 0.12, 0.12)
        self.layers: list[SpriteLayer] = []
        self.ctx: moderngl.Context | None = None
        self.renderer: SpriteRenderer | None = None

        self._animations: list[tuple[Animation, float | None]] = []
        self._released: list[SpriteLayer] = []
        self._pending_fit: tuple[float, float, float, float] | None = None
        self._fbo: moderngl.Framebuffer | None = None
        self._drag_pos: QPointF | None = None
        self._fps_frames = 0
        self._fps_start = time.perf_counter()

        self.frameSwapped.connect(self._on_frame_swapped)

    # Public API

    def add_layer(self, layer: SpriteLayer) -> SpriteLayer:
        self.layers.append(layer)
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

    def fit_to(self, x: float, y: float, w: float, h: float) -> None:
        """Frame the world rect; deferred until the widget has a size."""
        self._pending_fit = (x, y, w, h)
        self.update()

    def show_image(self, array: np.ndarray, fade_s: float = 0.4) -> SpriteLayer:
        """Replace all content with a single image, fading it in."""
        self.clear_animations()
        self.clear_layers()
        h, w = array.shape[:2]
        instances = make_instances(1)
        instances["pos"] = (w / 2, h / 2)
        instances["size"] = (w, h)
        instances["alpha"] = 0.0
        layer = self.add_layer(SpriteLayer(array, instances))
        self.fit_to(0, 0, w, h)

        def fade(t: float) -> bool:
            layer.instances["alpha"] = ease_out_cubic(min(t / fade_s, 1.0))
            layer.mark_dirty()
            return t < fade_s

        self.add_animation(fade)
        return layer

    # OpenGL

    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context()
        self.renderer = SpriteRenderer(self.ctx)

    def resizeGL(self, w: int, h: int) -> None:
        self.camera.viewport[:] = (self.width(), self.height())
        self._fbo = None  # Qt recreates its framebuffer on resize

    def paintGL(self) -> None:
        now = time.perf_counter()
        self._run_animations(now)
        if self._pending_fit is not None and self.width() > 0:
            self.camera.fit(*self._pending_fit)
            self._pending_fit = None

        if self._fbo is None:
            self._fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
        self._fbo.use()
        ratio = self.devicePixelRatioF()
        self.ctx.viewport = (0, 0, round(self.width() * ratio), round(self.height() * ratio))
        self.ctx.clear(*self.background, 1.0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        for layer in self._released:
            self.renderer.release(layer)
        self._released.clear()

        self.renderer.set_camera(self.camera)
        for layer in self.layers:
            if layer.visible:
                self.renderer.render(layer)

        self._count_frame(now)

    def _run_animations(self, now: float) -> None:
        alive = []
        for animation, start in self._animations:
            start = now if start is None else start
            if animation(now - start):
                alive.append((animation, start))
        self._animations = alive

    def _on_frame_swapped(self) -> None:
        if self._animations:
            self.update()

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

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos is not None:
            delta = event.position() - self._drag_pos
            self._drag_pos = event.position()
            self.camera.pan_pixels(delta.x(), delta.y())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_pos = None
        self.unsetCursor()

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 ** (event.angleDelta().y() / 120)
        pos = event.position()
        self.camera.zoom_at(pos.x(), pos.y(), factor)
        self.update()

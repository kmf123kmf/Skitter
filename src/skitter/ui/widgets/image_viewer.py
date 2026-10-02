"""Image viewer: GPU canvas with scrollbars, zoom controls, and animated edits.

By default world coordinates on the canvas equal image pixel coordinates:
the image occupies the world rect (0, 0, width, height). A world size can be
given instead, stretching the image over (0, 0, world_w, world_h); the
Slicing step uses this to show the source at mosaic scale.
"""

from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollBar,
    QToolButton,
    QWidget,
)

from skitter.core.easing import Easing, ease_in_out_cubic, ease_out_cubic, lerp
from skitter.ui import icons
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.render.sprites import SpriteLayer, make_instances

# Zoom steps (percent) used by zoom in/out, as in common image editors.
ZOOM_PRESETS = (1, 2, 3, 5, 8, 12.5, 100 / 6, 25, 100 / 3, 50, 200 / 3, 100, 150, 200,
                300, 400, 600, 800, 1200, 1600, 2400, 3200, 6400)  # fmt: skip
ZOOM_MENU = (10, 25, 100 / 3, 50, 200 / 3, 100, 150, 200, 300, 400, 800, 1600, 3200)
MIN_ZOOM_PERCENT = ZOOM_PRESETS[0]
MAX_ZOOM_PERCENT = ZOOM_PRESETS[-1]
TRANSITION_S = 0.35


def next_zoom_preset(percent: float, direction: int) -> float:
    """The next preset above (direction > 0) or below (direction < 0) percent."""
    if direction > 0:
        return next((p for p in ZOOM_PRESETS if p > percent * 1.001), ZOOM_PRESETS[-1])
    return next((p for p in reversed(ZOOM_PRESETS) if p < percent / 1.001), ZOOM_PRESETS[0])


def format_percent(percent: float) -> str:
    return f"{percent:.1f}%" if percent < 10 else f"{percent:.0f}%"


def parse_percent(text: str) -> float | None:
    try:
        value = float(text.strip().rstrip("%").strip())
    except ValueError:
        return None
    return value if value > 0 else None


class ZoomCombo(QComboBox):
    """Editable zoom box: pick a preset or "Fit", or type a percentage."""

    percent_requested = Signal(float)
    fit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setMinimumContentsLength(6)
        self.setToolTip("Zoom level")
        self.addItem("Fit")
        for percent in ZOOM_MENU:
            self.addItem(format_percent(percent))
        self._shown = 100.0
        self.textActivated.connect(self._apply_text)
        self.lineEdit().returnPressed.connect(lambda: self._apply_text(self.currentText()))

    def show_percent(self, percent: float) -> None:
        self._shown = percent
        editor = self.lineEdit()
        if not (editor.hasFocus() and editor.isModified()):
            self.setEditText(format_percent(percent))

    def _apply_text(self, text: str) -> None:
        self.lineEdit().setModified(False)
        if text.strip().lower() == "fit":
            self.fit_requested.emit()
            return
        percent = parse_percent(text)
        if percent is None:
            self.show_percent(self._shown)
        else:
            self.percent_requested.emit(percent)


class ImageViewer(QWidget):
    zoom_changed = Signal(float)  # percent; 100 = one image pixel per screen pixel

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image: np.ndarray | None = None
        self._layer: SpriteLayer | None = None
        self._transition: object | None = None
        self._finalize: Callable[[], None] | None = None
        self._dimming = 0.0
        self._world_size: tuple[float, float] | None = None  # None: image pixels
        self._content_bounds: tuple[float, float, float, float] | None = None

        self.canvas = MosaicCanvas()
        self.canvas.clamp_to_bounds = True
        self.canvas.fit_margin = 0.97
        self.canvas.pixelate_above = 4.0
        self.hbar = QScrollBar(Qt.Orientation.Horizontal)
        self.vbar = QScrollBar(Qt.Orientation.Vertical)
        for bar in (self.hbar, self.vbar):
            bar.hide()
            bar.valueChanged.connect(self._on_scroll)

        self._build_actions()
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        grid.addWidget(self.canvas, 0, 0)
        grid.addWidget(self.vbar, 0, 1)
        grid.addWidget(self.hbar, 1, 0)
        grid.addWidget(self._build_status_bar(), 2, 0, 1, 2)
        # All spare space goes to the canvas, never to the scrollbar row/column.
        grid.setRowStretch(0, 1)
        grid.setColumnStretch(0, 1)

        self.canvas.view_changed.connect(self._on_view_changed)
        self.canvas.cursor_moved.connect(self._show_cursor)
        self.canvas.cursor_left.connect(self._clear_cursor)
        self.canvas.double_clicked.connect(self._on_double_click)

    def _build_actions(self) -> None:
        def action(icon, text, slot, *shortcuts):
            act = QAction(icon, text, self)
            act.setShortcuts([QKeySequence(s) for s in shortcuts])
            keys = act.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
            act.setToolTip(f"{text} ({keys})")
            act.triggered.connect(slot)
            return act

        self.zoom_in_action = action(
            icons.zoom_in(), "Zoom In", lambda: self.step_zoom(1),
            QKeySequence.StandardKey.ZoomIn, "Ctrl+=",
        )  # fmt: skip
        self.zoom_out_action = action(
            icons.zoom_out(), "Zoom Out", lambda: self.step_zoom(-1),
            QKeySequence.StandardKey.ZoomOut,
        )  # fmt: skip
        self.fit_action = action(icons.zoom_fit(), "Fit to Window", self.fit, "Ctrl+0")
        self.fit_action.setCheckable(True)
        self.actual_size_action = action(
            icons.zoom_actual(), "Actual Size", lambda: self.set_zoom_percent(100), "Ctrl+1"
        )

    def _build_status_bar(self) -> QWidget:
        bar = QFrame()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 2, 4, 2)
        layout.setSpacing(4)

        self._swatch = QLabel()
        self._swatch.setFixedSize(12, 12)
        self._swatch.hide()
        self._cursor_label = QLabel()
        layout.addWidget(self._swatch)
        layout.addWidget(self._cursor_label)
        layout.addStretch()

        self.zoom_combo = ZoomCombo()
        self.zoom_combo.percent_requested.connect(self.set_zoom_percent)
        self.zoom_combo.fit_requested.connect(self.fit)
        for item in (
            self.zoom_out_action,
            self.zoom_combo,
            self.zoom_in_action,
            self.fit_action,
            self.actual_size_action,
        ):
            if isinstance(item, QAction):
                button = QToolButton()
                button.setDefaultAction(item)
                button.setAutoRaise(True)
                item = button
            layout.addWidget(item)
        return bar

    # Zoom

    def zoom_percent(self) -> float:
        return self.canvas.device_zoom() * 100

    def set_zoom_percent(
        self, percent: float, anchor: tuple[float, float] | None = None, animate: bool = True
    ) -> None:
        """Zoom to percent about a screen point (default: the view center)."""
        if self.image is None:
            return
        percent = float(np.clip(percent, MIN_ZOOM_PERCENT, MAX_ZOOM_PERCENT))
        cam = self.canvas.camera
        zoom = percent / 100 / self.canvas.devicePixelRatioF()
        center = cam.center
        if anchor is not None:
            center, zoom = cam.zoom_at_params(*anchor, zoom / cam.zoom)
        self.canvas.set_view(center, zoom, animate=animate)

    def step_zoom(self, direction: int) -> None:
        self.set_zoom_percent(next_zoom_preset(self.zoom_percent(), direction))

    def fit(self) -> None:
        self.canvas.zoom_to_fit(animate=True)
        self.fit_action.setChecked(True)

    def _on_double_click(self, x: float, y: float) -> None:
        if self.canvas.fit_mode:
            self.set_zoom_percent(100, anchor=(x, y))
        else:
            self.fit()

    def _on_view_changed(self) -> None:
        self._update_zoom_limits()
        self._sync_scrollbars()
        percent = self.zoom_percent()
        self.zoom_combo.show_percent(percent)
        self.fit_action.setChecked(self.canvas.fit_mode)
        self.zoom_changed.emit(percent)

    def _update_zoom_limits(self) -> None:
        if self.image is None or self.canvas.bounds is None:
            return
        cam, dpr = self.canvas.camera, self.canvas.devicePixelRatioF()
        _, _, w, h = self.canvas.bounds
        fit = self.canvas.fit_margin * min(cam.viewport[0] / w, cam.viewport[1] / h)
        cam.min_zoom = min(MIN_ZOOM_PERCENT / 100 / dpr, fit)
        cam.max_zoom = MAX_ZOOM_PERCENT / 100 / dpr

    # Scrollbars

    def _sync_scrollbars(self) -> None:
        bounds, cam = self.canvas.bounds, self.canvas.camera
        # Fit mode never needs scrollbars. Checking it (not just sizes) keeps them
        # from flashing mid-animation, which would resize the canvas and cut the
        # animation short.
        no_scroll = bounds is None or self.canvas.fit_mode
        for axis, bar in ((0, self.hbar), (1, self.vbar)):
            bar.blockSignals(True)
            content = 0.0 if no_scroll else bounds[axis + 2] * cam.zoom
            view = cam.viewport[axis]
            if content <= view + 0.5:
                bar.hide()
            else:
                bar.setRange(0, round(content - view))
                bar.setPageStep(round(view))
                bar.setSingleStep(max(1, round(view / 20)))
                bar.setValue(round((cam.center[axis] - bounds[axis]) * cam.zoom - view / 2))
                bar.show()
            bar.blockSignals(False)

    def _on_scroll(self) -> None:
        bounds, cam = self.canvas.bounds, self.canvas.camera
        center = cam.center.copy()
        for axis, bar in ((0, self.hbar), (1, self.vbar)):
            if not bar.isHidden():
                center[axis] = bounds[axis] + (bar.value() + cam.viewport[axis] / 2) / cam.zoom
        self.canvas.set_view(center, cam.zoom)

    # Cursor readout

    def _show_cursor(self, x: float, y: float) -> None:
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        world_w, world_h = self._world_dims(self.image)
        px, py = int(np.floor(x * w / world_w)), int(np.floor(y * h / world_h))
        if not (0 <= px < w and 0 <= py < h):
            self._clear_cursor()
            return
        r, g, b = (int(v) for v in self.image[py, px, :3])
        wx, wy = int(np.floor(x)), int(np.floor(y))
        self._cursor_label.setText(f"X {wx}   Y {wy}     RGB {r}, {g}, {b}")
        self._swatch.setStyleSheet(f"background: rgb({r},{g},{b}); border: 1px solid gray;")
        self._swatch.show()

    def _clear_cursor(self) -> None:
        self._cursor_label.clear()
        self._swatch.hide()

    # Showing images

    def show_image(self, image: np.ndarray, world_size: tuple[float, float] | None = None) -> None:
        """Display a new image: fit it to the view and fade it in.

        world_size stretches the image over (0, 0, *world_size) instead of
        its own pixel size.
        """
        self._finish_transition()
        self.image = image
        self._world_size = world_size
        self._content_bounds = None
        layer = self._make_layer(image, alpha=0.0)
        self._set_layer(layer)
        self._set_bounds(image)
        self.canvas.zoom_to_fit()

        def fade(e: float) -> None:
            layer.instances["alpha"] = e
            layer.mark_dirty()

        self._run_transition(fade, lambda: fade(1.0), ease_out_cubic)

    def replace_image(self, image: np.ndarray) -> None:
        """Swap in a different version of the image and refit."""
        self._finish_transition()
        self.image = image
        self._set_layer(self._make_layer(image))
        self._set_bounds(image)
        self.canvas.zoom_to_fit(animate=True, duration=TRANSITION_S)

    def flip(self, image: np.ndarray, horizontal: bool) -> None:
        """Show the flipped image, animating a mirror flip of the current one."""
        self._finish_transition()
        old = self._layer
        self.image = image
        axis = 0 if horizontal else 1
        start = float(old.instances["size"][0, axis])

        def step(e: float) -> None:
            old.instances["size"][0, axis] = start * (1 - 2 * e)
            old.mark_dirty()

        self._run_transition(step, lambda: self._set_layer(self._make_layer(image)))

    def rotate(self, image: np.ndarray, turns: int) -> None:
        """Show the rotated image (turns clockwise), animating the rotation."""
        self._finish_transition()
        old = self._layer
        self.image = image
        start_pos = old.instances["pos"][0].copy()
        h, w = image.shape[:2]
        end_pos = np.array([w / 2, h / 2])
        angle = np.pi / 2 * turns
        self._set_bounds(image)
        self.canvas.zoom_to_fit(animate=True, duration=TRANSITION_S, easing=ease_in_out_cubic)

        def step(e: float) -> None:
            old.instances["pos"][0] = lerp(start_pos, end_pos, e)
            old.instances["rotation"][0] = angle * e
            old.mark_dirty()

        self._run_transition(step, lambda: self._set_layer(self._make_layer(image)))

    def crop(self, image: np.ndarray, offset: tuple[int, int]) -> None:
        """Show a crop of the current image at offset, zooming in to fit it."""
        self._shift_to(image, -np.asarray(offset, dtype=float))

    def uncrop(self, image: np.ndarray, offset: tuple[int, int]) -> None:
        """Show the image the current one was cropped from at offset."""
        self._shift_to(image, np.asarray(offset, dtype=float))

    def _shift_to(self, image: np.ndarray, shift: np.ndarray) -> None:
        # Move the camera with the coordinate change so the view doesn't jump,
        # then animate to fit the new image.
        self._finish_transition()
        self.image = image
        self.canvas.camera.center = self.canvas.camera.center + shift
        self._set_layer(self._make_layer(image))
        self._set_bounds(image)
        self.canvas.zoom_to_fit(animate=True, duration=TRANSITION_S)

    def set_world_size(self, width: float, height: float) -> None:
        """Re-stretch the current image over (0, 0, width, height) and refit."""
        self._finish_transition()
        self._world_size = (width, height)
        if self._layer is not None:
            self._layer.instances["pos"] = (width / 2, height / 2)
            self._layer.instances["size"] = (width, height)
            self._layer.mark_dirty()
        self._set_bounds(self.image)
        self.canvas.zoom_to_fit(animate=True)

    def set_content_bounds(self, bounds: tuple[float, float, float, float] | None) -> None:
        """World rect (x, y, w, h) to fit and scroll over, if larger than the image.

        For example, tiles overhanging the image edges. Refits if in fit mode.
        """
        self._content_bounds = bounds
        self._set_bounds(self.image)
        if self.canvas.fit_mode:
            self.canvas.zoom_to_fit(animate=True)

    @property
    def world_size(self) -> tuple[float, float] | None:
        """The world size the image is stretched over, or None for its pixel size."""
        return self._world_size

    def _world_dims(self, image: np.ndarray) -> tuple[float, float]:
        if self._world_size is not None:
            return self._world_size
        h, w = image.shape[:2]
        return float(w), float(h)

    @property
    def image_layer(self) -> SpriteLayer | None:
        """The canvas layer currently showing the image (replaced by some edits)."""
        return self._layer

    def set_dimming(self, amount: float) -> None:
        """Darken the image by amount (0..1), e.g. under a partial overlay."""
        self._dimming = amount
        if self._layer is not None:
            self._layer.instances["tint"][:, 3] = amount
            self._layer.mark_dirty()
            self.canvas.update()

    def _make_layer(self, image: np.ndarray, alpha: float = 1.0) -> SpriteLayer:
        w, h = self._world_dims(image)
        instances = make_instances(1)
        instances["pos"] = (w / 2, h / 2)
        instances["size"] = (w, h)
        instances["alpha"] = alpha
        instances["tint"][:, 3] = self._dimming  # tint color is black
        return SpriteLayer(image, instances)

    def _set_layer(self, layer: SpriteLayer) -> None:
        # Replace the image layer in place so overlay layers stay above it.
        index = 0
        if self._layer is not None:
            index = self.canvas.layers.index(self._layer)
            self.canvas.remove_layer(self._layer)
        self._layer = self.canvas.add_layer(layer, index)

    def _set_bounds(self, image: np.ndarray) -> None:
        w, h = self._world_dims(image)
        bounds = (0.0, 0.0, w, h)
        if self._content_bounds is not None:
            x0 = min(0.0, self._content_bounds[0])
            y0 = min(0.0, self._content_bounds[1])
            x1 = max(w, self._content_bounds[0] + self._content_bounds[2])
            y1 = max(h, self._content_bounds[1] + self._content_bounds[3])
            bounds = (x0, y0, x1 - x0, y1 - y0)
        self.canvas.bounds = bounds
        self._update_zoom_limits()

    def _run_transition(
        self,
        step: Callable[[float], None],
        finalize: Callable[[], None],
        easing: Easing = ease_in_out_cubic,
    ) -> None:
        """Animate step(eased progress) over TRANSITION_S, then call finalize.

        Starting another transition first finalizes the running one, so rapid
        edits never leave a half-finished animation on screen.
        """
        token = self._transition = object()
        self._finalize = finalize

        def animate(t: float) -> bool:
            if self._transition is not token:
                return False
            step(float(easing(min(t / TRANSITION_S, 1.0))))
            if t >= TRANSITION_S:
                self._finish_transition()
                return False
            return True

        self.canvas.add_animation(animate)

    def _finish_transition(self) -> None:
        finalize, self._finalize = self._finalize, None
        self._transition = None
        if finalize is not None:
            finalize()

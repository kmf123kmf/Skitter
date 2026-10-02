"""Interactive crop box drawn over a canvas.

The overlay is a transparent child widget of the canvas. It handles left
button drags itself and ignores other input, so the wheel and middle-button
drags fall through to the canvas for zooming and panning while cropping.
"""

import math

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from skitter.core.geometry import (
    CORNERS,
    Rect,
    fit_aspect,
    move_rect,
    resize_rect,
    round_rect,
)
from skitter.ui.canvas import MosaicCanvas

HANDLE_PX = 8  # drawn handle size
HANDLE_GRAB_PX = 10  # corner grab radius
EDGE_GRAB_PX = 6  # edge grab distance
NEW_BOX_DRAG_PX = 4  # movement before a click outside starts a new box

_CURSORS = {
    "tl": Qt.CursorShape.SizeFDiagCursor,
    "br": Qt.CursorShape.SizeFDiagCursor,
    "tr": Qt.CursorShape.SizeBDiagCursor,
    "bl": Qt.CursorShape.SizeBDiagCursor,
    "t": Qt.CursorShape.SizeVerCursor,
    "b": Qt.CursorShape.SizeVerCursor,
    "l": Qt.CursorShape.SizeHorCursor,
    "r": Qt.CursorShape.SizeHorCursor,
    "move": Qt.CursorShape.SizeAllCursor,
    None: Qt.CursorShape.CrossCursor,
}


class CropOverlay(QWidget):
    box_changed = Signal()

    def __init__(self, canvas: MosaicCanvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.bounds: Rect = (0, 0, 1, 1)
        self.box: Rect = self.bounds  # crop box (left, top, right, bottom), whole pixels
        self.aspect: float | None = None
        self._drag: dict | None = None

        self.setMouseTracking(True)
        self.hide()
        canvas.installEventFilter(self)
        canvas.view_changed.connect(self.update)

    # Public API

    def start(self, width: int, height: int) -> None:
        """Show the overlay with the crop box covering the whole image."""
        self.bounds = (0, 0, width, height)
        self.aspect = None
        self._drag = None
        self.set_box(self.bounds)
        self.setGeometry(self.canvas.rect())
        self.show()
        self.raise_()

    def stop(self) -> None:
        self._drag = None
        self.hide()

    def set_box(self, rect: Rect) -> None:
        self.box = round_rect(rect, self.bounds)
        self.box_changed.emit()
        self.update()

    def set_aspect(self, aspect: float | None) -> None:
        """Lock to width / height (None for free), refitting the current box."""
        self.aspect = aspect
        if aspect:
            self.set_box(fit_aspect(self.box, aspect))

    def crop_box(self) -> tuple[int, int, int, int]:
        """(left, top, width, height) in image pixels."""
        left, top, right, bottom = (int(v) for v in self.box)
        return (left, top, right - left, bottom - top)

    # Coordinates

    def _to_image(self, pos: QPointF) -> np.ndarray:
        return self.canvas.camera.screen_to_world(pos.x(), pos.y())

    def _clamp_to_bounds(self, point: np.ndarray) -> np.ndarray:
        return np.clip(point, self.bounds[:2], self.bounds[2:])

    def _screen_rect(self) -> QRectF:
        cam = self.canvas.camera
        left, top, right, bottom = self.box
        return QRectF(
            QPointF(*cam.world_to_screen(left, top)), QPointF(*cam.world_to_screen(right, bottom))
        )

    @staticmethod
    def _handle_points(sr: QRectF) -> dict[str, QPointF]:
        cx, cy = sr.center().x(), sr.center().y()
        return {
            "tl": sr.topLeft(),
            "t": QPointF(cx, sr.top()),
            "tr": sr.topRight(),
            "r": QPointF(sr.right(), cy),
            "br": sr.bottomRight(),
            "b": QPointF(cx, sr.bottom()),
            "bl": sr.bottomLeft(),
            "l": QPointF(sr.left(), cy),
        }

    def _handle_image_point(self, handle: str) -> np.ndarray:
        left, top, right, bottom = self.box
        x = left if "l" in handle else right if "r" in handle else (left + right) / 2
        y = top if "t" in handle else bottom if "b" in handle else (top + bottom) / 2
        return np.array([x, y], dtype=float)

    def _hit(self, pos: QPointF) -> str | None:
        """Handle name, "move" for inside the box, or None for outside."""
        sr = self._screen_rect()
        points = self._handle_points(sr)
        for name in CORNERS:
            d = points[name] - pos
            if math.hypot(d.x(), d.y()) <= HANDLE_GRAB_PX:
                return name
        x, y = pos.x(), pos.y()
        within_x = sr.left() <= x <= sr.right()
        within_y = sr.top() <= y <= sr.bottom()
        if within_x and abs(y - sr.top()) <= EDGE_GRAB_PX:
            return "t"
        if within_x and abs(y - sr.bottom()) <= EDGE_GRAB_PX:
            return "b"
        if within_y and abs(x - sr.left()) <= EDGE_GRAB_PX:
            return "l"
        if within_y and abs(x - sr.right()) <= EDGE_GRAB_PX:
            return "r"
        return "move" if sr.contains(pos) else None

    # Events

    def eventFilter(self, obj, event) -> bool:
        if obj is self.canvas and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.canvas.rect())
        return False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()  # let the canvas pan
            return
        pos = event.position()
        hit = self._hit(pos)
        point = self._to_image(pos)
        self._drag = {"mode": hit, "rect": self.box, "press": point, "press_screen": pos}
        if hit is None:
            self._drag["anchor"] = self._clamp_to_bounds(point)
        elif hit != "move":
            self._drag["offset"] = self._handle_image_point(hit) - point

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        self.canvas.report_cursor(pos)
        drag = self._drag
        if drag is None:
            self.setCursor(_CURSORS[self._hit(pos)])
            return

        point = self._to_image(pos)
        mode = drag["mode"]
        if mode is None:
            moved = pos - drag["press_screen"]
            if math.hypot(moved.x(), moved.y()) < NEW_BOX_DRAG_PX:
                return
            ax, ay = drag["anchor"]
            px, py = self._clamp_to_bounds(point)
            handle = ("t" if py < ay else "b") + ("l" if px < ax else "r")
            rect = resize_rect((ax, ay, ax, ay), handle, (px, py), self.bounds, self.aspect)
        elif mode == "move":
            dx, dy = point - drag["press"]
            rect = move_rect(drag["rect"], dx, dy, self.bounds)
        else:
            target = point + drag["offset"]
            rect = resize_rect(drag["rect"], mode, tuple(target), self.bounds, self.aspect)
        self.set_box(rect)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = None

    def leaveEvent(self, event) -> None:
        self.canvas.cursor_left.emit()

    # Painting

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        sr = self._screen_rect()

        shade = QPainterPath()
        shade.setFillRule(Qt.FillRule.OddEvenFill)
        shade.addRect(QRectF(self.rect()))
        shade.addRect(sr)
        p.fillPath(shade, QColor(0, 0, 0, 150))

        p.setPen(QPen(QColor(255, 255, 255, 110), 1))
        for i in (1, 2):
            x = sr.left() + sr.width() * i / 3
            y = sr.top() + sr.height() * i / 3
            p.drawLine(QPointF(x, sr.top()), QPointF(x, sr.bottom()))
            p.drawLine(QPointF(sr.left(), y), QPointF(sr.right(), y))

        p.setPen(QPen(QColor(0, 0, 0, 160), 3))
        p.drawRect(sr)
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawRect(sr)

        p.setPen(QPen(QColor(0, 0, 0, 180), 1))
        p.setBrush(QColor(255, 255, 255))
        half = HANDLE_PX / 2
        for point in self._handle_points(sr).values():
            p.drawRect(QRectF(point.x() - half, point.y() - half, HANDLE_PX, HANDLE_PX))

        _, _, w, h = self.crop_box()
        text = f"{w} × {h}"
        metrics = p.fontMetrics()
        label = QRectF(0, 0, metrics.horizontalAdvance(text) + 14, metrics.height() + 6)
        label.moveCenter(QPointF(sr.center().x(), sr.bottom() + 10 + label.height() / 2))
        if label.bottom() > self.height() - 4:
            label.moveBottom(sr.bottom() - 10)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 180))
        p.drawRoundedRect(label, 4, 4)
        p.setPen(QColor(255, 255, 255))
        p.drawText(label, Qt.AlignmentFlag.AlignCenter, text)

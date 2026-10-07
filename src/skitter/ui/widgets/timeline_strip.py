"""A timeline strip: a time ruler, the video's holds, the playhead and keys.

It replaces a plain slider under a player. It has two lanes: the ruler and
track on top, where clicking or dragging scrubs (`scrubbed`) even right above
a key, and the keys' row below. Keys show there as marks (a diamond where the
camera stops, a circle where it passes through); a key can be clicked
(`key_clicked`), dragged to a new time (`key_moved`, on release) or
right-clicked (`key_menu`), and the cursor turns to a hand over one. Empty
ground in the keys' row scrubs too. Times map to x the same way for
everything, so keys and the playhead always line up.
"""

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

PAD = 10  # px at each end, so marks at the ends show whole
HIT = 7  # px around a key that pick it
KEY = 5.5  # key mark half size, px
TRACK_Y = 19  # px from the top: the middle of the track (the ruler is above it)
KEY_LANE = 11  # px from the bottom: the middle of the keys' row
HEIGHT = 46
TICKS = (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300)  # ruler steps, seconds
MIN_LABEL_GAP = 64  # px between labelled ticks


class TimelineStrip(QWidget):
    scrubbed = Signal(float)  # the user picked this time (click or drag on empty ground)
    key_clicked = Signal(int)
    key_moved = Signal(int, float)  # a key dragged to a new time (on release)
    key_menu = Signal(int, object)  # right click on a key: (index, global QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.duration = 0.0
        self.time = 0.0
        self.holds = (0.0, 0.0)  # video times where the animation starts and ends
        self.keys: list[tuple[float, bool, str]] = []  # (time, stop, motion), in time order
        self.selected = -1
        self._dragging_key = -1
        self._drag_time = 0.0
        self._scrubbing = False
        self.hovered = -1  # the key under the mouse
        self.setMinimumHeight(HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setToolTip("Click or drag to move through the video; drag a key (bottom row) "
                        "to retime it, right-click it for its options.")  # fmt: skip

    # State

    def set_timing(self, duration: float, holds=(0.0, 0.0)) -> None:
        self.duration, self.holds = max(float(duration), 0.0), holds
        self.update()

    def set_time(self, t: float) -> None:
        self.time = float(t)
        self.update()

    def set_keys(self, keys, selected: int = -1) -> None:
        self.keys, self.selected = list(keys), selected
        self.update()

    # Geometry

    def x_of(self, t: float) -> float:
        width = max(self.width() - 2 * PAD, 1)
        return PAD + (t / self.duration * width if self.duration > 0 else 0.0)

    def time_at(self, x: float) -> float:
        width = max(self.width() - 2 * PAD, 1)
        return min(max((x - PAD) / width * self.duration, 0.0), self.duration)

    def key_y(self) -> float:
        """The middle of the keys' row."""
        return self.height() - KEY_LANE

    def key_at(self, x: float, y: float) -> int:
        """The key under (x, y), the nearest if several (-1: none). Only the keys' row
        picks keys; the track above always scrubs."""
        if abs(y - self.key_y()) > HIT:
            return -1
        best, best_d = -1, HIT + 1
        for i, (t, _, _) in enumerate(self.keys):
            d = abs(self.x_of(t) - x)
            if d <= HIT and d < best_d:
                best, best_d = i, d
        return best

    # Input

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        i = self.key_at(pos.x(), pos.y())
        if event.button() == Qt.MouseButton.RightButton:
            if i >= 0:
                self.key_menu.emit(i, event.globalPosition().toPoint())
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if i >= 0:
            self._dragging_key, self._drag_time = i, self.keys[i][0]
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.key_clicked.emit(i)
        else:
            self._scrubbing = True
            self.scrubbed.emit(self.time_at(pos.x()))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x = event.position().x()
        if self._dragging_key >= 0:
            self._drag_time = self.time_at(x)
            self.update()
        elif self._scrubbing:
            self.scrubbed.emit(self.time_at(x))
        else:
            self._hover(self.key_at(x, event.position().y()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging_key >= 0:
            i, t = self._dragging_key, self._drag_time
            self._dragging_key = -1
            if abs(t - self.keys[i][0]) > 1e-6:
                self.key_moved.emit(i, t)
            self.update()
        self._scrubbing = False
        pos = event.position()
        self._hover(self.key_at(pos.x(), pos.y()), force=True)

    def leaveEvent(self, event) -> None:
        if self._dragging_key < 0:
            self._hover(-1)

    def _hover(self, i: int, force: bool = False) -> None:
        """Show which key the mouse is on: an open hand and a highlighted mark."""
        if i == self.hovered and not force:
            return
        self.hovered = i
        self.setCursor(Qt.CursorShape.OpenHandCursor if i >= 0 else Qt.CursorShape.ArrowCursor)
        self.update()

    # Drawing

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        text = pal.color(pal.ColorRole.WindowText)
        faint = QColor(text)
        faint.setAlphaF(0.35)
        accent = pal.color(pal.ColorRole.Highlight)
        h = self.height()
        mid = TRACK_Y
        key_y = self.key_y()

        # The track, and the holds shaded either side of the animation.
        x0, x1 = self.x_of(0.0), self.x_of(self.duration)
        track = QRectF(x0, mid - 3, x1 - x0, 6)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(pal.color(pal.ColorRole.Mid))
        p.drawRoundedRect(track, 3, 3)
        if self.duration > 0:
            shade = QColor(text)
            shade.setAlphaF(0.12)
            p.setBrush(shade)
            for a, b in ((0.0, self.holds[0]), (self.holds[1], self.duration)):
                if b > a:
                    p.drawRect(QRectF(self.x_of(a), mid - 3, self.x_of(b) - self.x_of(a), 6))
            p.setBrush(accent)
            played = QRectF(x0, mid - 3, self.x_of(self.time) - x0, 6)
            p.drawRoundedRect(played, 3, 3)

        self._draw_ruler(p, faint, text)

        # Lines between keys: dashed where the camera holds and then cuts.
        positions = [self._key_time(i) for i in range(len(self.keys))]
        for i in range(len(self.keys) - 1):
            pen = QPen(faint, 1.2)
            if self.keys[i][2] == "hold":
                pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            x0, x1 = self.x_of(positions[i]), self.x_of(positions[i + 1])
            p.drawLine(QPointF(x0, key_y), QPointF(x1, key_y))
        base = pal.color(pal.ColorRole.Base)
        for i, (_, stop, _) in enumerate(self.keys):
            fill = accent if i == self.selected else base
            grabbed = i in (self.hovered, self._dragging_key)
            line = accent if grabbed and i != self.selected else text
            size = KEY * 1.25 if grabbed else KEY
            self._draw_key(p, self.x_of(positions[i]), key_y, stop, fill, line, size)

        # The playhead.
        x = self.x_of(self.time)
        p.setPen(QPen(accent, 2))
        p.drawLine(QPointF(x, 2), QPointF(x, h - 2))
        p.end()

    def _key_time(self, i: int) -> float:
        return self._drag_time if i == self._dragging_key else self.keys[i][0]

    def _draw_ruler(self, p: QPainter, faint: QColor, text: QColor) -> None:
        if self.duration <= 0:
            return
        per_second = (self.width() - 2 * PAD) / self.duration
        step = next((s for s in TICKS if s * per_second >= MIN_LABEL_GAP), TICKS[-1])
        minor = step / 5
        font = p.font()
        font.setPointSizeF(max(font.pointSizeF() * 0.8, 6.5))
        p.setFont(font)
        count = int(math.floor(self.duration / minor + 1e-9))
        for n in range(count + 1):
            t = n * minor
            x = self.x_of(t)
            major = abs(t / step - round(t / step)) < 1e-6
            p.setPen(QPen(text if major else faint, 1))
            p.drawLine(QPointF(x, 1), QPointF(x, 6 if major else 4))
            if major:
                label = f"{t:g}s"
                p.drawText(QRectF(x + 2, -1, 60, 12), Qt.AlignmentFlag.AlignLeft, label)

    @staticmethod
    def _draw_key(p: QPainter, x: float, y: float, stop: bool, fill: QColor, line: QColor,
                  size: float = KEY):  # fmt: skip
        p.setPen(QPen(line, 1.6 if size > KEY else 1.2))
        p.setBrush(fill)
        if stop:
            p.drawPolygon(QPolygonF([QPointF(x, y - size), QPointF(x + size, y),
                                     QPointF(x, y + size), QPointF(x - size, y)]))  # fmt: skip
        else:
            path = QPainterPath()
            path.addEllipse(QPointF(x, y), size * 0.8, size * 0.8)
            p.drawPath(path)

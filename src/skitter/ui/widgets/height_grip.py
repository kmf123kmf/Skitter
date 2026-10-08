"""A drag bar under a widget that sets the widget's height (and remembers it)."""

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent, QPainter
from PySide6.QtWidgets import QWidget

from skitter.ui import preferences

GRIP = 8  # px, the bar's height


class HeightGrip(QWidget):
    """Put it right under target: dragging it makes target taller or shorter (between
    minimum and maximum px). key: where the height is remembered (preferences)."""

    def __init__(self, target: QWidget, key: str, default: int, minimum: int = 80,
                 maximum: int = 2000, parent=None):  # fmt: skip
        super().__init__(parent)
        self.target, self.key = target, key
        self.minimum, self.maximum = minimum, maximum
        self._drag: tuple[float, int] | None = None  # (press y, height then)
        self.setFixedHeight(GRIP)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setToolTip("Drag to make the list taller or shorter.")
        try:
            height = int(preferences.settings().value(key, default))
        except (TypeError, ValueError):
            height = default
        self.set_height(height)

    def set_height(self, height: float) -> None:
        self.target.setFixedHeight(int(min(max(height, self.minimum), self.maximum)))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = (event.globalPosition().y(), self.target.height())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is not None:
            y, height = self._drag
            self.set_height(height + event.globalPosition().y() - y)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag is not None:
            self._drag = None
            preferences.settings().setValue(self.key, self.target.height())

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.palette().color(self.palette().ColorRole.Mid)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        middle = self.width() / 2
        for dx in (-8, -4, 0, 4, 8):  # a row of dots: the usual "grab here"
            p.drawEllipse(QPointF(middle + dx, self.height() / 2), 1.3, 1.3)
        p.end()

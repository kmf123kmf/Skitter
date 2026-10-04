"""A grid of tile candidates to pick from, with badges and keyboard control.

Each cell shows one candidate crop fitted to the region's shape, its rank,
and badges: the tile shown now (accent frame), the matcher's own choice
(star), and a warning when picking it would break a reuse rule. Hovering a
cell, or moving to it with the arrow keys, asks for a preview (`hovered`);
a click, Enter or Space picks it (`activated`); a double-click picks it
and finishes (`finished`). Escape is passed on (`escaped`).
"""

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

COLUMNS = 4
SPACING = 6
ACCENT = QColor(59, 156, 255)
WARNING = QColor(255, 176, 32)
STAR = QColor(255, 214, 10)


@dataclass
class CandidateItem:
    image: QImage | None  # the crop as it would show (tinted, mirrored)
    rank: str  # shown in the corner: 1 = best match
    current: bool = False  # shown in the mosaic now
    auto: bool = False  # the matcher's choice
    warning: str = ""  # why picking it breaks a reuse rule ("" = it doesn't)
    tooltip: str = ""


class CandidateGrid(QWidget):
    hovered = Signal(int)  # index to preview, -1: none
    activated = Signal(int)
    finished = Signal(int)  # double-clicked: pick it and be done
    escaped = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: list[CandidateItem] = []
        self.aspect = 1.0  # width / height of the cells' crops
        self._hover = -1  # under the mouse
        self._focus = -1  # chosen with the keyboard
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    # Content

    def set_items(self, items: list[CandidateItem], aspect: float) -> None:
        self.items = items
        self.aspect = max(min(float(aspect), 4.0), 0.25)
        self._hover = self._focus = -1
        self._fit_height()
        self.update()

    def set_image(self, index: int, image: QImage) -> None:
        if 0 <= index < len(self.items):
            self.items[index].image = image
            self.update(self._cell(index).toAlignedRect())

    def clear_preview(self) -> None:
        """Stop previewing (until the mouse moves to a cell or an arrow key is pressed)."""
        self._hover = self._focus = -1
        self.update()

    @property
    def preview_index(self) -> int:
        """The candidate being previewed (mouse first, then keyboard), or -1."""
        return self._hover if self._hover >= 0 else self._focus

    # Geometry

    def _cell_size(self) -> tuple[float, float]:
        w = (self.width() - SPACING * (COLUMNS - 1)) / COLUMNS
        return w, w  # square cells; crops are fitted inside

    def _cell(self, index: int) -> QRectF:
        w, h = self._cell_size()
        row, col = divmod(index, COLUMNS)
        return QRectF(col * (w + SPACING), row * (h + SPACING), w, h)

    def _rows(self) -> int:
        return -(-len(self.items) // COLUMNS)

    def _fit_height(self) -> None:
        _, h = self._cell_size()
        rows = self._rows()
        self.setFixedHeight(max(int(rows * h + max(rows - 1, 0) * SPACING) + 1, 1))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_height()

    def sizeHint(self) -> QSize:
        return QSize(280, self.height())

    def index_at(self, pos: QPointF) -> int:
        for i in range(len(self.items)):
            if self._cell(i).contains(pos):
                return i
        return -1

    # Painting

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        palette = self.palette()
        small = QFont(self.font())
        small.setPointSizeF(max(self.font().pointSizeF() * 0.8, 6.5))
        small.setBold(True)
        p.setFont(small)
        preview = self.preview_index
        for i, item in enumerate(self.items):
            cell = self._cell(i)
            if not cell.intersects(QRectF(event.rect())):
                continue
            p.fillRect(cell, palette.base().color().darker(115))
            target = self._fit(cell, item)
            if item.image is not None:
                p.drawImage(target, item.image)
            if i == preview:
                p.fillRect(cell, QColor(255, 255, 255, 28))
            self._badges(p, cell, item)
            p.setBrush(Qt.BrushStyle.NoBrush)
            if item.current:
                p.setPen(QPen(ACCENT, 3))
                p.drawRect(cell.adjusted(1.5, 1.5, -1.5, -1.5))
            elif i == preview:
                p.setPen(QPen(palette.highlightedText().color(), 2))
                p.drawRect(cell.adjusted(1, 1, -1, -1))
            if i == self._focus and self.hasFocus():
                p.setPen(QPen(palette.highlight().color(), 1, Qt.PenStyle.DashLine))
                p.drawRect(cell.adjusted(4, 4, -4, -4))

    def _fit(self, cell: QRectF, item: CandidateItem) -> QRectF:
        """The crop's rectangle in a cell: the region's shape, as large as fits."""
        aspect = self.aspect
        w, h = cell.width() - 4, cell.height() - 4
        if w / h > aspect:
            w = h * aspect
        else:
            h = w / aspect
        return QRectF(cell.center().x() - w / 2, cell.center().y() - h / 2, w, h)

    def _badges(self, p: QPainter, cell: QRectF, item: CandidateItem) -> None:
        metrics = p.fontMetrics()
        text = item.rank
        pill = QRectF(cell.left() + 3, cell.top() + 3, metrics.horizontalAdvance(text) + 8,
                      metrics.height() + 1)  # fmt: skip
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 170))
        p.drawRoundedRect(pill, 4, 4)
        p.setPen(QColor(255, 255, 255))
        p.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)
        size = metrics.height() + 1
        if item.warning:
            badge = QRectF(cell.right() - size - 3, cell.top() + 3, size, size)
            path = QPainterPath()
            path.moveTo(badge.center().x(), badge.top())
            path.lineTo(badge.right(), badge.bottom())
            path.lineTo(badge.left(), badge.bottom())
            path.closeSubpath()
            p.setPen(QPen(QColor(0, 0, 0, 200), 1))
            p.setBrush(WARNING)
            p.drawPath(path)
            p.setPen(QColor(0, 0, 0))
            p.drawText(badge.adjusted(0, 2, 0, 0), Qt.AlignmentFlag.AlignCenter, "!")
        if item.auto:
            badge = QRectF(cell.left() + 3, cell.bottom() - size - 3, size, size)
            p.setPen(QPen(QColor(0, 0, 0, 200), 1))
            p.setBrush(QColor(0, 0, 0, 170))
            p.drawRoundedRect(badge, 4, 4)
            p.setPen(STAR)
            p.drawText(badge, Qt.AlignmentFlag.AlignCenter, "★")

    # Input

    def mouseMoveEvent(self, event) -> None:
        self._set_hover(self.index_at(event.position()))

    def leaveEvent(self, event) -> None:
        self._set_hover(-1)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.index_at(event.position())
            if index >= 0:
                self._focus = index
                self.activated.emit(index)
                self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.index_at(event.position())
            if index >= 0:
                self.finished.emit(index)

    def event(self, event) -> bool:
        if event.type() == event.Type.ToolTip:
            index = self.index_at(QPointF(event.pos()))
            if index >= 0 and self.items[index].tooltip:
                QToolTip.showText(event.globalPos(), self.items[index].tooltip, self,
                                  self._cell(index).toAlignedRect())  # fmt: skip
            else:
                QToolTip.hideText()
                event.ignore()
            return True
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        n = len(self.items)
        moves = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1, Qt.Key.Key_Up: -COLUMNS,
                 Qt.Key.Key_Down: COLUMNS}  # fmt: skip
        if key in moves and n:
            start = self._focus if self._focus >= 0 else self._current_index()
            index = min(max(start + moves[key], 0), n - 1) if start >= 0 else 0
            self._focus = index
            if self._hover < 0:
                self.hovered.emit(index)
            self.update()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space) and self._focus >= 0:
            self.activated.emit(self._focus)
        elif key == Qt.Key.Key_Escape:
            if self._focus >= 0 and self._hover < 0:
                self._focus = -1
                self.hovered.emit(-1)
                self.update()
            else:
                self.escaped.emit()
        else:
            super().keyPressEvent(event)

    def _current_index(self) -> int:
        return next((i for i, item in enumerate(self.items) if item.current), -1)

    def _set_hover(self, index: int) -> None:
        if index != self._hover:
            self._hover = index
            self.hovered.emit(self.preview_index)
            self.update()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self.update()

"""Toolbar icons drawn with QPainter.

Each icon is drawn as vectors on a 24x24 grid and rendered at several sizes
and screen scales, colored from the application palette (normal and
disabled), so the set looks consistent on any platform and theme.
"""

from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QIcon,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QPolygonF,
)

Draw = Callable[[QPainter, QColor], None]

_SIZES = (16, 20, 24, 32)
_SCALES = (1.0, 1.25, 1.5, 2.0)


def _make_icon(draw: Draw) -> QIcon:
    icon = QIcon()
    palette = QGuiApplication.palette()
    modes = (
        (QIcon.Mode.Normal, QPalette.ColorGroup.Active),
        (QIcon.Mode.Disabled, QPalette.ColorGroup.Disabled),
    )
    for mode, group in modes:
        color = palette.color(group, QPalette.ColorRole.WindowText)
        for size in _SIZES:
            for scale in _SCALES:
                px = round(size * scale)
                pixmap = QPixmap(px, px)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.scale(px / 24, px / 24)
                pen = QPen(color, 1.8)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                draw(painter, color)
                painter.end()
                pixmap.setDevicePixelRatio(scale)
                icon.addPixmap(pixmap, mode)
    return icon


def _mirrored(draw: Draw) -> Draw:
    def mirrored(p: QPainter, color: QColor) -> None:
        p.translate(24, 0)
        p.scale(-1, 1)
        draw(p, color)

    return mirrored


def _polyline(p: QPainter, *points: tuple[float, float]) -> None:
    p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in points]))


# Drawings


def _folder(p: QPainter, color: QColor) -> None:
    path = QPainterPath()
    path.moveTo(3, 18.5)
    for x, y in ((3, 5.5), (9, 5.5), (11, 7.5), (21, 7.5), (21, 18.5)):
        path.lineTo(x, y)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(3, 10.5), QPointF(21, 10.5))


def _undo(p: QPainter, color: QColor) -> None:
    path = QPainterPath()
    path.moveTo(5, 9)
    path.lineTo(14.5, 9)
    path.arcTo(QRectF(10, 9, 9, 9), 90, -180)
    path.lineTo(9, 18)
    p.drawPath(path)
    _polyline(p, (8.5, 5.5), (5, 9), (8.5, 12.5))


def _rotate_left(p: QPainter, color: QColor) -> None:
    rect = QRectF(5, 6, 14, 14)
    path = QPainterPath()
    path.arcMoveTo(rect, 90)
    path.arcTo(rect, 90, -270)
    p.drawPath(path)
    _polyline(p, (14.5, 3.5), (12, 6), (14.5, 8.5))


def _revert(p: QPainter, color: QColor) -> None:
    _rotate_left(p, color)
    _polyline(p, (12, 9.5), (12, 13), (14.5, 14.5))


def _flip_horizontal(p: QPainter, color: QColor) -> None:
    for y in (2.5, 7, 11.5, 16, 20.5):
        p.drawLine(QPointF(12, y), QPointF(12, y + 1.5))
    p.setBrush(color)
    p.drawPolygon(QPolygonF([QPointF(9.5, 6), QPointF(9.5, 18), QPointF(3, 18)]))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(QPolygonF([QPointF(14.5, 6), QPointF(14.5, 18), QPointF(21, 18)]))


def _flip_vertical(p: QPainter, color: QColor) -> None:
    p.translate(12, 12)
    p.rotate(90)
    p.translate(-12, -12)
    _flip_horizontal(p, color)


def _crop(p: QPainter, color: QColor) -> None:
    _polyline(p, (7, 2.5), (7, 17), (21.5, 17))
    _polyline(p, (2.5, 7), (17, 7), (17, 21.5))


def _magnifier(p: QPainter) -> None:
    p.drawEllipse(QPointF(10.5, 10.5), 6.5, 6.5)
    p.drawLine(QPointF(15.5, 15.5), QPointF(20.5, 20.5))


def _zoom_in(p: QPainter, color: QColor) -> None:
    _magnifier(p)
    p.drawLine(QPointF(7.5, 10.5), QPointF(13.5, 10.5))
    p.drawLine(QPointF(10.5, 7.5), QPointF(10.5, 13.5))


def _zoom_out(p: QPainter, color: QColor) -> None:
    _magnifier(p)
    p.drawLine(QPointF(7.5, 10.5), QPointF(13.5, 10.5))


def _zoom_fit(p: QPainter, color: QColor) -> None:
    _polyline(p, (3, 8), (3, 3), (8, 3))
    _polyline(p, (16, 3), (21, 3), (21, 8))
    _polyline(p, (21, 16), (21, 21), (16, 21))
    _polyline(p, (8, 21), (3, 21), (3, 16))
    p.drawRect(QRectF(8, 8, 8, 8))


def _zoom_actual(p: QPainter, color: QColor) -> None:
    font = QFont()
    font.setPixelSize(11)
    font.setBold(True)
    p.setFont(font)
    p.drawText(QRectF(0, 0, 24, 24), Qt.AlignmentFlag.AlignCenter, "1:1")


# Public icons


def open_folder() -> QIcon:
    return _make_icon(_folder)


def undo() -> QIcon:
    return _make_icon(_undo)


def redo() -> QIcon:
    return _make_icon(_mirrored(_undo))


def rotate_left() -> QIcon:
    return _make_icon(_rotate_left)


def rotate_right() -> QIcon:
    return _make_icon(_mirrored(_rotate_left))


def revert() -> QIcon:
    return _make_icon(_revert)


def flip_horizontal() -> QIcon:
    return _make_icon(_flip_horizontal)


def flip_vertical() -> QIcon:
    return _make_icon(_flip_vertical)


def crop() -> QIcon:
    return _make_icon(_crop)


def zoom_in() -> QIcon:
    return _make_icon(_zoom_in)


def zoom_out() -> QIcon:
    return _make_icon(_zoom_out)


def zoom_fit() -> QIcon:
    return _make_icon(_zoom_fit)


def zoom_actual() -> QIcon:
    return _make_icon(_zoom_actual)

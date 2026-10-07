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


def _filled(p: QPainter, color: QColor, *points: tuple[float, float]) -> None:
    p.setBrush(color)
    p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
    p.setBrush(Qt.BrushStyle.NoBrush)


def _play(p: QPainter, color: QColor) -> None:
    _filled(p, color, (7, 4.5), (19.5, 12), (7, 19.5))


def _pause(p: QPainter, color: QColor) -> None:
    p.setBrush(color)
    p.drawRoundedRect(QRectF(6.5, 5, 3.5, 14), 0.8, 0.8)
    p.drawRoundedRect(QRectF(14, 5, 3.5, 14), 0.8, 0.8)
    p.setBrush(Qt.BrushStyle.NoBrush)


def _to_start(p: QPainter, color: QColor) -> None:
    p.drawLine(QPointF(5.5, 5), QPointF(5.5, 19))
    _filled(p, color, (19, 5), (8.5, 12), (19, 19))


def _step_back(p: QPainter, color: QColor) -> None:
    p.drawLine(QPointF(7.5, 6.5), QPointF(7.5, 17.5))
    _filled(p, color, (17.5, 6.5), (10, 12), (17.5, 17.5))


def _loop(p: QPainter, color: QColor) -> None:
    p.drawArc(QRectF(4, 6, 16, 12), 30 * 16, 300 * 16)
    _filled(p, color, (16.5, 3.5), (21.5, 7.5), (16, 10))


def _diamond(p: QPainter, color: QColor, x: float, y: float, r: float) -> None:
    _filled(p, color, (x, y - r), (x + r, y), (x, y + r), (x - r, y))


def _add_key(p: QPainter, color: QColor) -> None:
    _diamond(p, color, 10, 13, 6.5)
    p.drawLine(QPointF(19, 3), QPointF(19, 9))
    p.drawLine(QPointF(16, 6), QPointF(22, 6))


def _delete_key(p: QPainter, color: QColor) -> None:
    _diamond(p, color, 10, 13, 6.5)
    p.drawLine(QPointF(16, 6), QPointF(22, 6))


def _previous_key(p: QPainter, color: QColor) -> None:
    _diamond(p, color, 15, 12, 6)
    _polyline(p, (8, 7), (3, 12), (8, 17))


def _viewfinder(p: QPainter, color: QColor) -> None:
    _polyline(p, (3, 8), (3, 4), (7, 4))
    _polyline(p, (17, 4), (21, 4), (21, 8))
    _polyline(p, (21, 16), (21, 20), (17, 20))
    _polyline(p, (7, 20), (3, 20), (3, 16))
    p.drawEllipse(QPointF(12, 12), 3.2, 3.2)


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


def play() -> QIcon:
    return _make_icon(_play)


def pause() -> QIcon:
    return _make_icon(_pause)


def to_start() -> QIcon:
    return _make_icon(_to_start)


def to_end() -> QIcon:
    return _make_icon(_mirrored(_to_start))


def step_back() -> QIcon:
    return _make_icon(_step_back)


def step_forward() -> QIcon:
    return _make_icon(_mirrored(_step_back))


def loop() -> QIcon:
    return _make_icon(_loop)


def add_key() -> QIcon:
    return _make_icon(_add_key)


def delete_key() -> QIcon:
    return _make_icon(_delete_key)


def previous_key() -> QIcon:
    return _make_icon(_previous_key)


def next_key() -> QIcon:
    return _make_icon(_mirrored(_previous_key))


def viewfinder() -> QIcon:
    return _make_icon(_viewfinder)

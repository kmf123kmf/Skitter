"""Mouse-wheel scrolling must never change a setting by accident.

Qt lets the wheel change any spin box, drop-down or slider under the
pointer, even one the user never clicked. Scrolling a settings panel then
silently edits whatever passes under the pointer (and, for matching
settings, marks the mosaic out of date). The guard, installed once for the
whole application, lets the wheel change such a control only while it has
keyboard focus (after a click or Tab); otherwise the wheel scrolls the
panel around it, as users expect.
"""

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QScrollBar,
    QWidget,
)

GUARDED = (QAbstractSpinBox, QComboBox, QAbstractSlider)


class WheelGuard(QObject):
    def eventFilter(self, obj, event) -> bool:
        if (
            event.type() == QEvent.Type.Wheel
            and isinstance(obj, GUARDED)
            and not isinstance(obj, QScrollBar)
            and not obj.hasFocus()
        ):
            area = _scroll_area(obj)
            if area is not None:
                QApplication.sendEvent(area.viewport(), event)  # scroll the panel instead
            return True
        if event.type() == QEvent.Type.Polish and isinstance(obj, GUARDED):
            # Take focus from clicks and Tab only, never from the wheel.
            if obj.focusPolicy() == Qt.FocusPolicy.WheelFocus:
                obj.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        return False


def _scroll_area(widget: QWidget) -> QAbstractScrollArea | None:
    parent = widget.parentWidget()
    while parent is not None and not isinstance(parent, QAbstractScrollArea):
        parent = parent.parentWidget()
    return parent


_guard: WheelGuard | None = None


def install_wheel_guard() -> None:
    """Install the guard for the whole application (once)."""
    global _guard
    app = QApplication.instance()
    if _guard is None and app is not None:
        _guard = WheelGuard(app)
        app.installEventFilter(_guard)

"""Actions and the flat tool buttons that show them, built alike everywhere.

An action's tooltip names it and its first shortcut ("Rotate Right (Ctrl+R)"),
so every button teaches its key.
"""

from collections.abc import Callable

from PySide6.QtCore import QObject, QSize, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import QToolButton

Key = str | QKeySequence | QKeySequence.StandardKey


def make_action(
    parent: QObject,
    text: str,
    slot: Callable | None = None,
    *keys: Key,
    icon: QIcon | None = None,
    tooltip: str | None = None,
    context: Qt.ShortcutContext | None = None,
    checkable: bool = False,
) -> QAction:
    """An action calling slot (with checkable: on `toggled`, given the new state).

    keys: its shortcuts, the first shown in the tooltip; context: where they work
    (default: anywhere in the window, while the parent widget shows).
    tooltip: what it does, if more than its text.
    """
    action = QAction(text, parent)
    if icon is not None:
        action.setIcon(icon)
    if keys:
        action.setShortcuts([QKeySequence(k) for k in keys])
    if context is not None:
        action.setShortcutContext(context)
    action.setCheckable(checkable)
    set_tooltip(action, tooltip or text)
    if slot is not None:
        (action.toggled if checkable else action.triggered).connect(slot)
    return action


def set_tooltip(action: QAction, text: str) -> None:
    """Set the tooltip to text and the action's shortcut: "Undo Crop (Ctrl+Z)"."""
    keys = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
    action.setToolTip(f"{text} ({keys})" if keys else text)


def tool_button(action: QAction, icon_size: int | None = None) -> QToolButton:
    """A flat button showing the action (its icon, else its text)."""
    button = QToolButton()
    button.setDefaultAction(action)
    button.setAutoRaise(True)
    if icon_size is not None:
        button.setIconSize(QSize(icon_size, icon_size))
    return button

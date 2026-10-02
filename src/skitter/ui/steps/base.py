"""Base class for workflow step pages."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QScrollArea, QStyle, QVBoxLayout, QWidget

from skitter.ui.session import Session

SIDE_PANEL_WIDTH = 320  # content width; the panel adds room for its scrollbar


class StepPage(QWidget):
    """One tab in the mosaic workflow.

    Subclasses set `title` and build their UI. The main window's Next button
    calls `advance()` to commit the step's work and then moves to the next
    tab; tabs after a step unlock only while it `is_complete()`. Emit
    `state_changed` whenever `is_complete()` or `can_advance()` may have
    changed.
    """

    title = ""

    state_changed = Signal()
    status_message = Signal(str)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session

    def is_complete(self) -> bool:
        """Whether this step's committed output is current, unlocking later steps."""
        return False

    def can_advance(self) -> bool:
        """Whether Next is enabled."""
        return self.is_complete()

    def advance(self) -> bool:
        """Commit this step's work when the user presses Next.

        Return True to move on to the next step.
        """
        return self.is_complete()

    def on_enter(self) -> None:
        """Called when this tab becomes the current tab."""

    def on_leave(self) -> None:
        """Called when another tab becomes current."""

    def shutdown(self) -> None:
        """Called when the window closes: stop any background work of the step's own."""


def side_panel(*widgets: QWidget, stretch_last: bool = False) -> QWidget:
    """Fixed-width settings panel shown to the right of a step's main view.

    Scrolls vertically when its contents are taller than the window.
    """
    content = QWidget()
    layout = QVBoxLayout(content)
    for i, widget in enumerate(widgets):
        last = i == len(widgets) - 1
        layout.addWidget(widget, stretch=1 if stretch_last and last else 0)
    if not stretch_last:
        layout.addStretch()

    scroll = QScrollArea()
    scroll.setWidget(content)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setFrameShape(QFrame.Shape.StyledPanel)
    # Leave room for the vertical scrollbar so content never gets clipped when it appears.
    style = scroll.style()
    scrollbar = style.pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
    scroll.setFixedWidth(SIDE_PANEL_WIDTH + scrollbar + 2 * scroll.frameWidth())
    return scroll

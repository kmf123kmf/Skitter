"""Base class for workflow step pages."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from skitter.ui.session import Session


class StepPage(QWidget):
    """One tab in the mosaic workflow.

    Subclasses set `title`, build their UI, and emit `completion_changed`
    whenever `is_complete()` may have changed.
    """

    title = ""

    completion_changed = Signal()
    status_message = Signal(str)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session

    def is_complete(self) -> bool:
        """Whether this step's output is ready, unlocking the next step."""
        return False

    def on_enter(self) -> None:
        """Called when this tab becomes the current tab."""

    def on_leave(self) -> None:
        """Called when another tab becomes current."""

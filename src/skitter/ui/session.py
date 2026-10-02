"""Observable wrapper around the Project shared by all step pages."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from skitter.core.project import Project


class Session(QObject):
    """Holds the current Project and announces changes to it.

    Step pages modify the project only through Session methods so that every
    page interested in a change gets notified.
    """

    source_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = Project()

    def set_source(self, path: Path, image: np.ndarray) -> None:
        self.project.source_path = Path(path)
        self.project.source_image = image
        self.source_changed.emit()

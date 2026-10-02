"""Observable wrapper around the Project shared by all step pages."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from skitter.core.edits import Edit, apply_edits
from skitter.core.project import Project


class Session(QObject):
    """Holds the current Project and announces changes to it.

    Step pages modify the project only through Session methods so that every
    page interested in a change gets notified.
    """

    source_changed = Signal()  # a new source image was loaded
    source_edited = Signal(object, bool)  # (edit, undone); edit is None after revert

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = Project()
        self._redo: list[Edit] = []

    def set_source(self, path: Path, image: np.ndarray) -> None:
        project = self.project
        project.source_path = Path(path)
        project.source_original = image
        project.source_edits = []
        project.source_image = image
        self._redo.clear()
        self.source_changed.emit()

    # Source edits

    @property
    def can_undo(self) -> bool:
        return bool(self.project.source_edits)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def next_redo(self) -> Edit | None:
        return self._redo[-1] if self._redo else None

    def apply_edit(self, edit: Edit) -> None:
        self.project.source_edits.append(edit)
        self._redo.clear()
        self._rebuild_source()
        self.source_edited.emit(edit, False)

    def undo(self) -> None:
        if not self.project.source_edits:
            return
        edit = self.project.source_edits.pop()
        self._redo.append(edit)
        self._rebuild_source()
        self.source_edited.emit(edit, True)

    def redo(self) -> None:
        if not self._redo:
            return
        edit = self._redo.pop()
        self.project.source_edits.append(edit)
        self._rebuild_source()
        self.source_edited.emit(edit, False)

    def revert_edits(self) -> None:
        """Drop all edits; they stay available to redo one at a time."""
        edits = self.project.source_edits
        if not edits:
            return
        self._redo.extend(reversed(edits))
        edits.clear()
        self._rebuild_source()
        self.source_edited.emit(None, True)

    def _rebuild_source(self) -> None:
        project = self.project
        project.source_image = apply_edits(project.source_original, project.source_edits)

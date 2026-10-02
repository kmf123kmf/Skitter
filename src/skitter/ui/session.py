"""Observable wrapper around the Project shared by all step pages."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from skitter.core.edits import Edit, apply_edits
from skitter.core.project import Project
from skitter.core.slicing import (
    MosaicLayout,
    SliceContext,
    SliceSummary,
    SlicingError,
    StageResult,
    summarize,
)


class Session(QObject):
    """Holds the current Project and announces changes to it.

    Step pages modify the project only through Session methods so that every
    page interested in a change gets notified.
    """

    source_changed = Signal()  # a new source image was loaded
    source_edited = Signal(object, bool)  # (edit, undone); edit is None after revert
    source_committed = Signal()  # project.source_final changed; later steps must refresh
    layout_changed = Signal()  # project.layout changed (tile size, aspect, columns)
    slicing_changed = Signal()  # project.regions recomputed (see slicing_error, slicing_summary)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = Project()
        self._redo: list[Edit] = []
        self._source_loads = 0  # identifies the loaded original
        self._committed_key: tuple | None = None
        self._slice_context: SliceContext | None = None
        self._slicing_cache: list[StageResult] = []
        self.slicing_error: str | None = None
        self.slicing_summary: SliceSummary | None = None

    def set_source(self, path: Path, image: np.ndarray) -> None:
        project = self.project
        project.source_path = Path(path)
        project.source_original = image
        project.source_edits = []
        project.source_image = image
        self._redo.clear()
        self._source_loads += 1
        self.source_changed.emit()

    # Committing the source

    def _source_key(self) -> tuple:
        return (self._source_loads, tuple(self.project.source_edits))

    @property
    def source_is_committed(self) -> bool:
        """Whether source_final matches the current loaded image and edits."""
        return self.project.has_source and self._committed_key == self._source_key()

    def commit_source(self) -> bool:
        """Snapshot the edited source as project.source_final.

        Returns True if the final image changed (and source_committed was
        emitted); committing an unchanged image leaves later steps' work intact.
        """
        if not self.project.has_source or self.source_is_committed:
            return False
        final = np.array(self.project.source_image, dtype=np.uint8, order="C", copy=True)
        final.setflags(write=False)
        self.project.source_final = final
        self._committed_key = self._source_key()
        self._rebuild_slice_context()
        self._evaluate_slicing()
        self.source_committed.emit()
        return True

    # Mosaic layout

    def set_layout(self, layout: MosaicLayout) -> None:
        """Change the base tile or column count; re-slices the final image."""
        if layout == self.project.layout:
            return
        self.project.layout = layout
        self._rebuild_slice_context()  # first, so listeners see the new mosaic size
        self.layout_changed.emit()
        self._evaluate_slicing()

    def mosaic_size(self) -> tuple[float, float] | None:
        """Canvas size in mosaic pixels, once the source is committed."""
        ctx = self._slice_context
        return None if ctx is None else (ctx.width, ctx.height)

    @property
    def slice_context(self) -> SliceContext | None:
        return self._slice_context

    # Slicing

    def slicing_edited(self) -> None:
        """Re-run slicing after project.slicing_plan (stages or parameters) changed."""
        self._evaluate_slicing()

    def _rebuild_slice_context(self) -> None:
        """New final image or layout: new context, no cached stage results."""
        final = self.project.source_final
        if final is not None:
            self._slice_context = SliceContext(final, self.project.layout)
        self._slicing_cache = []

    def _evaluate_slicing(self) -> None:
        project = self.project
        self.slicing_summary = None
        if self._slice_context is None:
            project.regions = None
        else:
            try:
                self._slicing_cache = project.slicing_plan.evaluate(
                    self._slice_context, self._slicing_cache
                )
            except SlicingError as exc:
                self.slicing_error = str(exc)
                self._slicing_cache = []
                project.regions = None
            else:
                self.slicing_error = None
                ctx = self._slice_context
                project.regions = (
                    self._slicing_cache[-1].regions if self._slicing_cache else ctx.canvas()
                )
                if project.regions:
                    self.slicing_summary = summarize(project.regions, ctx)
        self.slicing_changed.emit()

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

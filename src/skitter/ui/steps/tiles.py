"""Step 3: the tile library, the photos the mosaic is built from.

The user picks folders; Update reads new and changed images into the
library cache in the background (see core/tiles/library.py). The page shows
library statistics and a sample of tiles.
"""

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
)

from skitter.ui.canvas import MosaicCanvas
from skitter.ui.render.atlas import build_atlas
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.steps.base import StepPage, side_panel
from skitter.ui.style import muted, progress_bar, show_progress

SAMPLE_TILES = 2500
SAMPLE_CELL = 64.0  # world units per sample tile


def _format_bytes(n: float) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "bytes" else f"{n:,.1f} {unit}"
        n /= 1024
    raise AssertionError("unreachable")


class TilesStep(StepPage):
    title = "Tiles"

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.canvas = MosaicCanvas()
        self.canvas.clamp_to_bounds = True
        self._sample_layer: SpriteLayer | None = None
        self._empty = QLabel("Add folders of photos to use as tiles,\nthen press Update.")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet("color: palette(placeholder-text); font-size: 14pt;")
        self._view = QStackedWidget()
        self._view.addWidget(self._empty)
        self._view.addWidget(self.canvas)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._view, stretch=1)
        layout.addWidget(
            side_panel(
                self._build_folders_group(),
                self._build_update_group(),
                self._build_stats_group(),
                self._build_errors_group(),
            )  # fmt: skip
        )
        session.library_changed.connect(self._refresh)
        session.library_progress.connect(self._on_progress)
        session.busy_changed.connect(self._refresh)
        self._refresh()

    # Construction

    def _build_folders_group(self) -> QGroupBox:
        self.folders = QListWidget()
        self.folders.setMaximumHeight(120)
        self.add_button = QPushButton("Add Folder…")
        self.add_button.clicked.connect(self.add_folder_dialog)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_folder)
        buttons = QHBoxLayout()
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch()
        group = QGroupBox("Folders")
        layout = QVBoxLayout(group)
        layout.addWidget(self.folders)
        layout.addLayout(buttons)
        layout.addWidget(muted(QLabel("Images in these folders and their subfolders are tiles.")))
        return group

    def _build_update_group(self) -> QGroupBox:
        self.update_button = QPushButton("Update Library")
        self.update_button.setToolTip("Read new and changed images; unchanged ones are skipped.")
        self.update_button.clicked.connect(self.update_library)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.session.cancel_job)
        buttons = QHBoxLayout()
        buttons.addWidget(self.update_button)
        buttons.addWidget(self.cancel_button)
        self.progress = progress_bar()
        self.status = muted(QLabel())
        group = QGroupBox("Library")
        layout = QVBoxLayout(group)
        layout.addLayout(buttons)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        return group

    def _build_stats_group(self) -> QGroupBox:
        self._tiles = QLabel("—")
        self._failed = QLabel("—")
        self._missing = QLabel("—")
        self._disk = QLabel("—")
        self._location = muted(QLabel("—"))
        self._location.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        group = QGroupBox("Contents")
        form = QFormLayout(group)
        form.addRow("Tiles:", self._tiles)
        form.addRow("Unreadable:", self._failed)
        form.addRow("Missing:", self._missing)
        form.addRow("Cache size:", self._disk)
        form.addRow("Cache:", self._location)
        return group

    def _build_errors_group(self) -> QGroupBox:
        self.errors = QListWidget()
        self.errors.setMaximumHeight(100)
        self._errors_group = QGroupBox("Unreadable files")
        QVBoxLayout(self._errors_group).addWidget(self.errors)
        self._errors_group.hide()
        return self._errors_group

    # Actions

    def on_enter(self) -> None:
        if self.session.library is None:
            self.session.open_library()

    def add_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add Tile Folder")
        if folder:
            self.add_folder(folder)

    def add_folder(self, folder) -> None:
        library = self.session.library
        if library is None:  # not `or`: an empty library is falsy
            library = self.session.open_library()
        self.session.set_library_folders([*library.roots, folder])

    def remove_folder(self) -> None:
        row = self.folders.currentRow()
        library = self.session.library
        if library is None or row < 0:
            return
        roots = library.roots
        del roots[row]
        self.session.set_library_folders(roots)

    def update_library(self) -> None:
        if self.session.library is not None and not self.session.busy:
            self.progress.setRange(0, 0)
            self.status.setText("Starting…")
            self.session.update_library()

    def is_complete(self) -> bool:
        return self.session.library_ready

    # Display

    def _on_progress(self, message: str, fraction: float) -> None:
        self.status.setText(message)
        show_progress(self.progress, fraction)

    def _refresh(self) -> None:
        session = self.session
        library = session.library
        busy = session.busy
        updating = busy == "library"
        self.folders.clear()
        if library is not None:
            self.folders.addItems([str(p) for p in library.roots])
        has_folders = library is not None and bool(library.roots)
        for widget in (self.add_button, self.remove_button):
            widget.setEnabled(not busy)
        self.update_button.setEnabled(has_folders and not busy)
        self.cancel_button.setEnabled(updating)
        self.progress.setVisible(updating)

        if library is None:
            for label in (self._tiles, self._failed, self._missing, self._disk, self._location):
                label.setText("—")
        else:
            counts = library.counts()
            self._tiles.setText(f"{counts['ok']:,}")
            self._failed.setText(f"{counts['failed']:,}")
            self._missing.setText(f"{counts['missing']:,}")
            self._disk.setText(_format_bytes(library.disk_bytes()))
            self._location.setText(str(library.folder))

        report, error = session.library_report, session.library_error
        if not updating:
            if error:
                self.status.setText(f"Update failed: {error}")
            elif report is not None:
                text = (f"Added {report.added:,}, updated {report.updated:,}, "
                        f"{report.unchanged:,} unchanged, {report.missing:,} missing, "
                        f"{report.failed:,} unreadable.")  # fmt: skip
                self.status.setText(("Cancelled. " if report.cancelled else "") + text)
            elif library is not None and not has_folders:
                self.status.setText("Add a folder to begin.")
        self.errors.clear()
        if report is not None and report.errors:
            self.errors.addItems([f"{path}: {message}" for path, message in report.errors])
        self._errors_group.setVisible(self.errors.count() > 0)
        if not updating:
            self._show_sample()
        self.state_changed.emit()

    def _show_sample(self) -> None:
        library = self.session.library
        ids = library.ids if library is not None else np.zeros(0, np.int64)
        if self._sample_layer is not None:
            self.canvas.remove_layer(self._sample_layer)
            self._sample_layer = None
        if not len(ids):
            self._view.setCurrentWidget(self._empty)
            return
        self._view.setCurrentWidget(self.canvas)
        rng = np.random.default_rng(0)
        sample = np.sort(rng.choice(ids, min(len(ids), SAMPLE_TILES), replace=False))
        atlas = build_atlas(library.thumbs, library.thumb_size, sample)
        n = len(sample)
        columns = math.ceil(math.sqrt(n * 4 / 3))
        index = np.arange(n)
        instances = make_instances(n)
        instances["pos"] = np.stack([index % columns + 0.5, index // columns + 0.5], 1) * (
            SAMPLE_CELL
        )
        aspect = library.width[sample] / np.maximum(library.height[sample], 1)
        fit = 0.92 * SAMPLE_CELL
        instances["size"] = np.stack(
            [fit * np.minimum(1, aspect), fit * np.minimum(1, 1 / aspect)], axis=1
        )
        layer, uv = atlas.locate(sample, np.tile([0, 0, 1, 1], (n, 1)))
        instances["layer"], instances["uv"] = layer, uv
        self._sample_layer = self.canvas.add_layer(SpriteLayer(atlas.pages, instances))
        rows = math.ceil(n / columns)
        self.canvas.fit_to(0, 0, columns * SAMPLE_CELL, rows * SAMPLE_CELL)

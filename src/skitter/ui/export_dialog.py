"""Export Image window: save the mosaic, rendered at full detail, as PNG or JPEG."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from skitter.core.assembly import (
    FORMATS,
    SIZE_BY,
    SIZE_PARAMS,
    ExportReport,
    check_size,
    enlargement,
    export_frame,
)
from skitter.ui import preferences
from skitter.ui.style import WARNING_STYLE, progress_bar, show_progress
from skitter.ui.widgets.param_form import ParamForm

FILTERS = {"png": "PNG image (*.png)", "jpeg": "JPEG image (*.jpg *.jpeg)"}
SUFFIX_FORMATS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}
ENLARGED = 1.05  # a tile shown this much larger than its photo's pixels counts as enlarged


def _pixels(width: float, height: float) -> str:
    def number(value: float) -> str:
        return f"{value:,.1f}".removesuffix(".0")

    if abs(width - height) < 0.05:
        return f"{number(width)} px"
    return f"{number(width)} × {number(height)} px"


class ExportDialog(QDialog):
    """Modeless: export runs in the background and the window may be closed meanwhile."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.settings = session.project.export_settings
        self.setWindowTitle("Export Image")
        self.setMinimumWidth(480)
        self._last_path: str | None = None

        self.path = QLineEdit()
        self.path.editingFinished.connect(self._on_path_edited)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path, stretch=1)
        path_row.addWidget(browse)

        self.form = ParamForm()
        self.form.set_target(self.settings)
        self.form.changed.connect(self._on_setting_changed)
        self.size_label = QLabel()
        self.tile_label = QLabel()
        self.detail_label = QLabel()
        self.detail_label.setWordWrap(True)
        self.detail_label.setToolTip(
            "Tiles shown larger than their photos have pixels for look soft. "
            "A smaller size, or larger photos, avoid it."
        )
        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(WARNING_STYLE)

        options = QGroupBox("Options")
        options_layout = QVBoxLayout(options)
        options_layout.addWidget(self.form)
        info = QFormLayout()
        info.addRow("Image size:", self.size_label)
        info.addRow("Base tile:", self.tile_label)
        info.addRow("Tile photos:", self.detail_label)
        options_layout.addLayout(info)
        options_layout.addWidget(self.warning)

        self.progress = progress_bar()
        self.status = QLabel()
        self.status.setWordWrap(True)

        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.clicked.connect(self.open_folder)
        self.export_button = QPushButton("Export")
        self.export_button.setDefault(True)
        self.export_button.clicked.connect(self.export)
        self.cancel_button = QPushButton("Cancel Export")
        self.cancel_button.clicked.connect(session.cancel_job)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addWidget(self.open_folder_button)
        buttons.addStretch()
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(close_button)

        layout = QVBoxLayout(self)
        file_form = QFormLayout()
        file_form.addRow("Save as:", path_row)
        layout.addLayout(file_form)
        layout.addWidget(options)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addStretch()
        layout.addLayout(buttons)

        session.export_progress.connect(self._on_progress)
        session.export_finished.connect(self._on_finished)
        for signal in session.export_signals():
            signal.connect(self._refresh)
        self.path.setText(self._default_path())
        self._refresh()

    # Paths

    def _default_path(self) -> str:
        source = self.session.project.source_path
        stem = f"{source.stem} mosaic" if source else "mosaic"
        folder = preferences.settings().value("export/folder", "")
        if not folder or not Path(folder).is_dir():
            folder = source.parent if source else Path.home()
        return str(Path(folder) / f"{stem}{self.settings.extension}")

    def _on_path_edited(self) -> None:
        """A typed .png / .jpg extension chooses the format."""
        fmt = SUFFIX_FORMATS.get(Path(self.path.text().strip()).suffix.lower())
        if fmt is not None and fmt != self.settings.format:
            self.settings.format = fmt
            self.form.set_target(self.settings)
            self._refresh()

    def _match_extension(self) -> None:
        text = self.path.text().strip()
        if not text:
            return
        path = Path(text)
        if SUFFIX_FORMATS.get(path.suffix.lower()) != self.settings.format:
            stem = path.stem if path.suffix.lower() in SUFFIX_FORMATS else path.name
            self.path.setText(str(path.with_name(stem + self.settings.extension)))

    def browse(self) -> None:
        current = self.settings.format
        filters = [FILTERS[current]] + [f for k, f in FILTERS.items() if k != current]
        path, chosen = QFileDialog.getSaveFileName(
            self, "Export Image", self.path.text(), ";;".join(filters)
        )
        if not path:
            return
        fmt = SUFFIX_FORMATS.get(Path(path).suffix.lower())
        if fmt is None:  # no extension typed: use the chosen filter's
            fmt = next(k for k, f in FILTERS.items() if f == chosen)
            path += FORMATS[fmt][1]
        self.path.setText(path)
        self._on_path_edited()

    # Settings

    def _on_setting_changed(self, name: str) -> None:
        if name == "format":
            self._match_extension()
        self._refresh()

    def _size_problem(self) -> str | None:
        session = self.session
        if not session.can_export:
            return "Match tiles first: there is no mosaic to export."
        frame = export_frame(session.scene, self.settings)
        return check_size(frame.size, self.settings)

    def _refresh(self) -> None:
        session = self.session
        exporting = session.busy == "export"
        if session.can_export:
            frame = export_frame(session.scene, self.settings)
            w, h = frame.size
            channels = 4 if self.settings.alpha else 3
            memory = w * h * channels / 2**20
            self.size_label.setText(f"{w:,} × {h:,} px ({w * h / 1e6:,.1f} MP, {memory:,.0f} MB)")
            tw, th = session.slice_context.tile_size
            s = frame.scale
            self.tile_label.setText(_pixels(tw * s, th * s))
            self._show_detail(frame.scale)
            self._sync_sizes((max(1, round(tw * s)), w, h))
        else:
            for label in (self.size_label, self.tile_label, self.detail_label):
                label.setText("—")
                label.setStyleSheet("")
        problem = self._size_problem()
        self.warning.setText(problem or "")
        self.warning.setVisible(bool(problem))
        self.form.setEnabled(not exporting)
        self.path.setEnabled(not exporting)
        self.export_button.setVisible(not exporting)
        self.export_button.setEnabled(problem is None and not session.busy)
        self.cancel_button.setVisible(exporting)
        self.progress.setVisible(exporting)
        self.open_folder_button.setVisible(self._last_path is not None)
        if session.busy and not exporting:
            self.status.setText(f"Waiting for {session.busy} to finish…")
        elif not exporting and self.status.text().startswith("Waiting"):
            self.status.clear()

    def _sync_sizes(self, sizes: tuple[int, int, int]) -> None:
        """Show the resulting size in the inactive size fields, so switching keeps it."""
        for (key, _), name, value in zip(SIZE_BY, SIZE_PARAMS, sizes, strict=True):
            if key == self.settings.size_by:
                continue
            param = type(self.settings).__dict__[name]
            value = min(max(value, param.min), param.max)
            setattr(self.settings, name, value)
            widget = self.form.editor(name).widget
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

    def _show_detail(self, scale: float) -> None:
        """Whether the tile photos have enough pixels for this size."""
        library = self.session.library
        ratio = enlargement(self.session.scene, library.width, scale)
        enlarged = ratio > ENLARGED
        if not enlarged.any():
            self.detail_label.setText("Full detail (no tile is enlarged)")
            self.detail_label.setStyleSheet("")
            return
        share = enlarged.mean()
        self.detail_label.setText(
            f"{np.count_nonzero(enlarged):,} of {len(ratio):,} tiles ({share:.0%}) enlarged, "
            f"up to {ratio.max():,.1f}×"
        )
        self.detail_label.setStyleSheet(WARNING_STYLE if share > 0.1 else "")

    # Export

    def export(self) -> None:
        if self._size_problem() or self.session.busy:
            return
        self._match_extension()
        text = self.path.text().strip()
        if not text:
            self.status.setText("Choose where to save the image.")
            return
        path = Path(text).expanduser()
        if not path.parent.is_dir():
            self.status.setText(f"The folder {path.parent} does not exist.")
            return
        if path.exists() and not self._confirm_replace(path):
            return
        preferences.settings().setValue("export/folder", str(path.parent))
        self.progress.setRange(0, 0)
        self.status.setText("Starting…")
        self.session.start_export(path, self.settings)

    def _confirm_replace(self, path: Path) -> bool:
        answer = QMessageBox.question(
            self, "Export Image", f"{path.name} already exists. Replace it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )  # fmt: skip
        return answer == QMessageBox.StandardButton.Yes

    def open_folder(self) -> None:
        if self._last_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self._last_path).parent)))

    def _on_progress(self, message: str, fraction: float) -> None:
        self.status.setText(message)
        show_progress(self.progress, fraction)

    def _on_finished(self, path: str, report: ExportReport | None, error: str | None) -> None:
        if error:
            self.status.setText(f"Export failed: {error}")
        elif report is None:
            self.status.setText("Export cancelled.")
        else:
            w, h = report.size
            text = f"Saved {Path(path).name} ({w:,} × {h:,} px)."
            if report.failed:
                text += f" {report.failed:,} unreadable tile files used their thumbnails."
            self.status.setText(text)
            self._last_path = path
        self._refresh()

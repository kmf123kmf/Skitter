"""Export Animation window: render the build animation to a video file.

Modeless, like Export Image: the export runs in the background (a session
job) and the window may be closed meanwhile. Settings live in the project
(project.video_settings) and are shared with the Animate tab, which edits
the size and framing too: the window opens with what the tab shows, and
edits in either place show in both.
"""

from pathlib import Path

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

from skitter.core.animation.video import (
    FORMATS,
    estimate_bytes,
    free_bytes,
    plan_video,
)
from skitter.ui import preferences
from skitter.ui.render.tile_textures import detail_sizes
from skitter.ui.render.video_export import VideoReport
from skitter.ui.style import WARNING_STYLE, muted, progress_bar, show_progress
from skitter.ui.widgets.param_form import ParamForm

VIDEO_EXTENSIONS = {f.extension for f in FORMATS.values() if f.extension}


def _bytes(n: float) -> str:
    for unit, size in (("GB", 2**30), ("MB", 2**20)):
        if n >= size:
            return f"{n / size:,.1f} {unit}"
    return f"{n / 2**10:,.0f} KB"


class VideoExportDialog(QDialog):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.settings = session.project.video_settings
        self.setWindowTitle("Export Animation")
        self.setMinimumWidth(520)
        self._last_path: str | None = None

        self.path = QLineEdit()
        self.path.editingFinished.connect(self._refresh)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path, stretch=1)
        path_row.addWidget(browse)

        self.form = ParamForm()
        self.form.set_target(self.settings)
        self.form.changed.connect(self._on_setting_changed)
        self.format_note = muted(QLabel())
        self.size_label = QLabel()
        self.length_label = QLabel()
        self.detail_label = QLabel()
        self.detail_label.setWordWrap(True)
        self.disk_label = QLabel()
        self.disk_label.setWordWrap(True)
        self.problems = QLabel()
        self.problems.setWordWrap(True)
        self.problems.setStyleSheet(WARNING_STYLE)

        options = QGroupBox("Options")
        options_layout = QVBoxLayout(options)
        options_layout.addWidget(self.form)
        options_layout.addWidget(self.format_note)
        info = QFormLayout()
        info.addRow("Video:", self.size_label)
        info.addRow("Length:", self.length_label)
        info.addRow("Tile detail:", self.detail_label)
        info.addRow("Disk:", self.disk_label)
        options_layout.addLayout(info)
        options_layout.addWidget(self.problems)

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

        session.video_progress.connect(self._on_progress)
        session.video_finished.connect(self._on_finished)
        for signal in session.export_signals():
            signal.connect(self._refresh)
        session.animation_changed.connect(self._on_animation_changed)
        self.path.setText(self._default_path())
        self._refresh()

    # Paths

    @property
    def fmt(self):
        return self.settings.video_format

    def _default_path(self) -> str:
        source = self.session.project.source_path
        stem = f"{source.stem} animation" if source else "mosaic animation"
        folder = preferences.settings().value("export/video_folder", "")
        if not folder or not Path(folder).is_dir():
            folder = source.parent if source else Path.home()
        return str(Path(folder) / f"{stem}{self.fmt.extension}")

    def _match_extension(self) -> None:
        text = self.path.text().strip()
        if not text:
            return
        path = Path(text)
        stem = path.stem if path.suffix.lower() in VIDEO_EXTENSIONS else path.name
        self.path.setText(str(path.with_name(stem + self.fmt.extension)))

    def browse(self) -> None:
        if self.fmt.sequence:
            folder = QFileDialog.getExistingDirectory(
                self, "Folder for the new frame folder", str(Path(self.path.text()).parent)
            )
            if folder:
                self.path.setText(str(Path(folder) / Path(self.path.text()).name))
        else:
            ext = self.fmt.extension
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Animation", self.path.text(), f"{self.fmt.name} (*{ext})"
            )
            if path:
                self.path.setText(path if path.lower().endswith(ext) else path + ext)
        self._refresh()

    def _target(self) -> Path | None:
        text = self.path.text().strip()
        return Path(text).expanduser() if text else None

    # Settings and readouts

    def _on_setting_changed(self, name: str) -> None:
        if name == "format":
            self._match_extension()
        self.session.animation_edited()  # the Animate tab's export frame follows

    def _on_animation_changed(self) -> None:
        """Settings edited here or on the Animate tab (they share them)."""
        self.form.refresh()
        self._refresh()

    def _refresh(self) -> None:
        session = self.session
        exporting = session.busy == "video"
        self.format_note.setText(self.fmt.note)
        scene = session.scene
        problems = session.video_problems(self._target())
        if self._target() is None:
            problems.append("Choose where to save the animation.")
        if scene is not None and len(scene):
            self._show_plan(scene)
        else:
            for label in (self.size_label, self.length_label, self.detail_label, self.disk_label):
                label.setText("—")
        self.problems.setText("\n".join(problems))
        self.problems.setVisible(bool(problems))
        self.form.setEnabled(not exporting)
        self.path.setEnabled(not exporting)
        self.export_button.setVisible(not exporting)
        self.export_button.setEnabled(not problems and not session.busy)
        self.cancel_button.setVisible(exporting)
        self.progress.setVisible(exporting)
        self.open_folder_button.setVisible(self._last_path is not None)
        if session.busy and not exporting:
            self.status.setText(f"Waiting for {session.busy} to finish…")
        elif not exporting and self.status.text().startswith("Waiting"):
            self.status.clear()

    def _show_plan(self, scene) -> None:
        project = self.session.project
        duration = project.choreography.timeline(scene, project.animation_look).duration
        plan = plan_video(scene, duration, self.settings, project.animation_look.background)
        fps = float(plan.fps)
        self.size_label.setText(f"{plan.width:,} × {plan.height:,} px at {fps:g} fps")
        samples = 8 * self.settings.supersampling**2 * plan.samples
        self.length_label.setText(
            f"{plan.seconds:.1f} s, {plan.frames:,} frames ({samples:,} samples per pixel)"
        )
        _, scale, _ = detail_sizes(scene.size * plan.scale)
        tile_px = plan.scale * scene.tile_size[0]
        text = f"{tile_px:,.0f} px base tiles"
        text += "" if scale >= 1 else f", textures reduced to {scale:.0%} (memory)"
        self.detail_label.setText(text)
        estimate = estimate_bytes(plan, self.fmt)
        target = self._target()
        free = free_bytes(target) if target is not None else None
        if estimate is None:
            self.disk_label.setText("Size shows while exporting" + (
                f"; {_bytes(free)} free" if free is not None else ""))  # fmt: skip
            self.disk_label.setStyleSheet("")
        else:
            short = free is not None and free < estimate * 1.1
            self.disk_label.setText(
                f"About {_bytes(estimate)}" + (f"; {_bytes(free)} free" if free is not None else "")
                + (" — not enough space" if short else "")
            )  # fmt: skip
            self.disk_label.setStyleSheet(WARNING_STYLE if short else "")

    # Export

    def export(self) -> None:
        target = self._target()
        if target is None or self.session.video_problems(target) or self.session.busy:
            self._refresh()
            return
        if target.exists() and not self.fmt.sequence and not self._confirm_replace(target):
            return
        preferences.settings().setValue(
            "export/video_folder", str(target.parent if not self.fmt.sequence else target.parent)
        )
        self.progress.setRange(0, 0)
        self.status.setText("Starting…")
        self.session.start_video_export(target)

    def _confirm_replace(self, path: Path) -> bool:
        answer = QMessageBox.question(
            self, "Export Animation", f"{path.name} already exists. Replace it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )  # fmt: skip
        return answer == QMessageBox.StandardButton.Yes

    def open_folder(self) -> None:
        if self._last_path:
            path = Path(self._last_path)
            folder = path if path.is_dir() else path.parent
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _on_progress(self, message: str, fraction: float) -> None:
        self.status.setText(message)
        show_progress(self.progress, fraction)

    def _on_finished(self, path: str, report: VideoReport | None, error: str | None) -> None:
        if error:
            self.status.setText(f"Export failed: {error}")
        elif report is None:
            self.status.setText("Export cancelled; nothing was saved.")
        else:
            w, h = report.size
            text = (
                f"Saved {Path(path).name}: {w:,} × {h:,}, {report.seconds:.1f} s, "
                f"{_bytes(report.bytes)}, in {report.elapsed:.0f} s."
            )
            if report.failed:
                text += f" {report.failed:,} unreadable tile files used their thumbnails."
            self.status.setText(text)
            self._last_path = path
        self._refresh()

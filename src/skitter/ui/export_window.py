"""What the Export Image and Export Animation windows share.

Modeless: the export runs in the background (a session job) and the window
may be closed meanwhile. The window edits settings that live in the project
(a Configurable, shown by a ParamForm) and saves to the path typed or
browsed for, remembering the folder. A subclass supplies its settings, its
readouts (rows added to `info`), what blocks an export (`_problems`) and how
to start one, the path's extension and what to say once saved.
"""

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from skitter.ui import preferences
from skitter.ui.style import WARNING_STYLE, progress_bar, show_progress
from skitter.ui.widgets.param_form import ParamForm


class ExportWindow(QDialog):
    title = "Export"
    what = "file"  # what is saved, for messages ("Choose where to save the image.")
    job = ""  # Session.busy while this window's export runs
    folder_key = ""  # preference remembering the folder last saved to
    min_width = 480

    def __init__(self, session, progress_signal, finished_signal, parent=None):
        super().__init__(parent)
        self.session = session
        self.settings = self._project_settings()
        self.setWindowTitle(self.title)
        self.setMinimumWidth(self.min_width)
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
        self.info = QFormLayout()  # the subclass's readouts
        self.problems = QLabel()
        self.problems.setWordWrap(True)
        self.problems.setStyleSheet(WARNING_STYLE)
        options = QGroupBox("Options")
        self.options_layout = QVBoxLayout(options)
        self.options_layout.addWidget(self.form)
        self.options_layout.addLayout(self.info)
        self.options_layout.addWidget(self.problems)

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

        session.project_replaced.connect(self._on_project_replaced)
        progress_signal.connect(self._on_progress)
        finished_signal.connect(self._on_finished)
        for signal in session.export_signals():
            signal.connect(self._refresh)

    def _start(self) -> None:
        """Fill in the path and readouts (call at the end of a subclass's __init__)."""
        self.path.setText(self._default_path())
        self._refresh()

    # For subclasses

    def _project_settings(self):
        """The project's settings this window edits."""
        raise NotImplementedError

    def _default_name(self) -> str:
        """The file name to offer, with its extension."""
        raise NotImplementedError

    def _match_extension(self) -> None:
        """Give the typed path the extension the settings call for."""

    def _on_path_edited(self) -> None:
        self._refresh()

    def browse(self) -> None:
        raise NotImplementedError

    def _on_setting_changed(self, name: str) -> None:
        self._refresh()

    def _problems(self, target: Path | None) -> list[str]:
        """What keeps the export from starting (besides having no path)."""
        return []

    def _show_readouts(self) -> None:
        """Update the rows in `info` (there is a mosaic to export)."""

    def _clear_readouts(self) -> None:
        """No mosaic: the rows in `info` show a dash."""
        for row in range(self.info.rowCount()):
            item = self.info.itemAt(row, QFormLayout.ItemRole.FieldRole)
            if item is not None and isinstance(item.widget(), QLabel):
                item.widget().setText("—")
                item.widget().setStyleSheet("")

    def _replaces(self, target: Path) -> bool:
        """Whether exporting to target would replace a file there (ask first)."""
        return target.exists()

    def _begin(self, target: Path) -> None:
        """Start the export job."""
        raise NotImplementedError

    def _saved(self, path: str, report) -> str:
        """What to say once the export is saved."""
        return f"Saved {Path(path).name}."

    # Shared

    def _on_project_replaced(self) -> None:
        self.settings = self._project_settings()
        self.form.set_target(self.settings)
        self.path.setText(self._default_path())
        self._refresh()

    def _default_path(self) -> str:
        source = self.session.project.source_path
        folder = preferences.settings().value(self.folder_key, "")
        if not folder or not Path(folder).is_dir():
            folder = source.parent if source else Path.home()
        return str(Path(folder) / self._default_name())

    def _target(self) -> Path | None:
        text = self.path.text().strip()
        return Path(text).expanduser() if text else None

    def _all_problems(self) -> list[str]:
        target = self._target()
        problems = self._problems(target)
        if target is None:
            problems.append(f"Choose where to save the {self.what}.")
        return problems

    def _refresh(self) -> None:
        session = self.session
        exporting = session.busy == self.job
        if session.can_export:
            self._show_readouts()
        else:
            self._clear_readouts()
        problems = self._all_problems()
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

    def export(self) -> None:
        self._match_extension()
        target = self._target()
        if self._all_problems() or self.session.busy:
            self._refresh()
            return
        if self._replaces(target) and not self._confirm_replace(target):
            return
        preferences.settings().setValue(self.folder_key, str(target.parent))
        self.progress.setRange(0, 0)
        self.status.setText("Starting…")
        self._begin(target)

    def _confirm_replace(self, path: Path) -> bool:
        answer = QMessageBox.question(
            self, self.title, f"{path.name} already exists. Replace it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )  # fmt: skip
        return answer == QMessageBox.StandardButton.Yes

    def open_folder(self) -> None:
        """Show the folder of what was saved last (a folder of frames: that folder)."""
        if self._last_path:
            path = Path(self._last_path)
            folder = path if path.is_dir() else path.parent
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _on_progress(self, message: str, fraction: float) -> None:
        self.status.setText(message)
        show_progress(self.progress, fraction)

    def _on_finished(self, path: str, report, error: str | None) -> None:
        if error:
            self.status.setText(f"Export failed: {error}")
        elif report is None:
            self.status.setText("Export cancelled; nothing was saved.")
        else:
            text = self._saved(path, report)
            if report.failed:
                text += f" {report.failed:,} unreadable tile files used their thumbnails."
            self.status.setText(text)
            self._last_path = path
        self._refresh()

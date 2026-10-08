"""Top-level application window: one tab per workflow step.

The File menu saves and opens projects (core/project_file.py): New, Open,
Open Recent, Save, Save As. The title shows the project and `*` while it has
unsaved changes; New, Open and closing ask before dropping them.
"""

import threading
import time
from contextlib import contextmanager
from pathlib import Path

from PySide6.QtCore import QEventLoop, QSize, Qt
from PySide6.QtGui import QAction, QCursor, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from skitter.core.project_file import EXTENSION, ProjectFileError
from skitter.ui import preferences
from skitter.ui.export_dialog import ExportDialog
from skitter.ui.session import Session
from skitter.ui.steps import STEPS, StepPage
from skitter.ui.steps.animate import AnimateStep
from skitter.ui.steps.matching import MatchingStep
from skitter.ui.steps.slicing import SlicingStep
from skitter.ui.steps.source import SourceStep
from skitter.ui.video_dialog import VideoExportDialog
from skitter.ui.widgets.wheel_guard import install_wheel_guard

PROJECT_FILTER = f"Skitter projects (*{EXTENSION})"
RECENT_KEY, RECENT_COUNT = "project/recent", 8


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        install_wheel_guard()  # the wheel never changes a setting by accident
        self.setWindowTitle("Skitter")
        self._fit_to_screen()

        self.session = Session(self)
        self.steps: list[StepPage] = [cls(self.session) for cls in STEPS]

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        for step in self.steps:
            self.tabs.addTab(step, step.title)
            step.state_changed.connect(self._update_navigation)
            step.status_message.connect(self.statusBar().showMessage)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tabs, stretch=1)
        layout.addWidget(self._build_footer())
        self.setCentralWidget(central)

        self._current_step = self.steps[0]
        self.tabs.currentChanged.connect(self._on_tab_changed)
        session = self.session
        for signal in (
            session.project_replaced, session.source_changed, session.source_edited,
            session.source_committed, session.layout_changed, session.slicing_changed,
            session.matching_changed, session.mosaic_edited, session.animation_changed,
            session.settings_edited,
        ):  # fmt: skip
            signal.connect(self._update_title)

        self.export_dialog: ExportDialog | None = None
        self.video_dialog: VideoExportDialog | None = None
        self._build_menus()
        self._update_title()
        self._update_navigation()
        self.statusBar().showMessage("Choose a source image to begin")

    def _fit_to_screen(self, fraction: float = 0.8) -> None:
        """Size the window to a share of the screen it opens on, centred there."""
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 800)
            return
        available = screen.availableGeometry()  # excludes the taskbar
        size = QSize(round(available.width() * fraction), round(available.height() * fraction))
        self.setGeometry(
            QStyle.alignedRect(
                Qt.LayoutDirection.LeftToRight,
                Qt.AlignmentFlag.AlignCenter,
                size,
                available,
            )
        )

    def step(self, cls: type[StepPage]) -> StepPage:
        return next(s for s in self.steps if isinstance(s, cls))

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.aboutToShow.connect(self._update_title)  # Save follows unsaved changes

        def add(text, slot, shortcut=None) -> QAction:
            action = QAction(text, self)
            if shortcut is not None:
                action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(slot)
            file_menu.addAction(action)
            return action

        add("&New Project", self.new_project, QKeySequence.StandardKey.New)
        add("&Open Project...", self.open_project_dialog, QKeySequence.StandardKey.Open)
        self.recent_menu = file_menu.addMenu("Open &Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        file_menu.addSeparator()
        self.save_action = add("&Save Project", self.save_project, QKeySequence.StandardKey.Save)
        self.save_as_action = add("Save Project &As...", self.save_project_as, "Ctrl+Shift+S")
        file_menu.addSeparator()
        add("Open Source &Image...", self.open_source_image, "Ctrl+Shift+O")

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        mosaic_menu = self.menuBar().addMenu("&Mosaic")
        self.export_action = QAction("&Export Image...", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.triggered.connect(self.open_export)
        mosaic_menu.addAction(self.export_action)
        self.video_action = QAction("Export &Animation...", self)
        self.video_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self.video_action.triggered.connect(self.open_video_export)
        mosaic_menu.addAction(self.video_action)
        self.step(AnimateStep).export_requested.connect(self.open_video_export)
        self.step(MatchingStep).export_requested.connect(self.open_export)
        session = self.session
        for signal in session.export_signals():
            signal.connect(self._update_actions)
        session.export_finished.connect(self._on_exported)

        self._update_actions()

    # Projects

    def new_project(self) -> bool:
        if not self._may_replace_project():
            return False
        self.session.new_project()
        self._update_title()
        self.tabs.setCurrentWidget(self.step(SourceStep))
        self.statusBar().showMessage("New project: choose a source image to begin")
        return True

    def open_project_dialog(self) -> bool:
        if not self._may_replace_project():
            return False
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", self._project_folder(),
                                              PROJECT_FILTER)  # fmt: skip
        return bool(path) and self.open_project(path, ask=False)

    def open_project(self, path, ask: bool = True) -> bool:
        """Open a project file (ask: first offer to save unsaved changes)."""
        if ask and not self._may_replace_project():
            return False
        try:
            with self._busy(f"Opening {Path(path).name}…"):
                session = self.session
                if session.library is None:
                    session.open_library()  # here: it tells the tabs
                opened = self._in_background(lambda: session.read_project(path))
                session.open_project(path, opened)
                self._until_drawn()
        except (ProjectFileError, OSError) as exc:
            QMessageBox.warning(self, "Open Project", str(exc))
            self._forget_recent(path)
            return False
        self._remember(path)
        self._update_title()
        session = self.session
        self.tabs.setCurrentWidget(self._step_to_show(session.opened_view.get("step")))
        notes = list(session.load_problems)
        if session.missing_tiles:
            notes.insert(0, f"{session.missing_tiles:,} tiles show photos that are no longer in "
                            "the tile library; run matching to fill them.")  # fmt: skip
        if notes:
            QMessageBox.information(self, "Open Project", "Opened, with notes:\n\n"
                                    + "\n".join(f"• {note}" for note in notes))  # fmt: skip
        self.statusBar().showMessage(f"Opened {Path(path).name}", 10_000)
        return True

    @contextmanager
    def _busy(self, text: str):
        """A modal dialog with a bouncing bar while the work inside runs (it bounces
        whenever events are processed, see _in_background and _until_drawn)."""
        dialog = QProgressDialog(text, None, 0, 0, self)
        dialog.setWindowTitle("Skitter")
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)  # it can't stop the work
        dialog.setMinimumDuration(0)
        dialog.setMinimumWidth(360)
        dialog.show()
        self._pump()
        try:
            yield
        finally:
            dialog.close()
            dialog.deleteLater()

    def _pump(self, seconds: float = 0.05) -> None:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, round(seconds * 1000))

    def _in_background(self, work):
        """work() in a thread while the UI keeps going; its result (or its error)."""
        result = {}

        def run():
            try:
                result["value"] = work()
            except BaseException as exc:  # handed to the caller
                result["error"] = exc

        thread = threading.Thread(target=run, daemon=True, name="skitter-open")
        thread.start()
        while thread.is_alive():
            self._pump()
            thread.join(0.02)
        if "error" in result:
            raise result["error"]
        return result["value"]

    def _until_drawn(self) -> None:
        """Keep the UI going until the opened project is in: its slicing run (if it
        had no regions) and its tile textures."""
        session = self.session
        while session.slicing_running or (session.textures is not None
                                           and session.textures.loading):  # fmt: skip
            self._pump()
            time.sleep(0.01)
        self._pump()

    def save_project(self) -> bool:
        path = self.session.project_path
        return self.save_project_as() if path is None else self._save_to(path)

    def save_project_as(self) -> bool:
        session = self.session
        if not session.project.has_source:
            self.statusBar().showMessage("Choose a source image first: there is nothing to save")
            return False
        current = session.project_path
        if current is None:
            source = session.project.source_path
            name = f"{source.stem if source else 'mosaic'}{EXTENSION}"
            current = Path(self._project_folder()) / name
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", str(current), PROJECT_FILTER)
        if not path:
            return False
        path = Path(path)
        if path.suffix.lower() != EXTENSION:
            path = path.with_name(path.name + EXTENSION)
        return self._save_to(path)

    def _step_to_show(self, step_id) -> StepPage:
        """The tab an opened project was saved on (by id), if it's here and open; else
        the furthest one its work reaches."""
        self._update_navigation()
        for index, step in enumerate(self.steps):
            if step.id == step_id and self.tabs.isTabEnabled(index):
                return step
        session = self.session
        if session.project.matches is not None:
            return self.step(MatchingStep)
        if session.source_is_committed:
            return self.step(SlicingStep)
        return self.step(SourceStep)

    def _save_to(self, path) -> bool:
        try:
            self.session.save_project(path, {"step": self.steps[self.tabs.currentIndex()].id})
        except OSError as exc:
            QMessageBox.warning(self, "Save Project", f"Could not save {Path(path).name}: {exc}")
            return False
        self._remember(path)
        self._update_title()
        self.statusBar().showMessage(f"Saved {Path(path).name}", 10_000)
        return True

    def _may_replace_project(self) -> bool:
        """Whether the current project may go: saved or discarded on request, and no
        background job running."""
        session = self.session
        if session.busy:
            QMessageBox.information(self, "Skitter", f"Wait for the {session.busy} job to "
                                    "finish, or cancel it, first.")  # fmt: skip
            return False
        if not session.project.has_source or not session.modified:
            return True
        name = session.project_path.name if session.project_path else "this project"
        answer = QMessageBox.question(
            self, "Skitter", f"Save changes to {name}?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )  # fmt: skip
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return answer == QMessageBox.StandardButton.Discard

    def _project_folder(self) -> str:
        folder = preferences.settings().value("project/folder", "")
        return folder if folder and Path(folder).is_dir() else str(Path.home())

    def _recent(self) -> list[str]:
        value = preferences.settings().value(RECENT_KEY, [])
        return [value] if isinstance(value, str) else list(value or [])

    def _remember(self, path) -> None:
        path = str(Path(path))
        recent = [path] + [p for p in self._recent() if p != path]
        prefs = preferences.settings()
        prefs.setValue(RECENT_KEY, recent[:RECENT_COUNT])
        prefs.setValue("project/folder", str(Path(path).parent))

    def _forget_recent(self, path) -> None:
        path = str(Path(path))
        preferences.settings().setValue(RECENT_KEY, [p for p in self._recent() if p != path])

    def _fill_recent(self) -> None:
        menu = self.recent_menu
        menu.clear()
        recent = self._recent()
        for path in recent:
            menu.addAction(Path(path).name, lambda p=path: self.open_project(p)).setToolTip(path)
        menu.setToolTipsVisible(True)
        if not recent:
            menu.addAction("No recent projects").setEnabled(False)

    def open_source_image(self) -> None:
        source = self.step(SourceStep)
        self.tabs.setCurrentWidget(source)
        source.open_dialog()

    def open_export(self) -> ExportDialog:
        if self.export_dialog is None:
            self.export_dialog = ExportDialog(self.session, self)
        self.export_dialog.show()
        self.export_dialog.raise_()
        self.export_dialog.activateWindow()
        return self.export_dialog

    def open_video_export(self) -> VideoExportDialog:
        if self.video_dialog is None:
            self.video_dialog = VideoExportDialog(self.session, self)
        self.video_dialog.show()
        self.video_dialog.raise_()
        self.video_dialog.activateWindow()
        return self.video_dialog

    def _update_actions(self) -> None:
        self.export_action.setEnabled(self.session.can_export)
        scene = self.session.scene
        self.video_action.setEnabled(
            self.session.can_export and scene is not None and len(scene) > 0
        )

    def _on_exported(self, path: str, report, error) -> None:
        if report is not None:
            self.statusBar().showMessage(f"Exported {path}", 10_000)

    def _build_footer(self) -> QWidget:
        style = self.style()
        self.back_button = QPushButton("Back")
        self.back_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        self.back_button.clicked.connect(self.go_back)

        self.next_button = QPushButton()
        self.next_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ArrowForward))
        self.next_button.setLayoutDirection(Qt.LayoutDirection.RightToLeft)  # icon after text
        self.next_button.setDefault(True)
        self.next_button.setShortcut(QKeySequence("Ctrl+Return"))
        self.next_button.clicked.connect(self.go_next)

        footer = QFrame()
        footer.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self.back_button)
        layout.addStretch()
        layout.addWidget(self.next_button)
        return footer

    def go_next(self) -> None:
        """Commit the current step and move to the next one."""
        index = self.tabs.currentIndex()
        step = self.steps[index]
        if index + 1 >= len(self.steps) or not step.can_advance() or not step.advance():
            return
        self._update_navigation()
        if self.tabs.isTabEnabled(index + 1):
            self.tabs.setCurrentIndex(index + 1)

    def go_back(self) -> None:
        index = self.tabs.currentIndex()
        if index > 0:
            self.tabs.setCurrentIndex(index - 1)

    def _update_navigation(self) -> None:
        """Unlock tabs whose earlier steps are all complete; refresh Back/Next."""
        unlocked = True
        for index, step in enumerate(self.steps):
            self.tabs.setTabEnabled(index, unlocked)
            unlocked = unlocked and step.is_complete()

        index = self.tabs.currentIndex()
        step = self.steps[index]
        self.back_button.setVisible(index > 0)
        has_next = index + 1 < len(self.steps)
        self.next_button.setVisible(has_next)
        if has_next:
            self.next_button.setText(f"Next: {self.steps[index + 1].title}")
            self.next_button.setToolTip("Finish this step and continue (Ctrl+Enter)")
            self.next_button.setEnabled(step.can_advance())

    def _on_tab_changed(self, index: int) -> None:
        self._current_step.on_leave()
        self._current_step = self.steps[index]
        self._current_step.on_enter()
        self._update_navigation()

    def closeEvent(self, event) -> None:
        if not self._may_replace_project():
            event.ignore()
            return
        for step in self.steps:
            step.shutdown()
        self.session.shutdown()  # stop background work before the window goes
        super().closeEvent(event)

    def _update_title(self, *_) -> None:
        session = self.session
        if session.project_path is not None:
            name = session.project_path.name
        elif session.project.source_path is not None:
            name = session.project.source_path.name
        else:
            name = None
        self.setWindowTitle(f"Skitter — {name}[*]" if name else "Skitter[*]")
        unsaved = session.project.has_source and session.modified
        self.setWindowModified(unsaved)
        self.save_action.setEnabled(unsaved)
        self.save_as_action.setEnabled(session.project.has_source)

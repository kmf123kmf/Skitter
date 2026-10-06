"""Top-level application window: one tab per workflow step."""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QCursor, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from skitter.ui.export_dialog import ExportDialog
from skitter.ui.session import Session
from skitter.ui.steps import STEPS, StepPage
from skitter.ui.steps.animate import AnimateStep
from skitter.ui.steps.matching import MatchingStep
from skitter.ui.steps.source import SourceStep
from skitter.ui.video_dialog import VideoExportDialog
from skitter.ui.widgets.wheel_guard import install_wheel_guard


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
        self.session.source_changed.connect(self._update_title)

        self.export_dialog: ExportDialog | None = None
        self.video_dialog: VideoExportDialog | None = None
        self._build_menus()
        self._update_navigation()
        self.statusBar().showMessage("Choose a source image to begin")

    def _fit_to_screen(self, fraction: float = 0.8) -> None:
        """Size the window to a share of the screen it opens on, centred there."""
        screen = (
            QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        )
        if screen is None:
            self.resize(1280, 800)
            return
        available = screen.availableGeometry()  # excludes the taskbar
        size = QSize(
            round(available.width() * fraction), round(available.height() * fraction)
        )
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

        open_action = QAction("&Open Source Image...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_source_image)
        file_menu.addAction(open_action)

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
        self.next_button.setIcon(
            style.standardIcon(QStyle.StandardPixmap.SP_ArrowForward)
        )
        self.next_button.setLayoutDirection(
            Qt.LayoutDirection.RightToLeft
        )  # icon after text
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
        for step in self.steps:
            step.shutdown()
        self.session.shutdown()  # stop background work before the window goes
        super().closeEvent(event)

    def _update_title(self) -> None:
        path = self.session.project.source_path
        self.setWindowTitle(f"Skitter — {path.name}" if path else "Skitter")

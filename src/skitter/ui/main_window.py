"""Top-level application window: one tab per workflow step."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
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

from skitter.ui.demo import DemoWindow
from skitter.ui.export_dialog import ExportDialog
from skitter.ui.session import Session
from skitter.ui.steps import STEPS, StepPage
from skitter.ui.steps.source import SourceStep

DEMO_COUNTS = (1_000, 10_000, 50_000)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skitter")
        self.resize(1280, 800)

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

        self._demo_windows: list[DemoWindow] = []
        self.export_dialog: ExportDialog | None = None
        self._build_menus()
        self._update_navigation()
        self.statusBar().showMessage("Choose a source image to begin")

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
        session = self.session
        for signal in session.export_signals():
            signal.connect(self._update_actions)
        session.export_finished.connect(self._on_exported)
        self._update_actions()

        demo_menu = self.menuBar().addMenu("&Demo")
        for count in DEMO_COUNTS:
            action = QAction(f"Flying Tiles ({count:,})", self)
            action.triggered.connect(lambda _=False, n=count: self.open_demo(n))
            demo_menu.addAction(action)

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

    def _update_actions(self) -> None:
        self.export_action.setEnabled(self.session.can_export)

    def _on_exported(self, path: str, report, error) -> None:
        if report is not None:
            self.statusBar().showMessage(f"Exported {path}", 10_000)

    def open_demo(self, count: int) -> DemoWindow:
        window = DemoWindow(count)
        window.destroyed.connect(lambda: self._demo_windows.remove(window))
        self._demo_windows.append(window)
        window.show()
        return window

    # Step navigation

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
        self.session.cancel_job()  # stop background work before the window goes
        for step in self.steps:
            step.shutdown()
        self.session.wait_for_job(timeout=10)
        super().closeEvent(event)

    def _update_title(self) -> None:
        path = self.session.project.source_path
        self.setWindowTitle(f"Skitter — {path.name}" if path else "Skitter")

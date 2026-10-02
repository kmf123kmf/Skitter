"""Top-level application window: one tab per workflow step."""

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow, QTabWidget

from skitter.ui.demo import DemoWindow
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
            step.completion_changed.connect(self._update_tab_access)
            step.status_message.connect(self.statusBar().showMessage)
        self.setCentralWidget(self.tabs)

        self._current_step = self.steps[0]
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.session.source_changed.connect(self._update_title)

        self._demo_windows: list[DemoWindow] = []
        self._build_menus()
        self._update_tab_access()
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

        demo_menu = self.menuBar().addMenu("&Demo")
        for count in DEMO_COUNTS:
            action = QAction(f"Flying Tiles ({count:,})", self)
            action.triggered.connect(lambda _=False, n=count: self.open_demo(n))
            demo_menu.addAction(action)

    def open_source_image(self) -> None:
        source = self.step(SourceStep)
        self.tabs.setCurrentWidget(source)
        source.open_dialog()

    def open_demo(self, count: int) -> DemoWindow:
        window = DemoWindow(count)
        window.destroyed.connect(lambda: self._demo_windows.remove(window))
        self._demo_windows.append(window)
        window.show()
        return window

    def _update_tab_access(self) -> None:
        """Enable each tab only if every earlier step is complete."""
        unlocked = True
        for index, step in enumerate(self.steps):
            self.tabs.setTabEnabled(index, unlocked)
            unlocked = unlocked and step.is_complete()

    def _on_tab_changed(self, index: int) -> None:
        self._current_step.on_leave()
        self._current_step = self.steps[index]
        self._current_step.on_enter()

    def _update_title(self) -> None:
        path = self.session.project.source_path
        self.setWindowTitle(f"Skitter — {path.name}" if path else "Skitter")

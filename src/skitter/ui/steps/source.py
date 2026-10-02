"""Step 1: choose the source image the mosaic will reproduce."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from skitter.core.imaging import IMAGE_EXTENSIONS, load_image
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.steps.base import StepPage

IMAGE_FILTER = "Images (" + " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS)) + ")"
PANEL_WIDTH = 300


def _format_bytes(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    raise AssertionError("unreachable")


class SourceStep(StepPage):
    title = "Source"

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.setAcceptDrops(True)

        self.canvas = MosaicCanvas()
        self._empty = QLabel("Drop an image here\nor click Open Image...")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setObjectName("emptyState")
        self._empty.setStyleSheet(
            "#emptyState { border: 2px dashed palette(mid); border-radius: 12px;"
            " color: palette(placeholder-text); font-size: 16pt; margin: 24px; }"
        )
        self._view = QStackedWidget()
        self._view.addWidget(self._empty)
        self._view.addWidget(self.canvas)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._view, stretch=1)
        layout.addWidget(self._build_panel())

        session.source_changed.connect(self._on_source_changed)

    def _build_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFixedWidth(PANEL_WIDTH)
        panel.setFrameShape(QFrame.Shape.StyledPanel)

        open_button = QPushButton("Open Image...")
        open_button.clicked.connect(self.open_dialog)

        self._name = QLabel("—")
        self._dimensions = QLabel("—")
        self._megapixels = QLabel("—")
        self._file_size = QLabel("—")
        for label in (self._name, self._dimensions, self._megapixels, self._file_size):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._name.setWordWrap(True)

        info = QGroupBox("Image")
        form = QFormLayout(info)
        form.addRow("File:", self._name)
        form.addRow("Dimensions:", self._dimensions)
        form.addRow("Megapixels:", self._megapixels)
        form.addRow("File size:", self._file_size)

        layout = QVBoxLayout(panel)
        layout.addWidget(open_button)
        layout.addWidget(info)
        layout.addStretch()
        return panel

    # Loading

    def open_dialog(self) -> None:
        start_dir = ""
        if self.session.project.source_path:
            start_dir = str(self.session.project.source_path.parent)
        path, _ = QFileDialog.getOpenFileName(self, "Open Source Image", start_dir, IMAGE_FILTER)
        if path:
            self.load_file(Path(path))

    def load_file(self, path: Path) -> bool:
        try:
            image = load_image(path)
        except OSError as exc:
            QMessageBox.warning(self, "Skitter", f"Could not open image:\n{exc}")
            return False
        self.session.set_source(path, image)
        return True

    def _on_source_changed(self) -> None:
        project = self.session.project
        path, image = project.source_path, project.source_image
        h, w = image.shape[:2]

        self._view.setCurrentWidget(self.canvas)
        self.canvas.show_image(image)

        self._name.setText(path.name)
        self._name.setToolTip(str(path))
        self._dimensions.setText(f"{w:,} × {h:,} px")
        self._megapixels.setText(f"{w * h / 1e6:.1f}")
        self._file_size.setText(_format_bytes(path.stat().st_size))

        self.status_message.emit(f"Source image: {path.name}")
        self.completion_changed.emit()

    def is_complete(self) -> bool:
        return self.session.project.has_source

    # Drag and drop

    @staticmethod
    def _dropped_image(event: QDragEnterEvent | QDropEvent) -> Path | None:
        urls = event.mimeData().urls()
        if len(urls) != 1 or not urls[0].isLocalFile():
            return None
        path = Path(urls[0].toLocalFile())
        return path if path.suffix.lower() in IMAGE_EXTENSIONS else None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._dropped_image(event):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        path = self._dropped_image(event)
        if path and self.load_file(path):
            event.acceptProposedAction()

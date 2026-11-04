"""Choosing an image and cropping it to a shape: the Infinite Mosaic's Chosen image start.

The crop box (the Source tab's) is locked to the shape and stays inside the image,
so what comes out always fills it.
"""

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from skitter.core.imaging import IMAGE_EXTENSIONS, load_image
from skitter.ui import preferences
from skitter.ui.style import muted
from skitter.ui.widgets.crop_overlay import CropOverlay
from skitter.ui.widgets.image_viewer import ImageViewer

IMAGE_FILTER = "Images (" + " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS)) + ")"
FOLDER_KEY = "infinite/image_folder"  # preferences: where the last image came from


class CropDialog(QDialog):
    """An image with a crop box of this aspect (w / h), the largest that fits to start."""

    def __init__(self, image: np.ndarray, aspect: float, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Crop Image")
        self.image = image
        self.viewer = ImageViewer()
        self.viewer.show_image(image)
        self.overlay = CropOverlay(self.viewer.canvas)
        height, width = image.shape[:2]
        self.overlay.start(width, height)
        self.overlay.set_aspect(aspect)
        self.overlay.box_changed.connect(self._on_box)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)  # fmt: skip
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.viewer, stretch=1)
        layout.addWidget(muted(QLabel(
            "Drag the box to move it, its corners to resize it; the wheel zooms the view. "
            "What's inside becomes the first mosaic."
        )))  # fmt: skip
        layout.addWidget(self.buttons)
        self.resize(1000, 750)

    def _on_box(self) -> None:
        _, _, width, height = self.overlay.crop_box()
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(width > 0 and height > 0)

    def crop(self) -> np.ndarray:
        left, top, width, height = self.overlay.crop_box()
        return self.image[top : top + height, left : left + width]


def choose_image(parent: QWidget, aspect: float) -> np.ndarray | None:
    """Ask for an image and a crop of it of this aspect: its (H, W, 4) pixels, None if
    cancelled."""
    prefs = preferences.settings()
    path, _ = QFileDialog.getOpenFileName(parent, "Choose an Image",
                                          prefs.value(FOLDER_KEY, ""), IMAGE_FILTER)  # fmt: skip
    if not path:
        return None
    prefs.setValue(FOLDER_KEY, str(Path(path).parent))
    try:
        image = load_image(path)
    except (OSError, ValueError) as exc:
        QMessageBox.warning(parent, "Infinite Mosaic", f"Couldn't open that image: {exc}")
        return None
    dialog = CropDialog(image, aspect, parent)
    return dialog.crop() if dialog.exec() == QDialog.DialogCode.Accepted else None

"""Choosing an image file: the Infinite Mosaic's Choose Image start."""

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from skitter.core.imaging import IMAGE_EXTENSIONS, load_image
from skitter.ui import preferences

IMAGE_FILTER = "Images (" + " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS)) + ")"
FOLDER_KEY = "infinite/image_folder"  # preferences: where the last image came from


def choose_image(parent: QWidget) -> np.ndarray | None:
    """Ask for an image: its (H, W, 4) pixels, None if cancelled or unreadable."""
    prefs = preferences.settings()
    path, _ = QFileDialog.getOpenFileName(parent, "Choose an Image",
                                          prefs.value(FOLDER_KEY, ""), IMAGE_FILTER)  # fmt: skip
    if not path:
        return None
    prefs.setValue(FOLDER_KEY, str(Path(path).parent))
    try:
        return load_image(path)
    except (OSError, ValueError) as exc:
        QMessageBox.warning(parent, "Infinite Mosaic", f"Couldn't open that image: {exc}")
        return None

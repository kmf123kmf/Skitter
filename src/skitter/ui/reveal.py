"""Opening a photo file, or showing it in the system's file manager (Explorer,
Finder, or a folder window).

`file_menu` is the right-click menu of anything that stands for a photo file:
it offers to open the file or show where it is, or says the file is missing.
"""

import os
import subprocess
import sys

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMenu, QWidget

OPEN_TEXT = "Open Image"
REVEAL_TEXT = "Open Image Location"


def open_file(path: str) -> None:
    """Open the file with the system's default app for it."""
    QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.normpath(path)))


def reveal(path: str) -> None:
    """Open the file's folder with the file selected (on Linux: just the folder)."""
    path = os.path.normpath(path)
    if sys.platform == "win32":
        subprocess.Popen(f'explorer /select,"{path}"')
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))


def file_menu(parent: QWidget, path: str) -> QMenu:
    """A menu to open a photo file or show it in the file manager (just a note when the
    file is gone)."""
    menu = QMenu(parent)
    if os.path.isfile(path):
        menu.addAction(OPEN_TEXT, lambda: open_file(path))
        menu.addAction(REVEAL_TEXT, lambda: reveal(path))
    else:
        menu.addAction("Image file not found").setEnabled(False)
    return menu

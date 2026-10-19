"""Showing a file in the system's file manager (Explorer, Finder, or a folder window).

`file_menu` is the right-click menu of anything that stands for a photo file:
it offers to show the file, or says the file is missing.
"""

import os
import subprocess
import sys

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMenu, QWidget

if sys.platform == "win32":
    REVEAL_TEXT = "Show in Explorer"
elif sys.platform == "darwin":
    REVEAL_TEXT = "Show in Finder"
else:
    REVEAL_TEXT = "Show in Folder"


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
    """A menu to show a photo file in the file manager (disabled if the file is gone)."""
    menu = QMenu(parent)
    if os.path.isfile(path):
        menu.addAction(REVEAL_TEXT, lambda: reveal(path))
    else:
        menu.addAction(f"{REVEAL_TEXT} (file not found)").setEnabled(False)
    return menu

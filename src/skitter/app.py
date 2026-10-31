"""Application entry point."""

import argparse
import gc
import sys

from PySide6.QtWidgets import QApplication

from skitter import __version__
from skitter.ui.canvas import configure_opengl
from skitter.ui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    parser = argparse.ArgumentParser(prog="skitter")
    _, qt_args = parser.parse_known_args(argv[1:])  # the rest go to Qt

    configure_opengl()
    app = QApplication([argv[0], *qt_args])
    app.setApplicationName("Skitter")
    app.setApplicationVersion(__version__)

    window = MainWindow()
    window.show()
    window.session.open_library_in_background()  # ready, usually, before it's needed
    # Everything made so far (modules, Qt, numba, faiss...) lives as long as the app: keep
    # it out of the collector's full passes, which otherwise scan it all on the UI thread
    # (measured: 51 ms, a visible stutter mid-animation; 9 ms after freezing).
    gc.collect()
    gc.freeze()
    return app.exec()

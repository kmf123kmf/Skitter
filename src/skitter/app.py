"""Application entry point."""

import argparse
import gc
import logging
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from skitter import __version__
from skitter.core.tiles.library import default_library_folder
from skitter.ui.canvas import configure_opengl
from skitter.ui.main_window import MainWindow


def report_errors(window) -> None:
    """Log warnings and uncaught errors to a file, and show the errors: the built app has
    no console (sys.stderr is None), so otherwise they vanish."""
    log = default_library_folder().parent / "skitter.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)

    def hook(kind, value, tb):
        sys.__excepthook__(kind, value, tb)  # a console, when there is one
        logging.getLogger("skitter").error("uncaught error", exc_info=(kind, value, tb))
        QMessageBox.critical(window, "Skitter", f"Something went wrong: {value}\n\n"
                                                f"The details are in {log}.")  # fmt: skip

    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    parser = argparse.ArgumentParser(prog="skitter")
    _, qt_args = parser.parse_known_args(argv[1:])  # the rest go to Qt

    configure_opengl()
    app = QApplication([argv[0], *qt_args])
    app.setApplicationName("Skitter")
    app.setApplicationVersion(__version__)

    window = MainWindow()
    report_errors(window)
    window.show()
    window.session.open_library_in_background()  # ready, usually, before it's needed
    # Everything made so far (modules, Qt, numba, faiss...) lives as long as the app: keep
    # it out of the collector's full passes, which otherwise scan it all on the UI thread
    # (measured: 51 ms, a visible stutter mid-animation; 9 ms after freezing).
    gc.collect()
    gc.freeze()
    return app.exec()

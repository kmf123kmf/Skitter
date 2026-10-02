"""Application entry point."""

import argparse
import sys

from PySide6.QtWidgets import QApplication

from skitter import __version__
from skitter.ui.canvas import configure_opengl
from skitter.ui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    parser = argparse.ArgumentParser(prog="skitter")
    parser.add_argument("--demo", type=int, metavar="N", help="also open the flying-tiles demo")
    args, qt_args = parser.parse_known_args(argv[1:])

    configure_opengl()
    app = QApplication([argv[0], *qt_args])
    app.setApplicationName("Skitter")
    app.setApplicationVersion(__version__)

    window = MainWindow()
    window.show()
    if args.demo:
        window.open_demo(args.demo)
    return app.exec()

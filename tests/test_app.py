import logging
import sys

from skitter import app
from skitter.core.tiles.library import default_library_folder


def test_uncaught_errors_are_logged_and_shown(qapp, monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # put back afterwards
    shown = []
    monkeypatch.setattr(app.QMessageBox, "critical", lambda *args: shown.append(args))
    root = logging.getLogger()
    before = list(root.handlers)
    app.report_errors(None)
    try:
        try:
            raise RuntimeError("broken slot")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())
    finally:
        for handler in set(root.handlers) - set(before):
            root.removeHandler(handler)
            handler.close()
    assert "broken slot" in shown[0][2]
    log = default_library_folder().parent / "skitter.log"
    assert "RuntimeError: broken slot" in log.read_text(encoding="utf-8")

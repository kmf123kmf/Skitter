import os
import sys
import traceback

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _private_cache_dirs(tmp_path_factory, monkeypatch):
    """Keep tests out of the user's real tile library (default_library_folder)."""
    cache = tmp_path_factory.mktemp("user-cache")
    monkeypatch.setenv("LOCALAPPDATA", str(cache))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    from PySide6.QtCore import QSettings

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(cache))


@pytest.fixture(autouse=True)
def slot_errors(monkeypatch):
    """Fail a test whose Qt slots raised: Qt only prints such errors (through
    sys.excepthook) and carries on, so they would pass unnoticed. A test that expects
    one asks for this fixture by name and empties the list once it has checked it."""
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, value, tb: errors.append((kind, value, tb)))
    yield errors
    if errors:
        report = "\n".join("".join(traceback.format_exception(*e)) for e in errors)
        pytest.fail(f"{len(errors)} error(s) raised in Qt slots:\n{report}", pytrace=False)


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _inline_slicing(monkeypatch):
    """Slice inline, so tests see results at once (test_background_slicing.py covers the
    worker thread: use the `background_slicing` fixture)."""
    from skitter.ui.session import Session

    monkeypatch.setattr(Session, "slice_in_background", False)


@pytest.fixture
def background_slicing(monkeypatch):
    from skitter.ui.session import Session

    monkeypatch.setattr(Session, "slice_in_background", True)

import os

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

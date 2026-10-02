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

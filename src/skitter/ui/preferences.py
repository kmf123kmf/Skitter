"""User preferences, stored in an INI file under the per-user config location."""

from PySide6.QtCore import QSettings


def settings() -> QSettings:
    return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "Skitter", "Skitter")

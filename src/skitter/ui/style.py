"""Small shared looks: muted and warning text, labels that never widen a panel, progress
bars, sizes in bytes."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel, QProgressBar, QSizePolicy

WARNING_STYLE = "color: #c42b1c;"
MUTED_STYLE = "color: palette(placeholder-text);"
# The style draws a chosen tab much like the others: mark it plainly.
TAB_STYLE = (
    "QTabBar::tab { padding: 4px 6px; border-bottom: 2px solid transparent; }"
    "QTabBar::tab:selected { font-weight: bold; border-bottom: 2px solid palette(highlight); }"
)


def muted(label: QLabel) -> QLabel:
    """A label for secondary text: wrapped, in the palette's placeholder color."""
    label.setWordWrap(True)
    label.setStyleSheet(MUTED_STYLE)
    return label


def fixed_width(label: QLabel, lines: int = 0) -> QLabel:
    """A label whose text never widens the panel (and, given lines, never changes its height)."""
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setMinimumWidth(1)
    if lines:
        label.setFixedHeight(lines * QFontMetrics(label.font()).lineSpacing() + 2)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    return label


def format_bytes(n: float) -> str:
    """A size in bytes for people: "512 bytes", "1.5 KB", "2.3 MB", "1.1 GB"."""
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "bytes" else f"{n:,.1f} {unit}"
        n /= 1024
    raise AssertionError("unreachable")


def progress_bar() -> QProgressBar:
    """A bar without the percentage text (a status label says what is going on)."""
    bar = QProgressBar()
    bar.setTextVisible(False)
    return bar


def show_progress(bar: QProgressBar, fraction: float) -> None:
    """Show a job's progress: fraction 0..1, or negative when unknown (busy)."""
    if fraction < 0:
        bar.setRange(0, 0)
    else:
        bar.setRange(0, 1000)
        bar.setValue(round(fraction * 1000))

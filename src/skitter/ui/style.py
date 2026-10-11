"""Small shared looks: muted and warning text, progress bars, sizes in bytes."""

from PySide6.QtWidgets import QLabel, QProgressBar

WARNING_STYLE = "color: #c42b1c;"
MUTED_STYLE = "color: palette(placeholder-text);"


def muted(label: QLabel) -> QLabel:
    """A label for secondary text: wrapped, in the palette's placeholder color."""
    label.setWordWrap(True)
    label.setStyleSheet(MUTED_STYLE)
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

"""Tab framework and Source step behavior (headless; canvases are never shown)."""

import numpy as np
import pytest

from skitter.core.imaging import save_image


@pytest.fixture
def window(qapp, monkeypatch):
    from skitter.ui import main_window
    from skitter.ui.steps.base import StepPage

    class Second(StepPage):
        title = "Second"

    monkeypatch.setattr(main_window, "STEPS", [*main_window.STEPS, Second])
    win = main_window.MainWindow()
    yield win
    win.deleteLater()


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "photo.png"
    save_image(np.full((30, 40, 3), 90, np.uint8), path)
    return path


def test_source_is_first_tab(window):
    assert window.tabs.tabText(0) == "Source"


def test_later_steps_locked_until_source_chosen(window, image_file):
    from skitter.ui.steps.source import SourceStep

    assert window.tabs.isTabEnabled(0)
    assert not window.tabs.isTabEnabled(1)

    assert window.step(SourceStep).load_file(image_file)

    assert window.tabs.isTabEnabled(1)
    assert window.session.project.source_image.shape == (30, 40, 3)
    assert window.windowTitle() == "Skitter — photo.png"


def test_source_panel_shows_image_info(window, image_file):
    from skitter.ui.steps.source import SourceStep

    source = window.step(SourceStep)
    source.load_file(image_file)
    assert source._dimensions.text() == "40 × 30 px"
    assert source._name.text() == "photo.png"


def test_tab_change_calls_enter_and_leave(window, image_file):
    from skitter.ui.steps.source import SourceStep

    calls = []
    first, second = window.steps
    first.on_leave = lambda: calls.append("leave source")
    second.on_enter = lambda: calls.append("enter second")

    window.step(SourceStep).load_file(image_file)
    window.tabs.setCurrentIndex(1)
    assert calls == ["leave source", "enter second"]

"""Tab framework and Source step behavior (headless; canvases are never shown)."""

import numpy as np
import pytest

from skitter.core.edits import Crop
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
    image = np.zeros((30, 40, 3), np.uint8)
    image[:, :20] = 255  # left half white, so flips are detectable
    save_image(image, path)
    return path


@pytest.fixture
def source(window, image_file):
    from skitter.ui.steps.source import SourceStep

    step = window.step(SourceStep)
    step.load_file(image_file)
    return step


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


def test_source_panel_shows_image_info(source):
    assert source._size.text() == "40 × 30 px"
    assert source._name.text() == "photo.png"
    assert source._history.item(0).text() == "No edits"


def test_tab_change_calls_enter_and_leave(window, source):
    calls = []
    first, second = window.steps
    first.on_leave = lambda: calls.append("leave source")
    second.on_enter = lambda: calls.append("enter second")
    window.tabs.setCurrentIndex(1)
    assert calls == ["leave source", "enter second"]


def test_edit_actions_update_image_viewer_and_history(source):
    assert not source.undo_action.isEnabled()

    source.flip_h_action.trigger()
    assert source.session.project.source_image[0, 0, 0] == 0
    assert source.viewer.image is source.session.project.source_image

    source.rotate_right_action.trigger()
    assert source._size.text() == "30 × 40 px"
    assert [source._history.item(i).text() for i in range(2)] == [
        "Flip horizontal",
        "Rotate right",
    ]
    assert source.undo_action.toolTip().startswith("Undo Rotate right")

    source.undo_action.trigger()
    source.undo_action.trigger()
    assert source.session.project.source_image[0, 0, 0] == 255
    assert source.redo_action.isEnabled()


def test_crop_mode_applies_overlay_box(source):
    source.crop_action.trigger()
    assert source.is_cropping()
    assert not source.flip_h_action.isEnabled()
    assert source._crop_fields["w"].value() == 40

    source.crop_overlay.set_box((5, 4, 25, 14))
    assert source._crop_fields["w"].value() == 20

    source.apply_crop()
    assert not source.is_cropping()
    assert source.session.project.source_edits == [Crop(5, 4, 20, 10)]
    assert source.viewer.image.shape == (10, 20, 3)


def test_crop_fields_respect_aspect(source):
    source.crop_action.trigger()
    source._crop_aspect.setCurrentIndex(2)  # Square
    assert source.crop_overlay.crop_box()[2:] == (30, 30)
    source._crop_fields["w"].setValue(12)
    assert source.crop_overlay.crop_box()[2:] == (12, 12)


def test_cancel_crop_leaves_image_unchanged(source):
    source.crop_action.trigger()
    source.crop_overlay.set_box((0, 0, 10, 10))
    source.cancel_crop()
    assert source.session.project.source_edits == []


def test_zoom_helpers():
    from skitter.ui.widgets.image_viewer import format_percent, next_zoom_preset, parse_percent

    assert next_zoom_preset(100, 1) == 150
    assert next_zoom_preset(100, -1) == pytest.approx(200 / 3)
    assert next_zoom_preset(37, 1) == 50
    assert next_zoom_preset(6400, 1) == 6400
    assert parse_percent(" 150 % ") == 150
    assert parse_percent("abc") is None
    assert format_percent(5.25) == "5.2%"
    assert format_percent(66.67) == "67%"

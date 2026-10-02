"""Tab framework and Source step behavior (headless; canvases are never shown)."""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from skitter.core.edits import Crop
from skitter.core.imaging import save_image


@pytest.fixture
def window(qapp):
    from skitter.ui.main_window import MainWindow

    win = MainWindow()
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


def test_tabs_in_workflow_order(window):
    tabs = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert tabs == ["Source", "Slicing", "Tiles", "Matching"]


def test_next_unavailable_until_source_loaded(window, image_file):
    from skitter.ui.steps.source import SourceStep

    assert not window.next_button.isEnabled()
    assert window.next_button.text() == "Next: Slicing"
    assert window.back_button.isHidden()

    window.step(SourceStep).load_file(image_file)
    assert window.next_button.isEnabled()
    assert not window.tabs.isTabEnabled(1)  # loading alone doesn't finish the step
    assert window.windowTitle() == "Skitter — photo.png"


def test_next_commits_source_and_opens_slicing(window, source):
    from skitter.ui.steps.slicing import SlicingStep

    source.flip_h_action.trigger()
    window.next_button.click()

    final = window.session.project.source_final
    assert final[0, 0, 0] == 0  # the flip is baked in
    assert window.tabs.currentWidget() is window.step(SlicingStep)
    assert window.step(SlicingStep).viewer.image is final
    assert window.step(SlicingStep)._mosaic_px.text() == "3,200 × 2,400 px"  # 40 x 80 px tiles
    assert window.next_button.text() == "Next: Tiles"
    assert not window.back_button.isHidden()

    window.back_button.click()
    assert window.tabs.currentWidget() is source


def test_editing_after_commit_locks_slicing_until_next(window, source):
    window.next_button.click()
    window.back_button.click()

    source.rotate_right_action.trigger()
    assert not window.tabs.isTabEnabled(1)
    source.undo_action.trigger()
    assert window.tabs.isTabEnabled(1)  # unchanged from the committed image

    source.rotate_right_action.trigger()
    window.next_button.click()
    assert window.session.project.source_final.shape == (40, 30, 3)
    assert window.tabs.currentIndex() == 1


def test_next_disabled_while_cropping(window, source):
    source.crop_action.trigger()
    assert not window.next_button.isEnabled()
    window.go_next()
    assert window.tabs.currentIndex() == 0
    source.cancel_crop()
    assert window.next_button.isEnabled()


def test_source_panel_shows_image_info(source):
    assert source._size.text() == "40 × 30 px"
    assert source._name.text() == "photo.png"
    assert source._history.item(0).text() == "No edits"


def test_tab_change_calls_enter_and_leave(window, source):
    calls = []
    first, second, *_ = window.steps
    first.on_leave = lambda: calls.append("leave source")
    second.on_enter = lambda: calls.append("enter slicing")
    window.next_button.click()
    assert calls == ["leave source", "enter slicing"]


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


@pytest.fixture
def slicing(window, source, qapp):
    from skitter.ui.steps.slicing import SlicingStep

    window.next_button.click()
    return window.step(SlicingStep)


def flush(qapp):
    qapp.processEvents()  # runs the coalesced recompute timer


def test_slicing_shows_default_grid_of_base_tiles(slicing):
    # 40x30 image, default layout: 40 columns of 80 px square tiles -> 3200 x 2400 canvas.
    regions = slicing.session.project.regions
    assert len(regions) == 40 * 30
    assert slicing.viewer.world_size == (3200, 2400)
    assert slicing._count.text() == "1,200"
    assert slicing._density.text() == "1.00×"
    assert slicing._coverage.text() == "100.0%"
    assert slicing._rows.text() == "30"
    assert slicing.overlay.regions is regions
    assert slicing._stages.item(0).text() == "Grid — base tiles"
    assert "c42b1c" in slicing._source_px.styleSheet()  # 1 source px per tile: warn
    assert slicing.is_complete()


def test_layout_controls_resize_mosaic_and_reslice(slicing, qapp):
    slicing.columns.setValue(10)
    flush(qapp)
    assert slicing.session.project.layout.columns == 10
    assert slicing._mosaic_px.text() == "800 × 600 px"
    assert len(slicing.session.project.regions) == 10 * 8  # 7.5 rows -> 8 with overhang
    assert slicing._rows.text() == "7.50 (8 whole)"
    assert slicing.viewer.world_size == (800, 600)
    assert slicing._source_px.styleSheet() == ""  # 4 source px per tile

    slicing.tile_aspect.setCurrentIndex(1)  # 4:3
    flush(qapp)
    assert slicing._tile_px.text() == "80 × 60 px"
    assert len(slicing.session.project.regions) == 10 * 10

    slicing.tile_width.setValue(40)
    flush(qapp)
    assert slicing._mosaic_px.text() == "400 × 300 px"
    assert slicing.form.editor("cell_size").widget.findChild(QLabel).text() == "40 × 30 px"


def test_param_form_edits_recompute_regions(slicing, qapp):
    columns = slicing.form.editor("columns").widget
    assert not columns.isEnabled()  # count settings inactive in base-tile mode
    slicing.form.editor("mode").widget.setCurrentIndex(1)  # fixed count
    assert columns.isEnabled()
    assert not slicing.form.editor("cell_size").widget.isEnabled()
    columns.setValue(4)
    flush(qapp)
    assert len(slicing.session.project.regions) == 4 * 3
    assert slicing._stages.item(0).text() == "Grid — 4 columns, square cells"

    rows = slicing.form.editor("rows").widget
    assert not rows.isEnabled()  # inactive while square cells is on
    slicing.form.editor("square_cells").widget.setChecked(False)
    assert rows.isEnabled()
    rows.setValue(2)
    flush(qapp)
    assert len(slicing.session.project.regions) == 8


def test_add_reorder_disable_remove_stages(slicing, qapp):
    from skitter.core.slicing.operations import GapAdjust, GridSlicer

    gap_action = next(a for a in slicing.add_menu.actions() if a.text() == "Gap")
    gap_action.trigger()
    flush(qapp)
    assert [type(s.operation) for s in slicing.plan.stages] == [GridSlicer, GapAdjust]
    assert slicing.current_row() == 1
    assert slicing._settings_group.title() == "Gap Settings"
    assert slicing.session.project.regions.size[0, 0] == 78  # 80 px tiles, 2 px gap
    assert float(slicing._coverage.text().rstrip("%")) < 100
    assert "c42b1c" in slicing._coverage.styleSheet()

    slicing.move_stage(-1)
    assert [type(s.operation) for s in slicing.plan.stages] == [GapAdjust, GridSlicer]

    slicing._stages.item(1).setCheckState(Qt.CheckState.Unchecked)  # disable the grid
    flush(qapp)
    assert len(slicing.session.project.regions) == 1

    slicing._stages.setCurrentRow(1)
    slicing.remove_stage()
    flush(qapp)
    assert len(slicing.plan.stages) == 1 and len(slicing.session.project.regions) == 1


def test_slicing_error_is_reported(slicing, qapp):
    slicing.form.editor("mode").widget.setCurrentIndex(1)
    slicing.form.editor("square_cells").widget.setChecked(False)
    slicing.form.editor("columns").widget.setValue(1000)
    slicing.form.editor("rows").widget.setValue(1000)
    flush(qapp)
    assert slicing.session.project.regions is None
    assert "limit" in slicing._error.text()
    assert not slicing._error.isHidden()
    assert not slicing.is_complete()


def test_recommitted_source_keeps_layout_and_plan(window, slicing, qapp):
    from skitter.ui.steps.source import SourceStep

    slicing.columns.setValue(4)
    flush(qapp)
    window.back_button.click()
    window.step(SourceStep).rotate_right_action.trigger()
    window.next_button.click()
    assert slicing.session.project.layout.columns == 4
    assert slicing.columns.value() == 4
    assert slicing._mosaic_px.text() == "320 × 427 px"  # now portrait
    assert len(slicing.session.project.regions) == 4 * 6  # 5.33 rows -> 6


def test_overlay_draws_in_stacking_order():
    from skitter.core.slicing import RegionSet
    from skitter.ui.widgets.region_overlay import regions_to_instances

    regions = RegionSet.from_arrays([[1, 1], [2, 2], [3, 3]], (1, 1), z=[2, 0, 1])
    instances = regions_to_instances(regions, (1, 1, 1))
    assert instances["pos"][:, 0].tolist() == [2, 3, 1]  # bottom first


def test_pile_hover_reports_topmost_region(slicing, qapp):
    from skitter.core.slicing import Stage
    from skitter.core.slicing.operations import PileSlicer

    slicing.plan.stages[:] = [Stage(PileSlicer(rotation=20))]
    slicing._refresh_stages(select=0)
    slicing.session.slicing_edited()
    regions = slicing.session.project.regions
    assert len(regions) > 1200  # denser than the grid of base tiles

    slicing.viewer.canvas.cursor_moved.emit(1600.0, 1200.0)
    expected = regions.hit_test(1600.0, 1200.0)
    assert slicing.overlay.highlighted == expected
    assert f"of {len(regions):,}" in slicing._hover.text()
    slicing.viewer.canvas.cursor_left.emit()
    assert slicing.overlay.highlighted is None


def test_display_modes_control_overlay_and_dimming(slicing):
    from skitter.ui.widgets.region_overlay import OUTLINES

    assert slicing.overlay._layer.project_texture  # stacked by default
    assert slicing.viewer.image_layer.instances["tint"][0, 3] > 0  # uncovered areas dimmed
    slicing.display_mode.setCurrentIndex(1)  # outlines
    assert slicing.overlay.mode == OUTLINES and not slicing.overlay._layer.project_texture
    assert slicing.viewer.image_layer.instances["tint"][0, 3] == 0
    slicing.display_mode.setCurrentIndex(2)  # hidden
    assert not slicing.overlay.visible


def test_line_color_and_opacity_update_overlay_live(slicing, qapp):
    from skitter.ui.widgets.region_overlay import LINE_COLORS

    overlay = slicing.overlay
    index = slicing.line_color.findData("red")
    slicing.line_color.setCurrentIndex(index)
    red = next(rgb for cid, _, rgb in LINE_COLORS if cid == "red")
    assert overlay._layer.instances["tint"][:, :3].tolist()[0] == pytest.approx(red)

    slicing.line_opacity.setValue(40)
    assert overlay.line_alpha == pytest.approx(0.4)
    assert slicing._line_opacity_label.text() == "40%"
    assert overlay._highlight.line_alpha == 1.0

    slicing.columns.setValue(10)  # re-slice: new regions keep the chosen color
    flush(qapp)
    assert overlay._layer.instances["tint"][0, :3].tolist() == pytest.approx(red)


def test_line_style_persists_and_disables_when_hidden(window, slicing):
    from skitter.ui.steps.slicing import SlicingStep

    slicing.line_color.setCurrentIndex(slicing.line_color.findData("lime"))
    slicing.line_opacity.setValue(55)
    other = SlicingStep(slicing.session)
    assert other.line_color.currentData() == "lime"
    assert other.line_opacity.value() == 55
    assert other.overlay.line_alpha == pytest.approx(0.55)

    slicing.display_mode.setCurrentIndex(2)  # hidden
    assert not slicing.line_color.isEnabled() and not slicing.line_opacity.isEnabled()
    slicing.display_mode.setCurrentIndex(1)  # outlines
    assert slicing.line_color.isEnabled() and slicing.line_opacity.isEnabled()

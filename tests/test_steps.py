"""Tab framework and Source step behavior (headless; canvases are never shown)."""

import numpy as np
import pytest
from PySide6.QtCore import Qt

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
    assert tabs == ["Source", "Slicing", "Tiles", "Matching", "Animate"]


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
    assert window.step(SlicingStep)._rows.text() == "30"  # 40 columns over a 4:3 image
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
    assert window.session.project.source_final.shape == (40, 30, 4)
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
    assert source.viewer.image.shape == (10, 20, 4)


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
    # 40x30 image, default layout: 40 columns of square tiles, TILE_UNIT mosaic units each.
    regions = slicing.session.project.regions
    assert len(regions) == 40 * 30
    assert slicing.viewer.world_size == (4000, 3000)
    assert slicing._median.text() == "1.00 × 1.00 tiles"
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
    assert len(slicing.session.project.regions) == 10 * 8  # 7.5 rows -> 8 with overhang
    assert slicing._rows.text() == "7.50 (8 whole)"
    assert slicing.viewer.world_size == (1000, 750)
    assert slicing._source_px.styleSheet() == ""  # 4 source px per tile

    slicing.tile_aspect.setCurrentIndex(1)  # 4:3
    flush(qapp)
    assert len(slicing.session.project.regions) == 10 * 10
    assert slicing._rows.text() == "10"
    assert slicing.viewer.world_size == (1000, 750)  # same canvas, flatter tiles
    assert not hasattr(slicing, "tile_width")  # pixel sizes are chosen on export


def test_param_form_edits_recompute_regions(slicing, qapp):
    from skitter.core.slicing.operations import GridSlicer

    # Grid only scales the Mosaic tile: no counts of its own.
    assert [p.name for p in GridSlicer.params()] == ["cell_size", "anchor"]
    slicing.form.editor("cell_size").widget.setValue(2.0)
    flush(qapp)
    assert len(slicing.session.project.regions) == 20 * 15
    assert slicing._stages.item(0).text() == "Grid — 2× tiles"

    # Split divides each region on its own, keeping the tile shape by default.
    split = next(a for a in slicing.add_menu.actions() if a.text() == "Split")
    split.trigger()
    flush(qapp)
    assert len(slicing.session.project.regions) == 20 * 15 * 2 * 2
    assert slicing._stages.item(1).text() == "Split — 2 across, tile shape"
    down = slicing.form.editor("down").widget
    assert not down.isEnabled()  # chosen from the tile shape
    slicing.form.editor("keep_shape").widget.setChecked(False)
    assert down.isEnabled()
    down.setValue(1)
    flush(qapp)
    assert len(slicing.session.project.regions) == 20 * 15 * 2
    assert slicing._stages.item(1).text() == "Split — 2 × 1"


def test_add_reorder_disable_remove_stages(slicing, qapp):
    from skitter.core.slicing.operations import GridSlicer, JitterAdjust

    assert "Gap" not in [a.text() for a in slicing.add_menu.actions()]
    jitter_action = next(a for a in slicing.add_menu.actions() if a.text() == "Jitter")
    jitter_action.trigger()
    flush(qapp)
    assert [type(s.operation) for s in slicing.plan.stages] == [GridSlicer, JitterAdjust]
    assert slicing.current_row() == 1
    assert slicing._settings_group.title() == "Jitter Settings"
    assert float(slicing._coverage.text().rstrip("%")) < 100
    assert "c42b1c" in slicing._coverage.styleSheet()

    slicing.move_stage(-1)
    assert [type(s.operation) for s in slicing.plan.stages] == [JitterAdjust, GridSlicer]

    slicing._stages.item(1).setCheckState(Qt.CheckState.Unchecked)  # disable the grid
    flush(qapp)
    assert len(slicing.session.project.regions) == 1

    slicing._stages.setCurrentRow(1)
    slicing.remove_stage()
    flush(qapp)
    assert len(slicing.plan.stages) == 1 and len(slicing.session.project.regions) == 1


def test_slicing_error_is_reported(slicing, qapp):
    slicing.form.editor("cell_size").widget.setValue(0.05)  # 800 x 600 cells: over the limit
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
    assert slicing._rows.text() == "5.33 (6 whole)"  # now portrait
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


def test_range_rows_never_widen_a_settings_panel(qapp):
    """Spin boxes ask for room for their widest value ("20.00 turns"); a range row of
    two must not, or it widens the whole form past the side panel (its right edge
    then hides under the scrollbar, as once on the Animate tab)."""
    from PySide6.QtWidgets import QDoubleSpinBox, QLabel

    from skitter.core.slicing import RangeParam
    from skitter.ui.widgets.param_form import RANGE_BOX_MIN, create_editor

    param = RangeParam((0.0, 1.0), "Spin", min=0.0, max=20.0, suffix=" turns")
    param.name = "spin"
    row = create_editor(param).widget
    dash = QLabel("–").minimumSizeHint().width()
    assert row.minimumSizeHint().width() <= 2 * RANGE_BOX_MIN + dash + 2 * 4 + 2
    single = QDoubleSpinBox()
    single.setRange(0.0, 20.0)
    single.setSuffix(" turns")
    assert row.minimumSizeHint().width() < 2 * single.minimumSizeHint().width()


def test_transparent_source_slices_only_the_picture(window, tmp_path):
    from PIL import Image

    from skitter.ui.steps.slicing import SlicingStep
    from skitter.ui.steps.source import SourceStep

    image = np.zeros((40, 40, 4), np.uint8)
    image[..., :3] = 120
    image[:, 20:, 3] = 255  # the right half shows
    path = tmp_path / "half.png"
    Image.fromarray(image, "RGBA").save(path)
    source = window.step(SourceStep)
    source.load_file(path)
    assert source._transparent.text() == "50% of the image"
    window.next_button.click()
    assert window.tabs.currentWidget() is window.step(SlicingStep)
    project = window.session.project
    assert project.source_final.shape == (40, 40, 4)
    x0 = project.regions.center[:, 0] - project.regions.size[:, 0] / 2
    width = window.session.mosaic_size()[0]
    assert len(project.regions) and x0.min() >= width / 2 - 1e-9  # nothing on the hidden half
    assert window.session.slicing_summary.coverage == pytest.approx(1.0)


def test_entirely_transparent_source_cannot_go_on(window, tmp_path):
    from PIL import Image

    from skitter.ui.steps.source import SourceStep

    path = tmp_path / "clear.png"
    Image.fromarray(np.zeros((20, 20, 4), np.uint8), "RGBA").save(path)
    source = window.step(SourceStep)
    source.load_file(path)
    assert source._transparent.text() == "100% of the image"
    messages = []
    source.status_message.connect(messages.append)
    assert not source.advance()
    assert window.session.project.source_final is None
    assert "transparent" in messages[-1]


def test_show_structure_draws_what_contour_rows_follows(window, source):
    from skitter.core.slicing.operations import ContourSlicer
    from skitter.ui.steps.slicing import SlicingStep

    window.next_button.click()
    slicing = window.step(SlicingStep)
    slicing.show_structure.setChecked(True)
    assert not slicing.show_structure.isEnabled() and not slicing.structure.shown  # a grid
    slicing.add_stage(ContourSlicer())  # added after the grid, and selected
    window.session.slicing_edited()
    assert slicing.show_structure.isEnabled() and slicing.structure.shown
    slicing.show_structure.setChecked(False)
    assert not slicing.structure.shown
    slicing.show_structure.setChecked(True)
    assert slicing.structure.shown

    # Only for the selected stage: selecting the grid hides it, though Contour Rows
    # is still in the plan.
    slicing._stages.setCurrentRow(0)
    assert not slicing.structure.shown and not slicing.show_structure.isEnabled()
    slicing._stages.setCurrentRow(1)
    assert slicing.structure.shown

    # A disabled stage isn't doing anything: nothing to show.
    slicing._stages.item(1).setCheckState(Qt.CheckState.Unchecked)
    window.session.slicing_edited()
    assert not slicing.structure.shown and not slicing.show_structure.isEnabled()
    slicing._stages.item(1).setCheckState(Qt.CheckState.Checked)
    window.session.slicing_edited()
    assert slicing.structure.shown

    slicing.remove_stage()
    window.session.slicing_edited()
    assert not slicing.structure.shown and not slicing.show_structure.isEnabled()
    assert slicing.show_structure.isChecked()  # remembered for next time

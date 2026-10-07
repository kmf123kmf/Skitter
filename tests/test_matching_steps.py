"""Tiles and Matching steps, background jobs, and the tile atlas (headless)."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from skitter.core.imaging import save_image
from skitter.ui.render.atlas import PAGE, build_atlas, cell_size


@pytest.fixture
def window(qapp, tmp_path):
    from skitter.ui.main_window import MainWindow

    win = MainWindow()
    win.session.open_library(tmp_path / "cache")  # never the user's real library
    yield win
    win.session.cancel_job()
    win.session.wait_for_job()
    win.session.library.close()
    win.deleteLater()


@pytest.fixture
def photos(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i, color in enumerate(rng.integers(0, 256, (40, 3))):
        size = [(60, 60), (90, 60), (60, 90)][i % 3]
        Image.new("RGB", size, tuple(int(c) for c in color)).save(folder / f"t{i}.png")
    return folder


@pytest.fixture
def sliced(window, tmp_path, qapp):
    """Source loaded and committed, sliced into an 8 x 6 grid."""
    from skitter.ui.steps.slicing import SlicingStep
    from skitter.ui.steps.source import SourceStep

    image = np.zeros((30, 40, 3), np.uint8)
    image[:, 20:] = (200, 50, 50)
    save_image(image, tmp_path / "photo.png")
    window.step(SourceStep).load_file(tmp_path / "photo.png")
    window.next_button.click()
    window.step(SlicingStep).columns.setValue(8)
    qapp.processEvents()
    return window


def tiles_step(window):
    from skitter.ui.steps.tiles import TilesStep

    return window.step(TilesStep)


def matching_step(window):
    from skitter.ui.steps.matching import MatchingStep

    return window.step(MatchingStep)


def build_library(window, photos):
    step = tiles_step(window)
    step.add_folder(photos)
    step.update_library()
    assert window.session.busy == "library"
    window.session.wait_for_job()
    return step


def test_tiles_step_builds_library_in_background(sliced, photos):
    window = sliced
    window.next_button.click()  # to Tiles
    step = tiles_step(window)
    assert window.tabs.currentWidget() is step
    assert not step.is_complete() and not window.next_button.isEnabled()

    build_library(window, photos)
    assert window.session.busy is None
    assert step.is_complete() and window.next_button.isEnabled()
    assert step._tiles.text() == "40" and step.folders.count() == 1
    assert "Added 40" in step.status.text()
    assert window.tabs.isTabEnabled(3)

    step.folders.setCurrentRow(0)
    step.remove_folder()
    assert window.session.library.roots == []


def test_tiles_step_shows_what_the_library_can_paint(sliced, photos):
    from skitter.ui.steps.tiles import COLOR_MAP, ERROR, MAP_CELL, MAP_GAP, PAINTED, SAMPLE

    window = sliced
    window.next_button.click()  # to Tiles
    step = build_library(window, photos)
    step.wait_for_colors(10)
    assert step._colors is not None and len(step._colors) == 40
    assert step._photo_size.text().startswith("60 px")
    assert "portrait" in step._shapes.text() and step._detail.text() == "60 px tiles"

    # Color map: a swatch for every displayable cell, a tile in the cells tiles reach.
    assert step._shown[0] == COLOR_MAP and len(step._layers) == 2
    cmap = step._color_map()
    swatches, tiles = step._layers
    assert len(swatches.instances) == cmap.shown.sum()
    assert len(tiles.instances) == cmap.reached.sum() > 0
    assert "of these colors" in step._summary.text()
    row, column = np.argwhere(cmap.reached)[0]
    x = MAP_CELL * (column + 0.5) + (MAP_GAP if column else 0)
    step.canvas.cursor_moved.emit(float(x), float(MAP_CELL * (row + 0.5)))
    assert "near (closest ΔE" in step._hover.text()
    step.colorfulness.setCurrentIndex(2)  # vivid: a new map
    assert step._color_map().chroma > cmap.chroma

    # The source in the library's colors, and how far off they are.
    step.set_display(PAINTED)
    assert step._shown[0] == PAINTED and len(step._layers) == 1
    texture = step._layers[0].textures
    assert texture.shape[1:3] == step._coverage.error.shape
    assert "Mean ΔE" in step._summary.text()
    step.canvas.cursor_moved.emit(1.0, 1.0)
    assert step._hover.text().startswith("ΔE")
    step.set_display(ERROR)
    assert step._shown[0] == ERROR and not step.colorfulness.isVisibleTo(step)
    step.set_display(SAMPLE)
    assert len(step._layers[0].instances) == 40


def test_matching_step_runs_and_shows_mosaic(sliced, photos, qapp):
    window = sliced
    build_library(window, photos)
    window.next_button.click()  # Tiles
    window.next_button.click()  # Matching
    step = matching_step(window)
    assert window.tabs.currentWidget() is step
    assert step.run_button.isEnabled() and not step.is_complete()

    settings = window.session.project.match_settings
    settings.update(refine_seconds=0.5, adaptive_rounds=1)
    step.run_matching()
    assert window.session.busy == "matching" and not step.run_button.isEnabled()
    window.session.wait_for_job()

    result = window.session.project.matches
    assert result is not None and window.session.matching_is_current and step.is_complete()
    assert step._labels["score"].text().startswith("ΔE")
    assert step._labels["tiles"].text().endswith(f"of {len(result.regions)}")
    assert len(step._tile_layer.instances) == len(result.regions)
    assert step._tile_layer.instances["uv"].max() <= 1.0
    thumbnail_layer = step._tile_layer

    # Full-size crops (the session's shared textures) replace the thumbnails, below the heat map.
    textures = window.session.textures
    assert textures.scene is window.session.scene and len(textures.scene) == len(result.regions)
    textures.wait()
    assert not textures.loading and textures.detail is not None
    assert step._detail.text() == "Full size"
    layer = step._tile_layer
    assert layer is not thumbnail_layer and len(layer.instances) == len(result.regions)
    layers = step.viewer.canvas.layers
    assert layers.index(layer) < layers.index(step._heat_layer)
    uv = layer.instances["uv"]
    texels = np.abs(uv[:, 2:] - uv[:, :2]) * layer.textures.shape[1] + 1  # inset: 1 texel
    np.testing.assert_allclose(texels, np.ceil(result.regions.size), atol=1e-3)
    assert not step._heat_layer.visible
    step.display_mode.setCurrentIndex(1)  # heat map
    assert step._heat_layer.visible and step._tile_layer.visible
    step.display_mode.setCurrentIndex(2)  # source only
    assert not step._tile_layer.visible

    center = result.regions.center[0]
    step.viewer.canvas.cursor_moved.emit(float(center[0]), float(center[1]))
    assert "ΔE" in step._hover.text()

    # Changing a setting makes the result out of date until matching runs again, but
    # the mosaic stays valid: Animate stays open and Next still leads there.
    step.form.editor("max_uses").widget.setValue(1)
    assert not window.session.matching_is_current
    assert "Out of date" in step.status.text()
    assert step.is_complete() and window.next_button.isEnabled()
    assert window.tabs.isTabEnabled(window.tabs.count() - 1)


def test_matching_shows_the_run_in_progress(sliced, photos, qapp):
    import threading

    from PySide6.QtCore import Qt

    from skitter.core.matching.matcher import PreviewStage

    window = sliced
    build_library(window, photos)
    window.show()
    window.tabs.setCurrentWidget(matching_step(window))
    session, step = window.session, matching_step(window)
    session.project.match_settings.update(refine_seconds=0.5, adaptive_rounds=1, max_uses=1)
    step.run_matching()
    session.wait_for_job()
    old = step._tile_layer
    assert old is not None

    # Hold the next run once every region has a tile (in the worker thread).
    release, held = threading.Event(), threading.Event()

    def hold(preview):
        if preview.stage is PreviewStage.COMPLETE:
            held.set()
            release.wait(10)

    session.matching_preview.connect(hold, Qt.ConnectionType.DirectConnection)
    try:
        step.run_matching()
        assert held.wait(10)
        qapp.processEvents()  # the previews so far
        step._builder.wait(10)
        regions = len(session.project.regions)
        # The run replaces the mosaic: its last state, all tiles placed, no sketch left.
        assert step._tile_layer is None and step._heat_layer is None and step._shown is None
        layers = step.viewer.canvas.layers
        assert old not in layers and step._run_tiles in layers and step._run_sketch in layers
        assert len(step._run_tiles.instances) == regions
        assert len(step._run_sketch.instances) == 0
        assert layers.index(step._run_sketch) < layers.index(step._run_tiles)
        assert layers.index(step._run_tiles) < step.overlays.first
        assert step._detail.text().startswith(PreviewStage.COMPLETE.value)
        assert f"{regions:,} of {regions:,}" in step._detail.text()
        assert step._labels["score"].text() == "—"
        step.display_mode.setCurrentIndex(2)  # source only
        assert not step._run_tiles.visible and not step._run_sketch.visible
        step.display_mode.setCurrentIndex(0)
        assert step._run_tiles.visible
        run_layers = step._run_tiles, step._run_sketch
    finally:
        release.set()
        session.wait_for_job()
        session.matching_preview.disconnect(hold)

    # Done: the new mosaic in place of the run.
    assert step._run_tiles is None and step._run_sketch is None
    layers = step.viewer.canvas.layers
    assert step._tile_layer in layers and step._shown is session.scene
    assert not any(layer in layers for layer in run_layers)
    assert step._labels["score"].text().startswith("ΔE")


def test_indexing_shows_its_own_progress(sliced, photos, qapp):
    import threading

    from PySide6.QtCore import Qt

    window = sliced
    build_library(window, photos)
    window.show()
    session, step = window.session, matching_step(window)
    window.tabs.setCurrentWidget(step)
    assert not step.sub_progress.isVisible() and not step.sub_status.isVisible()
    session.project.match_settings.update(refine_seconds=0.2, adaptive_rounds=0)

    release, held = threading.Event(), threading.Event()

    def hold(message, fraction):  # in the worker: stop while describing tile crops
        if not held.is_set() and fraction > 0:
            held.set()
            release.wait(10)

    session.matching_detail.connect(hold, Qt.ConnectionType.DirectConnection)
    try:
        step.run_matching()
        assert held.wait(10)
        qapp.processEvents()
        assert step.status.text().startswith("Indexing tiles")
        assert step.sub_progress.isVisible() and step.sub_status.isVisible()
        assert step.sub_status.text().startswith("Describing tile crops:")
        assert step.sub_progress.value() == 1000  # one chunk: all described
        assert step.sub_progress.height() < step.progress.height()
    finally:
        release.set()
        session.wait_for_job()
        session.matching_detail.disconnect(hold)
    assert session.project.matches is not None
    assert not step.sub_progress.isVisible() and not step.sub_status.isVisible()

    # A cached index isn't built again (nothing to report), but the search still reports.
    seen = []
    session.matching_detail.connect(lambda message, _: seen.append(message))
    step.run_matching()
    session.wait_for_job()
    regions = len(session.project.regions)
    assert not any(m.startswith(("Describing", "Grouping", "Adding")) for m in seen)
    assert f"Searching: {regions:,} of {regions:,} regions" in seen
    assert not step.sub_progress.isVisible()


def test_run_frames_show_target_colors_then_tiles(sliced, photos):
    from skitter.ui.render.match_preview import build_frame

    window = sliced
    build_library(window, photos)
    session = window.session
    session.project.match_settings.update(refine_seconds=0.2, adaptive_rounds=0, max_uses=1)
    previews = []
    session.matching_preview.connect(previews.append)
    session.start_matching()
    session.wait_for_job()
    library, ctx = session.library, session.slice_context
    regions = len(session.project.regions)

    sketch = build_frame(previews[0], library, ctx, None)  # target colors only
    assert sketch.tiles is None and sketch.pages is None and sketch.placed == 0
    instances = sketch.sketch.instances
    assert len(instances) == sketch.needed == regions
    assert sketch.sketch.textures.shape == (1, 1, 1, 4) and (sketch.sketch.textures == 255).all()
    np.testing.assert_allclose(instances["tint"][:, :3], previews[0].target, atol=1e-6)
    assert (instances["tint"][:, 3] == 1).all()

    # Half the regions without a tile yet: they show their target color.
    best = next(p for p in previews if p.result is not None)
    tile = best.result.tile.copy()
    tile[::2] = -1
    assigned = replace(best, result=replace(best.result, tile=tile))
    frame = build_frame(assigned, library, ctx, None)
    waiting = int((assigned.result.tile < 0).sum())
    assert len(frame.sketch.instances) == waiting and frame.placed == regions - waiting
    assert frame.pages is not None and len(frame.tiles) == frame.placed
    # A later frame whose tiles the atlas already holds reuses its pages, uploading nothing.
    again = build_frame(assigned, library, ctx, frame.atlas)
    assert again.pages is None and again.patches == [] and again.atlas is frame.atlas
    np.testing.assert_array_equal(again.tiles, frame.tiles)
    # One with new tiles adds just their cells to the same pages.
    final = build_frame(previews[-1], library, ctx, frame.atlas)
    atlas = frame.atlas
    assert final.pages is None and final.atlas is atlas and final.patches
    shown = np.unique(session.project.matches.tile)
    assert (atlas.cell_of[shown] >= 0).all()
    texels = sum(w * h for _, _, _, w, h in final.patches)
    added = len(np.setdiff1d(shown, assigned.result.tile))
    assert texels == added * atlas.cell**2  # exactly the new cells


def test_growing_atlas_packs_cells_once_and_reports_them():
    from types import SimpleNamespace

    from skitter.ui.render.atlas import PAGE, GrowingAtlas

    thumbs = np.random.default_rng(0).integers(0, 256, (300, 32, 32, 3)).astype(np.uint8)
    library = SimpleNamespace(thumbs=thumbs, thumb_size=np.full((300, 2), 32))
    atlas = GrowingAtlas(library, 200)
    per_row = PAGE // atlas.cell
    assert atlas.add([5, 5, 7]) == [(0, 0, 0, 2 * atlas.cell, atlas.cell)]
    assert atlas.add([7]) == []  # already there
    # Crossing rows: the rest of row 0, whole rows, then the start of a row.
    count = per_row - 2 + 2 * per_row + 3
    rects = atlas.add(np.arange(10, 10 + count))
    c = atlas.cell
    assert rects == [(0, 2 * c, 0, (per_row - 2) * c, c), (0, 0, c, per_row * c, 2 * c),
                     (0, 0, 3 * c, 3 * c, c)]  # fmt: skip
    # Each cell holds its thumbnail, and locate points at it.
    layer, uv = atlas.locate([7], [[0, 0, 1, 1]])
    x, y = c, 0
    np.testing.assert_array_equal(atlas.pages[0, y : y + c, x : x + c, :3], thumbs[7])
    assert layer[0] == 0 and uv[0, 0] * PAGE == x + 0.5
    # Room is whole pages; beyond it nothing is written (the caller starts a new atlas).
    assert atlas.capacity == (PAGE // c) ** 2
    big = SimpleNamespace(thumbs=np.zeros((atlas.capacity + 5, 32, 32, 3), np.uint8),
                          thumb_size=np.full((atlas.capacity + 5, 2), 32))  # fmt: skip
    full = GrowingAtlas(big, 1)
    assert full.add(np.arange(10)) and full.add(np.arange(atlas.capacity + 5)) is None
    assert full.count == 10 and (full.cell_of[10:] < 0).all()


def test_reslicing_clears_the_mosaic(sliced, photos):
    window = sliced
    build_library(window, photos)
    session = window.session
    step = matching_step(window)
    session.project.match_settings.update(refine_seconds=0.5, adaptive_rounds=1)
    step.run_matching()
    session.wait_for_job()
    assert session.project.matches is not None and step._tile_layer is not None

    layout = session.project.layout
    session.set_layout(replace(layout, columns=layout.columns + 2))
    assert session.project.matches is None and not step.is_complete()
    assert step._tile_layer is None and step._heat_layer is None and step._shown is None
    assert step._labels["score"].text() == "—"
    assert "Press Match Tiles" in step.status.text()


def test_reslicing_smaller_fits_the_new_mosaic(sliced, photos):
    # The last mosaic's area must not stay in what the view fits and scrolls over.
    window = sliced
    build_library(window, photos)
    session = window.session
    step = matching_step(window)
    session.project.match_settings.update(refine_seconds=0.2, adaptive_rounds=0)
    step.run_matching()
    session.wait_for_job()
    old = session.mosaic_size()
    assert step.viewer.canvas.bounds == (0.0, 0.0, *old)

    layout = session.project.layout
    session.set_layout(replace(layout, columns=layout.columns - 3))
    new = session.mosaic_size()
    assert new[0] < old[0] and new[1] < old[1]
    assert step.viewer.canvas.bounds == (0.0, 0.0, *new)


def test_export_image_from_the_mosaic_menu(sliced, photos, tmp_path):
    window = sliced
    session = window.session
    button = matching_step(window).export_button
    assert not window.export_action.isEnabled() and not button.isEnabled()
    build_library(window, photos)
    session.project.match_settings.update(refine_seconds=0.5, adaptive_rounds=1)
    session.start_matching()
    session.wait_for_job()
    assert window.export_action.isEnabled() and button.isEnabled()

    button.click()  # the Matching panel's button opens the same window as the menu
    dialog = window.export_dialog
    assert dialog is not None and dialog is window.open_export()
    assert dialog.isVisible() and dialog.export_button.isEnabled()
    target = tmp_path / "out" / "mosaic.png"
    target.parent.mkdir()
    dialog.path.setText(str(target))
    # Pixels are chosen here: 8 columns of 40 px tiles over a 4:3 image.
    dialog.form.editor("tile_px").widget.setValue(40)
    size = (320, 240)
    assert dialog.size_label.text().startswith("320 × 240 px")
    assert dialog.tile_label.text() == "40 px"
    assert dialog.detail_label.text().startswith("Full detail")  # photos are 60-90 px

    # Or by the image's width; tiles follow, and outgrow the 60 px photos.
    assert dialog.form.editor("width_px").widget.value() == 320  # inactive sizes follow
    dialog.form.editor("size_by").widget.setCurrentIndex(1)
    assert dialog.size_label.text().startswith("320 × 240 px")  # switching keeps the size
    dialog.form.editor("width_px").widget.setValue(560)
    assert dialog.size_label.text().startswith("560 × 420 px")
    assert dialog.tile_label.text() == "70 px" and "enlarged" in dialog.detail_label.text()
    dialog.form.editor("size_by").widget.setCurrentIndex(0)
    assert dialog.form.editor("tile_px").widget.value() == 70  # back to tiles, same size
    dialog.form.editor("tile_px").widget.setValue(40)

    dialog.export()
    session.wait_for_job()
    with Image.open(target) as image:
        assert image.format == "PNG" and image.size == size
    assert "Saved mosaic.png" in dialog.status.text()
    assert not (tmp_path / "out" / "mosaic.png.part").exists()

    # Transparent is a background choice for PNG only; JPEG falls back to white.
    settings = session.project.export_settings
    background = dialog.form.editor("background").widget
    transparent = background.findData("transparent")
    assert background.model().item(transparent).isEnabled()
    background.setCurrentIndex(transparent)
    assert settings.alpha

    # Choosing JPEG switches the file extension.
    dialog.form.editor("format").widget.setCurrentIndex(1)
    assert dialog.path.text().endswith("mosaic.jpg")
    assert settings.background == "white" and background.currentData() == "white"
    assert not background.model().item(transparent).isEnabled() and not settings.alpha
    dialog.export()
    session.wait_for_job()
    with Image.open(target.with_suffix(".jpg")) as image:
        assert image.format == "JPEG" and image.size == size

    # Re-slicing drops the mosaic, so there is nothing to export.
    layout = session.project.layout
    session.set_layout(replace(layout, columns=layout.columns + 1))
    assert not window.export_action.isEnabled() and not dialog.export_button.isEnabled()


def test_export_needs_a_valid_mosaic(sliced, photos, tmp_path):
    from skitter.core.edits import FlipHorizontal

    window = sliced
    session = window.session
    build_library(window, photos)
    session.project.match_settings.update(refine_seconds=0.5, adaptive_rounds=1)
    session.start_matching()
    session.wait_for_job()
    action = window.export_action
    assert action.isEnabled()

    # Uncommitted source edits: disabled until undone (or committed, which re-slices).
    session.apply_edit(FlipHorizontal())
    assert not action.isEnabled()
    session.undo()
    assert action.isEnabled()

    # Settings changes leave a valid (if out of date) mosaic.
    session.project.match_settings.update(max_uses=1)
    session.match_settings_edited()
    assert action.isEnabled() and not session.matching_is_current

    # New library photos don't matter; a used tile changing does.
    Image.new("RGB", (50, 50), (1, 2, 3)).save(photos / "new.png")
    session.update_library(workers=0)
    session.wait_for_job()
    assert action.isEnabled()
    used = session.library.paths(session.project.matches.tile[:1])[0]
    Image.new("RGB", (70, 30), (9, 9, 9)).save(used)
    session.update_library(workers=0)
    session.wait_for_job()
    assert not action.isEnabled()

    # Another library: the mosaic is dropped.
    session.open_library(tmp_path / "other")
    assert session.project.matches is None and not action.isEnabled()


def test_animate_tab_plays_the_scene(sliced, photos, qapp):
    from skitter.ui.steps.animate import AnimateStep

    window = sliced
    session = window.session
    animate = window.step(AnimateStep)
    index = window.tabs.indexOf(animate)
    assert window.tabs.tabText(index) == "Animate" and index == len(window.steps) - 1
    assert not window.tabs.isTabEnabled(index) and "match tiles" in animate.status.text()
    build_library(window, photos)
    session.project.match_settings.update(refine_seconds=0.5, adaptive_rounds=1)
    session.start_matching()
    session.wait_for_job()
    assert window.tabs.isTabEnabled(index)

    window.tabs.setCurrentWidget(matching_step(window))
    assert window.next_button.text() == "Next: Animate"
    window.next_button.click()
    assert window.tabs.currentWidget() is animate and window.next_button.isHidden()
    scene, player = session.scene, animate.player
    assert animate.scene is scene and player.timeline is not None

    # It opens on the finished mosaic, exactly as the Matching preview draws it.
    assert player.time == pytest.approx(player.duration) and animate.slider.value() == 1000
    final = session.textures.instances()
    for field in ("pos", "size", "rotation", "layer", "uv", "offset"):
        np.testing.assert_allclose(player.layer.instances[field], final[field], atol=1e-5)
    assert np.all(player.layer.instances["alpha"] == 1)

    # Play starts from the beginning: nothing has landed yet.
    animate.toggle_play()
    assert player.playing and animate.play_button.text() == "Pause"
    assert player.time == 0 and np.all(player.layer.instances["alpha"] == 0)

    # Scrubbing pauses; settings changes replan and keep the moment.
    animate._on_slider(500)
    assert not player.playing and player.time == pytest.approx(4.0)  # half of 8 s
    animate.form.editor("duration").widget.setValue(20)
    assert player.duration == pytest.approx(20) and player.time == pytest.approx(4.0)

    # Leaving the tab pauses playback.
    animate.toggle_play()
    window.go_back()
    assert not player.playing

    # Full-size textures arrive: the layer swaps to them.
    session.textures.wait()
    qapp.processEvents()
    assert player.layer.textures.shape[1] == session.textures.pages.shape[1]

    # Re-slicing drops the mosaic and locks the tab.
    layout = session.project.layout
    session.set_layout(replace(layout, columns=layout.columns + 1))
    assert animate.scene is None and player.layer is None
    assert not window.tabs.isTabEnabled(index)


def test_export_animation_from_the_animate_tab(sliced, photos, tmp_path, qapp):
    import av

    from skitter.ui.steps.animate import AnimateStep

    window = sliced
    session = window.session
    build_library(window, photos)
    session.project.match_settings.update(refine_seconds=0.5, adaptive_rounds=1)
    session.start_matching()
    session.wait_for_job()
    animate = window.step(AnimateStep)
    assert window.video_action.isEnabled() and animate.export_button.isEnabled()

    # The look: camera and light settings reach the player (it draws shadows mid-toss).
    look_names = [p.name for p, _, _ in animate.look_form._rows]
    assert {"camera_height", "light_direction", "shadow_strength"} <= set(look_names)
    animate.look_form.editor("camera_height").widget.setValue(3.0)
    assert animate.player.camera.height == pytest.approx(3.0 * max(
        session.scene.bounds[2] - session.scene.bounds[0],
        session.scene.bounds[3] - session.scene.bounds[1]))  # fmt: skip
    animate.player.seek(animate.player.duration * 0.4)
    assert len(animate.player.shadow_layer.instances) > 0
    animate.player.seek(animate.player.duration)
    assert len(animate.player.shadow_layer.instances) == 0

    # Background: a color, or transparent (shown as a checkerboard).
    look = animate.look_form.editor("background")
    look.set_value("transparent")
    animate.project.animation_look.background = "transparent"
    session.animation_edited()
    assert animate.canvas.checkerboard

    # The export frame outlines what the video shows.
    animate.show_frame.setChecked(True)
    assert len(animate._shade.instances) == 4 and len(animate._frame_line.instances) == 1
    assert frame_aspect(animate) == pytest.approx(1920 / 1080)

    animate.export_button.click()
    dialog = window.video_dialog
    assert dialog is not None and dialog.isVisible()
    form = dialog.form
    form.editor("resolution").widget.setCurrentIndex(
        form.editor("resolution").widget.findData("custom")
    )
    form.editor("width").widget.setValue(160)
    form.editor("height").widget.setValue(120)
    assert frame_aspect(animate) == pytest.approx(160 / 120)  # the frame follows the settings
    form.editor("frame_rate").widget.setCurrentIndex(
        form.editor("frame_rate").widget.findData("custom")
    )
    form.editor("custom_fps").widget.setValue(10)
    form.editor("motion_blur").widget.setCurrentIndex(0)
    session.project.choreography.update(duration=1.0, travel=0.5)
    session.animation_edited()

    # MP4 can't keep transparency: Export stays off and says why.
    target = tmp_path / "build.mp4"
    dialog.path.setText(str(target))
    dialog._refresh()
    assert not dialog.export_button.isEnabled() and "transparent" in dialog.problems.text()

    # WebM keeps it.
    form.editor("format").widget.setCurrentIndex(form.editor("format").widget.findData("webm"))
    assert dialog.path.text().endswith("build.webm")
    dialog._refresh()
    assert dialog.export_button.isEnabled(), dialog.problems.text()
    dialog.export()
    session.wait_for_job()
    out = tmp_path / "build.webm"
    assert out.exists() and not (tmp_path / "build.webm.part").exists()
    assert "Saved build.webm" in dialog.status.text()
    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        assert (stream.width, stream.height) == (160, 120) and stream.average_rate == 10
        assert len(frames) == 1 * 10 + 2 * 10 + 1  # animation + 2 s hold at the end

    # Cancelling leaves nothing behind (a long export, so it can't finish first).
    session.project.video_settings.update(resolution="1080p", hold_end=60.0, motion_blur=16)
    dialog.path.setText(str(tmp_path / "cancelled.webm"))
    dialog._refresh()
    job = session.start_video_export(tmp_path / "cancelled.webm")
    session.cancel_job()
    session.wait_for_job()
    assert not job.running and "cancelled" in dialog.status.text().lower()
    assert not any(p.name.startswith("cancelled") for p in tmp_path.iterdir())


def test_animate_tab_and_export_window_share_video_framing(sliced, photos):
    from skitter.ui.steps.animate import FRAME_SETTINGS, AnimateStep

    window = sliced
    session = window.session
    build_library(window, photos)
    session.project.match_settings.update(refine_seconds=0.5, adaptive_rounds=1)
    session.start_matching()
    session.wait_for_job()
    animate = window.step(AnimateStep)
    tab = animate.video_form
    assert [p.name for p, _, _ in tab._rows] == list(FRAME_SETTINGS)
    animate.show_frame.setChecked(True)

    # Edited on the tab: the export frame follows, and the window opens with it.
    choose(tab.editor("resolution").widget, "vertical")
    assert frame_aspect(animate) == pytest.approx(1080 / 1920)
    shade_before = animate._shade.instances.copy()
    tab.editor("margin").widget.setValue(20.0)
    assert not np.array_equal(animate._shade.instances, shade_before)
    dialog = window.open_video_export()
    assert dialog.form.editor("resolution").widget.currentData() == "vertical"
    assert dialog.form.editor("margin").widget.value() == 20.0
    width, height = dialog.form.editor("width").widget, dialog.form.editor("height").widget
    assert not width.isEnabled() and (width.value(), height.value()) == (1080, 1920)
    assert dialog.size_label.text().startswith("1,080 × 1,920")

    # Edited in the window: the tab shows it too.
    choose(dialog.form.editor("resolution").widget, "custom")
    dialog.form.editor("width").widget.setValue(800)
    dialog.form.editor("height").widget.setValue(800)
    choose(dialog.form.editor("framing").widget, "fill")
    assert tab.editor("resolution").widget.currentData() == "custom"
    assert tab.editor("width").widget.isEnabled() and tab.editor("width").widget.value() == 800
    assert tab.editor("framing").widget.currentData() == "fill"
    assert frame_aspect(animate) == pytest.approx(1.0)
    # Settings only the window has stay out of the tab.
    choose(dialog.form.editor("format").widget, "webm")
    assert "format" not in [p.name for p, _, _ in tab._rows]


def test_back_from_animate_and_next_again(sliced, photos, qapp):
    from skitter.ui.steps.animate import AnimateStep

    window = sliced
    session = window.session
    build_library(window, photos)
    session.project.match_settings.update(refine_seconds=0.5, adaptive_rounds=1)
    session.start_matching()
    session.wait_for_job()
    matching, animate = matching_step(window), window.step(AnimateStep)
    window.tabs.setCurrentWidget(matching)
    assert window.next_button.isEnabled()
    window.next_button.click()
    assert window.tabs.currentWidget() is animate
    qapp.processEvents()

    window.go_back()  # nothing changed
    qapp.processEvents()
    assert window.tabs.currentWidget() is matching
    assert session.matching_is_current, "matching became out of date"
    assert matching.is_complete() and window.next_button.isEnabled()
    assert window.tabs.isTabEnabled(window.tabs.indexOf(animate))
    window.next_button.click()
    assert window.tabs.currentWidget() is animate


def test_wheel_never_changes_an_unfocused_setting(sliced, photos, qapp):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    def wheel(widget):
        center = QPointF(widget.width() / 2, widget.height() / 2)
        qapp.sendEvent(widget, QWheelEvent(
            center, widget.mapToGlobal(center), QPoint(), QPoint(0, -120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        ))  # fmt: skip
        qapp.processEvents()

    window = sliced
    session = window.session
    build_library(window, photos)
    session.project.match_settings.update(refine_seconds=0.5, adaptive_rounds=1)
    session.start_matching()
    session.wait_for_job()
    step = matching_step(window)
    window.tabs.setCurrentWidget(step)
    window.show()
    qapp.processEvents()
    settings = session.project.match_settings
    # Scrolling over the panel's controls (as when scrolling the panel) changes nothing.
    for name in ("max_uses", "tint", "candidates", "crops"):
        before = getattr(settings, name)
        wheel(step.form.editor(name).widget)
        assert getattr(settings, name) == before, name
    assert session.matching_is_current and window.tabs.isTabEnabled(4)
    # Once the user clicks into a control, the wheel adjusts it as usual.
    spin = step.form.editor("max_uses").widget
    spin.setFocus()
    qapp.processEvents()
    if spin.hasFocus():  # focus needs an active window, which offscreen runs may lack
        wheel(spin)
        assert settings.max_uses == 2


def frame_aspect(animate) -> float:
    """Width / height of the export frame outlined on the Animate tab."""
    w, h = animate._frame_line.instances["size"][0]
    return float(w / h)


def choose(combo, value) -> None:
    combo.setCurrentIndex(combo.findData(value))


def matched(window, photos, qapp):
    """The Matching tab showing a finished mosaic at full detail."""
    build_library(window, photos)
    window.next_button.click()  # Tiles
    window.next_button.click()  # Matching
    step = matching_step(window)
    window.session.project.match_settings.update(refine_seconds=0.3, adaptive_rounds=1)
    step.run_matching()
    window.session.wait_for_job()
    window.session.textures.wait()
    qapp.processEvents()
    return step


def click_region(step, r):
    x, y = step.session.project.matches.regions.center[r]
    step.viewer.canvas.clicked.emit(float(x), float(y))


def test_edit_mode_picks_tiles_by_hand(sliced, photos, qapp):
    from skitter.ui.steps.animate import AnimateStep

    window = sliced
    step = matched(window, photos, qapp)
    session = window.session
    result = session.project.matches
    picker, overlays = step.edit.picker, step.overlays
    assert picker.edit_button.isEnabled() and not step.edit.editing
    click_region(step, 0)  # not editing: clicks do nothing
    assert step.edit.selected < 0

    step.edit.set_editing(True)
    assert step.edit.editing and picker.edit_button.isChecked() and step.settings_group.isHidden()
    assert "Click a tile" in picker.hint.text()
    # Hovering an editable tile outlines it.
    x, y = result.regions.center[5]
    step.viewer.canvas.cursor_moved.emit(float(x), float(y))
    assert len(overlays.hover.instances) == 1
    np.testing.assert_allclose(overlays.hover.instances["pos"][0], result.regions.center[5])

    # Selecting a tile lists its remembered candidates, best first, the shown one framed.
    click_region(step, 0)
    assert step.edit.selected == 0 and not picker.grid.isHidden()
    items = picker.grid.items
    refs, _ = result.candidates.ranked(0)
    assert len(items) == len(refs) == 32
    assert [i.rank for i in items[:3]] == ["1", "2", "3"]
    current = [i for i, item in enumerate(items) if item.current]
    assert len(current) == 1 and items[current[0]].auto
    assert all(item.image is not None and item.tooltip for item in items)
    assert len(overlays.select.instances) == 1 and len(overlays.dim.instances) == 1
    tile = int(step.tile_of[0])
    assert (
        overlays.cover.instances["pos"][0].tolist()
        == step._tile_layer.instances["pos"][tile].tolist()
    )
    assert picker.target.pixmap() is not None and not picker.target.pixmap().isNull()

    # Hovering a candidate previews it in place; leaving restores the tile shown now.
    other = next(i for i, item in enumerate(items) if not item.current)
    picker.grid.hovered.emit(other)
    assert step.edit._previewing == other and len(overlays.preview.instances) == 1
    assert "Preview" in picker.info.text()
    picker.grid.hovered.emit(-1)
    assert step.edit._previewing == -1 and len(overlays.preview.instances) == 0

    # Picking it: the mosaic, scene, stats and markers update; the matcher's choice is kept.
    picker.grid.activated.emit(other)
    edited = session.project.matches
    slot = result.candidates.crops(0, [refs[other]])[0][0]
    assert edited is not result and edited.tile[0] == slot and edited.manual[0]
    assert session.scene.slot[tile] == slot and session.scene.result is edited
    assert step._labels["picks"].text() == "1" and len(overlays.markers.instances) == 1
    assert "Picked: ΔE" in picker.info.text() and picker.revert_button.isEnabled()
    assert [i for i, item in enumerate(picker.grid.items) if item.current] == [other]
    assert step.edit.undo_action.isEnabled() and not step.edit.redo_action.isEnabled()
    np.testing.assert_allclose(
        step._tile_layer.instances["uv"], session.textures.instances()["uv"]
    )  # fmt: skip
    # Its full-size crop is read in the background and patched into the pages in place.
    pages = session.textures.pages
    session.textures.wait()
    qapp.processEvents()
    assert session.textures.pages is pages
    uv = step._tile_layer.instances["uv"][tile]
    texels = np.abs(uv[2:] - uv[:2]) * pages.shape[1] + 1
    np.testing.assert_allclose(texels, np.ceil(result.regions.size[0]), atol=1e-3)

    # Undo, redo, revert.
    step.edit.undo_action.trigger()
    assert session.project.matches.tile[0] == result.tile[0]
    assert not session.project.matches.manual[0] and step._labels["picks"].text() == "0"
    step.edit.redo_action.trigger()
    assert session.project.matches.tile[0] == slot and session.manual_picks == 1
    picker.revert_button.click()
    assert session.project.matches.tile[0] == result.tile[0] and session.manual_picks == 0

    # Picking the matcher's own choice is a revert, not a manual pick.
    picker.grid.activated.emit(other)
    auto = next(i for i, item in enumerate(picker.grid.items) if item.auto)
    picker.grid.activated.emit(auto)
    assert session.manual_picks == 0

    # Find more lists the next best tiles.
    picker.more_button.click()
    assert len(picker.grid.items) > 32

    # The edited mosaic is what Animate shows.
    picker.grid.activated.emit(other)
    animate = window.step(AnimateStep)
    window.tabs.setCurrentWidget(animate)
    qapp.processEvents()
    assert animate.scene is session.scene and animate.scene.slot[tile] == slot
    window.tabs.setCurrentWidget(step)

    # Esc: deselect, then leave edit mode.
    step.edit.escape()
    assert step.edit.selected < 0 and step.edit.editing
    step.edit.escape()
    assert not step.edit.editing and not step.settings_group.isHidden()
    assert len(overlays.markers.instances) == 0


def test_double_click_edits_a_tile(sliced, photos, qapp):
    window = sliced
    step = matched(window, photos, qapp)
    canvas = step.viewer.canvas
    view = (canvas.camera.center.copy(), canvas.camera.zoom)
    x, y = step.session.project.matches.regions.center[7]
    sx, sy = canvas.camera.world_to_screen(float(x), float(y))
    canvas.double_clicked.emit(float(sx), float(sy))
    assert step.edit.editing and step.edit.selected == 7
    assert np.allclose(canvas.camera.center, view[0]) and canvas.camera.zoom == view[1]
    # While editing it selects another tile; off the mosaic it does nothing.
    x, y = step.session.project.matches.regions.center[12]
    canvas.double_clicked.emit(*map(float, canvas.camera.world_to_screen(float(x), float(y))))
    assert step.edit.selected == 12
    canvas.double_clicked.emit(-5000.0, -5000.0)
    assert step.edit.editing and step.edit.selected == 12

    # Double-clicking a candidate picks it and leaves edit mode.
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    grid = step.edit.picker.grid
    grid.resize(280, grid.height())
    grid._fit_height()
    other = next(i for i, item in enumerate(grid.items) if not item.current)
    ref = int(step.edit._refs[other])
    QTest.mouseDClick(grid, Qt.MouseButton.LeftButton, pos=grid._cell(other).center().toPoint())
    result = step.session.project.matches
    assert result.choice[12] == ref and result.manual[12]
    assert not step.edit.editing and step.edit.selected < 0
    assert step.session.can_undo_pick


def test_matching_again_asks_whether_to_keep_picks(sliced, photos, qapp):
    window = sliced
    step = matched(window, photos, qapp)
    session = window.session
    step.edit.set_editing(True)
    click_region(step, 3)
    other = next(i for i, item in enumerate(step.edit.picker.grid.items) if not item.current)
    step.edit.picker.grid.activated.emit(other)
    slot = session.project.matches.tile[3]
    asked = []

    def answer(value):
        def ask(count):
            asked.append(count)
            return value

        return ask

    step.ask_keep_picks = answer(None)  # Cancel: nothing happens
    step.run_matching()
    assert asked == [1] and session.busy is None and step.edit.editing

    step.ask_keep_picks = answer(True)
    step.run_matching()
    assert session.busy == "matching" and not step.edit.editing
    session.wait_for_job()
    kept = session.project.matches
    assert kept.tile[3] == slot and kept.manual[3] and session.manual_picks == 1

    step.ask_keep_picks = answer(False)
    step.run_matching()
    session.wait_for_job()
    assert session.manual_picks == 0 and asked == [1, 1, 1]

    step.ask_keep_picks = answer(True)  # no picks: no question
    step.run_matching()
    session.wait_for_job()
    assert asked == [1, 1, 1]


def test_long_file_names_never_resize_the_side_panel(sliced, tmp_path, qapp):
    folder = tmp_path / "long names"
    folder.mkdir()
    rng = np.random.default_rng(1)
    for i, color in enumerate(rng.integers(0, 256, (40, 3))):
        name = f"{i:02d}_" + "a_very_long_file_name_without_any_spaces_" * 4 + ".png"
        Image.new("RGB", (60 + 30 * (i % 2), 60), tuple(int(c) for c in color)).save(folder / name)
    window = sliced
    window.resize(1200, 800)
    window.show()
    step = matched(window, folder, qapp)
    step.edit.set_editing(True)
    click_region(step, 0)
    qapp.processEvents()
    panel = step.edit.picker.parentWidget()
    sizes = set()
    for index in range(len(step.edit.picker.grid.items)):
        step.edit.picker.grid.hovered.emit(index)
        x, y = step.session.project.matches.regions.center[index % 6]
        step.viewer.canvas.cursor_moved.emit(float(x), float(y))
        qapp.processEvents()
        sizes.add((panel.width(), step.edit.picker.width(), step.edit.picker.info.height(),
                   step._hover.width()))  # fmt: skip
        assert "…" in step.edit.picker.info.text() and "…" in step._hover.text()
    assert len(sizes) == 1
    window.hide()


def test_candidate_grid_keyboard_and_painting(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtTest import QTest

    from skitter.ui.widgets.candidate_grid import ACCENT, CandidateGrid, CandidateItem

    grid = CandidateGrid()
    grid.resize(280, 10)
    gray = QImage(8, 8, QImage.Format.Format_RGB888)
    gray.fill(QColor(128, 128, 128))
    items = [CandidateItem(gray, str(i + 1), warning="!" if i % 2 else "") for i in range(6)]
    items[1].current = True
    grid.set_items(items, aspect=1.5)
    assert grid.height() > 2 * grid._cell_size()[1]  # two rows of square cells
    hovered, activated = [], []
    grid.hovered.connect(hovered.append)
    grid.activated.connect(activated.append)
    grid.setFocus()
    QTest.keyClick(grid, Qt.Key.Key_Right)  # from the current one
    QTest.keyClick(grid, Qt.Key.Key_Down)
    assert hovered == [2, 5] and grid.preview_index == 5
    QTest.keyClick(grid, Qt.Key.Key_Return)
    assert activated == [5]
    QTest.keyClick(grid, Qt.Key.Key_Escape)  # stops previewing first
    assert hovered[-1] == -1 and grid.preview_index == -1

    # Frames never fill a cell: the crop shows through (badges draw only in corners).
    image = grid.grab().toImage()
    for index in (1, 3):  # the current one, and one with a warning badge
        center = grid._cell(index).center().toPoint()
        assert image.pixelColor(center) == QColor(128, 128, 128)
    cell = grid._cell(1)  # the current one is framed in the accent color
    assert image.pixelColor(int(cell.left()) + 1, int(cell.center().y())) == ACCENT


def test_matching_can_be_cancelled(sliced, photos):
    window = sliced
    build_library(window, photos)
    session = window.session
    job = session.start_matching()
    session.cancel_job()
    session.wait_for_job()
    assert not job.running and session.busy is None
    assert session.match_error is None


def test_matching_needs_a_library(sliced):
    with pytest.raises(RuntimeError):
        sliced.session.start_matching()


def test_atlas_packs_and_locates_crops():
    thumbs = np.zeros((5, 32, 32, 3), np.uint8)
    for i in range(5):
        thumbs[i, :16, :32] = i * 50  # 32 x 16 thumbnails
    sizes = np.tile([32, 16], (5, 1))
    atlas = build_atlas(thumbs, sizes, [4, 1, 1, 3])
    assert atlas.slots.tolist() == [1, 3, 4] and atlas.cell == 32 == cell_size(3)
    layer, uv = atlas.locate([3, 3], [[0, 0, 1, 1], [0, 0, 0.5, 1]], mirrored=[False, True])
    x, y = atlas.origin[1]
    np.testing.assert_allclose(uv[0] * PAGE, [x + 0.5, y + 0.5, x + 31.5, y + 15.5])
    np.testing.assert_allclose(uv[1] * PAGE, [x + 15.5, y + 0.5, x + 0.5, y + 15.5])  # mirrored
    assert layer.tolist() == [0, 0]
    assert atlas.pages[0, y + 2, x + 2, 0] == 150
    assert cell_size(100_000) == 16 and cell_size(1_000_000) == 8


def test_pack_images_places_every_image_once():
    from skitter.ui.render.atlas import pack_images

    rng = np.random.default_rng(1)
    images = [np.full((h, w, 3), i, np.uint8)
              for i, (w, h) in enumerate(rng.integers(5, 120, (60, 2)))]  # fmt: skip
    atlas = pack_images(images, max_page=256)
    assert atlas.pages.shape[1:] == (256, 256, 4) and len(atlas.pages) > 1
    covered = np.zeros(atlas.pages.shape[:3], np.int64)
    for i, im in enumerate(images):
        (x, y), (w, h), p = atlas.origin[i], atlas.size[i], atlas.page[i]
        assert (w, h) == (im.shape[1], im.shape[0]) and x + w <= 256 and y + h <= 256
        assert (atlas.pages[p, y : y + h, x : x + w, 0] == i).all()
        covered[p, y : y + h, x : x + w] += 1
    assert covered.max() == 1  # no overlaps
    layer, uv = atlas.locate([3], mirrored=[True])
    (x, y), (w, h) = atlas.origin[3], atlas.size[3]
    np.testing.assert_allclose(uv[0] * 256, [x + w - 0.5, y + 0.5, x + 0.5, y + h - 0.5])
    assert layer[0] == atlas.page[3]
    small = pack_images([np.zeros((10, 20, 3), np.uint8)])
    assert small.pages.shape[1] == 64  # a small atlas doesn't take a full page


def test_detail_sizes_keep_full_size_within_budget():
    from skitter.ui.render.tile_textures import detail_sizes

    sizes, scale, boost = detail_sizes([[100, 150], [99.5, 10.2]], budget=10**6)
    assert scale == 1 and boost == 1 and sizes.tolist() == [[100, 150], [100, 11]]
    sizes, scale, _ = detail_sizes(np.full((100, 2), 100.0), budget=250_000)
    assert scale == pytest.approx(0.5) and sizes.tolist() == [[50, 50]] * 100
    sizes, _, _ = detail_sizes([[9000, 3000]], budget=10**9, page=4096)
    assert sizes.tolist() == [[4096, 1366]]


def test_small_tiles_get_as_many_texels_as_a_base_tile():
    from skitter.ui.render.tile_textures import MAX_BOOST, crop_pixels, detail_sizes

    base = (100, 100)
    tiles = [[100, 100], [50, 50], [25, 25], [12.5, 12.5], [200, 200], [50, 100]]
    sizes, scale, boost = detail_sizes(tiles, budget=10**6, base=base)
    assert scale == 1 and boost == MAX_BOOST
    # Up to a base tile's texels (at most MAX_BOOST times the tile); large tiles as before;
    # a tile as long as the base on one side isn't small.
    assert sizes.tolist() == [[100, 100], [100, 100], [100, 100], [50, 50], [200, 200],
                              [50, 100]]  # fmt: skip
    # Picks made later size their crops the same way.
    assert crop_pixels(tiles, scale, boost, base).tolist() == sizes.tolist()
    # Without a base (video export), crops are their plain size.
    plain, _, boost = detail_sizes(tiles, budget=10**6)
    assert boost == 1 and plain.tolist()[1:4] == [[50, 50], [25, 25], [13, 13]]


def test_texture_budget_gives_up_the_boost_before_full_size():
    from skitter.ui.render.tile_textures import detail_sizes

    base, small = (100, 100), np.full((100, 2), 25.0)  # 62,500 texels plain, 1,000,000 boosted
    sizes, scale, boost = detail_sizes(small, budget=250_000, base=base)
    assert scale == 1 and 1 < boost < 4 and np.prod(sizes, axis=1).sum() <= 250_000
    assert sizes[0, 0] == pytest.approx(50, abs=1)  # 250,000 / 100 tiles: 50 x 50 each
    sizes, scale, boost = detail_sizes(small, budget=40_000, base=base)
    assert boost == 1 and scale == pytest.approx(0.8)  # then every crop shrinks, as before


def test_animate_tab_camera_moves_the_frame_and_the_view_follows(sliced, photos):
    from skitter.ui.steps.animate import AnimateStep

    window = sliced
    session = window.session
    build_library(window, photos)
    session.project.match_settings.update(refine_seconds=0.2, adaptive_rounds=0)
    session.start_matching()
    session.wait_for_job()
    window.show()
    window.tabs.setCurrentWidget(window.step(AnimateStep))
    animate = window.step(AnimateStep)
    animate.show_frame.setChecked(True)
    animate.follow.setChecked(True)
    base, _ = animate._frame_rect()
    assert animate.camera_box.currentData() == "static" and animate.shot_rect() == base

    # Pull back: the export frame starts small (close up) and ends on the whole video view.
    animate.camera_box.setCurrentIndex(animate.camera_box.findData("pull_back"))
    assert session.project.camera_move_id == "pull_back" and animate.camera_form._rows
    animate.camera_form.editor("zoom").widget.setValue(4.0)
    assert session.project.camera_move.zoom == 4.0
    animate.player.seek(0.0)
    w = animate._frame_line.instances["size"][0][0]
    assert w == pytest.approx(base[2] / 4) and animate.shot_rect()[2] == pytest.approx(base[2] / 4)
    close_view = animate.canvas.bounds[2]  # following: the view is fitted around the shot
    assert close_view < base[2] / 2
    animate.player.seek(animate.player.duration)
    assert animate.shot_rect() == pytest.approx(base)
    assert animate.canvas.bounds[2] > 3 * close_view

    # Panning by hand stops following; the frame still moves with the camera.
    animate.player.seek(0.0)
    camera = animate.canvas.camera
    animate.canvas.set_view(camera.center, camera.zoom * 1.5)
    assert not animate.follow.isChecked()
    zoom = camera.zoom
    animate.player.seek(animate.player.duration)
    assert camera.zoom == zoom
    assert animate._frame_line.instances["size"][0][0] == pytest.approx(base[2])

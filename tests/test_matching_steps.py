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
    assert step._sample_layer is not None and len(step._sample_layer.instances) == 40
    assert window.tabs.isTabEnabled(3)

    step.folders.setCurrentRow(0)
    step.remove_folder()
    assert window.session.library.roots == []


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


def test_export_image_from_the_mosaic_menu(sliced, photos, tmp_path):
    window = sliced
    session = window.session
    assert not window.export_action.isEnabled()
    build_library(window, photos)
    session.project.match_settings.update(refine_seconds=0.5, adaptive_rounds=1)
    session.start_matching()
    session.wait_for_job()
    assert window.export_action.isEnabled()

    dialog = window.open_export()
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

    sizes, scale = detail_sizes([[100, 150], [99.5, 10.2]], budget=10**6)
    assert scale == 1 and sizes.tolist() == [[100, 150], [100, 11]]
    sizes, scale = detail_sizes(np.full((100, 2), 100.0), budget=250_000)
    assert scale == pytest.approx(0.5) and sizes.tolist() == [[50, 50]] * 100
    sizes, _ = detail_sizes([[9000, 3000]], budget=10**9, page=4096)
    assert sizes.tolist() == [[4096, 1366]]

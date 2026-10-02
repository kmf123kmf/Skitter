"""Tiles and Matching steps, background jobs, and the tile atlas (headless)."""

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

    # Full-size crops replace the thumbnails once read, below the heat map.
    step._detail_job.wait()
    assert step._detail_job is None and step._detail.text() == "Full size"
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

    # Changing a setting makes the result out of date until matching runs again.
    step.form.editor("max_uses").widget.setValue(1)
    assert not window.session.matching_is_current
    assert "Out of date" in step.status.text()


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
    from skitter.ui.steps.matching import detail_sizes

    sizes, scale = detail_sizes([[100, 150], [99.5, 10.2]], budget=10**6)
    assert scale == 1 and sizes.tolist() == [[100, 150], [100, 11]]
    sizes, scale = detail_sizes(np.full((100, 2), 100.0), budget=250_000)
    assert scale == pytest.approx(0.5) and sizes.tolist() == [[50, 50]] * 100
    sizes, _ = detail_sizes([[9000, 3000]], budget=10**9, page=4096)
    assert sizes.tolist() == [[4096, 1366]]

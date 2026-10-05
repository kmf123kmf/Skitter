"""Rendering the finished mosaic at full detail, and saving it."""

import math

import numpy as np
import pytest
from PIL import Image

from skitter.core import assembly
from skitter.core.assembly import (
    ExportCancelled,
    ExportSettings,
    TileFiles,
    check_size,
    enlargement,
    export_frame,
    render_mosaic,
    save_mosaic,
)
from skitter.core.color import oklab_to_rgb8, rgb8_to_oklab
from skitter.core.matching.matcher import MatchResult
from skitter.core.scene import MosaicScene
from skitter.core.slicing import MosaicLayout, RegionSet, SliceContext


def context(width, height):
    """A context whose mosaic is width x height mosaic pixels."""
    return SliceContext(np.zeros((height, width, 3), np.uint8), MosaicLayout(1.0, width), 1.0)


def tile_files(tmp_path, images):
    paths = []
    for i, image in enumerate(images):
        path = tmp_path / f"tile{i}.png"
        Image.fromarray(image).save(path)
        paths.append(str(path))
    n = len(images)
    return TileFiles(np.arange(n), paths, np.zeros((n, 32, 32, 3), np.uint8), np.full((n, 2), 32))


def result_for(regions, tile, mirrored=None, shift=None):
    n = len(regions)
    mean = np.full((n, 3), 0.5, np.float32)
    return MatchResult(
        regions=regions, tile=np.asarray(tile),
        rect=np.tile([0, 0, 1, 1], (n, 1)).astype(np.float32),
        mirrored=np.zeros(n, bool) if mirrored is None else np.asarray(mirrored),
        cost=np.zeros(n, np.float32), tile_mean=mean,
        tint_target=mean if shift is None else mean + np.asarray(shift, np.float32),
        tint=1.0, quality=None, regret=[],
    )  # fmt: skip


def noise(size, seed):
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def render(result, ctx, files, **settings):
    settings.setdefault("tile_px", 1)  # one pixel per mosaic unit (the contexts' base tile)
    scene = MosaicScene.from_result(result, ctx)
    image, report = render_mosaic(scene, files, ExportSettings(**settings), workers=0)
    return image, report


def test_aligned_tiles_are_reproduced_exactly(tmp_path):
    tiles = [noise(20, i) for i in range(4)]
    files = tile_files(tmp_path, tiles)
    regions = RegionSet.grid(40, 40, 2, 2)
    image, report = render(result_for(regions, [0, 1, 2, 3]), context(40, 40), files)
    assert image.shape == (40, 40, 3) and report.tiles == 4 and report.failed == 0
    np.testing.assert_array_equal(image[:20, :20], tiles[0])
    np.testing.assert_array_equal(image[:20, 20:], tiles[1])
    np.testing.assert_array_equal(image[20:, 20:], tiles[3])


def test_mirroring_scale_and_missing_files(tmp_path):
    tile = noise(20, 0)
    files = tile_files(tmp_path, [tile])
    regions = RegionSet.grid(20, 20, 1, 1)
    image, _ = render(result_for(regions, [0], mirrored=[True]), context(20, 20), files)
    np.testing.assert_array_equal(image, tile[:, ::-1])

    image, _ = render(result_for(regions, [0]), context(20, 20), files, tile_px=2)
    assert image.shape == (40, 40, 3)

    files.paths[0] = str(tmp_path / "gone.png")
    image, report = render(result_for(regions, [0]), context(20, 20), files)
    assert report.failed == 1 and image.max() == 0  # the (black) thumbnail stands in


def test_tint_shifts_oklab_like_matching(tmp_path):
    color = np.array([90, 120, 160], np.uint8)
    files = tile_files(tmp_path, [np.tile(color, (16, 16, 1))])
    shift = np.array([0.05, 0.04, -0.02], np.float32)
    regions = RegionSet.grid(16, 16, 1, 1)
    image, _ = render(result_for(regions, [0], shift=[shift]), context(16, 16), files)
    want = oklab_to_rgb8(rgb8_to_oklab(color) + shift)
    assert np.abs(image.astype(int) - want).max() <= 1


def test_seams_and_rotated_edges_are_antialiased(tmp_path):
    files = tile_files(tmp_path, [np.full((10, 10, 3), 200, np.uint8)])
    # Two tiles meeting at x = 10.3: the seam shows no background.
    regions = RegionSet.from_rects([0.3, 10.3], 0, 10, 10)
    image, _ = render(result_for(regions, [0, 0]), context(21, 10), files, background="black")
    assert (image[:, 1:20] == 200).all()
    assert 0 < image[0, 0, 0] < 200  # left edge: 0.7 of the pixel covered
    assert abs(int(image[0, 0, 0]) - 140) <= 13

    # A tile turned 45° on black: soft edges, about the right area.
    regions = RegionSet.from_arrays([[20, 20]], [20, 20], math.pi / 4)
    image, _ = render(result_for(regions, [0]), context(40, 40), files, background="black")
    covered = image[..., 0].astype(float).sum() / 200
    assert covered == pytest.approx(400, rel=0.01)
    assert ((image[..., 0] > 0) & (image[..., 0] < 200)).any()
    assert image[0, 0, 0] == 0 and image[20, 20, 0] == 200


def test_framing_transparency_and_stacking(tmp_path):
    files = tile_files(tmp_path, [np.full((8, 8, 3), v, np.uint8) for v in (50, 250)])
    # A tile hanging past the right edge, and a smaller one on top of it.
    regions = RegionSet.from_rects([0, 2], [0, 2], [12, 4], [8, 4], z=[0, 1])
    result = result_for(regions, [0, 1])
    ctx = context(10, 8)
    assert export_frame(MosaicScene.from_result(result, ctx), ExportSettings(tile_px=1)).size == (
        10,
        8,
    )
    assert export_frame(
        MosaicScene.from_result(result, ctx), ExportSettings(tile_px=1, framing="tiles")
    ).size == (12, 8)
    by_width = ExportSettings(size_by="width", width_px=600, framing="tiles")
    assert export_frame(MosaicScene.from_result(result, ctx), by_width).size == (600, 400)
    by_height = ExportSettings(size_by="height", height_px=80)
    assert export_frame(MosaicScene.from_result(result, ctx), by_height).size == (100, 80)
    assert export_frame(MosaicScene.from_result(result, ctx), ExportSettings(tile_px=30)).size == (
        300,
        240,
    )

    image, _ = render(result, ctx, files)
    assert image[0, 0, 0] == 50 and image[3, 3, 0] == 250
    image, _ = render(result, context(14, 8), files, background="transparent")
    assert image.shape == (8, 14, 4)
    assert image[0, 0, 3] == 255 and image[0, 13, 3] == 0 and image[3, 3, 0] == 250
    # JPEG has no alpha: a transparent background renders white.
    image, _ = render(result, context(14, 8), files, format="jpeg", background="transparent")
    assert image.shape == (8, 14, 3) and image[0, 13, 0] == 255


def test_strips_render_the_same_image(tmp_path, monkeypatch):
    files = tile_files(tmp_path, [noise(30, i) for i in range(3)])
    regions = RegionSet.grid(90, 300, 3, 10).replace(rotation=0.1)
    result = result_for(regions, np.arange(30) % 3)
    whole, _ = render(result, context(90, 300), files)
    monkeypatch.setattr(assembly, "CROP_BUDGET", 3000)
    stripped, _ = render(result, context(90, 300), files)
    np.testing.assert_array_equal(whole, stripped)


def test_enlargement_compares_tile_size_with_photo_pixels():
    regions = RegionSet.from_rects([0, 10], 0, 10, 10)
    result = result_for(regions, [0, 1])
    result.rect[1] = [0.25, 0, 0.75, 1]  # half the photo's width
    scene = MosaicScene.from_result(result, context(20, 10))
    ratio = enlargement(scene, np.array([100, 100]), scale=20.0)  # 200 px tiles
    np.testing.assert_allclose(ratio, [2.0, 4.0])


def test_cancel(tmp_path):
    files = tile_files(tmp_path, [noise(10, 0)])
    regions = RegionSet.grid(10, 10, 1, 1)
    with pytest.raises(ExportCancelled):
        scene = MosaicScene.from_result(result_for(regions, [0]), context(10, 10))
        render_mosaic(scene, files, ExportSettings(),
                      cancelled=lambda: True, workers=0)  # fmt: skip


def test_save_png_and_jpeg(tmp_path):
    image = noise(64, 0)
    save_mosaic(image, tmp_path / "a.png", ExportSettings())
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "a.png")), image)
    settings = ExportSettings(format="jpeg", quality=90)
    save_mosaic(image, tmp_path / "a.jpg", settings)
    with Image.open(tmp_path / "a.jpg") as jpg:
        assert jpg.format == "JPEG" and jpg.size == (64, 64)
    assert check_size((70_000, 10), settings) and not check_size((70_000, 10), ExportSettings())


def test_transparent_source_exports_only_the_picture(tmp_path):
    from skitter.core.slicing import SlicingPlan

    image = np.zeros((8, 20, 4), np.uint8)
    image[:, 10:, 3] = 255  # the left half is hidden
    ctx = SliceContext(image, MosaicLayout(1.0, 20), 1.0)  # 1 x 1 tiles
    regions = SlicingPlan.default().regions(ctx)
    assert len(regions) == 10 * 8
    files = tile_files(tmp_path, [np.full((8, 8, 3), 200, np.uint8)])
    out, _ = render(result_for(regions, [0] * len(regions)), ctx, files, background="transparent")
    assert out.shape == (8, 20, 4)  # the frame is still the whole image
    assert (out[:, :10, 3] == 0).all() and (out[:, 10:, 3] == 255).all()

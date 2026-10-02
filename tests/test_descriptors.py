"""Color conversion, crop candidates, tile and region descriptors, stack rasters."""

import numpy as np
import pytest

from skitter.core.color import LINEAR_LUT, oklab_to_rgb8, rgb8_to_oklab, rgb8_to_oklab_nb
from skitter.core.matching.raster import rasterize
from skitter.core.matching.targets import target_descriptors
from skitter.core.slicing import MosaicLayout, RegionSet, SliceContext
from skitter.core.tiles.crops import aspect_classes, crop_candidates
from skitter.core.tiles.descriptors import (
    DIM,
    LEVEL1,
    LEVEL2,
    MEAN,
    TEXTURE,
    assemble,
    cell_colors,
    dim_weights,
    mirror,
    tile_descriptors,
)

# Color


def test_oklab_reference_points_and_roundtrip():
    np.testing.assert_allclose(rgb8_to_oklab([255, 255, 255]), [1, 0, 0], atol=1e-4)
    np.testing.assert_allclose(rgb8_to_oklab([0, 0, 0]), [0, 0, 0], atol=1e-6)
    colors = np.random.default_rng(0).integers(0, 256, (500, 3), dtype=np.uint8)
    back = oklab_to_rgb8(rgb8_to_oklab(colors)).astype(int)
    assert np.abs(back - colors).max() <= 1


def test_numba_conversion_matches_vectorized():
    for rgb in ([12, 200, 99], [255, 0, 0], [7, 7, 7]):
        np.testing.assert_allclose(
            rgb8_to_oklab_nb(*np.array(rgb, np.uint8), LINEAR_LUT), rgb8_to_oklab(rgb), atol=1e-5
        )


# Crops


def test_aspect_classes_merge_near_equal_shapes():
    classes, index = aspect_classes([1.5, 1.5004, 2 / 3, 0.66668, 1.0])
    assert len(classes) == 3
    assert index[0] == index[1] and index[2] == index[3] and len({*index}) == 3
    np.testing.assert_allclose(sorted(classes), [2 / 3, 1.0, 1.5], rtol=1e-3)


def test_crop_candidates_follow_tile_aspect():
    #            fits     square    long      far too long
    crops = crop_candidates([150, 100, 400, 1000], [100, 100, 100, 100], aspect=1.5)
    assert crops.tile.tolist() == [0, 1, 1, 1, 2, 2, 2]
    np.testing.assert_allclose(crops.rect[0], [0, 0, 1, 1])
    # A square tile in a landscape window: three windows down its height, centered one included.
    square = crops.rect[1:4]
    np.testing.assert_allclose(square[:, [0, 2]], [[0, 1]] * 3)
    np.testing.assert_allclose(square[:, 3] - square[:, 1], 2 / 3, rtol=1e-6)
    np.testing.assert_allclose(square[1, 1], 1 / 6, rtol=1e-6)  # the centered crop
    np.testing.assert_allclose(crops.retained[1], 2 / 3, rtol=1e-6)
    # A long tile: windows across its width.
    np.testing.assert_allclose(crops.rect[4:, 2] - crops.rect[4:, 0], 100 * 1.5 / 400, rtol=1e-6)


def test_crop_count_is_odd_and_capped():
    crops = crop_candidates([290], [100], aspect=1.0, max_crops=4)
    assert len(crops) == 3


# Tile descriptors


def thumb_set(*images):
    """Pack images (h, w, 3) into a (T, 32, 32, 3) thumbnail array with sizes."""
    thumbs = np.zeros((len(images), 32, 32, 3), np.uint8)
    sizes = np.zeros((len(images), 2), np.int64)
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        thumbs[i, :h, :w] = img
        sizes[i] = (w, h)
    return thumbs, sizes


def halves(h=24, w=32):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = 255
    return img


def test_uniform_tile_has_no_structure():
    thumbs, sizes = thumb_set(np.full((20, 30, 3), (200, 40, 90), np.uint8))
    desc = tile_descriptors(thumbs, sizes, [0], [[0, 0, 1, 1]])
    np.testing.assert_allclose(desc[0, MEAN], rgb8_to_oklab([200, 40, 90]), atol=1e-5)
    np.testing.assert_allclose(desc[0, LEVEL1.start :], 0, atol=1e-5)


def test_tile_structure_crops_and_mirror():
    thumbs, sizes = thumb_set(halves(), halves()[:, ::-1])
    full = [[0, 0, 1, 1]]
    left, flipped = (tile_descriptors(thumbs, sizes, [i], full)[0] for i in (0, 1))
    cells = cell_colors(left[None])[0, ..., 0]  # lightness of 4 x 4 cells
    np.testing.assert_allclose(cells[:, :2], 1, atol=1e-4)
    np.testing.assert_allclose(cells[:, 2:], 0, atol=1e-4)
    np.testing.assert_allclose(mirror(left), flipped, atol=1e-5)
    # The right half of the first tile is all black.
    right = tile_descriptors(thumbs, sizes, [0], [[0.5, 0, 1, 1]])[0]
    np.testing.assert_allclose(right[MEAN], 0, atol=1e-5)
    np.testing.assert_allclose(right[LEVEL1.start :], 0, atol=1e-5)


def test_fractional_crop_edges_are_area_weighted():
    # Crop starting mid-pixel: the half-covered white column counts half.
    img = np.zeros((4, 4, 3), np.uint8)
    img[:, 0] = 255
    thumbs, sizes = thumb_set(img)
    desc = tile_descriptors(thumbs, sizes, [0], [[0.125, 0, 0.625, 1]])  # x from 0.5 to 2.5 px
    expected_light = (0.5 * 1 + 1.5 * 0) / 2
    np.testing.assert_allclose(desc[0, 0], expected_light, atol=1e-5)


def test_level2_weights_give_grid_mean_squared_error():
    rng = np.random.default_rng(1)
    a, b = rng.random((2, 1, 4, 4, 3)).astype(np.float32)
    spread = np.zeros((1, 2, 2), np.float32)
    da, db = assemble(a, spread)[0], assemble(b, spread)[0]
    w = np.zeros(DIM)
    w[MEAN] = 1
    w[LEVEL2] = 1 / 16
    cost = (w * (da - db) ** 2).sum()
    assert cost == pytest.approx(((a - b) ** 2).sum(axis=-1).mean(), rel=1e-5)
    assert (dim_weights()[TEXTURE] > 0).all()


# Rasters and targets


def test_raster_paints_in_stacking_order():
    regions = RegionSet.from_rects([0, 5], [0, 0], 10, 10, z=[1, 0])  # first on top
    raster = rasterize(regions, px=1.0)
    assert raster.ids.shape == (10, 15)
    assert raster.lookup(np.array([[7.5, 5.0], [12.0, 5.0], [100.0, 0.0]])).tolist() == [0, 1, -1]


def make_ctx(image, columns, tile=1):
    return SliceContext(image, MosaicLayout(tile_width=tile, columns=columns))


def test_region_descriptor_matches_tile_of_same_picture():
    image = halves(240, 320)
    ctx = make_ctx(image, columns=320)
    target = target_descriptors(ctx.canvas(), ctx)
    thumbs, sizes = thumb_set(halves())
    tile = tile_descriptors(thumbs, sizes, [0], [[0, 0, 1, 1]])
    np.testing.assert_allclose(target.desc[0, : TEXTURE.start], tile[0, : TEXTURE.start], atol=0.02)
    assert target.mask[0].min() == 1 and target.visible[0] == 1


def test_hidden_parts_are_masked():
    ctx = make_ctx(np.zeros((100, 100, 3), np.uint8), columns=100)
    regions = RegionSet.from_rects([0, 50], [0, 0], [100, 50], 100, z=[0, 1])  # right half covered
    target = target_descriptors(regions, ctx, rasterize(regions))
    assert target.visible[0] == pytest.approx(0.5, abs=0.02)
    grid = target.mask[0, LEVEL2].reshape(4, 4, 3)[..., 0]
    np.testing.assert_allclose(grid[:, :2], 1)
    np.testing.assert_allclose(grid[:, 2:], 0)
    assert target.visible[1] == 1


def test_small_regions_match_on_coarse_levels_only():
    ctx = make_ctx(halves(40, 40), columns=40)
    regions = RegionSet.from_rects([0, 0], [0, 20], [6, 40], [6, 20])  # 6 px: cells of 1.5 px
    target = target_descriptors(regions, ctx)
    assert target.mask[0, LEVEL2].max() == 0 and target.mask[0, LEVEL1].min() == 1
    assert (target.desc[0, LEVEL2] == 0).all()
    assert target.mask[1, LEVEL2].min() == 1

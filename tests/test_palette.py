"""What the tile library can paint (core/tiles/palette.py)."""

from types import SimpleNamespace

import numpy as np
import pytest

from skitter.core.color import oklab_to_rgb8, rgb8_to_oklab
from skitter.core.tiles.ingest import THUMB
from skitter.core.tiles.palette import (
    NEAR,
    color_map,
    coverage,
    hue_of,
    photo_stats,
    tile_colors,
)


def solid_library(colors, sizes=None):
    """A stand-in library whose tiles are solid-colored thumbnails."""
    n = len(colors)
    thumbs = np.zeros((n, THUMB, THUMB, 3), np.uint8)
    thumbs[:] = np.asarray(colors, np.uint8)[:, None, None, :]
    sizes = np.full((n, 2), THUMB) if sizes is None else np.asarray(sizes)
    return SimpleNamespace(ids=np.arange(n), thumbs=thumbs, thumb_size=sizes)


def test_tile_colors_are_each_thumbnails_average():
    colors = tile_colors(solid_library([(200, 30, 30), (128, 128, 128)]))
    np.testing.assert_allclose(colors.lab, rgb8_to_oklab([[200, 30, 30], [128, 128, 128]]),
                               atol=2e-3)  # fmt: skip
    distance, index = colors.nearest(rgb8_to_oklab([[190, 40, 40]]))
    assert index.tolist() == [0] and distance[0] < NEAR


def test_color_map_marks_the_colors_no_tile_comes_near():
    red = oklab_to_rgb8([0.6, 0.09 * np.cos(0.5), 0.09 * np.sin(0.5)])  # medium chroma, ~30°
    gray = oklab_to_rgb8([0.6, 0, 0])
    colors = tile_colors(solid_library([red, gray]))
    cmap = color_map(colors, chroma=0.09, hues=36, levels=[0.6, 0.3])
    assert cmap.shape == (2, 37)
    assert cmap.reached[0, 0]  # the gray, at its lightness
    assert cmap.reached[0, 1 + 3] and cmap.near[0, 1 + 3] == 1  # 30°: column 3 of the hues
    assert hue_of(4) == pytest.approx(30)
    assert not cmap.reached[0, 1 + 18]  # 180°: cyan-ish, no tile near
    assert not cmap.reached[1].any()  # nothing that dark
    assert 0 < cmap.share_reached() < 0.2
    vivid = color_map(colors, chroma=0.3)
    assert not vivid.shown.all()  # very saturated colors fall outside sRGB at many hues
    assert not (vivid.reached & ~vivid.shown).any()


def test_coverage_shows_where_the_library_falls_short():
    colors = tile_colors(solid_library([(200, 40, 40)]))
    image = np.zeros((20, 40, 4), np.uint8)
    image[:, :20] = (200, 40, 40, 255)
    image[:, 20:] = (40, 40, 200, 255)
    image[:5, :] = 0  # transparent band: outside the picture
    cov = coverage(image, colors)
    assert cov.painted.shape == (20, 40, 4)
    assert (cov.painted[5:, :, :3] == cov.painted[10, 0, :3]).all()  # all the one red
    assert (cov.painted[:5, :, 3] == 0).all() and np.isnan(cov.error[:5]).all()
    assert cov.error[10, 5] < 1 and cov.error[10, 30] > 25
    assert cov.share_far == pytest.approx(0.5)
    small = coverage(np.zeros((1000, 500, 3), np.uint8), colors, max_side=100)
    assert small.error.shape == (100, 50)


def test_photo_stats_describe_sizes_shapes_and_sharpness():
    width = [3000, 2000, 1000, 1000] + [4000] * 6
    height = [2000, 3000, 1000, 1010] + [3000] * 6
    stats = photo_stats(width, height, tile_aspect=1.0)
    assert stats.count == 10 and stats.median_side == 3000
    assert stats.square == pytest.approx(0.2)  # within 5% of square
    assert stats.portrait == pytest.approx(0.1) and stats.landscape == pytest.approx(0.7)
    assert stats.detail_px == pytest.approx(1000, abs=1)  # 9 in 10 photos fill 1000 px squares
    wide = photo_stats([4000], [3000], tile_aspect=2.0)
    assert wide.detail_px == 4000  # a 2:1 window of a 4:3 photo is its full width
    assert photo_stats([], [], 1.0).count == 0

import math

import numpy as np
import pytest

from skitter.core.dive_motion import (
    LAST_OUT_ZOOM,
    PHOTO_IN,
    DiveMotion,
    approach_zoom,
    fade_out,
    flow_fades,
    glide,
    last_fade,
    moved,
)
from skitter.core.dive_path import FlowPath, View, flow_rate
from skitter.core.dive_styles import STYLES
from skitter.ui.render.sprites import make_instances


def grid(columns: int = 4, tile: float = 100.0) -> np.ndarray:
    """A grid of square tiles' instances, the third turned and tinted."""
    tiles = make_instances(columns * columns)
    x, y = np.meshgrid(np.arange(columns), np.arange(columns))
    tiles["pos"] = np.stack([x.ravel(), y.ravel()], axis=1) * tile + tile / 2
    tiles["size"] = tile
    tiles["rotation"][2] = 0.3
    tiles["offset"][2] = (0.1, 0.0, 0.0)
    return tiles


def test_a_glide_brings_its_point_from_where_it_shows_to_the_middle():
    center, screen0 = np.array([300.0, 200.0]), np.array([150.0, -60.0])
    start, zoom = glide(center, screen0, 0.5, 4.0, 0.0)
    assert zoom == 0.5
    np.testing.assert_allclose((center - start) * zoom, screen0)  # where it showed
    end, zoom = glide(center, screen0, 0.5, 4.0, 1.0)
    assert zoom == pytest.approx(4.0)
    np.testing.assert_allclose(end, center)  # in the middle
    assert glide(center, screen0, 0.5, 4.0, 0.5)[1] == pytest.approx(math.sqrt(0.5 * 4.0))
    # Zoom in to so many tiles across, never out.
    assert approach_zoom(1000.0, 100.0, 4.0, 0.5) == 2.5
    assert approach_zoom(1000.0, 100.0, 4.0, 3.0) == 3.0


@pytest.mark.parametrize("style", list(STYLES))
def test_a_dive_ends_on_its_tile_straightened_and_untinted_under_its_photo(style):
    tiles = grid()
    motion = DiveMotion(tiles, 2, (200.0, 200.0), 2.0, (800, 800), (0.0, 1.0), 8.0,
                        STYLES[style], 7)  # fmt: skip
    first = motion.frame(0.0, tiles.copy())
    np.testing.assert_allclose(first.center, (200.0, 200.0))  # from the view as it was
    assert first.zoom == 2.0 and first.photo_alpha == 0.0
    assert first.photo_rotation == pytest.approx(0.3)
    end = tiles.copy()
    last = motion.frame(1.0, end)
    np.testing.assert_allclose(last.center, tiles["pos"][2])  # onto the tile, filling the view
    assert last.zoom == pytest.approx(8.0) and last.photo_alpha == 1.0
    assert last.photo_rotation == pytest.approx(0.0)
    np.testing.assert_allclose(end["pos"][2], tiles["pos"][2])  # it stays put
    assert end["rotation"][2] == pytest.approx(0.0)
    np.testing.assert_allclose(end["offset"][2], 0.0)
    assert motion.frame(PHOTO_IN, tiles.copy()).photo_alpha == pytest.approx(1.0)


def test_after_a_dive_the_photo_fades_into_the_mosaic():
    assert fade_out(0.0, 0.6) == 1.0 and fade_out(0.3, 0.6) == pytest.approx(0.5)
    assert fade_out(1.0, 0.6) == 0.0 and fade_out(0.0, 0.0) == 0.0


def test_a_leg_fades_the_photo_in_then_into_its_mosaic():
    start = View((2000.0, 1500.0), 0.3)
    path = FlowPath(start, (2050.0, 1525.0), 0.0, 0.3 * 40, (1200, 900), flow_rate(40, 6.0))
    fades = [flow_fades(path, u, 0.35) for u in np.linspace(0, 1, 201)]
    photo, mosaic = (np.array(f) for f in zip(*fades, strict=True))
    assert photo[0] == 0.0 and mosaic[0] == 0.0 and photo[-1] == 1.0 and mosaic[-1] == 1.0
    assert np.all(np.diff(photo) >= 0) and np.all(np.diff(mosaic) >= 0)
    assert np.all(mosaic[photo < 1.0] == 0.0)  # into its mosaic only once all in


def test_the_last_level_fades_by_zoom_or_time_once_covered():
    assert last_fade(1.0, 0.0, 0.5) == 0.0
    assert last_fade(LAST_OUT_ZOOM, 0.0, 0.5) == pytest.approx(1.0)
    assert last_fade(1.0, math.log(LAST_OUT_ZOOM) / 0.5, 0.5) == pytest.approx(1.0)  # by time
    assert last_fade(0.5, 0.0, 0.5) == 0.0  # zoomed out: as at the start


def test_moving_sprites_into_another_world_scales_and_turns_them_about_a_point():
    tiles = grid(2)
    out = moved(tiles, tiles["pos"][0], (0.0, 0.0), 2.0, math.pi / 2)
    np.testing.assert_allclose(out["pos"][0], (0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(out["pos"][1], (0.0, 200.0), atol=1e-6)  # 100 right: 200 down
    np.testing.assert_allclose(out["size"], tiles["size"] * 2)
    np.testing.assert_allclose(out["rotation"], tiles["rotation"] + math.pi / 2)
    np.testing.assert_array_equal(tiles["pos"][0], grid(2)["pos"][0])  # the original stays

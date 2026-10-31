import numpy as np
import pytest

from skitter.core.dive_styles import RANDOM, STYLES, DiveTiles, dive_seed, pick_style

COLUMNS, ROWS, TILE = 12, 8, 100.0
MARGIN = 0.95  # the canvas's fit margin: the end view shows a little around the tile


def grid_tiles(index: int, seed: int = 7, zoomed: float = 1.0) -> DiveTiles:
    """A COLUMNS x ROWS grid, the view showing it all (or zoomed in that much on it)."""
    x, y = np.meshgrid(np.arange(COLUMNS), np.arange(ROWS))
    pos = (np.stack([x.ravel(), y.ravel()], axis=1) + 0.5) * TILE
    n = len(pos)
    view0 = np.array([COLUMNS, ROWS]) * TILE / zoomed
    view1 = np.array([TILE, TILE]) / MARGIN * np.array([COLUMNS / ROWS, 1.0])
    return DiveTiles(pos, np.full((n, 2), TILE), np.zeros(n), np.ones(n), index, view0, view1,
                     seed)  # fmt: skip


@pytest.mark.parametrize("style", list(STYLES))
@pytest.mark.parametrize("index", [0, COLUMNS * ROWS // 2 + 3, COLUMNS * ROWS - 1])
@pytest.mark.parametrize("zoomed", [1.0, 4.0])
def test_every_dive_style_clears_the_view(style, index, zoomed):
    tiles = grid_tiles(index, zoomed=zoomed)
    frames = {e: STYLES[style].frame(tiles, e) for e in (0.0, 0.5, 1.0)}

    start = frames[0.0]  # no jump as the dive starts
    np.testing.assert_allclose(start.pos, tiles.pos, atol=1e-6)
    np.testing.assert_allclose(start.size, tiles.size, atol=1e-6)
    np.testing.assert_allclose(start.alpha, tiles.alpha, atol=1e-6)
    assert start.shade is None or np.allclose(start.shade, 0.0)

    end = frames[1.0]  # every other tile out of the end view, gone, or invisible
    others = np.arange(len(tiles)) != index
    half_view = tiles.view_at(1.0) / 2
    radius = np.linalg.norm(end.size, axis=1) / 2
    outside = np.any(np.abs(end.pos - tiles.center) - radius[:, None] > half_view, axis=1)
    gone = (end.alpha <= 1e-3) | (end.size.min(axis=1) <= 1e-3)
    assert np.all((outside | gone)[others]), f"{style} leaves tiles in view"
    for frame in frames.values():
        assert all(np.all(np.isfinite(a)) for a in (frame.pos, frame.size, frame.rotation))


@pytest.mark.parametrize("style", list(STYLES))
def test_a_dive_style_repeats_with_its_seed(style):
    a = STYLES[style].frame(grid_tiles(5, seed=11), 0.6)
    b = STYLES[style].frame(grid_tiles(5, seed=11), 0.6)
    np.testing.assert_array_equal(a.pos, b.pos)
    np.testing.assert_array_equal(a.alpha, b.alpha)


def test_random_picks_every_style_by_the_dive():
    assert dive_seed(3, 40, 96) == dive_seed(3, 40, 96)  # a loop's replay: the same dive
    picked = {pick_style(RANDOM, dive_seed(i, 40, 96)).id for i in range(200)}
    assert picked == set(STYLES)
    assert pick_style(RANDOM, 123) is pick_style(RANDOM, 123)
    assert pick_style("vortex", 123).id == "vortex"
    assert pick_style("gone", 123) is next(iter(STYLES.values()))  # an old setting's

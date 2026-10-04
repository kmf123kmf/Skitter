"""Perceptual descriptors: coarse color structure as a fixed-length vector.

Tile crops and mosaic regions are described the same way, in OKLab:

- MEAN (3): the average color.
- LEVEL1 (12): a 2 x 2 grid of cell colors minus the average.
- LEVEL2 (48): a 4 x 4 grid of cell colors minus the average.
- TEXTURE (4): the spread (standard deviation) of lightness in each 2 x 2 cell,
  which tells flat cells from busy ones with the same average.

Cells are in the crop's or region's own frame, row by row, channels last.
Keeping the average apart from the structure makes tinting exact: tinting a
tile moves only its average, so the matching cost with tint strength t is

    (1 - t)^2 * |mean difference|^2 + |structure difference|^2

(each term weighted per dimension, see `dim_weights`). With level-2 cell
weights of 1/16 and no other terms, the cost is the mean squared OKLab error
over the 4 x 4 grid.
"""

import numpy as np
from numba import njit, prange

from skitter.core.color import LINEAR_LUT, rgb8_to_oklab_nb

GRID = 4
DIM = 67
MEAN = slice(0, 3)
LEVEL1 = slice(3, 15)
LEVEL2 = slice(15, 63)
TEXTURE = slice(63, 67)

STRUCTURE_CHANNELS = np.array([2.0, 1.0, 1.0])  # lightness carries structure


def assemble(grid4: np.ndarray, lightness_spread: np.ndarray) -> np.ndarray:
    """Descriptors from (N, 4, 4, 3) cell colors and (N, 2, 2) lightness spreads."""
    n = len(grid4)
    mean = grid4.mean(axis=(1, 2))
    grid2 = grid4.reshape(n, 2, 2, 2, 2, 3).mean(axis=(2, 4))
    out = np.empty((n, DIM), np.float32)
    out[:, MEAN] = mean
    out[:, LEVEL1] = (grid2 - mean[:, None, None]).reshape(n, 12)
    out[:, LEVEL2] = (grid4 - mean[:, None, None]).reshape(n, 48)
    out[:, TEXTURE] = lightness_spread.reshape(n, 4)
    return out


def _mirror_permutation() -> np.ndarray:
    perm = np.arange(DIM)
    for level, side in ((LEVEL1, 2), (LEVEL2, 4)):
        cells = np.arange(level.start, level.stop).reshape(side, side, 3)
        perm[level] = cells[:, ::-1].ravel()
    perm[TEXTURE] = np.arange(TEXTURE.start, TEXTURE.stop).reshape(2, 2)[:, ::-1].ravel()
    return perm


MIRROR_PERMUTATION = _mirror_permutation()


def mirror(desc: np.ndarray) -> np.ndarray:
    """Descriptors of the same crops flipped left to right."""
    return desc[..., MIRROR_PERMUTATION]


def cell_colors(desc: np.ndarray) -> np.ndarray:
    """(N, 4, 4, 3) OKLab cell colors recovered from descriptors."""
    desc = np.asarray(desc, dtype=np.float32)
    return desc[:, MEAN][:, None, None] + desc[:, LEVEL2].reshape(-1, GRID, GRID, 3)


def dim_weights(color: float = 1.0, structure: float = 1.0, texture: float = 0.5) -> np.ndarray:
    """Per-dimension weights before tinting.

    Grid cells are weighted by their share of the area, so each level counts
    about as much as the average color at weight 1. The 4 x 4 level counts a
    little less than the 2 x 2 one: coarse structure matters most at mosaic
    viewing distances.
    """
    w = np.zeros(DIM)
    w[MEAN] = color
    w[LEVEL1] = structure / 4 * np.tile(STRUCTURE_CHANNELS, 4)
    w[LEVEL2] = 0.75 * structure / 16 * np.tile(STRUCTURE_CHANNELS, 16)
    w[TEXTURE] = texture / 4
    return w


@njit(parallel=True, cache=True, nogil=True)
def _tile_cells(thumbs, sizes, starts, rects, lut, out_grid, out_spread):
    """Area-averaged 4 x 4 OKLab cells and 2 x 2 lightness spreads per crop.

    Crops of tile t are rects[starts[t]:starts[t + 1]], as fractions of the
    tile's thumbnail, which fills thumbs[t, :h, :w].
    """
    for t in prange(thumbs.shape[0]):
        first, last = starts[t], starts[t + 1]
        if first == last:
            continue
        w, h = sizes[t, 0], sizes[t, 1]
        lab = np.empty((h, w, 3), np.float32)
        for y in range(h):
            for x in range(w):
                px = thumbs[t, y, x]
                lab[y, x, 0], lab[y, x, 1], lab[y, x, 2] = rgb8_to_oklab_nb(
                    px[0], px[1], px[2], lut
                )
        for m in range(first, last):
            x0, y0 = rects[m, 0] * w, rects[m, 1] * h
            cw, ch = (rects[m, 2] - rects[m, 0]) * w / 4, (rects[m, 3] - rects[m, 1]) * h / 4
            s0 = np.zeros((2, 2))
            s1 = np.zeros((2, 2))
            s2 = np.zeros((2, 2))
            for cy in range(4):
                ya, yb = y0 + cy * ch, y0 + (cy + 1) * ch
                for cx in range(4):
                    xa, xb = x0 + cx * cw, x0 + (cx + 1) * cw
                    acc = np.zeros(4)
                    total = 0.0
                    for py in range(int(np.floor(ya)), int(np.ceil(yb))):
                        wy = min(yb, py + 1.0) - max(ya, float(py))
                        if wy <= 0.0:
                            continue
                        sy = min(max(py, 0), h - 1)
                        for px in range(int(np.floor(xa)), int(np.ceil(xb))):
                            wx = min(xb, px + 1.0) - max(xa, float(px))
                            if wx <= 0.0:
                                continue
                            sx = min(max(px, 0), w - 1)
                            wt = wx * wy
                            light = lab[sy, sx, 0]
                            acc[0] += wt * light
                            acc[1] += wt * lab[sy, sx, 1]
                            acc[2] += wt * lab[sy, sx, 2]
                            acc[3] += wt * light * light
                            total += wt
                    for c in range(3):
                        out_grid[m, cy, cx, c] = acc[c] / total
                    s0[cy // 2, cx // 2] += total
                    s1[cy // 2, cx // 2] += acc[0]
                    s2[cy // 2, cx // 2] += acc[3]
            for i in range(2):
                for j in range(2):
                    mean = s1[i, j] / s0[i, j]
                    out_spread[m, i, j] = np.sqrt(max(s2[i, j] / s0[i, j] - mean * mean, 0.0))


def tile_descriptors(thumbs, sizes, tile, rect, *, chunk: int = 32_768,
                     progress=None) -> np.ndarray:  # fmt: skip
    """Descriptors of crop windows of tile thumbnails.

    thumbs: (T, S, S, 3) uint8 (array or memmap), tile t filling [:h, :w].
    sizes: (T, 2) thumbnail (w, h). tile, rect: crops grouped by tile, in
    tile order (see crops.Crops). Thumbnails are read chunk tiles at a time.
    progress(done, total): crops described so far, after each chunk.
    """
    tile = np.asarray(tile, dtype=np.int64)
    rect = np.ascontiguousarray(rect, dtype=np.float32)
    out = np.empty((len(tile), DIM), np.float32)
    if not len(tile):
        return out
    sizes = np.asarray(sizes, dtype=np.int64)
    bounds = np.searchsorted(tile, np.arange(0, len(sizes) + chunk, chunk))
    for lo in range(0, len(sizes), chunk):
        hi = min(lo + chunk, len(sizes))
        first, last = bounds[lo // chunk], bounds[lo // chunk + 1]
        if first == last:
            continue
        starts = np.searchsorted(tile[first:last], np.arange(lo, hi + 1)).astype(np.int64)
        grid = np.empty((last - first, GRID, GRID, 3), np.float32)
        spread = np.empty((last - first, 2, 2), np.float32)
        _tile_cells(
            np.asarray(thumbs[lo:hi]), sizes[lo:hi], starts, rect[first:last], LINEAR_LUT,
            grid, spread,
        )  # fmt: skip
        out[first:last] = assemble(grid, spread)
        if progress is not None:
            progress(int(last), len(tile))
    return out

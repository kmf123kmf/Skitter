"""Thumbnail atlases: many small tile images packed into a few texture pages.

GPUs allow only a couple of thousand layers per texture array, far fewer than
the tiles of a large mosaic, so tile thumbnails are packed into square pages
(cells in a grid), and each sprite shows its own cell through its `uv` rect.
Cells shrink when many tiles are used, to keep GPU memory bounded (each page
is about 16 MB on the GPU).
"""

from dataclasses import dataclass

import numpy as np

PAGE = 2048
CELL_LIMITS = ((32_768, 32), (131_072, 16))  # (most tiles, cell size); beyond: 8 px


@dataclass(frozen=True)
class Atlas:
    pages: np.ndarray  # (P, PAGE, PAGE, 3) uint8
    cell: int  # cell size in texels
    slots: np.ndarray  # (N,) library slots packed, ascending
    page: np.ndarray  # (N,) page of each
    origin: np.ndarray  # (N, 2) texel (x, y) of each cell's corner
    size: np.ndarray  # (N, 2) texels (w, h) the thumbnail fills in its cell

    def locate(self, slots, rects, mirrored=None) -> tuple[np.ndarray, np.ndarray]:
        """Texture layer and (u0, v0, u1, v1) showing each crop rect of each slot.

        rects are fractions of the tile image; mirrored swaps u0 and u1.
        """
        i = np.searchsorted(self.slots, np.asarray(slots, dtype=np.int64))
        rects = np.asarray(rects, dtype=np.float64)
        size = self.size[i].astype(np.float64)
        x0 = self.origin[i, 0] + rects[:, 0] * size[:, 0]
        x1 = self.origin[i, 0] + rects[:, 2] * size[:, 0]
        y0 = self.origin[i, 1] + rects[:, 1] * size[:, 1]
        y1 = self.origin[i, 1] + rects[:, 3] * size[:, 1]
        # Keep half a texel inside, so filtering never reads the neighboring cell.
        inset_x = np.minimum(0.5, (x1 - x0) / 2)
        inset_y = np.minimum(0.5, (y1 - y0) / 2)
        uv = np.stack([x0 + inset_x, y0 + inset_y, x1 - inset_x, y1 - inset_y], axis=1) / PAGE
        if mirrored is not None:
            m = np.asarray(mirrored, bool)
            uv[m] = uv[m][:, [2, 1, 0, 3]]
        return self.page[i].astype(np.float32), uv.astype(np.float32)


def cell_size(count: int) -> int:
    return next((cell for limit, cell in CELL_LIMITS if count <= limit), 8)


def build_atlas(thumbs, thumb_size, slots, chunk: int = 8192) -> Atlas:
    """Pack the thumbnails of the given library slots (thumbs: (S, 32, 32, 3))."""
    slots = np.unique(np.asarray(slots, dtype=np.int64))
    n = len(slots)
    full = thumbs.shape[1]
    cell = min(cell_size(n), full)
    factor = full // cell
    per_row = PAGE // cell
    per_page = per_row * per_row
    pages = np.zeros((max(1, -(-n // per_page)), PAGE, PAGE, 3), np.uint8)
    index = np.arange(n)
    page = index // per_page
    within = index % per_page
    origin = np.stack([within % per_row, within // per_row], axis=1) * cell
    size = np.maximum(1, -(-np.asarray(thumb_size)[slots] // factor))
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        block = np.asarray(thumbs[slots[lo:hi]])
        if factor > 1:
            block = block.reshape(hi - lo, cell, factor, cell, factor, 3).mean(axis=(2, 4))
            block = block.astype(np.uint8)
        for j in range(hi - lo):
            x, y = origin[lo + j]
            pages[page[lo + j], y : y + cell, x : x + cell] = block[j]
    return Atlas(pages, cell, slots, page, origin, size)

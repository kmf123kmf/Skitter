"""Thumbnail atlases: many small tile images packed into a few texture pages.

GPUs allow only a couple of thousand layers per texture array, far fewer than
the tiles of a large mosaic, so tile thumbnails are packed into square pages
(cells in a grid), and each sprite shows its own cell through its `uv` rect.
Cells shrink when many tiles are used, to keep GPU memory bounded (each page
is about 16 MB on the GPU).

`GrowingAtlas` packs thumbnails as they are first needed, into fixed pages
(a matching run's previews). `pack_images` packs images of any sizes instead
(full-detail tile crops).
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
        origin = self.origin[i] + rects[:, :2] * size
        span = (rects[:, 2:] - rects[:, :2]) * size
        return self.page[i].astype(np.float32), _uv(origin, span, PAGE, mirrored)


def _uv(origin, size, page_size, mirrored=None) -> np.ndarray:
    """(u0, v0, u1, v1) of texel rects, half a texel inside so filtering stays in."""
    x0, y0 = origin[:, 0], origin[:, 1]
    x1, y1 = x0 + size[:, 0], y0 + size[:, 1]
    inset_x = np.minimum(0.5, (x1 - x0) / 2)
    inset_y = np.minimum(0.5, (y1 - y0) / 2)
    uv = np.stack([x0 + inset_x, y0 + inset_y, x1 - inset_x, y1 - inset_y], axis=1) / page_size
    if mirrored is not None:
        m = np.asarray(mirrored, bool)
        uv[m] = uv[m][:, [2, 1, 0, 3]]
    return uv.astype(np.float32)


@dataclass(frozen=True)
class PackedAtlas:
    """Images of any sizes packed into square RGBA pages (shelf packing)."""

    pages: np.ndarray  # (P, S, S, 4) uint8
    page: np.ndarray  # (N,) page of each image
    origin: np.ndarray  # (N, 2) texel (x, y) of each image's corner
    size: np.ndarray  # (N, 2) texels (w, h)

    def locate(self, index, mirrored=None) -> tuple[np.ndarray, np.ndarray]:
        """Texture layer and (u0, v0, u1, v1) of the given images; mirrored swaps u0 and u1."""
        i = np.asarray(index, dtype=np.int64)
        uv = _uv(self.origin[i].astype(np.float64), self.size[i], self.pages.shape[1], mirrored)
        return self.page[i].astype(np.float32), uv


def pack_images(images, max_page: int = 4096) -> PackedAtlas:
    """Pack (h, w, 3) uint8 images, each at most max_page on a side.

    Pages are the smallest power of two that holds everything (up to max_page),
    so a few small images don't take a full-size page.
    """
    size = np.array([(im.shape[1], im.shape[0]) for im in images], np.int64).reshape(-1, 2)
    if len(size) and size.max() > max_page:
        raise ValueError(f"image larger than the {max_page} px atlas page")
    need = max(np.sqrt(np.prod(size, axis=1).sum() * 1.15), size.max(initial=1), 64)
    side = min(max_page, 1 << (int(np.ceil(need)) - 1).bit_length())

    order = np.lexsort((-size[:, 0], -size[:, 1]))  # tallest first: shelves waste little
    page = np.zeros(len(size), np.int64)
    origin = np.zeros((len(size), 2), np.int64)
    p = x = y = shelf = 0
    for i in order:
        w, h = size[i]
        if x + w > side:  # next shelf
            x, y, shelf = 0, y + shelf, 0
        if y + h > side:  # next page
            p, x, y, shelf = p + 1, 0, 0, 0
        page[i], origin[i] = p, (x, y)
        x += w
        shelf = max(shelf, h)
    pages = np.zeros((p + 1, side, side, 4), np.uint8)
    pages[..., 3] = 255
    for i, im in enumerate(images):
        (x, y), (w, h) = origin[i], size[i]
        pages[page[i], y : y + h, x : x + w, :3] = im
    return PackedAtlas(pages, page, origin, size)


def shrink(thumbs: np.ndarray, cell: int) -> np.ndarray:
    """(N, S, S, 3) thumbnails averaged down to (N, cell, cell, 3) (cell divides S)."""
    factor = thumbs.shape[1] // cell
    if factor == 1:
        return thumbs
    n = len(thumbs)
    return thumbs.reshape(n, cell, factor, cell, factor, 3).mean(axis=(2, 4)).astype(np.uint8)


def cell_texels(thumb_size, full: int, cell: int) -> np.ndarray:
    """(N, 2) texels each thumbnail fills in its cell (its (w, h) at full size shrunk)."""
    return np.maximum(1, -(-np.asarray(thumb_size) // (full // cell)))


def cell_size(count: int) -> int:
    return next((cell for limit, cell in CELL_LIMITS if count <= limit), 8)


def build_atlas(thumbs, thumb_size, slots, chunk: int = 8192) -> Atlas:
    """Pack the thumbnails of the given library slots (thumbs: (S, 32, 32, 3))."""
    slots = np.unique(np.asarray(slots, dtype=np.int64))
    n = len(slots)
    full = thumbs.shape[1]
    cell = min(cell_size(n), full)
    per_row = PAGE // cell
    per_page = per_row * per_row
    pages = np.zeros((max(1, -(-n // per_page)), PAGE, PAGE, 3), np.uint8)
    index = np.arange(n)
    page = index // per_page
    within = index % per_page
    origin = np.stack([within % per_row, within // per_row], axis=1) * cell
    size = cell_texels(np.asarray(thumb_size)[slots], full, cell)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        block = shrink(np.asarray(thumbs[slots[lo:hi]]), cell)
        for j in range(hi - lo):
            x, y = origin[lo + j]
            pages[page[lo + j], y : y + cell, x : x + cell] = block[j]
    return Atlas(pages, cell, slots, page, origin, size)


class GrowingAtlas:
    """Library thumbnails packed into fixed RGBA pages as they are first needed (cells
    are never reused or moved), for views whose tiles change a few at a time.

    `add` reports the texels it wrote, so a GPU texture needs only those uploaded.
    When the pages are full, the caller starts a new atlas.
    """

    def __init__(self, library, capacity: int):
        self.library = library
        thumbs = library.thumbs
        self.cell = min(cell_size(capacity), thumbs.shape[1])
        self.per_row = PAGE // self.cell
        self.per_page = self.per_row * self.per_row
        pages = max(1, -(-capacity // self.per_page))
        self.capacity = pages * self.per_page
        self.pages = np.zeros((pages, PAGE, PAGE, 4), np.uint8)
        self.pages[..., 3] = 255
        self.cell_of = np.full(len(library.thumb_size), -1, np.int64)  # slot -> cell
        self.count = 0

    def add(self, slots) -> list[tuple] | None:
        """Pack the slots not packed yet. Returns the texels written, as (page, x, y, w,
        h) rects, or None if they don't fit (nothing is written then)."""
        new = np.unique(np.asarray(slots, dtype=np.int64))
        new = new[self.cell_of[new] < 0]
        start = self.count
        if start + len(new) > self.capacity:
            return None
        self.cell_of[new] = np.arange(start, start + len(new))
        self.count += len(new)
        for lo in range(0, len(new), 8192):
            block = shrink(np.asarray(self.library.thumbs[new[lo : lo + 8192]]), self.cell)
            for i, image in enumerate(block):
                page, x, y = self._place(start + lo + i)
                self.pages[page, y : y + self.cell, x : x + self.cell, :3] = image
        return self._rects(start, self.count)

    def locate(self, slots, rects, mirrored=None) -> tuple[np.ndarray, np.ndarray]:
        """Texture layer and (u0, v0, u1, v1) of each crop rect of each (packed) slot."""
        slots = np.asarray(slots, dtype=np.int64)
        page, within = np.divmod(self.cell_of[slots], self.per_page)
        corner = np.stack([within % self.per_row, within // self.per_row], axis=1) * self.cell
        size = cell_texels(self.library.thumb_size[slots], self.library.thumbs.shape[1],
                           self.cell).astype(np.float64)  # fmt: skip
        rects = np.asarray(rects, dtype=np.float64)
        origin = corner + rects[:, :2] * size
        return page.astype(np.float32), _uv(origin, (rects[:, 2:] - rects[:, :2]) * size,
                                            PAGE, mirrored)  # fmt: skip

    def _place(self, cell: int) -> tuple[int, int, int]:
        page, within = divmod(cell, self.per_page)
        return page, (within % self.per_row) * self.cell, (within // self.per_row) * self.cell

    def _rects(self, start: int, end: int) -> list[tuple]:
        """Cells [start, end) as a few rects: per page, a partial first row, whole rows,
        and a partial last row."""
        rects, c, row = [], self.cell, self.per_row
        while start < end:
            page, within = divmod(start, self.per_page)
            stop = min(end, (page + 1) * self.per_page) - page * self.per_page  # in this page
            first_row, col = divmod(within, row)
            if col:  # the rest of a started row
                n = min(row - col, stop - within)
                rects.append((page, col * c, first_row * c, n * c, c))
                within += n
            whole = (stop - within) // row
            if whole:
                rects.append((page, 0, (within // row) * c, row * c, whole * c))
                within += whole * row
            if within < stop:  # the start of a last row
                rects.append((page, 0, (within // row) * c, (stop - within) * c, c))
            start = page * self.per_page + stop
        return rects

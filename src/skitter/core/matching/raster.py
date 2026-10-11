"""Rasterized region stacks: which region is on top at each point.

Overlapping regions hide each other (see slicing/regions.py), so matching
must know which parts of a region stay visible, and the quality estimate
must paint regions in stacking order. Both use a raster of region indices,
painted bottom to top. Animations use the same painting to tell how much of
each tile the finished mosaic shows (`visible_share`).
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from skitter.core.slicing.regions import RegionSet

MAX_RASTER_PIXELS = 16_000_000


@njit(cache=True, nogil=True)
def _paint(ids, order, centers, sizes, rotations, x0, y0, px, painted):
    """Paint regions in order; painted[r] counts the pixels region r covers on its own."""
    height, width = ids.shape
    for r in order:
        cx, cy = centers[r, 0], centers[r, 1]
        hw, hh = sizes[r, 0] / 2, sizes[r, 1] / 2
        c, s = math.cos(rotations[r]), math.sin(rotations[r])
        ex = abs(c) * hw + abs(s) * hh  # bounding box half extents
        ey = abs(s) * hw + abs(c) * hh
        i0 = max(0, int(math.floor((cx - ex - x0) / px)))
        i1 = min(width, int(math.ceil((cx + ex - x0) / px)))
        j0 = max(0, int(math.floor((cy - ey - y0) / px)))
        j1 = min(height, int(math.ceil((cy + ey - y0) / px)))
        for j in range(j0, j1):
            dy = y0 + (j + 0.5) * px - cy
            for i in range(i0, i1):
                dx = x0 + (i + 0.5) * px - cx
                # Into the region's frame (rotate by -rotation).
                if abs(c * dx + s * dy) <= hw and abs(-s * dx + c * dy) <= hh:
                    ids[j, i] = r
                    painted[r] += 1


@dataclass(frozen=True)
class Raster:
    """ids[j, i] is the topmost region at pixel (i, j), or -1 for none."""

    ids: np.ndarray
    origin: tuple[float, float]  # world position of pixel (0, 0)'s corner
    px: float  # mosaic units per raster pixel

    def lookup(self, points: np.ndarray) -> np.ndarray:
        """Topmost region index at (..., 2) world points (-1 outside the raster)."""
        h, w = self.ids.shape
        i = np.floor((points[..., 0] - self.origin[0]) / self.px).astype(np.int64)
        j = np.floor((points[..., 1] - self.origin[1]) / self.px).astype(np.int64)
        inside = (i >= 0) & (i < w) & (j >= 0) & (j < h)
        out = np.full(i.shape, -1, np.int32)
        out[inside] = self.ids[j[inside], i[inside]]
        return out


@dataclass(frozen=True)
class RasterGrid:
    """Where a raster's pixels lie: windows of it paint exactly the same pixels."""

    origin: tuple[float, float]
    px: float
    shape: tuple[int, int]  # (height, width)

    @classmethod
    def of(
        cls,
        regions: RegionSet,
        bounds: tuple[float, float, float, float] | None = None,
        px: float | None = None,
        max_pixels: int = MAX_RASTER_PIXELS,
    ) -> "RasterGrid":
        """The grid `rasterize` uses for these arguments."""
        if bounds is None:
            b = regions.bounds()
            empty = (0, 0, 1, 1)
            bounds = (*b[:, :2].min(axis=0), *b[:, 2:].max(axis=0)) if len(regions) else empty
        x0, y0, x1, y1 = (float(v) for v in bounds)
        area = max(x1 - x0, 1e-9) * max(y1 - y0, 1e-9)
        if px is None:
            smallest = float(regions.size.min()) if len(regions) else 1.0
            px = smallest / 4
        px = max(px, math.sqrt(area / max_pixels), 1e-9)
        width = max(1, math.ceil((x1 - x0) / px))
        height = max(1, math.ceil((y1 - y0) / px))
        return cls((x0, y0), px, (height, width))

    def pixels(self, x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
        """Pixel range (i0, j0, i1, j1) covering a world rect, clipped to the grid."""
        (ox, oy), px, (h, w) = self.origin, self.px, self.shape
        return (
            min(max(int(math.floor((x0 - ox) / px)), 0), w),
            min(max(int(math.floor((y0 - oy) / px)), 0), h),
            min(max(int(math.ceil((x1 - ox) / px)), 0), w),
            min(max(int(math.ceil((y1 - oy) / px)), 0), h),
        )

    def paint(self, regions: RegionSet, i0: int = 0, j0: int = 0, i1=None, j1=None) -> Raster:
        """Pixels [j0:j1, i0:i1] of the raster (all of it by default)."""
        h, w = self.shape
        i1, j1 = w if i1 is None else i1, h if j1 is None else j1
        x0, y0 = self.origin[0] + i0 * self.px, self.origin[1] + j0 * self.px
        ids = np.full((max(j1 - j0, 0), max(i1 - i0, 0)), -1, np.int32)
        if len(regions) and ids.size:
            _paint(
                ids, regions.stacking_order().astype(np.int64), regions.center, regions.size,
                regions.rotation, x0, y0, self.px, np.zeros(len(regions), np.int64),
            )  # fmt: skip
        return Raster(ids, (x0, y0), self.px)


def rasterize(
    regions: RegionSet,
    bounds: tuple[float, float, float, float] | None = None,
    px: float | None = None,
    max_pixels: int = MAX_RASTER_PIXELS,
) -> Raster:
    """Paint regions bottom to top over bounds (x0, y0, x1, y1).

    bounds defaults to all regions' extent. px defaults to a quarter of the
    smallest region side, coarser if needed to stay within max_pixels.
    """
    return RasterGrid.of(regions, bounds, px, max_pixels).paint(regions)


def visible_share(center, size, rotation, max_pixels: int = 4_000_000) -> np.ndarray:
    """(N,) share of each rectangle left visible when all are painted bottom to top
    (array order): 1 uncovered, 0 hidden. Rectangles too small to cover a pixel
    count as visible."""
    center = np.ascontiguousarray(center, np.float64)
    size = np.ascontiguousarray(size, np.float64)
    rotation = np.ascontiguousarray(rotation, np.float64)
    n = len(center)
    if not n:
        return np.zeros(0)
    c, s = np.abs(np.cos(rotation)), np.abs(np.sin(rotation))
    ex = c * size[:, 0] / 2 + s * size[:, 1] / 2
    ey = s * size[:, 0] / 2 + c * size[:, 1] / 2
    x0, y0 = float((center[:, 0] - ex).min()), float((center[:, 1] - ey).min())
    x1, y1 = float((center[:, 0] + ex).max()), float((center[:, 1] + ey).max())
    area = max(x1 - x0, 1e-9) * max(y1 - y0, 1e-9)
    px = max(float(size.min()) / 8, math.sqrt(area / max_pixels), 1e-9)
    ids = np.full((max(1, math.ceil((y1 - y0) / px)), max(1, math.ceil((x1 - x0) / px))), -1,
                  np.int32)  # fmt: skip
    painted = np.zeros(n, np.int64)
    _paint(ids, np.arange(n, dtype=np.int64), center, size, rotation, x0, y0, px, painted)
    shown = np.bincount(ids[ids >= 0], minlength=n)
    return np.where(painted > 0, shown / np.maximum(painted, 1), 1.0)

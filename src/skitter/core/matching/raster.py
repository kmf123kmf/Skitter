"""Rasterized region stacks: which region is on top at each point.

Overlapping regions hide each other (see slicing/regions.py), so matching
must know which parts of a region stay visible, and the quality estimate
must paint regions in stacking order. Both use a raster of region indices,
painted bottom to top.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from skitter.core.slicing.regions import RegionSet

MAX_RASTER_PIXELS = 16_000_000


@njit(cache=True, nogil=True)
def _paint(ids, order, centers, sizes, rotations, x0, y0, px):
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
    if bounds is None:
        b = regions.bounds()
        bounds = (*b[:, :2].min(axis=0), *b[:, 2:].max(axis=0)) if len(regions) else (0, 0, 1, 1)
    x0, y0, x1, y1 = (float(v) for v in bounds)
    area = max(x1 - x0, 1e-9) * max(y1 - y0, 1e-9)
    if px is None:
        smallest = float(regions.size.min()) if len(regions) else 1.0
        px = smallest / 4
    px = max(px, math.sqrt(area / max_pixels), 1e-9)
    width = max(1, math.ceil((x1 - x0) / px))
    height = max(1, math.ceil((y1 - y0) / px))
    ids = np.full((height, width), -1, np.int32)
    if len(regions):
        _paint(
            ids, regions.stacking_order().astype(np.int64), regions.center, regions.size,
            regions.rotation, x0, y0, px,
        )  # fmt: skip
    return Raster(ids, (x0, y0), px)

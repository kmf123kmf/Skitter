"""Crop candidates: windows of each tile image that fit a region shape.

Regions come in a few shapes ("aspect classes"): usually the base tile and
the same tile turned 90°. A tile photo of another aspect must be cropped to
fit. The tile's own aspect decides how: a photo close to the region's shape
gets one centered crop; a photo much longer than the window gets several
crops spread along its long axis (always including the center), since the
best-matching part may be anywhere along it. Photos so elongated that a
crop would discard most of them are left out for that shape.

Each crop records `retained`, the share of the photo it keeps; matching
uses it as a small penalty so photos whose natural shape fits win ties.
"""

import math
from dataclasses import dataclass

import numpy as np


def aspect_classes(aspects, tolerance: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Group aspect ratios (width / height) that differ by less than tolerance.

    Returns (classes, index): each class's representative aspect (geometric
    mean of its members) and the class of every input.
    """
    logs = np.log(np.asarray(aspects, dtype=np.float64))
    bins = np.round(logs / math.log1p(tolerance)).astype(np.int64)
    _, index = np.unique(bins, return_inverse=True)
    sums = np.bincount(index, weights=logs)
    counts = np.bincount(index)
    return np.exp(sums / counts), index


@dataclass(frozen=True)
class Crops:
    """Crop windows for one aspect class, grouped by tile in tile order."""

    tile: np.ndarray  # (M,) tile index of each crop
    rect: np.ndarray  # (M, 4) float32 (x0, y0, x1, y1) as fractions of the tile image
    retained: np.ndarray  # (M,) float32 share of the photo inside the window

    def __len__(self) -> int:
        return len(self.tile)


def crop_candidates(
    widths,
    heights,
    aspect: float,
    *,
    max_crops: int = 3,
    min_slack: float = 1.15,
    max_slack: float = 3.0,
    step: float = 0.35,
) -> Crops:
    """Crop windows of the given aspect for tiles of the given pixel sizes.

    slack is how much longer a tile is than the window along one axis. Below
    min_slack a tile gets one centered crop; above it, one more crop per
    `step` of slack, up to max_crops (an odd count, so the center is always
    included). Tiles with slack above max_slack get none.
    """
    widths = np.asarray(widths, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    ratio = (widths / heights) / aspect
    wide = ratio >= 1.0
    slack = np.where(wide, ratio, 1.0 / ratio)

    max_crops = max(1, max_crops - (1 - max_crops % 2))  # largest odd count allowed
    count = np.minimum(max_crops, 1 + np.ceil((slack - 1.0) / step - 1e-9)).astype(np.int64)
    count = np.where(slack < min_slack, 1, count)
    count -= 1 - count % 2  # odd, so the centered crop is included
    count = np.where(slack > max_slack, 0, count)

    tile = np.repeat(np.arange(len(widths)), count)
    first = np.repeat(np.cumsum(count) - count, count)
    position = np.arange(len(tile)) - first  # 0 .. count-1 within each tile
    n = count[tile]
    along = np.where(n > 1, position / np.maximum(n - 1, 1), 0.5)  # 0..1 along the long axis

    span = 1.0 / slack[tile]  # window length as a fraction of the long axis
    start = along * (1.0 - span)
    rect = np.empty((len(tile), 4), np.float32)
    w = wide[tile]
    rect[:, 0] = np.where(w, start, 0.0)
    rect[:, 1] = np.where(w, 0.0, start)
    rect[:, 2] = np.where(w, start + span, 1.0)
    rect[:, 3] = np.where(w, 1.0, start + span)
    return Crops(tile, rect, span.astype(np.float32))

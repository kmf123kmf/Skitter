"""Pinned frames: patterns laid over a region from a pin, turned about it.

Grid, Brick Bond and Brick Pattern each lay a repeating pattern pinned to the
region's center or top-left corner and turned about that point; tiles at the
edges stay whole and overhang, and tiles that miss the region are dropped. A
PinnedFrame does that shared part, so a lattice slicer only says where its
tiles go:

    frame = PinnedFrame(region.width, region.height, anchor, rotation)
    lo, hi = frame.bounds()  # what to fill, in the pattern's own frame
    ...lay tiles over [lo, hi] (the pin at the origin, unturned)...
    return frame.place(centers, sizes, rotations)

`span` lays evenly spaced tiles along one axis of that frame, centered on
the pin or starting at it.
"""

import math
from dataclasses import dataclass

import numpy as np

from skitter.core.slicing.regions import RegionSet, overlapping, upright

CENTER, TOP_LEFT = "center", "top_left"
ANCHORS = ((CENTER, "Center"), (TOP_LEFT, "Top left"))


@dataclass(frozen=True)
class PinnedFrame:
    """A pattern's frame over a width x height region: pinned at its center or its
    top-left corner (anchor), turned about the pin by rotation (radians, clockwise)."""

    width: float
    height: float
    anchor: str = CENTER
    rotation: float = 0.0

    @property
    def pin(self) -> np.ndarray:
        """The pin, in the region's frame."""
        if self.anchor == CENTER:
            return np.array([self.width / 2, self.height / 2])
        return np.zeros(2)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi): the box around the region as the pattern sees it (unturned, the pin
        at the origin). Tiles reaching into it may show; the rest can't."""
        w, h = self.width, self.height
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float) - self.pin
        c, s = math.cos(self.rotation), math.sin(self.rotation)
        seen = corners @ np.array([[c, -s], [s, c]])  # turned back by -rotation
        return seen.min(axis=0), seen.max(axis=0)

    def place(self, centers, sizes, rotations=0.0) -> RegionSet:
        """The tiles laid in the pattern's frame, turned into the region about the pin,
        each facing as near upright as its outline allows (a half turn either way), and
        only those overlapping the region, in the order laid."""
        centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        n = len(centers)
        sizes = np.broadcast_to(np.asarray(sizes, dtype=float), (n, 2))
        c, s = math.cos(self.rotation), math.sin(self.rotation)
        placed = self.pin + centers @ np.array([[c, s], [-s, c]])  # turned by rotation
        turns = upright(np.broadcast_to(np.asarray(rotations, dtype=float), (n,)) + self.rotation)
        keep = overlapping(placed, sizes, turns, self.width, self.height)
        return RegionSet.from_arrays(placed[keep], sizes[keep], turns[keep])


def span(lo: float, hi: float, step: float, anchor: str) -> tuple[float, int, int]:
    """(origin, count, first): tiles `step` apart along an axis of a pinned frame,
    covering [lo, hi]; tile k (0..count-1) spans origin + [k, k + 1] * step and is
    number first + k from the pin.

    CENTER: the tiles are centered on the pin, and number 0 is the middle one (the
    region seen from its center is symmetric about it). TOP_LEFT: tile 0 starts at
    the pin, so the pattern runs from the corner.
    """
    if anchor == CENTER:
        count = max(1, math.ceil((hi - lo) / step - 1e-9))
        return -count * step / 2, count, -(count // 2)
    first = math.floor(lo / step + 1e-9)
    count = max(1, math.ceil(hi / step - 1e-9) - first)
    return first * step, count, first

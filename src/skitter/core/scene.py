"""The finished mosaic as a scene: every placed tile in its final state.

A MosaicScene is built once from a match result. It lists the placed tiles
(regions without a tile are left out) from bottom to top of the stack, so
drawing them in array order gives the finished mosaic. Everything that
shows the mosaic reads it: the Matching preview, the exported image, and
animations, whose last frame must be exactly this state.

Besides the final geometry, texture choice and tint, the scene carries
attributes that animations can order or group tiles by (position, reading
order, colors, match error, repeated photos). All lengths are in mosaic
units (see slicing/layout.py).
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numba import njit

from skitter.core.matching.matcher import MatchResult
from skitter.core.slicing import SliceContext

# Cached properties that depend only on where tiles lie (kept by with_result).
GEOMETRY = ("corners", "bounds", "position", "distance", "angle", "reading_order", "overlaps")


@dataclass(frozen=True, eq=False)
class MosaicScene:
    result: MatchResult
    region: np.ndarray  # (N,) index of each tile's region in result.regions
    slot: np.ndarray  # (N,) library slot of the tile's photo
    rect: np.ndarray  # (N, 4) crop window (x0, y0, x1, y1), fractions of the upright photo
    mirrored: np.ndarray  # (N,) bool: crop flipped left to right
    center: np.ndarray  # (N, 2) final center
    size: np.ndarray  # (N, 2) final (width, height)
    rotation: np.ndarray  # (N,) final rotation, radians clockwise
    tint_shift: np.ndarray  # (N, 3) OKLab shift matching applies to every pixel of the crop
    tile_color: np.ndarray  # (N, 3) OKLab average color the tile shows (tinted)
    target_color: np.ndarray  # (N, 3) OKLab color matching aimed for in its region
    error: np.ndarray  # (N,) ΔE of the region against the image (nan: unknown)
    canvas: tuple[float, float]  # (width, height) of the image frame
    tile_size: tuple[float, float]  # base tile (width, height)

    @classmethod
    def from_result(cls, result: MatchResult, ctx: SliceContext) -> "MosaicScene":
        regions = result.regions
        order = regions.stacking_order()
        order = np.asarray(order[result.tile[order] >= 0], dtype=np.int64)
        shown = result.tinted_mean()[order]
        error = (
            result.quality.region_error[order]
            if result.quality is not None
            else np.full(len(order), np.nan)
        )
        return cls(
            result=result,
            region=order,
            slot=result.tile[order].astype(np.int64),
            rect=result.rect[order].astype(np.float32),
            mirrored=result.mirrored[order].astype(bool),
            center=regions.center[order].astype(np.float64),
            size=regions.size[order].astype(np.float64),
            rotation=regions.rotation[order].astype(np.float64),
            tint_shift=(shown - result.tile_mean[order]).astype(np.float32),
            tile_color=shown.astype(np.float32),
            target_color=result.tint_target[order].astype(np.float32),
            error=np.asarray(error, dtype=np.float32),
            canvas=(float(ctx.width), float(ctx.height)),
            tile_size=tuple(float(v) for v in ctx.tile_size),
        )

    def __len__(self) -> int:
        return len(self.region)

    def with_result(self, result: MatchResult, ctx: SliceContext) -> "MosaicScene":
        """The scene of an edited result (other tiles picked, same regions placed).

        Geometry, and what is cached from it, carries over.
        """
        scene = MosaicScene.from_result(result, ctx)
        if np.array_equal(scene.region, self.region):
            for name in GEOMETRY:
                if name in self.__dict__:
                    scene.__dict__[name] = self.__dict__[name]
        return scene

    # Display

    @cached_property
    def corners(self) -> np.ndarray:
        """(N, 4, 2) final corners: top-left, top-right, bottom-right, bottom-left."""
        unit = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        local = unit[None] * self.size[:, None]
        c, s = np.cos(self.rotation)[:, None], np.sin(self.rotation)[:, None]
        x, y = local[..., 0], local[..., 1]
        return self.center[:, None] + np.stack([c * x - s * y, s * x + c * y], axis=-1)

    @cached_property
    def bounds(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) around every tile, overhang included."""
        if not len(self):
            return (0.0, 0.0, *self.canvas)
        corners = self.corners.reshape(-1, 2)
        x0, y0 = corners.min(axis=0)
        x1, y1 = corners.max(axis=0)
        return (float(x0), float(y0), float(x1), float(y1))

    # Attributes to order and group tiles by

    @cached_property
    def position(self) -> np.ndarray:
        """(N, 2) center as a fraction of the image frame (0..1 inside it)."""
        return self.center / np.asarray(self.canvas)

    @cached_property
    def distance(self) -> np.ndarray:
        """(N,) distance from the frame's center: 0 at the center, 1 at the corners."""
        half = np.asarray(self.canvas) / 2
        return np.hypot(*(self.center - half).T) / max(float(np.hypot(*half)), 1e-12)

    @cached_property
    def angle(self) -> np.ndarray:
        """(N,) direction from the frame's center, radians clockwise from +x."""
        half = np.asarray(self.canvas) / 2
        return np.arctan2(self.center[:, 1] - half[1], self.center[:, 0] - half[0])

    @cached_property
    def reading_order(self) -> np.ndarray:
        """(N,) rank in reading order: rows of base-tile height, left to right."""
        row = np.floor(self.center[:, 1] / self.tile_size[1])
        rank = np.empty(len(self), np.int64)
        rank[np.lexsort((self.center[:, 0], row))] = np.arange(len(self))
        return rank

    @cached_property
    def use_index(self) -> np.ndarray:
        """(N,) which use of its photo each tile is: 0 for the first (in reading order)."""
        order = np.lexsort((self.reading_order, self.slot))
        first = np.r_[True, self.slot[order][1:] != self.slot[order][:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
        uses = np.empty(len(self), np.int64)
        uses[order] = np.arange(len(order)) - start
        return uses

    @cached_property
    def lightness(self) -> np.ndarray:
        """(N,) OKLab lightness the tile shows (0 black, 1 white)."""
        return self.tile_color[:, 0].astype(np.float64)

    @cached_property
    def overlaps(self) -> np.ndarray:
        """(M, 2) pairs (lower, upper) of tiles whose final rectangles overlap.

        lower < upper (scene order is bottom to top). Tiles that only touch
        along an edge, like grid neighbors, don't count.
        """
        n = len(self)
        if n < 2:
            return np.zeros((0, 2), np.int64)
        corners = self.corners
        lo, hi = corners.min(axis=1), corners.max(axis=1)
        # Candidates: tiles sharing a cell of a grid about one typical tile wide.
        cell = float(np.median((hi - lo).max(axis=1))) or 1.0
        c0 = np.floor(lo / cell).astype(np.int64)
        c1 = np.floor(hi / cell).astype(np.int64)
        candidates = _cell_pairs(c0, c1)
        if not len(candidates):
            return candidates
        a, b = candidates[:, 0], candidates[:, 1]
        touching = np.all((lo[a] < hi[b]) & (lo[b] < hi[a]), axis=1)
        a, b = a[touching], b[touching]
        # Exact test for rotated rectangles: separated along one of their four axes?
        tol = 1e-6 * min(self.tile_size)
        d = self.center[b] - self.center[a]
        half_a, half_b = self.size[a] / 2, self.size[b] / 2
        axes_a = _axes(self.rotation[a])
        axes_b = _axes(self.rotation[b])
        separated = np.zeros(len(a), bool)
        for axis in (*axes_a, *axes_b):
            reach = _reach(half_a, axes_a, axis) + _reach(half_b, axes_b, axis)
            separated |= np.abs(np.sum(d * axis, axis=1)) >= reach - tol
        return np.stack([a[~separated], b[~separated]], axis=1)


def _axes(rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each rectangle's local x and y axes in the image (rotation clockwise)."""
    c, s = np.cos(rotation), np.sin(rotation)
    return np.stack([c, s], axis=1), np.stack([-s, c], axis=1)


def _reach(half: np.ndarray, axes, axis: np.ndarray) -> np.ndarray:
    """How far rectangles (half sizes along their axes) extend along a direction."""
    return half[:, 0] * np.abs(np.sum(axes[0] * axis, axis=1)) + half[:, 1] * np.abs(
        np.sum(axes[1] * axis, axis=1)
    )


def _cell_pairs(c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """Unique pairs (i < j) of boxes covering a common grid cell (inclusive cell ranges)."""
    span = (c1 - c0 + 1).prod(axis=1)
    tile = np.repeat(np.arange(len(c0)), span)
    k = np.arange(len(tile)) - np.repeat(np.cumsum(span) - span, span)
    width = (c1 - c0 + 1)[tile, 0]
    cx = c0[tile, 0] + k % width
    cy = c0[tile, 1] + k // width
    order = np.lexsort((tile, cy, cx))
    pairs = _group_pairs(cx[order], cy[order], tile[order])
    if not len(pairs):
        return pairs
    key = np.unique(pairs[:, 0] * len(c0) + pairs[:, 1])
    return np.stack([key // len(c0), key % len(c0)], axis=1)


@njit(cache=True, nogil=True)
def _group_pairs(cx, cy, tile):
    """All pairs (i < j) of tiles within each run of equal (cx, cy)."""
    count = 0
    start = 0
    n = len(tile)
    for end in range(1, n + 1):
        if end == n or cx[end] != cx[start] or cy[end] != cy[start]:
            k = end - start
            count += k * (k - 1) // 2
            start = end
    out = np.empty((count, 2), np.int64)
    m = 0
    start = 0
    for end in range(1, n + 1):
        if end == n or cx[end] != cx[start] or cy[end] != cy[start]:
            for i in range(start, end):
                for j in range(i + 1, end):
                    a, b = tile[i], tile[j]
                    out[m, 0] = min(a, b)
                    out[m, 1] = max(a, b)
                    m += 1
            start = end
    return out

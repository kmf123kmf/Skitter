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

from skitter.core.color import oklab_to_srgb
from skitter.core.matching.matcher import MatchResult
from skitter.core.slicing import SliceContext


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

    # Display

    @cached_property
    def tint_offset(self) -> np.ndarray:
        """(N, 3) sRGB shift of each tile's average color: the GPU preview's tint."""
        before = oklab_to_srgb(self.tile_color - self.tint_shift)
        return (oklab_to_srgb(self.tile_color) - before).astype(np.float32)

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

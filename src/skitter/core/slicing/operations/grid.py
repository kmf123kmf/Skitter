"""Grid: base-tile cells covering each region."""

import math

import numpy as np

from skitter.core.slicing.base import (
    NO_PROGRESS,
    Progress,
    SliceContext,
    Subdivider,
    check_region_count,
    register_operation,
)
from skitter.core.slicing.common import anchor_param, rotation_param
from skitter.core.slicing.frame import PinnedFrame, span
from skitter.core.slicing.params import TileSizeParam
from skitter.core.slicing.regions import Region, RegionSet


@register_operation
class GridSlicer(Subdivider):
    id = "grid"
    name = "Grid"
    description = (
        "Cover each region with a grid of base tiles (the Mosaic tile shape, scaled by "
        "Cell size). Partial cells at the edges become whole, overhanging cells."
    )

    cell_size = TileSizeParam(
        1.0, "Cell size", help="Cell size in base tiles (1.0: the Mosaic tile itself)."
    )
    angle = rotation_param("grid", "45° gives a diagonal grid")
    anchor = anchor_param("grid")

    def subdivide(
        self, region: Region, ctx: SliceContext, progress: Progress = NO_PROGRESS
    ) -> RegionSet:
        cell_w, cell_h = ctx.tile_dims(self.cell_size)
        frame = PinnedFrame(region.width, region.height, self.anchor, math.radians(self.angle))
        lo, hi = frame.bounds()
        x0, columns, _ = span(lo[0], hi[0], cell_w, self.anchor)
        y0, rows, _ = span(lo[1], hi[1], cell_h, self.anchor)
        check_region_count(columns * rows, self.name)
        # Row by row: left to right, top to bottom (as RegionSet.cells).
        xx, yy = np.meshgrid(x0 + (np.arange(columns) + 0.5) * cell_w,
                             y0 + (np.arange(rows) + 0.5) * cell_h)  # fmt: skip
        return frame.place(np.stack([xx.ravel(), yy.ravel()], axis=-1), (cell_w, cell_h))

    def summary(self) -> str:
        text = "base tiles" if self.cell_size == 1 else f"{self.cell_size:g}× tiles"
        return f"{text}, {self.angle:g}°" if self.angle else text

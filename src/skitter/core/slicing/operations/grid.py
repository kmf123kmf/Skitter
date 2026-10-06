"""Grid: base-tile cells covering each region."""

import math

from skitter.core.slicing.base import (
    NO_PROGRESS,
    Progress,
    SliceContext,
    Subdivider,
    check_region_count,
    register_operation,
)
from skitter.core.slicing.params import ChoiceParam, TileSizeParam
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
    anchor = ChoiceParam(
        "center", "Anchor",
        choices=[("center", "Center"), ("top_left", "Top left")],
        help="Where the grid is pinned; overhang goes to the opposite edges.",
    )  # fmt: skip

    def subdivide(
        self, region: Region, ctx: SliceContext, progress: Progress = NO_PROGRESS
    ) -> RegionSet:
        w, h = region.width, region.height
        cell_w, cell_h = (self.cell_size * size for size in ctx.tile_size)
        columns = max(1, math.ceil(w / cell_w - 1e-9))
        rows = max(1, math.ceil(h / cell_h - 1e-9))
        check_region_count(columns * rows, self.name)
        origin = (0.0, 0.0)
        if self.anchor == "center":
            origin = ((w - columns * cell_w) / 2, (h - rows * cell_h) / 2)
        return RegionSet.cells(cell_w, cell_h, columns, rows, origin)

    def summary(self) -> str:
        return "base tiles" if self.cell_size == 1 else f"{self.cell_size:g}× tiles"

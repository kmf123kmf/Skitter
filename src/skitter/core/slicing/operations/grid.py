"""Grid: equal cells."""

import math

from skitter.core.slicing.base import (
    SliceContext,
    Subdivider,
    check_region_count,
    register_operation,
)
from skitter.core.slicing.params import BoolParam, ChoiceParam, IntParam, TileSizeParam
from skitter.core.slicing.regions import Region, RegionSet

TILES, COUNT = "tiles", "count"


@register_operation
class GridSlicer(Subdivider):
    id = "grid"
    name = "Grid"
    description = (
        "Fill each region with a grid of equal cells: base-tile-sized cells that "
        "may overhang the edges, or a fixed number of cells that fit exactly."
    )

    mode = ChoiceParam(TILES, "Cells", choices=[(TILES, "Base tiles"), (COUNT, "Fixed count")])
    cell_size = TileSizeParam(
        1.0, "Cell size", when=lambda op: op.mode == TILES,
        help="Cell size in base tiles. Partial cells at the edges become whole, overhanging cells.",
    )  # fmt: skip
    anchor = ChoiceParam(
        "center", "Anchor",
        choices=[("center", "Center"), ("top_left", "Top left")],
        when=lambda op: op.mode == TILES,
        help="Where the grid is pinned; overhang goes to the opposite edges.",
    )  # fmt: skip
    columns = IntParam(4, "Columns", min=1, max=1000, when=lambda op: op.mode == COUNT)
    square_cells = BoolParam(
        True, "Square cells", when=lambda op: op.mode == COUNT,
        help="Choose the number of rows so cells are as square as possible.",
    )  # fmt: skip
    rows = IntParam(4, "Rows", min=1, max=1000,
                    when=lambda op: op.mode == COUNT and not op.square_cells)  # fmt: skip

    def rows_for(self, width: float, height: float) -> int:
        """Rows in count mode."""
        if self.square_cells:
            return max(1, round(height / (width / self.columns)))
        return self.rows

    def subdivide(self, region: Region, ctx: SliceContext) -> RegionSet:
        w, h = region.width, region.height
        if self.mode == COUNT:
            rows = self.rows_for(w, h)
            check_region_count(self.columns * rows, self.name)
            return RegionSet.grid(w, h, self.columns, rows)
        cell_w, cell_h = (self.cell_size * size for size in ctx.tile_size)
        columns = max(1, math.ceil(w / cell_w - 1e-9))
        rows = max(1, math.ceil(h / cell_h - 1e-9))
        check_region_count(columns * rows, self.name)
        origin = (0.0, 0.0)
        if self.anchor == "center":
            origin = ((w - columns * cell_w) / 2, (h - rows * cell_h) / 2)
        return RegionSet.cells(cell_w, cell_h, columns, rows, origin)

    def summary(self) -> str:
        if self.mode == TILES:
            return "base tiles" if self.cell_size == 1 else f"{self.cell_size:g}× tiles"
        if self.square_cells:
            return f"{self.columns} columns, square cells"
        return f"{self.columns} × {self.rows}"

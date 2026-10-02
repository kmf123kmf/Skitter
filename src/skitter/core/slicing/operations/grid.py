"""Grid: equal cells."""

from skitter.core.slicing.base import SliceContext, Subdivider, register_operation
from skitter.core.slicing.params import BoolParam, IntParam
from skitter.core.slicing.regions import Region, RegionSet


@register_operation
class GridSlicer(Subdivider):
    id = "grid"
    name = "Grid"
    description = "Divide each region into a grid of equal cells."

    columns = IntParam(24, "Columns", min=1, max=1000)
    square_cells = BoolParam(
        True, "Square cells", help="Choose the number of rows so cells are as square as possible."
    )
    rows = IntParam(16, "Rows", min=1, max=1000, when=lambda op: not op.square_cells)

    def rows_for(self, width: float, height: float) -> int:
        if self.square_cells:
            return max(1, round(height / (width / self.columns)))
        return self.rows

    def subdivide(self, region: Region, ctx: SliceContext) -> RegionSet:
        rows = self.rows_for(region.width, region.height)
        return RegionSet.grid(region.width, region.height, self.columns, rows)

    def summary(self) -> str:
        if self.square_cells:
            return f"{self.columns} columns, square cells"
        return f"{self.columns} × {self.rows}"

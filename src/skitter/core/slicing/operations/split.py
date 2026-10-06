"""Split: divide each region into a fixed number of equal pieces."""

from skitter.core.slicing.base import (
    NO_PROGRESS,
    Progress,
    SliceContext,
    Subdivider,
    check_region_count,
    register_operation,
)
from skitter.core.slicing.params import BoolParam, IntParam
from skitter.core.slicing.regions import Region, RegionSet


@register_operation
class SplitSlicer(Subdivider):
    id = "split"
    name = "Split"
    description = (
        "Divide each region into equal pieces, across by down. Counts are per region: "
        "after Jitter or Photo Pile, each photo is split on its own."
    )

    across = IntParam(2, "Across", min=1, max=1000, help="Pieces side by side in each region.")
    keep_shape = BoolParam(
        True, "Keep tile shape",
        help="Choose Down so the pieces come as close as possible to the Mosaic tile shape.",
    )  # fmt: skip
    down = IntParam(
        2, "Down", min=1, max=1000, when=lambda op: not op.keep_shape,
        help="Pieces stacked top to bottom in each region.",
    )  # fmt: skip

    def down_for(self, width: float, height: float, tile_aspect: float) -> int:
        """Rows of pieces for a region of this size (in its own frame)."""
        if not self.keep_shape:
            return self.down
        piece_w = width / self.across
        return max(1, round(height * tile_aspect / piece_w))

    def subdivide(
        self, region: Region, ctx: SliceContext, progress: Progress = NO_PROGRESS
    ) -> RegionSet:
        w, h = region.width, region.height
        down = self.down_for(w, h, ctx.tile_aspect)
        check_region_count(self.across * down, self.name)
        return RegionSet.grid(w, h, self.across, down)

    def summary(self) -> str:
        if self.keep_shape:
            return f"{self.across} across, tile shape"
        return f"{self.across} × {self.down}"

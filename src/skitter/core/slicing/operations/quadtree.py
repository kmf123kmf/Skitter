"""Quadtree: adaptive subdivision driven by image detail."""

import math

from skitter.core.slicing.base import SliceContext, Subdivider, register_operation
from skitter.core.slicing.params import FloatParam, IntParam, TileSizeParam
from skitter.core.slicing.regions import Region, RegionSet


@register_operation
class QuadtreeSlicer(Subdivider):
    id = "quadtree"
    name = "Quadtree"
    description = (
        "Split regions into quarters wherever the image has detail, leaving flat "
        "areas large. Use after a Grid so the largest cells have the tile shape."
    )

    threshold = FloatParam(
        12.0, "Detail threshold", min=0.5, max=128.0, step=0.5, decimals=1,
        help="Keep splitting while brightness varies more than this (standard deviation).",
    )  # fmt: skip
    min_size = TileSizeParam(
        0.25, "Minimum size", min=0.01, max=10.0,
        help="Never split a cell into parts smaller than this, in base tiles.",
    )  # fmt: skip
    max_depth = IntParam(4, "Maximum depth", min=1, max=12)

    def subdivide(self, region: Region, ctx: SliceContext) -> RegionSet:
        samples, scale = ctx.patch(region)
        min_w, min_h = (self.min_size * size for size in ctx.tile_size)
        leaves: list[tuple[float, float, float, float]] = []

        def split(x: float, y: float, w: float, h: float, depth: int) -> None:
            can_split = depth < self.max_depth and w / 2 >= min_w - 1e-9 and h / 2 >= min_h - 1e-9
            if can_split:
                block = samples[
                    int(y * scale) : max(int(y * scale) + 1, math.ceil((y + h) * scale)),
                    int(x * scale) : max(int(x * scale) + 1, math.ceil((x + w) * scale)),
                ]
                if block.std() > self.threshold:
                    hw, hh = w / 2, h / 2
                    for dx, dy in ((0, 0), (hw, 0), (0, hh), (hw, hh)):
                        split(x + dx, y + dy, hw, hh, depth + 1)
                    return
            leaves.append((x, y, w, h))

        split(0.0, 0.0, region.width, region.height, 0)
        x, y, w, h = zip(*leaves, strict=True)
        return RegionSet.from_rects(x, y, w, h)

    def summary(self) -> str:
        return f"detail {self.threshold:g}, min {self.min_size:g}× tile"

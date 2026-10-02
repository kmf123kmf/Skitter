"""Quadtree: adaptive subdivision driven by image detail."""

import math

from skitter.core.slicing.base import SliceContext, Subdivider, register_operation
from skitter.core.slicing.params import FloatParam, IntParam
from skitter.core.slicing.regions import Region, RegionSet


@register_operation
class QuadtreeSlicer(Subdivider):
    id = "quadtree"
    name = "Quadtree"
    description = (
        "Split regions into quarters wherever the image has detail, "
        "leaving flat areas as large regions."
    )

    threshold = FloatParam(
        12.0, "Detail threshold", min=0.5, max=128.0, step=0.5, decimals=1,
        help="Keep splitting while brightness varies more than this (standard deviation).",
    )  # fmt: skip
    min_size = IntParam(16, "Minimum size", min=2, max=4096, suffix=" px")
    max_depth = IntParam(6, "Maximum depth", min=1, max=12)

    def subdivide(self, region: Region, ctx: SliceContext) -> RegionSet:
        samples, scale = ctx.patch(region)
        leaves: list[tuple[float, float, float, float]] = []

        def split(x: float, y: float, w: float, h: float, depth: int) -> None:
            can_split = depth < self.max_depth and min(w, h) / 2 >= self.min_size
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
        return f"detail {self.threshold:g}, min {self.min_size} px"

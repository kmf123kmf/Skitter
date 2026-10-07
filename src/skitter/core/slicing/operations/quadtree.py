"""Quadtree: adaptive subdivision driven by image detail."""

import math

from skitter.core.slicing.base import (
    NO_PROGRESS,
    Progress,
    SliceContext,
    Subdivider,
    register_operation,
)
from skitter.core.slicing.params import ChoiceParam, FloatParam
from skitter.core.slicing.regions import Region, RegionSet

SIZES = {1: "½", 2: "¼", 3: "⅛", 4: "1/16", 5: "1/32", 6: "1/64"}  # by number of splits


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
    # Halving is all a quadtree does, so the only sizes it can reach are the
    # starting cell over a power of two: the limit is how many times to halve.
    splits = ChoiceParam(
        2, "Smallest cell",
        choices=[(n, f"{SIZES[n]} of the cell ({n} split{'s' if n > 1 else ''})") for n in SIZES],
        help="How small the detail can split each cell it starts from; ¼ after a base-tile "
             "Grid gives quarter-tile cells at most.",
    )  # fmt: skip

    def subdivide(
        self, region: Region, ctx: SliceContext, progress: Progress = NO_PROGRESS
    ) -> RegionSet:
        samples, scale = ctx.patch(region)
        leaves: list[tuple[float, float, float, float]] = []

        def split(x: float, y: float, w: float, h: float, depth: int) -> None:
            if depth < self.splits:
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
        return f"detail {self.threshold:g}, down to {SIZES[self.splits]}"

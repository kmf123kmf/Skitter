"""Adjusters: change existing regions without splitting them."""

import numpy as np

from skitter.core.slicing.base import (
    NO_PROGRESS,
    Progress,
    SliceContext,
    SlicingOperation,
    register_operation,
)
from skitter.core.slicing.params import ChoiceParam, FloatParam, IntParam
from skitter.core.slicing.regions import RegionSet


@register_operation
class JitterAdjust(SlicingOperation):
    id = "jitter"
    name = "Jitter"
    category = "Adjust"
    description = "Randomly shift, rotate and resize regions for a hand-placed look."

    offset = FloatParam(
        0.1, "Offset", min=0.0, max=1.0, step=0.05,
        help="Largest shift, as a fraction of each region's size.",
    )  # fmt: skip
    rotation = FloatParam(5.0, "Rotation", min=0.0, max=180.0, step=1.0, decimals=1, suffix="°")
    scale = FloatParam(
        0.0, "Scale", min=0.0, max=0.9, step=0.05,
        help="Largest size change, as a fraction of each region's size.",
    )  # fmt: skip
    seed = IntParam(1, "Seed", min=0, max=999_999)

    def apply(
        self, regions: RegionSet, ctx: SliceContext, progress: Progress = NO_PROGRESS
    ) -> RegionSet:
        rng = np.random.default_rng(self.seed)
        n = len(regions)
        shift = rng.uniform(-1, 1, (n, 2)) * self.offset * regions.size
        turn = rng.uniform(-1, 1, n) * np.deg2rad(self.rotation)
        factor = 1 + rng.uniform(-1, 1, n) * self.scale
        return regions.replace(
            center=regions.center + shift,
            size=regions.size * factor[:, None],
            rotation=regions.rotation + turn,
        )

    def summary(self) -> str:
        return f"offset {self.offset:g}, rotation {self.rotation:g}°"


@register_operation
class StackingAdjust(SlicingOperation):
    id = "stacking"
    name = "Stacking Order"
    category = "Adjust"
    description = "Change which regions lie on top where regions overlap."

    order = ChoiceParam(
        "random",
        "Order",
        choices=[
            ("random", "Random"),
            ("small_on_top", "Smaller on top"),
            ("large_on_top", "Larger on top"),
            ("reverse", "Reverse"),
        ],
    )
    seed = IntParam(1, "Seed", min=0, max=999_999, when=lambda op: op.order == "random")

    def apply(
        self, regions: RegionSet, ctx: SliceContext, progress: Progress = NO_PROGRESS
    ) -> RegionSet:
        if self.order == "random":
            key = np.random.default_rng(self.seed).random(len(regions))
        elif self.order == "small_on_top":
            key = -regions.area()
        elif self.order == "large_on_top":
            key = regions.area()
        else:
            key = -regions.stacking_rank()
        return regions.restacked(key)

    def summary(self) -> str:
        return dict(StackingAdjust.order.choices)[self.order]

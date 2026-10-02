"""Photo pile: overlapping, randomly rotated rectangles stacked in random order."""

import math

import numpy as np

from skitter.core.slicing.base import SliceContext, Subdivider, register_operation
from skitter.core.slicing.params import ChoiceParam, FloatParam, IntParam
from skitter.core.slicing.regions import Region, RegionSet

JITTER = 0.25  # center jitter, as a fraction of the cell spacing


@register_operation
class PileSlicer(Subdivider):
    id = "pile"
    name = "Photo Pile"
    description = (
        "Cover each region with a pile of overlapping, randomly rotated photos "
        "stacked in random order. Every point is covered."
    )

    photo_size = FloatParam(
        200.0, "Photo size", min=4.0, max=20_000.0, step=10.0, decimals=0, suffix=" px",
        help="Length of each photo's longer side.",
    )  # fmt: skip
    aspect = ChoiceParam(1.5, "Aspect", choices=[(1.0, "Square"), (4 / 3, "4:3"), (1.5, "3:2")])
    portrait = FloatParam(
        0.3, "Portrait share", min=0.0, max=1.0, step=0.05,
        help="Fraction of photos turned to portrait orientation.",
    )  # fmt: skip
    rotation = FloatParam(15.0, "Rotation", min=0.0, max=45.0, step=1.0, decimals=1, suffix="°")
    overlap = FloatParam(
        0.2, "Extra overlap", min=0.0, max=0.9, step=0.05,
        help="Overlap beyond what full coverage needs. Higher means more, denser photos.",
    )  # fmt: skip
    seed = IntParam(1, "Seed", min=0, max=999_999)

    def spacing(self) -> float:
        """Cell spacing that guarantees coverage.

        Each photo, at any rotation up to the limit, contains an axis-aligned
        square of half-size r around its center. Centers sit in cells of this
        spacing, jittered by up to JITTER of it, so every point of a cell is
        within (0.5 + JITTER) * spacing of its photo's center on each axis.
        """
        short = self.photo_size / self.aspect
        theta = math.radians(self.rotation)
        r = (short / 2) / (math.cos(theta) + math.sin(theta))
        return (1 - self.overlap) * 2 * r / (1 + 2 * JITTER)

    def subdivide(self, region: Region, ctx: SliceContext) -> RegionSet:
        # Seed per region (by position) so neighbors don't get identical piles.
        rng = np.random.default_rng([self.seed, round(region.cx * 64), round(region.cy * 64)])
        spacing = self.spacing()
        nx = max(1, math.ceil(region.width / spacing))
        ny = max(1, math.ceil(region.height / spacing))
        cell_w, cell_h = region.width / nx, region.height / ny  # <= spacing: still covered
        cells = RegionSet.grid(region.width, region.height, nx, ny)
        n = len(cells)

        jitter = rng.uniform(-JITTER, JITTER, (n, 2)) * (cell_w, cell_h)
        long_side, short_side = self.photo_size, self.photo_size / self.aspect
        portrait = rng.random(n) < self.portrait
        size = np.where(portrait[:, None], (short_side, long_side), (long_side, short_side))
        turn = rng.uniform(-1, 1, n) * math.radians(self.rotation)
        return RegionSet.from_arrays(cells.center + jitter, size, turn, z=rng.permutation(n))

    def summary(self) -> str:
        return f"{self.photo_size:g} px photos, ±{self.rotation:g}°"

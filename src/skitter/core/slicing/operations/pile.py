"""Photo pile: overlapping, randomly rotated rectangles stacked in random order."""

import math

import numpy as np

from skitter.core.slicing.base import SliceContext, Subdivider, register_operation
from skitter.core.slicing.params import FloatParam, IntParam, TileSizeParam
from skitter.core.slicing.regions import Region, RegionSet

JITTER = 0.25  # center jitter, as a fraction of the cell spacing


@register_operation
class PileSlicer(Subdivider):
    id = "pile"
    name = "Photo Pile"
    description = (
        "Cover each region with a pile of overlapping, randomly rotated photos "
        "in the base tile's shape, stacked in random order. At spread 1.0 every "
        "point is covered."
    )

    photo_size = TileSizeParam(1.0, "Photo size", help="Photo size in base tiles.")
    portrait = FloatParam(
        0.3, "Portrait share", min=0.0, max=1.0, step=0.05,
        help="Fraction of photos turned a quarter turn (portrait for landscape tiles).",
    )  # fmt: skip
    rotation = FloatParam(15.0, "Rotation", min=0.0, max=45.0, step=1.0, decimals=1, suffix="°")
    spread = FloatParam(
        1.25, "Spread", min=0.3, max=3.0, step=0.05,
        help=(
            "Photo spacing. 1.0 places photos as far apart as guaranteed full coverage "
            "allows; higher uses fewer photos but may leave gaps (see Coverage); lower "
            "piles them deeper."
        ),
    )  # fmt: skip
    seed = IntParam(1, "Seed", min=0, max=999_999)

    def photo_dims(self, ctx: SliceContext) -> tuple[float, float]:
        tile_w, tile_h = ctx.tile_size
        return self.photo_size * tile_w, self.photo_size * tile_h

    def spacing(self, ctx: SliceContext) -> float:
        """Cell spacing; at spread 1.0 it guarantees coverage.

        Each photo, at any rotation up to the limit, contains an axis-aligned
        square of half-size r around its center. Centers sit in cells of this
        spacing, jittered by up to JITTER of it, so every point of a cell is
        within (0.5 + JITTER) * spacing of its photo's center on each axis.
        """
        short = min(self.photo_dims(ctx))
        theta = math.radians(self.rotation)
        r = (short / 2) / (math.cos(theta) + math.sin(theta))
        return self.spread * 2 * r / (1 + 2 * JITTER)

    def subdivide(self, region: Region, ctx: SliceContext) -> RegionSet:
        # Seed per region (by position) so neighbors don't get identical piles.
        rng = np.random.default_rng([self.seed, round(region.cx * 64), round(region.cy * 64)])
        spacing = self.spacing(ctx)
        nx = max(1, math.ceil(region.width / spacing))
        ny = max(1, math.ceil(region.height / spacing))
        cell_w, cell_h = region.width / nx, region.height / ny  # <= spacing: still covered
        cells = RegionSet.grid(region.width, region.height, nx, ny)
        n = len(cells)

        jitter = rng.uniform(-JITTER, JITTER, (n, 2)) * (cell_w, cell_h)
        photo_w, photo_h = self.photo_dims(ctx)
        turned = rng.random(n) < self.portrait
        size = np.where(turned[:, None], (photo_h, photo_w), (photo_w, photo_h))
        turn = rng.uniform(-1, 1, n) * math.radians(self.rotation)
        return RegionSet.from_arrays(cells.center + jitter, size, turn, z=rng.permutation(n))

    def summary(self) -> str:
        size = "tile-size" if self.photo_size == 1 else f"{self.photo_size:g}× tile"
        return f"{size} photos, ±{self.rotation:g}°"

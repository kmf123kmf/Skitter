"""Brick bond: courses of base-tile bricks, each shifted along its length."""

import math

import numpy as np

from skitter.core.slicing.base import (
    NO_PROGRESS,
    Progress,
    SliceContext,
    Subdivider,
    check_region_count,
    region_rng,
    register_operation,
)
from skitter.core.slicing.common import (
    VERTICAL,
    anchor_param,
    orientation_param,
    rotation_param,
    seed_param,
)
from skitter.core.slicing.frame import PinnedFrame, span
from skitter.core.slicing.params import ChoiceParam, FloatParam, TileSizeParam
from skitter.core.slicing.regions import Region, RegionSet

BOND_STEPS = {"running": 1 / 2, "third": 1 / 3, "quarter": 1 / 4}  # shift per course


@register_operation
class BondSlicer(Subdivider):
    id = "bond"
    name = "Brick Bond"
    description = (
        "Lay base-tile bricks in courses like a brick wall, each course shifted "
        "along its length. Courses run across (horizontal) or down (vertical)."
    )

    orientation = orientation_param(
        "Courses", "Horizontal: rows shift sideways. Vertical: columns shift up and down."
    )
    bond = ChoiceParam(
        "running", "Bond",
        choices=[
            ("running", "Running (½)"),
            ("third", "Third (⅓)"),
            ("quarter", "Quarter (¼)"),
            ("custom", "Custom"),
            ("random", "Random"),
        ],
        help="How far each course shifts from the one before.",
    )  # fmt: skip
    step = FloatParam(
        0.5, "Shift", min=0.0, max=1.0, step=0.05, when=lambda op: op.bond == "custom",
        help="Shift from one course to the next, as a fraction of a brick's length. "
             "0 lines bricks up in a plain grid.",
    )  # fmt: skip
    seed = seed_param(when=lambda op: op.bond == "random")
    cell_size = TileSizeParam(
        1.0, "Brick size",
        help="Brick size in base tiles. Partial bricks at the edges become whole, "
             "overhanging bricks.",
    )  # fmt: skip
    angle = rotation_param("pattern", "45° gives diagonal brickwork")
    anchor = anchor_param("pattern")

    def course_shifts(self, count: int, first: int = 0, region: Region | None = None) -> np.ndarray:
        """Shift of each of count courses, numbered from first, as a fraction of a brick.

        Random shifts are seeded per region (by position) so neighbors differ.
        """
        if self.bond == "random":
            rng = region_rng(self.seed, region) if region else np.random.default_rng(self.seed)
            return rng.random(count)
        step = self.step if self.bond == "custom" else BOND_STEPS[self.bond]
        return ((np.arange(count) + first) * step) % 1.0

    def subdivide(
        self, region: Region, ctx: SliceContext, progress: Progress = NO_PROGRESS
    ) -> RegionSet:
        cell_w, cell_h = ctx.tile_dims(self.cell_size)
        vertical = self.orientation == VERTICAL
        frame = PinnedFrame(region.width, region.height, self.anchor, math.radians(self.angle))
        lo, hi = frame.bounds()
        # Course coordinates: u along the courses, v across them.
        if vertical:
            lo, hi = lo[::-1], hi[::-1]
        brick_u, brick_v = (cell_h, cell_w) if vertical else (cell_w, cell_h)
        # The unshifted grid pinned at the anchor; course 0 there stays unshifted.
        origin_u, _, _ = span(lo[0], hi[0], brick_u, self.anchor)
        origin_v, courses, first = span(lo[1], hi[1], brick_v, self.anchor)
        check_region_count(courses * (math.ceil((hi[0] - lo[0]) / brick_u) + 2), self.name)

        us, vs = [], []
        for index, shift in enumerate(self.course_shifts(courses, first, region)):
            start = origin_u + shift * brick_u
            # Every brick start + [i, i + 1] * brick_u that overlaps (lo_u, hi_u).
            i0 = math.floor((lo[0] - start) / brick_u - 1 + 1e-9) + 1
            i1 = math.ceil((hi[0] - start) / brick_u - 1e-9) - 1
            u = start + (np.arange(i0, i1 + 1) + 0.5) * brick_u
            us.append(u)
            vs.append(np.full(len(u), origin_v + (index + 0.5) * brick_v))
        u, v = np.concatenate(us), np.concatenate(vs)
        return frame.place(np.stack([v, u] if vertical else [u, v], axis=-1), (cell_w, cell_h))

    def summary(self) -> str:
        if self.bond == "custom":
            text = f"{self.step:g} shift"
        else:
            text = dict(BondSlicer.bond.choices)[self.bond].split(" (")[0].lower() + " bond"
        if self.orientation == VERTICAL:
            text += ", vertical"
        if self.angle:
            text += f", {self.angle:g}°"
        if self.cell_size != 1:
            text += f", {self.cell_size:g}× tiles"
        return text

"""Brick bond: courses of base-tile bricks, each shifted along its length."""

import math

import numpy as np

from skitter.core.slicing.base import SliceContext, Subdivider, region_rng, register_operation
from skitter.core.slicing.params import ChoiceParam, FloatParam, IntParam, TileSizeParam
from skitter.core.slicing.regions import Region, RegionSet

HORIZONTAL, VERTICAL = "horizontal", "vertical"
BOND_STEPS = {"running": 1 / 2, "third": 1 / 3, "quarter": 1 / 4}  # shift per course


@register_operation
class BondSlicer(Subdivider):
    id = "bond"
    name = "Brick Bond"
    description = (
        "Lay base-tile bricks in courses like a brick wall, each course shifted "
        "along its length. Courses run across (horizontal) or down (vertical)."
    )

    orientation = ChoiceParam(
        HORIZONTAL, "Courses", choices=[(HORIZONTAL, "Horizontal"), (VERTICAL, "Vertical")],
        help="Horizontal: rows shift sideways. Vertical: columns shift up and down.",
    )  # fmt: skip
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
    seed = IntParam(1, "Seed", min=0, max=999_999, when=lambda op: op.bond == "random")
    cell_size = TileSizeParam(
        1.0, "Brick size",
        help="Brick size in base tiles. Partial bricks at the edges become whole, "
             "overhanging bricks.",
    )  # fmt: skip
    anchor = ChoiceParam(
        "center", "Anchor",
        choices=[("center", "Center"), ("top_left", "Top left")],
        help="Where the pattern is pinned; overhang goes to the opposite edges.",
    )  # fmt: skip

    def course_shifts(self, count: int, first: int = 0, region: Region | None = None) -> np.ndarray:
        """Shift of each of count courses, numbered from first, as a fraction of a brick.

        Random shifts are seeded per region (by position) so neighbors differ.
        """
        if self.bond == "random":
            rng = region_rng(self.seed, region) if region else np.random.default_rng(self.seed)
            return rng.random(count)
        step = self.step if self.bond == "custom" else BOND_STEPS[self.bond]
        return ((np.arange(count) + first) * step) % 1.0

    def subdivide(self, region: Region, ctx: SliceContext) -> RegionSet:
        cell_w, cell_h = (self.cell_size * size for size in ctx.tile_size)
        vertical = self.orientation == VERTICAL
        # Work in course coordinates: u along the courses, v across them.
        w, h = region.width, region.height
        span_u, span_v = (h, w) if vertical else (w, h)
        brick_u, brick_v = (cell_h, cell_w) if vertical else (cell_w, cell_h)
        courses = max(1, math.ceil(span_v / brick_v - 1e-9))
        origin_u = origin_v = 0.0
        first = 0
        if self.anchor == "center":
            # Center the unshifted grid; the middle course stays unshifted.
            per_course = max(1, math.ceil(span_u / brick_u - 1e-9))
            origin_u = (span_u - per_course * brick_u) / 2
            origin_v = (span_v - courses * brick_v) / 2
            first = -(courses // 2)

        us, vs = [], []
        for index, shift in enumerate(self.course_shifts(courses, first, region)):
            start = origin_u + shift * brick_u
            # Every brick start + [i, i + 1] * brick_u that overlaps (0, span_u).
            i0 = math.floor(-start / brick_u - 1 + 1e-9) + 1
            i1 = math.ceil((span_u - start) / brick_u - 1e-9) - 1
            u = start + (np.arange(i0, i1 + 1) + 0.5) * brick_u
            us.append(u)
            vs.append(np.full(len(u), origin_v + (index + 0.5) * brick_v))
        u, v = np.concatenate(us), np.concatenate(vs)
        center = np.stack([v, u] if vertical else [u, v], axis=-1)
        return RegionSet.from_arrays(center, (cell_w, cell_h))

    def summary(self) -> str:
        if self.bond == "custom":
            text = f"{self.step:g} shift"
        else:
            text = dict(BondSlicer.bond.choices)[self.bond].split(" (")[0].lower() + " bond"
        if self.orientation == VERTICAL:
            text += ", vertical"
        if self.cell_size != 1:
            text += f", {self.cell_size:g}× tiles"
        return text

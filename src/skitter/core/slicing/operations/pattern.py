"""Brick patterns: herringbone, basketweave and other repeating layouts.

The patterns themselves are registered here; patterns.py has the framework.
To add a pattern, write a build function and decorate it with
`register_pattern`. If it needs a setting, add a parameter to PatternSlicer
(greyed out unless the selected pattern lists it in its options).
"""

import math

from skitter.core.slicing.base import (
    NO_PROGRESS,
    Progress,
    SliceContext,
    Subdivider,
    register_operation,
)
from skitter.core.slicing.params import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    IntParam,
    TileSizeParam,
)
from skitter.core.slicing.patterns import (
    PatternBuilder,
    PatternUnit,
    get_pattern,
    patterns,
    register_pattern,
    tile_pattern,
)
from skitter.core.slicing.regions import Region, RegionSet

HORIZONTAL, VERTICAL = "horizontal", "vertical"


# Patterns


@register_pattern(
    "herringbone", "Herringbone",
    description="Horizontal and vertical bricks in zigzag rows. Works with any tile shape.",
)  # fmt: skip
def herringbone(b: PatternBuilder) -> PatternUnit:
    length, thick = b.length, b.thickness
    # A horizontal brick, and a vertical one rising from its right end.
    b.add(length / 2, thick / 2, horizontal=True)
    b.add(length + thick / 2, thick - length / 2, horizontal=False)
    return b.unit((thick, thick), (length + thick, thick - length))


@register_pattern(
    "basketweave", "Basketweave", options=("weave_auto", "weave"),
    description="Square blocks of parallel bricks, alternating horizontal and vertical.",
)  # fmt: skip
def basketweave(b: PatternBuilder, weave_auto: bool, weave: int) -> PatternUnit:
    # Blocks are brick-length squares, so bricks are length / n thick: the
    # base tile shape exactly when the tile is n:1.
    n = max(1, round(b.length / b.thickness)) if weave_auto else weave
    length, thick = b.length, b.length / n
    for i in range(n):
        b.add(length / 2, (i + 0.5) * thick, horizontal=True, thickness=thick)
        b.add(length + (i + 0.5) * thick, length / 2, horizontal=False, thickness=thick)
    return b.unit((2 * length, 0), (length, length))


# Operation


def _uses(option: str):
    """`when` condition: the selected pattern uses this option."""
    return lambda op: option in get_pattern(op.pattern).options


@register_operation
class PatternSlicer(Subdivider):
    id = "pattern"
    name = "Brick Pattern"
    description = (
        "Fill each region with a repeating brick pattern such as herringbone or "
        "basketweave. Bricks lying the other way are base tiles turned 90°."
    )

    pattern = ChoiceParam(
        "herringbone", "Pattern", choices=lambda: [(p.id, p.name) for p in patterns()]
    )
    weave_auto = BoolParam(
        True, "Match tile shape", when=_uses("weave_auto"),
        help="Choose the bricks per block that keeps bricks closest to the tile shape.",
    )  # fmt: skip
    weave = IntParam(
        2, "Bricks per block", min=1, max=20,
        when=lambda op: _uses("weave")(op) and not op.weave_auto,
        help="Bricks side by side in each square block. Bricks keep the tile's length "
             "and get thinner or thicker to fill the block.",
    )  # fmt: skip
    orientation = ChoiceParam(
        HORIZONTAL, "Orientation", choices=[(HORIZONTAL, "Horizontal"), (VERTICAL, "Vertical")],
        help="Vertical turns the whole pattern 90°.",
    )  # fmt: skip
    angle = FloatParam(
        0.0, "Rotation", min=-90.0, max=90.0, step=5.0, decimals=1, suffix="°",
        help="Extra rotation of the whole pattern; 45° gives diagonal herringbone.",
    )  # fmt: skip
    cell_size = TileSizeParam(
        1.0, "Brick size",
        help="Brick size in base tiles. Bricks at the edges stay whole and overhang.",
    )  # fmt: skip
    anchor = ChoiceParam(
        "center", "Anchor",
        choices=[("center", "Center"), ("top_left", "Top left")],
        help="Which point of the region the pattern is pinned to and turns about.",
    )  # fmt: skip

    def unit(self, ctx: SliceContext) -> PatternUnit:
        pattern = get_pattern(self.pattern)
        builder = PatternBuilder(tuple(self.cell_size * size for size in ctx.tile_size))
        return pattern.build(builder, **{name: getattr(self, name) for name in pattern.options})

    def subdivide(
        self, region: Region, ctx: SliceContext, progress: Progress = NO_PROGRESS
    ) -> RegionSet:
        turn = math.radians(self.angle) + (math.pi / 2 if self.orientation == VERTICAL else 0.0)
        return tile_pattern(
            self.unit(ctx), region.width, region.height, anchor=self.anchor, rotation=turn
        )

    def summary(self) -> str:
        text = get_pattern(self.pattern).name
        if self.orientation == VERTICAL:
            text += ", vertical"
        if self.angle:
            text += f", {self.angle:g}°"
        if self.cell_size != 1:
            text += f", {self.cell_size:g}× tiles"
        return text

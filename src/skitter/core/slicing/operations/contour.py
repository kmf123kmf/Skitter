"""Contour rows: tiles laid in rows that follow the image's edges and outlines.

Like the rows of a Roman mosaic (opus vermiculatum), rows run parallel to
the strongest edges and ripple outward from them. structure.py finds the
edges and the row field; rows.py lays the tiles. Every tile keeps the base
tile's shape, its long side along its row.
"""

import math

import numpy as np

from skitter.core.slicing.base import (
    NO_PROGRESS,
    Progress,
    SliceContext,
    Subdivider,
    check_region_count,
    register_operation,
)
from skitter.core.slicing.params import BoolParam, FloatParam, IntParam, TileSizeParam
from skitter.core.slicing.regions import Region, RegionSet, upright
from skitter.core.slicing.rows import place_rows
from skitter.core.slicing.structure import StructureField, structure_field

SAMPLES_PER_ROW = 8  # analysis resolution: samples across one row (a tile's short side)
MAX_SAMPLES = 16_000_000
ANALYSIS_SHARE = 0.5  # of a region's work: finding its structure (then laying the rows)


@register_operation
class ContourSlicer(Subdivider):
    id = "contour"
    name = "Contour Rows"
    description = (
        "Lay tiles in rows that follow the image's edges and outlines and ripple outward "
        "from them, like the rows of a Roman mosaic. Rows nearer an edge lie on top; "
        "tiles underneath fill the gaps where rows meet."
    )

    tile_size = TileSizeParam(1.0, "Tile size", help="Tile size in base tiles.")
    strength = FloatParam(
        0.25, "Edge strength", min=0.02, max=1.0, step=0.05,
        help="How strong an edge must be to guide rows, as a share of the image's "
             "strongest edges. Lower follows fainter edges.",
    )  # fmt: skip
    smoothness = FloatParam(
        1.5, "Smoothness", min=0.0, max=10.0, step=0.25, suffix=" tiles",
        help="Blur before finding edges. More ignores fine detail, for smoother rows.",
    )  # fmt: skip
    min_length = FloatParam(
        3.0, "Shortest edge", min=0.0, max=100.0, step=0.5, suffix=" tiles",
        help="Edges shorter than this don't guide rows (they would ripple around noise).",
    )  # fmt: skip
    follow_mask = BoolParam(
        True, "Follow mask edge",
        help="Rows also follow the edge of the picture where the source is transparent.",
    )  # fmt: skip
    max_overlap = FloatParam(
        0.2, "Overlap", min=0.0, max=0.9, step=0.05,
        help="How much of a tile may lie over tiles already laid before its row stops.",
    )  # fmt: skip
    outline_rows = IntParam(
        3, "Outline rows", min=0, max=100,
        help="How many rows ripple out from each outline; past them, a plain background "
             "of straight rows (0: rows ripple everywhere).",
    )  # fmt: skip
    background_angle = FloatParam(
        0.0, "Background angle", min=-90.0, max=90.0, step=15.0, decimals=0, suffix="°",
        when=lambda op: op.outline_rows > 0,
        help="Which way the background's straight rows run, clockwise (0°: across). "
             "Also where the image has no outlines at all.",
    )  # fmt: skip
    follow_texture = BoolParam(
        True, "Follow texture",
        help="Where the image has a strong grain (stripes, bands, fine texture) away from "
             "the outlines, lay rows along it instead of rippling across it.",
    )  # fmt: skip
    fill_gaps = BoolParam(
        True, "Fill gaps",
        help="Cover what rows leave uncovered with tiles underneath them.",
    )  # fmt: skip

    def structure(
        self, region: Region, ctx: SliceContext
    ) -> tuple[StructureField, np.ndarray, float]:
        """(field, visible samples, samples per mosaic unit): what rows follow in a region,
        sampled in its own frame (see structure.py). The Slicing tab draws it."""
        short = min(ctx.tile_dims(self.tile_size))
        samples = min(int(region.area * (SAMPLES_PER_ROW / short) ** 2) + 1, MAX_SAMPLES)
        luminance, scale = ctx.patch(region, "luminance", max_samples=samples, antialias=True)
        reference = None
        if self.follow_texture:  # what counts as strong: before antialiasing
            reference, _ = ctx.patch(region, "luminance", max_samples=samples)
        visible, _ = ctx.patch(region, "visible", max_samples=samples)
        spacing = short * scale  # samples per row
        field = structure_field(
            luminance, spacing, smoothness=self.smoothness * spacing, strength=self.strength,
            min_length=self.min_length * spacing, visible=visible, follow_mask=self.follow_mask,
            follow_texture=self.follow_texture, reference=reference,
            outline_rows=self.outline_rows, background_angle=math.radians(self.background_angle),
        )  # fmt: skip
        return field, visible, scale

    def subdivide(
        self, region: Region, ctx: SliceContext, progress: Progress = NO_PROGRESS
    ) -> RegionSet:
        tile_w, tile_h = ctx.tile_dims(self.tile_size)
        long, short = max(tile_w, tile_h), min(tile_w, tile_h)
        check_region_count(math.ceil(1.5 * region.area / (tile_w * tile_h)), self.name)
        progress(0.0)
        field, visible, scale = self.structure(region, ctx)
        progress(ANALYSIS_SHARE)
        centers, angles, z = place_rows(
            field, visible, long * scale, short * scale, self.max_overlap, fill=self.fill_gaps
        )
        turn = math.pi / 2 if tile_h > tile_w else 0.0  # the long side along the row
        size = np.broadcast_to([tile_w, tile_h], (len(z), 2))
        # A row has no direction: tiles along it may as well face as near up as their
        # outline allows (a square: within a quarter turn either way).
        period = math.pi / 2 if math.isclose(tile_w, tile_h) else math.pi
        return RegionSet.from_arrays(centers / scale, size, upright(angles + turn, period), z)

    def summary(self) -> str:
        size = "" if self.tile_size == 1 else f"{self.tile_size:g}× tiles, "
        return f"{size}edges ≥ {self.strength:.0%}"

"""The source's visibility mask applied to slicing results.

Hidden pixels (alpha under half, see imaging.py) lie outside the picture,
as if beyond its border. Slicing fills the whole canvas, and layouts treat
the border by keeping every tile that touches the picture, whole: so after
the plan, tiles that touch no visible pixel are dropped, and the rest stay
whole, overhanging the mask's edge as edge tiles overhang the border.
(Colors past the edge are handled by SliceContext, which gives hidden
pixels the color of the nearest visible one.)
"""

import numpy as np

from skitter.core.slicing.base import SliceContext
from skitter.core.slicing.regions import RegionSet

SPACING = 0.7  # source pixels between test points: no visible pixel inside fits between
MAX_SAMPLES = 4_000_000  # test points at a time
INSET = 1e-6  # source pixels: an edge on a pixel boundary (give or take rounding) misses it


def touching(regions: RegionSet, ctx: SliceContext) -> np.ndarray:
    """(R,) bool: regions covering at least one visible source pixel.

    A region's bounding box (in source pixels) settles most cases exactly
    from a summed-area table: no visible pixel in it, none in the region;
    for an upright region the box is the region. Rotated regions whose box
    is partly visible are tested at points at most SPACING apart over the
    region, edges included, so no visible pixel lying inside it fits between
    them. Points past the image's border read its edge.
    An edge lying on a pixel boundary doesn't touch the pixel beyond it.
    """
    n = len(regions)
    visible = ctx.visible
    if visible is None or n == 0:
        return np.ones(n, bool)
    h, w = visible.shape
    table = ctx.visible_table
    box = regions.bounds() / ctx.scale  # (R, 4) x0, y0, x1, y1 in source pixels
    i0 = np.clip(np.floor(box[:, 0] + INSET), 0, w).astype(np.int64)
    j0 = np.clip(np.floor(box[:, 1] + INSET), 0, h).astype(np.int64)
    i1 = np.clip(np.ceil(box[:, 2] - INSET), 0, w).astype(np.int64)
    j1 = np.clip(np.ceil(box[:, 3] - INSET), 0, h).astype(np.int64)
    # Past the border: the edge pixels (as sampling does).
    i1, j1 = np.maximum(i1, np.minimum(i0 + 1, w)), np.maximum(j1, np.minimum(j0 + 1, h))
    i0, j0 = np.minimum(i0, i1 - 1), np.minimum(j0, j1 - 1)
    count = table[j1, i1].astype(np.int64) - table[j0, i1] - table[j1, i0] + table[j0, i0]
    keep = count > 0
    upright = np.abs(np.sin(2.0 * regions.rotation)) < 1e-9  # its box is itself
    unsure = np.flatnonzero(keep & ~upright & (count < (i1 - i0) * (j1 - j0)))
    # Regions needing as many points per side go together.
    points = np.ceil(regions.size[unsure].max(axis=1) / ctx.scale / SPACING).astype(np.int64) + 1
    for side in np.unique(points):
        group = unsure[points == side]
        u = np.linspace(-0.5, 0.5, side)
        step = max(1, MAX_SAMPLES // (side * side))
        for lo in range(0, len(group), step):
            rows = group[lo : lo + step]
            size = np.maximum(regions.size[rows] - 2 * INSET * ctx.scale, 0.0)
            rot = regions.rotation[rows]
            lx = u[None, None, :] * size[:, 0, None, None]
            ly = u[None, :, None] * size[:, 1, None, None]
            c, s = np.cos(rot)[:, None, None], np.sin(rot)[:, None, None]
            x = (regions.center[rows, 0, None, None] + c * lx - s * ly) / ctx.scale
            y = (regions.center[rows, 1, None, None] + s * lx + c * ly) / ctx.scale
            xi = np.clip(np.floor(x), 0, w - 1).astype(np.int64)
            yi = np.clip(np.floor(y), 0, h - 1).astype(np.int64)
            keep[rows] = visible[yi, xi].any(axis=(1, 2))
    return keep


def mask_regions(regions: RegionSet, ctx: SliceContext) -> RegionSet:
    """The regions that touch the visible picture (all of them without a mask)."""
    if ctx.visible is None:
        return regions
    return regions[np.flatnonzero(touching(regions, ctx))]

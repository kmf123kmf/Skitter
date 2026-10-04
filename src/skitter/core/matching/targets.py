"""Target descriptors: what each region of the source image looks like.

Each region is sampled on a regular grid in its own (rotated) frame and
described like a tile crop (see tiles/descriptors.py). Two things make a
region's descriptor less certain than a tile's, and its `mask` records them
as per-dimension weights in [0, 1]:

- Overlap: parts of a region hidden under higher regions don't show in the
  mosaic, so those cells don't count. A fully hidden region needs no tile.
- Size: a cell covering fewer than a couple of source pixels carries no real
  detail, so small regions match on the coarser levels only.

Hidden or masked cells get a structure value of zero (the region's average),
so approximate search, which can't weight per query, leans toward tiles that
are flat there instead of guessing.
"""

from dataclasses import dataclass

import numpy as np

from skitter.core.color import rgb8_to_oklab
from skitter.core.matching.raster import Raster
from skitter.core.slicing import RegionSet, SliceContext
from skitter.core.tiles.descriptors import DIM, GRID, LEVEL1, LEVEL2, MEAN, TEXTURE, assemble

SAMPLES = 6  # samples per axis in each 4 x 4 cell
MIN_SOURCE_PX = 2.0  # source pixels a cell needs (each way) to count
MIN_VISIBLE = 0.01  # regions showing less than this share need no tile


@dataclass(frozen=True)
class Targets:
    desc: np.ndarray  # (R, DIM) float32
    mask: np.ndarray  # (R, DIM) float32 weights in [0, 1]
    visible: np.ndarray  # (R,) share of each region that shows

    @property
    def needed(self) -> np.ndarray:
        """Regions that show enough to need a tile."""
        return self.visible >= MIN_VISIBLE


def target_descriptors(
    regions: RegionSet,
    ctx: SliceContext,
    raster: Raster | None = None,
    chunk: int = 2048,
    indices: np.ndarray | None = None,
) -> Targets:
    """Describe every region of the final image (raster: visibility, None = all visible).

    indices: describe only these regions (rows of the result, in this order).
    """
    which = np.arange(len(regions)) if indices is None else np.asarray(indices, np.int64)
    n = len(which)
    desc = np.zeros((n, DIM), np.float32)
    mask = np.zeros((n, DIM), np.float32)
    visible = np.zeros(n, np.float32)
    side = GRID * SAMPLES
    u = (np.arange(side) + 0.5) / side
    src_h, src_w = ctx.image.shape[:2]

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        rows = which[lo:hi]
        center, size, rot = regions.center[rows], regions.size[rows], regions.rotation[rows]
        m = hi - lo
        # Sample points in each region's frame, centered on the region.
        lx = (u[None, None, :] - 0.5) * size[:, 0, None, None]
        ly = (u[None, :, None] - 0.5) * size[:, 1, None, None]
        c, s = np.cos(rot)[:, None, None], np.sin(rot)[:, None, None]
        world = np.stack(
            [
                center[:, 0, None, None] + c * lx - s * ly,
                center[:, 1, None, None] + s * lx + c * ly,
            ],
            axis=-1,
        )  # (m, side, side, 2)
        xi = np.clip(np.floor(world[..., 0] / ctx.scale), 0, src_w - 1).astype(np.int64)
        yi = np.clip(np.floor(world[..., 1] / ctx.scale), 0, src_h - 1).astype(np.int64)
        lab = rgb8_to_oklab(ctx.image[yi, xi])  # (m, side, side, 3)
        if raster is None:
            seen = np.ones((m, side, side), np.float32)
        else:
            seen = (raster.lookup(world) == rows[:, None, None]).astype(np.float32)

        # Visibility-weighted cell means.
        cells = (m, GRID, SAMPLES, GRID, SAMPLES)
        wt = seen.reshape(cells)
        cell_vis = wt.sum(axis=(2, 4))  # (m, 4, 4)
        sums = (lab.reshape(*cells, 3) * wt[..., None]).sum(axis=(2, 4))
        vis_frac = cell_vis.sum(axis=(1, 2)) / (side * side)
        any_seen = cell_vis.sum(axis=(1, 2)) > 0
        mean = sums.sum(axis=(1, 2)) / np.maximum(cell_vis.sum(axis=(1, 2)), 1)[:, None]
        grid4 = np.where(
            cell_vis[..., None] > 0, sums / np.maximum(cell_vis, 1)[..., None], mean[:, None, None]
        )
        # Lightness spread per 2 x 2 cell, over visible samples.
        half = (m, 2, 2 * SAMPLES, 2, 2 * SAMPLES)
        w2 = seen.reshape(half)
        light = lab[..., 0].reshape(half)
        n2 = w2.sum(axis=(2, 4))
        m1 = (light * w2).sum(axis=(2, 4)) / np.maximum(n2, 1)
        m2 = (light * light * w2).sum(axis=(2, 4)) / np.maximum(n2, 1)
        spread = np.sqrt(np.maximum(m2 - m1 * m1, 0)) * (n2 > 0)

        d = assemble(grid4.astype(np.float32), spread.astype(np.float32))
        # Cells too small to carry detail: no structure.
        cell_src = size.min(axis=1) / ctx.scale / GRID  # source px across a 4 x 4 cell
        fine_ok = (cell_src >= MIN_SOURCE_PX).astype(np.float32)
        coarse_ok = (2 * cell_src >= MIN_SOURCE_PX).astype(np.float32)
        tex_ok = (2 * cell_src >= 2 * MIN_SOURCE_PX).astype(np.float32)

        frac4 = cell_vis / (SAMPLES * SAMPLES)
        frac2 = frac4.reshape(m, 2, 2, 2, 2).mean(axis=(2, 4))
        mk = np.zeros((m, DIM), np.float32)
        mk[:, MEAN] = any_seen[:, None]
        mk[:, LEVEL1] = np.repeat(frac2.reshape(m, 4), 3, axis=1) * coarse_ok[:, None]
        mk[:, LEVEL2] = np.repeat(frac4.reshape(m, 16), 3, axis=1) * fine_ok[:, None]
        mk[:, TEXTURE] = frac2.reshape(m, 4) * tex_ok[:, None]
        structure = slice(LEVEL1.start, DIM)
        d[:, structure] *= mk[:, structure] > 0  # unknown structure reads as flat

        desc[lo:hi], mask[lo:hi], visible[lo:hi] = d, mk, vis_frac
    return Targets(desc, mask, visible)

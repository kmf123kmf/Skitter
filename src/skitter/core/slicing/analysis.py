"""Measurements of a slicing result: coverage, density, tile sizes."""

import math
from dataclasses import dataclass

import numpy as np

from skitter.core.slicing.base import SliceContext
from skitter.core.slicing.regions import RegionSet

MAX_COVERAGE_TESTS = 4_000_000  # point-in-region tests per coverage estimate


def coverage(
    regions: RegionSet, width: float, height: float, spacing: float, ctx: SliceContext | None = None
) -> tuple[float, np.ndarray]:
    """Estimate the fraction of the canvas (0, 0, width, height) covered by regions
    (with ctx, of its visible part: samples on hidden pixels don't count).

    Divides the canvas into sample cells about `spacing` apart (wider if
    needed to stay within MAX_COVERAGE_TESTS) and tests one point per cell at
    a fixed pseudo-random position inside it. Random placement keeps the
    estimate unbiased even when regions line up with the sample grid (a
    regular grid can miss every gap between equal tiles). Returns the covered
    fraction and the boolean (rows, columns) sample grid. Each region is
    tested only against cells overlapping its bounding box, so the work grows
    with the number of regions, not with regions x samples.
    """
    if not regions:
        return 0.0, np.zeros((1, 1), dtype=bool)
    bounds = regions.bounds()
    box_area = np.clip(bounds[:, 2] - bounds[:, 0], 0, width) * np.clip(
        bounds[:, 3] - bounds[:, 1], 0, height
    )
    spacing = max(spacing, math.sqrt(box_area.sum() / MAX_COVERAGE_TESTS), 1e-9)
    nx = max(1, math.ceil(width / spacing))
    ny = max(1, math.ceil(height / spacing))
    sx, sy = width / nx, height / ny
    covered = np.zeros((ny, nx), dtype=bool)
    jitter = np.random.default_rng(0).random((ny, nx, 2))  # sample position within each cell

    # Cells overlapping each region's bounding box (clipped to the canvas).
    i0 = np.clip(np.floor(bounds[:, 0] / sx), 0, nx).astype(np.int64)
    i1 = np.clip(np.ceil(bounds[:, 2] / sx), 0, nx).astype(np.int64)
    j0 = np.clip(np.floor(bounds[:, 1] / sy), 0, ny).astype(np.int64)
    j1 = np.clip(np.ceil(bounds[:, 3] / sy), 0, ny).astype(np.int64)
    span_x = np.maximum(i1 - i0, 0)
    counts = span_x * np.maximum(j1 - j0, 0)
    total = int(counts.sum())
    if total == 0:
        return 0.0, covered

    # Expand every (region, sample-in-its-box) pair without a Python loop.
    owner = np.repeat(np.arange(len(regions)), counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    local = np.arange(total) - starts
    ix = i0[owner] + local % span_x[owner]
    iy = j0[owner] + local // span_x[owner]

    point = np.stack([(ix + jitter[iy, ix, 0]) * sx, (iy + jitter[iy, ix, 1]) * sy], axis=-1)
    offset = point - regions.center[owner]
    angle = -regions.rotation[owner]
    c, s = np.cos(angle), np.sin(angle)
    lx = c * offset[:, 0] - s * offset[:, 1]
    ly = s * offset[:, 0] + c * offset[:, 1]
    half = regions.size[owner] / 2
    inside = (np.abs(lx) <= half[:, 0]) & (np.abs(ly) <= half[:, 1])
    covered[iy[inside], ix[inside]] = True
    if ctx is None or ctx.visible is None:
        return float(covered.mean()), covered
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    h, w = ctx.visible.shape
    px = np.clip(np.floor((gx + jitter[..., 0]) * sx / ctx.scale), 0, w - 1).astype(np.int64)
    py = np.clip(np.floor((gy + jitter[..., 1]) * sy / ctx.scale), 0, h - 1).astype(np.int64)
    seen = ctx.visible[py, px]
    return float(covered[seen].mean()) if seen.any() else 0.0, covered


@dataclass(frozen=True)
class SliceSummary:
    count: int
    grid_tiles: float  # base tiles that fit on the visible canvas (its area / tile area)
    coverage: float  # estimated fraction of the visible canvas covered, 0..1
    smallest: tuple[float, float]  # region (w, h) in mosaic units, by area
    median: tuple[float, float]
    largest: tuple[float, float]
    source_px_per_tile: float  # source pixels across one base tile

    @property
    def density(self) -> float:
        """Regions relative to a plain grid of base tiles (a grid is ~1.0)."""
        return self.count / self.grid_tiles


def summarize(regions: RegionSet, ctx: SliceContext) -> SliceSummary:
    tile_w, tile_h = ctx.tile_size
    area = regions.area()
    order = np.argsort(area, kind="stable")

    def size_at(i: int) -> tuple[float, float]:
        w, h = regions.size[order[i]]
        return (float(w), float(h))

    fraction, _ = coverage(regions, ctx.width, ctx.height, spacing=min(tile_w, tile_h) / 8, ctx=ctx)
    shown = 1.0 if ctx.visible is None else float(ctx.visible.mean())
    return SliceSummary(
        count=len(regions),
        grid_tiles=shown * ctx.width * ctx.height / (tile_w * tile_h),
        coverage=fraction,
        smallest=size_at(0),
        median=size_at(len(order) // 2),
        largest=size_at(-1),
        source_px_per_tile=tile_w / ctx.scale,
    )

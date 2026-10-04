"""How good the whole mosaic is: a proxy render scored like a viewer sees it.

No tile images are loaded. Each region is painted, in stacking order, with
its chosen crop's 4 x 4 cell colors (tinted), on a raster of a few pixels per
cell. The result is compared with the final image at the same resolution in
OKLab after blurring at several scales measured in base tiles: a mosaic is
viewed from a distance, where neighboring errors partly cancel and only the
blurred error is visible.

The report gives an overall score (mean ΔE over the scales, lower is
better), the error at each scale, structural similarity of lightness, and a
per-region error used to focus extra search effort and for the heat map.

`rescore` updates a report after a few regions change their tiles (manual
picks) by redoing the work only in a window around each: a change can only
affect pixels within the widest blur's reach. The result equals a full
evaluation up to rounding.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from skitter.core.color import rgb8_to_oklab
from skitter.core.matching.raster import RasterGrid
from skitter.core.slicing import RegionSet, SliceContext

SCALES = (0.5, 1.0, 2.0)  # blur sigmas, in base tiles
PX_PER_TILE = 8  # raster resolution: pixels across a base tile's short side
MAX_PIXELS = 4_000_000
TRUNCATE = 4.0  # gaussian_filter's default reach, in sigmas
SSIM_SIGMA = 1.5


@dataclass(frozen=True)
class QualityReport:
    score: float  # mean blurred ΔE (x100 OKLab) over SCALES; lower is better
    by_scale: dict[float, float]
    ssim: float  # lightness structural similarity at the finest scale, 1 = identical
    region_error: np.ndarray  # (R,) mean ΔE over each region's visible pixels (nan: none)
    covered: float  # share of the canvas painted by placed regions
    extra: dict = field(default_factory=dict)


@njit(parallel=True, cache=True, nogil=True)
def _paint(ids, x0, y0, px, centers, sizes, rotations, cells, placed, out, hit):
    height, width = ids.shape
    for j in prange(height):
        for i in range(width):
            r = ids[j, i]
            if r < 0 or not placed[r]:
                continue
            dx = x0 + (i + 0.5) * px - centers[r, 0]
            dy = y0 + (j + 0.5) * px - centers[r, 1]
            c, s = math.cos(rotations[r]), math.sin(rotations[r])
            u = (c * dx + s * dy) / sizes[r, 0] + 0.5
            v = (-s * dx + c * dy) / sizes[r, 1] + 0.5
            cx = min(max(int(u * 4), 0), 3)
            cy = min(max(int(v * 4), 0), 3)
            for k in range(3):
                out[j, i, k] = cells[r, cy, cx, k]
            hit[j, i] = True


def _blur(values, weights, sigma):
    """Normalized Gaussian blur of (H, W, C) values over (H, W) weights."""
    w = gaussian_filter(weights, sigma, mode="constant")
    v = np.stack(
        [gaussian_filter(values[..., c] * weights, sigma, mode="constant") for c in
         range(values.shape[-1])], axis=-1,
    )  # fmt: skip
    return v / np.maximum(w, 1e-6)[..., None], w


def _ssim(a, b, mask, sigma=SSIM_SIGMA):
    s = _ssim_map(a, b, mask, sigma)
    return float(s[mask].mean()) if mask.any() else 0.0


def _ssim_map(a, b, mask, sigma=SSIM_SIGMA):
    c1, c2 = 0.01**2, 0.03**2
    m = mask.astype(np.float64)

    def blur(x):
        return gaussian_filter(x * m, sigma) / np.maximum(gaussian_filter(m, sigma), 1e-6)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    return numerator / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))


def target_raster(ctx: SliceContext, width: int, height: int, window=None) -> np.ndarray:
    """The final image resampled to (height, width) OKLab over the canvas.

    window: only pixels (i0, j0, i1, j1) of that raster.
    """
    image = np.asarray(ctx.image)
    if window is None:
        img = Image.fromarray(image).resize((width, height), Image.Resampling.BOX)
        return rgb8_to_oklab(np.asarray(img))
    i0, j0, i1, j1 = window
    sx, sy = image.shape[1] / width, image.shape[0] / height
    box = (i0 * sx, j0 * sy, i1 * sx, j1 * sy)
    x0, y0 = int(math.floor(box[0])), int(math.floor(box[1]))
    x1 = min(int(math.ceil(box[2])), image.shape[1])
    y1 = min(int(math.ceil(box[3])), image.shape[0])
    img = Image.fromarray(np.ascontiguousarray(image[y0:y1, x0:x1])).resize(
        (i1 - i0, j1 - j0), Image.Resampling.BOX,
        box=(box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0),
    )  # fmt: skip
    return rgb8_to_oklab(np.asarray(img))


def raster_size(ctx: SliceContext) -> tuple[float, int, int]:
    """(px, nx, ny): the evaluation raster's pixel size (mosaic units) and shape."""
    tile_short = min(ctx.tile_size)
    px = max(tile_short / PX_PER_TILE, math.sqrt(ctx.width * ctx.height / MAX_PIXELS))
    nx = max(1, round(ctx.width / px))
    px = ctx.width / nx
    ny = max(1, round(ctx.height / px))
    return px, nx, ny


def evaluate(
    regions: RegionSet,
    ctx: SliceContext,
    cells: np.ndarray,
    placed: np.ndarray,
    target: np.ndarray | None = None,
) -> QualityReport:
    """Score a mosaic whose region r shows cells[r] (4 x 4 OKLab), if placed[r].

    target: precomputed target_raster for the same canvas (optional).
    """
    tile_short = min(ctx.tile_size)
    px, nx, ny = raster_size(ctx)
    ids = _grid(regions, ctx).paint(regions, 0, 0, nx, ny).ids
    if target is None or target.shape[:2] != (ny, nx):
        target = target_raster(ctx, nx, ny)

    proxy = np.zeros((ny, nx, 3), np.float32)
    hit = np.zeros((ny, nx), bool)
    if len(regions):
        _paint(
            np.ascontiguousarray(ids), 0.0, 0.0, px, regions.center, regions.size,
            regions.rotation, np.ascontiguousarray(cells, dtype=np.float32),
            np.asarray(placed, bool), proxy, hit,
        )  # fmt: skip
    weight = hit.astype(np.float64)
    by_scale = {}
    valid_counts = {}
    fine_error = None
    for scale in SCALES:
        sigma = scale * tile_short / px
        a, wa = _blur(proxy, weight, sigma)
        b, _ = _blur(target, weight, sigma)
        de = 100 * np.sqrt(((a - b) ** 2).sum(-1))
        valid = wa > 0.5 * gaussian_filter(np.ones_like(weight), sigma, mode="constant")
        valid &= hit
        by_scale[scale] = float(de[valid].mean()) if valid.any() else float("nan")
        valid_counts[scale] = int(valid.sum())
        if fine_error is None:
            fine_error = np.where(hit, de, 0.0)

    flat = ids.ravel()
    owned = flat >= 0
    sums = np.bincount(flat[owned], weights=fine_error.ravel()[owned], minlength=len(regions))
    counts = np.bincount(flat[owned], minlength=len(regions))
    with np.errstate(invalid="ignore", divide="ignore"):
        region_error = np.where(counts > 0, sums / counts, np.nan)
    score = float(np.nanmean(list(by_scale.values()))) if hit.any() else float("nan")
    return QualityReport(
        score=score,
        by_scale=by_scale,
        ssim=_ssim(proxy[..., 0], target[..., 0], hit),
        region_error=region_error.astype(np.float32),
        covered=float(hit.mean()),
        extra={
            "raster_shape": (ny, nx),
            "valid": valid_counts,
            "pixels": counts.astype(np.int32),
            "hit": int(hit.sum()),
        },  # fmt: skip
    )


def _grid(regions: RegionSet, ctx: SliceContext) -> RasterGrid:
    px, nx, ny = raster_size(ctx)
    grid = RasterGrid.of(regions, bounds=(0, 0, ctx.width, ny * px), px=px)
    return RasterGrid(grid.origin, grid.px, (ny, nx))


def _reach(sigma: float) -> int:
    """Pixels a gaussian_filter of this sigma reaches (scipy's default truncation)."""
    return int(TRUNCATE * sigma + 0.5) + 1


def _gauss(values: np.ndarray, sigma: float) -> np.ndarray:
    """gaussian_filter with zero padding over the first two axes (channels independent)."""
    sigmas = (sigma, sigma) + (0,) * (values.ndim - 2)
    return gaussian_filter(values, sigmas, mode="constant")


def _ones_blur(shape, window, full_shape, sigma) -> np.ndarray:
    """gaussian_filter(ones(full_shape)) over a window (i0, j0, i1, j1): separable."""
    i0, j0, i1, j1 = window
    rows = gaussian_filter1d(np.ones(full_shape[0], np.float32), sigma, mode="constant")[j0:j1]
    cols = gaussian_filter1d(np.ones(full_shape[1], np.float32), sigma, mode="constant")[i0:i1]
    return rows[:, None] * cols[None, :]


def rescore(
    report: QualityReport,
    regions: RegionSet,
    ctx: SliceContext,
    placed: np.ndarray,
    cells_before: Callable[[np.ndarray], np.ndarray],
    cells_after: Callable[[np.ndarray], np.ndarray],
    changed,
) -> QualityReport:
    """The report after the `changed` regions' cells went from cells_before to cells_after.

    cells_*(ids) give the (len(ids), 4, 4, 3) cells those regions show.
    Regions are updated one at a time. For each, a scale's blurred error can
    only change within its blur's reach of the region, and computing that
    needs the image another reach further out; nothing else is touched.
    """
    px, nx, ny = raster_size(ctx)
    grid = _grid(regions, ctx)
    sigmas = [scale * min(ctx.tile_size) / px for scale in SCALES]
    widest = max(_reach(s) for s in (*sigmas, SSIM_SIGMA))
    by_scale = dict(report.by_scale)
    region_error = report.region_error.copy()
    ssim = report.ssim
    extra = report.extra
    placed = np.asarray(placed, bool)
    done: set[int] = set()
    for r in np.asarray(changed, dtype=np.int64):
        r = int(r)
        box = grid.pixels(*regions[r : r + 1].bounds()[0])
        done.add(r)
        if box[0] >= box[2] or box[1] >= box[3]:
            continue

        def grow(window, by):
            i0, j0, i1, j1 = window
            return max(i0 - by, 0), max(j0 - by, 0), min(i1 + by, nx), min(j1 + by, ny)

        def cut(array, window, within):
            return array[window[1] - within[1] : window[3] - within[1],
                         window[0] - within[0] : window[2] - within[0]]  # fmt: skip

        outer = grow(box, 2 * widest)
        ids = grid.paint(regions, *outer).ids
        local, inverse = np.unique(ids, return_inverse=True)
        inverse = inverse.reshape(ids.shape).astype(np.int32)
        if len(local) and local[0] < 0:
            local, inverse = local[1:], inverse - 1  # -1 stays outside every region
        old = cells_before(local)  # with every earlier change in place
        after = cells_after(local)
        earlier = np.isin(local, list(done - {r}))
        old[earlier] = after[earlier]
        mine = local == r
        new = old.copy()
        new[mine] = after[mine]

        origin = (grid.origin[0] + outer[0] * px, grid.origin[1] + outer[1] * px)
        proxy_old = np.zeros((*ids.shape, 3), np.float32)
        proxy_new = np.zeros_like(proxy_old)
        hit = np.zeros(ids.shape, bool)
        for cells, proxy in ((old, proxy_old), (new, proxy_new)):
            _paint(
                inverse, origin[0], origin[1], px, regions.center[local], regions.size[local],
                regions.rotation[local], np.ascontiguousarray(cells, dtype=np.float32),
                placed[local], proxy, hit,
            )  # fmt: skip
        target = target_raster(ctx, nx, ny, outer).astype(np.float32)
        weight = hit.astype(np.float32)
        change = (proxy_new - proxy_old) * weight[..., None]

        for k, (scale, sigma) in enumerate(zip(SCALES, sigmas, strict=True)):
            reach = _reach(sigma)
            inner = grow(box, reach)  # where this scale's error can change
            source = grow(inner, reach)  # what computing it there reads
            wt = cut(weight, source, outer)
            w = cut(_gauss(wt, sigma), inner, source)
            norm = np.maximum(w, 1e-6)[..., None]
            b = cut(_gauss(cut(target, source, outer) * wt[..., None], sigma), inner, source) / norm
            a_old = cut(_gauss(cut(proxy_old, source, outer) * wt[..., None], sigma), inner,
                        source) / norm  # fmt: skip
            a_new = a_old + _gauss(cut(change, inner, outer), sigma) / norm  # change: inside box
            de_old = 100 * np.sqrt(((a_old - b) ** 2).sum(-1))
            de_new = 100 * np.sqrt(((a_new - b) ** 2).sum(-1))
            seen = cut(hit, inner, outer)
            valid = seen & (w > 0.5 * _ones_blur(w.shape, inner, (ny, nx), sigma))
            if extra["valid"][scale]:
                delta = float(de_new[valid].sum() - de_old[valid].sum())
                by_scale[scale] += delta / extra["valid"][scale]
            if k == 0:  # the per-region error uses the finest scale
                owner = cut(inverse, inner, outer)
                owned = (owner >= 0) & seen
                sums = np.bincount(owner[owned], weights=(de_new - de_old)[owned],
                                   minlength=len(local))  # fmt: skip
                pixels = extra["pixels"][local]
                moved = np.flatnonzero((sums != 0) & (pixels > 0))
                region_error[local[moved]] += sums[moved] / pixels[moved]
        if extra["hit"]:  # lightness SSIM (reflecting edges: computed on the whole window)
            inner = grow(box, _reach(SSIM_SIGMA))
            source = grow(inner, _reach(SSIM_SIGMA))
            light, mask = cut(target, source, outer)[..., 0], cut(hit, source, outer)
            maps = [cut(_ssim_map(cut(p, source, outer)[..., 0], light, mask), inner, source)
                    for p in (proxy_old, proxy_new)]  # fmt: skip
            seen = cut(hit, inner, outer)
            ssim += float(maps[1][seen].sum() - maps[0][seen].sum()) / extra["hit"]
    score = float(np.nanmean(list(by_scale.values()))) if extra["hit"] else float("nan")
    return QualityReport(
        score=score, by_scale=by_scale, ssim=ssim, region_error=region_error,
        covered=report.covered, extra=extra,
    )  # fmt: skip

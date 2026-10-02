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
"""

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange
from PIL import Image
from scipy.ndimage import gaussian_filter

from skitter.core.color import rgb8_to_oklab
from skitter.core.matching.raster import rasterize
from skitter.core.slicing import RegionSet, SliceContext

SCALES = (0.5, 1.0, 2.0)  # blur sigmas, in base tiles
PX_PER_TILE = 8  # raster resolution: pixels across a base tile's short side
MAX_PIXELS = 4_000_000


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


def _ssim(a, b, mask, sigma=1.5):
    c1, c2 = 0.01**2, 0.03**2
    m = mask.astype(np.float64)

    def blur(x):
        return gaussian_filter(x * m, sigma) / np.maximum(gaussian_filter(m, sigma), 1e-6)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    s = numerator / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return float(s[mask].mean()) if mask.any() else 0.0


def target_raster(ctx: SliceContext, width: int, height: int) -> np.ndarray:
    """The final image resampled to (height, width) OKLab over the canvas."""
    img = Image.fromarray(np.asarray(ctx.image)).resize((width, height), Image.Resampling.BOX)
    return rgb8_to_oklab(np.asarray(img))


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
    px = max(tile_short / PX_PER_TILE, math.sqrt(ctx.width * ctx.height / MAX_PIXELS))
    nx = max(1, round(ctx.width / px))
    px = ctx.width / nx
    ny = max(1, round(ctx.height / px))
    raster = rasterize(regions, bounds=(0, 0, ctx.width, ny * px), px=px)
    ids = raster.ids[:ny, :nx]
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
    fine_error = None
    for scale in SCALES:
        sigma = scale * tile_short / px
        a, wa = _blur(proxy, weight, sigma)
        b, _ = _blur(target, weight, sigma)
        de = 100 * np.sqrt(((a - b) ** 2).sum(-1))
        valid = wa > 0.5 * gaussian_filter(np.ones_like(weight), sigma, mode="constant")
        valid &= hit
        by_scale[scale] = float(de[valid].mean()) if valid.any() else float("nan")
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
        extra={"raster_shape": (ny, nx)},
    )

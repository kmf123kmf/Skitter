"""Assembly: render the matched mosaic at full detail and save it.

Every region shows its chosen crop exactly as matching chose it: the crop
window of the original tile photo (read at full resolution, EXIF-upright,
resized to the region's output size with Lanczos), mirrored if chosen, and
tinted by matching's own model: each pixel's OKLab color shifted by the
region's tint (tinted average minus the crop's average). The tinted crop is
then placed, rotated and at sub-pixel position, with Catmull-Rom (bicubic)
sampling.

Edges are antialiased with SUB x SUB coverage samples per pixel. Regions
are drawn front to back and each pixel records which samples are already
covered, so a region only fills what the regions above it leave visible.
Neighboring tiles therefore meet without the background showing through
the seam, and hidden parts of lower regions cost nothing.

The image is rendered in horizontal strips so the tile crops held in memory
stay within CROP_BUDGET texels.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange
from PIL import Image

from skitter.core.color import LINEAR_LUT, rgb8_to_oklab_nb
from skitter.core.matching.matcher import MatchResult
from skitter.core.slicing import SliceContext
from skitter.core.slicing.params import ChoiceParam, Configurable, IntParam
from skitter.core.tiles.library import TileLibrary
from skitter.core.tiles.render import render_crops_or_thumbs

SUB = 4  # coverage samples per pixel along each axis
FULL = (1 << SUB * SUB) - 1  # every sample of a pixel covered
CROP_BUDGET = 64 * 2**20  # tile crop texels held at once (about 200 MB)
JPEG_MAX = 65_500  # largest JPEG side, pixels
BACKGROUNDS = {"white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128)}
TRANSPARENT = "transparent"
FORMATS = {"png": ("PNG", ".png"), "jpeg": ("JPEG", ".jpg")}

Progress = Callable[[str, float | None], None]


class ExportCancelled(Exception):
    pass


SIZE_BY = (("tile", "Tile width"), ("width", "Image width"), ("height", "Image height"))
SIZE_PARAMS = ("tile_px", "width_px", "height_px")  # the setting each SIZE_BY choice uses


class ExportSettings(Configurable):
    """How to export the mosaic. This is where the mosaic first gets a size in pixels."""

    format = ChoiceParam("png", "Format", choices=[("png", "PNG"), ("jpeg", "JPEG")])
    size_by = ChoiceParam(
        "tile", "Size by", choices=SIZE_BY,
        help="Set the image size from the width of a base tile, or from the whole "
             "image's width or height; the others follow.",
    )  # fmt: skip
    tile_px = IntParam(
        100, "Tile width", min=1, max=20_000, suffix=" px", when=lambda s: s.size_by == "tile",
        help="Width of a base tile in the image. Tiles are read from their original "
             "files, so large tiles keep real detail as long as the photos have it.",
    )  # fmt: skip
    width_px = IntParam(
        4000, "Image width", min=1, max=1_000_000, suffix=" px",
        when=lambda s: s.size_by == "width",
    )  # fmt: skip
    height_px = IntParam(
        3000, "Image height", min=1, max=1_000_000, suffix=" px",
        when=lambda s: s.size_by == "height",
    )  # fmt: skip
    framing = ChoiceParam(
        "frame", "Framing",
        choices=[("frame", "Image frame"), ("tiles", "Whole tiles")],
        help="Image frame trims tiles that hang past the edge of the image; "
             "Whole tiles keeps them, filling the corners with the background.",
    )  # fmt: skip
    background = ChoiceParam(
        "white", "Background",
        choices=[("white", "White"), ("black", "Black"), ("gray", "Gray"),
                 (TRANSPARENT, "Transparent")],
        available=lambda s, value: value != TRANSPARENT or s.format == "png",
        help="Shows wherever no tile covers the image: gaps between tiles, and the "
             "corners around overhanging tiles with Whole tiles framing. "
             "Transparent needs PNG.",
    )  # fmt: skip
    quality = IntParam(
        95, "JPEG quality", min=1, max=100, when=lambda s: s.format == "jpeg",
        help="Higher keeps more detail in a larger file. Colors are never subsampled.",
    )  # fmt: skip

    @property
    def alpha(self) -> bool:
        """Whether the image has an alpha channel."""
        return self.background == TRANSPARENT and self.format == "png"

    @property
    def extension(self) -> str:
        return FORMATS[self.format][1]


@dataclass(frozen=True)
class Frame:
    """The part of the mosaic exported, in mosaic units, and pixels per unit."""

    x: float
    y: float
    width: float
    height: float
    scale: float

    @property
    def size(self) -> tuple[int, int]:
        """Output (width, height) in pixels."""
        return (
            max(1, round(self.width * self.scale)),
            max(1, round(self.height * self.scale)),
        )


def export_frame(result: MatchResult, ctx: SliceContext, settings: ExportSettings) -> Frame:
    x, y, w, h = 0.0, 0.0, float(ctx.width), float(ctx.height)
    if settings.framing == "tiles":
        placed = result.tile >= 0
        if placed.any():
            bounds = result.regions.bounds()[placed]
            x, y = (float(v) for v in bounds[:, :2].min(axis=0))
            x1, y1 = bounds[:, 2:].max(axis=0)
            w, h = float(x1) - x, float(y1) - y
    if settings.size_by == "width":
        scale = settings.width_px / w
    elif settings.size_by == "height":
        scale = settings.height_px / h
    else:
        scale = settings.tile_px / ctx.tile_size[0]
    return Frame(x, y, w, h, scale)


def enlargement(result: MatchResult, photo_width, scale: float) -> np.ndarray:
    """How much each placed tile's crop is enlarged beyond its photo's resolution.

    photo_width is each library slot's upright image width (indexable by
    result.tile). Values above 1 mean the export shows the crop larger than
    the photo has pixels for.
    """
    placed = np.flatnonzero(result.tile >= 0)
    rect = result.rect[placed]
    available = (rect[:, 2] - rect[:, 0]) * np.asarray(photo_width)[result.tile[placed]]
    needed = result.regions.size[placed, 0] * scale
    return needed / np.maximum(available, 1e-9)


def check_size(size: tuple[int, int], settings: ExportSettings) -> str | None:
    """Why an image of this size can't be saved in the chosen format, if it can't."""
    if settings.format == "jpeg" and max(size) > JPEG_MAX:
        return (
            f"JPEG images can be at most {JPEG_MAX:,} pixels on a side; use PNG or a smaller size."
        )
    return None


@dataclass(frozen=True)
class TileFiles:
    """What rendering needs from the tile library (read on the library's thread)."""

    slots: np.ndarray  # (S,) sorted library slots
    paths: list[str]
    thumbs: np.ndarray  # (S, T, T, 3) fallback for unreadable files
    thumb_size: np.ndarray  # (S, 2)

    @classmethod
    def read(cls, library: TileLibrary, slots) -> "TileFiles":
        slots = np.unique(np.asarray(slots, dtype=np.int64))
        slots = slots[slots >= 0]
        return cls(
            slots, library.paths(slots), np.asarray(library.thumbs[slots]),
            library.thumb_size[slots].copy(),
        )  # fmt: skip

    def index(self, slots) -> np.ndarray:
        return np.searchsorted(self.slots, slots)


@dataclass
class ExportReport:
    size: tuple[int, int]
    tiles: int  # regions drawn
    failed: int  # tile files that could not be read (their thumbnails were used)


def render_mosaic(
    result: MatchResult,
    ctx: SliceContext,
    files: TileFiles,
    settings: ExportSettings,
    progress: Progress = lambda message, fraction: None,
    cancelled: Callable[[], bool] = lambda: False,
    workers: int | None = None,
) -> tuple[np.ndarray, ExportReport]:
    """The mosaic as an (H, W, 3) uint8 RGB image ((H, W, 4) RGBA if settings.alpha)."""
    frame = export_frame(result, ctx, settings)
    width, height = frame.size
    s = frame.scale
    regions = result.regions

    # Regions to draw, front to back, in output pixels.
    order = regions.stacking_order()[::-1]
    order = order[result.tile[order] >= 0]
    center = (regions.center[order] - (frame.x, frame.y)) * s
    size = regions.size[order] * s
    bounds = (regions.bounds()[order] - (frame.x, frame.y, frame.x, frame.y)) * s
    box = np.empty((len(order), 4), np.int64)
    box[:, :2] = np.floor(bounds[:, :2]) - 1
    box[:, 2:] = np.ceil(bounds[:, 2:]) + 1
    np.clip(box[:, 0::2], 0, width, out=box[:, 0::2])
    np.clip(box[:, 1::2], 0, height, out=box[:, 1::2])
    shown = (box[:, 2] > box[:, 0]) & (box[:, 3] > box[:, 1])
    order, center, size, box = order[shown], center[shown], size[shown], box[shown]
    n = len(order)

    rotation = regions.rotation[order]
    geom = np.column_stack([center, size / 2, np.cos(rotation), np.sin(rotation)]).astype(
        np.float64
    )
    crop_size = np.maximum(1, np.round(size)).astype(np.int64)  # (w, h) texels
    mirrored = result.mirrored[order].astype(np.bool_)
    shift = (result.tinted_mean() - result.tile_mean)[order].astype(np.float64)
    tinted = np.any(np.abs(shift) > 1e-6, axis=1)
    file_index = files.index(result.tile[order])
    rects = result.rect[order].astype(np.float64)

    # Strips of rows, each needing at most about CROP_BUDGET texels of crops.
    texels = crop_size[:, 0] * crop_size[:, 1]
    strips = max(1, math.ceil(float(texels.sum()) / CROP_BUDGET))
    strip_rows = max(64, math.ceil(height / strips))
    strips = math.ceil(height / strip_rows)

    channels = 4 if settings.alpha else 3
    image = np.empty((height, width, channels), np.uint8)
    background = np.array(BACKGROUNDS.get(settings.background, BACKGROUNDS["white"]), np.int64)
    crops: dict[int, np.ndarray] = {}
    failed: set[str] = set()
    by_top = np.argsort(box[:, 1], kind="stable")
    loaded = 0  # regions in by_top[:loaded] have been read
    for k in range(strips):
        top, bottom = k * strip_rows, min(height, (k + 1) * strip_rows)
        base = k / strips
        part = f" (part {k + 1} of {strips})" if strips > 1 else ""
        # Read the crops of regions that start above this strip's bottom.
        start = loaded
        while loaded < n and box[by_top[loaded], 1] < bottom:
            loaded += 1
        new = by_top[start:loaded]
        new = new[box[new, 3] > top]
        if len(new):
            fi = file_index[new]

            def report(done, total, base=base, part=part):
                progress(
                    f"Reading tile images: {done:,} of {total:,} files{part}",
                    base + 0.8 * done / max(total, 1) / strips,
                )

            rendered = render_crops_or_thumbs(
                [files.paths[i] for i in fi], rects[new], crop_size[new],
                files.thumbs[fi], files.thumb_size[fi],
                progress=report, cancelled=cancelled, workers=workers,
            )  # fmt: skip
            if rendered is None:
                raise ExportCancelled
            images, bad = rendered
            failed |= bad
            for r, crop in zip(new, images, strict=True):
                crops[int(r)] = crop
        if cancelled():
            raise ExportCancelled
        progress(f"Placing tiles{part}", base + 0.8 / strips)

        active = np.array(sorted(r for r in crops if box[r, 1] < bottom and box[r, 3] > top))
        active = active.astype(np.int64)  # front to back (order is front to back)
        offset = np.zeros(len(active) + 1, np.int64)
        offset[1:] = np.cumsum(texels[active] * 3)
        pixels = np.empty(int(offset[-1]), np.uint8)
        for j, r in enumerate(active):
            pixels[offset[j] : offset[j + 1]] = crops[int(r)].reshape(-1)
        need = tinted[active]
        _tint(pixels, offset, need, shift[active], LINEAR_LUT)

        rows = bottom - top
        mask = np.zeros((rows, width), np.uint16)
        acc = np.zeros((rows, width, 3), np.uint16)
        _composite(
            top, mask, acc, geom[active], box[active], crop_size[active], offset, pixels,
            mirrored[active],
        )  # fmt: skip
        _finish(mask, acc, background, image[top:bottom])

        for r in [r for r in crops if box[r, 3] <= bottom]:  # no longer needed
            del crops[r]
        if cancelled():
            raise ExportCancelled

    return image, ExportReport((width, height), n, len(failed))


def save_mosaic(image: np.ndarray, path: str | Path, settings: ExportSettings) -> None:
    error = check_size((image.shape[1], image.shape[0]), settings)
    if error:
        raise ValueError(error)
    pil = Image.fromarray(image, "RGBA" if image.shape[2] == 4 else "RGB")
    if settings.format == "jpeg":
        pil.save(path, "JPEG", quality=settings.quality, subsampling=0, optimize=True)
    else:
        pil.save(path, "PNG", compress_level=6)


# Kernels


@njit(cache=True, nogil=True, inline="always")
def _bits(m):
    count = 0
    while m:
        m &= m - 1
        count += 1
    return count


@njit(cache=True, nogil=True, inline="always")
def _catmull_rom(t):
    """Weights of the four samples around a point t in [0, 1) past the second."""
    t2 = t * t
    t3 = t2 * t
    return (
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2),
    )


@njit(cache=True, nogil=True, parallel=True)
def _composite(top, mask, acc, geom, box, crop_size, offset, pixels, mirrored):
    """Draw regions (front to back) into one strip's coverage mask and color sums.

    acc holds each pixel's color summed over its covered samples (sample
    count x color), so the final color is acc / SUB².
    """
    rows, width = mask.shape
    n = len(box)
    sub_x = np.empty(SUB * SUB)
    sub_y = np.empty(SUB * SUB)
    for k in range(SUB * SUB):
        sub_x[k] = (k % SUB + 0.5) / SUB - 0.5
        sub_y[k] = (k // SUB + 0.5) / SUB - 0.5
    reach = 0.5 * math.sqrt(2.0) + 1e-6  # farthest a sample lies from its pixel center
    for row in prange(rows):
        y = top + row
        py = y + 0.5
        wx = np.empty(4)
        wy = np.empty(4)
        for i in range(n):
            if y < box[i, 1] or y >= box[i, 3]:
                continue
            cx, cy, hw, hh, c, s = (
                geom[i, 0],
                geom[i, 1],
                geom[i, 2],
                geom[i, 3],
                geom[i, 4],
                geom[i, 5],
            )
            cw, ch = crop_size[i, 0], crop_size[i, 1]
            scale_x = cw / (2.0 * hw)
            scale_y = ch / (2.0 * hh)
            base = offset[i]
            flip = mirrored[i]
            for x in range(box[i, 0], box[i, 2]):
                covered = np.int64(mask[row, x])
                if covered == FULL:
                    continue
                dx = x + 0.5 - cx
                dy = py - cy
                lx = c * dx + s * dy  # in the region's frame
                ly = -s * dx + c * dy
                ex = abs(lx) - hw
                ey = abs(ly) - hh
                if ex >= reach or ey >= reach:
                    continue
                if ex <= -reach and ey <= -reach:
                    m = FULL
                else:
                    m = 0
                    for k in range(SUB * SUB):
                        sx = lx + c * sub_x[k] + s * sub_y[k]
                        sy = ly - s * sub_x[k] + c * sub_y[k]
                        if abs(sx) < hw and abs(sy) < hh:
                            m |= 1 << k
                m &= ~covered
                if m == 0:
                    continue
                count = _bits(m)

                # Color at the pixel center (nearest edge texels outside the region).
                u = (hw - lx if flip else lx + hw) * scale_x - 0.5
                v = (ly + hh) * scale_y - 0.5
                u0 = math.floor(u)
                v0 = math.floor(v)
                wx[0], wx[1], wx[2], wx[3] = _catmull_rom(u - u0)
                wy[0], wy[1], wy[2], wy[3] = _catmull_rom(v - v0)
                r = 0.0
                g = 0.0
                b = 0.0
                for j in range(4):
                    ty = min(max(v0 - 1 + j, 0), ch - 1)
                    row_r = 0.0
                    row_g = 0.0
                    row_b = 0.0
                    for q in range(4):
                        tx = min(max(u0 - 1 + q, 0), cw - 1)
                        p = base + (ty * cw + tx) * 3
                        row_r += wx[q] * pixels[p]
                        row_g += wx[q] * pixels[p + 1]
                        row_b += wx[q] * pixels[p + 2]
                    r += wy[j] * row_r
                    g += wy[j] * row_g
                    b += wy[j] * row_b
                acc[row, x, 0] += count * min(max(int(r + 0.5), 0), 255)
                acc[row, x, 1] += count * min(max(int(g + 0.5), 0), 255)
                acc[row, x, 2] += count * min(max(int(b + 0.5), 0), 255)
                mask[row, x] = covered | m


@njit(cache=True, nogil=True)
def _finish(mask, acc, background, out):
    """Turn coverage masks and color sums into pixels (RGBA if out has 4 channels)."""
    rows, width = mask.shape
    total = SUB * SUB
    alpha = out.shape[2] == 4
    for y in range(rows):
        for x in range(width):
            count = _bits(np.int64(mask[y, x]))
            if alpha:
                for ch in range(3):
                    out[y, x, ch] = (acc[y, x, ch] + count // 2) // count if count else 0
                out[y, x, 3] = (count * 255 + total // 2) // total
            else:
                rest = total - count
                for ch in range(3):
                    out[y, x, ch] = (acc[y, x, ch] + rest * background[ch] + total // 2) // total


@njit(cache=True, nogil=True, inline="always")
def _encode(c):
    """Linear light to an 8-bit sRGB level."""
    c = min(max(c, 0.0), 1.0)
    c = c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055
    return min(max(int(c * 255.0 + 0.5), 0), 255)


@njit(cache=True, nogil=True, parallel=True)
def _tint(pixels, offset, need, shift, lut):
    """Shift each crop's pixels by its OKLab tint (crops where need is set)."""
    for i in prange(len(need)):
        if not need[i]:
            continue
        dl, da, db = shift[i, 0], shift[i, 1], shift[i, 2]
        for p in range(offset[i], offset[i + 1], 3):
            lab_l, lab_a, lab_b = rgb8_to_oklab_nb(pixels[p], pixels[p + 1], pixels[p + 2], lut)
            lab_l += dl
            lab_a += da
            lab_b += db
            l_ = lab_l + 0.3963377774 * lab_a + 0.2158037573 * lab_b
            m_ = lab_l - 0.1055613458 * lab_a - 0.0638541728 * lab_b
            s_ = lab_l - 0.0894841775 * lab_a - 1.2914855480 * lab_b
            l_, m_, s_ = l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_
            pixels[p] = _encode(4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_)
            pixels[p + 1] = _encode(-1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_)
            pixels[p + 2] = _encode(-0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_)

"""What the tile library can paint: its colors, gaps and photos.

Everything here works from the library's analysis thumbnails, never the
original files, so it is quick enough to run whenever the library changes:

- `tile_colors`: each tile's average OKLab color.
- `color_map`: a hue x lightness grid at one colorfulness (chroma). Each cell
  gets the tile nearest its color, how far off that is, and how many tiles
  are near: cells no tile comes near are colors the library lacks.
- `coverage`: a source picture repainted with the nearest tile averages, and
  how far each pixel is from them; a quick preview of where a mosaic will
  struggle (matching also weighs structure, and tinting closes small gaps).
- `photo_stats`: photo sizes and shapes, and the largest base tile most
  photos can fill without enlarging.

Distances are OKLab Euclidean; ΔE values shown to people are 100 times that,
as in matching.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from skitter.core.color import oklab_to_linear, oklab_to_rgb8, rgb8_to_oklab
from skitter.core.imaging import visible_mask
from skitter.core.tiles.descriptors import MEAN, tile_descriptors

NEAR = 0.05  # OKLab distance counted as near (ΔE 5)
CHROMAS = (("muted", "Muted", 0.04), ("medium", "Medium", 0.09), ("vivid", "Vivid", 0.15))
HUES = 36  # hue columns (10° each)
LEVELS = np.linspace(0.95, 0.15, 12)  # lightness rows, light at the top
COVERAGE_SIDE = 480  # longest side of the coverage preview, pixels
DETAIL_SHARE = 0.9  # share of photos that must fill a tile for photo_stats.detail_px


@dataclass(frozen=True)
class TileColors:
    """The average OKLab color of each usable tile."""

    slots: np.ndarray  # (N,) library slots
    lab: np.ndarray  # (N, 3) float32

    def __len__(self) -> int:
        return len(self.slots)

    @cached_property
    def tree(self) -> cKDTree:
        return cKDTree(self.lab)

    def nearest(self, lab) -> tuple[np.ndarray, np.ndarray]:
        """(distance, index) of the tile color nearest each given OKLab color."""
        lab = np.asarray(lab, dtype=np.float64)
        distance, index = self.tree.query(lab.reshape(-1, 3))
        return distance.reshape(lab.shape[:-1]), index.reshape(lab.shape[:-1])


def tile_colors(library, progress=None) -> TileColors:
    """Average colors of the library's usable tiles (from their thumbnails).

    progress(done, total): tiles analyzed so far.
    """
    slots = library.ids
    whole = np.tile(np.array([0.0, 0.0, 1.0, 1.0], np.float32), (len(slots), 1))
    desc = tile_descriptors(library.thumbs, library.thumb_size, slots, whole, progress=progress)
    return TileColors(slots, np.ascontiguousarray(desc[:, MEAN]))


@dataclass(frozen=True)
class ColorMap:
    """A hue x lightness grid of colors at one chroma, and the library's reach.

    Column 0 holds the neutral grays; columns 1.. run round the hue circle
    from red (hue_of). Rows run from light to dark.
    """

    chroma: float
    target: np.ndarray  # (R, C, 3) OKLab color of each cell
    shown: np.ndarray  # (R, C) the color exists in sRGB (others are left empty)
    tile: np.ndarray  # (R, C) index into TileColors of the nearest tile
    distance: np.ndarray  # (R, C) OKLab distance to it
    near: np.ndarray  # (R, C) how many tiles are within NEAR

    @property
    def shape(self) -> tuple[int, int]:
        return self.shown.shape

    @property
    def reached(self) -> np.ndarray:
        """(R, C) cells some tile comes near."""
        return self.shown & (self.distance <= NEAR)

    def share_reached(self) -> float:
        """Share of the displayable cells some tile comes near."""
        return float(self.reached.sum() / max(self.shown.sum(), 1))


def hue_of(column: int, hues: int = HUES) -> float | None:
    """The hue (degrees, OKLab) of a color map column; None for the neutral column."""
    return None if column == 0 else (column - 1) * 360.0 / hues


def color_map(colors: TileColors, chroma: float, hues: int = HUES, levels=LEVELS) -> ColorMap:
    levels = np.asarray(levels, dtype=np.float64)
    angle = np.radians(np.arange(hues) * 360.0 / hues)
    a = np.concatenate([[0.0], chroma * np.cos(angle)])
    b = np.concatenate([[0.0], chroma * np.sin(angle)])
    rows, columns = len(levels), hues + 1
    target = np.empty((rows, columns, 3))
    target[..., 0] = levels[:, None]
    target[..., 1], target[..., 2] = a[None, :], b[None, :]
    linear = oklab_to_linear(target)
    shown = np.all((linear >= -1e-4) & (linear <= 1 + 1e-4), axis=-1)
    if len(colors):
        distance, tile = colors.nearest(target)
        flat = target.reshape(-1, 3)
        near = np.array(colors.tree.query_ball_point(flat, NEAR, return_length=True))
        near = near.reshape(rows, columns)
    else:
        distance = np.full((rows, columns), np.inf)
        tile = np.full((rows, columns), -1)
        near = np.zeros((rows, columns), np.int64)
    return ColorMap(chroma, target, shown, tile, distance, near)


@dataclass(frozen=True)
class Coverage:
    """A source picture repainted in the library's nearest average colors."""

    painted: np.ndarray  # (h, w, 4) uint8: nearest tile colors (hidden pixels transparent)
    error: np.ndarray  # (h, w) ΔE (x100 OKLab) to them; nan where hidden
    visible: np.ndarray  # (h, w) bool

    @property
    def mean_error(self) -> float:
        return float(np.nanmean(self.error)) if self.visible.any() else 0.0

    @property
    def share_far(self) -> float:
        """Share of the visible picture no tile color comes near."""
        if not self.visible.any():
            return 0.0
        return float((self.error[self.visible] > NEAR * 100).mean())


def coverage(image: np.ndarray, colors: TileColors, max_side: int = COVERAGE_SIDE) -> Coverage:
    """How near the library's tile colors come to each part of an (H, W, 3|4) picture,
    reduced to at most max_side pixels on its longest side."""
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BOX))
    visible = visible_mask(small)
    if visible is None:
        visible = np.ones(small.shape[:2], bool)
    lab = rgb8_to_oklab(small[..., :3])
    if len(colors):
        distance, index = colors.nearest(lab)
        painted = oklab_to_rgb8(colors.lab[index])
    else:
        distance = np.full(lab.shape[:2], np.inf)
        painted = np.zeros((*lab.shape[:2], 3), np.uint8)
    alpha = np.where(visible, 255, 0).astype(np.uint8)[..., None]
    error = np.where(visible, distance * 100, np.nan).astype(np.float32)
    return Coverage(np.concatenate([painted, alpha], axis=-1), error, visible)


@dataclass(frozen=True)
class PhotoStats:
    count: int
    median_side: int  # median short side, pixels
    small_side: int  # short side of the smallest tenth, pixels
    portrait: float  # shares of the photos by shape
    landscape: float
    square: float
    detail_px: int  # base tile width (px) DETAIL_SHARE of photos fill without enlarging


def photo_stats(width, height, tile_aspect: float) -> PhotoStats:
    """Statistics of photos of the given (upright) sizes, for base tiles of tile_aspect
    (width / height). A tile shows the largest window of its photo with its shape."""
    width = np.asarray(width, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    if not len(width):
        return PhotoStats(0, 0, 0, 0.0, 0.0, 0.0, 0)
    short = np.minimum(width, height)
    aspect = width / np.maximum(height, 1)
    square = np.abs(np.log(aspect)) < np.log(1.05)
    window = np.minimum(width, height * tile_aspect)  # widest crop with the tile's shape
    return PhotoStats(
        count=len(width),
        median_side=int(np.median(short)),
        small_side=int(np.percentile(short, 10)),
        portrait=float((~square & (aspect < 1)).mean()),
        landscape=float((~square & (aspect > 1)).mean()),
        square=float(square.mean()),
        detail_px=int(np.percentile(window, 100 * (1 - DETAIL_SHARE))),
    )

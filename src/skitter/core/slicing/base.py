"""Slicing operation framework: context, operation base classes, registry.

A slicing operation transforms a set of regions:

    apply(regions, ctx) -> regions

A plan starts from one region covering the whole image and runs its
operations in order. So one interface covers operations that split regions
(grid, quadtree), adjust them (jitter, stacking), filter or merge them, or add
regions drawn by hand.

To add an operation: subclass `Subdivider` (to split each region on its own)
or `SlicingOperation` (anything else), set `id`, `name`, `category` and
`description`, declare parameters (see params.py), implement `subdivide` or
`apply`, and decorate the class with `@register_operation`. Operations must
be deterministic for given parameters and image (take a seed parameter for
randomness) and must not modify their inputs.

Plans may run in the background. Operations get a `Progress` to report
their share done to and that stops them (raises SlicingCancelled) once the
run is cancelled: call it at natural points if a stage can take a while
(Subdivider does, between regions, and hands each region its slice).

Regions may overlap; their z values decide which lies on top (see
regions.py). Operations that only move or resize regions should keep z as
it is (RegionSet.replace does). Operations that create overlap or change it
set z, for example with RegionSet.restacked.
"""

import math
from abc import ABC, abstractmethod
from functools import cached_property
from typing import ClassVar

import numpy as np
from scipy import ndimage

from skitter.core.imaging import fill_hidden, visible_mask
from skitter.core.slicing.layout import TILE_UNIT, MosaicLayout
from skitter.core.slicing.params import Configurable
from skitter.core.slicing.regions import Region, RegionSet

# Menu order for categories; unknown categories sort after these.
CATEGORIES = ("Subdivide", "Adjust", "Filter", "Other")

MAX_REGIONS = 250_000
ANTIALIAS = 0.7  # blur before sampling, in sample spacings (see SliceContext.patch)


class SlicingError(Exception):
    """A plan could not be evaluated (for example, it produced too many regions)."""


class SlicingCancelled(Exception):
    """A plan's evaluation was cancelled (a newer one replaces it)."""


class Progress:
    """An operation's progress (0..1) toward its caller's report, and its cancel check.

    Calling it with the share done checks for cancellation (raising
    SlicingCancelled) and reports, at most every REPORT_STEP of progress.
    `part(lo, hi)` is a Progress for one piece of the work, mapped onto
    [lo, hi] of this one.
    """

    REPORT_STEP = 0.01

    def __init__(self, report=None, cancelled=None, lo: float = 0.0, hi: float = 1.0):
        self._report = report  # fraction -> None
        self._cancelled = cancelled  # () -> bool
        self._lo, self._hi = lo, hi
        self._last = [-1.0]  # shared with parts: reports go out in steps

    def __call__(self, fraction: float) -> None:
        if self._cancelled is not None and self._cancelled():
            raise SlicingCancelled
        if self._report is None:
            return
        done = self._lo + (self._hi - self._lo) * min(max(float(fraction), 0.0), 1.0)
        if done - self._last[0] >= self.REPORT_STEP or done >= 1.0 > self._last[0]:
            self._last[0] = done
            self._report(done)

    def part(self, lo: float, hi: float) -> "Progress":
        span = self._hi - self._lo
        piece = Progress(self._report, self._cancelled, self._lo + span * lo, self._lo + span * hi)
        piece._last = self._last
        return piece


NO_PROGRESS = Progress()


def check_region_count(count: int, name: str) -> None:
    """Raise SlicingError if an operation would produce more than MAX_REGIONS regions.

    Operations call this before allocating large results, so an oversized
    plan fails fast instead of exhausting memory.
    """
    if count > MAX_REGIONS:
        raise SlicingError(f"{name} produced {count:,} regions; the limit is {MAX_REGIONS:,}")


def region_rng(seed: int, region: Region) -> np.random.Generator:
    """A random generator for one region, seeded by seed and the region's position."""
    # SeedSequence entries must be non-negative; wrap coordinates left of or above the canvas.
    x, y = (round(v * 64) % 2**64 for v in (region.cx, region.cy))
    return np.random.default_rng([seed, x, y])


class SliceContext:
    """Read-only inputs available to slicing operations.

    Slicing works in mosaic units (see layout.py): the canvas is (0, 0,
    width, height), the final source image scaled uniformly to the mosaic
    layout. Operations size regions from `tile_size` (the base tile) and
    read the image through `patch`, which handles the scaling. Results do
    not depend on the unit size other than by scale.

    tile_width is the base tile's width in mosaic units: TILE_UNIT by
    default, or one unit per source pixel (1 px tiles) when no layout is
    given, which tests use.

    The image may be RGBA (sources are; see imaging.py). Its alpha is a
    mask: `visible` marks pixels at least half opaque (None: all of them),
    and hidden ones lie outside the picture, as if past its border. `image`
    is RGB with every hidden pixel taking the nearest visible one's color,
    as samples past the border take the nearest edge pixel, so everything
    reading the image treats the mask's edge like the border. Plans drop
    the regions that touch no visible pixel (mask.py).

    One context is created per final image and layout and reused across
    evaluations, so derived data (like luminance, or the filled-in image) is
    computed once, when first needed: building a context is cheap, and the
    work happens where slicing runs (in the background).
    """

    def __init__(
        self,
        image: np.ndarray,
        layout: MosaicLayout | None = None,
        tile_width: float | None = None,
    ):
        self._source = image
        self._smoothed: dict[float, np.ndarray] = {}  # luminance by blur (see patch)
        self.visible = visible_mask(image)  # (H, W) bool, or None: all visible
        h, w = image.shape[:2]
        self.source_height, self.source_width = h, w
        if tile_width is None:
            tile_width = 1.0 if layout is None else TILE_UNIT
        self.layout = layout or MosaicLayout(columns=w)
        self.tile_width = float(tile_width)
        self.width = self.layout.columns * self.tile_width
        self.height = self.width * h / w
        self.scale = self.width / w  # mosaic units per source pixel

    @cached_property
    def image(self) -> np.ndarray:
        """(H, W, 3) uint8 RGB, hidden pixels filled in (see the class docstring)."""
        rgb = self._source[..., :3]
        if self.visible is not None:
            rgb = fill_hidden(rgb, self.visible)
        return np.ascontiguousarray(rgb)

    @property
    def tile_size(self) -> tuple[float, float]:
        """Base tile (width, height) in mosaic units."""
        return (self.tile_width, self.tile_width / self.layout.tile_aspect)

    def tile_dims(self, scale: float = 1.0) -> tuple[float, float]:
        """A tile of `scale` base tiles: (width, height) in mosaic units."""
        width, height = self.tile_size
        return (scale * width, scale * height)

    @property
    def tile_aspect(self) -> float:
        return self.layout.tile_aspect

    def canvas(self) -> RegionSet:
        """One region covering the whole canvas: where every plan starts."""
        return RegionSet.covering(self.width, self.height)

    @cached_property
    def visible_table(self) -> np.ndarray | None:
        """(H + 1, W + 1) int32 summed-area table of `visible` (None without a mask):
        visible pixels in [j0, j1) x [i0, i1) are t[j1, i1] - t[j0, i1] - t[j1, i0] + t[j0, i0]."""
        if self.visible is None:
            return None
        h, w = self.visible.shape
        table = np.zeros((h + 1, w + 1), np.int32)
        np.cumsum(self.visible, axis=0, dtype=np.int32, out=table[1:, 1:])
        np.cumsum(table[1:, 1:], axis=1, out=table[1:, 1:])
        table.setflags(write=False)
        return table

    @cached_property
    def luminance(self) -> np.ndarray:
        """(H, W) float32 luma in 0..255 (Rec. 601 weights)."""
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        lum = self.image.astype(np.float32) @ weights
        lum.setflags(write=False)
        return lum

    def smoothed_luminance(self, sigma: float) -> np.ndarray:
        """luminance blurred by sigma source pixels (cached per sigma, to 0.1 px)."""
        sigma = round(sigma, 1)
        if sigma <= 0:
            return self.luminance
        if sigma not in self._smoothed:
            blurred = ndimage.gaussian_filter(self.luminance, sigma)
            blurred.setflags(write=False)
            self._smoothed[sigma] = blurred
        return self._smoothed[sigma]

    def patch(
        self,
        region: Region,
        source: str = "luminance",
        max_samples: int = 4_000_000,
        antialias: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Sample the source image inside a (canvas) region, aligned with its frame.

        Returns (samples, scale): samples[j, i] is the source pixel nearest to
        local point ((i + 0.5) / scale, (j + 0.5) / scale), so scale is samples
        per mosaic unit. It gives about one sample per source pixel, fewer if
        that would exceed max_samples. Points outside the image take the
        nearest edge pixel. source is "luminance", "rgb" or "visible" (the mask:
        bool, all True without one).

        antialias (luminance): sampling coarser than the image, blur it first
        (by ANTIALIAS of the sample spacing), so detail finer than the samples
        averages out instead of aliasing into false patterns (moire).
        """
        scale = min(1.0 / self.scale, math.sqrt(max_samples / max(region.area, 1.0)))
        pw = max(1, math.ceil(region.width * scale))
        ph = max(1, math.ceil(region.height * scale))
        if source == "visible":
            if self.visible is None:
                return np.ones((ph, pw), bool), scale
            image = self.visible
        elif source == "luminance":
            spacing = 1.0 / (scale * self.scale)  # source pixels per sample
            image = self.smoothed_luminance(ANTIALIAS * spacing) if antialias else self.luminance
        else:
            image = self.image
        u = (np.arange(pw) + 0.5) / scale
        v = (np.arange(ph) + 0.5) / scale
        src_w, src_h = self.source_width, self.source_height
        if region.is_axis_aligned:
            xs = np.floor((region.cx - region.width / 2 + u) / self.scale)
            ys = np.floor((region.cy - region.height / 2 + v) / self.scale)
            xs = np.clip(xs, 0, src_w - 1).astype(int)
            ys = np.clip(ys, 0, src_h - 1).astype(int)
            return image[np.ix_(ys, xs)], scale
        uu, vv = np.meshgrid(u, v)
        world = region.local_to_world(np.stack([uu, vv], axis=-1)) / self.scale
        xi = np.clip(np.floor(world[..., 0]), 0, src_w - 1).astype(int)
        yi = np.clip(np.floor(world[..., 1]), 0, src_h - 1).astype(int)
        return image[yi, xi], scale


class SlicingOperation(Configurable, ABC):
    """Base class for all slicing operations."""

    id: ClassVar[str] = ""  # stable identifier used in saved plans
    name: ClassVar[str] = ""  # shown in the UI
    category: ClassVar[str] = "Other"
    description: ClassVar[str] = ""

    def key(self) -> tuple:
        """Identifies this operation's type and settings (for result caching)."""
        return (self.id, super().key())

    def summary(self) -> str:
        """Short description of the current settings for lists; may be empty."""
        return ""

    def to_dict(self) -> dict:
        return {"type": self.id, "params": self.values()}

    @abstractmethod
    def apply(
        self, regions: RegionSet, ctx: SliceContext, progress: Progress = NO_PROGRESS
    ) -> RegionSet:
        """Return the transformed regions (in image coordinates)."""


class Subdivider(SlicingOperation):
    """An operation that splits each input region independently.

    Implement `subdivide`, which works in the region's local frame: the region
    is the axis-aligned rectangle [0, width] x [0, height], and the returned
    regions use the same frame. The base class maps them back to image
    coordinates, including the parent's position and rotation.

    Stacking: the parts of a region take its place in the overall stack, so
    they stay above the parts of every region below it and below the parts of
    every region above it. Among themselves, parts stack by the z values
    subdivide gives them (equal z: array order). The result's z values are
    0..N-1.
    """

    category = "Subdivide"

    def apply(
        self, regions: RegionSet, ctx: SliceContext, progress: Progress = NO_PROGRESS
    ) -> RegionSet:
        parent_rank = regions.stacking_rank()
        parts, parent_keys, part_keys = [], [], []
        total, n = 0, len(regions)
        for index, region in enumerate(regions):
            local = self.subdivide(region, ctx, progress.part(index / n, (index + 1) / n))
            total += len(local)
            check_region_count(total, self.name)
            parts.append(local.to_world(region))
            parent_keys.append(np.full(len(local), parent_rank[index]))
            part_keys.append(local.stacking_rank())
        result = RegionSet.concat(parts)
        if not result:
            return result
        return result.restacked(np.concatenate(parent_keys), np.concatenate(part_keys))

    @abstractmethod
    def subdivide(
        self, region: Region, ctx: SliceContext, progress: Progress = NO_PROGRESS
    ) -> RegionSet:
        """Split region; return the parts in its local frame. Report to progress (and
        so check for cancellation) at natural points if it can take a while."""


# Registry

_registry: dict[str, type[SlicingOperation]] = {}


def register_operation(cls: type[SlicingOperation]) -> type[SlicingOperation]:
    """Class decorator making an operation available to plans and the UI."""
    if not cls.id or not cls.name:
        raise TypeError(f"{cls.__name__} must define id and name")
    existing = _registry.get(cls.id)
    if existing is not None and existing is not cls:
        raise ValueError(f"operation id {cls.id!r} is already used by {existing.__name__}")
    _registry[cls.id] = cls
    return cls


def operation_types() -> list[type[SlicingOperation]]:
    """Registered operation classes in menu order (by category, then name)."""

    def order(cls):
        rank = CATEGORIES.index(cls.category) if cls.category in CATEGORIES else len(CATEGORIES)
        return (rank, cls.category, cls.name)

    return sorted(_registry.values(), key=order)


def get_operation_type(type_id: str) -> type[SlicingOperation]:
    try:
        return _registry[type_id]
    except KeyError:
        raise KeyError(f"unknown slicing operation {type_id!r}") from None


def operation_from_dict(data: dict) -> SlicingOperation:
    return get_operation_type(data["type"])(**data.get("params", {}))

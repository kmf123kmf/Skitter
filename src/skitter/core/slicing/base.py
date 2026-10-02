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

from skitter.core.slicing.layout import TILE_UNIT, MosaicLayout
from skitter.core.slicing.params import Configurable
from skitter.core.slicing.regions import Region, RegionSet

# Menu order for categories; unknown categories sort after these.
CATEGORIES = ("Subdivide", "Adjust", "Filter", "Other")

MAX_REGIONS = 250_000


class SlicingError(Exception):
    """A plan could not be evaluated (for example, it produced too many regions)."""


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

    One context is created per final image and layout and reused across
    evaluations, so derived data (like luminance) is computed once.
    """

    def __init__(
        self,
        image: np.ndarray,
        layout: MosaicLayout | None = None,
        tile_width: float | None = None,
    ):
        self.image = image  # (H, W, 3) uint8 RGB, read-only
        h, w = image.shape[:2]
        if tile_width is None:
            tile_width = 1.0 if layout is None else TILE_UNIT
        self.layout = layout or MosaicLayout(columns=w)
        self.tile_width = float(tile_width)
        self.width = self.layout.columns * self.tile_width
        self.height = self.width * h / w
        self.scale = self.width / w  # mosaic units per source pixel

    @property
    def source_width(self) -> int:
        return self.image.shape[1]

    @property
    def source_height(self) -> int:
        return self.image.shape[0]

    @property
    def tile_size(self) -> tuple[float, float]:
        """Base tile (width, height) in mosaic units."""
        return (self.tile_width, self.tile_width / self.layout.tile_aspect)

    @property
    def tile_aspect(self) -> float:
        return self.layout.tile_aspect

    def canvas(self) -> RegionSet:
        """One region covering the whole canvas: where every plan starts."""
        return RegionSet.covering(self.width, self.height)

    @cached_property
    def luminance(self) -> np.ndarray:
        """(H, W) float32 luma in 0..255 (Rec. 601 weights)."""
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        lum = self.image.astype(np.float32) @ weights
        lum.setflags(write=False)
        return lum

    def patch(
        self, region: Region, source: str = "luminance", max_samples: int = 4_000_000
    ) -> tuple[np.ndarray, float]:
        """Sample the source image inside a (canvas) region, aligned with its frame.

        Returns (samples, scale): samples[j, i] is the source pixel nearest to
        local point ((i + 0.5) / scale, (j + 0.5) / scale), so scale is samples
        per mosaic unit. It gives about one sample per source pixel, fewer if
        that would exceed max_samples. Points outside the image take the
        nearest edge pixel. source is "luminance" or "rgb".
        """
        image = self.luminance if source == "luminance" else self.image
        scale = min(1.0 / self.scale, math.sqrt(max_samples / max(region.area, 1.0)))
        pw = max(1, math.ceil(region.width * scale))
        ph = max(1, math.ceil(region.height * scale))
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
    def apply(self, regions: RegionSet, ctx: SliceContext) -> RegionSet:
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

    def apply(self, regions: RegionSet, ctx: SliceContext) -> RegionSet:
        parent_rank = regions.stacking_rank()
        parts, parent_keys, part_keys = [], [], []
        total = 0
        for index, region in enumerate(regions):
            local = self.subdivide(region, ctx)
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
    def subdivide(self, region: Region, ctx: SliceContext) -> RegionSet:
        """Split region; return the parts in its local frame."""


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

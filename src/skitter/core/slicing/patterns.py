"""Brick patterns: repeating arrangements of bricks, used by PatternSlicer.

A pattern is a repeating unit: a few bricks plus two period vectors. Copies
of the unit shifted by every combination of whole periods fill the plane,
and `tile_pattern` keeps the bricks that touch the region being filled.

Bricks are ordinary regions, so a unit may mix orientations and, if a
pattern needs it, brick shapes other than the base tile. A pattern is a
function that builds its unit with a `PatternBuilder`, which knows the brick
size the user chose:

    @register_pattern("stack", "Stack bond")
    def stack(b: PatternBuilder) -> PatternUnit:
        b.add(b.length / 2, b.thickness / 2)
        return b.unit((b.length, 0), (0, b.thickness))

Patterns are described in landscape terms: a brick is `length` long and
`thickness` thick, and a horizontal brick has its long side along x. `add`
makes each brick a region in the base tile's own shape, turned 90° where
needed, so a portrait base tile works too. Patterns with settings name them
in `options`; the settings are parameters of PatternSlicer with the same
names, passed to the build function as keyword arguments.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from skitter.core.slicing.base import SlicingError
from skitter.core.slicing.regions import RegionSet, _rotate

MAX_CANDIDATES = 4_000_000  # brick copies tested per region


@dataclass(frozen=True)
class PatternUnit:
    """One repeat of a pattern in pattern coordinates (mosaic units)."""

    bricks: RegionSet
    period: np.ndarray  # (2, 2): rows are the period vectors a and b
    center: np.ndarray  # pinned to the region's center by the "center" anchor
    corner: np.ndarray  # pinned to the region's top-left by the "top_left" anchor


class PatternBuilder:
    """Collects the bricks of a pattern unit."""

    def __init__(self, brick_size: tuple[float, float]):
        width, height = brick_size  # the base tile scaled by the cell size
        self.brick_size = (float(width), float(height))
        self.length = float(max(width, height))
        self.thickness = float(min(width, height))
        self._portrait = width < height
        self._bricks: list[tuple[float, float, float, float, float]] = []

    def add(
        self,
        cx: float,
        cy: float,
        horizontal: bool = True,
        *,
        length: float | None = None,
        thickness: float | None = None,
    ) -> None:
        """Add a brick centered at (cx, cy), lying horizontally or vertically.

        length and thickness default to the base brick; give them for bricks
        of another shape.
        """
        length = self.length if length is None else length
        thickness = self.thickness if thickness is None else thickness
        # Keep the region in the tile's shape and turn it, so tiles are never squashed.
        if self._portrait:
            width, height, upright = thickness, length, not horizontal
        else:
            width, height, upright = length, thickness, horizontal
        self._bricks.append((cx, cy, width, height, 0.0 if upright else math.pi / 2))

    def unit(self, a, b, *, center=None, corner=None) -> PatternUnit:
        """The unit made of the bricks added so far, repeating along a and b.

        center and corner default to the center and top-left of the bricks'
        bounding box.
        """
        if not self._bricks:
            raise ValueError("a pattern unit needs at least one brick")
        cx, cy, w, h, rotation = (np.array(v) for v in zip(*self._bricks, strict=True))
        positions, sizes = np.stack([cx, cy], axis=-1), np.stack([w, h], axis=-1)
        bricks = RegionSet.from_arrays(positions, sizes, rotation)
        period = np.array([a, b], dtype=float)
        if abs(np.linalg.det(period)) < 1e-9:
            raise ValueError("pattern period vectors must not be parallel")
        bounds = bricks.bounds()
        lo, hi = bounds[:, :2].min(axis=0), bounds[:, 2:].max(axis=0)
        return PatternUnit(
            bricks,
            period,
            np.asarray((lo + hi) / 2 if center is None else center, dtype=float),
            np.asarray(lo if corner is None else corner, dtype=float),
        )


@dataclass(frozen=True)
class BrickPattern:
    id: str  # stable identifier used in saved plans
    name: str  # shown in the UI
    build: Callable[..., PatternUnit]  # build(builder, **options) -> PatternUnit
    options: tuple[str, ...] = ()  # PatternSlicer parameters this pattern uses
    description: str = field(default="", compare=False)


_patterns: dict[str, BrickPattern] = {}


def register_pattern(id: str, name: str, *, options=(), description: str = ""):
    """Decorator registering a function build(builder, **options) as a pattern."""

    def register(build: Callable[..., PatternUnit]) -> Callable[..., PatternUnit]:
        existing = _patterns.get(id)
        if existing is not None and existing.build is not build:
            raise ValueError(f"pattern id {id!r} is already used by {existing.name!r}")
        _patterns[id] = BrickPattern(id, name, build, tuple(options), description)
        return build

    return register


def patterns() -> list[BrickPattern]:
    """Registered patterns in registration order."""
    return list(_patterns.values())


def get_pattern(pattern_id: str) -> BrickPattern:
    try:
        return _patterns[pattern_id]
    except KeyError:
        raise KeyError(f"unknown brick pattern {pattern_id!r}") from None


def tile_pattern(
    unit: PatternUnit,
    width: float,
    height: float,
    *,
    anchor: str = "center",
    rotation: float = 0.0,
) -> RegionSet:
    """Fill [0, width] x [0, height] with copies of unit, turned by rotation.

    Returns every brick that overlaps the area (edge bricks stay whole and
    overhang), ordered top to bottom, then left to right. anchor "center"
    pins unit.center to the area's center, "top_left" pins unit.corner to
    its top-left corner; the pattern turns about that point. Rotations are
    reduced to (-90°, 90°], which gives the same rectangle, so tiles are
    never drawn upside down.
    """
    place = np.array([width / 2, height / 2]) if anchor == "center" else np.zeros(2)
    pin = unit.center if anchor == "center" else unit.corner

    # Shifts t = i*a + j*b whose copy of the unit can reach the area.
    area = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    area = _rotate(area - place, -rotation) + pin  # the area in pattern coordinates
    bounds = unit.bricks.bounds()
    lo = area.min(axis=0) - bounds[:, 2:].max(axis=0)
    hi = area.max(axis=0) - bounds[:, :2].min(axis=0)
    box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    ij = np.linalg.solve(unit.period.T, box.T).T
    i0, j0 = np.floor(ij.min(axis=0)).astype(int)
    i1, j1 = np.ceil(ij.max(axis=0)).astype(int)
    if (i1 - i0 + 1) * (j1 - j0 + 1) * len(unit.bricks) > MAX_CANDIDATES:
        raise SlicingError("the brick pattern is too fine for the area; use larger bricks")

    ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
    shifts = np.stack([ii.ravel(), jj.ravel()], axis=-1) @ unit.period
    centers = (unit.bricks.center[None, :, :] + shifts[:, None, :]).reshape(-1, 2)
    centers = place + _rotate(centers - pin, rotation)
    sizes = np.tile(unit.bricks.size, (len(shifts), 1))
    turns = _upright(np.tile(unit.bricks.rotation, len(shifts)) + rotation)

    keep = _overlaps_area(centers, sizes, turns, width, height)
    centers, sizes, turns = centers[keep], sizes[keep], turns[keep]
    order = np.lexsort((np.round(centers[:, 0], 6), np.round(centers[:, 1], 6)))
    return RegionSet.from_arrays(centers[order], sizes[order], turns[order])


def _upright(rotation: np.ndarray) -> np.ndarray:
    """The same rectangles with rotations in (-pi/2, pi/2]."""
    return rotation - math.pi * np.ceil(rotation / math.pi - 0.5 - 1e-9)


def _overlaps_area(centers, sizes, rotation, width, height, tol=1e-6) -> np.ndarray:
    """Which rotated rectangles overlap [0, width] x [0, height] (more than touch).

    A separating-axis test on the area's axes and each rectangle's own axes.
    """
    c, s = np.abs(np.cos(rotation)), np.abs(np.sin(rotation))
    hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
    aw, ah = width / 2, height / 2
    d = np.array([aw, ah]) - centers
    # Area axes: the rectangles' bounding boxes against the area.
    keep = np.abs(d[:, 0]) < c * hw + s * hh + aw - tol
    keep &= np.abs(d[:, 1]) < s * hw + c * hh + ah - tol
    # Rectangle axes: the area projected onto each rectangle's sides.
    cos, sin = np.cos(rotation), np.sin(rotation)
    keep &= np.abs(d[:, 0] * cos + d[:, 1] * sin) < hw + c * aw + s * ah - tol
    keep &= np.abs(-d[:, 0] * sin + d[:, 1] * cos) < hh + s * aw + c * ah - tol
    return keep

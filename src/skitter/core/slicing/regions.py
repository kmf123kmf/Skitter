"""Regions: the output of slicing.

A region is a rectangle in mosaic pixel coordinates (see layout.py), given by its
center, size (width, height), and rotation in radians. Rotation is clockwise
on screen (image y points down), matching the GPU sprite shader, so regions
can be drawn directly as sprite instances.

Each region also has a *local frame*: origin at its top-left corner, x along
its top edge, y along its left edge, spanning [0, width] x [0, height].
Subdividing operations work in this frame and never deal with rotation.

Regions may overlap. Each has a stacking value `z`: regions with higher z
lie on top and hide the parts of lower regions they cover. Equal z values
stack in array order (later on top). Only the relative order of z values
matters; `stacking_order()` gives the bottom-to-top order.
"""

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

REGION_DTYPE = np.dtype([("center", "f8", 2), ("size", "f8", 2), ("rotation", "f8"), ("z", "f8")])


def _rotate(points: np.ndarray, angle) -> np.ndarray:
    """Rotate (..., 2) points by angle (scalar or broadcastable array)."""
    c, s = np.cos(angle), np.sin(angle)
    x, y = points[..., 0], points[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


@dataclass(frozen=True)
class Region:
    cx: float
    cy: float
    width: float
    height: float
    rotation: float = 0.0
    z: float = 0.0

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy])

    @property
    def size(self) -> np.ndarray:
        return np.array([self.width, self.height])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_axis_aligned(self) -> bool:
        return math.isclose(math.remainder(self.rotation, math.tau), 0.0, abs_tol=1e-9)

    def local_to_world(self, points) -> np.ndarray:
        """Map (..., 2) points from this region's local frame to image coordinates."""
        local = np.asarray(points, dtype=float) - self.size / 2
        return self.center + _rotate(local, self.rotation)

    def world_to_local(self, points) -> np.ndarray:
        """Map (..., 2) image points into this region's local frame."""
        offset = np.asarray(points, dtype=float) - self.center
        return _rotate(offset, -self.rotation) + self.size / 2


class RegionSet:
    """An immutable collection of regions stored as parallel numpy arrays.

    Operations never modify a RegionSet; they build new ones (see `replace`),
    which lets slicing plans cache and share intermediate results safely.
    """

    __slots__ = ("_data", "_order")

    def __init__(self, data: np.ndarray | None = None):
        if data is None:
            data = np.zeros(0, REGION_DTYPE)
        if data.dtype != REGION_DTYPE:
            raise TypeError(f"expected REGION_DTYPE array, got {data.dtype}")
        data.setflags(write=False)
        self._data = data
        self._order: np.ndarray | None = None

    # Construction

    @classmethod
    def from_arrays(cls, center, size, rotation=0.0, z=0.0) -> "RegionSet":
        center = np.asarray(center, dtype=float).reshape(-1, 2)
        n = len(center)
        data = np.zeros(n, REGION_DTYPE)
        data["center"] = center
        data["size"] = np.broadcast_to(np.asarray(size, dtype=float), (n, 2))
        data["rotation"] = np.broadcast_to(np.asarray(rotation, dtype=float), (n,))
        data["z"] = np.broadcast_to(np.asarray(z, dtype=float), (n,))
        return cls(data)

    @classmethod
    def from_rects(cls, x, y, width, height, rotation=0.0, z=0.0) -> "RegionSet":
        """Build from top-left corners and sizes (rotation about each center)."""
        x, y, width, height = np.broadcast_arrays(
            *(np.asarray(v, float) for v in (x, y, width, height))
        )
        center = np.stack([x + width / 2, y + height / 2], axis=-1)
        return cls.from_arrays(center, np.stack([width, height], axis=-1), rotation, z)

    @classmethod
    def covering(cls, width: float, height: float) -> "RegionSet":
        """One axis-aligned region covering [0, width] x [0, height]."""
        return cls.from_rects(0, 0, width, height)

    @classmethod
    def grid(cls, width: float, height: float, columns: int, rows: int) -> "RegionSet":
        """columns x rows equal cells exactly covering [0, width] x [0, height]."""
        return cls.cells(width / columns, height / rows, columns, rows)

    @classmethod
    def cells(
        cls, cell_width: float, cell_height: float, columns: int, rows: int, origin=(0.0, 0.0)
    ) -> "RegionSet":
        """columns x rows cells of the given size, the first with its top-left at origin.

        Ordered row by row: left to right, top to bottom.
        """
        cx = origin[0] + (np.arange(columns) + 0.5) * cell_width
        cy = origin[1] + (np.arange(rows) + 0.5) * cell_height
        xx, yy = np.meshgrid(cx, cy)
        return cls.from_arrays(
            np.stack([xx.ravel(), yy.ravel()], axis=-1), (cell_width, cell_height)
        )

    @classmethod
    def concat(cls, sets: Iterable["RegionSet"]) -> "RegionSet":
        arrays = [s._data for s in sets]
        return cls(np.concatenate(arrays) if arrays else None)

    # Access

    @property
    def data(self) -> np.ndarray:
        """The underlying read-only structured array."""
        return self._data

    @property
    def center(self) -> np.ndarray:
        return self._data["center"]

    @property
    def size(self) -> np.ndarray:
        return self._data["size"]

    @property
    def rotation(self) -> np.ndarray:
        return self._data["rotation"]

    @property
    def z(self) -> np.ndarray:
        return self._data["z"]

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return len(self._data) > 0

    def __getitem__(self, key) -> "Region | RegionSet":
        if isinstance(key, int | np.integer):
            (cx, cy), (w, h), rot, z = self._data[key]
            return Region(float(cx), float(cy), float(w), float(h), float(rot), float(z))
        return RegionSet(np.array(self._data[key]))

    def __iter__(self) -> Iterator[Region]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"RegionSet({len(self)} regions)"

    # Derived sets

    def replace(self, center=None, size=None, rotation=None, z=None) -> "RegionSet":
        """A copy with some fields replaced (arrays or broadcastable values)."""
        return RegionSet.from_arrays(
            self.center if center is None else center,
            self.size if size is None else size,
            self.rotation if rotation is None else rotation,
            self.z if z is None else z,
        )

    def to_world(self, parent: Region) -> "RegionSet":
        """Treat these regions as local to parent and map them to image coordinates.

        z values are kept as they are; Subdivider places the result within the
        parent's position in the overall stack.
        """
        return self.replace(
            center=parent.local_to_world(self.center),
            rotation=self.rotation + parent.rotation,
        )

    # Stacking

    def stacking_order(self) -> np.ndarray:
        """Indices from bottom to top: by z, then array order for equal z."""
        if self._order is None:
            order = np.lexsort((np.arange(len(self)), self.z))
            order.setflags(write=False)
            self._order = order
        return self._order

    def stacking_rank(self) -> np.ndarray:
        """Each region's position in the stack (0 = bottom)."""
        rank = np.empty(len(self), dtype=np.int64)
        rank[self.stacking_order()] = np.arange(len(self))
        return rank

    def restacked(self, key, tiebreak=None) -> "RegionSet":
        """A copy with z set to stack by key (higher on top).

        Equal keys keep the relative order of tiebreak (default: the current
        stacking). The new z values are 0..N-1.
        """
        tiebreak = self.stacking_rank() if tiebreak is None else tiebreak
        order = np.lexsort((tiebreak, np.asarray(key)))
        z = np.empty(len(self))
        z[order] = np.arange(len(self))
        return self.replace(z=z)

    # Geometry

    def corners(self) -> np.ndarray:
        """(N, 4, 2) corners in order top-left, top-right, bottom-right, bottom-left."""
        unit = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        local = unit[None, :, :] * self.size[:, None, :]
        return self.center[:, None, :] + _rotate(local, self.rotation[:, None])

    def bounds(self) -> np.ndarray:
        """(N, 4) axis-aligned bounding boxes as (x0, y0, x1, y1)."""
        corners = self.corners()
        return np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=1)

    def area(self) -> np.ndarray:
        return self.size[:, 0] * self.size[:, 1]

    def contains(self, x: float, y: float) -> np.ndarray:
        """Boolean mask of regions containing image point (x, y)."""
        offset = np.array([x, y]) - self.center
        local = _rotate(offset, -self.rotation)
        return np.all(np.abs(local) <= self.size / 2, axis=1)

    def hit_test(self, x: float, y: float) -> int:
        """Index of the topmost region containing (x, y), or -1."""
        hits = np.flatnonzero(self.contains(x, y))
        if not len(hits):
            return -1
        return int(hits[np.argmax(self.stacking_rank()[hits])])

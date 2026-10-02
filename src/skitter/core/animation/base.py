"""Animation framework: choreographies that build the mosaic tile by tile.

A choreography is a configurable recipe (settings declared as Params, like
slicing operations) that, given a MosaicScene, plans a Timeline. A timeline
is a pure function of time: `frame(t)` returns every tile's state at t, so
it can be played at any speed, paused, scrubbed, or rendered frame by frame
to a video. Its last frame, at `duration`, is exactly the finished mosaic
(TileFrame.final), drawn in stacking order.

To add a choreography: subclass `Choreography`, set `id`, `name` and
`description`, declare parameters, implement `timeline(scene)` (precompute
per-tile values there, keeping `frame` cheap), and decorate the class with
`@register_choreography`. `FlightTimeline` covers the common case of every
tile travelling from a start state to its final state.

Overlapping tiles (a photo pile) must land bottom first: a tile that lands
after one above it would jump underneath on arrival. `landing_order` turns
each tile's preferred position in the sequence into an order that respects
this, staying as close to the preference as it allows; choreographies then
choose when the k-th tile lands (for example, evenly spaced).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
from numba import njit

from skitter.core.easing import Easing, ease_out_cubic, lerp, progress
from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import Configurable


@dataclass(frozen=True, eq=False)
class TileFrame:
    """Every tile's state at one moment; arrays align with the scene's tiles."""

    center: np.ndarray  # (N, 2)
    size: np.ndarray  # (N, 2)
    rotation: np.ndarray  # (N,) radians clockwise
    alpha: np.ndarray  # (N,) 0 invisible .. 1 opaque
    tint: np.ndarray  # (N, 4) rgb blended over the tile with strength a (0: none)
    order: np.ndarray | None = None  # (N,) tile indices bottom to top; None: scene order

    @classmethod
    def final(cls, scene: MosaicScene) -> "TileFrame":
        """The finished mosaic."""
        n = len(scene)
        return cls(
            center=scene.center.copy(),
            size=scene.size.copy(),
            rotation=scene.rotation.copy(),
            alpha=np.ones(n),
            tint=np.zeros((n, 4)),
        )

    def __len__(self) -> int:
        return len(self.alpha)

    def replace(self, **changes) -> "TileFrame":
        return replace(self, **changes)

    def draw_order(self) -> np.ndarray:
        return np.arange(len(self)) if self.order is None else self.order


class Timeline(ABC):
    """A planned animation: every tile's state at any time in [0, duration]."""

    duration: float

    @abstractmethod
    def frame(self, t: float) -> TileFrame:
        """The state at time t (clamped to [0, duration]); frame(duration) is final."""


class Choreography(Configurable, ABC):
    """A configurable way to build the mosaic; plans a Timeline for a scene."""

    id: ClassVar[str] = ""
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    @abstractmethod
    def timeline(self, scene: MosaicScene) -> Timeline: ...


class FlightTimeline(Timeline):
    """Each tile travels from a start state to its final state.

    Tile i waits until delay[i], showing at its start state with the start
    alpha (0: hidden until it sets off), then moves over travel[i] seconds
    along easing. In flight it is at the final alpha (opaque), or with
    fade > 0 reaches it over that share of the trip. Tiles still waiting or
    in flight draw above landed ones (among themselves, and once landed, in
    stacking order).
    """

    def __init__(
        self,
        scene: MosaicScene,
        start: TileFrame,
        delay,
        travel,
        easing: Easing = ease_out_cubic,
        fade: float = 0.0,
    ):
        n = len(scene)
        self.final = TileFrame.final(scene)
        self.start = start
        self.delay = np.broadcast_to(np.asarray(delay, dtype=np.float64), (n,))
        self.travel = np.maximum(np.broadcast_to(np.asarray(travel, dtype=np.float64), (n,)), 1e-6)
        self.easing = easing
        self.fade = max(float(fade), 0.0)
        self.duration = float((self.delay + self.travel).max()) if n else 0.0

    def frame(self, t: float) -> TileFrame:
        t = min(max(float(t), 0.0), self.duration)
        a, b = self.start, self.final
        p = progress(t, self.delay, self.travel)
        e = self.easing(p)
        col = e[:, None]
        landed = p >= 1.0
        order = np.lexsort((np.arange(len(p)), ~landed))  # landed first, then in stacking order
        flying = t > self.delay
        if self.fade > 0:
            alpha = lerp(a.alpha, b.alpha, progress(t, self.delay, self.travel * self.fade))
        else:
            alpha = np.where(flying, b.alpha, a.alpha)
        return TileFrame(
            center=lerp(a.center, b.center, col),
            size=lerp(a.size, b.size, col),
            rotation=lerp(a.rotation, b.rotation, e),
            alpha=np.where(landed, b.alpha, alpha),
            tint=lerp(a.tint, b.tint, col),
            order=None if landed.all() else order,
        )


def landing_order(scene: MosaicScene, preference) -> np.ndarray:
    """(N,) each tile's place in the landing sequence (0 lands first).

    Tiles land by preference (lower values first; ties by stacking order),
    except that a tile never lands before every tile below it that it
    overlaps (scene.overlaps): at each step the most preferred tile whose
    support has fully landed goes next.
    """
    n = len(scene)
    preference = np.asarray(preference, dtype=np.float64)
    pairs = scene.overlaps
    pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
    start = np.searchsorted(pairs[:, 0], np.arange(n + 1))  # tiles above each, by lower
    waiting = np.bincount(pairs[:, 1], minlength=n).astype(np.int64)
    return _landing_order(preference, start, pairs[:, 1].copy(), waiting)


@njit(cache=True, nogil=True)
def _landing_order(preference, start, above, waiting):
    # A binary min-heap of ready tiles keyed by (preference, index).
    n = len(preference)
    heap = np.empty(n, np.int64)
    size = 0
    for i in range(n):
        if waiting[i] == 0:
            heap[size] = i
            size += 1
    for k in range(size // 2 - 1, -1, -1):
        _sift_down(heap, size, k, preference)
    rank = np.empty(n, np.int64)
    for step in range(n):
        tile = heap[0]
        size -= 1
        heap[0] = heap[size]
        _sift_down(heap, size, 0, preference)
        rank[tile] = step
        for k in range(start[tile], start[tile + 1]):
            upper = above[k]
            waiting[upper] -= 1
            if waiting[upper] == 0:  # its support has landed: ready
                heap[size] = upper
                j = size
                size += 1
                while j > 0:
                    parent = (j - 1) // 2
                    if not _before(heap[j], heap[parent], preference):
                        break
                    heap[j], heap[parent] = heap[parent], heap[j]
                    j = parent
    return rank


@njit(cache=True, nogil=True, inline="always")
def _before(a, b, preference):
    return preference[a] < preference[b] or (preference[a] == preference[b] and a < b)


@njit(cache=True, nogil=True)
def _sift_down(heap, size, j, preference):
    while True:
        child = 2 * j + 1
        if child >= size:
            return
        if child + 1 < size and _before(heap[child + 1], heap[child], preference):
            child += 1
        if not _before(heap[child], heap[j], preference):
            return
        heap[j], heap[child] = heap[child], heap[j]
        j = child


_registry: dict[str, type[Choreography]] = {}


def register_choreography(cls: type[Choreography]) -> type[Choreography]:
    """Class decorator making a choreography available to the UI."""
    if not cls.id or not cls.name:
        raise TypeError(f"{cls.__name__} must define id and name")
    existing = _registry.get(cls.id)
    if existing is not None and existing is not cls:
        raise ValueError(f"choreography id {cls.id!r} is already used by {existing.__name__}")
    _registry[cls.id] = cls
    return cls


def choreography_types() -> list[type[Choreography]]:
    """Registered choreographies, by name."""
    return sorted(_registry.values(), key=lambda cls: cls.name)


def get_choreography(type_id: str) -> type[Choreography]:
    try:
        return _registry[type_id]
    except KeyError:
        raise KeyError(f"unknown choreography {type_id!r}") from None

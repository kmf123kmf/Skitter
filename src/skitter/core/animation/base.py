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
tile travelling from a start state to its final state; `TossTimeline` throws
tiles onto the table under gravity, with bounces (see look.py for the camera
looking down at the table, and the shadows).

Overlapping tiles (a photo pile) must land bottom first: a tile that lands
after one above it would jump underneath on arrival. `landing_order` turns
each tile's preferred position in the sequence into an order that respects
this, staying as close to the preference as it allows; choreographies then
choose when the k-th tile lands (for example, evenly spaced).
"""

import math
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
    height: np.ndarray | None = None  # (N,) above the table, mosaic units; None: all 0
    rest: np.ndarray | None = None  # (N,) bool: at rest on the table; None: all
    facing: np.ndarray | None = None  # (N,) cos of the flip angle: 1 face up, < 0 back up

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
        """Tiles bottom to top: those at rest (in stacking order), then the others."""
        return np.arange(len(self)) if self.order is None else self.order

    def heights(self) -> np.ndarray:
        return np.zeros(len(self)) if self.height is None else self.height

    def at_rest(self) -> np.ndarray:
        """Tiles lying still on the table; they draw below everything moving."""
        return np.ones(len(self), bool) if self.rest is None else self.rest

    def facings(self) -> np.ndarray:
        return np.ones(len(self)) if self.facing is None else self.facing


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


@dataclass(frozen=True)
class Flips:
    """Tiles turning over in flight about one of their own edges' axes.

    Each tile makes a whole number of turns (`turns`, signed) over its trip,
    so it always lands face up and unmirrored. Seen from above, a tile
    flipping about its local x axis (axis 0) foreshortens in height, about
    its y axis (axis 1) in width; `facing` (the cosine of the flip angle) is
    negative while its back is up.
    """

    turns: np.ndarray  # (N,) whole turns, signed
    axis: np.ndarray  # (N,) 0 or 1
    THINNEST = 0.02  # edge-on, a tile stays this share of its width (never vanishes)

    @classmethod
    def none(cls, n: int) -> "Flips":
        return cls(np.zeros(n), np.zeros(n, np.int64))

    def __bool__(self) -> bool:
        return bool(np.any(self.turns))

    def apply(self, size: np.ndarray, progress: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(size, facing) at this share of the trip (1: landed, face up)."""
        facing = np.cos(2.0 * math.pi * self.turns * progress)
        size = size.copy()
        rows = np.arange(len(size))
        squash = np.maximum(np.abs(facing), self.THINNEST)
        size[rows, 1 - self.axis] *= squash  # axis 0 (x): height shrinks; axis 1 (y): width
        return size, facing


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
        flips: Flips | None = None,
    ):
        n = len(scene)
        self.flips = flips if flips is not None else Flips.none(n)
        self.final = TileFrame.final(scene)
        self.start = start
        self.delay = np.broadcast_to(np.asarray(delay, dtype=np.float64), (n,))
        self.travel = np.maximum(np.broadcast_to(np.asarray(travel, dtype=np.float64), (n,)), 1e-6)
        self.easing = easing
        self.fade = max(float(fade), 0.0)
        self.duration = float((self.delay + self.travel).max()) if n else 0.0

    def frame(self, t: float) -> TileFrame:
        t = min(max(float(t), 0.0), self.duration)
        if t >= self.duration:
            return self.final  # exactly, whatever the rounding of the times
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
        size, facing = lerp(a.size, b.size, col), None
        if self.flips:
            size, facing = self.flips.apply(size, np.where(landed, 1.0, p))
        return TileFrame(
            center=lerp(a.center, b.center, col),
            size=size,
            rotation=lerp(a.rotation, b.rotation, e),
            alpha=np.where(landed, b.alpha, alpha),
            tint=lerp(a.tint, b.tint, col),
            order=None if landed.all() else order,
            rest=None if landed.all() else landed,
            facing=facing,
        )


class TossTimeline(Timeline):
    """Tiles thrown onto the table: gravity arcs, bounces and a settling wobble.

    Tile i leaves start_center at delay[i] (hidden before) and crosses to its
    place at constant speed over `travel` seconds while its height follows a
    gravity arc peaking at `apex` (mosaic units): gravity g = 8 apex / travel².
    It spins at a constant rate from start_rotation and stops spinning on
    impact. Then it bounces in place, each bounce `bounce` times the speed of
    the last (heights shrink by bounce²), and its rotation settles with a
    damped wobble of `wobble` radians. Everything is closed form, so frames
    are exact at any time. A tile counts as at rest (and draws among the
    tiles on the table) only once it has settled, `settle` seconds after
    impact.
    """

    MAX_BOUNCES = 4
    MIN_BOUNCE = 0.01  # bounces lower than this share of the apex are dropped

    def __init__(self, scene: MosaicScene, start_center, start_rotation, delay, travel: float,
                 apex: float, bounce: float = 0.0, wobble: float = 0.0,
                 flips: Flips | None = None):  # fmt: skip
        n = len(scene)
        self.flips = flips if flips is not None else Flips.none(n)
        self.final = TileFrame.final(scene)
        self.start_center = np.asarray(start_center, dtype=np.float64)
        self.start_rotation = np.asarray(start_rotation, dtype=np.float64)
        self.delay = np.broadcast_to(np.asarray(delay, dtype=np.float64), (n,))
        self.travel = max(float(travel), 1e-6)
        self.apex = max(float(apex), 0.0)
        self.gravity = 8.0 * self.apex / self.travel**2
        self.speed = 4.0 * self.apex / self.travel  # upward speed at take-off = at impact
        self.bounce_speed = self._bounce_speeds(self.travel, self.apex, bounce)
        self.bounce_time = 2.0 * self.bounce_speed / max(self.gravity, 1e-12)
        self.bounce_start = np.concatenate([[0.0], np.cumsum(self.bounce_time)])
        self.wobble = float(wobble)
        self.settle = self.settle_time(self.travel, self.apex, bounce, wobble)
        self.duration = float(self.delay.max() + self.travel + self.settle) if n else 0.0

    @classmethod
    def _bounce_speeds(cls, travel: float, apex: float, bounce: float) -> np.ndarray:
        """Take-off speed of each bounce (each keeps `bounce` of the last one's speed)."""
        speeds = []
        if apex <= 0 or bounce <= 0:
            return np.zeros(0)
        gravity = 8.0 * apex / travel**2
        v = 4.0 * apex / travel * bounce
        while len(speeds) < cls.MAX_BOUNCES and v * v / (2 * gravity) >= cls.MIN_BOUNCE * apex:
            speeds.append(v)
            v *= bounce
        return np.array(speeds)

    @classmethod
    def settle_time(cls, travel: float, apex: float, bounce: float, wobble: float) -> float:
        """Seconds from first impact until a tile lies still."""
        travel = max(float(travel), 1e-6)
        speeds = cls._bounce_speeds(travel, max(float(apex), 0.0), bounce)
        bounces = float((2.0 * speeds / (8.0 * apex / travel**2)).sum()) if len(speeds) else 0.0
        return max(bounces, 0.3 * travel) if wobble else bounces

    def frame(self, t: float) -> TileFrame:
        t = min(max(float(t), 0.0), self.duration)
        if t >= self.duration:
            return self.final  # exactly, whatever the rounding of the times
        b = self.final
        tau = t - self.delay  # time since take-off
        p = np.clip(tau / self.travel, 0.0, 1.0)
        center = lerp(self.start_center, b.center, p[:, None])
        rotation = lerp(self.start_rotation, b.rotation, p)
        flight = np.clip(tau, 0.0, self.travel)
        height = self.speed * flight - 0.5 * self.gravity * flight**2

        after = tau - self.travel  # time since first impact
        landed = after >= 0
        if len(self.bounce_speed):
            k = np.clip(np.searchsorted(self.bounce_start, after, side="right") - 1,
                        0, len(self.bounce_speed) - 1)  # fmt: skip
            s = after - self.bounce_start[k]
            bouncing = landed & (after < self.bounce_start[-1])
            hop = self.bounce_speed[k] * s - 0.5 * self.gravity * s**2
            height = np.where(bouncing, np.maximum(hop, 0.0), np.where(landed, 0.0, height))
        else:
            height = np.where(landed, 0.0, height)
        if self.wobble and self.settle > 0:
            u = np.clip(after / self.settle, 0.0, 1.0)
            swing = self.wobble * np.exp(-4.0 * u) * (1.0 - u) * np.sin(3.0 * math.tau * u)
            rotation = np.where(landed, b.rotation + swing, rotation)

        rest = after >= self.settle
        order = np.lexsort((np.arange(len(tau)), ~rest))  # at rest first, then stacking order
        size, facing = b.size.copy(), None
        if self.flips:  # whole turns over the flight: face up from the first impact on
            size, facing = self.flips.apply(size, p)
        return TileFrame(
            center=center,
            size=size,
            facing=facing,
            rotation=np.where(rest, b.rotation, rotation),
            alpha=np.where(tau > 0, 1.0, 0.0),
            tint=b.tint.copy(),
            order=None if rest.all() else order,
            height=np.where(rest, 0.0, np.maximum(height, 0.0)),
            rest=None if rest.all() else rest,
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

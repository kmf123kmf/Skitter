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
after one above it would jump underneath on arrival. Two ways to get such an
order, for two kinds of ordering:

- An ordering with a direction (center outward, reading order, by
  lightness): `landing_order` turns each tile's preferred place into a valid
  order as close to it as it allows.
- A random ordering: `random_landing_order` draws a uniformly random valid
  order, so tiles land close together in space and time exactly as often as
  chance has it. Don't use `landing_order` with random preferences for this:
  it lands a tile held back by the one under it as soon as it may, right
  next to it, so random piles came out clumpy (a landing with another one
  nearby within 0.15 s 92% of the time, against 50% by chance).

Choreographies then choose when the k-th tile lands (for example, evenly
spaced). A new ordering must say which kind it is; tests check random
orderings against chance (tests/test_animation.py).
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numba import njit

from skitter.core.easing import Easing, ease_out_cubic, lerp, progress
from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import Configurable

if TYPE_CHECKING:
    from skitter.core.animation.look import AnimationLook


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
    def timeline(self, scene: MosaicScene, look: "AnimationLook | None" = None) -> Timeline:
        """The plan for a scene; `look` gives the camera (default: AnimationLook())."""


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
    gravity arc peaking at `apex` (mosaic units): gravity g = 8 apex / travel²
    for a throw from the table. With `start_height` (a drop: start_height =
    apex) a tile starts that high and falls from rest, g = 2 apex / travel².
    In general the arc runs from start_height up to apex and down to the
    table in `travel` seconds.
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
                 flips: Flips | None = None, start_height: float = 0.0):  # fmt: skip
        n = len(scene)
        self.flips = flips if flips is not None else Flips.none(n)
        self.final = TileFrame.final(scene)
        self.start_center = np.asarray(start_center, dtype=np.float64)
        self.start_rotation = np.asarray(start_rotation, dtype=np.float64)
        self.delay = np.broadcast_to(np.asarray(delay, dtype=np.float64), (n,))
        self.travel = max(float(travel), 1e-6)
        self.start_height = max(float(start_height), 0.0)
        self.apex = max(float(apex), self.start_height)
        self.gravity = self._gravity(self.travel, self.apex, self.start_height)
        # Upward speed at take-off (0 for a drop from rest).
        self.speed = math.sqrt(2.0 * self.gravity * (self.apex - self.start_height))
        self.bounce_speed = self._bounce_speeds(self.travel, self.apex, bounce, self.start_height)
        self.bounce_time = 2.0 * self.bounce_speed / max(self.gravity, 1e-12)
        self.bounce_start = np.concatenate([[0.0], np.cumsum(self.bounce_time)])
        self.wobble = float(wobble)
        self.settle = self.settle_time(self.travel, self.apex, bounce, wobble, self.start_height)
        self.duration = float(self.delay.max() + self.travel + self.settle) if n else 0.0

    @staticmethod
    def _gravity(travel: float, apex: float, start_height: float = 0.0) -> float:
        """Gravity for an arc from start_height up to apex and down to 0 in `travel`:
        rising takes sqrt(2 (apex - start) / g), falling sqrt(2 apex / g)."""
        rise_and_fall = math.sqrt(max(apex - start_height, 0.0)) + math.sqrt(max(apex, 0.0))
        return 2.0 * rise_and_fall**2 / travel**2

    @classmethod
    def _bounce_speeds(cls, travel: float, apex: float, bounce: float,
                       start_height: float = 0.0) -> np.ndarray:  # fmt: skip
        """Take-off speed of each bounce (each keeps `bounce` of the last one's speed)."""
        speeds = []
        apex = max(apex, start_height)
        if apex <= 0 or bounce <= 0:
            return np.zeros(0)
        gravity = cls._gravity(travel, apex, start_height)
        v = math.sqrt(2.0 * gravity * apex) * bounce  # impact speed falling from the apex
        while len(speeds) < cls.MAX_BOUNCES and v * v / (2 * gravity) >= cls.MIN_BOUNCE * apex:
            speeds.append(v)
            v *= bounce
        return np.array(speeds)

    @classmethod
    def settle_time(cls, travel: float, apex: float, bounce: float, wobble: float,
                    start_height: float = 0.0) -> float:  # fmt: skip
        """Seconds from first impact until a tile lies still."""
        travel = max(float(travel), 1e-6)
        apex = max(float(apex), float(start_height), 0.0)
        speeds = cls._bounce_speeds(travel, apex, bounce, start_height)
        gravity = cls._gravity(travel, apex, start_height) if apex > 0 else 1.0
        bounces = float((2.0 * speeds / gravity).sum()) if len(speeds) else 0.0
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
        height = self.start_height + self.speed * flight - 0.5 * self.gravity * flight**2

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
        height = np.where(rest, 0.0, np.maximum(height, 0.0))
        # At rest first, then tiles that have touched down (still bouncing or settling),
        # both in stacking order: a tile may land on one still settling. Then tiles
        # still falling, from low to high: nearer the camera covers what is further down.
        falling = ~landed
        order = np.lexsort((np.arange(len(tau)), np.where(falling, height, 0.0), falling, ~rest))
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
            height=height,
            rest=None if rest.all() else rest,
        )


RANDOM_SWEEPS = 100  # Gibbs sweeps for random_landing_order (settled after about 20)


def random_landing_order(scene: MosaicScene, seed: int, sweeps: int = RANDOM_SWEEPS) -> np.ndarray:
    """(N,) a uniformly random landing order in which every tile lands after those
    it lies on (scene.overlaps): what a random ordering means under that rule.

    Each tile gets an independent uniform key in [0, 1) and tiles land in key
    order; conditioning the keys on lower < upper for every overlapping pair
    makes that order uniform over the valid orders. The conditioned keys are
    sampled by Gibbs sweeps: each key is redrawn uniformly between the largest
    key of the tiles under it and the smallest of the tiles on it (starting
    from stacking order, which is valid). Without overlaps the first sweep is
    already a plain random order.
    """
    n = len(scene)
    pairs = scene.overlaps
    by_upper = pairs[np.argsort(pairs[:, 1], kind="stable")]
    by_lower = pairs[np.argsort(pairs[:, 0], kind="stable")]
    keys = (np.arange(n) + 0.5) / max(n, 1)
    _conditioned_keys(
        keys, np.searchsorted(by_upper[:, 1], np.arange(n + 1)), by_upper[:, 0].copy(),
        np.searchsorted(by_lower[:, 0], np.arange(n + 1)), by_lower[:, 1].copy(),
        int(sweeps), int(seed),
    )  # fmt: skip
    rank = np.empty(n, np.int64)
    rank[np.argsort(keys, kind="stable")] = np.arange(n)
    return rank


@njit(cache=True, nogil=True)
def _conditioned_keys(keys, below_start, below, above_start, above, sweeps, seed):
    np.random.seed(seed)
    for _ in range(max(sweeps, 1)):
        for i in range(len(keys)):
            lo = 0.0
            for k in range(below_start[i], below_start[i + 1]):
                lo = max(lo, keys[below[k]])
            hi = 1.0
            for k in range(above_start[i], above_start[i + 1]):
                hi = min(hi, keys[above[k]])
            keys[i] = lo + np.random.random() * (hi - lo)


def landing_gap(n: int) -> int:
    """Landings between a tile and the next one allowed on top of it (see landing_order)."""
    return int(np.clip(n // 150, 3, 50))


def landing_order(scene: MosaicScene, preference, gap: int | None = None) -> np.ndarray:
    """(N,) each tile's place in the landing sequence (0 lands first).

    Tiles land by preference (lower values first; ties by stacking order),
    except that a tile never lands before every tile below it that it
    overlaps (scene.overlaps): at each step the most preferred tile whose
    support has fully landed goes next.

    For orderings with a direction; random ones use random_landing_order
    (see the module docstring).

    A tile whose last support has just landed waits `gap` more landings
    (default landing_gap) while any other tile is ready. Without that,
    landing one tile frees the tiles on it, which, held back, are usually
    the most preferred: the sequence would climb stacks in one spot after
    another (on a photo pile, the next tile landed on one of the last few
    about 70% of the time). With it, landings spread over the mosaic as
    they would without the overlap rule, keeping the preferred order.
    """
    n = len(scene)
    preference = np.asarray(preference, dtype=np.float64)
    pairs = scene.overlaps
    pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
    start = np.searchsorted(pairs[:, 0], np.arange(n + 1))  # tiles above each, by lower
    waiting = np.bincount(pairs[:, 1], minlength=n).astype(np.int64)
    gap = landing_gap(n) if gap is None else max(int(gap), 0)
    return _landing_order(preference, start, pairs[:, 1].copy(), waiting, gap)


@njit(cache=True, nogil=True)
def _landing_order(preference, start, above, waiting, gap):
    # A binary min-heap of ready tiles keyed by (preference, index), and a queue of
    # tiles freed recently (each with the step from which it may go).
    n = len(preference)
    heap = np.empty(n, np.int64)
    size = 0
    for i in range(n):
        if waiting[i] == 0:
            heap[size] = i
            size += 1
    for k in range(size // 2 - 1, -1, -1):
        _sift_down(heap, size, k, preference)
    cooling = np.empty(n, np.int64)
    eligible = np.empty(n, np.int64)
    first = last = 0
    rank = np.empty(n, np.int64)
    for step in range(n):
        while first < last and (eligible[first] <= step or size == 0):
            size = _push(heap, size, cooling[first], preference)
            first += 1
        tile = heap[0]
        size -= 1
        heap[0] = heap[size]
        _sift_down(heap, size, 0, preference)
        rank[tile] = step
        for k in range(start[tile], start[tile + 1]):
            upper = above[k]
            waiting[upper] -= 1
            if waiting[upper] == 0:  # its support has landed: ready after the gap
                if gap == 0:
                    size = _push(heap, size, upper, preference)
                else:
                    cooling[last] = upper
                    eligible[last] = step + 1 + gap
                    last += 1
    return rank


@njit(cache=True, nogil=True)
def _push(heap, size, tile, preference):
    heap[size] = tile
    j = size
    while j > 0:
        parent = (j - 1) // 2
        if not _before(heap[j], heap[parent], preference):
            break
        heap[j], heap[parent] = heap[parent], heap[j]
        j = parent
    return size + 1


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

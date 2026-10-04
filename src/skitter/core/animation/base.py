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

Overlapping tiles (a photo pile) must land bottom first; landing.py makes
such orders, for orderings with a direction and for random ones.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar

import numpy as np

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
    damped wobble of `wobble` radians (its sign is the way the tile first
    rocks). `bounce` and `wobble` are one value or one per tile. Everything
    is closed form, so frames are exact at any time. A tile counts as at
    rest (and draws among the tiles on the table) only once it has settled,
    `settle[i]` seconds after impact.

    Stacked tiles (`cover`: seconds from each tile's impact until a tile
    lying on it lands; see cover_after_impact): a covered tile makes only the
    hops that end back on the table before the cover lands; a later hop
    would show it rising (and, seen from the camera, growing) under a tile
    lying flat on it. Its rocking goes on past the cover and dies out within
    COVER_DAMP seconds, as if pressed down. Impacts are never moved, so
    landing orders and pacing are unaffected.
    """

    MAX_BOUNCES = 4
    MIN_BOUNCE = 0.01  # bounces lower than this share of the apex are dropped
    COVER_DAMP = 0.15  # seconds a covered tile keeps rocking after a tile lands on it

    def __init__(self, scene: MosaicScene, start_center, start_rotation, delay, travel: float,
                 apex: float, bounce=0.0, wobble=0.0, flips: Flips | None = None,
                 start_height: float = 0.0, cover=None):  # fmt: skip
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
        bounce = np.broadcast_to(np.asarray(bounce, dtype=np.float64), (n,))
        speeds = self.bounce_table(self.travel, self.apex, bounce, self.start_height)
        g = max(self.gravity, 1e-12)
        free_end = (2.0 * speeds / g).sum(axis=1)  # hopping time with nothing on top
        self.cover = (np.full(n, np.inf) if cover is None
                      else np.broadcast_to(np.asarray(cover, dtype=np.float64), (n,)))  # fmt: skip
        in_time = np.cumsum(2.0 * speeds / g, axis=1) <= self.cover[:, None] + 1e-12
        self.bounce_speed = np.where(in_time, speeds, 0.0)  # (N, MAX_BOUNCES), 0: none
        self.bounce_time = 2.0 * self.bounce_speed / g
        self.bounce_start = np.concatenate(
            [np.zeros((n, 1)), np.cumsum(self.bounce_time, axis=1)], axis=1
        )  # (N, MAX_BOUNCES + 1): each hop's start, then the end of the last
        self.wobble = np.broadcast_to(np.asarray(wobble, dtype=np.float64), (n,))
        # Rocking takes as long as a free tile's settling; a covered tile's dies out soon
        # after the cover lands.
        self.rock = np.where(self.wobble != 0, np.maximum(free_end, 0.3 * self.travel), 0.0)
        rock_end = np.minimum(self.rock, self.cover + self.COVER_DAMP)
        self.settle = np.maximum(self.bounce_start[:, -1], rock_end)  # (N,)
        self.duration = float((self.delay + self.travel + self.settle).max()) if n else 0.0

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
    def bounce_table(cls, travel: float, apex: float, bounce, start_height: float = 0.0):
        """(N, MAX_BOUNCES) take-off speed of each tile's bounces (0: no more bounces),
        as _bounce_speeds for each tile's own `bounce`."""
        bounce = np.asarray(bounce, dtype=np.float64).reshape(-1)
        apex = max(apex, start_height)
        speeds = np.zeros((len(bounce), cls.MAX_BOUNCES))
        if apex <= 0:
            return speeds
        gravity = cls._gravity(travel, apex, start_height)
        k = np.arange(1, cls.MAX_BOUNCES + 1)
        speeds = math.sqrt(2.0 * gravity * apex) * np.maximum(bounce, 0.0)[:, None] ** k
        high = speeds**2 / (2 * gravity) >= cls.MIN_BOUNCE * apex
        return np.where(np.cumprod(high, axis=1).astype(bool), speeds, 0.0)

    @classmethod
    def settle_times(cls, travel: float, apex: float, bounce, wobble,
                     start_height: float = 0.0) -> np.ndarray:  # fmt: skip
        """(N,) settle_time of each tile on its own (nothing landing on it)."""
        travel = max(float(travel), 1e-6)
        apex = max(float(apex), float(start_height), 0.0)
        bounce, wobble = np.broadcast_arrays(np.asarray(bounce, float), np.asarray(wobble, float))
        speeds = cls.bounce_table(travel, apex, bounce, start_height)
        gravity = cls._gravity(travel, apex, start_height) if apex > 0 else 1.0
        bounces = (2.0 * speeds / gravity).sum(axis=1)
        return np.where(wobble.reshape(-1) != 0, np.maximum(bounces, 0.3 * travel), bounces)

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
        rows = np.arange(len(tau))
        # Each tile's current hop: the last one started (all start at 0 once there are no more).
        k = (after[:, None] >= self.bounce_start[:, :-1]).sum(axis=1) - 1
        k = np.clip(k, 0, self.MAX_BOUNCES - 1)
        s = after - self.bounce_start[rows, k]
        bouncing = landed & (after < self.bounce_start[:, -1])
        hop = self.bounce_speed[rows, k] * s - 0.5 * self.gravity * s**2
        height = np.where(bouncing, np.maximum(hop, 0.0), np.where(landed, 0.0, height))
        if np.any(self.wobble):
            u = np.clip(after / np.maximum(self.rock, 1e-9), 0.0, 1.0)
            swing = self.wobble * np.exp(-4.0 * u) * (1.0 - u) * np.sin(3.0 * math.tau * u)
            pressed = np.clip((after - self.cover) / self.COVER_DAMP, 0.0, 1.0)  # 1: still
            swing = swing * (1.0 - pressed) ** 2
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


def cover_after_impact(scene: MosaicScene, impact) -> np.ndarray:
    """(N,) seconds from each tile's impact until the first tile lying on it lands
    (inf: nothing does). impact: (N,) when each tile first touches the table."""
    impact = np.asarray(impact, dtype=np.float64)
    first = np.full(len(scene), np.inf)
    lower, upper = scene.overlaps.T
    np.minimum.at(first, lower, impact[upper])
    return first - impact


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

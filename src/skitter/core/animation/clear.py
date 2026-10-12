"""Built-in clear choreographies: taking the finished mosaic away (phases.py).

A clear timeline starts exactly at the finished mosaic and ends with every
tile gone, so the video can loop into a build from an empty table.

Blow Away: a gust sweeps across the picture (`line_front`, as a Wave
travels) and blows the tiles off the table. Everything is closed form, so
frames are exact at any time:

- When the gust reaches a tile it sets off from rest and picks up the
  wind's speed W at its own rate k (a drag model: speed W (1 - e^(-k tau)),
  so it has blown `downwind` W (tau - (1 - e^(-k tau)) / k) after tau
  seconds). Its share of the wind's speed, s, drives the rest:
- It skids along the table, rattling (Skid jitter), until s reaches its
  skid share, then lifts (height rising with the square of the speed above
  that, as lift grows with airspeed) toward its own ceiling, bobbing and
  drifting sideways in the gusts. It spins and tumbles at rates in turns a
  second at full speed.
- Piles leave top first: a tile can't blow away while another lies on it,
  so a covered tile sets off only once every tile lying on it is PEEL of
  the way clear of it (`_release_times`). Not all the way: tiles moving
  draw from low to high, so the tile on top stays on top as the one under
  it slides out, and waiting for each to clear the next would add up along
  a pile's chains of overlapping tiles (15 deep in a 2,655-tile pile), all
  but stopping the gust in a pile.
- Tiles fly `distance` picture lengths (along the wind) past the picture's
  downwind edge, and are gone once all of a tile is that far out. How far
  the video or the Animate view shows around the picture depends on their
  framing and zoom, so the distance is a setting: far enough, and tiles
  leave the frame before they go. Lifting only carries a tile further out
  (perspective), so the table-plane distance decides. The wind blows
  exactly as hard as it takes for the last tile to be gone at the
  duration.
"""

import math

import numpy as np
from numba import njit

from skitter.core.animation import compass
from skitter.core.animation.base import (
    Choreography,
    Flips,
    Note,
    TileFrame,
    Timeline,
    register_choreography,
    seconds,
)
from skitter.core.animation.look import AnimationLook, scene_extent
from skitter.core.animation.show import HEIGHT_LIMIT
from skitter.core.scene import MosaicScene
from skitter.core.slicing.common import seed_param
from skitter.core.slicing.params import FloatParam, RangeParam, TileSizeParam

BOB = 0.25  # gusts move a tile up and down by up to this share of its height (at 100%)
DRIFT = 2.0  # gusts drift a tile sideways by up to this many base tiles (at 100%)
RATTLE_DECAY = 12.0  # the rattle dies out as the share of the wind's speed grows past the skid
PEEL = 0.25  # a covered tile sets off once the tiles on it are this share of the way clear
SEARCH = 1e6  # the wind's speed is searched within this factor of the mosaic a second


@njit(cache=True, nogil=True)
def _blown_time(k, distance, wind):
    """Seconds until a tile setting off from rest has blown `distance` downwind:
    W (tau - (1 - e^(-k tau)) / k) = distance. With u = k tau that is
    u - 1 + e^(-u) = k distance / W, convex and rising: Newton from the right."""
    if distance <= 0.0:
        return 0.0
    c = k * distance / wind
    u = c + 1.0
    for _ in range(100):
        e = math.expm1(-u)  # e^(-u) - 1, exact for small u
        step = (u + e - c) / -e
        u -= step
        if step <= 1e-12 * u:
            break
    return u / k


@njit(cache=True, nogil=True)
def _release_times(wind, k, front, exit_distance, lower, upper, clear_distance, release, gone):
    """In place: when each tile sets off (release) and is gone (gone); returns
    the last to leave. A tile sets off when the gust reaches it (front), or later, once
    every tile lying on it has blown `clear_distance` (pairs sorted by lower tile,
    highest first, so each upper tile is final before the tiles under it)."""
    release[:] = front
    for p in range(len(lower)):
        i, j = lower[p], upper[p]
        r = release[j] + _blown_time(k[j], clear_distance[p], wind)
        if r > release[i]:
            release[i] = r
    end = 0.0
    for i in range(len(k)):
        gone[i] = release[i] + _blown_time(k[i], exit_distance[i], wind)
        if gone[i] > end:
            end = gone[i]
    return end


def half_extent(size: np.ndarray, rotation: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """(N,) half of each rotated rectangle's extent along a unit direction."""
    cos, sin = np.cos(rotation), np.sin(rotation)
    along_x = np.abs(cos * direction[0] + sin * direction[1])  # the tile's own x axis
    along_y = np.abs(-sin * direction[0] + cos * direction[1])
    return size[:, 0] / 2 * along_x + size[:, 1] / 2 * along_y


class WindTimeline(Timeline):
    """Tiles blown away downwind (see the module docstring).

    Per tile: release (seconds), rate k (1/s), skid (share of the wind's speed),
    ceiling (mosaic units), spin and tumble (turns a second at full speed), rattle
    (radians), and the gusts' bob and drift.
    """

    def __init__(self, scene: MosaicScene, wind: float, direction: np.ndarray, release, gone,
                 k, skid, ceiling, spin, flips: Flips, rattle, waves, duration: float):  # fmt: skip
        self.final = TileFrame.final(scene)
        self.wind = float(wind)
        self.direction = np.asarray(direction, np.float64)
        self.across = np.array([-self.direction[1], self.direction[0]])
        self.release, self.gone, self.k = release, gone, k
        self.skid, self.ceiling, self.spin, self.flips = skid, ceiling, spin, flips
        self.rattle = rattle
        # (amplitude, angular frequency, phase) of the bob, the drift, the rattle and
        # the rattle's sideways shuffle, each (N,)
        self.bob, self.drift, self.shake, self.shuffle = waves
        self.reach = np.hypot(scene.size[:, 0], scene.size[:, 1]) / 2
        safe = np.minimum(skid, 1.0 - 1e-9)
        lift_off = -np.log1p(-safe) / k  # seconds until the tile lifts
        self.lift_off_distance = self.wind * (lift_off - safe / k)
        self.duration = float(duration)

    @staticmethod
    def _wave(wave, tau):
        amplitude, omega, phase = wave
        return amplitude * np.sin(omega * tau + phase)

    def frame(self, t: float) -> TileFrame:
        t = min(max(float(t), 0.0), self.duration)
        if t <= 0.0:
            return self.final  # exactly, whatever the rounding of the times
        b = self.final
        tau = t - self.release
        moving = tau > 0
        tau = np.maximum(tau, 0.0)
        speed = -np.expm1(-self.k * tau)  # share of the wind's speed, 0 at rest
        downwind = self.wind * (tau - speed / self.k)
        skid = self.skid
        rising = np.clip((speed - skid) / np.maximum(1.0 - skid, 1e-9), 0.0, 1.0)
        height = self.ceiling * rising**2 * (1.0 + self._wave(self.bob, tau))
        # The rattle grows as the tile picks up speed on the table and dies out as it lifts.
        skidding = np.where(speed <= skid, speed / np.maximum(skid, 1e-9),
                            np.exp(-RATTLE_DECAY * (speed - skid)))  # fmt: skip
        skidding = np.where(skid > 0, skidding, 0.0)
        sideways = speed * self._wave(self.drift, tau) + skidding * self._wave(self.shuffle, tau)
        center = (b.center + downwind[:, None] * self.direction
                  + sideways[:, None] * self.across)  # fmt: skip
        rotation = (b.rotation + self.spin * math.tau * downwind / self.wind
                    + skidding * self._wave(self.shake, tau))  # fmt: skip
        size, facing = b.size, None
        if self.flips:  # tumbling once in the air, at full speed `turns` a second
            airborne = np.maximum(downwind - self.lift_off_distance, 0.0) / self.wind
            size, facing = self.flips.apply(b.size, airborne)
        height = np.where(moving, np.maximum(height, 0.0), 0.0)
        # At rest in stacking order, then everything moving from low to high.
        order = np.lexsort((np.arange(len(tau)), height, moving))
        return TileFrame(
            center=center,
            size=size,
            rotation=rotation,
            alpha=np.where(t >= self.gone, 0.0, 1.0),
            tint=b.tint,
            order=order,
            height=height,
            rest=~moving,
            facing=facing,
        )


@register_choreography
class BlowAwayChoreography(Choreography):
    id = "blow_away"
    name = "Blow Away"
    phase = "clear"
    description = (
        "A gust sweeps across the mosaic: tiles skid, lift and tumble away on the wind "
        "until none is left. In a pile, the top tiles go first."
    )

    direction = compass.direction_param("right", "wind")
    angle = compass.angle_param(90.0, "wind", "direction")
    duration = FloatParam(
        4.0, "Duration", min=0.5, max=600.0, step=0.5, decimals=1, suffix=" s",
        help="From the first gust until the last tile has flown its distance; the wind "
             "blows as hard as that takes.",
    )  # fmt: skip
    distance = FloatParam(
        1.0, "Distance", min=0.1, max=10.0, step=0.25, decimals=2, suffix=" × picture",
        help="How far tiles fly past the picture's downwind edge before they're gone, in "
             "lengths of the picture along the wind. Raise it if tiles vanish inside the "
             "video's frame; farther takes a stronger wind.",
    )  # fmt: skip
    sweep = FloatParam(
        60.0, "Sweep", min=0.0, max=90.0, step=5.0, decimals=0, suffix="%",
        help="How long the gust takes to cross the picture, as a share of the duration: "
             "it reaches the tiles one after another (0: all at once).",
    )  # fmt: skip
    raggedness = FloatParam(
        30.0, "Raggedness", min=0.0, max=100.0, step=5.0, decimals=0, suffix="%",
        under="sweep", when=lambda c: c.sweep > 0,
        help="How unevenly the gust reaches the tiles: 0 strictly in line across the "
             "picture, 100 in random order.",
    )  # fmt: skip
    pickup = FloatParam(
        0.5, "Pickup", min=0.05, max=5.0, step=0.05, decimals=2, suffix=" s",
        help="How long a tile takes to pick up most of the wind's speed (each tile a "
             "little more or less).",
    )  # fmt: skip
    skid = FloatParam(
        30.0, "Skid", min=0.0, max=90.0, step=5.0, decimals=0, suffix="%",
        help="How fast a tile slides along the table before the wind lifts it, as a share "
             "of the wind's speed (0: it lifts at once).",
    )  # fmt: skip
    skid_jitter = FloatParam(
        8.0, "Skid jitter", min=0.0, max=45.0, step=1.0, decimals=0, suffix="°",
        under="skid", when=lambda c: c.skid > 0,
        help="How much tiles rattle while they skid: a quick shake of up to this many "
             "degrees, and a little sideways (0: they slide smoothly).",
    )  # fmt: skip
    lift = TileSizeParam(
        3.0, "Lift", min=0.0, max=50.0, step=0.5,
        help="How high tiles rise in the wind, in base tiles (each at least half of it; "
             "shadows grow with it).",
    )  # fmt: skip
    gusts = FloatParam(
        40.0, "Gusts", min=0.0, max=100.0, step=5.0, decimals=0, suffix="%",
        help="How much tiles bob up and down and drift sideways in the gusting wind "
             "(0: they fly straight).",
    )  # fmt: skip
    spin = RangeParam(
        (0.0, 0.5), "Spin", min=0.0, max=5.0, step=0.1, suffix=" turns/s",
        help="How fast tiles turn flat in the wind at full speed, either way: each tile "
             "draws its own rate from this range.",
    )  # fmt: skip
    tumble = RangeParam(
        (0.25, 1.0), "Tumble", min=0.0, max=5.0, step=0.1, suffix=" turns/s",
        help="How fast tiles flip over in the air at full speed: each tile draws its own "
             "rate from this range (0: they stay face up).",
    )  # fmt: skip
    seed = seed_param()

    @property
    def degrees(self) -> float:
        """The way the wind blows, degrees clockwise from the top."""
        return compass.degrees(self.direction, self.angle)

    def timeline(self, scene: MosaicScene, look: AnimationLook | None = None) -> Timeline:
        look = look or AnimationLook()
        params = type(self)
        rng = np.random.default_rng(self.seed)
        n, total = len(scene), float(self.duration)
        tile = scene.tile_size[0]
        direction = compass.heading(self.degrees)

        # The gust reaches each tile in turn (raggedness mixes in a random order).
        along = scene.center @ direction if n else np.zeros(0)
        span = float(np.ptp(along)) if n else 0.0
        line = (along - along.min()) / span if span > 0 else np.zeros(n)
        ragged = self.raggedness / 100.0
        front = self.sweep / 100.0 * total * ((1.0 - ragged) * line + ragged * rng.random(n))
        if n:
            front -= front.min()

        k = rng.uniform(0.7, 1.3, n) / self.pickup
        skid = np.clip(self.skid / 100.0 * rng.uniform(0.6, 1.4, n), 0.0, 0.95)
        gusts = self.gusts / 100.0
        limit = HEIGHT_LIMIT * look.camera_height * scene_extent(scene)  # bobbing aside
        top = min(self.lift * tile, limit)
        ceiling = top * rng.uniform(0.5, 1.0, n)
        spin = params.spin.draw(self.spin, rng, n) * rng.choice([-1.0, 1.0], n)
        flips = Flips(turns=params.tumble.draw(self.tumble, rng, n) * rng.choice([-1.0, 1.0], n),
                      axis=rng.integers(0, 2, n))  # fmt: skip
        rattle = math.radians(self.skid_jitter) * rng.uniform(0.5, 1.0, n)
        reach = np.hypot(scene.size[:, 0], scene.size[:, 1]) / 2

        def wave(amplitude, low, high):  # (amplitude, angular frequency, phase), per tile
            return (amplitude, math.tau * rng.uniform(low, high, n),
                    rng.uniform(0.0, math.tau, n))  # fmt: skip

        waves = (
            wave(BOB * gusts, 0.5, 1.2),
            wave(DRIFT * gusts * tile * rng.uniform(0.3, 1.0, n), 0.3, 0.8),
            wave(rattle, 5.0, 8.0),
            wave(0.5 * rattle * reach, 6.0, 9.0),
        )

        # Gone: `distance` picture lengths past the downwind edge, all of the tile.
        x0, y0, x1, y1 = scene.bounds if n else (0.0, 0.0, 1.0, 1.0)
        middle = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        length = abs((x1 - x0) * direction[0]) + abs((y1 - y0) * direction[1])
        out = middle @ direction + (0.5 + self.distance) * length
        exit_distance = out + reach - along

        # Piles: a tile under another sets off once the one on top is PEEL of the way clear.
        pairs = scene.overlaps
        pairs = pairs[np.argsort(-pairs[:, 0], kind="stable")]
        lower = np.ascontiguousarray(pairs[:, 0])
        upper = np.ascontiguousarray(pairs[:, 1])
        half = half_extent(scene.size, scene.rotation, direction)
        clear_distance = PEEL * np.maximum(half[lower] + reach[upper]
                                           - (along[upper] - along[lower]), 0.0)  # fmt: skip

        release, gone = np.zeros(n), np.zeros(n)

        def last_gone(wind: float) -> float:
            return _release_times(wind, k, front, exit_distance, lower, upper, clear_distance,
                                  release, gone)  # fmt: skip

        # The wind's speed: the last tile leaves the view at the duration (a stronger wind
        # clears sooner). Search between a breeze and a blast, ending on one in time.
        extent = scene_extent(scene)
        lo, hi = extent / SEARCH, extent * SEARCH
        if n and last_gone(hi) <= total:
            for _ in range(100):
                mid = math.sqrt(lo * hi)
                lo, hi = (lo, mid) if last_gone(mid) <= total else (mid, hi)
                if hi / lo < 1.0 + 1e-9:
                    break
        wind = hi
        last_gone(wind)

        timeline = WindTimeline(scene, wind, direction, release, gone, k, skid, ceiling, spin,
                                flips, rattle, waves, total)  # fmt: skip
        notes = [Note("duration", f"The wind blows {wind / tile:.1f} tiles a second."),
                 Note("distance", f"{self.distance * length / tile:.0f} tiles past the "
                                  "edge.")]  # fmt: skip
        if self.sweep > 0:
            notes.append(Note("sweep", f"The gust takes {seconds(float(front.max(initial=0)))} "
                                       "to cross the picture."))  # fmt: skip
        waiting = int(np.count_nonzero(release > front + 1e-9))
        if waiting:
            notes.append(Note("sweep", f"{waiting} tiles wait for the tiles lying on them to "
                                       "blow away first."))  # fmt: skip
        if top < self.lift * tile - 1e-9:
            notes.append(Note("lift", f"Limited to {top / tile:.1f} tiles: higher would "
                                      "come too near the camera (Look: Camera height).",
                              True))  # fmt: skip
        timeline.notes = tuple(notes)
        return timeline

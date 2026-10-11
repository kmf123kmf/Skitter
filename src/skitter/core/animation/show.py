"""Built-in show choreographies: what happens on the finished mosaic (phases.py).

A show timeline starts and ends exactly at the finished mosaic; its effects
are offsets from it that vanish at both ends, so a show fits between any
build and any clear. A plain pause on the finished mosaic needs no show: it
is the build's hold after (phases.PhaseHolds).

Shows that lift tiles describe a surface under them and leave piles to
lift.py (which tiles move, settling, drawing order, slope shading). Ripple
and Wave build their surface from two parts:

- a front (`Front`): how far it has travelled when it reaches each point,
  and which way it travels there. Ripple's is a ring spreading from the drop
  point (`ring_front`), Wave's a straight line sweeping across
  (`line_front`).
- a pulse (`wave_pulse`): the surface's height after the front passes, a
  few raised-cosine bumps, each lower than the last.

`SurfaceTimeline` runs any front with the pulse at constant speed, so that
the last bump leaves the farthest point exactly at the duration.
"""

import math
from dataclasses import dataclass

import numpy as np

from skitter.core.animation import compass
from skitter.core.animation.base import Choreography, TileFrame, Timeline, register_choreography
from skitter.core.animation.lift import Lift
from skitter.core.animation.look import AnimationLook, scene_extent
from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import (
    BoolParam,
    FloatParam,
    IntParam,
    TileSizeParam,
)

DECAY = 0.6  # each bump's height, relative to the one before it
HEIGHT_LIMIT = 0.25  # surfaces rise at most this share of the camera height


def wave_pulse(tau: np.ndarray, period: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """(height, its rate of change) of the surface tau seconds after the front passed:
    `count` raised-cosine bumps of `period` seconds each, peaking at 1, DECAY, ...
    Zero (and flat) before the front and after the last bump."""
    k = np.floor(tau / period)
    on = (tau >= 0) & (k < count)
    k = np.where(on, k, 0)
    w = 2.0 * math.pi / period
    s = tau - k * period
    scale = np.where(on, DECAY**k, 0.0)
    return scale * 0.5 * (1.0 - np.cos(w * s)), scale * 0.5 * w * np.sin(w * s)


@dataclass(frozen=True)
class Front:
    """Where a front reaches: `distance` (N, 5) it has travelled when it reaches each of a
    tile's points (Lift.points), and `direction` (N, 2) it travels at each tile's center
    (unit vectors)."""

    distance: np.ndarray
    direction: np.ndarray

    @property
    def reach(self) -> float:
        """How far it travels to reach every point."""
        return float(self.distance.max()) if self.distance.size else 0.0


def ring_front(points: np.ndarray, origin) -> Front:
    """A ring spreading from origin."""
    offset = points - np.asarray(origin, np.float64)
    center = offset[:, 0]
    outward = center / np.maximum(np.hypot(*center.T), 1e-9)[:, None]
    return Front(np.hypot(offset[..., 0], offset[..., 1]), outward)


def line_front(points: np.ndarray, angle: float) -> Front:
    """A straight line sweeping across, travelling `angle` degrees clockwise from the
    top; it starts at the first point it meets."""
    direction = compass.heading(angle)
    along = points @ direction
    start = float(along.min()) if along.size else 0.0
    return Front(along - start, np.broadcast_to(direction, (len(points), 2)).copy())


class SurfaceTimeline(Timeline):
    """A front (see Front) carrying `count` bumps across the tiles at constant speed: the
    first leaves at 0, the last leaves the farthest point exactly at `duration`.
    wavelength and amplitude in mosaic units; fading (0..1) lowers the bumps with the
    distance travelled, to `1 - fading` at the far end."""

    def __init__(self, lift: Lift, front: Front, duration: float, wavelength: float,
                 amplitude: float, count: int, fading: float):  # fmt: skip
        self.lift = lift
        self.final = lift.final
        self.front = front
        self.duration = float(duration)
        self.count = int(count)
        reach = front.reach
        self.speed = (reach + self.count * wavelength) / self.duration
        self.period = wavelength / self.speed
        self.amplitude = amplitude * (1.0 - fading * front.distance / max(reach, 1e-9))  # (N, 5)

    def frame(self, t: float) -> TileFrame:
        t = float(t)
        if t <= 0.0 or t >= self.duration:
            return self.final  # exactly, whatever the rounding of the times
        pulse, rate = wave_pulse(t - self.front.distance / self.speed, self.period, self.count)
        # h(x, t) = A pulse(t - d / speed): the slope runs along the travel, -A rate / speed.
        slope = (-self.amplitude[:, 0] * rate[:, 0] / self.speed)[:, None] * self.front.direction
        return self.lift.frame(self.amplitude * pulse, slope)


def _surface(choreography, scene: MosaicScene, look: AnimationLook | None, front_of,
             count: int) -> SurfaceTimeline:  # fmt: skip
    """The timeline of a show with the shared settings below; front_of(points) -> Front."""
    look = look or AnimationLook()
    lift = Lift(scene, choreography.still_below / 100.0, look, choreography.shading)
    tile = scene.tile_size[0]
    amplitude = min(choreography.amplitude * tile,
                    HEIGHT_LIMIT * look.camera_height * scene_extent(scene))  # fmt: skip
    return SurfaceTimeline(lift, front_of(lift.points), choreography.duration,
                           choreography.wavelength * tile, amplitude, count,
                           choreography.fading / 100.0)  # fmt: skip


# Settings Ripple and Wave share (a Param belongs to one class: each call makes one).


def duration_param(what: str) -> FloatParam:
    return FloatParam(
        4.0, "Duration", min=0.5, max=600.0, step=0.5, decimals=1, suffix=" s",
        help=f"From the start until the last {what} has passed the farthest tile; they "
             "travel as fast as that takes.",
    )  # fmt: skip


def wavelength_param(what: str) -> TileSizeParam:
    return TileSizeParam(
        6.0, "Wavelength", min=0.5, max=50.0, step=0.5,
        help=f"Distance from one {what}'s crest to the next, in base tiles: how wide each "
             f"{what} is.",
    )  # fmt: skip


def amplitude_param(what: str) -> TileSizeParam:
    return TileSizeParam(
        3.0, "Amplitude", min=0.05, max=10.0, step=0.1,
        help=f"How high the first {what} lifts tiles, in base tiles (shadows and shading "
             "grow with it).",
    )  # fmt: skip


def fading_param(what: str, default: float) -> FloatParam:
    return FloatParam(
        default, "Fading", min=0.0, max=100.0, step=5.0, decimals=0, suffix="%",
        help=f"How much lower the {what}s are by the time they reach the farthest tiles "
             "(100: flat there).",
    )  # fmt: skip


def shading_param(what: str) -> BoolParam:
    return BoolParam(
        True, "Slope shading",
        help=f"Shade tiles as if they tilted with the {what}s: lighter facing the light, "
             "darker facing away.",
    )  # fmt: skip


def still_below_param() -> FloatParam:
    return FloatParam(
        25.0, "Still below", min=0.0, max=100.0, step=5.0, decimals=0, suffix="% visible",
        help="In overlapping layouts, tiles showing less of themselves than this stay "
             "still (tiles lying on a moving tile move too). 0: every tile moves.",
    )  # fmt: skip


@register_choreography
class RippleChoreography(Choreography):
    id = "ripple"
    name = "Ripple"
    phase = "show"
    description = (
        "A stone dropped on the mosaic: rings of tiles rise and settle as they spread. "
        "In a pile, only tiles you can see move, and none passes through another."
    )

    center_x = FloatParam(
        50.0, "Drop X", min=0.0, max=100.0, step=5.0, decimals=0, suffix="%",
        help="Where the stone drops, across the picture (0: left edge, 100: right edge).",
    )  # fmt: skip
    center_y = FloatParam(
        50.0, "Drop Y", min=0.0, max=100.0, step=5.0, decimals=0, suffix="%",
        help="Where the stone drops, down the picture (0: top edge, 100: bottom edge).",
    )  # fmt: skip
    duration = duration_param("ring")
    wavelength = wavelength_param("ring")
    amplitude = amplitude_param("ring")
    rings = IntParam(
        2, "Rings", min=1, max=6,
        help="Rings following one another, each lower than the one before.",
    )  # fmt: skip
    fading = fading_param("ring", 50.0)
    shading = shading_param("ring")
    still_below = still_below_param()

    def timeline(self, scene: MosaicScene, look: AnimationLook | None = None) -> Timeline:
        origin = np.asarray(scene.canvas) * (self.center_x / 100.0, self.center_y / 100.0)
        return _surface(self, scene, look, lambda points: ring_front(points, origin), self.rings)


@register_choreography
class WaveChoreography(Choreography):
    id = "wave"
    name = "Wave"
    phase = "show"
    description = (
        "A wave sweeps across the mosaic: rows of tiles rise and settle as it passes. "
        "In a pile, only tiles you can see move, and none passes through another."
    )

    direction = compass.direction_param("right", "wave")
    angle = compass.angle_param(90.0, "wave", "direction")
    duration = duration_param("wave")
    wavelength = wavelength_param("wave")
    amplitude = amplitude_param("wave")
    count = IntParam(
        2, "Num waves", min=1, max=6,
        help="Waves following one another, each lower than the one before.",
    )  # fmt: skip
    fading = fading_param("wave", 0.0)
    shading = shading_param("wave")
    still_below = still_below_param()

    @property
    def degrees(self) -> float:
        """The direction of travel, degrees clockwise from the top."""
        return compass.degrees(self.direction, self.angle)

    def timeline(self, scene: MosaicScene, look: AnimationLook | None = None) -> Timeline:
        return _surface(self, scene, look, lambda points: line_front(points, self.degrees),
                        self.count)  # fmt: skip

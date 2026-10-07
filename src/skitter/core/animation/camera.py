"""Camera moves: how the video's frame travels over the table as the mosaic builds.

A camera move is independent of the choreography: both are functions of the
same animation time t (0 .. the timeline's duration), one telling where the
tiles are, the other what the frame shows. The renderer combines them.

A move animates the framing, not the table camera of look.py: that camera
stays above the middle of the mosaic, so perspective, shadows and the
choreographies' rules about it (tiles kept off its axis near it) hold. A
move is a long lens panning, zooming and turning over that picture. A `Shot`
is the table point at the middle of the frame, a zoom relative to the
video's framing (zoom 1 shows exactly the view_rect framing, as Static does)
and a turn (radians, clockwise: the picture looks turned the other way).

Moves plan a `CameraPath` for a scene, its timeline and that framing;
`path.shot(t)` is the shot at time t. Timing settings are shares of the
animation, so a move stays in step when the duration changes.

Most moves are made of `Segment`s, each a spiral similarity: the frame
scales and turns about one fixed table point, so nothing slides across the
frame on its own (a pure zoom when the turn doesn't change; a straight pan
when neither does). Zoom changes geometrically, so each step feels as large
as the last. Follow samples the timeline instead (`SampledPath`).

Add a move by subclassing CameraMove (declare Params, plan a path) and
decorating it with @register_camera_move.
"""

import cmath
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.ndimage import gaussian_filter1d

from skitter.core import easing
from skitter.core.animation.base import Timeline
from skitter.core.animation.video import content_rect
from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import ChoiceParam, Configurable, FloatParam, RangeParam

WorldRect = tuple[float, float, float, float]  # x, y, width, height (mosaic units)

EASES = {
    "smooth": easing.ease_in_out_cubic,
    "in": easing.ease_in_cubic,
    "out": easing.ease_out_cubic,
    "linear": easing.linear,
}
EASE_CHOICES = (
    ("smooth", "Smooth"),
    ("in", "Gentle start"),
    ("out", "Gentle stop"),
    ("linear", "Steady"),
)
PLACES = {  # named spots of the mosaic, as fractions across and down
    "top_left": (0.0, 0.0), "top": (0.5, 0.0), "top_right": (1.0, 0.0),
    "left": (0.0, 0.5), "center": (0.5, 0.5), "right": (1.0, 0.5),
    "bottom_left": (0.0, 1.0), "bottom": (0.5, 1.0), "bottom_right": (1.0, 1.0),
}  # fmt: skip
PLACE_CHOICES = tuple((key, key.replace("_", " ").capitalize()) for key in PLACES)


@dataclass(frozen=True)
class Shot:
    """What the frame shows at a moment."""

    center: tuple[float, float]  # the table point at the middle of the frame
    zoom: float = 1.0  # 1: the video's framing; 2: twice as close
    rotation: float = 0.0  # the frame's turn over the table, radians clockwise

    def size(self, base: WorldRect) -> tuple[float, float]:
        """The frame's width and height on the table, given the video's framing."""
        return base[2] / self.zoom, base[3] / self.zoom

    def corners(self, base: WorldRect) -> np.ndarray:
        """(4, 2) the frame's corners on the table: top left, top right, bottom right,
        bottom left (as the video shows them)."""
        w, h = self.size(base)
        local = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
        c, s = math.cos(self.rotation), math.sin(self.rotation)
        return np.asarray(self.center) + local @ np.array([[c, s], [-s, c]])


def home_shot(base: WorldRect) -> Shot:
    """The video's framing itself."""
    return Shot((base[0] + base[2] / 2, base[1] + base[3] / 2))


class CameraPath(ABC):
    """A planned camera move: the shot at any time."""

    @abstractmethod
    def shot(self, t: float) -> Shot:
        """The shot at animation time t (before the start and after the end: still)."""

    @property
    @abstractmethod
    def max_zoom(self) -> float:
        """The closest the path gets (textures need that much more detail)."""


@dataclass(frozen=True)
class Segment:
    """From one shot to another over [t0, t1], eased: a spiral similarity (see the
    module docstring). In complex numbers, the center is a - e^(i rotation) K / zoom
    for a fixed point a, which stays where it is in the frame throughout."""

    t0: float
    t1: float
    start: Shot
    end: Shot
    ease: str = "smooth"

    def progress(self, t: float) -> float:
        span = self.t1 - self.t0
        u = 1.0 if span <= 0 else min(max((t - self.t0) / span, 0.0), 1.0)
        return float(EASES[self.ease](u))

    @property
    def fixed_point(self) -> complex | None:
        """The table point that stays put in the frame (None: a straight pan)."""
        a, b = self.start, self.end
        k = (a.zoom / b.zoom) * cmath.exp(1j * (b.rotation - a.rotation))
        if abs(1 - k) < 1e-9:
            return None
        return (complex(*b.center) - k * complex(*a.center)) / (1 - k)

    def shot(self, t: float) -> Shot:
        a, b = self.start, self.end
        u = self.progress(t)
        if u <= 0.0 or u >= 1.0:
            return a if u <= 0.0 else b  # exactly, so the holds before and after match
        zoom = a.zoom * (b.zoom / a.zoom) ** u
        rotation = a.rotation + (b.rotation - a.rotation) * u
        fixed = self.fixed_point
        if fixed is None:
            c = complex(*a.center) + (complex(*b.center) - complex(*a.center)) * u
        else:
            k = (fixed - complex(*a.center)) * a.zoom * cmath.exp(-1j * a.rotation)
            c = fixed - cmath.exp(1j * rotation) * k / zoom
        return Shot((c.real, c.imag), zoom, rotation)


@dataclass(frozen=True)
class SegmentPath(CameraPath):
    """Segments in time order; still between and around them."""

    segments: tuple[Segment, ...]

    def shot(self, t: float) -> Shot:
        segments = self.segments
        if t <= segments[0].t0:
            return segments[0].start
        for segment in segments:
            if t <= segment.t1:
                return segment.shot(t) if t >= segment.t0 else segment.start
        return segments[-1].end

    @property
    def max_zoom(self) -> float:
        return max(max(s.start.zoom, s.end.zoom) for s in self.segments)


@dataclass(frozen=True)
class StillPath(CameraPath):
    still: Shot

    def shot(self, t: float) -> Shot:
        return self.still

    @property
    def max_zoom(self) -> float:
        return self.still.zoom


@dataclass(frozen=True)
class SampledPath(CameraPath):
    """Shots sampled over time, interpolated (centers linearly, zoom geometrically)."""

    times: np.ndarray  # (S,) increasing
    centers: np.ndarray  # (S, 2)
    zooms: np.ndarray  # (S,)

    def shot(self, t: float) -> Shot:
        x = float(np.interp(t, self.times, self.centers[:, 0]))
        y = float(np.interp(t, self.times, self.centers[:, 1]))
        zoom = float(np.exp(np.interp(t, self.times, np.log(self.zooms))))
        return Shot((x, y), zoom)

    @property
    def max_zoom(self) -> float:
        return float(self.zooms.max())


def keep_on_mosaic(scene: MosaicScene, base: WorldRect, point, zoom: float) -> tuple[float, float]:
    """A frame center near point (table units) that keeps a frame at this zoom on the
    mosaic where it can (centered on any axis where the frame is the wider)."""
    x0, y0, x1, y1 = content_rect(scene)
    w, h = base[2] / zoom, base[3] / zoom
    center = []
    for lo, hi, size, c in ((x0, x1, w, point[0]), (y0, y1, h, point[1])):
        if size >= hi - lo:
            c = (lo + hi) / 2
        else:
            c = min(max(c, lo + size / 2), hi - size / 2)
        center.append(float(c))
    return center[0], center[1]


def spot(scene: MosaicScene, fractions) -> tuple[float, float]:
    """A point of the mosaic given as fractions across and down."""
    x0, y0, x1, y1 = content_rect(scene)
    return x0 + fractions[0] * (x1 - x0), y0 + fractions[1] * (y1 - y0)


def close_up(scene: MosaicScene, base: WorldRect, focus, zoom: float) -> tuple[float, float]:
    """Where a close-up at this zoom centers: on the focus (fractions of the mosaic),
    moved just enough to keep the frame on the mosaic where it can."""
    return keep_on_mosaic(scene, base, spot(scene, focus), zoom)


def zoom_path(scene, base: WorldRect, duration: float, focus, zoom: float, timing, ease: str,
              closing: bool) -> CameraPath:  # fmt: skip
    """From the close-up to the full view (closing False: pull back) or the reverse."""
    home = home_shot(base)
    if zoom <= 1.0 + 1e-9:
        return StillPath(home)
    near = Shot(close_up(scene, base, focus, zoom), zoom)
    t0, t1 = (duration * share / 100.0 for share in timing)
    start, end = (home, near) if closing else (near, home)
    return SegmentPath((Segment(t0, t1, start, end, ease),))


class CameraMove(Configurable, ABC):
    """A configurable way to move the camera; plans a CameraPath."""

    id: ClassVar[str] = ""
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    @abstractmethod
    def path(self, scene: MosaicScene, timeline: Timeline, base: WorldRect) -> CameraPath:
        """The move over this scene and timeline; base: the video's framing (zoom 1)."""


_registry: dict[str, type[CameraMove]] = {}


def register_camera_move(cls: type[CameraMove]) -> type[CameraMove]:
    """Class decorator making a camera move available to the UI."""
    if not cls.id or not cls.name:
        raise TypeError(f"{cls.__name__} must define id and name")
    existing = _registry.get(cls.id)
    if existing is not None and existing is not cls:
        raise ValueError(f"camera move id {cls.id!r} is already used by {existing.__name__}")
    _registry[cls.id] = cls
    return cls


def camera_move_types() -> list[type[CameraMove]]:
    """Registered camera moves: Static first, the rest by name."""
    return sorted(_registry.values(), key=lambda cls: (cls.id != "static", cls.name))


def get_camera_move(type_id: str) -> type[CameraMove]:
    try:
        return _registry[type_id]
    except KeyError:
        raise KeyError(f"unknown camera move {type_id!r}") from None


# Shared settings


def _focus(axis: str) -> FloatParam:
    across = axis == "x"
    return FloatParam(
        50.0, "Focus across" if across else "Focus down", min=0.0, max=100.0, step=5.0,
        decimals=0, suffix="%",
        help=f"Where the close-up is, from the mosaic's {'left' if across else 'top'} edge "
             f"(0%) to its {'right' if across else 'bottom'} (100%).",
    )  # fmt: skip


def _zoom(default: float = 4.0, label: str = "Close-up", help: str = "") -> FloatParam:
    return FloatParam(
        default, label, min=1.0, max=20.0, step=0.5, decimals=1, suffix="×",
        help=help or "How close the close-up is, compared with the video's framing.",
    )  # fmt: skip


def _timing(default) -> RangeParam:
    return RangeParam(
        default, "Moves during", min=0.0, max=100.0, step=5.0, decimals=0, suffix="%",
        help="When the camera moves, as shares of the animation; it holds still before "
             "and after.",
    )  # fmt: skip


def _ease() -> ChoiceParam:
    return ChoiceParam("smooth", "Easing", choices=EASE_CHOICES,
                       help="How the move speeds up and slows down.")  # fmt: skip


def _span(duration: float, timing) -> tuple[float, float]:
    return duration * timing[0] / 100.0, duration * timing[1] / 100.0


# Built-in moves


@register_camera_move
class StaticMove(CameraMove):
    id = "static"
    name = "Static"
    description = "The camera holds still on the video's framing."

    def path(self, scene, timeline, base) -> CameraPath:
        return StillPath(home_shot(base))


@register_camera_move
class PullBackMove(CameraMove):
    id = "pull_back"
    name = "Pull back"
    description = "Starts close on a spot and pulls back to show the whole mosaic."

    focus_x = _focus("x")
    focus_y = _focus("y")
    zoom = _zoom()
    timing = _timing((0.0, 80.0))
    ease = _ease()

    def path(self, scene, timeline, base) -> CameraPath:
        return zoom_path(scene, base, timeline.duration, (self.focus_x / 100, self.focus_y / 100),
                         self.zoom, self.timing, self.ease, closing=False)  # fmt: skip


@register_camera_move
class PushInMove(CameraMove):
    id = "push_in"
    name = "Push in"
    description = "Starts on the whole mosaic and moves in close on a spot."

    focus_x = _focus("x")
    focus_y = _focus("y")
    zoom = _zoom()
    timing = _timing((20.0, 100.0))
    ease = _ease()

    def path(self, scene, timeline, base) -> CameraPath:
        return zoom_path(scene, base, timeline.duration, (self.focus_x / 100, self.focus_y / 100),
                         self.zoom, self.timing, self.ease, closing=True)  # fmt: skip


@register_camera_move
class PanMove(CameraMove):
    id = "pan"
    name = "Pan"
    description = (
        "Glides across the mosaic close up, from one side or corner to another, then "
        "pulls back to show it all (or stays close)."
    )

    start = ChoiceParam("left", "From", choices=PLACE_CHOICES,
                        help="Where the pan starts, close up.")  # fmt: skip
    finish = ChoiceParam("right", "To", choices=PLACE_CHOICES,
                         help="Where the pan ends.")  # fmt: skip
    zoom = _zoom(2.5, help="How close the camera pans, compared with the video's framing.")
    timing = _timing((0.0, 85.0))
    ending = ChoiceParam(
        "whole", "Then", choices=(("whole", "Pull back to the whole mosaic"),
                                  ("stay", "Stay close")),
        help="Whole mosaic: the last quarter of the move pulls back to show everything.",
    )  # fmt: skip
    ease = _ease()

    def path(self, scene, timeline, base) -> CameraPath:
        t0, t1 = _span(timeline.duration, self.timing)
        zoom = self.zoom
        a = Shot(keep_on_mosaic(scene, base, spot(scene, PLACES[self.start]), zoom), zoom)
        b = Shot(keep_on_mosaic(scene, base, spot(scene, PLACES[self.finish]), zoom), zoom)
        if self.ending == "stay":
            return SegmentPath((Segment(t0, t1, a, b, self.ease),))
        mid = t0 + 0.75 * (t1 - t0)
        return SegmentPath((Segment(t0, mid, a, b, self.ease),
                            Segment(mid, t1, b, home_shot(base), self.ease)))  # fmt: skip


@register_camera_move
class RotateMove(CameraMove):
    id = "rotate"
    name = "Rotate"
    description = (
        "Starts turned (and, if you like, closer) and turns upright onto the whole mosaic."
    )

    angle = FloatParam(
        30.0, "Turn", min=-180.0, max=180.0, step=5.0, decimals=0, suffix="°",
        help="How far the picture turns on its way upright: positive turns clockwise, "
             "negative counterclockwise.",
    )  # fmt: skip
    zoom = _zoom(1.5, "Start", help="How close the camera starts (1: the video's framing).")
    timing = _timing((0.0, 80.0))
    ease = _ease()

    def path(self, scene, timeline, base) -> CameraPath:
        t0, t1 = _span(timeline.duration, self.timing)
        home = home_shot(base)
        # The frame turns the other way to the picture.
        start = Shot(home.center, self.zoom, -math.radians(self.angle))
        if start == home:
            return StillPath(home)
        return SegmentPath((Segment(t0, t1, start, home, self.ease),))


FOLLOW_SAMPLES = 96  # moments the timeline is looked at to find the action


def _nearest(found: np.ndarray) -> np.ndarray:
    """For each index, the nearest index where found is True (found has one at least)."""
    have = np.flatnonzero(found)
    index = np.arange(len(found))
    right = np.searchsorted(have, index).clip(0, len(have) - 1)
    left = (right - 1).clip(0, len(have) - 1)
    closer = np.abs(have[left] - index) <= np.abs(have[right] - index)
    return np.where(closer, have[left], have[right])


@register_camera_move
class FollowMove(CameraMove):
    id = "follow"
    name = "Follow"
    description = (
        "Watches where the mosaic is being built: the camera keeps the landing spots of "
        "the tiles in flight in view, then pulls back to the whole mosaic at the end."
    )

    closest = _zoom(3.0, "Closest", help="The closest the camera gets, compared with the "
                                        "video's framing.")  # fmt: skip
    room = FloatParam(
        1.5, "Room", min=1.0, max=4.0, step=0.25, decimals=2, suffix="×",
        help="How much room the frame leaves around the action (1: just enough).",
    )  # fmt: skip
    smoothness = FloatParam(
        10.0, "Smoothness", min=0.0, max=50.0, step=2.5, decimals=1, suffix="%",
        help="How calmly the camera follows: it averages the action over this share of "
             "the animation.",
    )  # fmt: skip
    ending = FloatParam(
        20.0, "Pull back over", min=0.0, max=100.0, step=5.0, decimals=0, suffix="%",
        help="The last share of the animation, over which the camera eases back to the "
             "whole mosaic.",
    )  # fmt: skip

    def path(self, scene, timeline, base) -> CameraPath:
        duration = timeline.duration
        home = home_shot(base)
        if duration <= 0 or not len(scene):
            return StillPath(home)
        times = np.linspace(0.0, duration, FOLLOW_SAMPLES)
        spots = np.asarray(scene.center, dtype=np.float64)
        reach = 0.5 * np.asarray(scene.size, dtype=np.float64).max(axis=1)
        centers = np.full((len(times), 2), np.nan)
        zooms = np.full(len(times), np.nan)
        for i, t in enumerate(times):
            frame = timeline.frame(float(t))
            moving = frame.alpha > 0
            if frame.rest is not None:
                moving &= ~frame.rest
            else:
                moving &= np.abs(frame.center - spots).sum(axis=1) > 1e-6
            if not moving.any():
                continue
            where = spots[moving]
            lo = np.percentile(where, 10, axis=0) - reach[moving].mean()
            hi = np.percentile(where, 90, axis=0) + reach[moving].mean()
            centers[i] = (lo + hi) / 2
            need = np.maximum(hi - lo, 1e-9) * self.room
            zooms[i] = min(self.closest, base[2] / need[0], base[3] / need[1])
        found = ~np.isnan(zooms)
        if not found.any():
            return StillPath(home)
        # Moments without action hold the shot of the nearest moment with some.
        nearest = _nearest(found)
        centers, zooms = centers[nearest], np.maximum(zooms[nearest], 1.0)
        sigma = self.smoothness / 100.0 * (len(times) - 1)
        log_zoom = np.log(zooms)
        if sigma > 0:
            centers = gaussian_filter1d(centers, sigma, axis=0, mode="nearest")
            log_zoom = gaussian_filter1d(log_zoom, sigma, mode="nearest")
        zooms = np.exp(log_zoom)
        # Ease back to the whole mosaic over the last share.
        start = duration * (1 - self.ending / 100.0)
        u = np.clip((times - start) / max(duration - start, 1e-9), 0.0, 1.0)
        u = easing.ease_in_out_cubic(u) if self.ending > 0 else (times >= duration) * 1.0
        zooms = np.exp(np.log(zooms) * (1 - u))
        centers = centers + (np.asarray(home.center) - centers) * u[:, None]
        centers = np.array([keep_on_mosaic(scene, base, c, z)
                            for c, z in zip(centers, zooms, strict=True)])  # fmt: skip
        return SampledPath(times, centers, zooms)


__all__ = [
    "CameraMove",
    "CameraPath",
    "SampledPath",
    "Segment",
    "SegmentPath",
    "Shot",
    "StillPath",
    "camera_move_types",
    "get_camera_move",
    "home_shot",
    "register_camera_move",
]

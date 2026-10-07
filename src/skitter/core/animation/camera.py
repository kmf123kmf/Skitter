"""Camera moves: how the video's frame travels over the table as the mosaic builds.

A camera move is independent of the choreography: both are functions of the
same animation time t (0 .. the timeline's duration), one telling where the
tiles are, the other what the frame shows. The renderer combines them.

A move animates the framing, not the table camera of look.py: that camera
stays above the middle of the mosaic, so perspective, shadows and the
choreographies' rules about it (tiles kept off its axis near it) hold. A
move is a long lens panning and zooming over that picture. A `Shot` is the
point of the table at the middle of the frame and a zoom relative to the
video's framing (zoom 1 shows exactly the video_rect framing, as Static does).

Moves plan a `CameraPath` for a scene, its timeline and that framing;
`path.shot(t)` is the shot at time t. Timing settings are shares of the
animation, so a move stays in step when the duration changes. Zoom moves
are pure zooms about one fixed point of the table (nothing slides across
the frame while it zooms), with the zoom changing geometrically (each step
feels as large as the last).

Add a move by subclassing CameraMove (declare Params, plan a path) and
decorating it with @register_camera_move.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

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


@dataclass(frozen=True)
class Shot:
    """What the frame shows at a moment: centered on a table point, at a zoom."""

    center: tuple[float, float]  # the table point at the middle of the frame
    zoom: float = 1.0  # 1: the video's framing; 2: twice as close

    def view(self, base: WorldRect) -> WorldRect:
        """The world rect shown, given the video's framing (the view at zoom 1)."""
        w, h = base[2] / self.zoom, base[3] / self.zoom
        return (self.center[0] - w / 2, self.center[1] - h / 2, w, h)


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
class StillPath(CameraPath):
    still: Shot

    def shot(self, t: float) -> Shot:
        return self.still

    @property
    def max_zoom(self) -> float:
        return self.still.zoom


@dataclass(frozen=True)
class ZoomPath(CameraPath):
    """A pure zoom about a fixed table point: zoom z0 at time t0 to z1 at t1, eased.

    The frame's center at zoom z is anchor + (home - anchor) / z, where home is
    the center at zoom 1, so every table point moves straight toward or away
    from the anchor and the anchor itself stays put in the frame.
    """

    anchor: tuple[float, float]
    home: tuple[float, float]
    z0: float
    z1: float
    t0: float
    t1: float
    ease: str = "smooth"

    def shot(self, t: float) -> Shot:
        span = self.t1 - self.t0
        u = 1.0 if span <= 0 else float(np.clip((t - self.t0) / span, 0.0, 1.0))
        u = float(EASES[self.ease](u))
        zoom = self.z0 * (self.z1 / self.z0) ** u
        anchor, home = np.asarray(self.anchor), np.asarray(self.home)
        center = anchor + (home - anchor) / zoom
        return Shot((float(center[0]), float(center[1])), zoom)

    @property
    def max_zoom(self) -> float:
        return max(self.z0, self.z1)


def close_up(scene: MosaicScene, base: WorldRect, focus, zoom: float) -> tuple[float, float]:
    """Where a close-up at this zoom centers: on the focus (fractions of the mosaic),
    moved just enough to keep the frame on the mosaic where it can."""
    x0, y0, x1, y1 = content_rect(scene)
    w, h = base[2] / zoom, base[3] / zoom
    center = []
    for lo, hi, size, f in ((x0, x1, w, focus[0]), (y0, y1, h, focus[1])):
        c = lo + f * (hi - lo)
        if size >= hi - lo:
            c = (lo + hi) / 2
        else:
            c = min(max(c, lo + size / 2), hi - size / 2)
        center.append(c)
    return center[0], center[1]


def zoom_path(scene, base: WorldRect, duration: float, focus, zoom: float, timing, ease: str,
              closing: bool) -> CameraPath:  # fmt: skip
    """From the close-up to the full view (closing False: pull back) or the reverse."""
    home = (base[0] + base[2] / 2, base[1] + base[3] / 2)
    if zoom <= 1.0 + 1e-9:
        return StillPath(Shot(home))
    near = np.asarray(close_up(scene, base, focus, zoom))
    # The fixed point whose pure zoom from home (zoom 1) reaches `near` at `zoom`.
    anchor = (zoom * near - np.asarray(home)) / (zoom - 1.0)
    t0, t1 = (duration * share / 100.0 for share in timing)
    z0, z1 = (1.0, zoom) if closing else (zoom, 1.0)
    return ZoomPath((float(anchor[0]), float(anchor[1])), home, z0, z1, t0, t1, ease)


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


# Built-in moves


@register_camera_move
class StaticMove(CameraMove):
    id = "static"
    name = "Static"
    description = "The camera holds still on the video's framing."

    def path(self, scene, timeline, base) -> CameraPath:
        return StillPath(Shot((base[0] + base[2] / 2, base[1] + base[3] / 2)))


def _focus(axis: str) -> FloatParam:
    across = axis == "x"
    return FloatParam(
        50.0, "Focus across" if across else "Focus down", min=0.0, max=100.0, step=5.0,
        decimals=0, suffix="%",
        help=f"Where the close-up is, from the mosaic's {'left' if across else 'top'} edge "
             f"(0%) to its {'right' if across else 'bottom'} (100%).",
    )  # fmt: skip


def _zoom() -> FloatParam:
    return FloatParam(
        4.0, "Close-up", min=1.0, max=20.0, step=0.5, decimals=1, suffix="×",
        help="How close the close-up is, compared with the video's framing.",
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


__all__ = [
    "CameraMove",
    "CameraPath",
    "Shot",
    "StillPath",
    "ZoomPath",
    "camera_move_types",
    "get_camera_move",
    "register_camera_move",
]

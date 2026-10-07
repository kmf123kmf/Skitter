"""Camera shots: what the video's frame shows of the table at a moment.

The camera animates the framing, not the table camera of look.py: that
camera stays above the middle of the mosaic, so perspective, shadows and the
choreographies' rules about it (tiles kept off its axis near it) hold. The
video camera is a long lens panning, zooming and turning over that picture.
A `Shot` is the table point at the middle of the frame, a zoom relative to
the video's framing (zoom 1 shows exactly the view_rect framing) and a turn
(radians, clockwise: the picture looks turned the other way).

A `CameraPath` gives the shot at any time on the video's clock: the
keyframes of keyframes.py make one, and camera_moves.py writes keys for
ready-made moves. `keep_on_mosaic` places a shot on a scene.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from skitter.core.animation.video import content_rect
from skitter.core.scene import MosaicScene

WorldRect = tuple[float, float, float, float]  # x, y, width, height (mosaic units)


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
    """The camera over time: the shot at any moment."""

    @abstractmethod
    def shot(self, t: float) -> Shot:
        """The shot at time t."""

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

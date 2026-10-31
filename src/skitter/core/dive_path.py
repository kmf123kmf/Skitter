"""The camera's course through one mosaic of a continuous dive (Infinite Mosaic).

Continuous auto-dive never stops: the view zooms at a constant rate (constant
in log zoom, so it looks equally fast at every scale) from the whole mosaic
onto a tile picked in advance, and when that tile fills the view the next
mosaic (that tile's photo) takes over and the zoom goes on. `FlowPath` is one
mosaic's leg, from where the view starts (`View`) to the tile filling it:

- Zoom: geometric from the start's to the tile's (`end_zoom`), at `rate`
  (natural log of zoom per second), so a leg lasts log(end / start) / rate.
- Pan: the tile's offset on screen shrinks to nothing along an eased curve
  (Straight in: over the whole leg; Center first: over its first part), so
  panning starts and ends at rest and nothing jolts where legs meet.
- Bounds: the view never shows beyond what it started with (the mosaic, at
  a leg's start): the pan waits where centering the tile early would (see
  `_max_pan`).
- Turn: the view turns with the tile (photos in a pile lie at angles), so it
  ends upright on screen, as the next mosaic starts.

The next leg starts where this one ends, in the next mosaic's own world:
the tile's view there is the whole mosaic, upright (`View`, rotation 0).
"""

import math
from dataclasses import dataclass

import numpy as np

STRAIGHT, CENTER_FIRST = "straight", "center"
CENTER_SHARE = 0.35  # Center first: the share of a leg the tile takes to reach the middle
MIN_SECONDS = 0.5  # the shortest leg (a view already about as close as the tile)


@dataclass(frozen=True)
class View:
    """Where the camera looks: world center, zoom (screen px per unit), rotation."""

    center: tuple[float, float]
    zoom: float
    rotation: float = 0.0


def _smooth(u):
    """Smoothstep: 0 to 1 with zero slope at both ends."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _angle_between(a: float, b: float) -> float:
    """b - a, the short way round."""
    return (b - a + math.pi) % (2 * math.pi) - math.pi


class FlowPath:
    """One leg: from start onto the tile at target (world center) with this rotation,
    until the view (viewport, screen px) frames it at end_zoom."""

    def __init__(self, start: View, target, rotation: float, end_zoom: float, viewport,
                 rate: float, path: str = STRAIGHT):  # fmt: skip
        self.start = start
        self.c0 = np.asarray(start.center, np.float64)
        self.target = np.asarray(target, np.float64)
        self.z0, self.z1 = float(start.zoom), float(end_zoom)
        self.r0 = float(start.rotation)
        self.turn = _angle_between(self.r0, float(rotation))
        self.path = path
        self.span = math.log(self.z1 / self.z0)  # log zoom it covers
        self.seconds = max(self.span / rate, MIN_SECONDS) if rate > 0 else MIN_SECONDS
        # How far the tile is from the middle, as a share of the start view's half-size
        # (the farther axis); below 1 - k1 so the pan can always finish.
        half = np.asarray(viewport, np.float64) / (2 * self.z0)
        offset = np.abs(self.target - self.c0) / half
        self.k1 = self.z0 / self.z1  # the end view's size relative to the start's
        self.reach = float(min(offset.max(), max(1 - self.k1, 0.0))) if self.k1 < 1 else 0.0

    def at(self, u: float) -> View:
        """The view at u (0 to 1) of the leg."""
        u = float(np.clip(u, 0.0, 1.0))
        k = (self.z0 / self.z1) ** u  # the view's size relative to the start's
        pan = min(self._wanted_pan(u), self._max_pan(k))
        along = 1 - (1 - pan) * k  # share of the way from the start's center to the tile
        center = self.c0 + (self.target - self.c0) * along
        rotation = self.r0 + self.turn * float(_smooth(u))
        return View((float(center[0]), float(center[1])), self.z0 / k, rotation)

    def _wanted_pan(self, u: float) -> float:
        """How much of the tile's offset on screen is gone at u (1: in the middle)."""
        if self.path == CENTER_FIRST:
            return float(_smooth(u / CENTER_SHARE))
        return float(_smooth(u))

    def _max_pan(self, k: float) -> float:
        """The most of it that can be gone with the view (k the start's size) still
        inside the start view: its center may be at most (1 - k) half-sizes off."""
        if self.reach <= 0:
            return 1.0
        allowed = (1 - k) / self.reach  # share of the way to the tile the center may go
        return float(np.clip(1 - (1 - allowed) / k, 0.0, 1.0))

    def u_at_share(self, share: float) -> float:
        """Where along the leg the tile fills this share of the view (0 if it already
        does at the start)."""
        if self.z0 >= self.z1:
            return 0.0
        return float(np.clip(1 - math.log(share) / math.log(self.z0 / self.z1), 0.0, 1.0))

    def tile_share(self, u: float) -> float:
        """How much of the view the tile fills at u (its zoom over the end's): 1 at the end."""
        u = float(np.clip(u, 0.0, 1.0))
        return (self.z0 / self.z1) ** (1 - u)


def flow_rate(columns: int, seconds: float) -> float:
    """The zoom rate (log zoom per second) that takes about seconds from a whole mosaic of
    this many columns onto one of its base tiles."""
    return math.log(max(columns, 2)) / max(seconds, 1e-3)

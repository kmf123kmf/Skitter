"""The camera's course through one mosaic of a continuous dive (Infinite Mosaic).

Continuous auto-dive never stops: the view zooms at a constant rate (constant
in log zoom, so it looks equally fast at every scale) from the whole mosaic
onto a tile picked in advance, and when that tile fills the view the next
mosaic (that tile's photo) takes over and the zoom goes on. `FlowPath` is one
mosaic's leg, from where the view starts (`View`) to the tile filling it:

- Zoom: geometric from the start's to the tile's (`end_zoom`), at `rate`
  (natural log of zoom per second), so the zoom takes log(end / start) / rate.
- Straight in: the zoom is even (it goes on at the same speed from leg to
  leg), and the tile's offset on screen shrinks to nothing along an eased
  curve over the whole leg. That curve never shows beyond what it started with
  (the mosaic, at a leg's start): the pan waits where centering the tile
  early would (see `_max_pan`), easing into that limit (`_soft_min`).
  A tile off the middle keeps its place on screen at first, so the view
  itself sets off at once, at the tile's offset per unit of log zoom. Left to
  that, the view would stop dead on each tile and leap off to the next: once
  the next tile is known, `leave` has the leg end moving at that speed, still
  landing exactly on its tile (it passes through, coming in from the far
  side, past the mosaic's edge if need be), so the view's motion carries
  through where legs meet.
- Center first: the view pans, not zooming, until the tile is in the middle
  (beyond the mosaic's edge if need be), then zooms straight in, easing in
  and out (the zoom rests where legs meet). The pan takes CENTER_SHARE of the
  zoom's time per half view the tile is off (`pan_share`).
- Turn: the view turns with the tile (photos in a pile lie at angles), so it
  ends upright on screen, as the next mosaic starts.

The next leg starts where this one ends, in the next mosaic's own world:
the tile's view there is the whole mosaic, upright (`View`, rotation 0).
"""

import math
from dataclasses import dataclass

import numpy as np

STRAIGHT, CENTER_FIRST = "straight", "center"
CENTER_SHARE = 0.35  # Center first: a pan of half the view takes this share of the zoom's time
MIN_SECONDS = 0.5  # the shortest leg (a view already about as close as the tile)
EDGE_ROOM = 0.995  # of a tile on the edge's way: released just before the end, to land at rest
CARRY_FROM = 0.5  # `leave`: where along a leg it starts bending the view toward the next tile
CARRY_LATEST = 0.9  # too late to bend it past here: the leg ends at rest


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


def _unsmooth(y: float) -> float:
    """The u in [0, 1] where _smooth(u) is y (in [0, 1])."""
    return 0.5 - math.sin(math.asin(1 - 2 * y) / 3)


def _soft_min(a: float, b: float, width: float) -> float:
    """min(a, b), rounded off where they are within width of each other: no kink."""
    if width <= 0:
        return min(a, b)
    h = max(width - abs(a - b), 0.0) / width
    return min(a, b) - h * h * width / 4


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
        self.heading: np.ndarray | None = None  # see `leave`
        self.carry_from = 1.0
        # How far the tile is from the middle, as a share of the start view's half-size
        # (the farther axis); well below 1 - k1 so the pan is done waiting in time to come
        # to rest on a tile on the edge, not arrive still catching up.
        half = self.half0 = np.asarray(viewport, np.float64) / (2 * self.z0)
        offset = np.abs(self.target - self.c0) / half
        self.k1 = self.z0 / self.z1  # the end view's size relative to the start's
        room = EDGE_ROOM * max(1 - self.k1, 0.0)
        self.reach = float(min(offset.max(), room)) if self.k1 < 1 else 0.0
        self.pan_share = 0.0  # Center first: the share of the leg that pans (not zooming)
        if path == CENTER_FIRST:
            pan = CENTER_SHARE * float(offset.max())  # the pan's time over the zoom's
            self.pan_share = pan / (1 + pan)
        self.seconds = self.duration(rate)

    def duration(self, rate: float) -> float:
        """How long the leg takes at this zoom rate (Time per mosaic may change mid-leg)."""
        zoom = max(self.span / rate, MIN_SECONDS) if rate > 0 else MIN_SECONDS
        return zoom / (1 - self.pan_share)

    def at(self, u: float) -> View:
        """The view at u (0 to 1) of the leg."""
        u = float(np.clip(u, 0.0, 1.0))
        zoomed = self._zoomed(u)
        k = (self.z0 / self.z1) ** zoomed  # the view's size relative to the start's
        if self.path == CENTER_FIRST:  # the pan, all before the zoom
            along = float(_smooth(u / self.pan_share)) if self.pan_share > 0 else 1.0
        else:  # the share of the tile's offset on screen gone, the view kept in bounds
            limit = self._max_pan(k)  # (its rounding gone by the ends: they stay exact)
            width = 2 * u * (1 - u) ** 2 if limit < 1 else 0.0
            pan = max(_soft_min(float(_smooth(u)), limit, width), 0.0)
            along = 1 - (1 - pan) * k  # share of the way from the start's center to the tile
        center = self.c0 + (self.target - self.c0) * along + self._carried(u, k)
        rotation = self.r0 + self.turn * float(_smooth(zoomed))
        return View((float(center[0]), float(center[1])), self.z0 / k, rotation)

    def leave(self, heading, from_u: float = 0.0) -> None:
        """End the leg moving on: heading is the next leg's tile's offset from the middle
        as this leg's tile fills the view (screen px, that tile upright), which is how far
        the view moves per unit of log zoom as the next leg sets off. from_u: where the leg
        has got to (what is behind it stays as it was). Straight in only (Center first
        rests where legs meet); past CARRY_LATEST it ends at rest after all. Coming in from
        the far side of a tile on the mosaic's edge shows a little past it, for a while."""
        from_u = max(from_u, CARRY_FROM)
        if self.path != STRAIGHT or self.span <= 0 or from_u >= CARRY_LATEST:
            return
        self.heading, self.carry_from = np.asarray(heading, np.float64), from_u

    def _carried(self, u: float, k: float) -> np.ndarray:
        """How far `leave` has the view from the eased course at u (world units; k the
        view's size relative to the start's). A curve that starts flat and ends at 0 with
        slope -1, so the tile's offset goes through nothing at the speed asked for."""
        if self.heading is None or u <= self.carry_from:
            return np.zeros(2)
        window = 1 - self.carry_from
        e = (u - self.carry_from) / window
        x, y = self.heading * (self.span * window * e**3 * (1 - e))
        c, s = math.cos(self.r0 + self.turn), math.sin(self.r0 + self.turn)
        return -np.array([c * x - s * y, s * x + c * y]) * k / self.z0

    def _zoomed(self, u: float) -> float:
        """The share of the leg's log zoom done at u."""
        if self.path == CENTER_FIRST:
            return float(_smooth((u - self.pan_share) / (1 - self.pan_share)))
        return u

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
        zoomed = float(np.clip(1 - math.log(share) / math.log(self.z0 / self.z1), 0.0, 1.0))
        if self.path != CENTER_FIRST or zoomed <= 0:
            return zoomed
        return self.pan_share + (1 - self.pan_share) * _unsmooth(zoomed)

    def tile_share(self, u: float) -> float:
        """How much of the view the tile fills at u (its zoom over the end's): 1 at the end."""
        u = float(np.clip(u, 0.0, 1.0))
        return (self.z0 / self.z1) ** (1 - self._zoomed(u))


def flow_rate(columns: int, seconds: float) -> float:
    """The zoom rate (log zoom per second) that takes about seconds from a whole mosaic of
    this many columns onto one of its base tiles."""
    return math.log(max(columns, 2)) / max(seconds, 1e-3)

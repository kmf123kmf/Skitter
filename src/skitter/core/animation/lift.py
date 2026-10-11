"""Lifting tiles off the finished mosaic without breaking the pile (for shows).

A lifting show (a ripple, a wave) describes a surface rising under the
tiles: its height at five points of each tile (`Lift.points`: the center,
then the corners) and its slope at the center. `Lift.frame` turns that into
a frame that never shows what a pile of flat photos couldn't do:

- Only tiles the camera sees enough of move: those showing at least
  `still_below` of themselves in the finished mosaic. The rest stay on the
  table. A tile lying on a moving tile always moves too, so nothing still
  ever lies on something lifting.
- A rigid tile rests on the highest of its five points, and is never lower
  than a moving tile under it. Flat tiles can't tilt, so where the surface
  would put a tile below one it lies on, the two settle between their
  heights, weighted by how much of each the camera sees: the visible tile
  keeps close to the surface, the buried one gives way. (Carrying the upper
  tile all the way up instead passes heights along chains of overlapping
  tiles, far from the wave: on a 2,655-tile pile it put 30% of the visible
  area more than a quarter of the peak off the surface; settling, 8%.)
- Tiles draw from low to high, then in stacking order. For flat tiles seen
  from above that is exactly what covers what, and it never contradicts the
  stacking order, since no tile is lower than one under it. Order changes
  never pop: neighbors swap only at equal heights, where the camera scales
  them alike. Tiles at height 0 are at rest on the table.
- Slope shading (optional) tints each lifted tile as if it tilted with the
  surface under it, like a shaded-relief map: lighter where the surface
  faces the look's light, darker where it faces away, alike on both sides.
  (True lighting relative to a flat tile mostly darkens, the light being
  high, so a ring read as a dark band rather than a bump.) Tiles stay flat;
  only their shade follows the slope.
"""

import math

import numpy as np
from numba import njit

from skitter.core.animation.base import TileFrame
from skitter.core.animation.look import AnimationLook
from skitter.core.scene import MosaicScene

SHADE_GAIN = 0.4  # shading tint per unit of slope toward (or away from) the light
SHADE_LIGHT = 0.7  # lightening relative to darkening (blending toward white washes photos out)
SHADE_MAX = 0.5  # strongest shading tint
SETTLE_SWEEPS = 50  # settling passes over the overlapping pairs (then carried, to be exact)
BURIED_WEIGHT = 0.02  # a hidden tile's say in settling (a visible tile has its share, up to 1)


@njit(cache=True, nogil=True)
def _carry(height, lower, upper):
    """In place: no tile lower than a tile under it. Pairs sorted by lower tile, so
    each lower tile is final before it carries the tiles lying on it."""
    for k in range(len(lower)):
        if height[lower[k]] > height[upper[k]]:
            height[upper[k]] = height[lower[k]]


@njit(cache=True, nogil=True)
def _settle(height, weight, lower, upper, sweeps):
    """In place: each tile above one it lies on and the tile under it meet at their
    weighted mean, pass after pass (pairs sorted by lower tile), until none is."""
    for _ in range(sweeps):
        changed = False
        for k in range(len(lower)):
            a, b = lower[k], upper[k]
            if height[a] > height[b]:
                mean = (weight[a] * height[a] + weight[b] * height[b]) / (weight[a] + weight[b])
                height[a] = mean
                height[b] = mean
                changed = True
        if not changed:
            return


def moving_tiles(scene: MosaicScene, still_below: float) -> np.ndarray:
    """(N,) bool: tiles showing at least `still_below` of themselves, and every tile
    lying on one of those (upward closed: nothing still lies on a moving tile)."""
    moving = (scene.visible_share >= still_below).astype(np.float64)
    lower, upper = _sorted_pairs(scene)
    _carry(moving, lower, upper)
    return moving > 0


def _sorted_pairs(scene: MosaicScene) -> tuple[np.ndarray, np.ndarray]:
    pairs = scene.overlaps
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return np.ascontiguousarray(pairs[:, 0]), np.ascontiguousarray(pairs[:, 1])


def slope_tint(slope: np.ndarray, fall: np.ndarray) -> np.ndarray:
    """(N, 4) tints shading tiles tilted by these surface slopes (N, 2) under a light
    whose shadows fall along `fall` (unit): white where the surface rises toward
    `fall` (it faces the light), black where it falls, none tilted across the light."""
    facing = slope @ fall  # > 0: the surface rises away from the light, facing it
    strength = np.minimum(SHADE_GAIN * np.abs(facing), SHADE_MAX)
    tint = np.zeros((len(slope), 4))
    lit = facing > 0
    tint[lit, :3] = 1.0
    tint[:, 3] = np.where(lit, SHADE_LIGHT * strength, strength)
    return tint


class Lift:
    """Frames of tiles lifted by a surface under them (see the module docstring).

    still_below: share of a tile (0..1) the finished mosaic must show for it to
    move. shading: slope shading on or off.
    """

    def __init__(self, scene: MosaicScene, still_below: float = 0.25,
                 look: AnimationLook | None = None, shading: bool = True):  # fmt: skip
        look = look or AnimationLook()
        self.final = TileFrame.final(scene)
        self.points = np.concatenate([scene.center[:, None], scene.corners], axis=1)  # (N, 5, 2)
        self.moving = moving_tiles(scene, still_below)
        self.lower, self.upper = _sorted_pairs(scene)
        self.weight = scene.visible_share + BURIED_WEIGHT
        angle = math.radians(look.light_direction)  # 0: light from the top, clockwise
        self.fall = -np.array([math.sin(angle), -math.cos(angle)])  # the way shadows fall
        self.shading = shading

    def frame(self, surface: np.ndarray, slope: np.ndarray | None = None) -> TileFrame:
        """The frame for surface heights (N, 5) at `points` and slopes (N, 2) at the
        centers (None: no shading). Exactly the finished mosaic when nothing is lifted."""
        height = np.where(self.moving, np.maximum(surface.max(axis=1), 0.0), 0.0)
        _settle(height, self.weight, self.lower, self.upper, SETTLE_SWEEPS)
        _carry(height, self.lower, self.upper)  # whatever settling left, exactly
        lifted = height > 0
        if not lifted.any():
            return self.final
        tint = self.final.tint
        if self.shading and slope is not None:
            tint = np.where(lifted[:, None], slope_tint(slope, self.fall), tint)
        return self.final.replace(
            tint=tint,
            order=np.lexsort((np.arange(len(height)), height)),
            height=height,
            rest=~lifted,
        )

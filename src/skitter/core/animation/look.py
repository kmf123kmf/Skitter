"""How animations are seen: background, a camera looking down at the table, light.

Tiles move over a table, the plane the mosaic lies in, and gravity pulls
them onto it. The camera looks straight down from `camera_height` above the
middle of the mosaic, so a tile at height h above the table looks larger by
H / (H - h) and further from the middle (perspective). A distant light casts
each airborne tile's shadow onto the table: offset away from the light by
SHADOW_SLOPE x h, blurred by `shadow_softness` x h, fainter the higher the
tile. Tiles at rest (height 0) cast no shadow and look exactly as in the
finished mosaic.

`TableCamera.project` turns a TileFrame into what the camera sees, in three
draw groups: tiles at rest on the table (stacking order), shadows, and
tiles in the air (above everything at rest).

Nothing at or above the camera is seen: tiles higher than NEAR x camera
height are hidden, and below it fully opaque. Choreographies keep tiles
that high off the picture (`clear_of_axis`): near the camera everything
projects far out, so a tile that isn't right under the camera's axis there
is out of view and slides in from the picture's edge instead of popping in.
"""

import math
from dataclasses import dataclass

import numpy as np

from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import ColorParam, Configurable, FloatParam

SHADOW_SLOPE = 0.6  # shadow offset per unit of height (light about 60° above the table)
FLIP_SHADE = 0.45  # how much darker a tile is edge-on than flat (flips)
BACK = (0.933, 0.910, 0.863)  # default back of a photo (flips)
NEAR = 0.97  # tiles higher than this share of the camera height are not seen (33x size)
VIEW = 1.5  # the widest view, in half diagonals of the mosaic (framing margins included)


class AnimationLook(Configurable):
    """How animations look in the Animate tab and in exported videos."""

    background = ColorParam(
        "#1e1e1e", "Background", allow_transparent=True,
        help="Behind the tiles. Transparent needs a video format that keeps it "
             "(WebM, WebP, ProRes, PNG sequence).",
    )  # fmt: skip
    camera_height = FloatParam(
        2.0, "Camera height", min=0.5, max=20.0, step=0.25, suffix=" × mosaic",
        help="How far above the table the camera looks down from. Lower makes tiles "
             "in the air look bigger (stronger perspective).",
    )  # fmt: skip
    light_direction = FloatParam(
        315.0, "Light from", min=0.0, max=360.0, step=15.0, decimals=0, suffix="°",
        help="Where the light comes from: 0° the top of the picture, clockwise. "
             "Shadows fall the opposite way.",
    )  # fmt: skip
    shadow_strength = FloatParam(
        0.5, "Shadows", min=0.0, max=1.0, step=0.05,
        help="Darkness of the shadows tiles in the air cast on the table (0: none).",
    )  # fmt: skip
    back_color = ColorParam(
        "#eee8dc", "Photo backs",
        help="The color of a tile's back, seen while it flips over in flight.",
    )  # fmt: skip
    shadow_softness = FloatParam(
        0.15, "Shadow softness", min=0.0, max=2.0, step=0.05,
        help="How blurred a shadow is, relative to the tile's height.",
    )  # fmt: skip


def flip_tint(frame, back: tuple[float, float, float]) -> np.ndarray:
    """Tints showing flipping tiles: the back color while back up, and darker as a
    tile turns edge-on to the light (shading that makes the turn read as 3D)."""
    if frame.facing is None:
        return frame.tint
    facing = frame.facing
    light = 1.0 - FLIP_SHADE * (1.0 - np.abs(facing))  # 1 flat, darker edge-on
    tint = frame.tint.copy()
    front = (facing >= 0) & (facing < 1.0 - 1e-9)
    tint[front, :3] = 0.0  # mix toward black by the shade
    tint[front, 3] = 1.0 - light[front]
    up = facing < 0  # back up
    tint[up, :3] = np.asarray(back)[None, :] * light[up][:, None]
    tint[up, 3] = 1.0  # the back shows its own color, shaded
    return tint


def clear_of_axis(scene: MosaicScene) -> np.ndarray:
    """(N,) how far from the camera's axis (horizontally) each tile's center must be
    while it is just below the near plane, for the whole tile to be off the picture."""
    x0, y0, x1, y1 = scene.bounds
    view = VIEW * math.hypot(x1 - x0, y1 - y0) / 2
    reach = np.hypot(scene.size[:, 0], scene.size[:, 1]) / 2  # rotated tile, any way
    return reach + view * (1.0 - NEAR)  # the view, seen at the near plane's size


def scene_extent(scene: MosaicScene) -> float:
    """The mosaic's size for heights and the camera: its longer side, overhang included."""
    x0, y0, x1, y1 = scene.bounds
    return max(x1 - x0, y1 - y0, 1e-9)


@dataclass(frozen=True)
class Projected:
    """What the camera sees of a frame; tile arrays are in scene order."""

    center: np.ndarray  # (N, 2) on screen (the table plane), perspective applied
    size: np.ndarray  # (N, 2)
    rotation: np.ndarray
    alpha: np.ndarray
    tint: np.ndarray
    ground: np.ndarray  # tile indices at rest, bottom to top
    air: np.ndarray  # tile indices in the air, bottom to top
    shadow: np.ndarray  # tile indices casting a shadow (in air order)
    shadow_center: np.ndarray  # (S, 2)
    shadow_size: np.ndarray  # (S, 2)
    shadow_rotation: np.ndarray  # (S,)
    shadow_alpha: np.ndarray  # (S,)
    shadow_blur: np.ndarray  # (S,) blur radius, mosaic units


@dataclass(frozen=True)
class TableCamera:
    point: np.ndarray  # (2,) where the camera's axis meets the table
    height: float  # camera height above the table, mosaic units
    light: np.ndarray  # (2,) unit vector: the way shadows fall
    strength: float
    softness: float
    back: tuple[float, float, float] = BACK  # color of a tile's back (flips)

    @classmethod
    def for_scene(cls, scene: MosaicScene, look: AnimationLook) -> "TableCamera":
        x0, y0, x1, y1 = scene.bounds
        angle = math.radians(look.light_direction)  # light from here: 0 = top, clockwise
        toward_light = np.array([math.sin(angle), -math.cos(angle)])
        return cls(
            point=np.array([(x0 + x1) / 2, (y0 + y1) / 2]),
            height=look.camera_height * scene_extent(scene),
            light=-toward_light,
            strength=look.shadow_strength,
            softness=look.shadow_softness,
            back=ColorParam.rgb(look.back_color),
        )

    def scale(self, height: np.ndarray) -> np.ndarray:
        """Apparent size factor of things at these heights."""
        h = np.minimum(np.maximum(height, 0.0), NEAR * self.height)
        return self.height / (self.height - h)

    def visibility(self, height: np.ndarray) -> np.ndarray:
        """1 below the near plane, 0 above it (no fading: tiles are never see-through)."""
        return (np.asarray(height, np.float64) <= NEAR * self.height).astype(np.float64)

    def project(self, frame) -> Projected:
        """A TileFrame as seen from the camera (see the module docstring)."""
        if not len(frame):
            return _empty()
        height = frame.heights()
        scale = self.scale(height)
        center = self.point + (frame.center - self.point) * scale[:, None]
        order = frame.draw_order()
        rest = frame.at_rest()
        ground, air = order[rest[order]], order[~rest[order]]
        casting = air[(height[air] > 0) & (frame.alpha[air] > 0)] if self.strength > 0 else air[:0]
        h = height[casting]
        fade = np.clip(1.0 - 0.6 * h / max(self.height, 1e-9), 0.0, 1.0)
        return Projected(
            center=center,
            size=frame.size * scale[:, None],
            rotation=frame.rotation,
            alpha=frame.alpha * self.visibility(height),
            tint=flip_tint(frame, self.back),
            ground=ground,
            air=air,
            shadow=casting,
            shadow_center=frame.center[casting] + self.light * (SHADOW_SLOPE * h)[:, None],
            shadow_size=frame.size[casting],
            shadow_rotation=frame.rotation[casting],
            shadow_alpha=self.strength * frame.alpha[casting] * fade,
            shadow_blur=self.softness * h,
        )


def _empty() -> Projected:
    z1, z2, i = np.zeros(0), np.zeros((0, 2)), np.zeros(0, np.int64)
    return Projected(z2, z2, z1, z1, np.zeros((0, 4)), i, i, i, z2, z2, z1, z1, z1)


def project_flat(frame) -> Projected:
    """A frame seen without a camera: heights ignored, no shadows."""
    if not len(frame):
        return _empty()
    order = frame.draw_order()
    rest = frame.at_rest()
    none = np.zeros(0, np.int64)
    return Projected(
        frame.center, frame.size, frame.rotation, frame.alpha, flip_tint(frame, BACK),
        order[rest[order]], order[~rest[order]], none,
        np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0),
    )  # fmt: skip

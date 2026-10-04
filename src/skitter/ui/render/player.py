"""Playing an animation timeline (core/animation) on a GPU canvas.

The player shows a scene's tiles as three sprite layers, drawn in this
order: tiles at rest on the table, the shadows of tiles in the air, and the
tiles in the air (see core/animation/look.py: a camera looks down at the
table, so tiles in the air look bigger and cast shadows). Any moment can be
shown with `seek(t)` (scrubbing), and `play()` advances time with the
canvas's frame clock. Textures can be swapped while playing (for example
when full-size tiles finish loading); the current moment is redrawn.
"""

import numpy as np
from PySide6.QtCore import QObject, Signal

from skitter.core.animation import TileFrame, Timeline
from skitter.core.animation.look import Projected, TableCamera, project_flat
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.render.sprites import SpriteLayer, make_instances

SHADOW_COLOR = (0.0, 0.0, 0.0, 1.0)


def frame_layers(
    base: np.ndarray, frame: TileFrame, camera: TableCamera | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sprite instances for a frame: (tiles at rest, shadows, tiles in the air).

    base holds each tile's texture choice and tint (scene order); the frame,
    seen through the camera, supplies geometry, alpha, tint and draw order.
    """
    seen = camera.project(frame) if camera is not None else project_flat(frame)
    return _tiles(base, seen, seen.ground), _shadows(seen), _tiles(base, seen, seen.air)


def _tiles(base: np.ndarray, seen: Projected, index: np.ndarray) -> np.ndarray:
    tiles = base[index]
    tiles["pos"] = seen.center[index]
    tiles["size"] = seen.size[index]
    tiles["rotation"] = seen.rotation[index]
    tiles["alpha"] = seen.alpha[index]
    tiles["tint"] = seen.tint[index]
    return tiles


def _shadows(seen: Projected) -> np.ndarray:
    shadows = make_instances(len(seen.shadow))
    shadows["pos"] = seen.shadow_center
    shadows["size"] = seen.shadow_size
    shadows["rotation"] = seen.shadow_rotation
    shadows["alpha"] = seen.shadow_alpha
    shadows["blur"] = np.maximum(seen.shadow_blur, 1e-3)
    shadows["tint"] = SHADOW_COLOR
    return shadows


class TimelinePlayer(QObject):
    time_changed = Signal(float)
    playing_changed = Signal(bool)

    def __init__(self, canvas: MosaicCanvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.timeline: Timeline | None = None
        self.camera: TableCamera | None = None
        self.layer: SpriteLayer | None = None  # tiles at rest
        self.shadow_layer: SpriteLayer | None = None
        self.air_layer: SpriteLayer | None = None  # tiles in the air (layer's textures)
        self._base: np.ndarray | None = None
        self.time = 0.0
        self._run: object | None = None  # identifies the running playback

    @property
    def duration(self) -> float:
        return self.timeline.duration if self.timeline is not None else 0.0

    @property
    def playing(self) -> bool:
        return self._run is not None

    def set_content(self, pages: np.ndarray | None, base: np.ndarray | None) -> None:
        """Textures and per-tile instance data (see TileTextures.instances); None clears.

        The tile layers draw first on the canvas, below any overlays added later.
        """
        index = 0
        if self.layer is not None:
            index = self.canvas.layers.index(self.layer)
            for layer in (self.air_layer, self.shadow_layer, self.layer):
                self.canvas.remove_layer(layer)
            self.layer = self.shadow_layer = self.air_layer = None
        self._base = base
        if base is not None:
            empty = base[:0].copy()
            self.layer = self.canvas.add_layer(SpriteLayer(pages, empty), index)
            self.shadow_layer = self.canvas.add_layer(SpriteLayer(None, make_instances(0)),
                                                      index + 1)  # fmt: skip
            self.air_layer = self.canvas.add_layer(
                SpriteLayer(None, empty.copy(), texture_from=self.layer), index + 2
            )
            self._draw()

    def set_camera(self, camera: TableCamera | None) -> None:
        """How tiles in the air are seen (perspective and shadows); None: flat."""
        self.camera = camera
        self._draw()

    def set_timeline(self, timeline: Timeline | None, time: float | None = None) -> None:
        """Show a new timeline at `time` (default: where the old one was, clamped)."""
        self.timeline = timeline
        self.seek(self.time if time is None else time)

    def seek(self, t: float) -> None:
        self.time = min(max(float(t), 0.0), self.duration)
        self._draw()
        self.time_changed.emit(self.time)

    def play(self) -> None:
        if self.timeline is None or self.playing:
            return
        if self.time >= self.duration:
            self.seek(0.0)  # draw the first frame now, not the finished one
        run = self._run = object()
        start = self.time

        def step(elapsed: float) -> bool:
            if run is not self._run:
                return False
            self.seek(start + elapsed)
            if self.time >= self.duration:
                self.pause()
                return False
            return True

        self.canvas.add_animation(step)
        self.playing_changed.emit(True)

    def pause(self) -> None:
        if self._run is not None:
            self._run = None
            self.playing_changed.emit(False)

    def _draw(self) -> None:
        if self.layer is None or self.timeline is None or self._base is None:
            return
        layers = frame_layers(self._base, self.timeline.frame(self.time), self.camera)
        for layer, instances in zip((self.layer, self.shadow_layer, self.air_layer), layers,
                                    strict=True):  # fmt: skip
            layer.instances = instances
            layer.mark_dirty()
        self.canvas.update()

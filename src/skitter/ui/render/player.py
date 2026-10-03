"""Playing an animation timeline (core/animation) on a GPU canvas.

The player owns one sprite layer showing a scene's tiles. Any moment can be
shown with `seek(t)` (scrubbing), and `play()` advances time with the
canvas's frame clock. Textures can be swapped while playing (for example
when full-size tiles finish loading); the current moment is redrawn.
"""

import numpy as np
from PySide6.QtCore import QObject, Signal

from skitter.core.animation import TileFrame, Timeline
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.render.sprites import SpriteLayer


def frame_instances(base: np.ndarray, frame: TileFrame) -> np.ndarray:
    """Sprite instances for a frame, in its draw order.

    base holds each tile's texture choice and tint offset (scene order);
    the frame supplies geometry, alpha, tint and draw order.
    """
    order = frame.draw_order()
    tiles = base[order]
    tiles["pos"] = frame.center[order]
    tiles["size"] = frame.size[order]
    tiles["rotation"] = frame.rotation[order]
    tiles["alpha"] = frame.alpha[order]
    tiles["tint"] = frame.tint[order]
    return tiles


class TimelinePlayer(QObject):
    time_changed = Signal(float)
    playing_changed = Signal(bool)

    def __init__(self, canvas: MosaicCanvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.timeline: Timeline | None = None
        self.layer: SpriteLayer | None = None
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

        The tile layer draws first on the canvas, below any overlays added later.
        """
        index = 0
        if self.layer is not None:
            index = self.canvas.layers.index(self.layer)
            self.canvas.remove_layer(self.layer)
            self.layer = None
        self._base = base
        if base is not None:
            self.layer = self.canvas.add_layer(SpriteLayer(pages, base.copy()), index)
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
        self.layer.instances = frame_instances(self._base, self.timeline.frame(self.time))
        self.layer.mark_dirty()
        self.canvas.update()

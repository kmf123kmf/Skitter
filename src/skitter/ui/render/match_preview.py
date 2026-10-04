"""A matching run in progress, drawn: the matcher's previews (MatchPreview,
core/matching/matcher.py) turned into sprite layers off the UI thread.

A frame shows the tiles chosen so far, from library thumbnails, over flat
squares of the target color in regions still without one. Packing the
thumbnails takes a while for big mosaics, so frames are built in the
background, newest first: a preview arriving while another builds waits,
replacing any that already waited, and builds start at most every
MIN_INTERVAL seconds. Frames reuse the last thumbnail pages when they hold
every tile shown (only the instances change).
"""

import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from skitter.core.matching.matcher import MatchPreview, PreviewStage
from skitter.core.scene import MosaicScene
from skitter.ui.jobs import Job
from skitter.ui.render.atlas import Atlas, build_atlas
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.render.tile_textures import scene_instances

MIN_INTERVAL = 0.2  # seconds between the starts of two builds


@dataclass
class PreviewFrame:
    stage: PreviewStage
    sketch: SpriteLayer  # flat target colors where no tile is shown (may have no instances)
    tiles: np.ndarray | None  # tile sprite instances; None: no tiles yet
    pages: SpriteLayer | None  # a new tile layer (its pages), or None: reuse the last one's
    atlas: Atlas | None
    placed: int  # regions showing a tile
    needed: int  # regions that get one


def build_frame(preview: MatchPreview, library, ctx, atlas: Atlas | None) -> PreviewFrame:
    """Sprite layers for a preview (atlas: the last frame's, reused if it holds every tile)."""
    regions, result = preview.regions, preview.result
    needed = np.isfinite(preview.target[:, 0])
    waiting = needed if result is None else needed & (result.tile < 0)
    order = regions.stacking_order()
    order = order[waiting[order]]
    sketch = make_instances(len(order))
    sketch["pos"] = regions.center[order]
    sketch["size"] = regions.size[order]
    sketch["rotation"] = regions.rotation[order]
    sketch["tint"][:, :3] = preview.target[order]
    sketch["tint"][:, 3] = 1.0
    frame = PreviewFrame(preview.stage, SpriteLayer(None, sketch), None, None, None, 0,
                         int(needed.sum()))  # fmt: skip
    if result is None:
        return frame
    scene = MosaicScene.from_result(result, ctx)
    frame.placed = len(scene)
    if not len(scene):
        return frame
    if atlas is None or not np.isin(scene.slot, atlas.slots).all():
        atlas = build_atlas(library.thumbs, library.thumb_size, scene.slot)
        new = True
    else:
        new = False
    layer, uv = atlas.locate(scene.slot, scene.rect, scene.mirrored)
    frame.tiles = scene_instances(scene, layer, uv)
    frame.atlas = atlas
    if new:
        frame.pages = SpriteLayer(atlas.pages, frame.tiles)  # RGBA conversion here, not in the UI
    return frame


class PreviewBuilder(QObject):
    """Builds frames for a run's previews in the background, newest first."""

    ready = Signal(object)  # PreviewFrame

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job: Job | None = None
        self._pending: tuple | None = None
        self._atlas: Atlas | None = None
        self._library = None
        self._token = 0  # frames of runs before the last reset are dropped
        self._last = -float("inf")  # when the last build started
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._start)

    @property
    def busy(self) -> bool:
        return self._job is not None or self._pending is not None

    def submit(self, preview: MatchPreview, library, ctx) -> None:
        self._pending = (preview, library, ctx)
        self._schedule()

    def reset(self) -> None:
        """Drop waiting and building frames and the kept atlas (a run ended)."""
        self._token += 1
        self._pending = None
        self._timer.stop()
        if self._job is not None:
            self._job.cancel()
        self._atlas = self._library = None

    def wait(self, timeout: float | None = None) -> None:
        """Build every waiting frame now, delivering them (tests)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.busy and (deadline is None or time.monotonic() < deadline):
            self._timer.stop()
            self._last = -float("inf")
            if self._job is None:
                self._start()
            if self._job is not None:
                self._job.wait(timeout)

    def _schedule(self) -> None:
        if self._job is not None or self._pending is None or self._timer.isActive():
            return
        wait = MIN_INTERVAL - (time.monotonic() - self._last)
        if wait > 0:
            self._timer.start(max(1, round(wait * 1000)))
        else:
            self._start()

    def _start(self) -> None:
        if self._job is not None or self._pending is None:
            return
        preview, library, ctx = self._pending
        self._pending = None
        atlas = self._atlas if library is self._library else None
        token = self._token

        def work(progress, cancelled):
            return build_frame(preview, library, ctx, atlas)

        job = self._job = Job(work, parent=self)
        job.finished.connect(lambda frame: self._built(token, library, frame))
        job.stopped.connect(self._stopped)
        self._last = time.monotonic()
        job.start()

    def _built(self, token: int, library, frame: PreviewFrame) -> None:
        if token != self._token:
            return
        if frame.atlas is not None:
            self._atlas, self._library = frame.atlas, library
        self.ready.emit(frame)

    def _stopped(self) -> None:
        self._job = None
        self._schedule()

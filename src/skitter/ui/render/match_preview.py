"""A matching run in progress, drawn: the matcher's previews (MatchPreview,
core/matching/matcher.py) turned into sprite layers off the UI thread.

A frame shows the tiles chosen so far, from library thumbnails, over flat
squares of the target color in regions still without one. Packing the
thumbnails takes a while for big mosaics, so frames are built in the
background, newest first: a preview arriving while another builds waits,
replacing any that already waited, and builds start at most every
MIN_INTERVAL seconds.

Thumbnails go into a GrowingAtlas (atlas.py): frames of a run mostly show
tiles earlier frames showed, so each frame packs only the new ones and the GPU
uploads just those (patches), not every page again. Only when the pages are
full does a frame start a new atlas (and a new layer).
"""

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, Signal

from skitter.core.matching.matcher import MatchPreview, PreviewStage
from skitter.core.scene import MosaicScene
from skitter.ui.render.atlas import GrowingAtlas
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.render.tile_textures import scene_instances

logger = logging.getLogger(__name__)

MIN_INTERVAL = 0.2  # seconds between the starts of two builds
ROOM = 2.0  # a new atlas has cells for this many times the regions that need a tile


@dataclass
class PreviewFrame:
    stage: PreviewStage
    sketch: SpriteLayer  # flat target colors where no tile is shown (may have no instances)
    tiles: np.ndarray | None  # tile sprite instances; None: no tiles yet
    pages: SpriteLayer | None  # a new tile layer (its pages), or None: the last one's...
    patches: list[tuple] | None  # ... with these texels of its pages rewritten
    atlas: GrowingAtlas | None
    placed: int  # regions showing a tile
    needed: int  # regions that get one


def build_frame(preview: MatchPreview, library, ctx, atlas: GrowingAtlas | None) -> PreviewFrame:
    """Sprite layers for a preview (atlas: the last frame's, which grows if it can)."""
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
    frame = PreviewFrame(preview.stage, SpriteLayer(None, sketch), None, None, None, None, 0,
                         int(needed.sum()))  # fmt: skip
    if result is None:
        return frame
    scene = MosaicScene.from_result(result, ctx)
    frame.placed = len(scene)
    if not len(scene):
        return frame
    patches = None if atlas is None or atlas.library is not library else atlas.add(scene.slot)
    if patches is None:  # a new atlas (and layer)
        room = max(int(ROOM * frame.needed), int(ROOM * len(np.unique(scene.slot))), 1024)
        atlas = GrowingAtlas(library, room)
        atlas.add(scene.slot)
    layer, uv = atlas.locate(scene.slot, scene.rect, scene.mirrored)
    frame.tiles = scene_instances(scene, layer, uv)
    frame.atlas = atlas
    if patches is None:
        frame.pages = SpriteLayer(atlas.pages, frame.tiles)  # shares the RGBA pages
    else:
        frame.patches = patches
    return frame


class PreviewBuilder(QObject):
    """Builds frames for a run's previews in a background thread, newest first.

    One thread lives as long as the builder: starting a thread per frame could
    hold up the UI thread for half a second while matching keeps every core
    busy (it waits for the new thread to be scheduled).
    """

    ready = Signal(object)  # PreviewFrame
    _built = Signal(int, object, object)  # (token, library, frame), from the thread

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Condition()
        self._pending: tuple | None = None  # (token, preview, library, ctx)
        self._building = False
        self._atlas: GrowingAtlas | None = None  # the last frame's (built in the thread)
        self._token = 0  # frames of runs before the last reset are dropped
        self._closed = False
        self._thread: threading.Thread | None = None
        self._built.connect(self._deliver)

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._pending is not None or self._building

    def submit(self, preview: MatchPreview, library, ctx) -> None:
        with self._lock:
            self._pending = (self._token, preview, library, ctx)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True,
                                                name="skitter-preview")  # fmt: skip
                self._thread.start()
            self._lock.notify()

    def reset(self) -> None:
        """Drop waiting and building frames and the kept atlas (a run ended)."""
        with self._lock:
            self._token += 1
            self._pending = None
            self._atlas = None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pending = None
            self._lock.notify()

    def wait(self, timeout: float | None = None) -> None:
        """Until every waiting frame is built, then deliver them (tests)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.busy and (deadline is None or time.monotonic() < deadline):
            time.sleep(0.005)
        QCoreApplication.processEvents()

    def _run(self) -> None:
        last = -float("inf")
        while True:
            with self._lock:
                while self._pending is None and not self._closed:
                    self._lock.wait()
                if self._closed:
                    return
                wait = MIN_INTERVAL - (time.monotonic() - last)
                if wait > 0:  # newer previews may replace this one meanwhile
                    self._lock.wait(wait)
                    continue
                token, preview, library, ctx = self._pending
                self._pending = None
                self._building = True
                atlas = self._atlas
            last = time.monotonic()
            try:
                frame = build_frame(preview, library, ctx, atlas)
            except Exception:
                logger.exception("building a matching preview failed")
                frame = None
            with self._lock:
                self._building = False
                if frame is not None and token == self._token:
                    self._atlas = frame.atlas if frame.atlas is not None else self._atlas
                    self._built.emit(token, library, frame)

    def _deliver(self, token: int, library, frame: PreviewFrame) -> None:
        if token == self._token:
            self.ready.emit(frame)

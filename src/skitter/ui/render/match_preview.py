"""A matching run in progress, drawn: the matcher's previews (MatchPreview,
core/matching/matcher.py) turned into sprite layers off the UI thread.

A frame shows the tiles chosen so far, from library thumbnails, over flat
squares of the target color in regions still without one. Packing the
thumbnails takes a while for big mosaics, so frames are built in the
background, newest first: a preview arriving while another builds waits,
replacing any that already waited, and builds start at most every
MIN_INTERVAL seconds.

Thumbnails go into a GrowingAtlas: frames of a run mostly show tiles earlier
frames showed, so each frame packs only the new ones into free cells and the
GPU uploads just those (patches), not every page again. Only when the pages
are full does a frame start a new atlas (and a new layer).
"""

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, Signal

from skitter.core.matching.matcher import MatchPreview, PreviewStage
from skitter.core.scene import MosaicScene
from skitter.ui.render.atlas import PAGE, _uv, cell_size
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.render.tile_textures import scene_instances

logger = logging.getLogger(__name__)

MIN_INTERVAL = 0.2  # seconds between the starts of two builds
ROOM = 2.0  # a new atlas has cells for this many times the regions that need a tile


class GrowingAtlas:
    """Library thumbnails packed into fixed RGBA pages as they are first needed (cells
    are never reused or moved)."""

    def __init__(self, library, capacity: int):
        self.library = library
        thumbs = library.thumbs
        self.cell = min(cell_size(capacity), thumbs.shape[1])
        self.per_row = PAGE // self.cell
        self.per_page = self.per_row * self.per_row
        pages = max(1, -(-capacity // self.per_page))
        self.capacity = pages * self.per_page
        self.pages = np.zeros((pages, PAGE, PAGE, 4), np.uint8)
        self.pages[..., 3] = 255
        self.cell_of = np.full(len(library.thumb_size), -1, np.int64)  # slot -> cell
        self.count = 0

    def add(self, slots) -> list[tuple] | None:
        """Pack the slots not packed yet. Returns the texels written, as (page, x, y, w,
        h) rects, or None if they don't fit (nothing is written then)."""
        new = np.unique(np.asarray(slots, dtype=np.int64))
        new = new[self.cell_of[new] < 0]
        start = self.count
        if start + len(new) > self.capacity:
            return None
        self.cell_of[new] = np.arange(start, start + len(new))
        self.count += len(new)
        thumbs = self.library.thumbs
        factor = thumbs.shape[1] // self.cell
        for lo in range(0, len(new), 8192):
            block = np.asarray(thumbs[new[lo : lo + 8192]])
            if factor > 1:
                size = len(block), self.cell, factor, self.cell, factor, 3
                block = block.reshape(size).mean(axis=(2, 4)).astype(np.uint8)
            for i, image in enumerate(block):
                page, x, y = self._place(start + lo + i)
                self.pages[page, y : y + self.cell, x : x + self.cell, :3] = image
        return self._rects(start, self.count)

    def locate(self, slots, rects, mirrored=None) -> tuple[np.ndarray, np.ndarray]:
        """Texture layer and (u0, v0, u1, v1) of each crop rect of each (packed) slot."""
        slots = np.asarray(slots, dtype=np.int64)
        page, within = np.divmod(self.cell_of[slots], self.per_page)
        corner = np.stack([within % self.per_row, within // self.per_row], axis=1) * self.cell
        factor = self.library.thumbs.shape[1] // self.cell
        size = np.maximum(1, -(-self.library.thumb_size[slots] // factor)).astype(np.float64)
        rects = np.asarray(rects, dtype=np.float64)
        origin = corner + rects[:, :2] * size
        return page.astype(np.float32), _uv(origin, (rects[:, 2:] - rects[:, :2]) * size,
                                            PAGE, mirrored)  # fmt: skip

    def _place(self, cell: int) -> tuple[int, int, int]:
        page, within = divmod(cell, self.per_page)
        return page, (within % self.per_row) * self.cell, (within // self.per_row) * self.cell

    def _rects(self, start: int, end: int) -> list[tuple]:
        """Cells [start, end) as a few rects: per page, a partial first row, whole rows,
        and a partial last row."""
        rects, c, row = [], self.cell, self.per_row
        while start < end:
            page, within = divmod(start, self.per_page)
            stop = min(end, (page + 1) * self.per_page) - page * self.per_page  # in this page
            first_row, col = divmod(within, row)
            if col:  # the rest of a started row
                n = min(row - col, stop - within)
                rects.append((page, col * c, first_row * c, n * c, c))
                within += n
            whole = (stop - within) // row
            if whole:
                rects.append((page, 0, (within // row) * c, row * c, whole * c))
                within += whole * row
            if within < stop:  # the start of a last row
                rects.append((page, 0, (within // row) * c, (stop - within) * c, c))
            start = page * self.per_page + stop
        return rects


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

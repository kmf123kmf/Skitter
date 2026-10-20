"""Tiles at their photos' own resolution where a view zooms in on them.

The tile textures (tile_textures.TileTextures) hold each crop at about one
texel per mosaic unit: plenty for the whole mosaic, blocky up close.
`ZoomDetail` watches one view. Where a tile in view shows larger than its
texture has texels, its crop is read again from the photo at the density the
screen needs (rounded up in steps of DENSITY_STEP, so small zooms reuse what
was read), never beyond the photo's own pixels. The crops go onto the tile
layer's detail pages (SpriteLayer.detail_pages) and those tiles point at them;
everywhere else, and once the view zooms back out, tiles show their usual
textures.

None of it holds up the UI thread, however fast the view zooms and pans:
- the view is looked at again at most every UPDATE_S while it moves; zooming
  out drops detail no longer needed at once (only instance edits);
- photos are read by a few threads, nearest the middle of the view first, and
  crops that leave the view are dropped before they are read;
- crops read are kept in memory (CROP_CACHE_BYTES beyond those on the pages),
  so zooming back in reads nothing again;
- uploads go out a frame at a time, at most UPLOAD_BYTES a frame, each tile
  switching over in the frame its crop is uploaded.

When the crops in view would need more texels than the pages hold
(PAGE_FILL of all but one page), every tile in view gets the same lower
density. A crop that doesn't fit anywhere frees the page that keeps the
fewest texels in view, putting those back first.
"""

import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal

from skitter.core.scene import MosaicScene
from skitter.core.tiles.library import TileLibrary
from skitter.core.tiles.render import render_crops
from skitter.ui.render.atlas import texel_uv
from skitter.ui.render.sprites import SpriteLayer

DETAIL_PAGES = 3  # detail pages of a tile layer (each 48 MB on the GPU at 4096 texels)
DETAIL_SIDE = 4096
DENSITY_STEP = 2**0.5  # texel densities read: powers of this (per mosaic unit)
MIN_GAIN = 1.05  # read a crop again only for at least this much more density
KEEP_FACTOR = 2.0  # detail this many times denser than the screen needs is dropped
PAGE_FILL = 0.6  # share of the pages (all but one) the crops in view may take
VIEW_MARGIN = 0.1  # of the view's size, on each side: tiles just outside count as in view
UPDATE_S = 0.1  # most often the view is looked at again while it moves
UPLOAD_BYTES = 8 * 2**20  # most texel bytes uploaded a frame (at least one crop)
CROP_CACHE_BYTES = 128 * 2**20  # crops kept in memory beyond those on the pages
READERS = max(1, min(3, (os.cpu_count() or 2) // 2))  # threads reading photos


def step_up(density) -> np.ndarray:
    """The smallest power of DENSITY_STEP at least density."""
    exponent = np.ceil(np.log(np.asarray(density, np.float64)) / math.log(DENSITY_STEP) - 1e-9)
    return DENSITY_STEP**exponent


def crop_pixels(sizes, density, side: int = DETAIL_SIDE) -> np.ndarray:
    """(N, 2) texels (w, h) of crops of tiles of the given sizes (mosaic units) at
    `density` texels per unit (N,), none larger than a detail page (side texels)."""
    scaled = np.abs(np.asarray(sizes, np.float64)) * np.asarray(density, np.float64)[:, None]
    scaled *= np.minimum(1.0, side / scaled.max(axis=1, initial=1.0))[:, None]
    return np.clip(np.ceil(scaled - 1e-6), 1, side).astype(np.int64)


def detail_densities(
    zoom: float, main, native, sizes, budget: float, side: int = DETAIL_SIDE
) -> np.ndarray:
    """(N,) texels per unit to read each tile at for a view showing `zoom` device pixels
    per unit, 0 where its usual texture (`main` texels per unit) is enough.

    The density steps up from zoom, never beyond `native` (the photo's own pixels),
    and steps down for every tile alike while the crops would take more than budget
    texels.
    """
    main, native = np.asarray(main, np.float64), np.asarray(native, np.float64)
    level = float(step_up(max(zoom, 1e-9)))
    while True:
        density = np.minimum(level, native)
        density[density <= main * MIN_GAIN] = 0.0
        wanted = density > 0
        if not wanted.any():
            return density
        total = float(np.prod(crop_pixels(sizes[wanted], density[wanted], side), axis=1).sum())
        if total <= budget:
            return density
        level = float(step_up(level * math.sqrt(budget / total))) / DENSITY_STEP


def crop_keys(slots, rects, pixels) -> list[tuple]:
    """Identifies each crop image: (slot, x0, y0, x1, y1, w, h), its photo, window (to 5
    decimals) and size."""
    rects = np.round(np.asarray(rects, np.float64), 5).tolist()
    pixels = np.asarray(pixels, np.int64).tolist()
    return [(s, *r, w, h) for s, r, (w, h) in zip(np.asarray(slots).tolist(), rects, pixels,
                                                  strict=True)]  # fmt: skip


@dataclass
class _Crop:
    id: int
    image: np.ndarray | None  # (h, w, 3) uint8; None: the photo couldn't be read
    page: int = -1  # detail page it is on (-1: none)
    x: int = 0
    y: int = 0

    @property
    def nbytes(self) -> int:
        return 0 if self.image is None else self.image.nbytes


@dataclass
class _Want:
    """A crop the view needs: the tiles in view showing it, and how much it matters."""

    tiles: list | np.ndarray
    density: float
    priority: float  # distance of its nearest tile from the middle of the view


class _Reader(QObject):
    """A few threads reading crops from their photos, in the order last asked for."""

    read = Signal(object, object)  # key, (h, w, 3) uint8 or None (unreadable)

    def __init__(self, threads: int = READERS, parent=None):
        super().__init__(parent)
        self._lock = threading.Condition()
        self._queue: list[tuple] = []  # (key, path, rect, size), most urgent last
        self._reading: set = set()
        self._closed = False
        self._threads = [
            threading.Thread(target=self._run, daemon=True, name="skitter-zoom-detail")
            for _ in range(threads)
        ]
        for thread in self._threads:
            thread.start()

    def ask(self, items: list[tuple]) -> None:
        """Read these (most urgent first) instead of anything asked before and not
        started yet."""
        with self._lock:
            self._queue = [item for item in reversed(items) if item[0] not in self._reading]
            self._lock.notify_all()

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._queue or self._reading)

    def wait(self, timeout: float | None = None) -> None:
        """Block until everything asked for is read (tests)."""
        with self._lock:
            self._lock.wait_for(lambda: not (self._queue or self._reading), timeout)

    def close(self, timeout: float | None = 5.0) -> None:
        with self._lock:
            self._closed = True
            self._queue = []
            self._lock.notify_all()
        for thread in self._threads:
            thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self._lock:
                self._lock.wait_for(lambda: self._queue or self._closed)
                if self._closed:
                    return
                key, path, rect, size = self._queue.pop()
                self._reading.add(key)
            try:
                image = render_crops([path], [rect], [size], workers=0)[0]
            except Exception:  # never fatal: the tile keeps its usual texture
                image = None
            try:
                self.read.emit(key, None if isinstance(image, str) else image)
            finally:
                with self._lock:
                    self._reading.discard(key)
                    self._lock.notify_all()


class ZoomDetail(QObject):
    """Sharper crops for the tiles a canvas zooms in on (see the module docstring).

    The view calls `show` with its tile layer (made with detail pages) whenever
    it replaces the layer or its instances, which must then be the layer's
    usual ones (TileTextures.instances()).
    """

    def __init__(self, canvas, parent=None, readers: int = READERS):
        super().__init__(parent)
        self.canvas = canvas
        self._layer: SpriteLayer | None = None
        self._scene: MosaicScene | None = None
        self._library: TileLibrary | None = None
        self._version = None  # the library's version the crops were read from
        self._crops: OrderedDict[tuple, _Crop] = OrderedDict()  # least recently wanted first
        self._key_of_id: dict[int, tuple] = {}
        self._ids = 0
        self._spare = 0  # texel bytes of crops kept but on no page
        self._want: dict[tuple, _Want] = {}
        self._uploads: list[tuple] = []  # keys to put on the pages, most urgent first
        self._uploading = False  # the canvas runs _upload_frame every frame
        self._pages: list[list] = []  # per page: [x, y, shelf height, keys on it, last fill]
        self._fills = 0
        self._reader = _Reader(readers, self)
        self._reader.read.connect(self._on_read)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(round(UPDATE_S * 1000))
        self._timer.timeout.connect(self.update)
        canvas.view_changed.connect(self._on_view_changed)

    # What the view shows

    def show(self, layer: SpriteLayer | None, scene: MosaicScene | None,
             library: TileLibrary | None) -> None:  # fmt: skip
        """Sharpen this layer's tiles (one per tile of scene, in order); None: nothing."""
        if (layer is None or scene is None or library is None or scene.picture
                or not len(scene) or not layer.detail_pages):  # fmt: skip
            self._detach()
            return
        if library is not self._library or library.version != self._version:
            self._crops.clear()
            self._key_of_id.clear()
            self._spare = 0
            layer_changed = True
        else:
            layer_changed = layer is not self._layer
        self._layer, self._scene, self._library = layer, scene, library
        self._version = library.version
        if layer_changed:  # empty pages
            for crop in self._crops.values():
                if crop.page >= 0:
                    crop.page = -1
                    self._spare += crop.nbytes
            side = layer.detail_side
            self._pages = [[0, 0, 0, [], 0] for _ in range(layer.detail_pages)]
            self._side = side
            self._budget = max(1, layer.detail_pages - 1) * side * side * PAGE_FILL
        n = len(scene)
        self._base_layer = layer.instances["layer"].copy()
        self._base_uv = layer.instances["uv"].copy()
        self._crop_of = np.full(n, -1, np.int64)  # id of the crop each tile shows (-1: none)
        self._density = np.zeros(n)  # texels per unit of that crop (0: its usual)
        self._sizes = np.abs(scene.size.astype(np.float64))
        uv = np.abs(self._base_uv[:, 2:] - self._base_uv[:, :2]).astype(np.float64)
        texels = uv * layer.textures.shape[2:0:-1] + 1  # uv rects lie half a texel inside
        self._main = (texels / np.maximum(self._sizes, 1e-9)).min(axis=1)
        photo = np.stack([library.width[scene.slot], library.height[scene.slot]], axis=1)
        span = (scene.rect[:, 2:] - scene.rect[:, :2]) * photo
        native = (span / np.maximum(self._sizes, 1e-9)).min(axis=1)
        self._native = np.where(photo.min(axis=1) > 0, native, np.inf)  # unknown: no limit
        half = 0.5 * self._sizes
        c, s = np.abs(np.cos(scene.rotation)), np.abs(np.sin(scene.rotation))
        self._reach = np.stack([half[:, 0] * c + half[:, 1] * s,
                                half[:, 0] * s + half[:, 1] * c], axis=1)  # fmt: skip
        slots = np.unique(scene.slot)
        self._paths = dict(zip(slots.tolist(), library.paths(slots), strict=True))
        self.update()

    def _detach(self) -> None:
        self._layer = self._scene = None
        self._want = {}
        self._uploads = []
        self._reader.ask([])

    @property
    def sharpened(self) -> int:
        """Tiles showing a crop read for the view."""
        return 0 if self._layer is None else int(np.count_nonzero(self._density))

    @property
    def loading(self) -> bool:
        return bool(self._uploads) or self._reader.busy

    def wait(self, timeout: float | None = None) -> None:
        """Read and upload everything the view asked for (tests, scripts)."""
        self._timer.stop()
        self.update()
        while True:
            self._reader.wait(timeout)
            QCoreApplication.processEvents()
            while self._uploads:
                self._upload_frame(0.0)
            if not self._reader.busy:
                return

    def shutdown(self) -> None:
        self._timer.stop()
        self._reader.close()

    # Looking at the view

    def _on_view_changed(self) -> None:
        if self._layer is None:
            return
        zoom = self.canvas.device_zoom()
        drop = self._density > KEEP_FACTOR * zoom  # zoomed out: the usual textures do
        if drop.any():
            self._revert(np.flatnonzero(drop))
            self.canvas.update()
        if not self._timer.isActive():
            self._timer.start()

    def _in_view(self) -> np.ndarray:
        """Tiles in view (or nearly), by index."""
        canvas = self.canvas
        cam = canvas.camera
        w, h = cam.viewport
        corners = np.array([cam.screen_to_world(x, y) for x, y in ((0, 0), (w, 0), (0, h), (w, h))])
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        margin = (hi - lo) * VIEW_MARGIN
        lo, hi = lo - margin, hi + margin
        center, reach = self._scene.center, self._reach
        inside = np.all((center + reach >= lo) & (center - reach <= hi), axis=1)
        return np.flatnonzero(inside)

    def update(self) -> None:
        """Look at the view: ask for the crops it needs, show those already read."""
        if self._layer is None:
            return
        scene = self._scene
        tiles = self._in_view()
        zoom = self.canvas.device_zoom()
        density = detail_densities(zoom, self._main[tiles], self._native[tiles],
                                   self._sizes[tiles], self._budget, self._side)  # fmt: skip
        wanted = density > 0
        self._revert(tiles[~wanted & (self._density[tiles] > 0)])  # zoomed out
        tiles, density = tiles[wanted], density[wanted]
        # Tiles already showing a crop at least as dense keep it (zooming out a little).
        kept = self._density[tiles] >= density * (1 - 1e-9)
        density[kept] = self._density[tiles[kept]]
        pixels = crop_pixels(self._sizes[tiles], density, self._side)
        keys = crop_keys(scene.slot[tiles], scene.rect[tiles], pixels)
        for i in np.flatnonzero(kept):
            keys[i] = self._key_of_id[int(self._crop_of[tiles[i]])]
        middle = self.canvas.camera.center
        distance = np.hypot(*(scene.center[tiles] - middle).T)
        want: dict[tuple, _Want] = {}
        for i, key in enumerate(keys):
            entry = want.get(key)
            if entry is None:
                want[key] = _Want([tiles[i]], float(density[i]), float(distance[i]))
            else:
                entry.tiles.append(tiles[i])
                entry.priority = min(entry.priority, float(distance[i]))
        for entry in want.values():
            entry.tiles = np.asarray(entry.tiles, np.int64)
        self._want = want

        uploads, reads, shown = [], [], False
        for key, entry in sorted(want.items(), key=lambda item: item[1].priority):
            crop = self._crops.get(key)
            if crop is None:
                reads.append((key, self._paths[key[0]], key[1:5], key[5:7]))
                continue
            self._crops.move_to_end(key)
            if crop.page >= 0:
                shown |= self._show_crop(key, crop, entry)
            elif crop.image is not None:
                uploads.append(key)
        self._reader.ask(reads)
        self._uploads = uploads
        self._trim()
        if shown:
            self.canvas.update()
        self._start_uploads()

    # Crops arriving

    def _on_read(self, key: tuple, image: np.ndarray | None) -> None:
        if self._layer is None:
            return
        if key not in self._crops:
            self._ids += 1
            self._crops[key] = crop = _Crop(self._ids, image)
            self._key_of_id[crop.id] = key
            self._spare += crop.nbytes
        if image is not None and key in self._want:
            self._uploads.append(key)  # after those asked for earlier: they matter more
            self._start_uploads()
        self._trim()

    def _start_uploads(self) -> None:
        if self._uploads and not self._uploading:
            self._uploading = True
            self.canvas.add_animation(self._upload_frame)

    def _upload_frame(self, _elapsed: float) -> bool:
        """Put crops on the pages for the next frame, up to UPLOAD_BYTES; False when done
        (a canvas animation: called every frame)."""
        spent, shown = 0, False
        while self._uploads and self._layer is not None and (spent < UPLOAD_BYTES or not shown):
            key = self._uploads.pop(0)
            crop, entry = self._crops.get(key), self._want.get(key)
            if crop is None or entry is None or crop.image is None or crop.page >= 0:
                continue
            if not self._place(key, crop):
                continue  # no room: the tiles keep what they show
            spent += crop.image.nbytes
            shown |= self._show_crop(key, crop, entry)
        if shown:
            self._trim()
            self.canvas.update()
        self._uploading = bool(self._uploads) and self._layer is not None
        return self._uploading

    # The pages

    def _place(self, key: tuple, crop: _Crop) -> bool:
        """Find room for a crop on the pages and write it there (freeing a page if none
        has room); False if it can't have any."""
        h, w = crop.image.shape[:2]
        for p in sorted(range(len(self._pages)), key=lambda p: -self._pages[p][4]):
            if self._allocate(p, key, crop, w, h):
                return True
        keep = [sum(self._crops[k].nbytes for k in page[3] if k in self._want) // 3
                for page in self._pages]  # fmt: skip
        victim = min(range(len(self._pages)), key=lambda p: (keep[p], self._pages[p][4]))
        if keep[victim] + w * h > self._side**2:
            return False
        kept = self._clear_page(victim)
        for k in sorted(kept, key=lambda k: -self._crops[k].image.shape[0]):  # tallest first
            other = self._crops[k]
            ih, iw = other.image.shape[:2]
            if self._allocate(victim, k, other, iw, ih):
                self._show_crop(k, other, self._want[k])
            elif k not in self._uploads:
                self._uploads.append(k)
        return self._allocate(victim, key, crop, w, h)

    def _allocate(self, p: int, key: tuple, crop: _Crop, w: int, h: int) -> bool:
        page = self._pages[p]
        x, y, shelf = page[0], page[1], page[2]
        if x + w > self._side:
            x, y, shelf = 0, y + shelf, 0
        if y + h > self._side:
            return False
        page[0], page[1], page[2] = x + w, y, max(shelf, h)
        page[3].append(key)
        self._fills += 1
        page[4] = self._fills
        crop.page, crop.x, crop.y = p, x, y
        self._spare -= crop.image.nbytes
        self._layer.write_detail(p, x, y, crop.image)
        return True

    def _clear_page(self, p: int) -> list[tuple]:
        """Take every crop off page p; returns those the view still needs."""
        page = self._pages[p]
        keys, page[:] = page[3], [0, 0, 0, [], page[4]]
        kept = []
        for key in keys:
            crop = self._crops[key]
            crop.page = -1
            self._spare += crop.image.nbytes
            self._revert(np.flatnonzero(self._crop_of == crop.id))
            if key in self._want:
                kept.append(key)
        return kept

    def _show_crop(self, key: tuple, crop: _Crop, entry: _Want) -> bool:
        """Point the entry's tiles at the crop; True if any changed."""
        tiles = entry.tiles[self._crop_of[entry.tiles] != crop.id]
        if not len(tiles):
            return False
        h, w = crop.image.shape[:2]
        origin = np.tile([crop.x, crop.y], (len(tiles), 1)).astype(np.float64)
        size = np.tile([w, h], (len(tiles), 1)).astype(np.float64)
        instances = self._layer.instances
        instances["layer"][tiles] = self._layer.first_detail + crop.page
        instances["uv"][tiles] = texel_uv(origin, size, self._side, self._scene.mirrored[tiles])
        self._crop_of[tiles] = crop.id
        self._density[tiles] = entry.density
        self._layer.mark_dirty()
        return True

    def _revert(self, tiles) -> None:
        """Show these tiles' usual textures again."""
        if not len(tiles):
            return
        instances = self._layer.instances
        instances["layer"][tiles] = self._base_layer[tiles]
        instances["uv"][tiles] = self._base_uv[tiles]
        self._crop_of[tiles] = -1
        self._density[tiles] = 0.0
        self._layer.mark_dirty()

    def _trim(self) -> None:
        """Forget the least recently wanted crops on no page beyond CROP_CACHE_BYTES."""
        if self._spare <= CROP_CACHE_BYTES:
            return
        for key in list(self._crops):
            crop = self._crops[key]
            if crop.page < 0 and key not in self._want and crop.image is not None:
                del self._crops[key]
                del self._key_of_id[crop.id]
                self._spare -= crop.image.nbytes
                if self._spare <= CROP_CACHE_BYTES:
                    return

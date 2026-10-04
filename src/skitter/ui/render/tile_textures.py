"""Textures for every tile of a mosaic scene, shared by every view of it.

Tiles first show from the library's thumbnails, packed into atlas pages
straight away. A background job then reads each tile's crop from its
original file at one texel per mosaic unit (reduced uniformly if the total
would exceed DETAIL_TEXELS), packs the crops, and replaces the pages; views
redraw on `changed`. `pages`, `layer` and `uv` always describe the best
textures available, aligned with the scene's tiles (mirroring included).

Manual picks change a few tiles at a time (`set_scene`). With full detail
loaded, a newly needed crop goes into free space in the existing pages:
first cut from its thumbnail, then replaced by the real crop once read in
the background. Views redraw just those texels on `patched`. When the
pages fill up, the crops in use are repacked (no files are read again) and
`changed` is emitted as for new pages.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, Signal

from skitter.core.assembly import TileFiles
from skitter.core.scene import MosaicScene
from skitter.core.tiles.library import TileLibrary
from skitter.core.tiles.render import cut, render_crops_or_thumbs
from skitter.ui.jobs import Job, JobCancelled
from skitter.ui.render.atlas import PackedAtlas, build_atlas, pack_images
from skitter.ui.render.sprites import make_instances

DETAIL_TEXELS = 96 * 2**20  # full-detail budget: about 400 MB of GPU memory (RGBA) + mipmaps
DETAIL_PAGE = 4096  # atlas page size, texels


@dataclass(frozen=True)
class TileDetail:
    """Full-detail crops, packed for the GPU."""

    atlas: PackedAtlas
    scale: float  # texels per mosaic unit (1.0 = full size)
    failed: int  # files that could not be read (their thumbnails are shown)


def detail_sizes(sizes, budget: int = DETAIL_TEXELS, page: int = DETAIL_PAGE):
    """Pixel sizes (w, h) for crops of the given mosaic sizes, and the uniform scale used.

    Full size unless the total would exceed budget texels; no crop exceeds page.
    """
    sizes = np.asarray(sizes, dtype=np.float64).reshape(-1, 2)
    area = float(np.prod(np.ceil(sizes), axis=1).sum())
    scale = min(1.0, (budget / area) ** 0.5) if area else 1.0
    scaled = sizes * scale
    scaled *= np.minimum(1.0, page / scaled.max(axis=1, initial=1))[:, None]
    return np.clip(np.ceil(scaled - 1e-6), 1, page).astype(np.int64), scale


def build_detail(
    paths, rects, sizes, thumbs, thumb_size, progress, cancelled, budget: int = DETAIL_TEXELS
) -> TileDetail:
    """Read and pack full-detail crops (runs in a background job).

    paths, rects, sizes describe each distinct crop; thumbs / thumb_size are
    each crop's library thumbnail, shown instead when its file can't be read.
    """
    pixels, scale = detail_sizes(sizes, budget)

    def report(done, total):
        progress(f"Loading full-size tiles: {done:,} of {total:,} files", done / max(total, 1))

    rendered = render_crops_or_thumbs(
        paths, rects, pixels, thumbs, thumb_size, progress=report, cancelled=cancelled
    )
    if rendered is None:
        raise JobCancelled
    images, failed = rendered
    progress("Packing tiles…", None)
    return TileDetail(pack_images(images, DETAIL_PAGE), scale, len(failed))


@dataclass(frozen=True)
class DetailRequest:
    """The distinct crops a scene shows, with what reading them needs.

    Made on the library's thread (it isn't threadsafe); `build` may then run
    in a background job.
    """

    image: np.ndarray  # (N,) which crop each tile shows
    slots: np.ndarray  # (C,) library slot of each crop
    paths: list[str]
    rects: np.ndarray  # (C, 4)
    sizes: np.ndarray  # (C, 2) wanted size, texels
    thumbs: np.ndarray
    thumb_size: np.ndarray

    @classmethod
    def for_scene(cls, scene: MosaicScene, files: TileFiles, scale: float = 1.0):
        """Crops at `scale` texels per mosaic unit; tiles showing the same crop at the
        same size share one image."""
        size = np.ceil(scene.size * scale - 1e-6)
        keys, image = np.unique(
            np.column_stack([scene.slot, scene.rect.astype(np.float64), size]),
            axis=0,
            return_inverse=True,
        )
        index = files.index(keys[:, 0].astype(np.int64))
        return cls(
            image=image.reshape(-1),
            slots=keys[:, 0].astype(np.int64),
            paths=[files.paths[i] for i in index],
            rects=keys[:, 1:5],
            sizes=keys[:, 5:7],
            thumbs=files.thumbs[index],
            thumb_size=files.thumb_size[index],
        )

    def build(self, progress, cancelled, budget: int = DETAIL_TEXELS) -> TileDetail:
        return build_detail(self.paths, self.rects, self.sizes, self.thumbs, self.thumb_size,
                            progress, cancelled, budget)  # fmt: skip


def scene_instances(scene: MosaicScene, layer: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sprite instances showing the finished mosaic, in stacking order."""
    tiles = make_instances(len(scene))
    tiles["pos"] = scene.center
    tiles["size"] = scene.size
    tiles["rotation"] = scene.rotation
    tiles["layer"], tiles["uv"] = layer, uv
    tiles["offset"] = scene.tint_shift
    return tiles


def crop_key(slot, rect, size) -> tuple:
    """Identifies one crop image: the photo, its window and its requested size."""
    return (int(slot), *(round(float(v), 5) for v in rect), int(size[0]), int(size[1]))


def thumb_crop(thumbs, thumb_size, slot: int, rect, size) -> np.ndarray:
    """(h, w, 3) a crop cut from a library thumbnail: a stand-in until the file is read."""
    tw, th = (int(v) for v in thumb_size[slot])
    return cut(Image.fromarray(np.asarray(thumbs[slot])[:th, :tw]), rect, size)


class TileTextures(QObject):
    """The textures of a scene's tiles: thumbnails at once, full-size crops when read."""

    changed = Signal()  # pages replaced: views upload them again (layer and uv changed too)
    patched = Signal(object)  # [(page, x, y, w, h)] texels of pages rewritten in place
    status_changed = Signal(str)

    def __init__(self, scene: MosaicScene, library: TileLibrary, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.detail: TileDetail | None = None
        self.status = "Thumbnails"
        self._job: Job | None = None
        self._crop_jobs: list[Job] = []
        self._library = library
        if not len(scene):
            self.pages = np.zeros((1, 1, 1, 3), np.uint8)
            self.layer, self.uv = np.zeros(0), np.zeros((0, 4))
            return
        self._atlas = build_atlas(library.thumbs, library.thumb_size, scene.slot)
        self.pages: np.ndarray = self._atlas.pages
        self.layer, self.uv = self._atlas.locate(scene.slot, scene.rect, scene.mirrored)
        self._load_detail(library)

    @property
    def loading(self) -> bool:
        return self._job is not None

    def instances(self) -> np.ndarray:
        """Sprite instances showing the finished mosaic, in stacking order."""
        return scene_instances(self.scene, self.layer, self.uv)

    def cancel(self) -> None:
        for job in (self._job, *self._crop_jobs):
            if job is not None:
                job.cancel()

    def wait(self, timeout: float | None = None) -> None:
        """Block until full detail (and crops for picks) are loaded, delivering signals."""
        for job in (self._job, *list(self._crop_jobs)):
            if job is not None:
                job.wait(timeout)

    def _set_status(self, text: str) -> None:
        self.status = text
        self.status_changed.emit(text)

    def _load_detail(self, library: TileLibrary) -> None:
        scene = self.scene
        request = DetailRequest.for_scene(scene, TileFiles.read(library, scene.slot))
        job = Job(request.build, parent=self)
        job.progress.connect(lambda message, _: self._set_status(message))
        job.finished.connect(lambda detail: self._show_detail(detail, request, scene))
        job.failed.connect(lambda message: self._set_status(f"Thumbnails ({message})"))
        job.cancelled.connect(lambda: self._set_status("Thumbnails"))
        job.stopped.connect(self._job_stopped)
        self._job = job
        self._set_status("Loading full-size tiles…")
        job.start()

    def _job_stopped(self) -> None:
        self._job = None

    def _show_detail(self, detail: TileDetail, request: DetailRequest, scene) -> None:
        atlas = detail.atlas
        self.detail = detail
        self.pages = atlas.pages
        self._page, self._origin, self._size = (a.copy() for a in (atlas.page, atlas.origin,
                                                                   atlas.size))  # fmt: skip
        self._images = {
            crop_key(s, r, z): i
            for i, (s, r, z) in enumerate(
                zip(request.slots, request.rects, request.sizes, strict=True)
            )  # fmt: skip
        }
        self._image = request.image.astype(np.int64)
        self._start_shelf()
        if scene is not self.scene:  # picks made while loading
            self._update_detail(scene, self.scene, emit=False)
        self._locate()
        text = "Full size" if detail.scale >= 1 else f"Reduced to {detail.scale:.0%} (memory)"
        if detail.failed:
            text += f"; {detail.failed:,} unreadable files show thumbnails"
        self._set_status(text)
        self.changed.emit()

    # Picks

    def set_scene(self, scene: MosaicScene, library: TileLibrary) -> None:
        """Show an edited scene (same tiles placed, some showing other crops).

        Emits `patched` for texels rewritten in place, or `changed` when the
        pages had to be replaced; layer and uv are always updated.
        """
        old, self.scene, self._library = self.scene, scene, library
        if not len(scene):
            return
        if self.detail is None:
            if not np.isin(scene.slot, self._atlas.slots).all():  # new photos: new atlas
                self._atlas = build_atlas(library.thumbs, library.thumb_size, scene.slot)
                self.pages = self._atlas.pages
                self.layer, self.uv = self._atlas.locate(scene.slot, scene.rect, scene.mirrored)
                self.changed.emit()
                return
            self.layer, self.uv = self._atlas.locate(scene.slot, scene.rect, scene.mirrored)
            return
        self._update_detail(old, scene, emit=True)
        self._locate()

    def _sizes(self, scene) -> np.ndarray:
        return np.ceil(scene.size - 1e-6)  # requested texels (at scale 1, as DetailRequest)

    def _pixels(self, sizes) -> np.ndarray:
        """Texel sizes of crops at the loaded detail's scale."""
        scaled = np.asarray(sizes, np.float64).reshape(-1, 2) * self.detail.scale
        scaled *= np.minimum(1.0, DETAIL_PAGE / scaled.max(axis=1, initial=1))[:, None]
        return np.clip(np.ceil(scaled - 1e-6), 1, DETAIL_PAGE).astype(np.int64)

    def _update_detail(self, old, scene, emit: bool) -> None:
        changed = np.flatnonzero(
            (old.slot != scene.slot)
            | np.any(np.abs(old.rect - scene.rect) > 1e-6, axis=1)
            | (old.mirrored != scene.mirrored)
        )
        sizes = self._sizes(scene)
        new: list[tuple] = []
        for i in changed:
            key = crop_key(scene.slot[i], scene.rect[i], sizes[i])
            index = self._images.get(key)
            if index is None:
                index = self._images[key] = len(self._page) + len(new)
                new.append(key)
            self._image[i] = index
        if not new:
            return
        pixels = self._pixels([k[5:7] for k in new])
        thumbs, thumb_size = self._library.thumbs, self._library.thumb_size
        images = [
            thumb_crop(thumbs, thumb_size, k[0], k[1:5], p)
            for k, p in zip(new, pixels, strict=True)
        ]
        spots = [self._allocate(*p) for p in pixels]
        if any(s is None for s in spots):
            self._repack(images)
            if emit:
                self._locate()
                self.changed.emit()
        else:
            self._page = np.concatenate([self._page, [s[0] for s in spots]])
            self._origin = np.concatenate([self._origin, [s[1:] for s in spots]]).astype(np.int64)
            self._size = np.concatenate([self._size, pixels])
            first = len(self._page) - len(new)
            rects = [self._write(first + k, image) for k, image in enumerate(images)]
            if emit:
                self.patched.emit(rects)
        self._read_crops(new)

    def _locate(self) -> None:
        atlas = PackedAtlas(self.pages, self._page, self._origin, self._size)
        self.layer, self.uv = atlas.locate(self._image, self.scene.mirrored)

    def _start_shelf(self) -> None:
        last = len(self.pages) - 1
        on_last = self._page == last
        top = int((self._origin[on_last, 1] + self._size[on_last, 1]).max(initial=0))
        self._shelf = [last, 0, top, 0]  # page, x, y, shelf height

    def _allocate(self, w: int, h: int):
        """(page, x, y) of free space for a crop, or None when the pages are full."""
        side = self.pages.shape[1]
        page, x, y, shelf = self._shelf
        if x + w > side:
            x, y, shelf = 0, y + shelf, 0
        if y + h > side or w > side:
            return None
        self._shelf = [page, x + w, y, max(shelf, h)]
        return page, x, y

    def _write(self, index: int, image: np.ndarray) -> tuple:
        page, (x, y), (w, h) = self._page[index], self._origin[index], self._size[index]
        self.pages[page, y : y + h, x : x + w, :3] = image[:h, :w]
        return int(page), int(x), int(y), int(w), int(h)

    def _repack(self, new_images) -> None:
        """Pack the crops in use (and new ones) into fresh pages: frees abandoned space."""
        used = np.unique(self._image)
        first_new = len(self._page)
        old_used = used[used < first_new]
        images = [self.pages[self._page[i], y : y + h, x : x + w, :3]
                  for i, (x, y), (w, h) in zip(old_used, self._origin[old_used],
                                               self._size[old_used], strict=True)]  # fmt: skip
        atlas = pack_images([*images, *new_images], DETAIL_PAGE)
        remap = np.full(first_new + len(new_images), -1, np.int64)
        remap[old_used] = np.arange(len(old_used))
        remap[first_new:] = len(old_used) + np.arange(len(new_images))
        self._image = remap[self._image]
        self._images = {k: int(remap[i]) for k, i in self._images.items() if remap[i] >= 0}
        self.pages = atlas.pages
        self._page, self._origin, self._size = atlas.page, atlas.origin, atlas.size
        self._start_shelf()

    def _read_crops(self, keys: list[tuple]) -> None:
        """Read new crops from their files in the background, then swap them in."""
        library = self._library
        files = TileFiles.read(library, [k[0] for k in keys])
        index = files.index(np.array([k[0] for k in keys], np.int64))
        paths = [files.paths[i] for i in index]
        rects = np.array([k[1:5] for k in keys], np.float64)
        pixels = self._pixels([k[5:7] for k in keys])
        thumbs, thumb_size = files.thumbs[index], files.thumb_size[index]

        def work(progress, cancelled):
            done = render_crops_or_thumbs(paths, rects, pixels, thumbs, thumb_size, workers=0,
                                          cancelled=cancelled)  # fmt: skip
            if done is None:
                raise JobCancelled
            return done[0]

        job = Job(work, parent=self)
        job.finished.connect(lambda images: self._swap_in(keys, images))
        job.stopped.connect(lambda: self._crop_jobs.remove(job))
        self._crop_jobs.append(job)
        job.start()

    def _swap_in(self, keys: list[tuple], images: list[np.ndarray]) -> None:
        rects = []
        for key, image in zip(keys, images, strict=True):
            index = self._images.get(key)
            if index is not None and tuple(self._size[index]) == (image.shape[1], image.shape[0]):
                rects.append(self._write(index, image))
        if rects:
            self.patched.emit(rects)

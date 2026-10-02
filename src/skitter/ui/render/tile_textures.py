"""Textures for every tile of a mosaic scene, shared by every view of it.

Tiles first show from the library's thumbnails, packed into atlas pages
straight away. A background job then reads each tile's crop from its
original file at one texel per mosaic unit (reduced uniformly if the total
would exceed DETAIL_TEXELS), packs the crops, and replaces the pages; views
redraw on `changed`. `pages`, `layer` and `uv` always describe the best
textures available, aligned with the scene's tiles (mirroring included).
"""

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal

from skitter.core.scene import MosaicScene
from skitter.core.tiles.library import TileLibrary
from skitter.core.tiles.render import render_crops_or_thumbs
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


def build_detail(paths, rects, sizes, thumbs, thumb_size, progress, cancelled) -> TileDetail:
    """Read and pack full-detail crops (runs in a background job).

    paths, rects, sizes describe each distinct crop; thumbs / thumb_size are
    each crop's library thumbnail, shown instead when its file can't be read.
    """
    pixels, scale = detail_sizes(sizes)

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


class TileTextures(QObject):
    """The textures of a scene's tiles: thumbnails at once, full-size crops when read."""

    changed = Signal()  # pages, layer and uv now hold the full-detail textures
    status_changed = Signal(str)

    def __init__(self, scene: MosaicScene, library: TileLibrary, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.detail: TileDetail | None = None
        self.status = "Thumbnails"
        self._job: Job | None = None
        if not len(scene):
            self.pages = np.zeros((1, 1, 1, 3), np.uint8)
            self.layer, self.uv = np.zeros(0), np.zeros((0, 4))
            return
        atlas = build_atlas(library.thumbs, library.thumb_size, scene.slot)
        self.pages: np.ndarray = atlas.pages
        self.layer, self.uv = atlas.locate(scene.slot, scene.rect, scene.mirrored)
        self._load_detail(library)

    @property
    def loading(self) -> bool:
        return self._job is not None

    def instances(self) -> np.ndarray:
        """Sprite instances showing the finished mosaic, in stacking order."""
        scene = self.scene
        tiles = make_instances(len(scene))
        tiles["pos"] = scene.center
        tiles["size"] = scene.size
        tiles["rotation"] = scene.rotation
        tiles["layer"], tiles["uv"] = self.layer, self.uv
        tiles["offset"] = scene.tint_offset
        return tiles

    def cancel(self) -> None:
        if self._job is not None:
            self._job.cancel()

    def wait(self, timeout: float | None = None) -> None:
        """Block until full detail is loaded (or failed), delivering its signals."""
        if self._job is not None:
            self._job.wait(timeout)

    def _set_status(self, text: str) -> None:
        self.status = text
        self.status_changed.emit(text)

    def _load_detail(self, library: TileLibrary) -> None:
        scene = self.scene
        size = np.ceil(scene.size - 1e-6)
        # Tiles showing the same crop at the same size share one image.
        keys, image = np.unique(
            np.column_stack([scene.slot, scene.rect.astype(np.float64), size]),
            axis=0,
            return_inverse=True,
        )
        image = image.reshape(-1)
        crop_slot = keys[:, 0].astype(np.int64)
        paths = library.paths(crop_slot)
        thumbs = np.asarray(library.thumbs[crop_slot])  # read here: the library isn't threadsafe
        thumb_size = library.thumb_size[crop_slot]

        def work(progress, cancelled):
            return build_detail(paths, keys[:, 1:5], keys[:, 5:7], thumbs, thumb_size,
                                progress, cancelled)  # fmt: skip

        job = Job(work, parent=self)
        job.progress.connect(lambda message, _: self._set_status(message))
        job.finished.connect(lambda detail: self._show_detail(detail, image))
        job.failed.connect(lambda message: self._set_status(f"Thumbnails ({message})"))
        job.cancelled.connect(lambda: self._set_status("Thumbnails"))
        job.stopped.connect(self._job_stopped)
        self._job = job
        self._set_status("Loading full-size tiles…")
        job.start()

    def _job_stopped(self) -> None:
        self._job = None

    def _show_detail(self, detail: TileDetail, image: np.ndarray) -> None:
        self.detail = detail
        self.pages = detail.atlas.pages
        self.layer, self.uv = detail.atlas.locate(image, self.scene.mirrored)
        text = "Full size" if detail.scale >= 1 else f"Reduced to {detail.scale:.0%} (memory)"
        if detail.failed:
            text += f"; {detail.failed:,} unreadable files show thumbnails"
        self._set_status(text)
        self.changed.emit()

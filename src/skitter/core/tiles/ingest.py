"""Reading tile images quickly: small thumbnails, in parallel worker processes.

Decoding is the slow part of building a large library. JPEGs are decoded at
reduced scale (`Image.draft`, which lets libjpeg skip most of the work), HEIC
photos from their embedded thumbnail when it is large enough, and images are
read by a pool of processes.
"""

import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

THUMB = 32  # analysis thumbnail size (longest side), pixels
_ORIENTATION = 0x0112
_TURNED = (5, 6, 7, 8)  # EXIF orientations that swap width and height


class Thumbnail:
    __slots__ = ("width", "height", "pixels")

    def __init__(self, width: int, height: int, pixels: np.ndarray):
        self.width = width  # full image size, upright
        self.height = height
        self.pixels = pixels  # (h, w, 3) uint8, longest side THUMB


def load_thumbnail(path: str | Path, size: int = THUMB) -> Thumbnail:
    """Upright image size and a thumbnail fitting size x size (keeps aspect)."""
    with Image.open(path) as img:
        width, height = img.size
        if img.getexif().get(_ORIENTATION, 1) in _TURNED:
            width, height = height, width
        img.draft("RGB", (2 * size, 2 * size))  # JPEG: reduced scale; HEIC: a thumbnail
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((size, size), Image.Resampling.BOX)
        return Thumbnail(width, height, np.asarray(img, dtype=np.uint8))


def _load(path: str) -> Thumbnail | str:
    try:
        return load_thumbnail(path)
    except Exception as exc:  # any unreadable file is recorded, never fatal
        return f"{type(exc).__name__}: {exc}"


def load_thumbnails(
    paths: Iterable[str],
    workers: int | None = None,
    chunksize: int = 64,
    cancelled: Callable[[], bool] = lambda: False,
) -> Iterator[tuple[int, Thumbnail | str]]:
    """Yield (index, thumbnail or error message) for each path, in order.

    workers: processes to use (default: all cores; 0 reads in this process).
    Stops early once cancelled() returns True.
    """
    paths = list(paths)
    if workers == 0 or len(paths) < 2 * chunksize:
        for i, path in enumerate(paths):
            if cancelled():
                return
            yield i, _load(path)
        return
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        results = pool.map(_load, paths, chunksize=chunksize)
        try:
            for i, result in enumerate(results):
                if cancelled():
                    break
                yield i, result
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

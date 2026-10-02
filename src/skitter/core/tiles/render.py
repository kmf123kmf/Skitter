"""Tile crops at full detail, read from the original image files.

Analysis thumbnails (32 px) are enough to match tiles but far too coarse to
show them. `render_crops` reads each tile image once (JPEGs decoded at
reduced scale when that is still large enough), cuts out every crop of it
that is asked for, and resizes each to its own pixel size. Files are read by
a pool of processes.
"""

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageOps

_ORIENTATION = 0x0112
_TURNED = (5, 6, 7, 8)  # EXIF orientations that swap width and height


def cut(image: Image.Image, rect, size) -> np.ndarray:
    """(h, w, 3) uint8 of a crop window (fractions of the image), resized to size (w, h)."""
    width, height = image.size
    x0, y0, x1, y1 = (float(v) for v in rect)
    box = (x0 * width, y0 * height, x1 * width, y1 * height)
    out = image.resize((int(size[0]), int(size[1])), Image.Resampling.LANCZOS, box=box)
    return np.asarray(out, dtype=np.uint8)


def _render_file(job) -> list[np.ndarray] | str:
    path, rects, sizes = job
    try:
        with Image.open(path) as img:
            raw_w, raw_h = img.size
            turned = img.getexif().get(_ORIENTATION, 1) in _TURNED
            width, height = (raw_h, raw_w) if turned else (raw_w, raw_h)
            # Smallest scale of the whole image at which every crop has enough pixels.
            spans = np.maximum(np.asarray(rects)[:, 2:] - np.asarray(rects)[:, :2], 1e-6)
            scale = float(np.max(np.asarray(sizes) / (spans * (width, height))))
            need = (int(np.ceil(raw_w * scale)), int(np.ceil(raw_h * scale)))
            img.draft("RGB", need)  # JPEG: decode at reduced scale, never below need
            img = ImageOps.exif_transpose(img).convert("RGB")
            return [cut(img, rect, size) for rect, size in zip(rects, sizes, strict=True)]
    except Exception as exc:  # an unreadable or vanished file is reported, never fatal
        return f"{type(exc).__name__}: {exc}"


def render_crops(
    paths: Sequence[str],
    rects,
    sizes,
    *,
    workers: int | None = None,
    progress: Callable[[int, int], None] = lambda done, total: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> list[np.ndarray | str] | None:
    """Each crop of each image, resized: (h, w, 3) uint8, or an error message.

    paths, rects (N, 4) fractions of the upright image and sizes (N, 2) output
    (w, h) pixels describe N crops; crops of the same file share one read.
    workers: processes to use (default: all cores; 0 reads in this thread).
    progress(files done, files total) is called as files finish. Returns None
    if cancelled.
    """
    rects = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
    sizes = np.maximum(1, np.asarray(sizes, dtype=np.int64).reshape(-1, 2))
    groups: dict[str, list[int]] = {}
    for i, path in enumerate(paths):
        groups.setdefault(path, []).append(i)
    jobs = [(path, rects[ix], sizes[ix]) for path, ix in groups.items()]
    out: list[np.ndarray | str] = [""] * len(rects)
    total = len(jobs)
    progress(0, total)

    def store(done, ix, result) -> None:
        for k, i in enumerate(ix):
            out[i] = result if isinstance(result, str) else result[k]
        if done % 64 == 0 or done == total:
            progress(done, total)

    if workers == 0 or total < 64:
        for done, (job, ix) in enumerate(zip(jobs, groups.values(), strict=True), 1):
            if cancelled():
                return None
            store(done, ix, _render_file(job))
        return out
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        results = pool.map(_render_file, jobs, chunksize=8)
        try:
            for done, (ix, result) in enumerate(zip(groups.values(), results, strict=True), 1):
                if cancelled():
                    return None
                store(done, ix, result)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    return out

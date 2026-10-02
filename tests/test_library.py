"""Tile library cache and thumbnail ingest."""

import os

import numpy as np
from PIL import Image

from skitter.core.tiles.ingest import THUMB, load_thumbnail, load_thumbnails
from skitter.core.tiles.library import FAILED, MISSING, OK, TileLibrary


def save(path, size=(40, 20), color=(255, 0, 0), orientation=None):
    img = Image.new("RGB", size, color)
    kwargs = {}
    if orientation:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kwargs["exif"] = exif
    img.save(path, **kwargs)
    return path


def test_thumbnail_honors_exif_orientation(tmp_path):
    thumb = load_thumbnail(save(tmp_path / "turned.jpg", (400, 200), orientation=6))
    assert (thumb.width, thumb.height) == (200, 400)
    assert thumb.pixels.shape == (THUMB, THUMB // 2, 3)
    plain = load_thumbnail(save(tmp_path / "plain.png", (90, 30)))
    assert plain.pixels.shape == (11, THUMB, 3) and (plain.width, plain.height) == (90, 30)


def test_parallel_loading_keeps_order_and_reports_errors(tmp_path):
    paths = [str(save(tmp_path / f"{i}.png", color=(i * 40, 0, 0))) for i in range(4)]
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    paths.insert(2, str(bad))
    results = dict(load_thumbnails(paths, workers=2, chunksize=1))
    assert sorted(results) == list(range(5))
    assert isinstance(results[2], str)
    assert results[4].pixels[0, 0, 0] == 120


def test_library_ingests_rescans_and_tracks_changes(tmp_path):
    photos = tmp_path / "photos"
    (photos / "sub").mkdir(parents=True)
    a = save(photos / "a.png", (60, 30), (10, 200, 30))
    save(photos / "sub" / "b.jpg", (30, 60))
    (photos / "notes.txt").write_text("ignored")
    (photos / "broken.png").write_bytes(b"\x89PNG broken")

    lib = TileLibrary(tmp_path / "cache")
    lib.set_roots([photos])
    report = lib.update(workers=0)
    assert (report.added, report.failed) == (3, 1)
    assert len(lib) == 2 and lib.counts() == {"ok": 2, "failed": 1, "missing": 0}
    version = lib.version

    slot = next(s for s in lib.ids if lib.paths([s])[0].endswith("a.png"))
    tw, th = lib.thumb_size[slot]
    assert (tw, th) == (THUMB, THUMB // 2) and (lib.width[slot], lib.height[slot]) == (60, 30)
    np.testing.assert_array_equal(lib.thumbs[slot, 0, 0], (10, 200, 30))
    assert (lib.thumbs[slot, th:] == 0).all()

    # Nothing changed: nothing re-read, same version.
    again = lib.update(workers=0)
    assert (again.unchanged, again.added, again.updated) == (3, 0, 0)
    assert lib.version == version

    # A changed file is re-read in place; a deleted one goes missing.
    save(a, (60, 30), (0, 0, 255))
    os.utime(a, (1e9, 1e9))
    os.remove(photos / "sub" / "b.jpg")
    later = lib.update(workers=0)
    assert (later.updated, later.missing) == (1, 1)
    assert lib.version > version
    np.testing.assert_array_equal(lib.thumbs[slot, 0, 0], (0, 0, 255))
    assert sorted(lib.status.tolist()) == sorted([OK, MISSING, FAILED])
    lib.close()

    # Everything persists.
    reopened = TileLibrary(tmp_path / "cache")
    assert len(reopened) == 1 and reopened.roots == [photos.resolve()]
    np.testing.assert_array_equal(reopened.thumbs[slot, 0, 0], (0, 0, 255))
    reopened.close()


def test_library_grows_thumbnail_store(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    for i in range(1100):  # beyond the initial capacity of 1024 slots
        save(photos / f"{i:04}.png", (8, 8), (i % 256, 0, 0))
    lib = TileLibrary(tmp_path / "cache")
    lib.set_roots([photos])
    lib.update(workers=0)
    assert len(lib) == 1100 and lib.thumbs.shape[0] == 1100
    slot = next(s for s in lib.ids if lib.paths([s])[0].endswith("1099.png"))
    assert lib.thumbs[slot, 0, 0, 0] == 1099 % 256
    lib.close()


def test_cancel_stops_reading(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    for i in range(10):
        save(photos / f"{i}.png")
    lib = TileLibrary(tmp_path / "cache")
    lib.set_roots([photos])
    calls = iter(range(100))
    report = lib.update(workers=0, cancelled=lambda: next(calls) >= 3)
    read = len(lib)
    assert report.cancelled and read < 10
    assert lib.update(workers=0).added == 10 - read  # the rest, on the next update
    assert len(lib) == 10
    lib.close()


def test_render_crops_cuts_each_crop_at_its_size(tmp_path):
    from skitter.core.tiles.render import render_crops

    # Left half red, right half blue, 400 x 200, stored turned (EXIF 6: upright 200 x 400).
    raw = np.zeros((200, 400, 3), np.uint8)
    raw[:, :200] = (255, 0, 0)
    raw[:, 200:] = (0, 0, 255)
    turned = tmp_path / "turned.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(raw).save(turned, exif=exif, quality=95)
    plain = tmp_path / "plain.png"
    Image.fromarray(raw).save(plain)

    paths = [str(plain), str(plain), str(turned), str(tmp_path / "gone.png")]
    rects = [[0, 0, 0.5, 1], [0.5, 0, 1, 1], [0, 0, 1, 0.5], [0, 0, 1, 1]]
    sizes = [[50, 50], [30, 60], [40, 40], [8, 8]]
    left, right, top, missing = render_crops(paths, rects, sizes, workers=0)
    assert left.shape == (50, 50, 3) and right.shape == (60, 30, 3)
    assert left[:, :-3, 0].min() > 250 and right[:, 3:, 2].min() > 250  # edges blend a little
    # Upright, the turned image's top half is its stored left half (red).
    assert top.shape == (40, 40, 3) and top[5:-5, 5:-5, 0].min() > 200
    assert isinstance(missing, str)

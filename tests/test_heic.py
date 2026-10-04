"""HEIC / HEIF photos (pillow-heif): sources, tile library, full-detail crops."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from PIL import Image

from skitter.core.imaging import IMAGE_EXTENSIONS, load_image
from skitter.core.tiles.ingest import load_thumbnail
from skitter.core.tiles.library import walk_images
from skitter.core.tiles.render import _render_file, render_crops


def halves(width=400, height=200) -> np.ndarray:
    """Left half red, right half blue."""
    image = np.zeros((height, width, 3), np.uint8)
    image[:, : width // 2] = (255, 0, 0)
    image[:, width // 2 :] = (0, 0, 255)
    return image


@pytest.fixture(params=[[], [64]], ids=["no-thumbnail", "thumbnail"])
def heic(tmp_path, request):
    """A photo stored sideways, like a phone's in portrait: upright it is 200 x 400.

    HEIF keeps the rotation in the container (irot), which readers apply; the
    EXIF orientation 6 an iPhone also writes is informational only. Putting
    the EXIF in image.info makes the encoder write both, as a phone does.
    Phones also embed a small preview, which reading uses when it is enough.
    """
    path = tmp_path / "phone.HEIC"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° clockwise to show upright
    image = Image.fromarray(halves())
    image.info["exif"] = exif.tobytes()
    image.save(path, format="HEIF", quality=95, thumbnails=request.param)
    return path


def test_heic_is_a_known_image_type(tmp_path, heic):
    assert {".heic", ".heif"} <= IMAGE_EXTENSIONS
    found = [path for path, _, _ in walk_images(tmp_path)]
    assert found == [os.path.normcase(str(heic))]  # upper-case extension too


def test_heic_source_loads_upright(heic):
    image = load_image(heic)
    assert image.shape == (400, 200, 3)
    # Upright, the stored left half (red) is on top.
    assert image[20, 100, 0] > 200 and image[20, 100, 2] < 60
    assert image[380, 100, 2] > 200 and image[380, 100, 0] < 60


def test_heic_thumbnail_and_size_are_upright(heic):
    thumb = load_thumbnail(heic)
    assert (thumb.width, thumb.height) == (200, 400)
    assert thumb.pixels.shape[:2] == (32, 16)
    assert thumb.pixels[2, 8, 0] > 200 and thumb.pixels[-3, 8, 2] > 200


def test_heic_full_detail_crop(heic):
    (top,) = render_crops([str(heic)], [[0, 0, 1, 0.5]], [[50, 100]], workers=0)
    assert top.shape == (100, 50, 3)
    assert top[5:-5, 5:-5, 0].min() > 200  # the upright top half is red


def test_heic_decodes_in_worker_processes(heic):
    # Tile reading runs in fresh processes: they must register the decoder themselves.
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=spawn) as pool:
        result = pool.submit(_render_file, (str(heic), [[0, 0, 1, 1]], [[20, 40]])).result()
    assert not isinstance(result, str), result
    assert result[0].shape == (40, 20, 3)


def test_heic_tiles_join_the_library(tmp_path, heic):
    from skitter.core.tiles.library import TileLibrary

    library = TileLibrary(tmp_path / "cache")
    try:
        library.set_roots([tmp_path])
        report = library.update(workers=0)
        assert report.added == 1 and len(library) == 1
        slot = library.ids[0]
        assert (library.width[slot], library.height[slot]) == (200, 400)
        assert library.paths([slot])[0].lower().endswith(".heic")
    finally:
        library.close()

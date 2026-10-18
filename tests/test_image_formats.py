"""Every listed image type Pillow can write loads as a source, a tile and a crop."""

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

# extension -> Pillow save format and options (lossy ones at high quality)
WRITABLE = {
    ".jpg": ("JPEG", {"quality": 95}),
    ".jfif": ("JPEG", {"quality": 95}),
    ".png": ("PNG", {}),
    ".apng": ("PNG", {}),
    ".gif": ("GIF", {}),
    ".bmp": ("BMP", {}),
    ".dib": ("DIB", {}),
    ".tiff": ("TIFF", {}),
    ".webp": ("WEBP", {"quality": 95}),
    ".avif": ("AVIF", {"quality": 95}),
    ".jxl": ("JXL", {"quality": 95}),
    ".jp2": ("JPEG2000", {}),
    ".j2k": ("JPEG2000", {}),
    ".tga": ("TGA", {}),
    ".qoi": ("QOI", {}),
    ".ppm": ("PPM", {}),
    ".pcx": ("PCX", {}),
    ".sgi": ("SGI", {}),
}


def halves(width=400, height=200) -> np.ndarray:
    """Top half red, bottom half blue."""
    image = np.zeros((height, width, 3), np.uint8)
    image[: height // 2] = (255, 0, 0)
    image[height // 2 :] = (0, 0, 255)
    return image


@pytest.fixture(params=sorted(WRITABLE))
def picture(tmp_path, request):
    ext = request.param
    save_format, options = WRITABLE[ext]
    path = tmp_path / f"picture{ext.upper()}"  # upper-case extensions count too
    Image.fromarray(halves()).save(path, format=save_format, **options)
    return path


def is_red(rgb) -> bool:
    return rgb[0] > 200 and rgb[2] < 60


def is_blue(rgb) -> bool:
    return rgb[2] > 200 and rgb[0] < 60


def test_writable_types_are_listed():
    assert set(WRITABLE) <= IMAGE_EXTENSIONS


def test_found_in_a_folder(tmp_path, picture):
    found = [path for path, _, _ in walk_images(tmp_path)]
    assert found == [os.path.normcase(str(picture))]


def test_loads_as_a_source(picture):
    image = load_image(picture)
    assert image.shape == (200, 400, 4)
    assert (image[..., 3] == 255).all()
    assert is_red(image[20, 200]) and is_blue(image[180, 200])


def test_loads_as_a_tile(picture):
    thumb = load_thumbnail(picture)
    assert (thumb.width, thumb.height) == (400, 200)
    assert thumb.pixels.shape[:2] == (16, 32)
    assert is_red(thumb.pixels[2, 16]) and is_blue(thumb.pixels[-3, 16])


def test_full_detail_crop(picture):
    (top,) = render_crops([str(picture)], [[0, 0, 1, 0.5]], [[100, 50]], workers=0)
    assert top.shape == (50, 100, 3)
    assert top[5:-5, 5:-5, 0].min() > 200


def test_jpeg_xl_decodes_in_worker_processes(tmp_path):
    # Tile reading runs in fresh processes: they must register the decoder themselves.
    path = tmp_path / "picture.jxl"
    Image.fromarray(halves()).save(path, format="JXL")
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=spawn) as pool:
        result = pool.submit(_render_file, (str(path), [[0, 0, 1, 1]], [[40, 20]])).result()
    assert not isinstance(result, str), result
    assert result[0].shape == (20, 40, 3)


def test_jpeg_xl_orientation_is_honored(tmp_path):
    # A JPEG XL photo from a camera keeps its EXIF orientation; upright it is 200 x 400.
    path = tmp_path / "phone.jxl"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° clockwise to show upright
    Image.fromarray(halves()).save(path, format="JXL", exif=exif.tobytes())
    image = load_image(path)
    assert image.shape == (400, 200, 4)
    assert is_blue(image[200, 20]) and is_red(image[200, 180])  # the top went right
    thumb = load_thumbnail(path)
    assert (thumb.width, thumb.height) == (200, 400)

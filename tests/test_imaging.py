import numpy as np

from skitter.core.imaging import (
    average_color,
    center_crop_square,
    find_images,
    load_image,
    resize,
    save_image,
)


def test_save_and_load_roundtrip(tmp_path):
    array = np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    save_image(array, path)
    np.testing.assert_array_equal(load_image(path), array)


def test_find_images_filters_and_sorts(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ["b.JPG", "a.png", "sub/c.webp", "notes.txt"]:
        (tmp_path / name).touch()
    assert [p.name for p in find_images(tmp_path)] == ["a.png", "b.JPG", "c.webp"]


def test_resize_shape():
    array = np.zeros((40, 60, 3), dtype=np.uint8)
    assert resize(array, (16, 8)).shape == (8, 16, 3)


def test_center_crop_square():
    array = np.zeros((40, 60, 3), dtype=np.uint8)
    assert center_crop_square(array).shape == (40, 40, 3)


def test_average_color():
    array = np.zeros((2, 2, 3), dtype=np.uint8)
    array[0, :] = [200, 100, 0]
    np.testing.assert_allclose(average_color(array), [100, 50, 0])

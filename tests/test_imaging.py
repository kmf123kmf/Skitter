import numpy as np

from skitter.core.imaging import load_image, save_image


def test_save_and_load_roundtrip(tmp_path):
    array = np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    save_image(array, path)
    np.testing.assert_array_equal(load_image(path), array)

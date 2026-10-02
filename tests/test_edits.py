import numpy as np
import pytest

from skitter.core.edits import Crop, FlipHorizontal, FlipVertical, Rotate90, apply_edits

# 2x3 image whose pixel values encode their position: value = 10 * row + col
IMAGE = (np.arange(2)[:, None] * 10 + np.arange(3)[None, :]).astype(np.uint8)[..., None]


def values(image):
    return image[..., 0].tolist()


def test_flips():
    assert values(FlipHorizontal().apply(IMAGE)) == [[2, 1, 0], [12, 11, 10]]
    assert values(FlipVertical().apply(IMAGE)) == [[10, 11, 12], [0, 1, 2]]


def test_rotate_right_is_clockwise():
    # Clockwise: the left column (bottom to top) becomes the top row.
    assert values(Rotate90(1).apply(IMAGE)) == [[10, 0], [11, 1], [12, 2]]
    assert values(Rotate90(-1).apply(IMAGE)) == [[2, 12], [1, 11], [0, 10]]


def test_crop():
    assert values(Crop(1, 0, 2, 2).apply(IMAGE)) == [[1, 2], [11, 12]]


def test_crop_validation():
    with pytest.raises(ValueError):
        Crop(0, 0, 0, 1)
    with pytest.raises(ValueError):
        Crop(2, 0, 2, 1).apply(IMAGE)


def test_apply_edits_chains_and_is_contiguous():
    result = apply_edits(IMAGE, [Rotate90(1), FlipHorizontal(), Crop(0, 0, 2, 1)])
    assert values(result) == [[0, 10]]
    assert result.flags["C_CONTIGUOUS"]


def test_describe():
    assert Rotate90(1).describe() == "Rotate right"
    assert Rotate90(-1).describe() == "Rotate left"
    assert Crop(0, 0, 640, 480).describe() == "Crop to 640 × 480"

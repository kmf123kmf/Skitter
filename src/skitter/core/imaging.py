"""Image loading and basic array operations.

Images are represented as uint8 numpy arrays of shape (H, W, 3), RGB.

Every image is read with Pillow. Importing this module (which importing
skitter.core does, so worker processes get it too) adds decoders Pillow
lacks: HEIC / HEIF through pillow-heif. HEIF images decode upright, and
`draft` picks an embedded thumbnail when one is large enough, like JPEG's
reduced-scale decoding.
"""

from pathlib import Path

import numpy as np
import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".heic", ".heif",
}  # fmt: skip


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as an RGB uint8 array, honoring EXIF orientation."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def save_image(array: np.ndarray, path: str | Path) -> None:
    Image.fromarray(array).save(path)

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


def find_images(folder: str | Path) -> list[Path]:
    """Recursively list image files under a folder, sorted for stable ordering."""
    return sorted(
        p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def resize(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to (width, height)."""
    return np.asarray(Image.fromarray(array).resize(size, Image.Resampling.LANCZOS))


def center_crop_square(array: np.ndarray) -> np.ndarray:
    h, w = array.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return array[top : top + side, left : left + side]


def average_color(array: np.ndarray) -> np.ndarray:
    """Mean RGB of an image as a float array of shape (3,)."""
    return array.reshape(-1, array.shape[-1]).mean(axis=0)

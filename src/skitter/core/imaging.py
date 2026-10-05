"""Image loading and basic array operations.

Images are represented as uint8 numpy arrays of shape (H, W, 3), RGB.
Source images are (H, W, 4), RGBA: every source gets an alpha channel when
loaded (opaque if the file has none), and edits carry it along, so a mask
is just alpha, whether from the file or (later) drawn in the app. Alpha is
a solid mask: pixels at least half opaque are visible (`visible_mask`),
the rest lie outside the picture, as if beyond its border.

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
from scipy import ndimage

pillow_heif.register_heif_opener()

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".heic", ".heif",
}  # fmt: skip


ALPHA_THRESHOLD = 128  # alpha at least this (half opaque) is visible


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as an RGBA uint8 array (opaque if the file has no
    transparency), honoring EXIF orientation."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return np.asarray(img.convert("RGBA"), dtype=np.uint8)


def visible_mask(image: np.ndarray) -> np.ndarray | None:
    """(H, W) bool: pixels at least half opaque. None when all are (or the image
    has no alpha), so callers can skip masking altogether."""
    if image.ndim != 3 or image.shape[2] < 4:
        return None
    visible = image[..., 3] >= ALPHA_THRESHOLD
    return None if visible.all() else visible


def fill_hidden(rgb: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """RGB with every hidden pixel taking the color of the nearest visible one,
    as sampling past the image's border takes the nearest edge pixel. Unchanged
    if nothing is visible."""
    if visible.all() or not visible.any():
        return rgb
    iy, ix = ndimage.distance_transform_edt(~visible, return_distances=False, return_indices=True)
    return rgb[iy, ix]


def save_image(array: np.ndarray, path: str | Path) -> None:
    Image.fromarray(array).save(path)

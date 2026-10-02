"""Perceptual color: sRGB <-> OKLab, vectorized and inside numba kernels.

OKLab (Björn Ottosson, 2020) is close to perceptually uniform: Euclidean
distance approximates visible difference. L is lightness in [0, 1]; a and b
are roughly within ±0.4. Multiplying distances by 100 gives a ΔE scale on
which about 2 is a just noticeable difference.
"""

import numpy as np
from numba import njit

# Linear sRGB -> LMS, and cube-rooted LMS -> Lab.
_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_M2_INV = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ]
)
_M1_INV = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ]
)


def srgb_to_linear(c):
    """sRGB components in [0, 1] to linear light."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


# Linear value of each 8-bit sRGB level.
LINEAR_LUT = srgb_to_linear(np.arange(256) / 255).astype(np.float32)


def linear_to_oklab(rgb) -> np.ndarray:
    lms = np.asarray(rgb, dtype=np.float64) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_linear(lab) -> np.ndarray:
    lms = (np.asarray(lab, dtype=np.float64) @ _M2_INV.T) ** 3
    return lms @ _M1_INV.T


def rgb8_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 sRGB to float32 OKLab."""
    return linear_to_oklab(LINEAR_LUT[np.asarray(rgb, dtype=np.uint8)]).astype(np.float32)


def oklab_to_srgb(lab) -> np.ndarray:
    """OKLab to sRGB components in [0, 1] (out-of-gamut colors are clipped)."""
    return linear_to_srgb(oklab_to_linear(lab))


def oklab_to_rgb8(lab) -> np.ndarray:
    return np.round(oklab_to_srgb(lab) * 255).astype(np.uint8)


@njit(cache=True, inline="always")
def rgb8_to_oklab_nb(r, g, b, lut):
    """One uint8 sRGB pixel to OKLab inside numba kernels (lut = LINEAR_LUT)."""
    lr, lg, lb = lut[r], lut[g], lut[b]
    l_ = np.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb)
    m_ = np.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb)
    s_ = np.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )

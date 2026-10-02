"""Vectorized easing curves.

Each curve maps progress t in [0, 1] to [0, 1] and accepts scalars or arrays,
so a whole set of tiles can be eased in one call.
"""

from collections.abc import Callable

import numpy as np

Easing = Callable[[float], float]


def progress(t, start, duration):
    """Normalized progress through [start, start + duration], clamped to [0, 1]."""
    return np.clip((np.asarray(t) - start) / duration, 0.0, 1.0)


def lerp(a, b, t):
    return a + (b - a) * t


def linear(t):
    return np.asarray(t)


def ease_in_cubic(t):
    t = np.asarray(t)
    return t**3


def ease_out_cubic(t):
    t = np.asarray(t)
    return 1 - (1 - t) ** 3


def ease_in_out_cubic(t):
    t = np.asarray(t)
    return np.where(t < 0.5, 4 * t**3, 1 - (-2 * t + 2) ** 3 / 2)


def ease_out_back(t, overshoot=1.70158):
    t = np.asarray(t)
    c3 = overshoot + 1
    return 1 + c3 * (t - 1) ** 3 + overshoot * (t - 1) ** 2

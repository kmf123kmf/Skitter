"""The color error heat map ramp, shared by every view that shows ΔE."""

import numpy as np

HEAT_MAX_DE = 25.0  # ΔE shown fully red


def heat_colors(error: np.ndarray) -> np.ndarray:
    """(N, 3) colors from green (no error) through yellow to red (HEAT_MAX_DE and above)."""
    t = np.clip(np.nan_to_num(error, nan=0.0) / HEAT_MAX_DE, 0, 1)[:, None]
    green, yellow, red = np.array([[0.2, 0.75, 0.3], [1.0, 0.85, 0.2], [0.9, 0.2, 0.15]])
    low = green + (yellow - green) * np.minimum(t * 2, 1)
    return np.where(t < 0.5, low, yellow + (red - yellow) * (t * 2 - 1))

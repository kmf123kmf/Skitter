"""2D pan/zoom camera mapping world coordinates to the screen.

World coordinates follow image conventions: x to the right, y downward.
Screen coordinates are logical (device-independent) pixels.
"""

import numpy as np

WorldRect = tuple[float, float, float, float]  # x, y, width, height


def clamp_center(center, zoom: float, viewport, rect: WorldRect) -> np.ndarray:
    """Constrain a camera center so rect covers the viewport where it can.

    On each axis where rect is smaller than the viewport it is centered;
    otherwise the view may not scroll past its edges.
    """
    center = np.array(center, dtype=float)
    for axis, (origin, size) in enumerate(((rect[0], rect[2]), (rect[1], rect[3]))):
        half = viewport[axis] / 2 / zoom
        if size <= 2 * half:
            center[axis] = origin + size / 2
        else:
            center[axis] = np.clip(center[axis], origin + half, origin + size - half)
    return center


class Camera2D:
    def __init__(self, min_zoom: float = 1e-3, max_zoom: float = 1e3):
        self.center = np.zeros(2)
        self.zoom = 1.0  # screen pixels per world unit
        self.viewport = np.ones(2)  # screen width, height
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

    def screen_to_world(self, x: float, y: float) -> np.ndarray:
        return self.center + (np.array([x, y]) - self.viewport / 2) / self.zoom

    def world_to_screen(self, x: float, y: float) -> np.ndarray:
        return (np.array([x, y]) - self.center) * self.zoom + self.viewport / 2

    def pan_pixels(self, dx: float, dy: float) -> None:
        """Move the view so content follows a drag of (dx, dy) screen pixels."""
        self.center -= np.array([dx, dy]) / self.zoom

    def zoom_at_params(self, x: float, y: float, factor: float) -> tuple[np.ndarray, float]:
        """Center and zoom after zooming by factor about screen point (x, y)."""
        anchor = self.screen_to_world(x, y)
        zoom = float(np.clip(self.zoom * factor, self.min_zoom, self.max_zoom))
        return anchor - (np.array([x, y]) - self.viewport / 2) / zoom, zoom

    def zoom_at(self, x: float, y: float, factor: float) -> None:
        """Zoom by factor, keeping the world point under screen (x, y) fixed."""
        self.center, self.zoom = self.zoom_at_params(x, y, factor)

    def fit_params(
        self, x: float, y: float, w: float, h: float, margin: float = 0.95
    ) -> tuple[np.ndarray, float]:
        """Center and zoom that frame the world rect (x, y, w, h).

        Not limited by min_zoom, so very large content can always be fitted.
        """
        zoom = min(margin * min(self.viewport[0] / w, self.viewport[1] / h), self.max_zoom)
        return np.array([x + w / 2, y + h / 2]), float(zoom)

    def fit(self, x: float, y: float, w: float, h: float, margin: float = 0.95) -> None:
        self.center, self.zoom = self.fit_params(x, y, w, h, margin)

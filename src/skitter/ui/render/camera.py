"""2D pan/zoom camera mapping world coordinates to the screen.

World coordinates follow image conventions: x to the right, y downward.
Screen coordinates are logical (device-independent) pixels.
"""

import numpy as np


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

    def zoom_at(self, x: float, y: float, factor: float) -> None:
        """Zoom by factor, keeping the world point under screen (x, y) fixed."""
        anchor = self.screen_to_world(x, y)
        self.zoom = float(np.clip(self.zoom * factor, self.min_zoom, self.max_zoom))
        self.center = anchor - (np.array([x, y]) - self.viewport / 2) / self.zoom

    def fit(self, x: float, y: float, w: float, h: float, margin: float = 0.95) -> None:
        """Center and zoom so the world rect (x, y, w, h) fills the viewport."""
        self.center = np.array([x + w / 2, y + h / 2])
        zoom = margin * min(self.viewport[0] / w, self.viewport[1] / h)
        self.zoom = float(np.clip(zoom, self.min_zoom, self.max_zoom))

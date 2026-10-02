"""Flying-tiles demo: exercises the GPU canvas with many animated tiles."""

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow

from skitter.core.easing import ease_in_cubic, ease_out_cubic, lerp, progress
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.render.sprites import SpriteLayer, make_instances

TILE = 32.0  # world units per tile
PERIOD = 6.0  # seconds per assemble / hold / scatter cycle


def _procedural_textures(count: int, px: int, rng: np.random.Generator) -> np.ndarray:
    """Colored tiles with a diagonal gradient and a dark border."""
    yy, xx = np.mgrid[0:px, 0:px] / (px - 1)
    gradient = 0.6 + 0.4 * (1 - (xx + yy) / 2)
    border = np.minimum.reduce([xx, yy, 1 - xx, 1 - yy]) < 0.06
    colors = rng.uniform(0.2, 1.0, (count, 1, 1, 3))
    textures = colors * gradient[None, :, :, None]
    textures[:, border] *= 0.35
    return (textures * 255).astype(np.uint8)


def flying_tiles(count: int, seed: int = 0):
    """Build a demo layer, its animation, and the world rect to frame."""
    rng = np.random.default_rng(seed)
    cols = math.ceil(math.sqrt(count * 16 / 10))
    rows = math.ceil(count / cols)
    width, height = cols * TILE, rows * TILE

    index = np.arange(count)
    grid = np.stack([(index % cols + 0.5) * TILE, (index // cols + 0.5) * TILE], axis=1)
    center = np.array([width / 2, height / 2])
    offset = grid - center
    dist = np.hypot(*offset.T) / np.hypot(*center)  # 0 at center, 1 at corners

    angle = rng.uniform(0, 2 * np.pi, count)
    radius = rng.uniform(1.0, 2.5, count) * max(width, height)
    scatter = center + radius[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    spin = rng.uniform(-3 * np.pi, 3 * np.pi, count)
    delay = 1.2 * dist + rng.uniform(0, 0.3, count)

    # Tint the assembled grid toward a sunset gradient so it reads as a picture.
    u, v = grid[:, 0] / width, grid[:, 1] / height
    target = np.stack([0.9 - 0.3 * v, 0.3 + 0.4 * u * (1 - v), 0.3 + 0.6 * v], axis=1)

    instances = make_instances(count)
    instances["layer"] = rng.integers(0, 64, count)
    instances["tint"][:, :3] = target
    instances["tint"][:, 3] = 0.55
    layer = SpriteLayer(_procedural_textures(64, 64, rng), instances)

    def animate(t: float) -> bool:
        c = t % PERIOD
        if c < 4.5:
            e = ease_out_cubic(progress(c, delay, 1.5))
        else:
            e = 1 - ease_in_cubic(progress(c, 4.5 + delay * 0.4, 1.0))
        ripple = 1 + 0.12 * e * np.sin(dist * 25 - t * 6)
        instances["pos"] = lerp(scatter, grid, e[:, None])
        instances["rotation"] = spin * (1 - e)
        instances["size"] = (TILE * lerp(0.3, 0.92, e) * ripple)[:, None]
        layer.mark_dirty()
        return True

    return layer, animate, (0.0, 0.0, width, height)


class DemoWindow(QMainWindow):
    """Standalone window running the flying-tiles demo with an fps readout."""

    def __init__(self, count: int):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Skitter Demo — {count:,} tiles")
        self.resize(1280, 800)

        self.canvas = MosaicCanvas(self)
        self.setCentralWidget(self.canvas)
        fps_label = QLabel()
        self.statusBar().addPermanentWidget(fps_label)
        self.canvas.fps_changed.connect(lambda fps: fps_label.setText(f"{fps:.0f} fps"))

        layer, animate, rect = flying_tiles(count)
        self.canvas.add_layer(layer)
        self.canvas.add_animation(animate)
        self.canvas.fit_to(*rect)

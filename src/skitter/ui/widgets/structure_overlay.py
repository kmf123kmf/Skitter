"""Draws what Contour Rows follows over the image: guide edges and the flow.

Edges found in the image show yellow, the mask's edge cyan, texture (where
tiles follow the grain) a magenta tint, and short white strokes, about one
per row, show which way tiles run (the flow). The
picture is built on the CPU from a StructureField (core/slicing/structure.py)
and shown as one sprite stretched over the region it was computed for.
"""

import numpy as np
from PIL import Image, ImageDraw

from skitter.core.slicing.structure import StructureField
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.render.sprites import SpriteLayer, make_instances

EDGE_COLOR = (255, 214, 10, 255)
MASK_COLOR = (40, 220, 255, 255)
FLOW_COLOR = (255, 255, 255, 200)
FLOW_SHADOW = (0, 0, 0, 140)
TEXTURE_TINT = (255, 60, 200, 70)
MAX_SIDE = 4096  # the picture's longest side, in pixels
STROKE = 0.7  # flow stroke length, in rows


def structure_image(field: StructureField, visible: np.ndarray) -> np.ndarray:
    """(H, W, 4) uint8 picture of a field: edges, mask edge, and flow strokes a row
    apart, at up to MAX_SIDE pixels (a whole number per sample)."""
    h, w = field.rows.shape
    zoom = max(1, min(MAX_SIDE // max(h, w), 4))
    picture = Image.new("RGBA", (w * zoom, h * zoom), (0, 0, 0, 0))
    draw = ImageDraw.Draw(picture)
    step = max(field.spacing, 2.0)
    half = 0.5 * STROKE * step
    width = max(1, round(0.12 * step * zoom))
    for y in np.arange(step / 2, h, step):
        for x in np.arange(step / 2, w, step):
            i, j = int(x), int(y)
            if not visible[j, i]:
                continue
            dx, dy = field.flow[:, j, i] * half
            line = [((x - dx) * zoom, (y - dy) * zoom), ((x + dx) * zoom, (y + dy) * zoom)]
            draw.line(line, fill=FLOW_SHADOW, width=width + 2)
            draw.line(line, fill=FLOW_COLOR, width=width)
    pixels = np.array(picture)
    if field.texture.any():  # under the strokes: tint only where nothing is drawn
        big = np.repeat(np.repeat(field.texture, zoom, axis=0), zoom, axis=1)
        pixels[big & (pixels[..., 3] == 0)] = TEXTURE_TINT
    mask_edge = field.guides & ~field.edges
    for where, color in ((field.edges, EDGE_COLOR), (mask_edge, MASK_COLOR)):
        if where.any():
            big = np.repeat(np.repeat(where, zoom, axis=0), zoom, axis=1)
            pixels[big] = color
    return pixels


class StructureOverlay:
    """Add after the region overlay so it draws on top of everything."""

    def __init__(self, canvas: MosaicCanvas):
        self.canvas = canvas
        self._layer: SpriteLayer | None = None

    @property
    def shown(self) -> bool:
        return self._layer is not None

    def show(self, field: StructureField, visible: np.ndarray, scale: float) -> None:
        """Show a field computed over the canvas from (0, 0), `scale` samples per unit."""
        self.hide()
        h, w = field.rows.shape
        instances = make_instances(1)
        instances["pos"] = (w / scale / 2, h / scale / 2)
        instances["size"] = (w / scale, h / scale)
        self._layer = self.canvas.add_layer(SpriteLayer(structure_image(field, visible), instances))
        self.canvas.update()

    def hide(self) -> None:
        if self._layer is not None:
            self.canvas.remove_layer(self._layer)
            self._layer = None
            self.canvas.update()

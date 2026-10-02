"""Draws a RegionSet over the image on the GPU, honoring stacking order.

Regions draw bottom to top. In "stacked" mode each region is filled with the
image pixels at its own position, so a region covers the outlines of lower
regions exactly where it overlaps them, while the image looks unchanged
inside covered areas. "outlines" mode draws only lines (every region's full
extent is visible, still stacked by z). Line widths are in screen pixels, so
outlines stay crisp at every zoom.
"""

import numpy as np

from skitter.core.slicing import RegionSet
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.render.sprites import SpriteLayer, make_instances

LINE_COLORS = (  # (id, label, rgb)
    ("amber", "Amber", (1.0, 0.78, 0.2)),  # reads well over most photos
    ("white", "White", (1.0, 1.0, 1.0)),
    ("black", "Black", (0.0, 0.0, 0.0)),
    ("red", "Red", (0.95, 0.15, 0.15)),
    ("magenta", "Magenta", (1.0, 0.2, 0.9)),
    ("lime", "Lime", (0.5, 1.0, 0.2)),
)
DEFAULT_LINE_COLOR = "amber"
LINE_COLOR = LINE_COLORS[0][2]
HIGHLIGHT_COLOR = (0.25, 0.85, 1.0)
LINE_PX = 1.25
EDGE_PX = 1.5

STACKED, OUTLINES = "stacked", "outlines"


def regions_to_instances(regions: RegionSet | None, color) -> np.ndarray:
    """Sprite instances for regions in stacking order (bottom first)."""
    if regions is None or not regions:
        return make_instances(0)
    order = regions.stacking_order()
    instances = make_instances(len(regions))
    instances["pos"] = regions.center[order]
    instances["size"] = regions.size[order]
    instances["rotation"] = regions.rotation[order]
    instances["tint"][:, :3] = color
    instances["tint"][:, 3] = 1.0
    return instances


class RegionOverlay:
    """Add after the image layer so it draws on top of the image."""

    def __init__(self, canvas: MosaicCanvas):
        self.canvas = canvas
        self.regions: RegionSet | None = None
        self.highlighted: int | None = None
        self.mode = STACKED
        self.line_color = LINE_COLOR
        self._layer = SpriteLayer(None, make_instances(0), outline_px=LINE_PX, edge_px=EDGE_PX)
        self._highlight = SpriteLayer(
            None, make_instances(0), outline_px=LINE_PX + 1, edge_px=EDGE_PX
        )
        canvas.add_layer(self._layer)
        canvas.add_layer(self._highlight)
        self._apply_style()

    def set_image(self, image_layer: SpriteLayer | None, width: float, height: float) -> None:
        """The image layer whose pixels fill regions in stacked mode."""
        self._layer.texture_from = image_layer
        self._layer.texture_size = (width, height)
        self._apply_style()

    def set_regions(self, regions: RegionSet | None) -> None:
        self.regions = regions
        self._layer.instances = regions_to_instances(regions, self.line_color)
        self._layer.mark_dirty()
        self.set_highlight(None)

    def set_line_color(self, color) -> None:
        self.line_color = tuple(color)
        self._layer.instances["tint"][:, :3] = self.line_color
        self._layer.mark_dirty()
        self.canvas.update()

    @property
    def line_alpha(self) -> float:
        return self._layer.line_alpha

    def set_line_alpha(self, alpha: float) -> None:
        """Opacity of the region outlines (the hover highlight stays opaque)."""
        self._layer.line_alpha = min(max(float(alpha), 0.0), 1.0)
        self.canvas.update()

    def set_highlight(self, index: int | None) -> None:
        """Outline one region's full extent on top of everything (None clears)."""
        if index is not None and (self.regions is None or not 0 <= index < len(self.regions)):
            index = None
        self.highlighted = index
        selected = None if index is None else self.regions[index : index + 1]
        self._highlight.instances = regions_to_instances(selected, HIGHLIGHT_COLOR)
        self._highlight.mark_dirty()
        self.canvas.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._apply_style()

    @property
    def visible(self) -> bool:
        return self._layer.visible

    def set_visible(self, visible: bool) -> None:
        self._layer.visible = self._highlight.visible = visible
        self.canvas.update()

    def _apply_style(self) -> None:
        layer = self._layer
        stacked = self.mode == STACKED and layer.texture_from is not None
        layer.project_texture = stacked
        # Outlines mode draws lines only: any fill would build up where regions overlap.
        layer.fill_alpha = 0.0
        self.canvas.update()

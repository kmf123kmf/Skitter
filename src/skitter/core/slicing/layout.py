"""Mosaic layout: the base tile and the size of the mosaic canvas.

The user chooses a base tile (width in mosaic pixels and an aspect ratio)
and the number of columns. The mosaic canvas is the final source image
scaled uniformly to columns x tile width; its height follows the source's
aspect ratio, so the canvas usually holds a fractional number of tile rows.
Slicing works in canvas (mosaic pixel) coordinates and tries to cover the
whole canvas; tiles at the edges may overhang it. Cropping or squaring off
the assembled mosaic is a later, separate choice.
"""

import math
from dataclasses import asdict, dataclass

TILE_ASPECTS = (
    (1.0, "Square (1:1)"),
    (4 / 3, "4:3"),
    (3 / 2, "3:2"),
    (16 / 9, "16:9"),
    (3 / 4, "3:4 (portrait)"),
    (2 / 3, "2:3 (portrait)"),
    (9 / 16, "9:16 (portrait)"),
)


@dataclass(frozen=True)
class MosaicLayout:
    tile_width: int = 80  # base tile width in mosaic pixels
    tile_aspect: float = 1.0  # base tile width / height
    columns: int = 40

    def __post_init__(self):
        if self.tile_width < 1 or self.columns < 1 or self.tile_aspect <= 0:
            raise ValueError(f"invalid layout {self}")

    @property
    def tile_height(self) -> float:
        return self.tile_width / self.tile_aspect

    @property
    def tile_size(self) -> tuple[float, float]:
        return (float(self.tile_width), self.tile_height)

    @property
    def canvas_width(self) -> int:
        return self.columns * self.tile_width

    def canvas_height(self, source_width: int, source_height: int) -> float:
        return self.canvas_width * source_height / source_width

    def canvas_size(self, source_width: int, source_height: int) -> tuple[float, float]:
        return (float(self.canvas_width), self.canvas_height(source_width, source_height))

    def scale(self, source_width: int) -> float:
        """Mosaic pixels per source pixel."""
        return self.canvas_width / source_width

    def rows(self, source_width: int, source_height: int) -> float:
        """Tile rows the canvas holds (usually fractional)."""
        return self.canvas_height(source_width, source_height) / self.tile_height

    def whole_rows(self, source_width: int, source_height: int) -> int:
        """Rows of whole tiles needed to cover the canvas."""
        return max(1, math.ceil(self.rows(source_width, source_height) - 1e-9))

    def replace(self, **changes) -> "MosaicLayout":
        return MosaicLayout(**{**asdict(self), **changes})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MosaicLayout":
        return cls(**data)

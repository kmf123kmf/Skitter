"""Mosaic layout: the shape of the base tile and how many fit across.

The user chooses a base tile shape (aspect ratio) and the number of
columns. Nothing before export has a size in pixels: slicing and matching
work in *mosaic units*, with the canvas being the final source image
scaled uniformly to CANVAS_WIDTH units across, whatever the layout; a base
tile is CANVAS_WIDTH / columns units wide. Its height follows the source's
aspect ratio, so the canvas usually holds a fractional number of tile rows.
Slicing tries to cover the whole canvas; tiles at the edges may overhang it.

Because the canvas keeps its size, a point in mosaic units stays at the
same place on the picture when the layout changes: what is kept by
position (camera keys, say) needs no converting when the mosaic is sliced
again with other columns or another tile shape.

Pixels are chosen only when the mosaic is exported (see core/assembly.py),
which also decides whether overhanging tiles are trimmed.
"""

import math
from dataclasses import asdict, dataclass

CANVAS_WIDTH = 4000.0  # mosaic units across the canvas (base tiles of 100 at 40 columns)

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
    tile_aspect: float = 1.0  # base tile width / height
    columns: int = 40

    def __post_init__(self):
        if self.columns < 1 or self.tile_aspect <= 0:
            raise ValueError(f"invalid layout {self}")

    @property
    def tile_width(self) -> float:
        """A base tile's width in mosaic units."""
        return CANVAS_WIDTH / self.columns

    def rows(self, source_width: int, source_height: int) -> float:
        """Tile rows the canvas holds (usually fractional)."""
        return self.columns * self.tile_aspect * source_height / source_width

    def whole_rows(self, source_width: int, source_height: int) -> int:
        """Rows of whole tiles needed to cover the canvas."""
        return max(1, math.ceil(self.rows(source_width, source_height) - 1e-9))

    def source_per_tile(self, source_width: int) -> float:
        """Source pixels across one base tile."""
        return source_width / self.columns

    def replace(self, **changes) -> "MosaicLayout":
        return MosaicLayout(**{**asdict(self), **changes})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MosaicLayout":
        return cls(**data)

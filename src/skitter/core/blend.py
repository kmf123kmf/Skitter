"""Blending: mixing the picture into the matched tiles.

With a small library the closest photo can still be far from the picture
in places. Blending mixes the picture into each tile: a tile showing
(1 - a) tile + a picture (per pixel, in OKLab) differs from the picture by
(1 - a) times what the tile alone does. Every candidate's error shrinks by
the same factor, so blending never changes which tile matching would
choose: it is applied after matching, to the finished mosaic, and changing
it needs no new match.

Each tile carries its own amount and its own piece of the picture (the
footprint it covers in the finished mosaic, scene.picture_uv), so the
blend moves with the tile in animations. "Weak matches" blends only tiles
whose error is above a target, each just enough to bring it down to the
target (up to the amount); tiles already within it stay pure photos.
"""

import numpy as np

from skitter.core.slicing.params import ChoiceParam, Configurable, FloatParam

WHERE = (("weak", "Weak matches"), ("all", "Every tile"))


class BlendSettings(Configurable):
    """How much of the picture shows through the tiles."""

    amount = FloatParam(
        0.0, "Blend", min=0.0, max=100.0, step=5.0, decimals=0, suffix="%",
        help="Mix the picture into the tiles, so the mosaic reads better with fewer "
             "photos. 0%: tiles show only their photos. Changing it needs no new match.",
    )  # fmt: skip
    where = ChoiceParam(
        "weak", "Where", choices=WHERE, under="amount", when=lambda s: s.amount > 0,
        help="Weak matches: only tiles far from the picture blend, just enough to come "
             "within the target (at most the Blend amount). Every tile: all blend by the "
             "Blend amount.",
    )  # fmt: skip
    target = FloatParam(
        3.0, "Target error", min=0.5, max=50.0, step=0.5, decimals=1, suffix=" ΔE",
        under="where", when=lambda s: s.amount > 0 and s.where == "weak",
        help="Tiles that differ from the picture by at most this much (seen from a "
             "distance, as in the Score) stay pure photos. About 2 is barely visible.",
    )  # fmt: skip

    @property
    def active(self) -> bool:
        return self.amount > 0


def tile_blend(error: np.ndarray, settings: BlendSettings | None) -> np.ndarray:
    """(N,) share of the picture each tile shows, from its matching error (ΔE; nan:
    unknown, left pure)."""
    error = np.asarray(error, dtype=np.float64)
    if settings is None or not settings.active:
        return np.zeros(len(error), np.float32)
    amount = settings.amount / 100.0
    if settings.where == "all":
        return np.full(len(error), amount, np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        needed = 1.0 - settings.target / error
    needed = np.where(np.isfinite(needed), needed, 0.0)
    return np.clip(needed, 0.0, amount).astype(np.float32)

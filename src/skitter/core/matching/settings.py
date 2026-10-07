"""Matching settings, declared as Params so the UI form is generated."""

from skitter.core.slicing.params import BoolParam, ChoiceParam, Configurable, FloatParam, IntParam
from skitter.core.tiles.descriptors import dim_weights

TINT_PRESETS = {"none": 0.0, "subtle": 0.2}
MAX_TINT = 0.6


class MatchSettings(Configurable):
    tint = ChoiceParam(
        "subtle", "Tinting",
        choices=[("none", "None (purist)"), ("subtle", "Subtle"), ("custom", "Custom")],
        help="Shift each tile's average color toward its region's. Tiles are never blended "
             "with the source image; only their average color moves.",
    )  # fmt: skip
    tint_strength = FloatParam(
        0.35, "Tint strength", min=0.0, max=MAX_TINT, step=0.05,
        when=lambda s: s.tint == "custom",
        help="0 leaves tiles untouched; higher values let structure matter more than color.",
    )  # fmt: skip
    max_uses = IntParam(
        3, "Max uses per tile", min=0, max=100_000,
        help="How many times one tile image may appear (0: no limit).",
    )  # fmt: skip
    min_spacing = FloatParam(
        4.0, "Repeat spacing", min=0.0, max=1000.0, step=0.5, decimals=1, suffix=" × tile",
        help="Minimum distance between two uses of the same tile, in base tiles (0: off).",
    )  # fmt: skip
    crops = IntParam(
        3, "Crops per tile", min=1, max=7,
        help="Most crop windows tried along a tile that is longer than the region shape.",
    )  # fmt: skip
    mirror = BoolParam(False, "Mirrored tiles", help="Also try every crop flipped left to right.")
    color_weight = FloatParam(1.0, "Color weight", min=0.0, max=5.0, step=0.1)
    structure_weight = FloatParam(
        1.0, "Structure weight", min=0.0, max=5.0, step=0.1,
        help="How much the arrangement of light and color inside a region counts.",
    )  # fmt: skip
    texture_weight = FloatParam(
        0.5, "Texture weight", min=0.0, max=5.0, step=0.1,
        help="How much busy versus flat areas count.",
    )  # fmt: skip
    crop_penalty = FloatParam(
        0.25, "Crop penalty", min=0.0, max=1.0, step=0.05,
        help="Preference for tiles whose own shape fits the region (less cropped away).",
    )  # fmt: skip
    candidates = IntParam(
        48, "Candidates per region", min=4, max=256,
        help="Nearest tiles kept per region for reranking and the reuse rules.",
    )  # fmt: skip
    search_effort = IntParam(
        16, "Search effort", min=1, max=65_536,
        help="Index cells searched per region at first (faiss nprobe). Raised automatically "
             "while the sampled search error is too high and more effort clearly helps, up "
             "to every cell of the index.",
    )  # fmt: skip
    error_diffusion = BoolParam(
        False, "Error diffusion",
        help="Pass each region's leftover color error on to its neighbors, so large areas "
             "keep their color even with a limited library.",
    )  # fmt: skip
    refine_seconds = FloatParam(
        5.0, "Refinement time", min=0.0, max=600.0, step=1.0, decimals=0, suffix=" s",
        help="Time spent improving the assignment by moving and swapping tiles.",
    )  # fmt: skip
    adaptive_rounds = IntParam(
        2, "Adaptive passes", min=0, max=10,
        help="Passes that search harder for the regions the quality check finds worst.",
    )  # fmt: skip

    @property
    def tint_value(self) -> float:
        return self.tint_strength if self.tint == "custom" else TINT_PRESETS[self.tint]

    def weights(self):
        return dim_weights(self.color_weight, self.structure_weight, self.texture_weight)

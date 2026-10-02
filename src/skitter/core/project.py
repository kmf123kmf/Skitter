"""Project data: everything chosen across the mosaic generation steps."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from skitter.core.assembly import ExportSettings
from skitter.core.edits import Edit
from skitter.core.matching.matcher import MatchResult
from skitter.core.matching.settings import MatchSettings
from skitter.core.slicing import MosaicLayout, RegionSet, SlicingPlan


@dataclass
class Project:
    source_path: Path | None = None
    source_original: np.ndarray | None = None  # (H, W, 3) uint8 RGB, as loaded
    source_edits: list[Edit] = field(default_factory=list)
    source_image: np.ndarray | None = None  # original with edits applied
    # Read-only snapshot of source_image taken when the user finishes the Source
    # step. Every later step works from this, not from the editable image.
    source_final: np.ndarray | None = None

    layout: MosaicLayout = field(default_factory=MosaicLayout)  # base tile and columns
    slicing_plan: SlicingPlan = field(default_factory=SlicingPlan.default)
    regions: RegionSet | None = None  # slicing_plan evaluated on source_final, mosaic units

    match_settings: MatchSettings = field(default_factory=MatchSettings)
    matches: MatchResult | None = None  # a tile for each region (None after re-slicing)

    export_settings: ExportSettings = field(default_factory=ExportSettings)

    @property
    def has_source(self) -> bool:
        return self.source_image is not None

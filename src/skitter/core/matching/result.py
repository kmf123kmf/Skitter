"""What matching produces: the result, and looks at a run in progress."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from skitter.core.matching.candidates import Candidates
from skitter.core.matching.quality import QualityReport
from skitter.core.matching.settings import MatchSettings
from skitter.core.slicing import RegionSet

Progress = Callable[[str, float | None], None]


class MatchCancelled(Exception):
    pass


class PreviewStage(Enum):
    """What a preview's tiles are, in the order a run reaches them."""

    SKETCH = "Target colors (no tiles yet)"
    BEST = "Best tiles, before reuse rules"  # each region's own best: repeats unlimited
    ASSIGNED = "Assigned, some regions waiting"  # reuse rules left these without a tile
    # (waiting regions show their favorite, the BEST tile, until widening replaces it)
    WIDENING = "Widening: tentative tiles"  # waiting regions: cheapest tile allowed so far
    COMPLETE = "Every region has a tile"
    REFINING = "Refining: swapping tiles"
    ADAPTIVE = "Adaptive pass: worst regions"  # searched again, harder


@dataclass(frozen=True)
class RegretStats:
    samples: int
    mean: float  # mean extra cost of the approximate best over the exact best
    p95: float
    relative: float  # mean regret / median exact cost
    effort: int  # search effort (nprobe) finally used
    exact: bool  # the index searched exactly (small library)


@dataclass
class MatchResult:
    regions: RegionSet
    tile: np.ndarray  # (R,) library slot, -1 for regions without a tile (hidden)
    rect: np.ndarray  # (R, 4) crop window, fractions of the tile image
    mirrored: np.ndarray  # (R,) bool
    cost: np.ndarray  # (R,) matching cost (inf: none)
    tile_mean: np.ndarray  # (R, 3) OKLab average of the chosen crop
    tint_target: np.ndarray  # (R, 3) OKLab color each tile is tinted toward
    tint: float
    quality: QualityReport
    regret: list[RegretStats]
    stats: dict = field(default_factory=dict)
    # For manual picks (None in results made without them):
    candidates: Candidates | None = None
    choice: np.ndarray | None = None  # (R,) int32 shown candidate, index into its class's set
    manual: np.ndarray | None = None  # (R,) bool: picked by hand
    settings: MatchSettings | None = None  # the settings matched with

    @property
    def manual_count(self) -> int:
        return 0 if self.manual is None else int(self.manual.sum())

    def tinted_mean(self) -> np.ndarray:
        return self.tile_mean + self.tint * (self.tint_target - self.tile_mean)


@dataclass(frozen=True)
class MatchPreview:
    """A look at a run in progress."""

    stage: PreviewStage
    regions: RegionSet
    target: np.ndarray  # (R, 3) sRGB (0..1) each region aims for; nan: hidden, no tile needed
    result: MatchResult | None = None  # tiles so far (tile -1: none yet; quality None)

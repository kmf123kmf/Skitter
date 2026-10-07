"""The matching pipeline: regions + tile library + settings -> a mosaic.

See the package docstring for the steps; run.py does them (MatchRun, one
method per phase) and result.py has what comes out. `Matcher` keeps candidate sets and
search indexes between runs (keyed by library version and settings), so
changing reuse rules or tint reuses them; changing the crop settings or the
library rebuilds only what depends on them.

Detail levels: an index ranks with fixed weights, but regions too small for
some structure blocks (or hidden where they lie) leave them out of their
cost. Searching them with the full index ranks tiles on detail the cost
ignores, and no search effort fixes that. So regions are searched per shape
class and detail level (targets.detail_levels), each level with its own
index weighted like its cost (indexing only the dimensions it uses).

Good-enough search: before searching the regions of a shape and level, a
sample of them is also searched exactly (every candidate). If the
approximate results cost noticeably more (relative regret above
REGRET_LIMIT), search effort is raised (x4) and the sample checked again,
as long as that clearly helps (REGRET_GAIN), up to every cell of the index.
After assignment, adaptive
passes search harder only for the regions the quality check finds worst,
keeping a pass only if the overall score improves and it places no more
regions against the reuse rules (the score doesn't see repetition).

The result remembers each region's best few candidates (candidates.py) for
manual picks. Pins (manual picks kept from an earlier run) are placed
first and never moved; everything else is matched around them.

Previews: `run(preview=...)` is handed a MatchPreview at each point where
the run has something to show (see PreviewStage), copied so it stays valid
while the run goes on. Previews only read the run's state: results are the
same with or without them. `run(detail=...)` reports progress within the
long steps (building candidates and indexes, searching, widening), and
cancelling stops those too.

Widening: after the first assignment, regions whose candidates the reuse
rules used up are searched again. Such regions usually share their favorite
photo (a large plain area: every region wants the same few photos), so they
are grouped by it. Each group is searched once, from its average target,
skipping photos at their use limit, as deep as the group needs (about
WIDEN_SLACK times the photos it must use, counting each photo's crops, up to
WIDEN_MAX candidates). Each member
gets an interleaved slice of that list (members m apart share none), reranked
for its own target, so a plain area spreads over many photos instead of
exhausting the same ones again. Adaptive passes search their worst regions
again the same way.

Searches run in batches of regions sized to take about SEARCH_SECONDS each,
in a shuffled order so previews fill in all over the mosaic. Each region's
search is independent, so batching never changes results. Small batches cost
throughput, more inside the app than in a plain script (measured at search
effort 1024, 300,000 photos: 512 regions +47%, 1,024 +16%, 2,048 +9%,
4,096 none), hence MIN_BATCH. The tuning constants are in run.py.
"""

from collections.abc import Callable

import numpy as np

from skitter.core.matching.candidates import Pins
from skitter.core.matching.index import CandidateSet, SearchIndex, build_candidates
from skitter.core.matching.result import (
    MatchCancelled,
    MatchPreview,
    MatchResult,
    PreviewStage,
    Progress,
    RegretStats,
)
from skitter.core.matching.run import MatchRun, Reporter
from skitter.core.matching.settings import MatchSettings
from skitter.core.slicing import RegionSet, SliceContext
from skitter.core.tiles.library import TileLibrary

__all__ = [
    "MatchCancelled",
    "MatchPreview",
    "MatchResult",
    "Matcher",
    "PreviewStage",
    "Progress",
    "RegretStats",
]

TINT_REBUILD = 0.15  # rebuild an index when the tint moves this far from its reference


class Matcher:
    def __init__(self, library: TileLibrary):
        self.library = library
        self._candidates: dict[tuple, CandidateSet] = {}
        self._indexes: dict[tuple, SearchIndex] = {}

    def candidates(
        self, aspect: float, settings: MatchSettings, progress=lambda message, fraction: None
    ) -> tuple[tuple, CandidateSet]:
        key = (self.library.version, round(float(aspect), 4), settings.crops, settings.mirror)
        if key not in self._candidates:
            cands = build_candidates(self.library, aspect, settings.crops, settings.mirror,
                                     progress)  # fmt: skip
            self._candidates = {k: v for k, v in self._candidates.items() if k[0] == key[0]}
            self._candidates[key] = cands
        return key, self._candidates[key]

    def index(
        self, key: tuple, cands: CandidateSet, settings: MatchSettings,
        progress=lambda message, fraction: None, weights=None,
    ) -> SearchIndex:  # fmt: skip
        """The search index over cands at the settings' weights (or the given ones, for
        a detail level; see targets.detail_levels), cached."""
        weights = settings.weights() if weights is None else np.asarray(weights)
        full = (key, tuple(np.round(weights, 6)))
        index = self._indexes.get(full)
        if index is None or abs(index.tint_ref - settings.tint_value) > TINT_REBUILD:
            index = SearchIndex(cands, weights, settings.tint_value, progress=progress)
            self._indexes = {k: v for k, v in self._indexes.items() if k[0][0] == key[0]}
            self._indexes[full] = index
        return index

    def run(
        self,
        regions: RegionSet,
        ctx: SliceContext,
        settings: MatchSettings,
        progress: Progress = lambda message, fraction: None,
        cancelled: Callable[[], bool] = lambda: False,
        pins: Pins | None = None,
        preview: Callable[[MatchPreview], None] | None = None,
        detail: Progress | None = None,
    ) -> MatchResult:
        """Match tiles to regions: progress(message, fraction) at each step, detail(...)
        within long steps, preview(MatchPreview) whenever there is something to show;
        cancelled() stops it (MatchCancelled). Pins are kept in place."""
        run = MatchRun(self, regions, ctx, settings, pins,
                       Reporter(progress, cancelled, detail, preview))  # fmt: skip
        run.analyze()
        run.search()
        run.pin()
        run.assign()
        run.refine()
        run.adapt()
        return run.result()

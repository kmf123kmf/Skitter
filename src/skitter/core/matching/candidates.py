"""Each region's ranked candidates, remembered after matching for manual picks.

Matching ranks a few dozen nearest crops per region by exact cost and then
keeps only the one it assigns. `Candidates` keeps the best KEEP of each list
(indices into the per-shape candidate sets, which matching has in memory
anyway), so a user can later pick another tile for a region. About 8 bytes
per remembered candidate.

`Pins` are manual picks carried into a new matching run: the matcher places
them first, counts them toward the reuse rules, and fits everything else
around them.
"""

from dataclasses import dataclass, field

import numpy as np

from skitter.core.matching.index import CandidateSet
from skitter.core.tiles.descriptors import DIM, MEAN

KEEP = 32  # candidates remembered per region


@dataclass(eq=False)
class Candidates:
    sets: tuple[CandidateSet | None, ...]  # crop candidates of each shape class
    cls: np.ndarray  # (R,) int16 shape class of each region
    ref: np.ndarray  # (R, KEEP) int32 ranked candidates, indices into their class's set (-1: none)
    cost: np.ndarray  # (R, KEEP) float32 their exact costs, ascending
    auto: np.ndarray  # (R,) int32 the candidate matching chose (-1: none)
    auto_cost: np.ndarray  # (R,) float32
    # More candidates found later for single regions ("Find more"), after the stored ones.
    more: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def set_of(self, region: int) -> CandidateSet | None:
        c = int(self.cls[region])
        return self.sets[c] if 0 <= c < len(self.sets) else None

    def ranked(self, region: int) -> tuple[np.ndarray, np.ndarray]:
        """(refs, costs) of a region's remembered candidates, best first."""
        ok = self.ref[region] >= 0
        refs, costs = self.ref[region][ok], self.cost[region][ok]
        if region in self.more:
            extra, extra_cost = self.more[region]
            refs, costs = np.concatenate([refs, extra]), np.concatenate([costs, extra_cost])
        return refs.astype(np.int64), costs.astype(np.float32)

    def crops(self, region: int, refs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(slot, rect, mirrored, OKLab mean) of candidates of one region."""
        s = self.set_of(region)
        refs = np.asarray(refs, dtype=np.int64)
        return (s.tile[refs], s.rect[refs], s.mirrored[refs],
                s.desc[refs][:, MEAN].astype(np.float32))  # fmt: skip

    def desc(self, regions, refs) -> np.ndarray:
        """(n, DIM) descriptors of the given candidates of the given regions (-1: zeros)."""
        regions = np.asarray(regions, dtype=np.int64)
        refs = np.asarray(refs, dtype=np.int64)
        out = np.zeros((len(regions), DIM), np.float32)
        cls = self.cls[regions]
        for c in np.unique(cls):
            s = self.sets[c] if 0 <= c < len(self.sets) else None
            rows = np.flatnonzero((cls == c) & (refs >= 0))
            if s is not None and len(rows):
                out[rows] = s.desc[refs[rows]].astype(np.float32)
        return out

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.cls, self.ref, self.cost, self.auto, self.auto_cost))


@dataclass(frozen=True)
class Pins:
    """Manual picks to keep when matching again: region r shows this crop."""

    region: np.ndarray  # (P,) int64
    tile: np.ndarray  # (P,) library slot
    rect: np.ndarray  # (P, 4) crop window
    mirrored: np.ndarray  # (P,) bool

    def __len__(self) -> int:
        return len(self.region)

    def subset(self, keep) -> "Pins":
        keep = np.asarray(keep)
        return Pins(self.region[keep], self.tile[keep], self.rect[keep], self.mirrored[keep])

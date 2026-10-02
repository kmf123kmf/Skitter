"""Candidate crops, their nearest-neighbor index, and exact reranking.

Per aspect class, every tile contributes a few crop windows (crops.py),
optionally mirrored. Their descriptors are indexed with faiss: an exact flat
index for small libraries, an inverted-file index with 8-bit scalar
quantization (IVF-SQ8) for large ones. Search returns each region's top
candidates approximately; `rerank` then computes their exact cost with the
current weights, tint and per-region masks.

Vectors are scaled so that plain L2 distance equals the weighted cost:
each dimension by sqrt(weight), the average color additionally by
(1 - t_ref) for a reference tint t_ref. The index stays useful as the tint
changes, since reranking always uses the real tint.

`exact_best` searches every candidate exactly for a sample of regions; the
gap between that and the approximate result (regret) tells whether the
approximate search is good enough.
"""

import math
from dataclasses import dataclass

import faiss
import numpy as np

from skitter.core.tiles.crops import crop_candidates
from skitter.core.tiles.descriptors import DIM, MEAN, mirror, tile_descriptors
from skitter.core.tiles.library import TileLibrary

EXACT_LIMIT = 50_000  # candidates up to which a flat (exact) index is used
CROP_PENALTY_SCALE = 0.01  # cost of keeping none of a photo at crop_penalty 1


@dataclass(frozen=True)
class CandidateSet:
    """Crop candidates of every tile for one aspect class."""

    aspect: float
    tile: np.ndarray  # (M,) library slot
    rect: np.ndarray  # (M, 4) float32 crop window, fractions of the image
    mirrored: np.ndarray  # (M,) bool
    retained: np.ndarray  # (M,) float32 share of the photo kept
    desc: np.ndarray  # (M, DIM) float16

    def __len__(self) -> int:
        return len(self.tile)


def build_candidates(
    library: TileLibrary, aspect: float, crops: int = 3, mirrored: bool = False
) -> CandidateSet:
    ids = library.ids
    c = crop_candidates(library.width[ids], library.height[ids], aspect, max_crops=crops)
    tile = ids[c.tile]
    desc = tile_descriptors(library.thumbs, library.thumb_size, tile, c.rect)
    flags = np.zeros(len(tile), bool)
    rect, retained = c.rect, c.retained
    if mirrored:
        tile, rect, retained = (np.concatenate([a, a]) for a in (tile, rect, retained))
        desc = np.concatenate([desc, mirror(desc)])
        flags = np.concatenate([flags, ~flags])
    return CandidateSet(aspect, tile, rect, flags, retained, desc.astype(np.float16))


def scaled(desc: np.ndarray, weights: np.ndarray, tint: float) -> np.ndarray:
    """Vectors whose L2 distances are weighted costs at the given tint."""
    scale = np.sqrt(np.asarray(weights, dtype=np.float64))
    scale[MEAN] *= 1.0 - tint
    return np.ascontiguousarray(np.asarray(desc, dtype=np.float32) * scale.astype(np.float32))


class SearchIndex:
    """faiss index over one CandidateSet at reference weights and tint."""

    def __init__(self, cands: CandidateSet, weights: np.ndarray, tint_ref: float, chunk=262_144):
        self.cands = cands
        self.weights = np.asarray(weights, dtype=np.float64)
        self.tint_ref = tint_ref
        m = len(cands)
        if m <= EXACT_LIMIT:
            self.index = faiss.IndexFlatL2(DIM)
            self.exact = True
        else:
            nlist = int(np.clip(4 * math.sqrt(m), 64, 65_536))
            self.index = faiss.index_factory(DIM, f"IVF{nlist},SQ8")
            sample = np.random.default_rng(0).choice(m, min(m, 64 * nlist, 300_000), False)
            self.index.train(scaled(cands.desc[np.sort(sample)], self.weights, tint_ref))
            self.exact = False
        for lo in range(0, m, chunk):
            self.index.add(scaled(cands.desc[lo : lo + chunk], self.weights, tint_ref))

    def search(self, desc: np.ndarray, k: int, nprobe: int = 16) -> np.ndarray:
        """(R, k) candidate indices nearest to target descriptors (-1 where fewer exist)."""
        if not self.exact:
            faiss.extract_index_ivf(self.index).nprobe = nprobe
        k = min(k, len(self.cands))
        if not len(desc) or not k:
            return np.full((len(desc), k), -1, np.int64)
        _, ids = self.index.search(scaled(desc, self.weights, self.tint_ref), k)
        return ids


def region_weights(weights, mask, tint: float) -> np.ndarray:
    """(R, DIM) exact cost weights: dimension weights x region masks, tinted mean."""
    w = np.asarray(weights, dtype=np.float32) * mask
    w[:, MEAN] *= (1.0 - tint) ** 2
    return w


def rerank(
    cands: CandidateSet,
    ids: np.ndarray,
    desc: np.ndarray,
    weights: np.ndarray,
    crop_penalty: float,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact costs of each region's candidates, sorted best first.

    ids: (R, K) candidate indices (-1: none). desc: (R, DIM) targets.
    weights: (R, DIM) from region_weights. Returns (ids, costs) reordered.
    """
    ids = np.asarray(ids, dtype=np.int64)
    costs = np.full(ids.shape, np.inf, np.float32)
    penalty = crop_penalty * CROP_PENALTY_SCALE * (1.0 - cands.retained)
    for lo in range(0, len(ids), chunk):
        sl = slice(lo, lo + chunk)
        idx = ids[sl]
        valid = idx >= 0
        safe = np.where(valid, idx, 0)
        x = cands.desc[safe].astype(np.float32)  # (r, K, DIM)
        diff = x - desc[sl, None, :]
        c = np.einsum("rkd,rkd,rd->rk", diff, diff, weights[sl]) + penalty[safe]
        costs[sl] = np.where(valid, c, np.inf)
    order = np.argsort(costs, axis=1, kind="stable")
    return np.take_along_axis(ids, order, 1), np.take_along_axis(costs, order, 1)


def exact_best(
    cands: CandidateSet,
    desc: np.ndarray,
    weights: np.ndarray,
    crop_penalty: float,
    chunk: int = 131_072,
) -> tuple[np.ndarray, np.ndarray]:
    """Best candidate and cost for each region, searching every candidate exactly.

    Uses sum_d w (x - q)^2 = (x^2) @ w - 2 x @ (w q) + sum_d w q^2, so the
    work is two matrix products per chunk of candidates.
    """
    q = np.asarray(desc, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    wq = (w * q).T
    const = (w * q * q).sum(axis=1)
    penalty = crop_penalty * CROP_PENALTY_SCALE * (1.0 - cands.retained)
    best = np.full(len(q), -1, np.int64)
    best_cost = np.full(len(q), np.inf, np.float32)
    for lo in range(0, len(cands), chunk):
        x = cands.desc[lo : lo + chunk].astype(np.float32)
        cost = (x * x) @ w.T - 2 * (x @ wq) + const + penalty[lo : lo + chunk, None]
        i = cost.argmin(axis=0)
        c = cost[i, np.arange(len(q))]
        better = c < best_cost
        best[better], best_cost[better] = lo + i[better], c[better]
    return best, best_cost

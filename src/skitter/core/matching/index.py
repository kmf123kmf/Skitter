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

Building candidates and indexes takes a while for big libraries (training
the IVF clusters most of all), so both report progress as they go: an
optional `progress(message, fraction)` that may raise to stop the work.
Clusters are trained one k-means round at a time, which gives exactly the
centroids of faiss's own training (tests check it).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import faiss
import numpy as np

from skitter.core.tiles.crops import crop_candidates
from skitter.core.tiles.descriptors import DIM, MEAN, mirror, tile_descriptors
from skitter.core.tiles.library import TileLibrary

EXACT_LIMIT = 50_000  # candidates up to which a flat (exact) index is used
CROP_PENALTY_SCALE = 0.01  # cost of keeping none of a photo at crop_penalty 1

Progress = Callable[[str, float], None]  # (what is being done, fraction of it done)


def _quiet(message: str, fraction: float) -> None:
    pass


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

    def find(self, tile, rect, mirrored) -> np.ndarray:
        """Index of each given crop in this set (-1: not in it)."""
        tile = np.asarray(tile, dtype=np.int64)
        out = np.full(len(tile), -1, np.int64)
        if not len(tile) or not len(self):
            return out
        by_tile = np.argsort(self.tile, kind="stable")
        sorted_tile = self.tile[by_tile]
        for i, (t, r, m) in enumerate(zip(tile, np.asarray(rect), np.asarray(mirrored, bool),
                                          strict=True)):  # fmt: skip
            lo, hi = np.searchsorted(sorted_tile, t), np.searchsorted(sorted_tile, t, side="right")
            same = by_tile[lo:hi]
            near = np.abs(self.rect[same] - r).max(axis=1) < 1e-4
            hit = same[near & (self.mirrored[same] == m)]
            if len(hit):
                out[i] = hit.min()
        return out

    def with_crops(
        self, library: TileLibrary, tile, rect, mirrored
    ) -> tuple["CandidateSet", np.ndarray]:
        """This set plus the given crops (those it lacks), and each crop's index.

        Used for manual picks kept across matching runs: a crop picked under
        other settings (more crops, mirroring) may not be in this set.
        """
        tile = np.asarray(tile, dtype=np.int64)
        rect = np.asarray(rect, dtype=np.float32).reshape(-1, 4)
        mirrored = np.asarray(mirrored, bool)
        index = self.find(tile, rect, mirrored)
        new = np.flatnonzero(index < 0)
        if not len(new):
            return self, index
        order = new[np.argsort(tile[new], kind="stable")]  # tile_descriptors wants tile order
        desc = tile_descriptors(library.thumbs, library.thumb_size, tile[order], rect[order])
        desc[mirrored[order]] = mirror(desc[mirrored[order]])
        retained = np.prod(rect[order, 2:] - rect[order, :2], axis=1)
        index[order] = len(self) + np.arange(len(order))
        extended = CandidateSet(
            self.aspect,
            np.concatenate([self.tile, tile[order]]),
            np.concatenate([self.rect, rect[order]]),
            np.concatenate([self.mirrored, mirrored[order]]),
            np.concatenate([self.retained, retained.astype(np.float32)]),
            np.concatenate([self.desc, desc.astype(np.float16)]),
        )
        return extended, index


def build_candidates(
    library: TileLibrary, aspect: float, crops: int = 3, mirrored: bool = False,
    progress: Progress = _quiet,
) -> CandidateSet:  # fmt: skip
    ids = library.ids
    c = crop_candidates(library.width[ids], library.height[ids], aspect, max_crops=crops)
    tile = ids[c.tile]

    def described(done: int, total: int) -> None:
        progress(f"Describing tile crops: {done:,} of {total:,}", done / max(total, 1))

    described(0, len(tile))
    desc = tile_descriptors(library.thumbs, library.thumb_size, tile, c.rect, progress=described)
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

    def __init__(self, cands: CandidateSet, weights: np.ndarray, tint_ref: float, chunk=262_144,
                 progress: Progress = _quiet):  # fmt: skip
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
            train_ivf(self.index, scaled(cands.desc[np.sort(sample)], self.weights, tint_ref),
                      progress)  # fmt: skip
            self.exact = False
        for lo in range(0, m, chunk):
            progress(f"Adding tiles to the index: {lo:,} of {m:,}", lo / max(m, 1))
            self.index.add(scaled(cands.desc[lo : lo + chunk], self.weights, tint_ref))

    def search(self, desc: np.ndarray, k: int, nprobe: int = 16, exclude=None) -> np.ndarray:
        """(R, k) candidate indices nearest to target descriptors (-1 where fewer exist).

        exclude: candidate indices never to return (photos that can't be used anyway).
        """
        if not self.exact:
            faiss.extract_index_ivf(self.index).nprobe = nprobe
        k = min(k, len(self.cands))
        if not len(desc) or not k:
            return np.full((len(desc), k), -1, np.int64)
        x = scaled(desc, self.weights, self.tint_ref)
        if exclude is None or not len(exclude):
            return self.index.search(x, k)[1]
        batch = faiss.IDSelectorBatch(np.ascontiguousarray(exclude, dtype=np.int64))
        selector = faiss.IDSelectorNot(batch)  # keeps a pointer to batch: both stay alive here
        params = (faiss.SearchParameters(sel=selector) if self.exact
                  else faiss.SearchParametersIVF(sel=selector, nprobe=nprobe))  # fmt: skip
        return self.index.search(x, k, params=params)[1]


def train_ivf(index, x: np.ndarray, progress: Progress = _quiet) -> None:
    """index.train(x) for an untrained IVF index with a flat quantizer, one k-means
    round at a time (centroids carried from round to round), reporting each."""
    ivf = faiss.extract_index_ivf(index)
    params, nlist = ivf.cp, ivf.nlist
    centroids = None
    for i in range(params.niter):
        progress(f"Grouping similar tiles: round {i + 1} of {params.niter}", i / params.niter)
        clustering = faiss.Clustering(ivf.d, nlist, params)
        clustering.niter = 1
        if centroids is not None:  # start where the last round ended
            faiss.copy_array_to_vector(centroids.ravel(), clustering.centroids)
        clustering.train(x, faiss.IndexFlatL2(ivf.d))
        centroids = faiss.vector_to_array(clustering.centroids).reshape(nlist, ivf.d)
    progress("Grouping similar tiles: finishing", 1.0)
    ivf.quantizer.reset()
    ivf.quantizer.add(centroids)
    index.train(x)  # the quantizer is trained, so only the scalar quantizer trains here


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


def exact_top(
    cands: CandidateSet,
    desc: np.ndarray,
    weights: np.ndarray,
    crop_penalty: float,
    count: int,
    chunk: int = 131_072,
) -> tuple[np.ndarray, np.ndarray]:
    """The `count` cheapest candidates for one region, searching every candidate exactly.

    desc, weights: (DIM,) the region's target and cost weights (region_weights).
    Returns (ids, costs), best first.
    """
    q = np.asarray(desc, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    penalty = crop_penalty * CROP_PENALTY_SCALE * (1.0 - cands.retained)
    best_ids = np.zeros(0, np.int64)
    best_costs = np.zeros(0, np.float32)
    for lo in range(0, len(cands), chunk):
        x = cands.desc[lo : lo + chunk].astype(np.float32)
        cost = ((x - q) ** 2 @ w + penalty[lo : lo + chunk]).astype(np.float32)
        keep = min(count, len(cost))
        top = np.argpartition(cost, keep - 1)[:keep] if keep < len(cost) else np.arange(len(cost))
        best_ids = np.concatenate([best_ids, lo + top])
        best_costs = np.concatenate([best_costs, cost[top]])
        if len(best_ids) > count:
            keep = np.argpartition(best_costs, count - 1)[:count]
            best_ids, best_costs = best_ids[keep], best_costs[keep]
    order = np.argsort(best_costs, kind="stable")
    return best_ids[order], best_costs[order]

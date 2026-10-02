"""Benchmark matching at library scale with synthetic descriptors.

    python scripts/bench_matching.py --candidates 1000000 --regions 50000

Measures index build time and size, search speed and accuracy (recall@1 and
relative regret against exact search on a sample) for several search
efforts, reranking, and constrained assignment. Synthetic descriptors are
clustered like real photo collections (colors and structures recur), but
real libraries vary; treat the numbers as a guide.
"""

import argparse
import time

import faiss
import numpy as np

from skitter.core.matching.assign import Assignment
from skitter.core.matching.index import (
    CandidateSet,
    SearchIndex,
    exact_best,
    region_weights,
    rerank,
)
from skitter.core.tiles.descriptors import DIM, MEAN, dim_weights


def synthetic(n: int, rng, clusters: int = 2000) -> np.ndarray:
    """Clustered descriptors: a random mean color and structure per cluster, plus noise."""
    centers = np.zeros((clusters, DIM), np.float32)
    centers[:, MEAN] = rng.uniform([0.1, -0.15, -0.15], [0.95, 0.15, 0.15], (clusters, 3))
    centers[:, 3:] = rng.normal(0, 0.04, (clusters, DIM - 3))
    which = rng.integers(0, clusters, n)
    return centers[which] + rng.normal(0, 0.02, (n, DIM)).astype(np.float32)


def timed(label: str, fn):
    start = time.perf_counter()
    result = fn()
    print(f"  {label:<34} {time.perf_counter() - start:8.2f} s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=1_000_000)
    parser.add_argument("--regions", type=int, default=50_000)
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--efforts", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--sample", type=int, default=256)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    m, r = args.candidates, args.regions
    print(f"{m:,} candidates, {r:,} regions, {faiss.omp_get_max_threads()} threads")

    desc = timed("generate", lambda: synthetic(m, rng).astype(np.float16))
    tiles = np.arange(m) // 2  # two crops per tile image
    cands = CandidateSet(1.0, tiles, np.zeros((m, 4), np.float32), np.zeros(m, bool),
                         np.ones(m, np.float32), desc)  # fmt: skip
    targets = synthetic(r, rng)
    weights = region_weights(dim_weights(), np.ones((r, DIM), np.float32), 0.2)
    index = timed("build index", lambda: SearchIndex(cands, dim_weights(), 0.2))
    size = len(faiss.serialize_index(index.index)) / 1e6
    print(f"  index: {'exact' if index.exact else 'IVF-SQ8'}, {size:,.0f} MB; "
          f"descriptors {desc.nbytes / 1e6:,.0f} MB (float16)")  # fmt: skip

    sample = rng.choice(r, args.sample, replace=False)
    best, exact = timed(
        f"exact search ({args.sample} regions)",
        lambda: exact_best(cands, targets[sample], weights[sample], 0.0),
    )
    for effort in args.efforts:
        ids = timed(f"search nprobe={effort}", lambda e=effort: index.search(targets, args.k, e))
        found, costs = timed("  rerank", lambda i=ids: rerank(cands, i, targets, weights, 0.0))
        recall = float((found[sample, 0] == best).mean())
        regret = np.maximum(costs[sample, 0] - exact, 0).mean() / np.median(exact)
        print(f"    recall@1 {recall:.3f}, relative regret {regret:.4f}")

    a = Assignment(cands.tile[found], costs, rng.random((r, 2)) * 1e4, tiles[-1] + 1, 3, 200.0)
    unplaced = timed("assign (greedy)", lambda: a.greedy(np.arange(r)))
    saved = timed("refine (one pass)", lambda: a.refine_pass(np.arange(r), np.ones(r)))
    print(f"  unplaced {unplaced}, refinement saved {saved:.4f}, rules {a.check()}")


if __name__ == "__main__":
    main()

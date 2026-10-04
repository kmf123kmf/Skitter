"""Landing orders: which tile lands when, bottom first.

Overlapping tiles (a photo pile) must land bottom first: a tile that lands
after one above it would jump underneath on arrival. Two ways to get such an
order, for two kinds of ordering:

- An ordering with a direction (center outward, reading order, by
  lightness): `landing_order` turns each tile's preferred place into a valid
  order as close to it as it allows.
- A random ordering: `random_landing_order` draws a uniformly random valid
  order, so tiles land close together in space and time exactly as often as
  chance has it. Don't use `landing_order` with random preferences for this:
  it lands a tile held back by the one under it as soon as it may, right
  next to it, so random piles came out clumpy (a landing with another one
  nearby within 0.15 s 92% of the time, against 50% by chance).

Choreographies then choose when the k-th tile lands (for example, evenly
spaced). A new ordering must say which kind it is; tests check random
orderings against chance (tests/test_animation.py).
"""

import numpy as np
from numba import njit

from skitter.core.scene import MosaicScene

RANDOM_SWEEPS = 100  # Gibbs sweeps for random_landing_order (settled after about 20)


def random_landing_order(scene: MosaicScene, seed: int, sweeps: int = RANDOM_SWEEPS) -> np.ndarray:
    """(N,) a uniformly random landing order in which every tile lands after those
    it lies on (scene.overlaps): what a random ordering means under that rule.

    Each tile gets an independent uniform key in [0, 1) and tiles land in key
    order; conditioning the keys on lower < upper for every overlapping pair
    makes that order uniform over the valid orders. The conditioned keys are
    sampled by Gibbs sweeps: each key is redrawn uniformly between the largest
    key of the tiles under it and the smallest of the tiles on it (starting
    from stacking order, which is valid). Without overlaps the first sweep is
    already a plain random order.
    """
    n = len(scene)
    pairs = scene.overlaps
    by_upper = pairs[np.argsort(pairs[:, 1], kind="stable")]
    by_lower = pairs[np.argsort(pairs[:, 0], kind="stable")]
    keys = (np.arange(n) + 0.5) / max(n, 1)
    _conditioned_keys(
        keys, np.searchsorted(by_upper[:, 1], np.arange(n + 1)), by_upper[:, 0].copy(),
        np.searchsorted(by_lower[:, 0], np.arange(n + 1)), by_lower[:, 1].copy(),
        int(sweeps), int(seed),
    )  # fmt: skip
    rank = np.empty(n, np.int64)
    rank[np.argsort(keys, kind="stable")] = np.arange(n)
    return rank


@njit(cache=True, nogil=True)
def _conditioned_keys(keys, below_start, below, above_start, above, sweeps, seed):
    np.random.seed(seed)
    for _ in range(max(sweeps, 1)):
        for i in range(len(keys)):
            lo = 0.0
            for k in range(below_start[i], below_start[i + 1]):
                lo = max(lo, keys[below[k]])
            hi = 1.0
            for k in range(above_start[i], above_start[i + 1]):
                hi = min(hi, keys[above[k]])
            keys[i] = lo + np.random.random() * (hi - lo)


def landing_gap(n: int) -> int:
    """Landings between a tile and the next one allowed on top of it (see landing_order)."""
    return int(np.clip(n // 150, 3, 50))


def landing_order(scene: MosaicScene, preference, gap: int | None = None) -> np.ndarray:
    """(N,) each tile's place in the landing sequence (0 lands first).

    Tiles land by preference (lower values first; ties by stacking order),
    except that a tile never lands before every tile below it that it
    overlaps (scene.overlaps): at each step the most preferred tile whose
    support has fully landed goes next.

    For orderings with a direction; random ones use random_landing_order
    (see the module docstring).

    A tile whose last support has just landed waits `gap` more landings
    (default landing_gap) while any other tile is ready. Without that,
    landing one tile frees the tiles on it, which, held back, are usually
    the most preferred: the sequence would climb stacks in one spot after
    another (on a photo pile, the next tile landed on one of the last few
    about 70% of the time). With it, landings spread over the mosaic as
    they would without the overlap rule, keeping the preferred order.
    """
    n = len(scene)
    preference = np.asarray(preference, dtype=np.float64)
    pairs = scene.overlaps
    pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
    start = np.searchsorted(pairs[:, 0], np.arange(n + 1))  # tiles above each, by lower
    waiting = np.bincount(pairs[:, 1], minlength=n).astype(np.int64)
    gap = landing_gap(n) if gap is None else max(int(gap), 0)
    return _landing_order(preference, start, pairs[:, 1].copy(), waiting, gap)


@njit(cache=True, nogil=True)
def _landing_order(preference, start, above, waiting, gap):
    # A binary min-heap of ready tiles keyed by (preference, index), and a queue of
    # tiles freed recently (each with the step from which it may go).
    n = len(preference)
    heap = np.empty(n, np.int64)
    size = 0
    for i in range(n):
        if waiting[i] == 0:
            heap[size] = i
            size += 1
    for k in range(size // 2 - 1, -1, -1):
        _sift_down(heap, size, k, preference)
    cooling = np.empty(n, np.int64)
    eligible = np.empty(n, np.int64)
    first = last = 0
    rank = np.empty(n, np.int64)
    for step in range(n):
        while first < last and (eligible[first] <= step or size == 0):
            size = _push(heap, size, cooling[first], preference)
            first += 1
        tile = heap[0]
        size -= 1
        heap[0] = heap[size]
        _sift_down(heap, size, 0, preference)
        rank[tile] = step
        for k in range(start[tile], start[tile + 1]):
            upper = above[k]
            waiting[upper] -= 1
            if waiting[upper] == 0:  # its support has landed: ready after the gap
                if gap == 0:
                    size = _push(heap, size, upper, preference)
                else:
                    cooling[last] = upper
                    eligible[last] = step + 1 + gap
                    last += 1
    return rank


@njit(cache=True, nogil=True)
def _push(heap, size, tile, preference):
    heap[size] = tile
    j = size
    while j > 0:
        parent = (j - 1) // 2
        if not _before(heap[j], heap[parent], preference):
            break
        heap[j], heap[parent] = heap[parent], heap[j]
        j = parent
    return size + 1


@njit(cache=True, nogil=True, inline="always")
def _before(a, b, preference):
    return preference[a] < preference[b] or (preference[a] == preference[b] and a < b)


@njit(cache=True, nogil=True)
def _sift_down(heap, size, j, preference):
    while True:
        child = 2 * j + 1
        if child >= size:
            return
        if child + 1 < size and _before(heap[child + 1], heap[child], preference):
            child += 1
        if not _before(heap[child], heap[j], preference):
            return
        heap[j], heap[child] = heap[child], heap[j]
        j = child

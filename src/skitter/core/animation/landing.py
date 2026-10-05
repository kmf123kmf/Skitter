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

Choreographies then choose when the k-th tile lands: `landing_times` paces
them evenly, slowing down at the end so the last few land one by one. A new
ordering must say which kind it is; tests check random orderings against
chance (tests/test_animation.py).
"""

import math

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


def landing_times(n: int, window: float, wind_down: float = 0.0,
                  last_gap: float = 0.0) -> np.ndarray:  # fmt: skip
    """(N,) when the k-th landing happens, from 0 (the first) to `window` (the last).

    Landings come at a steady pace, then over the last `wind_down` seconds the
    pace slows down exponentially, so that the last two land `last_gap`
    seconds apart (at most half the wind-down). Evenly spaced landings at any
    frame rate land a fixed number of tiles per frame: in the last frames,
    with little else moving, the last tiles would visibly land all at once.
    Evenly spaced if there's no wind-down, or if they are already at least
    `last_gap` apart.
    """
    even = np.linspace(0.0, window, n) if n > 1 else np.zeros(n)
    tail = min(max(float(wind_down), 0.0), float(window))
    gap = min(max(float(last_gap), 0.0), tail / 2)
    if n < 2 or tail <= 0 or gap * (n - 1) <= window:
        return even
    steady = window - tail
    # The pace (landings per second) over the wind-down, s in [0, tail]: rate e^(-k s / tail),
    # ending at rate e^-k with exactly one landing in the last `gap` seconds. Find the
    # decay k that fits the n - 1 landings after the first into the window.

    def shape(k: float) -> tuple[float, float]:
        end = k / (tail * math.expm1(k * gap / tail))  # one landing in the last gap
        rate = end * math.exp(k)
        return rate, rate * (steady + tail * -math.expm1(-k) / k)

    lo, hi = 1e-9, 1.0
    while shape(hi)[1] < n - 1 and hi < 600:
        lo, hi = hi, hi * 2
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if shape(mid)[1] < n - 1 else (lo, mid)
    k = (lo + hi) / 2
    rate = shape(k)[0]
    count = np.arange(n, dtype=np.float64)
    late = np.maximum(count - rate * steady, 0.0)  # landings into the wind-down
    into = -tail / k * np.log(np.maximum(1.0 - late * k / (rate * tail), 1e-300))
    times = np.where(count <= rate * steady, count / rate, steady + np.minimum(into, tail))
    times[-1] = window
    return np.maximum.accumulate(times)


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

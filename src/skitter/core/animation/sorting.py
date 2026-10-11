"""Sorting algorithms that unscramble a mosaic, round by round (for the Sort build).

The mosaic's spots are lined up along a path (`path_order`: reading order, a
snake, a Hilbert curve or a spiral). Position p on the path is spot
path[p]; the photo whose home is spot path[v] has value v. A scrambled mosaic
is then an unsorted array a (a[p]: the value at position p), and sorting it
puts every photo home.

Each algorithm sorts `a` in place, in rounds: steps that happen at once on
screen (all the compare-and-swaps of one stage of a sorting network, one
merge pass, one quicksort level...), yielding after each round that changed
anything. `sort_states` records the array after each round, grouping rounds
when there are more than can be shown, and finishes with one cleanup round
if an algorithm left anything out of place (none does). The slow sorts
(insertion, bubble, cocktail shaker, "I can't believe it can sort",
pancake) are here for their patterns: every algorithm gets the same time.
"""

import math
from collections.abc import Callable, Iterator

import numpy as np
from numba import njit

from skitter.core.scene import MosaicScene

# Paths: the order the sorted mosaic reads in.


def path_order(scene: MosaicScene, path: str) -> np.ndarray:
    """(N,) the spots in path order: path[p] is the spot at position p."""
    n = len(scene)
    if n == 0:
        return np.zeros(0, np.int64)
    tw, th = scene.tile_size
    x, y = scene.center[:, 0], scene.center[:, 1]
    if path == "snake":  # rows of base-tile height, every other one right to left
        row = np.floor(y / th)
        return np.lexsort((np.where(row % 2 == 0, x, -x), row))
    if path == "hilbert":
        x0, y0, x1, y1 = scene.bounds
        side = 1 << max(0, math.ceil(math.log2(max((x1 - x0) / tw, (y1 - y0) / th, 1.0))))
        cx = np.clip(((x - x0) / tw).astype(np.int64), 0, side - 1)
        cy = np.clip(((y - y0) / th).astype(np.int64), 0, side - 1)
        return np.lexsort((scene.reading_order, hilbert_index(side, cx, cy)))
    if path == "spiral":  # ring by ring outward from the middle, each ring clockwise from the top
        w, h = scene.canvas
        dx, dy = x - w / 2, y - h / 2
        turn = np.arctan2(dx, -dy) % math.tau / math.tau
        return np.argsort(np.floor(np.hypot(dx, dy) / tw) + turn, kind="stable")
    return np.argsort(scene.reading_order, kind="stable")  # reading order


def hilbert_index(side: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance along a Hilbert curve filling a side x side grid (side a power of two)."""
    x, y = np.asarray(x, np.int64).copy(), np.asarray(y, np.int64).copy()
    d = np.zeros(len(x), np.int64)
    s = side // 2
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx.astype(np.int64)) ^ ry.astype(np.int64))
        flip = ~ry & rx
        x = np.where(flip, side - 1 - x, x)
        y = np.where(flip, side - 1 - y, y)
        x, y = np.where(~ry, y, x), np.where(~ry, x, y)
        s //= 2
    return d


# Algorithms: each sorts `a` in place, yielding after every round that changed it.

Algorithm = Callable[[np.ndarray, np.random.Generator, dict], Iterator[None]]


def odd_even(a, rng, info) -> Iterator[None]:
    """Odd-even transposition sort (bubble sort in parallel): neighbors at even, then odd,
    positions compare and swap, all at once."""
    n, parity, calm = len(a), 0, 0
    while calm < 2 and n > 1:
        i = np.arange(parity, n - 1, 2)
        swap = i[a[i] > a[i + 1]]
        if len(swap):
            a[swap], a[swap + 1] = a[swap + 1], a[swap].copy()
            calm = 0
            yield
        else:
            calm += 1
        parity ^= 1


def bitonic(a, rng, info) -> Iterator[None]:
    """Bitonic sorting network (for any length): stages of compare-and-swaps, each stage
    all at once."""
    n = len(a)
    lo, hi, up = _bitonic_network(n)
    for at in _by_stage(_stages(n, lo, hi)):
        i, j, asc = lo[at], hi[at], up[at]
        swap = np.where(asc, a[i] > a[j], a[i] < a[j])
        if swap.any():
            i, j = i[swap], j[swap]
            a[i], a[j] = a[j], a[i].copy()
            yield


def _bitonic_network(n: int):
    """Comparators (i, j, ascending) of a bitonic sort for any n (after H. W. Lang)."""
    lo, hi, up = [], [], []

    def merge(start, count, asc):
        if count > 1:
            m = 1 << ((count - 1).bit_length() - 1)  # the greatest power of two below count
            lo.extend(range(start, start + count - m))
            hi.extend(range(start + m, start + count))
            up.extend([asc] * (count - m))
            merge(start, m, asc)
            merge(start + m, count - m, asc)

    def sort(start, count, asc):
        if count > 1:
            m = count // 2
            sort(start, m, not asc)
            sort(start + m, count - m, asc)
            merge(start, count, asc)

    sort(0, n, True)
    return np.array(lo, np.int64), np.array(hi, np.int64), np.array(up, bool)


@njit(cache=True, nogil=True)
def _stages(n, lo, hi):
    """Each operation's round, as soon as it can go: after the last one on its positions
    (operations on separate positions happen at once; the outcome is unchanged)."""
    free = np.zeros(n, np.int64)
    stage = np.empty(len(lo), np.int64)
    for k in range(len(lo)):
        s = max(free[lo[k]], free[hi[k]])
        stage[k] = s
        free[lo[k]] = s + 1
        free[hi[k]] = s + 1
    return stage


def merge(a, rng, info) -> Iterator[None]:
    """Bottom-up merge sort: sorted runs of 1, 2, 4... merged in pairs, a pass at a time."""
    n, width = len(a), 1
    while width < n:
        block = np.arange(n) // (2 * width)
        a[:] = a[np.lexsort((a, block))]
        width *= 2
        yield


def quick(a, rng, info) -> Iterator[None]:
    """Quicksort, a level at a time: every unsorted stretch splits around a random pivot
    of its own (smaller, the pivot, larger), all at once."""
    n = len(a)
    group = np.zeros(n, np.int64)  # stretches, numbered left to right
    while n:
        starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
        sizes = np.diff(np.r_[starts, n])
        if sizes.max() <= 1:
            return
        pivot = a[starts + (rng.random(len(starts)) * sizes).astype(np.int64)]
        g = np.repeat(np.arange(len(starts)), sizes)
        side = np.sign(a - pivot[g]) + 1  # 0 smaller, 1 the pivot's, 2 larger
        order = np.lexsort((np.arange(n), side, g))
        a[:] = a[order]
        key = (g * 3 + side)[order]
        group = np.cumsum(np.r_[True, key[1:] != key[:-1]]) - 1
        yield


def radix(a, rng, info) -> Iterator[None]:
    """Radix sort, most significant digit first (base 4): each pass sorts by one more digit,
    so photos land in the right quarter, then the right sixteenth..."""
    n = len(a)
    digits = max(1, math.ceil(math.log(max(n, 2), 4)))
    for d in range(1, digits + 1):
        a[:] = a[np.argsort(a // 4 ** (digits - d), kind="stable")]
        yield


def heap(a, rng, info) -> Iterator[None]:
    """Heap sort: the array becomes a max-heap, then the largest is swapped to the end
    again and again; swaps on separate positions happen at once."""
    lo, hi = _heap_swaps(a.copy())
    yield from _swap_rounds(a, lo, hi, _stages(len(a), lo, hi))


def selection(a, rng, info) -> Iterator[None]:
    """Selection sort: the smallest of the rest swapped into the next position, one at a
    time."""
    lo, hi = _selection_swaps(a.copy())
    yield from _swap_rounds(a, lo, hi, np.arange(len(lo)))


def _swap_rounds(a, lo, hi, stage) -> Iterator[None]:
    for at in _by_stage(stage):
        i, j = lo[at], hi[at]
        a[i], a[j] = a[j], a[i].copy()
        yield


def _by_stage(stage: np.ndarray) -> Iterator[np.ndarray]:
    """The operations of each round in turn (indices, in their original order)."""
    order = np.argsort(stage, kind="stable")
    edges = np.flatnonzero(np.r_[True, np.diff(stage[order]) > 0, True])
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        yield order[lo:hi]


@njit(cache=True, nogil=True)
def _sift(a, start, end, lo, hi, k):
    """Sift a[start] down the max-heap a[:end], recording swaps from lo[k], hi[k] on."""
    root = start
    while 2 * root + 1 < end:
        child = 2 * root + 1
        if child + 1 < end and a[child] < a[child + 1]:
            child += 1
        if a[root] >= a[child]:
            return k
        a[root], a[child] = a[child], a[root]
        lo[k], hi[k] = root, child
        k += 1
        root = child
    return k


@njit(cache=True, nogil=True)
def _heap_swaps(a):
    n = len(a)
    size = n * (int(np.log2(max(n, 2))) + 2) + 1  # at most one sift's depth per step
    lo, hi = np.empty(size, np.int64), np.empty(size, np.int64)
    k = 0
    for start in range(n // 2 - 1, -1, -1):
        k = _sift(a, start, n, lo, hi, k)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        lo[k], hi[k] = 0, end
        k += 1
        k = _sift(a, 0, end, lo, hi, k)
    return lo[:k], hi[:k]


@njit(cache=True, nogil=True)
def _selection_swaps(a):
    n = len(a)
    lo, hi = np.empty(n, np.int64), np.empty(n, np.int64)
    k = 0
    for i in range(n - 1):
        m = i
        for j in range(i + 1, n):
            if a[j] < a[m]:
                m = j
        if m != i:
            a[i], a[m] = a[m], a[i]
            lo[k], hi[k] = i, m
            k += 1
    return lo[:k], hi[:k]


def shear(a, rng, info) -> Iterator[None]:
    """Shear sort on a grid `info["columns"]` wide: rows sort (every other one backwards),
    then columns, until sorted in a snake; a last pass turns every row forwards. The
    snake runs forwards in the last row, so a short last row works: its missing cells
    act as the largest values, at the row's end, and never move."""
    n = len(a)
    width = max(1, min(int(info.get("columns", 1)), n))
    p = np.arange(n)
    r, c = p // width, p % width
    rows = -(-n // width)
    by_row = np.lexsort((c, r))  # positions row by row
    by_col = np.lexsort((r, c))  # positions column by column

    def sort_rows(snake: bool):
        vals = a[by_row]
        key = np.where(snake & ((rows - 1 - r[by_row]) % 2 == 1), -vals, vals)
        a[by_row] = vals[np.lexsort((key, r[by_row]))]

    def sort_columns():
        vals = a[by_col]
        a[by_col] = vals[np.lexsort((vals, c[by_col]))]

    for _ in range(max(1, math.ceil(math.log2(max(rows, 2)))) + 1):
        for step in (lambda: sort_rows(True), sort_columns):
            before = a.copy()
            step()
            if not np.array_equal(before, a):
                yield
    before = a.copy()
    sort_rows(False)
    if not np.array_equal(before, a):
        yield


# Slow sorts, shown for their patterns (their inner loops compiled; a round each step).


def insertion(a, rng, info) -> Iterator[None]:
    """Insertion sort: each photo in turn slides back into the sorted stretch behind it."""
    for i in range(1, len(a)):
        if _insert(a, i):
            yield


@njit(cache=True, nogil=True)
def _insert(a, i):
    value, j = a[i], i
    while j > 0 and a[j - 1] > value:
        a[j] = a[j - 1]
        j -= 1
    a[j] = value
    return j != i


def bubble(a, rng, info) -> Iterator[None]:
    """Bubble sort, a pass at a time: each pass carries the largest photo left to the end,
    so the sorted stretch grows back from the end."""
    end = len(a)
    while end > 1 and _bubble_pass(a, 0, end - 1, 1):
        end -= 1
        yield


def cocktail(a, rng, info) -> Iterator[None]:
    """Cocktail shaker sort: bubble passes back and forth, the sorted stretches growing
    from both ends toward the middle."""
    lo, hi = 0, len(a) - 1
    while lo < hi:
        if not _bubble_pass(a, lo, hi, 1):
            return
        hi -= 1
        yield
        if not _bubble_pass(a, lo, hi, -1):
            return
        lo += 1
        yield


@njit(cache=True, nogil=True)
def _bubble_pass(a, lo, hi, step):
    """Neighbors from lo to hi (forward) or hi down to lo (backward) swap when out of
    order; whether any did."""
    swapped = False
    if step > 0:
        for j in range(lo, hi):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
    else:
        for j in range(hi, lo, -1):
            if a[j - 1] > a[j]:
                a[j - 1], a[j] = a[j], a[j - 1]
                swapped = True
    return swapped


def cant_believe(a, rng, info) -> Iterator[None]:
    """ "I can't believe it can sort" (Fung, 2021): for every i, for every j, swap if
    a[i] < a[j]. It looks backwards, and sorts: the stretch up to i ends up sorted, the
    largest photo riding at its front."""
    for i in range(len(a)):
        if _sweep(a, i):
            yield


@njit(cache=True, nogil=True)
def _sweep(a, i):
    swapped = False
    for j in range(len(a)):
        if a[i] < a[j]:
            a[i], a[j] = a[j], a[i]
            swapped = True
    return swapped


def pancake(a, rng, info) -> Iterator[None]:
    """Pancake sort: the only move is flipping the stretch from the start over. Flip the
    largest unsorted photo to the start, then flip it down to the end of the unsorted
    stretch; repeat."""
    for size in range(len(a), 1, -1):
        m = int(np.argmax(a[:size]))
        if m == size - 1:
            continue
        if m > 0:
            a[: m + 1] = a[m::-1].copy()
            yield
        a[:size] = a[size - 1 :: -1].copy()
        yield


def bogo(a, rng, info) -> Iterator[None]:
    """Bogosort: shuffle and hope. A few hopeless shuffles, then (luckily) sorted."""
    for _ in range(int(info.get("shuffles", 5))):
        a[:] = rng.permutation(a)
        yield
    a.sort()
    yield


ALGORITHMS: dict[str, Algorithm] = {
    "bitonic": bitonic, "odd_even": odd_even, "merge": merge, "quick": quick, "radix": radix,
    "heap": heap, "selection": selection, "shear": shear, "insertion": insertion,
    "bubble": bubble, "cocktail": cocktail, "cant_believe": cant_believe, "pancake": pancake,
    "bogo": bogo,
}  # fmt: skip


def sort_states(a: np.ndarray, algorithm: str, seed: int, max_rounds: int,
                info: dict | None = None) -> tuple[np.ndarray, bool]:  # fmt: skip
    """((S, N) the array before the first round and after each shown round, whether a
    cleanup round was needed). Rounds beyond `max_rounds` are grouped: each shown round
    is a few of the algorithm's, in order. The last state is sorted."""
    info = info or {}
    run = ALGORITHMS[algorithm]
    a = np.asarray(a, np.int64)

    def changes(work: np.ndarray) -> Iterator[None]:
        """The algorithm's rounds that changed anything (skipping idle ones)."""
        before = work.copy()
        for _ in run(work, np.random.default_rng(seed), info):
            if not np.array_equal(before, work):
                before[:] = work
                yield

    count = sum(1 for _ in changes(a.copy()))
    work = a.copy()
    group = max(1, math.ceil(count / max(max_rounds, 1)))
    states = [work.copy()]
    for k, _ in enumerate(changes(work), 1):
        if k % group == 0 or k == count:
            states.append(work.copy())
    cleanup = not np.array_equal(work, np.sort(work))
    if cleanup:
        states.append(np.sort(work))
    return np.array(states, np.int32).reshape(len(states), len(a)), cleanup

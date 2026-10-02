"""Choosing one candidate per region under the reuse rules.

Each region has a short list of candidates sorted by cost (see index.py).
The rules: a tile image may be used at most `max_uses` times (0: no limit),
and two uses of the same image must be at least `spacing` apart (mosaic units;
0: off). Crops and mirrored copies of one image count as that image.

`Assignment.greedy` visits regions in priority order and gives each its
cheapest allowed candidate. `refine_pass` then improves the result: a region
moves to a better candidate when that is allowed, or when the one region
blocking it can move to another allowed candidate for a lower total cost.
Costs are weighted by region importance (visible area), so refinement
trades fairly between large and small regions.

With error diffusion, regions are visited in reading order and each one's
remaining average-color error is passed to the regions after it (on a grid
of cells about one region wide, with Floyd-Steinberg weights), so areas keep
their average color even when no single tile matches well.

Uses of each image are kept in a doubly linked list over regions, so
checking the rules costs time proportional to that image's uses.
"""

import numpy as np
from numba import njit

DIFFUSION_DAMPING = 0.85


@njit(cache=True)
def _allowed(r, tile, uses, head, nxt, center, max_uses, spacing2):
    if max_uses > 0 and uses[tile] >= max_uses:
        return False
    if spacing2 > 0.0:
        q = head[tile]
        while q != -1:
            dx = center[q, 0] - center[r, 0]
            dy = center[q, 1] - center[r, 1]
            if q != r and dx * dx + dy * dy < spacing2:
                return False
            q = nxt[q]
    return True


@njit(cache=True)
def _place(r, j, tiles, choice, uses, head, nxt, prv):
    tile = tiles[r, j]
    choice[r] = j
    uses[tile] += 1
    prv[r] = -1
    nxt[r] = head[tile]
    if head[tile] != -1:
        prv[head[tile]] = r
    head[tile] = r


@njit(cache=True)
def _remove(r, tiles, choice, uses, head, nxt, prv):
    tile = tiles[r, choice[r]]
    uses[tile] -= 1
    if prv[r] != -1:
        nxt[prv[r]] = nxt[r]
    else:
        head[tile] = nxt[r]
    if nxt[r] != -1:
        prv[nxt[r]] = prv[r]
    nxt[r] = prv[r] = -1
    choice[r] = -1


@njit(cache=True, nogil=True)
def _greedy(order, tiles, costs, center, max_uses, spacing2, choice, uses, head, nxt, prv):
    unplaced = 0
    for r in order:
        placed = False
        for j in range(tiles.shape[1]):
            tile = tiles[r, j]
            if tile < 0 or not np.isfinite(costs[r, j]):
                break
            if _allowed(r, tile, uses, head, nxt, center, max_uses, spacing2):
                _place(r, j, tiles, choice, uses, head, nxt, prv)
                placed = True
                break
        if not placed:
            unplaced += 1
    return unplaced


@njit(cache=True, nogil=True)
def _greedy_diffused(order, tiles, costs, cand_mean, target_mean, mean_weight, tint, cell_of,
                     grid_shape, center, max_uses, spacing2, choice, uses, head, nxt, prv,
                     tint_target):  # fmt: skip
    gy, gx = grid_shape
    err = np.zeros((gy, gx, 3))
    err_w = np.zeros((gy, gx))
    offsets = ((0, 1, 7 / 16), (1, -1, 3 / 16), (1, 0, 5 / 16), (1, 1, 1 / 16))
    unplaced = 0
    for r in order:
        cy, cx = cell_of[r, 0], cell_of[r, 1]
        want = target_mean[r].copy()
        if err_w[cy, cx] > 0:
            want += DIFFUSION_DAMPING * err[cy, cx] / err_w[cy, cx]
        tint_target[r] = want
        w = mean_weight[r] * (1.0 - tint) ** 2
        best_j, best_c = -1, np.inf
        for j in range(tiles.shape[1]):
            tile = tiles[r, j]
            if tile < 0 or not np.isfinite(costs[r, j]):
                break
            old = 0.0
            new = 0.0
            for c in range(3):
                old += (cand_mean[r, j, c] - target_mean[r, c]) ** 2
                new += (cand_mean[r, j, c] - want[c]) ** 2
            cost = costs[r, j] + w * (new - old)
            if cost < best_c and _allowed(r, tile, uses, head, nxt, center, max_uses, spacing2):
                best_j, best_c = j, cost
        if best_j < 0:
            unplaced += 1
            continue
        _place(r, best_j, tiles, choice, uses, head, nxt, prv)
        # What shows: the tile's average tinted toward the wanted color.
        for c in range(3):
            shown = cand_mean[r, best_j, c] + tint * (want[c] - cand_mean[r, best_j, c])
            e = want[c] - shown
            for dy, dx, k in offsets:
                y, x = cy + dy, cx + dx
                if 0 <= y < gy and 0 <= x < gx:
                    err[y, x, c] += k * e
        for dy, dx, k in offsets:
            y, x = cy + dy, cx + dx
            if 0 <= y < gy and 0 <= x < gx:
                err_w[y, x] += k
    return unplaced


@njit(cache=True, nogil=True)
def _refine(order, tiles, costs, importance, center, max_uses, spacing2,
            choice, uses, head, nxt, prv):  # fmt: skip
    """One improvement pass; returns the total (importance-weighted) cost saved."""
    saved = 0.0
    for r in order:
        cur = choice[r]
        if cur <= 0:
            continue  # unplaced, or already on its best candidate
        for j in range(cur):
            tile = tiles[r, j]
            if tile < 0:
                break
            gain = importance[r] * (costs[r, cur] - costs[r, j])
            _remove(r, tiles, choice, uses, head, nxt, prv)
            if _allowed(r, tile, uses, head, nxt, center, max_uses, spacing2):
                _place(r, j, tiles, choice, uses, head, nxt, prv)
                saved += gain
                break
            # Try moving one region that holds this tile out of the way.
            moved = False
            q = head[tile]
            tries = 0
            while q != -1 and tries < 8 and not moved:
                tries += 1
                nq = nxt[q]
                cq = choice[q]
                _remove(q, tiles, choice, uses, head, nxt, prv)
                if _allowed(r, tile, uses, head, nxt, center, max_uses, spacing2):
                    _place(r, j, tiles, choice, uses, head, nxt, prv)
                    for j2 in range(tiles.shape[1]):
                        t2 = tiles[q, j2]
                        if t2 < 0 or not np.isfinite(costs[q, j2]):
                            break
                        if j2 == cq:
                            continue
                        loss = importance[q] * (costs[q, j2] - costs[q, cq])
                        if loss >= gain:
                            break  # sorted: no later candidate is cheaper
                        if _allowed(q, t2, uses, head, nxt, center, max_uses, spacing2):
                            _place(q, j2, tiles, choice, uses, head, nxt, prv)
                            saved += gain - loss
                            moved = True
                            break
                    if not moved:
                        _remove(r, tiles, choice, uses, head, nxt, prv)
                if not moved:
                    _place(q, cq, tiles, choice, uses, head, nxt, prv)
                q = nq
            if moved:
                break
            _place(r, cur, tiles, choice, uses, head, nxt, prv)
    return saved


class Assignment:
    """Chosen candidate per region, with the bookkeeping the rules need.

    tiles: (R, K) image id of each candidate (-1: none); costs: (R, K)
    ascending per row; center: (R, 2) region centers; n_tiles: number of
    image ids.
    """

    def __init__(self, tiles, costs, center, n_tiles: int, max_uses: int = 0, spacing=0.0):
        self.tiles = np.ascontiguousarray(tiles, dtype=np.int64)
        self.costs = np.ascontiguousarray(costs, dtype=np.float32)
        self.center = np.ascontiguousarray(center, dtype=np.float64)
        self.max_uses = int(max_uses)
        self.spacing2 = float(spacing) ** 2
        r = len(self.tiles)
        self.choice = np.full(r, -1, np.int64)
        self.uses = np.zeros(n_tiles, np.int64)
        self.head = np.full(n_tiles, -1, np.int64)
        self.nxt = np.full(r, -1, np.int64)
        self.prv = np.full(r, -1, np.int64)
        self.tint_target: np.ndarray | None = None  # set by greedy_diffused

    def _state(self):
        return (self.max_uses, self.spacing2, self.choice, self.uses, self.head, self.nxt,
                self.prv)  # fmt: skip

    def greedy(self, order) -> int:
        """Place the regions in order; returns how many found no allowed candidate."""
        order = np.asarray(order, dtype=np.int64)
        return _greedy(order, self.tiles, self.costs, self.center, *self._state())

    def greedy_diffused(self, order, cand_mean, target_mean, mean_weight, tint, cell_of, shape):
        """Greedy in reading order with average-color error diffusion (see module doc)."""
        self.tint_target = np.array(target_mean, dtype=np.float64)
        return _greedy_diffused(
            np.asarray(order, dtype=np.int64), self.tiles, self.costs,
            np.ascontiguousarray(cand_mean, dtype=np.float64),
            np.ascontiguousarray(target_mean, dtype=np.float64),
            np.ascontiguousarray(mean_weight, dtype=np.float64), float(tint),
            np.ascontiguousarray(cell_of, dtype=np.int64), shape, self.center, *self._state(),
            self.tint_target,
        )  # fmt: skip

    def refine_pass(self, order, importance) -> float:
        return _refine(
            np.asarray(order, dtype=np.int64), self.tiles, self.costs,
            np.ascontiguousarray(importance, dtype=np.float64), self.center, *self._state(),
        )  # fmt: skip

    def set_choices(self, choice) -> None:
        """Replace the whole assignment (choice[r]: candidate index, -1 for none)."""
        self.unplace(np.flatnonzero(self.choice >= 0))
        for r, j in enumerate(np.asarray(choice, dtype=np.int64)):
            if j >= 0:
                _place(r, j, self.tiles, self.choice, self.uses, self.head, self.nxt, self.prv)

    def unplace(self, regions) -> None:
        for r in np.asarray(regions, dtype=np.int64):
            if self.choice[r] >= 0:
                _remove(r, self.tiles, self.choice, self.uses, self.head, self.nxt, self.prv)

    def force(self, regions) -> int:
        """Give unplaced regions their best candidate regardless of the rules.

        Returns how many were placed this way (rule violations).
        """
        count = 0
        for r in np.asarray(regions, dtype=np.int64):
            if self.choice[r] < 0 and self.tiles[r, 0] >= 0:
                _place(r, 0, self.tiles, self.choice, self.uses, self.head, self.nxt, self.prv)
                count += 1
        return count

    def chosen_tiles(self) -> np.ndarray:
        rows = np.arange(len(self.choice))
        return np.where(self.choice >= 0, self.tiles[rows, np.maximum(self.choice, 0)], -1)

    def chosen_costs(self) -> np.ndarray:
        rows = np.arange(len(self.choice))
        return np.where(self.choice >= 0, self.costs[rows, np.maximum(self.choice, 0)], np.inf)

    def check(self) -> tuple[int, int]:
        """(uses over the limit, pairs closer than spacing): both 0 when the rules hold."""
        tiles = self.chosen_tiles()
        placed = tiles >= 0
        counts = np.bincount(tiles[placed], minlength=len(self.uses))
        over = int(np.maximum(counts - self.max_uses, 0).sum()) if self.max_uses else 0
        close = 0
        if self.spacing2 > 0:
            for t in np.flatnonzero(counts > 1):
                pts = self.center[(tiles == t)]
                d2 = ((pts[:, None] - pts[None]) ** 2).sum(-1)
                close += int((d2[np.triu_indices(len(pts), 1)] < self.spacing2).sum())
        return over, close

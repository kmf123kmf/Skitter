"""Tiles moved between places by invisible hands.

Places are the tiles' spots in the finished mosaic: place i is tile i's final
spot. `swap_plan` makes the moves of unscrambling: tiles go home in a given
order, in batches of simultaneous swaps, each swap putting a tile home and
taking the tile that lay there to the first one's old place.

`MovesTimeline` plays them. A tile lies still at its current place, at its
own size and turn, until a move slides it along the table to another place,
`travel` seconds later; paths curve to one side ("keep right"), so two tiles
trading places pass side by side. A photo lying in a spot not its own may
hang over its neighbors (tiles of different sizes or turns, a pile), so every
tile has a level and keeps it throughout: it slides over the tiles below it
and under those above, and which of two lies on top never changes. Levels
that keep every tile above those it lies on in the finished mosaic (a pile)
leave nothing to pop at the end.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from skitter.core.animation.base import TileFrame, Timeline
from skitter.core.easing import ease_in_out_cubic
from skitter.core.scene import MosaicScene

BULGE = 0.15  # how far paths curve aside, as a share of the distance travelled


@dataclass(frozen=True)
class Moves:
    """Moves, one per row: `tile` leaves place `source` for place `target` at `start`."""

    tile: np.ndarray  # (M,) int
    start: np.ndarray  # (M,) seconds
    source: np.ndarray  # (M,) place
    target: np.ndarray  # (M,) place

    @classmethod
    def none(cls) -> "Moves":
        i = np.zeros(0, np.int64)
        return cls(i, np.zeros(0), i, i)

    def __len__(self) -> int:
        return len(self.tile)

    def sorted(self) -> "Moves":
        """The same moves, by tile, then start."""
        order = np.lexsort((self.start, self.tile))
        return Moves(self.tile[order], self.start[order], self.source[order],
                     self.target[order])  # fmt: skip


class _Playback:
    """Which move each tile is making at a time: moves sorted by tile, then start,
    found with one search (key = tile x span + start)."""

    def __init__(self, n: int, moves: Moves, places, travel: float, horizon: float):
        self.moves = moves.sorted()
        self.places = np.asarray(places, np.int64)
        self.travel = max(float(travel), 1e-6)
        self._span = horizon + self.travel + 1.0
        self._key = self.moves.tile * self._span + self.moves.start
        self._tiles = np.arange(n)

    def at(self, t: float):
        """(k, p, moving, place): each tile's latest move begun before t (k, valid
        where it has one), its progress p (1: done), whether it's moving, and the
        place it lies at or is going to."""
        m, tiles = self.moves, self._tiles
        last = np.searchsorted(self._key, tiles * self._span + t, side="left") - 1
        has = last >= 0
        has[has] = m.tile[last[has]] == tiles[has]
        k = np.where(has, last, 0)
        p = np.where(has, np.clip((t - m.start[k]) / self.travel, 0.0, 1.0), 1.0)
        moving = has & (p < 1.0)
        return k, p, moving, np.where(has, m.target[k], self.places)


class MovesTimeline(Timeline):
    """Tiles starting at `places` (N,) and moved by `moves`, each taking `travel`
    seconds, sliding along the table at their `levels` (N,: lower drawn first; see the
    module docstring). The last frame, at `duration`, is the finished mosaic: every
    tile must end at home."""

    def __init__(self, scene: MosaicScene, places, moves: Moves, travel: float,
                 duration: float, levels):  # fmt: skip
        self.final = TileFrame.final(scene)
        self.duration = float(duration)
        self.play = _Playback(len(scene), moves, places, travel, self.duration)
        self.moves, self.places, self.travel = self.play.moves, self.play.places, self.play.travel
        self.order = np.argsort(np.asarray(levels), kind="stable")

    def frame(self, t: float) -> TileFrame:
        t = float(t)
        if t >= self.duration:
            return self.final  # exactly, whatever the rounding of the times
        b, m = self.final, self.moves
        if not len(m):  # nothing moves: every tile lies where it starts
            return b.replace(center=b.center[self.places], order=self.order)
        k, p, moving, place = self.play.at(max(t, 0.0))
        center = b.center[place].copy()
        if moving.any():
            i = np.flatnonzero(moving)
            mk = k[i]
            e = ease_in_out_cubic(p[i])
            a, z = b.center[m.source[mk]], b.center[m.target[mk]]
            way = z - a
            side = np.stack([-way[:, 1], way[:, 0]], axis=1)  # perpendicular, as long as the way
            center[i] = a + way * e[:, None] + side * (BULGE * np.sin(math.pi * e))[:, None]
        return b.replace(center=center, order=self.order)


def swap_plan(n: int, home_order: np.ndarray, times: np.ndarray, rng: np.random.Generator,
              centers=None, reach: float = math.inf) -> tuple[np.ndarray, Moves]:  # fmt: skip
    """(places, moves) unscrambling n tiles: `home_order` lists the scrambled tiles in the
    order they go home and `times` (one per tile, ascending) when. Tiles with the same
    time go home together, in one batch of simultaneous swaps; batches must be at
    least a move's time apart. Each swap takes a tile home and the tile lying there to
    the first one's old place: that one goes home in a later batch, or, if its home is
    that old place, in the same swap (a two-tile swap; the last batch has only those).
    So every swap puts at least one tile home. A tile that can't go home in its batch
    (no partner) goes home in an earlier one, or, failing that, starts at home. places:
    (n,) where each tile starts (tiles not listed start, and stay, at home).

    With `centers` (n, 2), the places' positions, a tile goes home from at most
    `reach` away where it can (else from as near as it can): moves stay short, as if
    a shuffled puzzle were being put right, rather than tiles flying across it.

    Built backwards from the finished mosaic, batch by batch from the last: undoing a
    swap trades a tile of the batch (home from then on) with a tile still to go home.
    """
    order = np.asarray(home_order, np.int64)
    times = np.asarray(times, np.float64)
    starts, batch = np.unique(times, return_inverse=True)  # batch start times; each tile's
    home_batch = np.full(n, -1, np.int64)  # batch each tile goes home in (-1: stays home)
    home_batch[order] = batch.reshape(-1)
    if centers is not None and n > 1:
        from scipy.spatial import cKDTree

        centers = np.asarray(centers, np.float64)
        # Places never move (only which tile lies where): each home's nearest, once.
        near_dist, near_at = cKDTree(centers).query(centers, k=min(n, 96))
        near_dist, near_at = near_dist.reshape(n, -1), near_at.reshape(n, -1)
    else:  # no positions: partners drawn at random
        centers = np.zeros((n, 2))
        near_dist, near_at = np.zeros((n, 0)), np.zeros((n, 0), np.int64)
    places, tiles, partners, batches = _undo_swaps(
        n, order, home_batch, len(starts), near_at, near_dist, centers, float(reach),
        int(rng.integers(2**31)))  # fmt: skip
    found = zip(batches[::-1], tiles[::-1], partners[::-1], strict=True)  # first batch first
    swaps = [(float(starts[b]), int(a), int(c)) for b, a, c in found]
    # Replay forwards to record each move's source and target.
    now = places.copy()
    rows = []
    for s, tile, partner in swaps:
        a = now[tile]  # tile's wrong place; partner lies at tile's home
        assert now[partner] == tile
        rows.append((tile, s, a, tile))
        rows.append((partner, s, tile, a))
        now[tile], now[partner] = tile, a
    if not rows:
        return places, Moves.none()
    tile, start, source, target = (np.array(c) for c in zip(*rows, strict=True))
    return places, Moves(tile.astype(np.int64), start.astype(np.float64),
                         source.astype(np.int64), target.astype(np.int64))  # fmt: skip


@njit(cache=True, nogil=True)
def _undo_swaps(n, order, home_batch, batches, near_at, near_dist, centers, reach, seed):
    """swap_plan's backwards pass: (places at the start, then each swap's tile going home,
    partner and batch, the last batch first). home_batch is updated as tiles that find
    no partner move to an earlier batch."""
    np.random.seed(seed)
    places = np.arange(n)  # where each tile lies, while undoing
    lying = np.arange(n)  # which tile lies at each place
    taken = np.zeros(n, np.bool_)  # in a swap of the batch being undone
    out_tile = np.empty(n, np.int64)
    out_partner = np.empty(n, np.int64)
    out_batch = np.empty(n, np.int64)
    pick = np.empty(near_at.shape[1], np.int64)
    count = 0
    for b in range(batches - 1, -1, -1):
        first = count
        for i in range(len(order) - 1, -1, -1):
            tile = order[i]
            if home_batch[tile] != b or taken[tile]:
                continue  # another batch's, or going home in a two-tile swap already
            partner = -1
            # A tile going home later (preferred), else one of the batch's own (a two-tile
            # swap): within reach at random, else the nearest.
            if near_at.shape[1] == 0 or reach > near_dist[tile, -1]:
                for _ in range(64):  # reach beyond the nearest places: any tile within it
                    y = order[np.random.randint(len(order))]
                    if home_batch[y] > b and not taken[y] and y != tile:
                        dx = centers[places[y], 0] - centers[tile, 0]
                        dy = centers[places[y], 1] - centers[tile, 1]
                        if dx * dx + dy * dy <= reach * reach:
                            partner = y
                            break
            for later in (True, False):
                if partner >= 0:
                    break
                found = 0
                for j in range(near_at.shape[1]):
                    if near_dist[tile, j] > reach:
                        break
                    y = lying[near_at[tile, j]]
                    h = home_batch[y]
                    if (h > b if later else h == b) and not taken[y] and y != tile:
                        pick[found] = y
                        found += 1
                if found:
                    partner = pick[np.random.randint(found)]
            for later in (True, False):  # nothing within reach: the nearest, anywhere
                if partner >= 0:
                    break
                best = np.inf
                for y in order:
                    h = home_batch[y]
                    if (h > b if later else h == b) and not taken[y] and y != tile:
                        dx = centers[places[y], 0] - centers[tile, 0]
                        dy = centers[places[y], 1] - centers[tile, 1]
                        if dx * dx + dy * dy < best:
                            best = dx * dx + dy * dy
                            partner = y
            if partner < 0:  # no partner: try an earlier batch
                home_batch[tile] = b - 1
                continue
            taken[tile] = True
            taken[partner] = True
            out_tile[count], out_partner[count], out_batch[count] = tile, partner, b
            count += 1
        for k in range(first, count):  # undo the batch: trade places
            tile, partner = out_tile[k], out_partner[k]
            taken[tile] = False
            taken[partner] = False
            a, c = places[tile], places[partner]
            places[tile], places[partner] = c, a
            lying[a], lying[c] = partner, tile
    return places, out_tile[:count], out_partner[:count], out_batch[:count]

"""Tiles moved between places by invisible hands.

Places are the tiles' spots in the finished mosaic: place i is tile i's final
spot. `swap_plan` makes the moves of unscrambling: tiles go home in a given
order, in batches of simultaneous swaps, each swap putting a tile home and
taking the tile that lay there to the first one's old place. Two timelines
play such moves:

- `MovesTimeline`, for tiles that don't overlap (a grid): a tile lies still at
  its current place, at its own size and turn, until a move takes it to
  another place, `travel` seconds later. Carried, it rises in an arc (casting
  its shadow), tipping a little as if held; slid, it glides just above the
  others (a hover too low to show a shadow). Paths curve to one side ("keep
  right"), so two tiles trading places pass side by side, and one of them
  flies lower (`low`), passing beneath the other.
- `LayeredTimeline`, for tiles lying on each other (a pile): moving a tile out
  from under others would pass through them, so the pile first explodes. Each
  tile's layer is how many tiles lie beneath it at its deepest (`layers`);
  tiles of one layer never overlap. The layers rise apart, evenly spaced and
  spreading outward from the middle the higher they are (an exploded view),
  swaps happen in the gaps between them, and the pile settles back down. A
  photo takes on the turn and size of whatever spot it lies in, so layers
  stay free of overlaps. Swaps trade spots of one layer: the tile going home
  travels in the gap just above the layer, its partner in the gap just below,
  each at its own height, so nothing passes through anything. (Swaps between
  neighboring layers can't promise that: two tiles trading layers where they
  overlap must cross heights on the way, and moves of a batch go in step, so
  they'd cross where other tiles pass.) Every tile is a flat plate at some
  height, so drawing them from low to high is always right.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from skitter.core.animation.base import TileFrame, Timeline
from skitter.core.easing import ease_in_out_cubic
from skitter.core.scene import MosaicScene

HOVER = 0.01  # sliding tiles' height, in base tiles: above the others, no visible shadow
BULGE = 0.15  # how far paths curve aside, as a share of the distance travelled
LOW = 0.55  # a low move's arc, relative to the full lift
CLIMB = 0.2  # share of a layered move spent rising into its gap (and again settling)


@dataclass(frozen=True)
class Moves:
    """Moves, one per row: `tile` leaves place `source` for place `target` at `start`.
    `low` moves fly lower (a swap's second tile); `tilt` (radians) is how far a
    carried tile tips on the way; `lane` (layered moves) is the height it travels
    at, in layer gaps above the table."""

    tile: np.ndarray  # (M,) int
    start: np.ndarray  # (M,) seconds
    source: np.ndarray  # (M,) place
    target: np.ndarray  # (M,) place
    low: np.ndarray  # (M,) bool
    tilt: np.ndarray  # (M,) radians
    lane: np.ndarray | None = None  # (M,)

    @classmethod
    def none(cls) -> "Moves":
        z = np.zeros(0)
        i = np.zeros(0, np.int64)
        return cls(i, z, i, i, np.zeros(0, bool), z, z)

    def __len__(self) -> int:
        return len(self.tile)

    def sorted(self) -> "Moves":
        """The same moves, by tile, then start."""
        order = np.lexsort((self.start, self.tile))
        lane = None if self.lane is None else np.asarray(self.lane)[order]
        fields = (self.tile, self.start, self.source, self.target, self.low, self.tilt)
        return Moves(*(np.asarray(a)[order] for a in fields), lane)

    def delayed(self, seconds: float) -> "Moves":
        return Moves(self.tile, self.start + seconds, self.source, self.target, self.low,
                     self.tilt, self.lane)  # fmt: skip


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
    seconds; `carry` (else slide) and `lift` (mosaic units) shape the moves. The
    last frame, at `duration`, is the finished mosaic: every tile must end at home.
    For tiles that don't overlap (see the module docstring)."""

    def __init__(self, scene: MosaicScene, places, moves: Moves, travel: float,
                 duration: float, carry: bool = True, lift: float = 0.0):  # fmt: skip
        self.final = TileFrame.final(scene)
        self.duration = float(duration)
        self.play = _Playback(len(scene), moves, places, travel, self.duration)
        self.moves, self.places, self.travel = self.play.moves, self.play.places, self.play.travel
        self.carry = carry
        self.lift = float(lift) if carry else HOVER * scene.tile_size[0]

    def frame(self, t: float) -> TileFrame:
        t = float(t)
        if t >= self.duration:
            return self.final  # exactly, whatever the rounding of the times
        b, m = self.final, self.moves
        if not len(m):  # nothing moves: every tile lies where it starts
            return b.replace(center=b.center[self.places])
        k, p, moving, place = self.play.at(max(t, 0.0))
        center = b.center[place].copy()
        rotation = b.rotation.copy()
        height = np.zeros(len(place))
        if moving.any():
            i = np.flatnonzero(moving)
            mk = k[i]
            e = ease_in_out_cubic(p[i])
            a, z = b.center[m.source[mk]], b.center[m.target[mk]]
            way = z - a
            side = np.stack([-way[:, 1], way[:, 0]], axis=1)  # perpendicular, as long as the way
            arc = np.sin(math.pi * e)
            center[i] = a + way * e[:, None] + side * (BULGE * arc)[:, None]
            height[i] = self.lift * np.where(m.low[mk], LOW, 1.0) * arc
            if self.carry:
                rotation[i] += m.tilt[mk] * arc
        resting, flying = np.flatnonzero(~moving), np.flatnonzero(moving)
        order = np.concatenate([resting, flying[np.argsort(height[flying], kind="stable")]])
        return b.replace(center=center, rotation=rotation, order=order,
                         height=height if moving.any() else None,
                         rest=~moving if moving.any() else None)  # fmt: skip


@njit(cache=True, nogil=True)
def _layers(n, lower, upper):
    """Each tile's layer: one above the highest layer under it (pairs sorted by lower
    tile, so a tile's layer is final before the tiles on it are)."""
    layer = np.zeros(n, np.int64)
    for k in range(len(lower)):
        if layer[lower[k]] + 1 > layer[upper[k]]:
            layer[upper[k]] = layer[lower[k]] + 1
    return layer


def layers(scene: MosaicScene) -> np.ndarray:
    """(N,) each tile's layer in the finished mosaic: 0 on the table, else one above the
    highest tile under it. Tiles of one layer never overlap."""
    pairs = scene.overlaps
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return _layers(len(scene), np.ascontiguousarray(pairs[:, 0]),
                   np.ascontiguousarray(pairs[:, 1]))  # fmt: skip


def swap_lanes(moves: Moves, layer: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(M,) the height (in layer gaps above the table; layer k lies k + 1 up) each move
    of a layered swap travels at: the tile going home in the lower part of the gap
    above its layer, its partner in the upper part of the gap below. Tiles rising
    into a gap and tiles dropping into it never meet, and lanes differ."""
    level = layer[moves.source] + 1.0
    above = level + rng.uniform(0.2, 0.45, len(level))
    below = level - 1.0 + rng.uniform(0.55, 0.8, len(level))
    return np.where(moves.low, below, above)


class LayeredTimeline(Timeline):
    """A pile exploding into layers, tiles swapping between them, and the pile settling
    back (see the module docstring). Over the first `explode` seconds the layers rise
    to `gap` (mosaic units) apart, layer k spreading out from the middle by `spread`
    x k / top layer; the moves come after that; over the last `explode` seconds the
    pile settles. `shade` (0..1) darkens lower layers while apart."""

    def __init__(self, scene: MosaicScene, places, moves: Moves, travel: float,
                 duration: float, layer: np.ndarray, explode: float, gap: float,
                 spread: float, shade: float):  # fmt: skip
        self.final = TileFrame.final(scene)
        self.duration = float(duration)
        self.explode = min(float(explode), self.duration / 2)
        self.play = _Playback(len(scene), moves, places, travel, self.duration)
        self.moves, self.places, self.travel = self.play.moves, self.play.places, self.play.travel
        self.layer = np.asarray(layer, np.int64)
        self.gap = float(gap)
        self.shade = float(shade)
        top = max(int(self.layer.max()) if len(self.layer) else 0, 1)
        x0, y0, x1, y1 = scene.bounds
        middle = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        self._out = (scene.center - middle) * (spread * self.layer / top)[:, None]
        self._top = (top + 1) * self.gap  # the top layer's height, apart

    def apart(self, t: float) -> float:
        """How far the pile has come apart: 0 flat, 1 exploded."""
        rise = ease_in_out_cubic(min(t / max(self.explode, 1e-9), 1.0))
        fall = ease_in_out_cubic(min((self.duration - t) / max(self.explode, 1e-9), 1.0))
        return float(min(rise, fall))

    def frame(self, t: float) -> TileFrame:
        t = float(t)
        if t >= self.duration:
            return self.final  # exactly, whatever the rounding of the times
        t = max(t, 0.0)
        b, m = self.final, self.moves
        x = self.apart(t)
        spot = b.center + self._out * x  # each place, spread out as far as the pile is apart
        rise = (self.layer + 1.0) * self.gap * x  # each place's height
        if len(m):
            k, p, moving, place = self.play.at(t)
        else:
            k, p, moving, place = 0, 1.0, np.zeros(len(b), bool), self.places
        center = spot[place].copy()
        height = rise[place].copy()
        size = b.size[place].copy()
        rotation = b.rotation[place].copy()
        if moving.any():
            i = np.flatnonzero(moving)
            mk = k[i]
            src, dst = m.source[mk], m.target[mk]
            q = p[i]
            e = ease_in_out_cubic(np.clip((q - CLIMB) / (1.0 - 2 * CLIMB), 0.0, 1.0))
            lift = ease_in_out_cubic(np.clip(np.minimum(q, 1.0 - q) / CLIMB, 0.0, 1.0))
            base = rise[src] + (rise[dst] - rise[src]) * e
            height[i] = base + (m.lane[mk] * self.gap - base) * lift  # in its lane mid-move
            center[i] = spot[src] + (spot[dst] - spot[src]) * e[:, None]
            size[i] = b.size[src] + (b.size[dst] - b.size[src]) * e[:, None]
            turn = (b.rotation[dst] - b.rotation[src] + math.pi) % math.tau - math.pi
            rotation[i] = b.rotation[src] + turn * e
        if x <= 0 and not moving.any():  # flat on the table, in stacking order
            return b.replace(center=center, size=size, rotation=rotation,
                             order=np.argsort(place, kind="stable"))  # fmt: skip
        tint = b.tint
        if self.shade > 0:
            tint = b.tint.copy()
            tint[:, :3] = 0.0
            tint[:, 3] = self.shade * x * np.clip(1.0 - height / max(self._top, 1e-9), 0, 1)
        order = np.lexsort((place, height))  # low to high; within a layer, stacking order
        return b.replace(center=center, size=size, rotation=rotation, tint=tint, order=order,
                         height=height, rest=height <= 0)  # fmt: skip


def swap_plan(n: int, home_order: np.ndarray, times: np.ndarray, travel: float,
              rng: np.random.Generator, tilt: float = 0.0, centers=None,
              reach: float = math.inf, layer=None) -> tuple[np.ndarray, Moves]:  # fmt: skip
    """(places, moves) unscrambling n tiles: `home_order` lists the scrambled tiles in the
    order they go home and `times` (one per tile, ascending) when. Tiles with the same
    time go home together, in one batch of simultaneous swaps; batches must be at
    least `travel` apart. Each swap takes a tile home and the tile lying there to the
    first one's old place: that one goes home in a later batch, or, if its home is that
    old place, in the same swap (a two-tile swap; the last batch has only those). So
    every swap puts at least one tile home. A tile that can't go home in its batch (no
    partner) goes home in an earlier one, or, failing that, starts at home. places:
    (n,) where each tile starts (tiles not listed start, and stay, at home). Each move
    tips up to `tilt` radians, either way.

    With `centers` (n, 2), the places' positions, a tile goes home from at most
    `reach` away where it can (else from as near as it can): moves stay short, as if
    a shuffled puzzle were being put right, rather than tiles flying across it. With
    `layer` (n,), each place's layer, swaps only trade places of one layer, and each
    move gets its lane (swap_lanes).

    Built backwards from the finished mosaic, batch by batch from the last: undoing a
    swap trades a tile of the batch (home from then on) with a tile still to go home.
    """
    order = np.asarray(home_order, np.int64)
    times = np.asarray(times, np.float64)
    starts, batch = np.unique(times, return_inverse=True)  # batch start times; each tile's
    home_batch = np.full(n, -1, np.int64)  # batch each tile goes home in (-1: stays home)
    home_batch[order] = batch.reshape(-1)
    layered = layer is not None
    layer = np.asarray(layer, np.int64) if layered else np.zeros(n, np.int64)
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
        n, order, home_batch, len(starts), near_at, near_dist, centers, float(reach), layer,
        layered, int(rng.integers(2**31)))  # fmt: skip
    found = zip(batches[::-1], tiles[::-1], partners[::-1], strict=True)  # first batch first
    swaps = [(float(starts[b]), int(a), int(c)) for b, a, c in found]
    # Replay forwards to record each move's source and target.
    now = places.copy()
    rows = []
    for s, tile, partner in swaps:
        a = now[tile]  # tile's wrong place; partner lies at tile's home
        assert now[partner] == tile
        rows.append((tile, s, a, tile, False))
        rows.append((partner, s, tile, a, True))
        now[tile], now[partner] = tile, a
    if not rows:
        return places, Moves.none()
    tile, start, source, target, low = (np.array(c) for c in zip(*rows, strict=True))
    moves = Moves(
        tile.astype(np.int64),
        start.astype(np.float64),
        source.astype(np.int64),
        target.astype(np.int64),
        low.astype(bool),
        rng.uniform(-tilt, tilt, len(tile)),
    )
    if layer is not None:
        moves = Moves(moves.tile, moves.start, moves.source, moves.target, moves.low,
                      moves.tilt, swap_lanes(moves, layer, rng))  # fmt: skip
    return places, moves


@njit(cache=True, nogil=True)
def _undo_swaps(n, order, home_batch, batches, near_at, near_dist, centers, reach, layer,
                layered, seed):  # fmt: skip
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
            # swap), lying in the tile's layer: within reach at random, else the nearest.
            if near_at.shape[1] == 0 or reach > near_dist[tile, -1]:
                for _ in range(64):  # reach beyond the nearest places: any tile within it
                    y = order[np.random.randint(len(order))]
                    if (home_batch[y] > b and not taken[y] and y != tile
                            and (not layered or layer[places[y]] == layer[tile])):  # fmt: skip
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
                    if ((h > b if later else h == b) and not taken[y] and y != tile
                            and (not layered or layer[places[y]] == layer[tile])):  # fmt: skip
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
                    if ((h > b if later else h == b) and not taken[y] and y != tile
                            and (not layered or layer[places[y]] == layer[tile])):  # fmt: skip
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

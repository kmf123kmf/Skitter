"""Built-in build: the mosaic starts scrambled and invisible hands put it right.

Every tile starts on the table, most of them in another tile's place. Swap by
swap, tiles go home in the chosen order: each swap slides one tile home and
the tile lying there to the first one's old place, until the picture is
whole. Swaps come in batches: every swap of a batch starts at
once (or, with Stagger, each at its own moment over that many seconds:
`stagger_starts`), and once they have all landed the whole mosaic lies still
for a pause before the next batch (`batch_times`). Over the wind-down
batches shrink, down to a single swap at the end.

Every tile keeps a level throughout, sliding over the tiles below it and
under those above (photos in spots not their own may overlap: tiles of
different sizes, a pile), so no tile ever pops over or under another. The
levels are a random order that keeps every tile above those it lies on in
the finished mosaic, so a pile ends in its own order. With Scatter, places
not yet solved lie loose, away from their spots and turned, so the start
doesn't show the shape of the slicing (many small spots where the picture
has detail); a photo going home goes to its true spot. See moves.py for the
moves and the swap plan.
"""

import math

import numpy as np

from skitter.core.animation.base import (
    Choreography,
    Note,
    Timeline,
    pace,
    register_choreography,
    seconds,
)
from skitter.core.animation.landing import random_landing_order, wind_down_param
from skitter.core.animation.look import AnimationLook
from skitter.core.animation.moves import Moves, MovesTimeline, swap_plan
from skitter.core.animation.orders import (
    order_key,
    order_param,
    spread_param,
    sweep_angle_param,
    sweep_param,
)
from skitter.core.scene import MosaicScene
from skitter.core.slicing.common import seed_param
from skitter.core.slicing.params import (
    ChoiceParam,
    FloatParam,
    Section,
    TileSizeParam,
)

# Settings of earlier versions, now gone (tiles slide; piles no longer explode).
PATHS = (
    ("mixed", "Mixed"),
    ("counter", "Counter-clockwise"),
    ("clockwise", "Clockwise"),
    ("straight", "Straight"),
)
BENDS = {"counter": 1.0, "clockwise": -1.0, "straight": 0.0}  # Mixed: either, at random
RETIRED = frozenset({"motion", "lift", "tilt", "stack_height", "explode", "pile_spread",
                     "depth_shading"})  # fmt: skip


def batch_sizes(count: int, batches: int, tail: int) -> np.ndarray:
    """(batches,) tiles going home in each batch, adding up to `count`: alike, then over
    the last `tail` batches shrinking geometrically, down to a single swap's two."""
    if count <= 2 * batches:  # a swap per batch at most, spread out, the last included
        sizes = np.zeros(batches, np.int64)
        swaps = count // 2
        at = np.round(np.linspace(batches - 1, 0, max(swaps, 1), endpoint=False)).astype(int)
        np.add.at(sizes, at[:swaps], 2)
        sizes[batches - 1] += count - sizes.sum()  # an odd one out goes last
        return sizes
    tail = min(max(tail, 0), batches - 1)

    def profile(c: float) -> np.ndarray:  # c per batch, then down to 2 by the last
        shrink = 2.0 * (c / 2.0) ** (1.0 - np.arange(1, tail + 1) / tail) if tail else []
        return np.concatenate([np.full(batches - tail, c), shrink])

    lo, hi = 2.0, float(count)
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if profile(mid).sum() < count else (lo, mid)
    real = profile((lo + hi) / 2)
    real *= count / real.sum()
    edges = np.round(np.concatenate([[0.0], np.cumsum(real)])).astype(np.int64)
    edges[-1] = count
    return np.diff(edges)


def batch_times(count: int, duration: float, travel: float, pause: float,
                wind_down: float = 0.0) -> np.ndarray:  # fmt: skip
    """(count,) when each tile goes home, in batches: as many batches as fit in
    `duration` with each batch's moves (`travel`) and a pause of at least `pause` after
    them, the first starting at 0 and the last ending at `duration`. Batches are
    alike, then shrink over the last `wind_down` seconds (batch_sizes)."""
    window = max(duration - travel, 0.0)  # when the last batch starts
    batches = int(window // max(travel + pause, 1e-9)) + 1
    if batches < 2 or count < 1:
        return np.zeros(count)
    step = window / (batches - 1)  # >= travel + pause
    tail = int(round(wind_down / step)) if wind_down > 0 else 0
    sizes = batch_sizes(count, batches, max(tail, 1 if wind_down > 0 else 0))
    return np.repeat(np.arange(batches) * step, sizes)


def stagger_starts(moves: Moves, stagger: float, rng: np.random.Generator) -> Moves:
    """The same moves, each swap of a batch (moves with the same start; rows 2k and
    2k + 1 are a swap) starting at its own random moment over `stagger` seconds: the
    first at the batch's start, the last `stagger` later. A batch of one swap starts
    on time, except the last, which starts `stagger` late so it still ends the build."""
    if stagger <= 0 or not len(moves):
        return moves
    begin = moves.start[::2]  # each swap's batch start
    batches, batch = np.unique(begin, return_inverse=True)
    batch = batch.reshape(-1)
    draw = rng.random(len(begin))
    low = np.full(len(batches), np.inf)
    high = np.full(len(batches), -np.inf)
    np.minimum.at(low, batch, draw)
    np.maximum.at(high, batch, draw)
    width = (high - low)[batch]
    offset = np.divide(draw - low[batch], width, out=np.zeros_like(draw), where=width > 0)
    offset[(width == 0) & (batch == len(batches) - 1)] = 1.0  # the last batch, one swap
    start = np.repeat(begin + stagger * offset, 2)
    return Moves(moves.tile, start, moves.source, moves.target, moves.bend)


@register_choreography
class UnscrambleChoreography(Choreography):
    id = "unscramble"
    name = "Unscramble"
    phase = "build"
    description = (
        "The mosaic starts scrambled, its tiles in each other's places; invisible hands "
        "slide them back, one tile home with every swap, until the picture is whole."
    )

    _scramble = Section("Scramble")
    scrambled = FloatParam(
        100.0, "Scrambled", min=0.0, max=100.0, step=5.0, decimals=0, suffix="%",
        help="Share of the tiles that start out of place (the rest start at home).",
    )  # fmt: skip
    reach = TileSizeParam(
        6.0, "Reach", min=0.5, max=200.0, step=0.5,
        help="About how far from home tiles start, in base tiles: small for a lightly "
             "shuffled puzzle, large for tiles flying across the whole mosaic. (The last "
             "few tiles may have further to go.)",
    )  # fmt: skip
    scatter = FloatParam(
        50.0, "Scatter", min=0.0, max=100.0, step=5.0, decimals=0, suffix="%",
        help="How loosely tiles out of place lie: 0% in their spots (the scrambled "
             "mosaic keeps the shape of its tiles, e.g. dense where it was split by "
             "detail), 100% anywhere over the mosaic, turned any way. A tile going home "
             "goes to its true spot.",
    )  # fmt: skip

    _order = Section("Order")
    order = order_param("random", "go home")
    sweep = sweep_param()
    sweep_angle = sweep_angle_param()
    spread = spread_param(0.05, "order tiles go home in")

    _timing = Section("Timing")
    duration = FloatParam(
        10.0, "Duration", min=0.5, max=3600.0, step=0.5, decimals=1, suffix=" s",
        help="From the first swap until the last tile is home.",
    )  # fmt: skip
    travel = FloatParam(
        0.8, "Move time", min=0.1, max=10.0, step=0.1, decimals=2, suffix=" s",
        help="How long each move takes (at most the duration).",
    )  # fmt: skip
    paths = ChoiceParam(
        "mixed", "Paths", choices=PATHS,
        help="How the two tiles of a swap pass each other: curving aside, the pair "
             "turning around its middle one way or the other (Mixed: each swap at "
             "random, so the moves don't seem to swirl around the mosaic), or straight, "
             "one over the other.",
    )  # fmt: skip
    pause = FloatParam(
        0.5, "Pause", min=0.0, max=10.0, step=0.1, decimals=2, suffix=" s",
        help="Swaps come in batches; from the last swap of a batch landing to the next "
             "batch, the whole mosaic lies still at least this long. (As many batches "
             "as fit in the duration.)",
    )  # fmt: skip
    stagger = FloatParam(
        0.0, "Stagger", min=0.0, max=10.0, step=0.1, decimals=2, suffix=" s",
        help="The swaps of a batch start at random moments over this long, instead of "
             "all at once, so they land at different times too (0: a batch moves as "
             "one). The pause starts when the last of them lands.",
    )  # fmt: skip
    wind_down = wind_down_param(
        0.0,
        "Batches shrink over this last share of the duration, so the last few "
        "swaps happen one by one (0: batches alike to the end).",
    )

    _end = Section()
    seed = seed_param()

    @classmethod
    def from_values(cls, values: dict, problems: list[str] | None = None, what: str = ""):
        if "stack_height" in values and "pile_spread" not in values:
            values = {k: v for k, v in values.items() if k != "spread"}  # was Pile spread
        values = {k: v for k, v in values.items() if k not in RETIRED}
        return super().from_values(values, problems, what)

    def timeline(self, scene: MosaicScene, look: AnimationLook | None = None) -> Timeline:
        rng = np.random.default_rng(self.seed)
        n = len(scene)
        travel = min(self.travel, self.duration)
        stagger = min(self.stagger, self.duration - travel)
        count = int(round(n * self.scrambled / 100.0))
        chosen = np.zeros(n, bool)
        chosen[rng.choice(n, size=count, replace=False)] = True
        # By the order's preference (lower first), each tile's place loosened by Spread.
        key = np.argsort(np.argsort(order_key(scene, self, rng), kind="stable"))
        key = key / max(n - 1, 1) + rng.uniform(-1.0, 1.0, n) * self.spread
        home_order = np.flatnonzero(chosen)[np.argsort(key[chosen], kind="stable")]
        # A batch moves for its swaps' time plus the stagger.
        times = batch_times(len(home_order), self.duration, travel + stagger, self.pause,
                            self.wind_down * self.duration)  # fmt: skip
        places, moves = swap_plan(n, home_order, times, rng, scene.center,
                                  self.reach * scene.tile_size[0])  # fmt: skip
        # Levels: random, every tile above those it lies on in the finished mosaic.
        levels = random_landing_order(scene, int(rng.integers(2**31)))
        # Each swap's pair curves alike: Mixed draws a way for each swap.
        swaps = len(moves) // 2
        if self.paths == "mixed":
            way = rng.choice([-1.0, 1.0], swaps)
        else:
            way = np.full(swaps, BENDS[self.paths])
        moves = moves.bent(np.repeat(way, 2))
        # Loose places: toward a random point of the mosaic, turned, by Scatter.
        x0, y0, x1, y1 = scene.bounds
        anywhere = rng.uniform((x0, y0), (x1, y1), (n, 2))
        share = self.scatter / 100.0
        loose = scene.center + share * (anywhere - scene.center)
        loose_turn = share * rng.uniform(-math.pi, math.pi, n)
        # Last, so changing the stagger changes nothing else.
        moves = stagger_starts(moves, stagger, rng)
        timeline = MovesTimeline(scene, places, moves, travel, self.duration, levels, loose,
                                 loose_turn)  # fmt: skip
        timeline.notes = tuple(self._notes(times, travel, stagger))
        return timeline

    def _notes(self, times: np.ndarray, travel: float, stagger: float) -> list[Note]:
        """What the timing settings came to (Timeline.notes), from when each tile goes
        home (`times`: batch_times)."""
        notes = []
        if travel < self.travel:
            notes.append(Note("travel", f"Shortened to {seconds(travel)}, the duration.", True))
        if stagger < self.stagger - 1e-9:
            notes.append(Note("stagger", f"Shortened to {seconds(stagger)}: with the move "
                                         "time, it must fit in the duration.", True))  # fmt: skip
        starts, sizes = np.unique(times, return_counts=True)
        if len(starts) < 2:
            if len(times) > 2:
                batch = seconds(self.travel + self.stagger + self.pause)
                notes.append(Note("duration", f"Only one batch fits (its moves and the pause "
                                              f"take {batch}): every swap in one go.",
                                  True))  # fmt: skip
            return notes
        gaps = np.diff(starts)
        notes.append(Note("duration", f"{len(starts)} batches of about "
                                      f"{len(times) / 2 / len(starts):.0f} swaps, "
                                      f"{pace(float(np.median(gaps)))}.", False))  # fmt: skip
        # The shortest stillness between batches (a batch of one swap moves for less).
        still = float(gaps.min()) - travel - stagger
        if still > 1.1 * self.pause + 0.02:  # (not just rounding)
            notes.append(Note("pause", f"{seconds(still)} or more: the batches spread out to "
                                       "fill the duration."))  # fmt: skip
        if self.wind_down > 0:
            if sizes.max() <= 2:
                notes.append(Note("wind_down", "No effect: every batch is already a single "
                                               "swap."))  # fmt: skip
            else:
                shrinking = int((sizes < np.median(sizes) - 1).sum())  # (not rounding)
                tail = seconds(self.wind_down * self.duration)
                last = (
                    "the last batch shrinks"
                    if shrinking == 1
                    else f"the last {shrinking} batches shrink"
                )
                notes.append(Note("wind_down", f"= {tail}: {last}."))
        return notes

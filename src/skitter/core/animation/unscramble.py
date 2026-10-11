"""Built-in build: the mosaic starts scrambled and invisible hands put it right.

Every tile starts on the table, most of them in another tile's place. Swap by
swap, tiles go home in the chosen order: each swap carries (or slides) one
tile home and the tile lying there to the first one's old place, until the
picture is whole. Swaps come in batches: every swap of a batch starts at
once, and once they have all landed the whole mosaic lies still for a pause
before the next batch (`batch_times`). Over the wind-down batches shrink,
down to a single swap at the end.

A pile (tiles lying on each other) explodes first: its layers rise apart and
spread out, the swaps happen in the gaps between them, and the pile settles
back (moves.LayeredTimeline). See moves.py for the moves and the swap plan.
"""

import math

import numpy as np

from skitter.core.animation.base import Choreography, Timeline, register_choreography
from skitter.core.animation.landing import wind_down_param
from skitter.core.animation.look import AnimationLook, scene_extent
from skitter.core.animation.moves import LayeredTimeline, MovesTimeline, layers, swap_plan
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

LIFT_LIMIT = 0.25  # carried tiles rise at most this share of the camera height
MOTIONS = (("carry", "Carry"), ("slide", "Slide"))


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


@register_choreography
class UnscrambleChoreography(Choreography):
    id = "unscramble"
    name = "Unscramble"
    phase = "build"
    photos_take_spots = True  # a pile's photos take on the spots they lie in
    description = (
        "The mosaic starts scrambled, its tiles in each other's places; invisible hands "
        "swap them back, one tile home with every swap, until the picture is whole."
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

    _motion = Section("Motion")
    motion = ChoiceParam(
        "carry", "Motion", choices=MOTIONS,
        help="Carry: tiles are lifted over the mosaic, casting shadows, and set down. "
             "Slide: they glide across it. (A pile explodes instead: see Stack height.)",
    )  # fmt: skip
    lift = TileSizeParam(
        1.5, "Lift", min=0.1, max=10.0, step=0.1,
        under="motion", when=lambda c: c.motion == "carry",
        help="How high carried tiles rise, in base tiles (one of two trading places "
             "flies lower, so they pass one over the other).",
    )  # fmt: skip
    tilt = FloatParam(
        8.0, "Tilt", min=0.0, max=45.0, step=1.0, decimals=0, suffix="°",
        under="motion", when=lambda c: c.motion == "carry",
        help="How far a carried tile turns on the way, either way, as if held by hand; "
             "it lands straight.",
    )  # fmt: skip

    _order = Section("Order")
    order = order_param("random", "go home")
    sweep = sweep_param()
    sweep_angle = sweep_angle_param()
    spread = spread_param(0.05, "order tiles go home in")

    _timing = Section("Timing")
    duration = FloatParam(
        10.0, "Duration", min=0.5, max=3600.0, step=0.5, decimals=1, suffix=" s",
        help="From the first swap until the last tile is home (a pile: from coming "
             "apart to settling back).",
    )  # fmt: skip
    travel = FloatParam(
        0.8, "Move time", min=0.1, max=10.0, step=0.1, decimals=2, suffix=" s",
        help="How long each move takes (at most the duration).",
    )  # fmt: skip
    pause = FloatParam(
        0.5, "Pause", min=0.0, max=10.0, step=0.1, decimals=2, suffix=" s",
        help="Swaps come in batches, all of a batch at once; between batches the whole "
             "mosaic lies still at least this long. (As many batches as fit in the "
             "duration.)",
    )  # fmt: skip
    wind_down = wind_down_param(
        0.0,
        "Batches shrink over this last share of the duration, so the last few "
        "swaps happen one by one (0: batches alike to the end).",
    )

    _pile = Section("Pile", "Only when tiles lie on each other.")
    stack_height = FloatParam(
        0.25, "Stack height", min=0.02, max=1.0, step=0.05, suffix=" × mosaic",
        help="A pile (tiles lying on each other) first comes apart into evenly spaced "
             "layers, so tiles can swap between them; this is how high its top layer rises.",
    )  # fmt: skip
    explode = FloatParam(
        1.5, "Explode time", min=0.1, max=30.0, step=0.1, decimals=2, suffix=" s",
        help="A pile: how long it takes to come apart at the start, and to settle back "
             "at the end (both within the duration).",
    )  # fmt: skip
    pile_spread = FloatParam(
        15.0, "Pile spread", min=0.0, max=100.0, step=5.0, decimals=0, suffix="%",
        help="A pile: how far its top layer spreads out from the middle while apart "
             "(lower layers less), opening gaps to see the layers below.",
    )  # fmt: skip
    depth_shading = FloatParam(
        30.0, "Depth shading", min=0.0, max=100.0, step=5.0, decimals=0, suffix="%",
        help="A pile: how much darker its bottom layer looks while apart (layers above "
             "it less), so the stack reads as deep.",
    )  # fmt: skip

    _end = Section()
    seed = seed_param()

    @classmethod
    def from_values(cls, values: dict, problems: list[str] | None = None, what: str = ""):
        if "pile_spread" not in values and "spread" in values:  # saved before Pile spread
            values = {**values, "pile_spread": values["spread"]}
            del values["spread"]
        return super().from_values(values, problems, what)

    def timeline(self, scene: MosaicScene, look: AnimationLook | None = None) -> Timeline:
        look = look or AnimationLook()
        rng = np.random.default_rng(self.seed)
        n = len(scene)
        travel = min(self.travel, self.duration)
        count = int(round(n * self.scrambled / 100.0))
        chosen = np.zeros(n, bool)
        chosen[rng.choice(n, size=count, replace=False)] = True
        # By the order's preference (lower first), each tile's place loosened by Spread.
        key = np.argsort(np.argsort(order_key(scene, self, rng), kind="stable"))
        key = key / max(n - 1, 1) + rng.uniform(-1.0, 1.0, n) * self.spread
        home_order = np.flatnonzero(chosen)[np.argsort(key[chosen], kind="stable")]
        layer = layers(scene)
        piled = n > 0 and layer.max() > 0
        explode = min(self.explode, self.duration / 2) if piled else 0.0
        window = self.duration - 2 * explode  # the swaps' time
        travel = min(travel, window) if piled else travel
        times = batch_times(len(home_order), window, travel, self.pause,
                            self.wind_down * window)  # fmt: skip
        places, moves = swap_plan(n, home_order, times, travel, rng, math.radians(self.tilt),
                                  scene.center, self.reach * scene.tile_size[0],
                                  layer if piled else None)  # fmt: skip
        if piled:
            top = min(self.stack_height * scene_extent(scene),
                      LIFT_LIMIT * look.camera_height * scene_extent(scene))  # fmt: skip
            return LayeredTimeline(
                scene, places, moves.delayed(explode), travel, self.duration, layer, explode,
                top / (layer.max() + 1), self.pile_spread / 100.0, self.depth_shading / 100.0,
            )  # fmt: skip
        lift = min(self.lift * scene.tile_size[0],
                   LIFT_LIMIT * look.camera_height * scene_extent(scene))  # fmt: skip
        return MovesTimeline(scene, places, moves, travel, self.duration,
                             carry=self.motion == "carry", lift=lift)  # fmt: skip

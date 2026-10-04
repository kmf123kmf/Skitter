"""Built-in choreography: tiles fly in from all around, or fall from the camera, and land."""

import math

import numpy as np

from skitter.core.animation.base import (
    Choreography,
    FlightTimeline,
    Flips,
    TileFrame,
    Timeline,
    TossTimeline,
    cover_after_impact,
    register_choreography,
)
from skitter.core.animation.landing import landing_order, random_landing_order
from skitter.core.animation.look import NEAR, AnimationLook, clear_of_axis, scene_extent
from skitter.core.easing import ease_out_back, ease_out_cubic
from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import BoolParam, ChoiceParam, FloatParam, IntParam, RangeParam

TOSS_LIMIT = 0.8  # throws peak at most this share of the camera height (5x size)

# Orders without a direction: they land in a uniformly random bottom-first order
# (random_landing_order), never through landing_order (see landing.py). A new
# ordering must be listed here if it is random; tests hold both kinds to account.
RANDOM_ORDERS = frozenset({"random"})

ORDERS = (
    ("center", "Center outward"),
    ("edges", "Edges inward"),
    ("reading", "Reading order"),
    ("dark", "Dark to light"),
    ("light", "Light to dark"),
    ("random", "Random"),
)


@register_choreography
class AssembleChoreography(Choreography):
    id = "assemble"
    name = "Assemble"
    description = (
        "Tiles fly in from all around the mosaic, spinning, and land in place: tossed "
        "onto the table under gravity (with bounces and a settling wobble), dropped "
        "from the camera, or gliding in. Overlapping tiles always land bottom first."
    )

    motion = ChoiceParam(
        "toss", "Motion",
        choices=[("toss", "Toss (gravity)"), ("drop", "Drop (from the camera)"),
                 ("glide", "Glide")],
        help="Toss: thrown in an arc, falling faster and faster onto the table, then "
             "bouncing. Drop: falls from beside the camera, looming large at first and "
             "shrinking onto its spot. Glide: slides in flat and slows to a stop.",
    )  # fmt: skip

    order = ChoiceParam("center", "Order", choices=ORDERS, help="Which tiles land first.")
    duration = FloatParam(
        8.0, "Duration", min=0.5, max=600.0, step=0.5, decimals=1, suffix=" s",
        help="From the first tile leaving to the last tile coming to rest."
    )  # fmt: skip
    travel = FloatParam(
        1.5, "Flight time", min=0.05, max=60.0, step=0.1, decimals=2, suffix=" s",
        help="How long each tile is in the air (at most the duration).",
    )  # fmt: skip
    spread = FloatParam(
        0.15, "Spread", min=0.0, max=1.0, step=0.05,
        when=lambda c: c.order not in RANDOM_ORDERS,
        help="Randomness in the landing order, as a share of the whole sequence.",
    )  # fmt: skip
    distance = FloatParam(
        1.5, "Distance", min=0.0, max=10.0, step=0.25, suffix=" × mosaic",
        when=lambda c: c.motion != "drop",
        help="How far out tiles start, in mosaic sizes from the center.",
    )  # fmt: skip
    slant = FloatParam(
        0.25, "Slant", min=0.0, max=3.0, step=0.05, suffix=" × mosaic",
        when=lambda c: c.motion == "drop",
        help="How far beside its spot a tile starts, outward from the middle. 0 drops "
             "straight down (tiles near the middle start just far enough out to come in "
             "from the edge): they loom huge and cover the view. More slides tiles in "
             "from the edges of the picture, smaller when they appear.",
    )  # fmt: skip
    spin = RangeParam(
        (0.0, 1.0), "Spin", min=0.0, max=20.0, step=0.25, suffix=" turns",
        help="How far each tile turns on the way, either way: each tile draws its own "
             "amount from this range (equal ends: all the same).",
    )  # fmt: skip
    flips = RangeParam(
        (0, 0), "Flips", min=0, max=8, step=1, whole=True,
        help="Whole times each tile turns over in flight (about one of its edges), "
             "showing its back on the way; each tile draws its own number from this range. "
             "Tiles always land face up, the right way round.",
    )  # fmt: skip
    arc = FloatParam(
        0.4, "Arc height", min=0.0, max=5.0, step=0.05, suffix=" × mosaic",
        when=lambda c: c.motion == "toss",
        help="How high tiles fly at the top of their arc. Higher arcs fall harder: "
             "gravity follows from this and the flight time.",
    )  # fmt: skip
    bounce = RangeParam(
        (0.2, 0.4),
        "Bounce",
        min=0.0,
        max=0.8,
        step=0.05,
        when=lambda c: c.motion != "glide",
        help="How springy a landing is: each bounce keeps this share of the speed "
        "(0: no bounce); each tile draws its own from this range. A tile that "
        "another lands on stops hopping before it, and soon stops rocking.",
    )
    wobble = RangeParam(
        (2.0, 6.0), "Settle wobble", min=0.0, max=30.0, step=1.0, decimals=0, suffix="°",
        when=lambda c: c.motion != "glide",
        help="A small rocking turn that dies away after landing, either way: each tile "
             "draws its own amount from this range (equal ends: all the same).",
    )  # fmt: skip
    shrink = FloatParam(
        0.3, "Start size", min=0.0, max=10.0, step=0.1, suffix=" ×",
        when=lambda c: c.motion == "glide",
        help="Size of a tile at the start of its flight, relative to its final size.",
    )  # fmt: skip
    overshoot = BoolParam(
        False, "Overshoot", when=lambda c: c.motion == "glide",
        help="Glide slightly past the spot and settle back.",
    )  # fmt: skip
    seed = IntParam(1, "Seed", min=0, max=999_999)

    def timeline(self, scene: MosaicScene, look: AnimationLook | None = None) -> Timeline:
        n = len(scene)
        rng = np.random.default_rng(self.seed)
        look = look if look is not None else AnimationLook()
        extent = scene_extent(scene)
        drop = self.motion == "drop"
        camera = look.camera_height * extent
        start_height = camera if drop else 0.0  # at the camera, at rest
        # A throw never reaches the camera (it would pass the lens and pop into view).
        apex = start_height if drop else min(self.arc * extent, TOSS_LIMIT * camera)
        tossed = self.motion != "glide"
        params = type(self)
        # Bouncing and settling take a share of the flight time (at most this, for the
        # springiest tile); every tile's whole trip must fit in the duration.
        share = (
            TossTimeline.settle_time(
                1.0, apex, self.bounce[1], math.radians(self.wobble[1]), start_height
            )  # fmt: skip
            if tossed
            else 0
        )
        travel = min(self.travel, self.duration / (1.0 + share))

        # Preferred place in the sequence (0..1), loosened by the spread; overlapping
        # tiles still land bottom first. Landings are evenly paced.
        if self.order in RANDOM_ORDERS:  # uniformly random among bottom-first orders
            order = random_landing_order(scene, self.seed)
        else:
            key = self._order_key(scene, rng)
            preferred = np.empty(n)
            preferred[np.argsort(key, kind="stable")] = np.linspace(0.0, 1.0, n) if n else []
            preferred += rng.uniform(-1.0, 1.0, n) * self.spread
            order = landing_order(scene, preferred)
        place = order / max(n - 1, 1)  # 0: lands first, 1: last
        window = max(self.duration - travel, 0.0)  # when the last tile lands, after the first
        if tossed:  # each tile's own bounce and rocking, drawn from the ranges
            bounce = params.bounce.draw(self.bounce, rng, n)
            wobble = (rng.choice([-1.0, 1.0], n)  # rocking either way first
                      * np.radians(params.wobble.draw(self.wobble, rng, n)))  # fmt: skip
            # Room for each tile to settle after landing, with nothing on top (a tile
            # something lands on settles sooner). Whether one will be covered depends
            # only on the order, so the window comes from the others: the last to come
            # to rest does so exactly at the end. All of them, if a covered tile still
            # wouldn't fit (a cover that itself settles very fast).
            free = TossTimeline.settle_times(travel, apex, bounce, wobble, start_height)
            room = self.duration - travel - free
            covered = np.zeros(n, bool)
            covered[scene.overlaps[:, 0]] = True

            def fit(tiles) -> float:
                tiles = tiles & (place > 0)
                if not tiles.any():
                    return window
                return max(min(window, float((room[tiles] / place[tiles]).min())), 0.0)

            windows = (fit(~covered), fit(np.ones(n, bool)))
        delay = place * window

        x0, y0, x1, y1 = scene.bounds
        middle = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        reach = self.distance * math.hypot(x1 - x0, y1 - y0) / 2
        angle = rng.uniform(0.0, 2 * math.pi, n)
        radius = reach * rng.uniform(1.0, 1.6, n)
        final = TileFrame.final(scene)
        start_center = middle + radius[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        way = rng.choice([-1.0, 1.0], n)  # each tile spins either way
        start_rotation = final.rotation + way * params.spin.draw(self.spin, rng, n) * 2 * math.pi
        flips = Flips(  # random direction and edge per tile, always whole turns
            turns=params.flips.draw(self.flips, rng, n) * rng.choice([-1.0, 1.0], n),
            axis=rng.integers(0, 2, n),
        )
        if drop:
            start_center = self._drop_starts(scene, final, middle, extent, rng)
        if tossed:
            for window in windows:
                delay = place * window
                cover = cover_after_impact(scene, delay + travel)  # stacked tiles stop hopping
                timeline = TossTimeline(
                    scene, start_center, start_rotation, delay, travel, apex, bounce, wobble,
                    flips, start_height, cover,
                )  # fmt: skip
                if timeline.duration <= self.duration + 1e-9:
                    break
            return timeline
        start = final.replace(
            center=start_center,
            size=final.size * self.shrink,
            rotation=start_rotation,
            alpha=np.zeros(n),
        )
        easing = ease_out_back if self.overshoot else ease_out_cubic
        return FlightTimeline(scene, start, delay, travel, easing=easing, flips=flips)

    def _drop_starts(self, scene, final: TileFrame, middle, extent: float, rng) -> np.ndarray:
        """Where dropped tiles start: beside their spots by the slant, outward from the
        middle (the camera's axis), a little varied, and always far enough out to be off
        the picture when they come into view below the near plane (see look.py)."""
        n = len(final)
        out = final.center - middle
        angle = np.where(
            np.hypot(out[:, 0], out[:, 1]) > 1e-9 * extent,
            np.arctan2(out[:, 1], out[:, 0]),
            rng.uniform(0.0, 2 * math.pi, n),  # right under the camera: any way
        )
        angle = angle + rng.uniform(-0.35, 0.35, n)  # about 20 degrees either way
        way = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        reach = self.slant * extent * rng.uniform(0.8, 1.2, n)
        # Falling from rest, a tile is at the near plane after this share of its fall,
        # still this share of its sideways way from its spot.
        left = 1.0 - math.sqrt(1.0 - NEAR)
        # Smallest k = reach x left with |out + k way| >= clear (a little to spare).
        clear = 1.05 * clear_of_axis(scene)
        along = (out * way).sum(axis=1)
        k = -along + np.sqrt(np.maximum(along**2 - (out**2).sum(axis=1) + clear**2, 0.0))
        reach = np.maximum(reach, np.maximum(k, 0.0) / left)
        return final.center + reach[:, None] * way

    def _order_key(self, scene: MosaicScene, rng: np.random.Generator) -> np.ndarray:
        if self.order == "center":
            return scene.distance
        if self.order == "edges":
            return -scene.distance
        if self.order == "reading":
            return scene.reading_order.astype(np.float64)
        if self.order == "dark":
            return scene.lightness
        if self.order == "light":
            return -scene.lightness
        return rng.random(len(scene))

    def summary(self) -> str:
        return f"{dict(ORDERS)[self.order]}, {self.duration:g} s"

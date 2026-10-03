"""Built-in choreography: tiles fly in from all around and land in place."""

import math

import numpy as np

from skitter.core.animation.base import (
    Choreography,
    FlightTimeline,
    Flips,
    TileFrame,
    Timeline,
    TossTimeline,
    landing_order,
    register_choreography,
)
from skitter.core.animation.look import scene_extent
from skitter.core.easing import ease_out_back, ease_out_cubic
from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import BoolParam, ChoiceParam, FloatParam, IntParam

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
        "onto the table under gravity (with bounces and a settling wobble), or gliding "
        "in. Overlapping tiles always land bottom first."
    )

    motion = ChoiceParam(
        "toss", "Motion", choices=[("toss", "Toss (gravity)"), ("glide", "Glide")],
        help="Toss: thrown in an arc, falling faster and faster onto the table, then "
             "bouncing. Glide: slides in flat and slows to a stop.",
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
        help="Randomness in the landing order, as a share of the whole sequence.",
    )  # fmt: skip
    distance = FloatParam(
        1.5, "Distance", min=0.0, max=10.0, step=0.25, suffix=" × mosaic",
        help="How far out tiles start, in mosaic sizes from the center.",
    )  # fmt: skip
    spin = FloatParam(1.0, "Spin", min=0.0, max=20.0, step=0.25, suffix=" turns")
    flips = IntParam(
        0, "Flips", min=0, max=8,
        help="Whole times each tile turns over in flight (about one of its edges), "
             "showing its back on the way. Tiles always land face up, the right way round.",
    )  # fmt: skip
    arc = FloatParam(
        0.4, "Arc height", min=0.0, max=5.0, step=0.05, suffix=" × mosaic",
        when=lambda c: c.motion == "toss",
        help="How high tiles fly at the top of their arc. Higher arcs fall harder: "
             "gravity follows from this and the flight time.",
    )  # fmt: skip
    bounce = FloatParam(
        0.3, "Bounce", min=0.0, max=0.8, step=0.05, when=lambda c: c.motion == "toss",
        help="How springy a landing is: each bounce keeps this share of the speed "
             "(0: no bounce).",
    )  # fmt: skip
    wobble = FloatParam(
        4.0, "Settle wobble", min=0.0, max=30.0, step=1.0, decimals=0, suffix="°",
        when=lambda c: c.motion == "toss",
        help="A small rocking turn that dies away after landing.",
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

    def timeline(self, scene: MosaicScene) -> Timeline:
        n = len(scene)
        rng = np.random.default_rng(self.seed)
        apex = self.arc * scene_extent(scene)
        wobble = math.radians(self.wobble)
        # Bouncing and settling take a fixed share of the flight time; a tile's whole
        # trip must fit in the duration.
        share = (
            TossTimeline.settle_time(1.0, apex, self.bounce, wobble) if self.motion == "toss" else 0
        )
        travel = min(self.travel, self.duration / (1.0 + share))
        settle = share * travel
        window = max(self.duration - travel - settle, 0.0)

        # Preferred place in the sequence (0..1), loosened by the spread; overlapping
        # tiles still land bottom first. Landings are evenly paced.
        key = self._order_key(scene, rng)
        preferred = np.empty(n)
        preferred[np.argsort(key, kind="stable")] = np.linspace(0.0, 1.0, n) if n else []
        preferred += rng.uniform(-1.0, 1.0, n) * self.spread
        order = landing_order(scene, preferred)
        delay = order / max(n - 1, 1) * window

        x0, y0, x1, y1 = scene.bounds
        middle = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        reach = self.distance * math.hypot(x1 - x0, y1 - y0) / 2
        angle = rng.uniform(0.0, 2 * math.pi, n)
        radius = reach * rng.uniform(1.0, 1.6, n)
        final = TileFrame.final(scene)
        start_center = middle + radius[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        start_rotation = final.rotation + rng.uniform(-1.0, 1.0, n) * self.spin * 2 * math.pi
        flips = Flips(  # random direction and edge per tile, always whole turns
            turns=self.flips * rng.choice([-1.0, 1.0], n), axis=rng.integers(0, 2, n)
        )
        if self.motion == "toss":
            return TossTimeline(scene, start_center, start_rotation, delay, travel, apex,
                                self.bounce, wobble, flips)  # fmt: skip
        start = final.replace(
            center=start_center,
            size=final.size * self.shrink,
            rotation=start_rotation,
            alpha=np.zeros(n),
        )
        easing = ease_out_back if self.overshoot else ease_out_cubic
        return FlightTimeline(scene, start, delay, travel, easing=easing, flips=flips)

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

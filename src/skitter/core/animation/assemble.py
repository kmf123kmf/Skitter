"""Built-in choreography: tiles fly in from all around and land in place."""

import math

import numpy as np

from skitter.core.animation.base import (
    Choreography,
    FlightTimeline,
    TileFrame,
    Timeline,
    landing_order,
    register_choreography,
)
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
        "Tiles fly in from all around the mosaic, spinning, and land in place. "
        "Overlapping tiles always land bottom first."
    )

    order = ChoiceParam("center", "Order", choices=ORDERS, help="Which tiles land first.")
    duration = FloatParam(
        8.0, "Duration", min=0.5, max=600.0, step=0.5, decimals=1, suffix=" s",
        help="From the first tile leaving to the last tile landing.",
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
    shrink = FloatParam(
        0.3, "Start size", min=0.0, max=10.0, step=0.1, suffix=" ×",
        help="Size of a tile at the start of its flight, relative to its final size.",
    )  # fmt: skip
    bounce = BoolParam(False, "Bounce", help="Overshoot slightly and settle when landing.")
    seed = IntParam(1, "Seed", min=0, max=999_999)

    def timeline(self, scene: MosaicScene) -> Timeline:
        n = len(scene)
        rng = np.random.default_rng(self.seed)
        travel = min(self.travel, self.duration)
        window = self.duration - travel

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
        start = final.replace(
            center=middle + radius[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1),
            size=final.size * self.shrink,
            rotation=final.rotation + rng.uniform(-1.0, 1.0, n) * self.spin * 2 * math.pi,
            alpha=np.zeros(n),
        )
        easing = ease_out_back if self.bounce else ease_out_cubic
        return FlightTimeline(scene, start, delay, travel, easing=easing)

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

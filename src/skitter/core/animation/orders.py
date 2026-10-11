"""The orders builds share: which tiles land (or go home) first.

A build declares its order settings with the functions below, in the place
they belong among its own settings (a Param belongs to one class: each call
makes one); the settings are named `order`, `sweep`, `sweep_angle` and
`spread`, and `order_key` reads them. A build may put orders of its own
before ORDERS (Deal's Nearest first), handling them itself.

Two kinds of order: those with a direction give each tile a preference
(`order_key`, lower first), which `landing_ranks` turns into a valid
bottom-first sequence; random ones (RANDOM_ORDERS) have none, and land in a
uniformly random bottom-first order (see landing.py).
"""

import math

import numpy as np

from skitter.core.animation import compass
from skitter.core.animation.landing import landing_order, random_landing_order
from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import ChoiceParam, FloatParam

# Orders without a direction: they land in a uniformly random bottom-first order
# (random_landing_order), never through landing_order (see landing.py). A new
# ordering must be listed here if it is random; tests hold both kinds to account.
RANDOM_ORDERS = frozenset({"random"})
SWEEP = "sweep"

ORDERS = (
    ("center", "Center outward"),
    ("edges", "Edges inward"),
    (SWEEP, "Sweep"),
    ("reading", "Rows (reading order)"),
    ("clockwise", "Clockwise"),
    ("counter", "Counter-clockwise"),
    ("dark", "Dark to light"),
    ("light", "Light to dark"),
    ("random", "Random"),
)


def order_param(default: str, what: str = "land", extra=()) -> ChoiceParam:
    """The order; `extra`: the build's own orders, listed first."""
    return ChoiceParam(
        default, "Order", choices=(*extra, *ORDERS),
        help=f"Which tiles {what} first. Sweep: a straight line crossing the picture. "
             "Rows: row by row from the top, each left to right. Clockwise and "
             "counter-clockwise: round the middle, from straight up.",
    )  # fmt: skip


def sweep_param() -> ChoiceParam:
    return compass.direction_param("down", "sweep", when=lambda c: c.order == SWEEP,
                                   under="order")  # fmt: skip


def sweep_angle_param() -> FloatParam:
    return compass.angle_param(180.0, "sweep", "sweep", when=lambda c: c.order == SWEEP)


def spread_param(default: float, what: str = "landing order") -> FloatParam:
    return FloatParam(
        default, "Spread", min=0.0, max=1.0, step=0.05,
        under="order", when=lambda c: c.order not in RANDOM_ORDERS,
        help=f"Randomness in the {what}, as a share of the whole sequence.",
    )  # fmt: skip


def order_key(scene: MosaicScene, settings, rng: np.random.Generator) -> np.ndarray:
    """(N,) each tile's preference for the order `settings` choose (lower first)."""
    order = settings.order
    if order == "center":
        return scene.distance
    if order == "edges":
        return -scene.distance
    if order == SWEEP:
        return scene.center @ compass.heading(compass.degrees(settings.sweep,
                                                              settings.sweep_angle))  # fmt: skip
    if order == "reading":
        return scene.reading_order.astype(np.float64)
    if order in ("clockwise", "counter"):
        offset = scene.center - np.asarray(scene.canvas) / 2
        turn = np.arctan2(offset[:, 0], -offset[:, 1])  # clockwise from straight up
        return (turn if order == "clockwise" else -turn) % math.tau
    if order == "dark":
        return scene.lightness
    if order == "light":
        return -scene.lightness
    return rng.random(len(scene))


def describe_order(settings, extra=()) -> str:
    """The order for a summary: "Center outward", "Sweep down"."""
    if settings.order == SWEEP:
        return f"Sweep {compass.describe(settings.sweep, settings.sweep_angle)}"
    return dict((*extra, *ORDERS))[settings.order]


def landing_ranks(scene: MosaicScene, key, spread: float, seed: int,
                  rng: np.random.Generator) -> np.ndarray:  # fmt: skip
    """(N,) each tile's place in a bottom-first landing sequence: by `key` (lower
    first), each tile's place loosened by `spread` (a share of the sequence); or,
    with no key, a uniformly random one (see landing.py)."""
    if key is None:
        return random_landing_order(scene, seed)
    n = len(scene)
    preferred = np.empty(n)
    preferred[np.argsort(key, kind="stable")] = np.linspace(0.0, 1.0, n) if n else []
    preferred += rng.uniform(-1.0, 1.0, n) * spread
    return landing_order(scene, preferred)

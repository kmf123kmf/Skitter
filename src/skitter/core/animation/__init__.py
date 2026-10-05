"""Animated construction of the finished mosaic (see base.py).

Nothing here imports Qt: a timeline only computes tile states, which the
UI draws as sprites (or a renderer could draw frame by frame).
"""

from skitter.core.animation import assemble, deal  # registers the built-in choreographies
from skitter.core.animation.base import (
    Choreography,
    FlightTimeline,
    Flips,
    TileFrame,
    Timeline,
    TossTimeline,
    choreography_types,
    cover_after_impact,
    get_choreography,
    register_choreography,
)
from skitter.core.animation.landing import landing_order, landing_times, random_landing_order

__all__ = [
    "Choreography",
    "FlightTimeline",
    "Flips",
    "Timeline",
    "TossTimeline",
    "cover_after_impact",
    "TileFrame",
    "assemble",
    "deal",
    "choreography_types",
    "get_choreography",
    "landing_order",
    "landing_times",
    "random_landing_order",
    "register_choreography",
]

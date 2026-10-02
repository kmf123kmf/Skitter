"""Animated construction of the finished mosaic (see base.py).

Nothing here imports Qt: a timeline only computes tile states, which the
UI draws as sprites (or a renderer could draw frame by frame).
"""

from skitter.core.animation import assemble  # registers the built-in choreographies
from skitter.core.animation.base import (
    Choreography,
    FlightTimeline,
    TileFrame,
    Timeline,
    choreography_types,
    get_choreography,
    register_choreography,
)

__all__ = [
    "Choreography",
    "FlightTimeline",
    "Timeline",
    "TileFrame",
    "assemble",
    "choreography_types",
    "get_choreography",
    "register_choreography",
]

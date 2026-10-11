"""Directions across the picture, as a Wave travels or a Sweep order lands.

A direction is a compass choice (or a custom angle), in degrees clockwise
from the top of the picture, as the light's direction is given.
"""

import math

import numpy as np

from skitter.core.slicing.params import ChoiceParam, FloatParam

COMPASS = {"up": 0.0, "up_right": 45.0, "right": 90.0, "down_right": 135.0, "down": 180.0,
           "down_left": 225.0, "left": 270.0, "up_left": 315.0}  # fmt: skip
CUSTOM = "custom"
DIRECTIONS = (
    ("up", "Up"), ("up_right", "Up right"), ("right", "Right"), ("down_right", "Down right"),
    ("down", "Down"), ("down_left", "Down left"), ("left", "Left"), ("up_left", "Up left"),
    (CUSTOM, "Custom"),
)  # fmt: skip


def direction_param(default: str, what: str, when=None) -> ChoiceParam:
    """A compass direction (a Param belongs to one class: each call makes one)."""
    return ChoiceParam(default, "Direction", choices=DIRECTIONS, when=when,
                       help=f"The way the {what} travels across the picture.")  # fmt: skip


def angle_param(default: float, what: str, direction: str, when=None) -> FloatParam:
    """The custom angle, active when the `direction` setting is Custom (and `when`)."""
    return FloatParam(
        default, "Angle", min=0.0, max=360.0, step=15.0, decimals=0, suffix="°",
        when=lambda c: getattr(c, direction) == CUSTOM and (when is None or when(c)),
        help=f"The way the {what} travels: 0° toward the top of the picture, clockwise "
             "(90°: to the right).",
    )  # fmt: skip


def degrees(direction: str, angle: float) -> float:
    """The direction of travel, degrees clockwise from the top."""
    return angle if direction == CUSTOM else COMPASS[direction]


def describe(direction: str, angle: float) -> str:
    """The direction for a summary: "down", "up right", "30°"."""
    return f"{angle:g}°" if direction == CUSTOM else dict(DIRECTIONS)[direction].lower()


def heading(angle: float) -> np.ndarray:
    """(2,) unit vector travelling `angle` degrees clockwise from the top (y runs down)."""
    a = math.radians(angle)
    return np.array([math.sin(a), -math.cos(a)])

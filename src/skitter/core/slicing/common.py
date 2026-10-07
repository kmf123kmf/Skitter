"""Settings several slicing operations share, so they read and behave alike.

Each function returns a new parameter to declare on an operation class (a
Param belongs to one class), with the shared label, range and wording:

    class MySlicer(Subdivider):
        angle = rotation_param("pattern", "45° gives diagonal rows")
        anchor = anchor_param("pattern")
"""

from collections.abc import Callable

from skitter.core.slicing.frame import ANCHORS, CENTER
from skitter.core.slicing.params import ChoiceParam, FloatParam, IntParam

HORIZONTAL, VERTICAL = "horizontal", "vertical"
ORIENTATIONS = ((HORIZONTAL, "Horizontal"), (VERTICAL, "Vertical"))


def anchor_param(what: str) -> ChoiceParam:
    """Where a pinned pattern (frame.py) is pinned and turns about."""
    return ChoiceParam(
        CENTER, "Anchor", choices=ANCHORS,
        help=f"Which point of the region the {what} is pinned to and turns about; "
             "overhang goes to the opposite edges.",
    )  # fmt: skip


def rotation_param(what: str, example: str) -> FloatParam:
    """The turn of a whole pinned pattern about its anchor, in degrees."""
    return FloatParam(
        0.0, "Rotation", min=-90.0, max=90.0, step=5.0, decimals=1, suffix="°",
        help=f"Rotation of the whole {what} about its anchor; {example}. Tiles at the "
             "edges stay whole and overhang.",
    )  # fmt: skip


def orientation_param(label: str, help: str) -> ChoiceParam:
    return ChoiceParam(HORIZONTAL, label, choices=ORIENTATIONS, help=help)


def seed_param(when: Callable | None = None) -> IntParam:
    """The seed of an operation's randomness (same seed, same result)."""
    return IntParam(1, "Seed", min=0, max=999_999, when=when)

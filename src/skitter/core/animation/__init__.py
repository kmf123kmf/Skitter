"""Animation of the mosaic: built, shown, cleared away (see base.py, phases.py).

Nothing here imports Qt: a timeline only computes tile states, which the
UI draws as sprites (or a renderer could draw frame by frame).
"""

from skitter.core.animation import assemble, deal, show  # registers the built-ins
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
    unregister_choreography,
)
from skitter.core.animation.landing import landing_order, landing_times, random_landing_order
from skitter.core.animation.phases import (
    PHASE_NAMES,
    PHASES,
    PhasedTimeline,
    PhaseHolds,
    PhaseSpan,
    VideoClock,
    plan_phases,
)

__all__ = [
    "PHASES",
    "PHASE_NAMES",
    "PhaseHolds",
    "PhaseSpan",
    "VideoClock",
    "PhasedTimeline",
    "plan_phases",
    "show",
    "unregister_choreography",
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

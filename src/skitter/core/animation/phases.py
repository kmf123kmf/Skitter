"""Phases: the animation as build, show and clear, played one after another.

An animation has up to three phases, each made by a choreography of that
phase (`Choreography.phase`):

- **build** puts the mosaic together: it starts wherever it likes (usually an
  empty table) and ends exactly at the finished mosaic (`TileFrame.final`).
  Every animation has one.
- **show** happens on the finished mosaic: it starts *and* ends exactly at
  the finished mosaic (its effects are offsets from it that vanish at both
  ends), so it fits between any build and any clear.
- **clear** takes the mosaic away: it starts exactly at the finished mosaic
  and ends wherever it likes; ending with nothing left on the table (every
  tile gone or out of the frame) lets the video loop seamlessly into a build
  that starts from an empty table.

Because every boundary is the finished mosaic at rest, any choreographies of
the three phases combine with no special cases. Show and clear are optional
(None).

Holds: each phase can hold still before and after its motion
(`PhaseHolds`: its first frame, then its last; for the build, the empty
table and the finished mosaic). They belong to the phase: a phase's span is
its hold before, its motion, its hold after. A phase left out has no holds.

`PhasedTimeline` plays the chosen phases in order, holds included, and keeps
where each lies (`spans`; a phase left out is an empty span where it would
be). That is the whole video: `VideoClock` (the spans and the length) is
what camera keys anchor to (keyframes.py) and what the export plans with.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skitter.core.animation.base import Choreography, TileFrame, Timeline
from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import Configurable, FloatParam

if TYPE_CHECKING:
    from skitter.core.animation.look import AnimationLook

PHASES = ("build", "show", "clear")  # in playing order
PHASE_NAMES = {"build": "Build", "show": "Show", "clear": "Clear"}
OPTIONAL = ("show", "clear")  # phases that can be left out (None)


class PhaseHolds(Configurable):
    """How long a phase holds still before and after its motion."""

    hold_before = FloatParam(
        0.0, "Hold before", min=0.0, max=60.0, step=0.5, decimals=1, suffix=" s",
        help="Show the phase's first frame this long before anything moves (for Build: "
             "the empty table).",
    )  # fmt: skip
    hold_after = FloatParam(
        0.0, "Hold after", min=0.0, max=60.0, step=0.5, decimals=1, suffix=" s",
        help="Show the phase's last frame this long after its motion ends (for Build: the "
             "finished mosaic).",
    )  # fmt: skip


def default_holds() -> dict[str, PhaseHolds]:
    """New projects: the finished mosaic holds 2 s after the build."""
    holds = {phase: PhaseHolds() for phase in PHASES}
    holds["build"].update(hold_after=2.0)
    return holds


@dataclass(frozen=True)
class PhaseSpan:
    """Where a phase lies in the video: its hold before, its motion, its hold after
    (start == end: the phase is left out)."""

    phase: str
    start: float
    end: float
    before: float = 0.0  # seconds held at the start
    after: float = 0.0  # seconds held at the end

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def motion(self) -> tuple[float, float]:
        """(start, end) of the motion, between the holds."""
        return self.start + self.before, self.end - self.after


@dataclass(frozen=True)
class VideoClock:
    """The video's time: its length and where each phase lies (all three, in order;
    empty: all build). Camera keys anchor to it."""

    duration: float
    spans: tuple[PhaseSpan, ...] = ()

    @property
    def phase_spans(self) -> tuple[PhaseSpan, ...]:
        if self.spans:
            return self.spans
        d = float(self.duration)
        return (PhaseSpan("build", 0.0, d), PhaseSpan("show", d, d), PhaseSpan("clear", d, d))

    def phase_span(self, phase: str) -> tuple[float, float]:
        """(start, end) of a phase, holds included (start == end: it's left out)."""
        span = self.phase_spans[PHASES.index(phase)]
        return span.start, span.end


class PhasedTimeline(Timeline):
    """The phases' timelines played one after another, each between its holds (see the
    module docstring).

    parts: {phase: timeline or None}; build is required. holds: {phase: (before,
    after)} seconds (default none).
    """

    def __init__(self, parts: dict[str, Timeline | None], holds=None):
        if parts.get("build") is None:
            raise ValueError("an animation needs a build phase")
        holds = holds or {}
        self.parts = {phase: parts.get(phase) for phase in PHASES}
        spans, t = [], 0.0
        for phase in PHASES:
            timeline = self.parts[phase]
            if timeline is None:
                spans.append(PhaseSpan(phase, t, t))
                continue
            before, after = (max(float(s), 0.0) for s in holds.get(phase, (0.0, 0.0)))
            end = t + before + timeline.duration + after
            spans.append(PhaseSpan(phase, t, end, before, after))
            t = end
        self.spans = tuple(spans)
        self.duration = t
        self._playing = [(span, self.parts[span.phase]) for span in self.spans
                         if self.parts[span.phase] is not None]  # fmt: skip

    @property
    def clock(self) -> VideoClock:
        return VideoClock(self.duration, self.spans)

    def span(self, phase: str) -> PhaseSpan:
        return self.spans[PHASES.index(phase)]

    def phase_at(self, t: float) -> str:
        """The phase playing at time t (at a boundary, the earlier phase: it ends there)."""
        t = min(max(float(t), 0.0), self.duration)
        for span, _ in self._playing:
            if t <= span.end:
                return span.phase
        return self._playing[-1][0].phase

    def frame(self, t: float) -> TileFrame:
        t = min(max(float(t), 0.0), self.duration)
        span, timeline = next(((s, tl) for s, tl in self._playing if t <= s.end),
                              self._playing[-1])  # fmt: skip
        # Still during the holds: the motion's first frame before it, its last after.
        return timeline.frame(min(max(t - span.motion[0], 0.0), timeline.duration))


def plan_phases(
    choreographies: dict[str, Choreography | None], scene: MosaicScene,
    look: "AnimationLook | None" = None, holds: dict[str, PhaseHolds] | None = None,
) -> PhasedTimeline:  # fmt: skip
    """The animation of the chosen choreographies ({phase: choreography or None}),
    each phase between its holds."""
    parts = {}
    for phase in PHASES:
        choreography = choreographies.get(phase)
        if choreography is not None and choreography.phase != phase:
            raise ValueError(f"{choreography.name} is a {choreography.phase} choreography, "
                             f"not a {phase} one")  # fmt: skip
        parts[phase] = None if choreography is None else choreography.timeline(scene, look)
    seconds = {phase: (h.hold_before, h.hold_after) for phase, h in (holds or {}).items()}
    return PhasedTimeline(parts, seconds)

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

Chains: a build or a clear may be several choreographies played one after
another (`plan_chain`), planned outward from the finished mosaic so that each
link starts exactly where the one before it ends. A build's last link ends at
the finished mosaic; each one before it lands on the next one's first frame,
planned on the scene laid out that way (MosaicScene.laid_out): Assemble
tossing the photos onto Unscramble's scrambled table, say. A clear's first
link starts at the finished mosaic; each one after it starts from the last
frame of the one before. So a link that another leads into (or follows)
must rest on the table there (`Choreography.rests_on_table`), and the link
on the other side must plan on any layout (`fits_any_layout`): `can_chain`.
The phase's `hold_between` pauses between links.

Holds: each phase can hold still before and after its motion
(`PhaseHolds`: its first frame, then its last; for the build, the empty
table and the finished mosaic). They belong to the phase: a phase's span is
its hold before, its motion, its hold after. A phase left out has no holds.

`PhasedTimeline` plays the chosen phases in order, holds included, and keeps
where each lies (`spans`; a phase left out is an empty span where it would
be). That is the whole video: `VideoClock` (the spans and the length) is
what camera keys anchor to (keyframes.py) and what the export plans with.
"""

from collections.abc import Sequence
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
        0.0, "Hold before", min=0.0, max=600.0, step=0.5, decimals=1, suffix=" s",
        help="Show the phase's first frame this long before anything moves (for Build: "
             "the empty table).",
    )  # fmt: skip
    hold_after = FloatParam(
        0.0, "Hold after", min=0.0, max=600.0, step=0.5, decimals=1, suffix=" s",
        help="Show the phase's last frame this long after its motion ends (for Build: the "
             "finished mosaic).",
    )  # fmt: skip
    hold_between = FloatParam(
        0.0, "Hold between", min=0.0, max=600.0, step=0.5, decimals=1, suffix=" s",
        when=lambda c: c.chained,
        help="Where one choreography of the phase hands over to the next (a lead-in "
             "landing, say), hold still this long before the next one starts.",
    )  # fmt: skip

    chained = False  # the phase is a chain (shows Hold between); not a setting


def default_holds() -> dict[str, PhaseHolds]:
    """New projects: no holds."""
    return {phase: PhaseHolds() for phase in PHASES}


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


def can_chain(first, then) -> bool:
    """Whether choreography `first` (a class or one) can play right before `then` in
    one phase's chain (see the module docstring)."""
    if first.phase != then.phase or first.id == then.id:
        return False
    if first.phase == "build":  # planned backward: the first lands on the next one's start
        return first.fits_any_layout and then.rests_on_table
    if first.phase == "clear":  # planned forward: the next starts from the first one's end
        return first.rests_on_table and then.fits_any_layout
    return False


class ChainTimeline(Timeline):
    """A phase's chain: its links' timelines played one after another, holding still
    `pause` seconds between each and the next. `starts`: when each link begins."""

    def __init__(self, links: Sequence[Timeline], pause: float = 0.0):
        if not links:
            raise ValueError("a chain needs a link")
        self.links = tuple(links)
        self.pause = max(float(pause), 0.0)
        starts, t = [], 0.0
        for link in self.links:
            starts.append(t)
            t += link.duration + self.pause
        self.starts = tuple(starts)
        self.duration = t - self.pause

    def link_at(self, t: float) -> int:
        """Which link plays at time t (in the pause after one: that one, at its end)."""
        k = len(self.starts) - 1
        while k > 0 and t < self.starts[k]:
            k -= 1
        return k

    def frame(self, t: float) -> TileFrame:
        t = min(max(float(t), 0.0), self.duration)
        k = self.link_at(t)
        link = self.links[k]
        return link.frame(min(t - self.starts[k], link.duration))


def plan_chain(chain: Sequence[Choreography], scene: MosaicScene,
               look: "AnimationLook | None" = None, pause: float = 0.0) -> Timeline:  # fmt: skip
    """The timeline of one phase's choreographies, in playing order: one alone plans as
    usual; several, a ChainTimeline, each link planned on the scene laid out where the
    link next to it (toward the finished mosaic) starts or ends."""
    chain = list(chain)
    if not chain:
        raise ValueError("a chain needs a choreography")
    for first, then in zip(chain, chain[1:], strict=False):
        if not can_chain(first, then):
            raise ValueError(f"{first.name} can't lead into {then.name}")
    if len(chain) == 1:
        return chain[0].timeline(scene, look)
    links: list = [None] * len(chain)
    on = scene
    if chain[0].phase == "build":  # from the last link back: each lands where the next starts
        for k in range(len(chain) - 1, -1, -1):
            links[k] = chain[k].timeline(on, look)
            on = scene.laid_out(links[k].frame(0.0))
    else:  # a clear, from the first: each starts where the one before ended
        for k, choreography in enumerate(chain):
            links[k] = choreography.timeline(on, look)
            on = scene.laid_out(links[k].frame(links[k].duration))
    return ChainTimeline(links, pause)


def plan_phases(
    choreographies: "dict[str, Choreography | Sequence[Choreography] | None]",
    scene: MosaicScene, look: "AnimationLook | None" = None,
    holds: dict[str, PhaseHolds] | None = None,
) -> PhasedTimeline:  # fmt: skip
    """The animation of the chosen choreographies ({phase: a choreography, a chain of
    them in playing order, or None}), each phase between its holds."""
    parts = {}
    holds = holds or {}
    for phase in PHASES:
        chosen = choreographies.get(phase)
        if chosen is None:
            chain = []
        elif isinstance(chosen, Choreography):
            chain = [chosen]
        else:
            chain = list(chosen)
        for choreography in chain:
            if choreography.phase != phase:
                raise ValueError(f"{choreography.name} is a {choreography.phase} "
                                 f"choreography, not a {phase} one")  # fmt: skip
        pause = holds[phase].hold_between if phase in holds else 0.0
        parts[phase] = plan_chain(chain, scene, look, pause) if chain else None
    seconds = {phase: (h.hold_before, h.hold_after) for phase, h in holds.items()}
    return PhasedTimeline(parts, seconds)

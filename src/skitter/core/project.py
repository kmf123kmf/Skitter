"""Project data: everything chosen across the mosaic generation steps."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from skitter.core.animation import PHASES, Choreography, PhasedTimeline, choreography_types
from skitter.core.animation.keyframes import CameraTrack
from skitter.core.animation.look import AnimationLook
from skitter.core.animation.phases import PhaseHolds, default_holds, plan_phases
from skitter.core.animation.video import VideoSettings
from skitter.core.assembly import ExportSettings
from skitter.core.edits import Edit
from skitter.core.matching.matcher import MatchResult
from skitter.core.matching.settings import MatchSettings
from skitter.core.slicing import MosaicLayout, RegionSet, SlicingPlan


@dataclass
class Project:
    source_path: Path | None = None
    source_original: np.ndarray | None = None  # (H, W, 4) uint8 RGBA, as loaded
    source_edits: list[Edit] = field(default_factory=list)
    source_image: np.ndarray | None = None  # original with edits applied
    # Read-only snapshot of source_image taken when the user finishes the Source
    # step. Every later step works from this, not from the editable image.
    source_final: np.ndarray | None = None

    layout: MosaicLayout = field(default_factory=MosaicLayout)  # base tile and columns
    slicing_plan: SlicingPlan = field(default_factory=SlicingPlan.default)
    regions: RegionSet | None = None  # slicing_plan evaluated on source_final, mosaic units

    match_settings: MatchSettings = field(default_factory=MatchSettings)
    matches: MatchResult | None = None  # a tile for each region (None after re-slicing)

    export_settings: ExportSettings = field(default_factory=ExportSettings)

    # Animate step: one settings object per choreography (of every phase), the one chosen
    # for each phase (phases.py; show and clear may be None) and each phase's holds,
    # the camera, the look.
    choreographies: dict[str, Choreography] = field(
        default_factory=lambda: {cls.id: cls() for cls in choreography_types()}
    )
    phase_choices: dict[str, str | None] = field(
        default_factory=lambda: {"build": "assemble", "show": None, "clear": None}
    )
    phase_holds: dict[str, PhaseHolds] = field(default_factory=default_holds)
    camera_track: CameraTrack = field(default_factory=CameraTrack)  # keyframes (keyframes.py)
    animation_look: AnimationLook = field(default_factory=AnimationLook)
    video_settings: VideoSettings = field(default_factory=VideoSettings)

    def phase_choreography(self, phase: str) -> Choreography | None:
        """The choreography chosen for a phase (None: the phase is left out)."""
        choice = self.phase_choices.get(phase)
        return None if choice is None else self.choreographies[choice]

    def animation(self, scene, look: AnimationLook | None = None) -> PhasedTimeline:
        """The whole animation of a scene (the whole video): the chosen phases, one after
        another, each between its holds."""
        chosen = {phase: self.phase_choreography(phase) for phase in PHASES}
        return plan_phases(chosen, scene, self.animation_look if look is None else look,
                           self.phase_holds)  # fmt: skip

    @property
    def choreography(self) -> Choreography:
        """The build's choreography."""
        return self.phase_choreography("build")

    @property
    def choreography_id(self) -> str:
        """The build's choreography id."""
        return self.phase_choices["build"]

    @choreography_id.setter
    def choreography_id(self, value: str) -> None:
        self.phase_choices["build"] = value

    @property
    def has_source(self) -> bool:
        return self.source_image is not None

"""Step pages, one per tab, in workflow order."""

from skitter.ui.steps.animate import AnimateStep
from skitter.ui.steps.base import StepPage
from skitter.ui.steps.matching import MatchingStep
from skitter.ui.steps.slicing import SlicingStep
from skitter.ui.steps.source import SourceStep
from skitter.ui.steps.tiles import TilesStep

# Tab order. Each step unlocks once every step before it is complete. Steps are named
# by `id` in project files, so reordering or retitling them keeps old files working.
STEPS: list[type[StepPage]] = [SourceStep, SlicingStep, TilesStep, MatchingStep, AnimateStep]

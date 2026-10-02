"""Step pages, one per tab, in workflow order."""

from skitter.ui.steps.base import StepPage
from skitter.ui.steps.matching import MatchingStep
from skitter.ui.steps.slicing import SlicingStep
from skitter.ui.steps.source import SourceStep
from skitter.ui.steps.tiles import TilesStep

# Tab order. Each step unlocks once every step before it is complete.
STEPS: list[type[StepPage]] = [SourceStep, SlicingStep, TilesStep, MatchingStep]

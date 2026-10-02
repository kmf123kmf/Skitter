"""Step pages, one per tab, in workflow order."""

from skitter.ui.steps.base import StepPage
from skitter.ui.steps.source import SourceStep

# Tab order. Each step unlocks once every step before it is complete.
STEPS: list[type[StepPage]] = [SourceStep]

"""Slicing plans: an ordered list of operations applied to the final image."""

from collections.abc import Callable
from dataclasses import dataclass

from skitter.core.slicing.base import (
    MAX_REGIONS,
    Progress,
    SliceContext,
    SlicingOperation,
    check_region_count,
    operation_from_dict,
)
from skitter.core.slicing.mask import mask_regions
from skitter.core.slicing.operations import GridSlicer
from skitter.core.slicing.regions import RegionSet

__all__ = ["MAX_REGIONS", "SlicingPlan", "Stage", "StageResult"]


@dataclass
class Stage:
    operation: SlicingOperation
    enabled: bool = True

    def key(self) -> tuple:
        return (self.enabled, self.operation.key())


@dataclass(frozen=True)
class StageResult:
    key: tuple
    regions: RegionSet  # regions after this stage


class SlicingPlan:
    """Stages run in order, starting from one region covering the whole canvas.

    Disabled stages pass their input through unchanged.
    """

    def __init__(self, stages=()):
        self.stages: list[Stage] = list(stages)

    @classmethod
    def default(cls) -> "SlicingPlan":
        return cls([Stage(GridSlicer())])

    def copy(self) -> "SlicingPlan":
        """An independent copy (to evaluate in the background while this one is edited)."""
        return SlicingPlan(Stage(s.operation.copy(), s.enabled) for s in self.stages)

    def evaluate(
        self,
        ctx: SliceContext,
        cache: list[StageResult] | None = None,
        progress: Callable[[str, float], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> list[StageResult]:
        """Run the plan and return one result per stage.

        Leading entries of cache whose keys still match are reused, so after
        editing stage k only stages k onward are recomputed. A cache is only
        valid for the context it was computed with.

        progress(message, fraction) hears which stage runs and how far the
        whole run is; once cancelled() is true the run stops, raising
        SlicingCancelled (between stages, or within one that reports).
        """
        regions = ctx.canvas()
        results: list[StageResult] = []
        reusing = cache is not None
        n = len(self.stages)
        for i, stage in enumerate(self.stages):
            key = stage.key()
            if reusing and i < len(cache) and cache[i].key == key:
                regions = cache[i].regions
            else:
                reusing = False
                if stage.enabled:
                    name = stage.operation.name
                    message = name if n == 1 else f"{name} (stage {i + 1} of {n})"
                    report = None if progress is None else (lambda f, m=message: progress(m, f))
                    step = Progress(report, cancelled).part(i / n, (i + 1) / n)
                    step(0.0)
                    regions = stage.operation.apply(regions, ctx, step)
                    check_region_count(len(regions), stage.operation.name)
                    step(1.0)
            results.append(StageResult(key, regions))
        return results

    def regions(self, ctx: SliceContext) -> RegionSet:
        """The plan's regions that touch the visible picture (see mask.py)."""
        results = self.evaluate(ctx)
        return mask_regions(results[-1].regions if results else ctx.canvas(), ctx)

    def to_dict(self) -> dict:
        return {
            "stages": [
                {"enabled": stage.enabled, **stage.operation.to_dict()} for stage in self.stages
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SlicingPlan":
        return cls(
            Stage(operation_from_dict(item), item.get("enabled", True))
            for item in data.get("stages", [])
        )

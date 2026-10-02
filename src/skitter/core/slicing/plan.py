"""Slicing plans: an ordered list of operations applied to the final image."""

from dataclasses import dataclass

from skitter.core.slicing.base import (
    SliceContext,
    SlicingError,
    SlicingOperation,
    operation_from_dict,
)
from skitter.core.slicing.operations import GridSlicer
from skitter.core.slicing.regions import RegionSet

MAX_REGIONS = 250_000


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

    def evaluate(
        self, ctx: SliceContext, cache: list[StageResult] | None = None
    ) -> list[StageResult]:
        """Run the plan and return one result per stage.

        Leading entries of cache whose keys still match are reused, so after
        editing stage k only stages k onward are recomputed. A cache is only
        valid for the context it was computed with.
        """
        regions = ctx.canvas()
        results: list[StageResult] = []
        reusing = cache is not None
        for i, stage in enumerate(self.stages):
            key = stage.key()
            if reusing and i < len(cache) and cache[i].key == key:
                regions = cache[i].regions
            else:
                reusing = False
                if stage.enabled:
                    regions = stage.operation.apply(regions, ctx)
                    if len(regions) > MAX_REGIONS:
                        raise SlicingError(
                            f"{stage.operation.name} produced {len(regions):,} regions; "
                            f"the limit is {MAX_REGIONS:,}"
                        )
            results.append(StageResult(key, regions))
        return results

    def regions(self, ctx: SliceContext) -> RegionSet:
        results = self.evaluate(ctx)
        return results[-1].regions if results else ctx.canvas()

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

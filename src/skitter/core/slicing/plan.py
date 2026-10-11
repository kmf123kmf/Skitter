"""Slicing plans: a slicer, then (optionally) its cells split by detail.

The slicer (any registered operation: Grid, Bond, Pattern, Contour Rows,
Photo Pile) cuts the whole mosaic; "Split by detail" (a quadtree) then
splits each of its cells into quarters wherever the image has detail. A
plan keeps settings for every slicer, so switching to another and back
finds them as they were, and keeps the detail split's settings while it is
off.

Files from before plans took this shape listed stages (`{"stages": [...]}`);
`from_dict` reads them: the first slicer is kept, a quadtree after it
becomes the detail split, and anything else is left out (and reported).
"""

from collections.abc import Callable
from dataclasses import dataclass

from skitter.core.slicing.base import (
    MAX_REGIONS,
    Progress,
    SliceContext,
    SlicingOperation,
    check_region_count,
    get_operation_type,
    operation_types,
)
from skitter.core.slicing.mask import mask_regions
from skitter.core.slicing.operations import GridSlicer, QuadtreeSlicer
from skitter.core.slicing.regions import RegionSet

__all__ = ["MAX_REGIONS", "SlicingPlan", "StageResult"]

DEFAULT_SLICER = GridSlicer.id


@dataclass(frozen=True)
class StageResult:
    key: tuple
    regions: RegionSet  # regions after this stage


class SlicingPlan:
    """The chosen slicer (and every slicer's settings) and the detail split."""

    def __init__(
        self,
        slicer: SlicingOperation | None = None,
        detail: QuadtreeSlicer | None = None,
        split_by_detail: bool = False,
    ):
        self.slicers: dict[str, SlicingOperation] = {cls.id: cls() for cls in operation_types()}
        slicer = GridSlicer() if slicer is None else slicer
        self.slicers[slicer.id] = slicer
        self.slicer_id = slicer.id
        self.detail = QuadtreeSlicer() if detail is None else detail
        self.split_by_detail = split_by_detail

    @classmethod
    def default(cls) -> "SlicingPlan":
        return cls()

    @property
    def slicer(self) -> SlicingOperation:
        return self.slicers[self.slicer_id]

    def operations(self) -> list[SlicingOperation]:
        """What runs, in order."""
        return [self.slicer, self.detail] if self.split_by_detail else [self.slicer]

    def copy(self) -> "SlicingPlan":
        """An independent copy (to evaluate in the background while this one is edited)."""
        plan = SlicingPlan(self.slicer.copy(), self.detail.copy(), self.split_by_detail)
        plan.slicers.update({key: op.copy() for key, op in self.slicers.items()
                             if key != self.slicer_id})  # fmt: skip
        return plan

    def evaluate(
        self,
        ctx: SliceContext,
        cache: list[StageResult] | None = None,
        progress: Callable[[str, float], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> list[StageResult]:
        """Run the plan and return one result per operation run.

        Leading entries of cache whose keys still match are reused, so changing
        only the detail split doesn't run the slicer again. A cache is only
        valid for the context it was computed with.

        progress(message, fraction) hears which operation runs and how far the
        whole run is; once cancelled() is true the run stops, raising
        SlicingCancelled (between operations, or within one that reports).
        """
        regions = ctx.canvas()
        results: list[StageResult] = []
        reusing = cache is not None
        operations = self.operations()
        n = len(operations)
        for i, operation in enumerate(operations):
            key = operation.key()
            if reusing and i < len(cache) and cache[i].key == key:
                regions = cache[i].regions
            else:
                reusing = False
                name = operation.name
                report = None if progress is None else (lambda f, m=name: progress(m, f))
                step = Progress(report, cancelled).part(i / n, (i + 1) / n)
                step(0.0)
                regions = operation.apply(regions, ctx, step)
                check_region_count(len(regions), name)
                step(1.0)
            results.append(StageResult(key, regions))
        return results

    def regions(self, ctx: SliceContext) -> RegionSet:
        """The plan's regions that touch the visible picture (see mask.py)."""
        return mask_regions(self.evaluate(ctx)[-1].regions, ctx)

    def to_dict(self) -> dict:
        return {
            "slicer": self.slicer_id,
            "slicers": {key: op.values() for key, op in self.slicers.items()},
            "split_by_detail": self.split_by_detail,
            "detail": self.detail.values(),
        }

    @classmethod
    def from_dict(cls, data: dict, problems: list[str] | None = None) -> "SlicingPlan":
        """The plan to_dict described (or an older file's stages; see above). With a
        problems list, slicers this version doesn't have and settings it doesn't know
        are left out or keep their defaults, each reported there instead of raising."""
        if "stages" in data:
            return cls._from_stages(data["stages"], problems)
        plan = cls()
        for key, values in (data.get("slicers") or {}).items():
            try:
                op_type = get_operation_type(key)
            except KeyError:
                if problems is None:
                    raise
                problems.append(f"Slicer left out: {key!r} isn't available")
                continue
            plan.slicers[key] = op_type.from_values(values, problems, op_type.name)
        chosen = data.get("slicer", DEFAULT_SLICER)
        if chosen in plan.slicers:
            plan.slicer_id = chosen
        elif problems is None:
            raise KeyError(f"unknown slicer {chosen!r}")
        else:
            problems.append(f"Slicer: {chosen!r} isn't available; Grid is used")
        plan.detail = QuadtreeSlicer.from_values(data.get("detail", {}), problems,
                                                 "Split by detail")  # fmt: skip
        plan.split_by_detail = bool(data.get("split_by_detail", False))
        return plan

    @classmethod
    def _from_stages(cls, stages: list, problems: list[str] | None) -> "SlicingPlan":
        """An older file's list of stages: its first slicer, then a quadtree as the detail
        split; disabled stages count only for their settings."""
        plan = cls()
        slicer = None
        for item in stages:
            kind, enabled = item.get("type"), item.get("enabled", True)
            if kind == QuadtreeSlicer.id:
                if enabled and slicer is None and problems is not None:
                    problems.append("Quadtree split the whole mosaic; it now splits a Grid's "
                                    "cells by detail")  # fmt: skip
                plan.detail = QuadtreeSlicer.from_values(item.get("params", {}), problems,
                                                         "Split by detail")  # fmt: skip
                plan.split_by_detail = plan.split_by_detail or enabled
                continue
            try:
                op_type = get_operation_type(kind)
            except KeyError:
                if problems is None:
                    raise
                if enabled:
                    problems.append(f"Slicing stage left out: {kind!r} is no longer available")
                continue
            operation = op_type.from_values(item.get("params", {}), problems, op_type.name)
            plan.slicers[op_type.id] = operation
            if enabled and slicer is None:
                slicer = operation
            elif enabled and problems is not None:
                problems.append(f"Slicing stage left out: {op_type.name} after "
                                f"{slicer.name} (a plan has one slicer now)")  # fmt: skip
        if slicer is not None:
            plan.slicer_id = slicer.id
        return plan

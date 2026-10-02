"""Slicing: divide the final source image into regions for tile matching.

The result of slicing is always a RegionSet: rectangles with a center, size
and rotation. See base.py for how to write a new slicing operation.
"""

from skitter.core.slicing import operations  # registers the built-in operations
from skitter.core.slicing.base import (
    CATEGORIES,
    SliceContext,
    SlicingError,
    SlicingOperation,
    Subdivider,
    get_operation_type,
    operation_from_dict,
    operation_types,
    register_operation,
)
from skitter.core.slicing.params import BoolParam, ChoiceParam, FloatParam, IntParam, Param
from skitter.core.slicing.plan import MAX_REGIONS, SlicingPlan, Stage, StageResult
from skitter.core.slicing.regions import REGION_DTYPE, Region, RegionSet

__all__ = [
    "CATEGORIES",
    "MAX_REGIONS",
    "REGION_DTYPE",
    "BoolParam",
    "ChoiceParam",
    "FloatParam",
    "IntParam",
    "Param",
    "Region",
    "RegionSet",
    "SliceContext",
    "SlicingError",
    "SlicingOperation",
    "SlicingPlan",
    "Stage",
    "StageResult",
    "Subdivider",
    "get_operation_type",
    "operation_from_dict",
    "operation_types",
    "operations",
    "register_operation",
]

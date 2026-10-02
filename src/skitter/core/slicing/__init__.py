"""Slicing: divide the final source image into regions for tile matching.

The result of slicing is always a RegionSet: rectangles with a center, size,
rotation and stacking order, in mosaic units (see layout.py). See base.py
for how to write a new slicing operation.
"""

from skitter.core.slicing import operations  # registers the built-in operations
from skitter.core.slicing.analysis import SliceSummary, coverage, summarize
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
from skitter.core.slicing.layout import TILE_ASPECTS, TILE_UNIT, MosaicLayout
from skitter.core.slicing.params import (
    BoolParam,
    ChoiceParam,
    Configurable,
    FloatParam,
    IntParam,
    Param,
    TileSizeParam,
)
from skitter.core.slicing.patterns import (
    BrickPattern,
    PatternBuilder,
    PatternUnit,
    get_pattern,
    patterns,
    register_pattern,
    tile_pattern,
)
from skitter.core.slicing.plan import MAX_REGIONS, SlicingPlan, Stage, StageResult
from skitter.core.slicing.regions import REGION_DTYPE, Region, RegionSet

__all__ = [
    "CATEGORIES",
    "MAX_REGIONS",
    "REGION_DTYPE",
    "TILE_ASPECTS",
    "TILE_UNIT",
    "BoolParam",
    "BrickPattern",
    "ChoiceParam",
    "Configurable",
    "FloatParam",
    "IntParam",
    "MosaicLayout",
    "Param",
    "PatternBuilder",
    "PatternUnit",
    "Region",
    "RegionSet",
    "SliceContext",
    "SliceSummary",
    "SlicingError",
    "SlicingOperation",
    "SlicingPlan",
    "Stage",
    "StageResult",
    "Subdivider",
    "TileSizeParam",
    "coverage",
    "get_operation_type",
    "get_pattern",
    "operation_from_dict",
    "operation_types",
    "operations",
    "patterns",
    "register_operation",
    "register_pattern",
    "summarize",
    "tile_pattern",
]

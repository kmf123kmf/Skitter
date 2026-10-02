"""Built-in slicing operations. Importing this package registers them."""

from skitter.core.slicing.operations.adjust import JitterAdjust, StackingAdjust
from skitter.core.slicing.operations.bond import BondSlicer
from skitter.core.slicing.operations.grid import GridSlicer
from skitter.core.slicing.operations.pattern import PatternSlicer
from skitter.core.slicing.operations.pile import PileSlicer
from skitter.core.slicing.operations.quadtree import QuadtreeSlicer

__all__ = [
    "BondSlicer",
    "GridSlicer",
    "JitterAdjust",
    "PatternSlicer",
    "PileSlicer",
    "QuadtreeSlicer",
    "StackingAdjust",
]

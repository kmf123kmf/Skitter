"""Built-in slicing operations. Importing this package registers them."""

from skitter.core.slicing.operations.adjust import JitterAdjust, StackingAdjust
from skitter.core.slicing.operations.bond import BondSlicer
from skitter.core.slicing.operations.contour import ContourSlicer
from skitter.core.slicing.operations.grid import GridSlicer
from skitter.core.slicing.operations.pattern import PatternSlicer
from skitter.core.slicing.operations.pile import PileSlicer
from skitter.core.slicing.operations.quadtree import QuadtreeSlicer
from skitter.core.slicing.operations.split import SplitSlicer

__all__ = [
    "BondSlicer",
    "ContourSlicer",
    "GridSlicer",
    "JitterAdjust",
    "PatternSlicer",
    "PileSlicer",
    "QuadtreeSlicer",
    "SplitSlicer",
    "StackingAdjust",
]

"""Built-in slicing operations. Importing this package registers them."""

from skitter.core.slicing.operations.adjust import GapAdjust, JitterAdjust, StackingAdjust
from skitter.core.slicing.operations.grid import GridSlicer
from skitter.core.slicing.operations.pile import PileSlicer
from skitter.core.slicing.operations.quadtree import QuadtreeSlicer

__all__ = [
    "GapAdjust",
    "GridSlicer",
    "JitterAdjust",
    "PileSlicer",
    "QuadtreeSlicer",
    "StackingAdjust",
]

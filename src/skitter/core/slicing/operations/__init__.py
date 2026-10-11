"""Built-in slicing operations. Importing this package registers the slicers."""

from skitter.core.slicing.operations.bond import BondSlicer
from skitter.core.slicing.operations.contour import ContourSlicer
from skitter.core.slicing.operations.grid import GridSlicer
from skitter.core.slicing.operations.pattern import PatternSlicer
from skitter.core.slicing.operations.pile import PileSlicer
from skitter.core.slicing.operations.quadtree import QuadtreeSlicer

__all__ = [
    "BondSlicer",
    "ContourSlicer",
    "GridSlicer",
    "PatternSlicer",
    "PileSlicer",
    "QuadtreeSlicer",
]

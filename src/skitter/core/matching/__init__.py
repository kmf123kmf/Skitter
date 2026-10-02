"""Matching: choose a tile crop for every region of the sliced mosaic.

The pipeline (see matcher.py):

1. `targets`: describe each region of the final image, noting which parts
   of it stay visible under overlapping regions.
2. `index`: per region shape, build crop candidates of every tile, describe
   them, and index them for fast approximate nearest-neighbor search.
3. Search every region's top candidates, then rerank them exactly with the
   current tint and weights.
4. `assign`: choose among those candidates under the reuse rules (how often a
   tile may repeat and how far apart repeats must be), then refine.
5. `quality`: render a coarse proxy of the mosaic and score it against the
   image as seen from a distance; spend more search effort where it is worst.

Nothing here imports Qt; long steps take progress and cancel callbacks.
"""

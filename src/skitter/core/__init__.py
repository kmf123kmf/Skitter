"""Image processing and mosaic algorithms.

Everything in this package works on numpy arrays and must not import Qt,
so it can be tested headless and reused outside the GUI.
"""

# Registers extra image decoders (HEIC) wherever skitter.core is used,
# including the worker processes that read tile images.
from skitter.core import imaging as _imaging  # noqa: F401

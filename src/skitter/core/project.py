"""Project data: everything chosen across the mosaic generation steps."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from skitter.core.edits import Edit


@dataclass
class Project:
    source_path: Path | None = None
    source_original: np.ndarray | None = None  # (H, W, 3) uint8 RGB, as loaded
    source_edits: list[Edit] = field(default_factory=list)
    source_image: np.ndarray | None = None  # original with edits applied

    @property
    def has_source(self) -> bool:
        return self.source_image is not None

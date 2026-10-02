"""Project data: everything chosen across the mosaic generation steps."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Project:
    source_path: Path | None = None
    source_image: np.ndarray | None = None  # (H, W, 3) uint8 RGB

    @property
    def has_source(self) -> bool:
        return self.source_image is not None

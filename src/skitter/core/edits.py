"""Non-destructive image edits.

Each edit is a small immutable value applied to an (H, W, C) array. A project
keeps the original image plus its list of edits, so edits can be undone,
redone, and saved alongside the project.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlipHorizontal:
    def apply(self, image: np.ndarray) -> np.ndarray:
        return image[:, ::-1]

    def describe(self) -> str:
        return "Flip horizontal"


@dataclass(frozen=True)
class FlipVertical:
    def apply(self, image: np.ndarray) -> np.ndarray:
        return image[::-1]

    def describe(self) -> str:
        return "Flip vertical"


@dataclass(frozen=True)
class Rotate90:
    turns: int = 1  # clockwise quarter turns; negative turns counterclockwise

    def apply(self, image: np.ndarray) -> np.ndarray:
        return np.rot90(image, k=-self.turns)

    def describe(self) -> str:
        quarter = self.turns % 4
        return {1: "Rotate right", 3: "Rotate left"}.get(quarter, f"Rotate {90 * quarter}°")


@dataclass(frozen=True)
class Crop:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self):
        if self.left < 0 or self.top < 0 or self.width < 1 or self.height < 1:
            raise ValueError(f"invalid crop {self}")

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if self.left + self.width > w or self.top + self.height > h:
            raise ValueError(f"{self} exceeds image size {w}x{h}")
        return image[self.top : self.top + self.height, self.left : self.left + self.width]

    def describe(self) -> str:
        return f"Crop to {self.width} × {self.height}"


Edit = FlipHorizontal | FlipVertical | Rotate90 | Crop


def apply_edits(image: np.ndarray, edits: list[Edit]) -> np.ndarray:
    for edit in edits:
        image = edit.apply(image)
    return np.ascontiguousarray(image)

"""Picking tiles by hand on the Matching tab: the picker panel and canvas overlays.

`TilePicker` is the side-panel group: the edit-mode switch, undo/redo,
the selected region's target and status, and a grid of its candidates.
`EditOverlays` draws on the mosaic: the region under the cursor, the
selected region (the rest dimmed), a live preview of a candidate in place,
and markers on tiles picked by hand.

Candidate crops show from library thumbnails at once; `CropReader` then
reads the real crops from their files in the background.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont, QFontMetrics, QImage, QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
)

from skitter.core.color import oklab_to_srgb, rgb8_to_oklab
from skitter.core.scene import MosaicScene
from skitter.core.tiles.render import render_crops_or_thumbs
from skitter.ui import icons
from skitter.ui.jobs import Job, JobCancelled
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.render.tile_textures import thumb_crop
from skitter.ui.widgets.candidate_grid import CandidateGrid

PICKER_PX = 192  # longest side of candidate crops read for the picker (and the preview)
STAND_IN_PX = 48  # longest side of crops cut from thumbnails until the files are read
TARGET_PX = 72  # longest side of the target patch
DIM_ALPHA = 0.45  # how far the rest of the mosaic darkens around a selection
SELECT_COLOR = (0.23, 0.61, 1.0)
HOVER_COLOR = (1.0, 1.0, 1.0)
MARKER_COLOR = (1.0, 0.77, 0.19)
MARKER_SHARE = 0.16  # marker size, share of the tile's shorter side


# Images


def crop_size(aspect: float, longest: int = PICKER_PX) -> tuple[int, int]:
    """(w, h) pixels of a crop of this aspect whose longest side is `longest`."""
    if aspect >= 1:
        return longest, max(1, round(longest / aspect))
    return max(1, round(longest * aspect)), longest


def tinted(rgb: np.ndarray, shift, mirrored: bool) -> np.ndarray:
    """A crop as the mosaic shows it: OKLab-shifted by matching's tint, maybe mirrored."""
    shift = np.asarray(shift, np.float32)
    out = rgb
    if np.any(np.abs(shift) > 1e-6):
        out = np.round(oklab_to_srgb(rgb8_to_oklab(rgb) + shift) * 255).astype(np.uint8)
    return np.ascontiguousarray(out[:, ::-1] if mirrored else out)


def to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def region_patch(image: np.ndarray, scale: float, center, size, rotation: float,
                 longest: int = TARGET_PX) -> np.ndarray:  # fmt: skip
    """(h, w, 3) the part of the source image a region covers, upright in its own frame.

    center, size: mosaic units; scale: mosaic units per source pixel.
    """
    w, h = crop_size(size[0] / size[1], longest)
    c, s = math.cos(rotation), math.sin(rotation)
    # Output pixel (u, v) -> mosaic point -> source pixel (affine, PIL's inverse mapping).
    ax, ay = size[0] / w, size[1] / h
    x0 = center[0] - c * size[0] / 2 + s * size[1] / 2
    y0 = center[1] - s * size[0] / 2 - c * size[1] / 2
    coeffs = (c * ax / scale, -s * ay / scale, x0 / scale, s * ax / scale, c * ay / scale,
              y0 / scale)  # fmt: skip
    src = Image.fromarray(np.asarray(image))
    out = src.transform((w, h), Image.Transform.AFFINE, coeffs, Image.Resampling.BICUBIC)
    return np.asarray(out, dtype=np.uint8)


class CropReader(QObject):
    """Reads candidate crops from their files in the background, one at a time, and
    tints them as the mosaic would."""

    ready = Signal(object, int, object, object)  # (token, index, crop, tinted crop)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job: Job | None = None

    def read(self, token, crops: list[tuple]) -> None:
        """crops: (index, path, rect, size, thumb, thumb_size, tint shift, mirrored)."""
        self.cancel()

        def work(progress, cancelled):
            for index, path, rect, size, thumb, thumb_size, shift, mirrored in crops:
                if cancelled():
                    raise JobCancelled
                done = render_crops_or_thumbs([path], [rect], [size], thumb[None],
                                              thumb_size[None], workers=0)  # fmt: skip
                if done is not None:
                    crop = done[0][0]
                    self.ready.emit(token, index, crop, tinted(crop, shift, mirrored))

        self._job = Job(work, parent=self).start()

    def cancel(self) -> None:
        if self._job is not None:
            self._job.cancel()
            self._job = None

    def wait(self, timeout: float | None = None) -> None:
        if self._job is not None:
            self._job.wait(timeout)


def thumb_candidate(library, slot: int, rect, aspect: float) -> np.ndarray:
    """A small stand-in crop cut from the library thumbnail."""
    size = crop_size(aspect, STAND_IN_PX)
    return thumb_crop(library.thumbs, library.thumb_size, int(slot), rect, size)


def file_name(path: str) -> str:
    return Path(path.replace("\\", "/")).name


def elide(label: QLabel, text: str, width: int | None = None, bold: bool = False) -> str:
    """Text shortened in the middle to fit a label's width (keeps a file's extension)."""
    font = QFont(label.font())
    font.setBold(bold)
    width = max(width if width is not None else label.contentsRect().width(), 40)
    return QFontMetrics(font).elidedText(text, Qt.TextElideMode.ElideMiddle, width)


def fixed_width(label: QLabel, lines: int = 0) -> QLabel:
    """A label whose text never widens the panel (and, given lines, never changes its height)."""
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setMinimumWidth(1)
    if lines:
        label.setFixedHeight(lines * QFontMetrics(label.font()).lineSpacing() + 2)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    return label


# Canvas overlays


def upper_closure(scene: MosaicScene, tile: int) -> np.ndarray:
    """Tiles drawn over `tile` (directly or over ones that are), with it, bottom to top."""
    pairs = scene.overlaps
    if not len(pairs):
        return np.array([tile], np.int64)
    order = np.argsort(pairs[:, 0], kind="stable")
    lower, upper = pairs[order, 0], pairs[order, 1]
    seen = np.zeros(len(scene), bool)
    seen[tile] = True
    frontier = np.array([tile], np.int64)
    while len(frontier):
        lo = np.searchsorted(lower, frontier)
        hi = np.searchsorted(lower, frontier, side="right")
        above = np.concatenate([upper[a:b] for a, b in zip(lo, hi, strict=True)])
        above = np.unique(above[~seen[above]]) if len(above) else above
        seen[above] = True
        frontier = above
    return np.flatnonzero(seen)


def outline(center, size, rotation, color) -> np.ndarray:
    inst = make_instances(1)
    inst["pos"], inst["size"], inst["rotation"] = center, size, rotation
    inst["tint"][:, :3], inst["tint"][:, 3] = color, 1.0
    return inst


def marker_instances(scene: MosaicScene, tiles: np.ndarray) -> np.ndarray:
    """Small diamonds in the top-left corner of the given scene tiles."""
    inst = make_instances(len(tiles))
    size = scene.size[tiles]
    side = MARKER_SHARE * size.min(axis=1)
    rot = scene.rotation[tiles]
    local = -size / 2 + side[:, None] * 0.9  # inside the top-left corner
    c, s = np.cos(rot), np.sin(rot)
    x, y = local[:, 0], local[:, 1]
    offset = np.stack([c * x - s * y, s * x + c * y], axis=1)
    inst["pos"] = scene.center[tiles] + offset
    inst["size"] = np.stack([side, side], axis=1) / math.sqrt(2)
    inst["rotation"] = rot + math.pi / 4
    inst["tint"][:, :3], inst["tint"][:, 3] = MARKER_COLOR, 1.0
    return inst


class EditOverlays:
    """Layers drawn over the mosaic in edit mode (all empty when not editing)."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.dim = SpriteLayer(None, make_instances(0))
        self.preview = SpriteLayer(None, make_instances(0))
        self.cover = SpriteLayer(None, make_instances(0))  # tiles from the mosaic's textures
        self.markers = SpriteLayer(None, make_instances(0), outline_px=1.0, edge_px=0.0,
                                   fill_alpha=1.0)  # fmt: skip
        self.hover = SpriteLayer(None, make_instances(0), outline_px=2.0, edge_px=1.0,
                                 line_alpha=0.9)  # fmt: skip
        self.select = SpriteLayer(None, make_instances(0), outline_px=3.0, edge_px=1.5)
        for layer in self.layers:
            canvas.add_layer(layer)

    @property
    def layers(self) -> tuple[SpriteLayer, ...]:
        return self.dim, self.preview, self.cover, self.markers, self.hover, self.select

    @property
    def first(self) -> int:
        """Index in the canvas below which the mosaic's own layers go."""
        return self.canvas.layers.index(self.dim)

    def _set(self, layer: SpriteLayer, instances: np.ndarray) -> None:
        layer.instances = instances
        layer.mark_dirty()
        self.canvas.update()

    def clear(self) -> None:
        for layer in self.layers:
            self._set(layer, make_instances(0))
        self.cover.texture_from = None

    def show_hover(self, center=None, size=None, rotation=0.0) -> None:
        if center is None:
            self._set(self.hover, make_instances(0))
        else:
            self._set(self.hover, outline(center, size, rotation, HOVER_COLOR))

    def show_selection(self, center, size, rotation, bounds) -> None:
        self._set(self.select, outline(center, size, rotation, SELECT_COLOR))
        if bounds is None:
            self._set(self.dim, make_instances(0))
            return
        x0, y0, x1, y1 = bounds
        dim = make_instances(1)
        dim["pos"] = ((x0 + x1) / 2, (y0 + y1) / 2)
        dim["size"] = (x1 - x0, y1 - y0)
        dim["tint"] = (0.0, 0.0, 0.0, 1.0)
        dim["alpha"] = DIM_ALPHA
        self._set(self.dim, dim)

    def show_cover(self, tiles: np.ndarray, tile_layer: SpriteLayer | None, bright=-1) -> None:
        """Mosaic tiles drawn again over the dimming (and over a preview), dimmed as
        well except `bright`: so only the selected tile's visible part stands out."""
        self.cover.texture_from = tile_layer
        if tile_layer is None:
            self._set(self.cover, make_instances(0))
            return
        cover = tile_layer.instances[tiles].copy()
        dark = tiles != bright
        cover["tint"][dark] = (0.0, 0.0, 0.0, DIM_ALPHA)  # as dark as the dimming
        self._set(self.cover, cover)

    def show_preview(self, rgb: np.ndarray | None, instance: np.ndarray | None) -> None:
        """A candidate in place: its untinted, unmirrored crop and the instance showing it."""
        index = self.canvas.layers.index(self.preview)
        self.canvas.remove_layer(self.preview)
        if rgb is None:
            self.preview = SpriteLayer(None, make_instances(0))
        else:
            self.preview = SpriteLayer(rgb, instance)
        self.canvas.add_layer(self.preview, index)

    def show_markers(self, instances: np.ndarray) -> None:
        self._set(self.markers, instances)


# The side-panel group


class TilePicker(QGroupBox):
    """Edit mode controls and the candidate grid (see module doc)."""

    def __init__(self, parent=None):
        super().__init__("Edit Tiles", parent)
        self.edit_button = QPushButton("Edit Tiles")
        self.edit_button.setCheckable(True)
        self.edit_button.setToolTip(
            "Pick tiles by hand: click a tile in the mosaic, then choose another photo for it "
            "(E). Double-clicking a tile starts here too."
        )
        self.undo_button = QToolButton()
        self.undo_button.setIcon(icons.undo())
        self.redo_button = QToolButton()
        self.redo_button.setIcon(icons.redo())
        self.revert_all_button = QPushButton("Revert All")
        self.revert_all_button.setToolTip("Give every hand-picked tile back the matcher's choice.")
        top = QHBoxLayout()
        top.addWidget(self.edit_button, stretch=1)
        top.addWidget(self.undo_button)
        top.addWidget(self.redo_button)
        top.addWidget(self.revert_all_button)

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: palette(placeholder-text);")

        self.target = QLabel()
        self.target.setFixedSize(TARGET_PX + 4, TARGET_PX + 4)
        self.target.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target.setToolTip("The part of the image this tile covers.")
        self.info = fixed_width(QLabel(), lines=3)
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        header = QHBoxLayout()
        header.addWidget(self.target, alignment=Qt.AlignmentFlag.AlignTop)
        header.addWidget(self.info, stretch=1)

        self.grid = CandidateGrid()
        self.more_button = QPushButton("Find More")
        self.more_button.setToolTip("Search every tile for this region and list the next best.")
        self.revert_button = QPushButton("Revert Tile")
        self.revert_button.setToolTip("Give this region back the matcher's choice.")
        buttons = QHBoxLayout()
        buttons.addWidget(self.more_button)
        buttons.addWidget(self.revert_button)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.hint)
        self.selection = [header, self.grid, buttons]
        layout.addLayout(header)
        layout.addWidget(self.grid)
        layout.addLayout(buttons)

    def show_selection(self, shown: bool) -> None:
        self.target.setVisible(shown)
        self.info.setVisible(shown)
        self.grid.setVisible(shown)
        self.more_button.setVisible(shown)
        self.revert_button.setVisible(shown)

    def set_target(self, rgb: np.ndarray | None) -> None:
        self.target.setPixmap(QPixmap() if rgb is None else QPixmap.fromImage(to_qimage(rgb)))

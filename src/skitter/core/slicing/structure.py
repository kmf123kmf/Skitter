"""Image structure for slicers that follow it: guide edges and a row field.

Works on a sampled patch of the image (SliceContext.patch), in samples:

1. Edges: the luminance is blurred (`smoothness`, in samples), and edge
   pixels are local maxima of the gradient magnitude across the edge (thin,
   as in Canny) at least `strength` of a strong edge (the 99th percentile of
   the magnitude). Fragments shorter than `min_length` samples are dropped:
   they would start ripples around noise.
2. Guides: those edges on visible samples (hidden ones lie outside the
   picture: their colors, filled in from the nearest visible pixel, have
   seams that are no edges), plus the mask's edge (visible pixels next to
   hidden ones) if asked.
3. Row field: `rows` is each sample's distance to the nearest guide in row
   heights (`spacing` samples per row), so row k's centerline is the level
   set rows = k + 1/2. Its gradient points away from the guides; rows run
   across it.
   Background: past `outline_rows` rows from any guide (0: no limit), and
   everywhere if there is no guide at all, rows run straight instead, at
   `background_angle` (radians, clockwise; 0: across), as in a mosaic's
   plain background. `straight` is their row field, the same way: rows
   from the patch's top left, one row height apart.
4. Grain: the image's own direction at fine scale (the structure tensor of
   the luminance blurred by GRAIN_BLUR of a row, so stripes thinner than a
   tile count, averaged over a row): which way it runs, how clearly
   (coherence) and how strongly (gradient energy). Gradients within half a
   row of a guide are left out: the outline has its rows, and averaged in,
   its strength would make a band of "texture" beside it.
5. Texture (if asked): where the grain is strong (at least `strength` of
   the image's strong gradients, outlines included, as for edges, measured
   on `reference`: the luminance before it was antialiased, so the faint
   leftovers of detail finer than the samples, which would alias into a
   false grain, don't count) and clear, at least a row from any guide, rows rippling out from the
   outlines would cut across it: tiles there follow the grain instead
   (rows.py). Patches smaller than MIN_TEXTURE tiles
   don't count.
6. Flow: a smooth direction everywhere, for tiles laid off the outline
   rows: the grain in texture; elsewhere the rows' direction where it is
   clear and the grain where it isn't (where rows from two guides meet, or
   the middle of a curve). Both are averaged as doubled angles (a direction
   and its opposite are the same), weighted by how clear each is, over
   about a row.
"""

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np
from scipy import ndimage

STRONG_PERCENTILE = 99.0  # gradient magnitude counted as a strong edge
FLAT = 1e-6  # gradient magnitude below this (of the strong edge) is no edge at all
GRAIN_BLUR = 0.07  # of a row (about half a sample): just steadies the gradient
TEXTURE_CLARITY = 0.4  # coherence of a grain clear enough to follow
MIN_TEXTURE = 4.0  # tiles: smaller patches of texture don't count


@dataclass(frozen=True)
class StructureField:
    edges: np.ndarray  # (H, W) bool: guide edges found in the image
    guides: np.ndarray  # (H, W) bool: everything rows follow (edges, mask edge)
    rows: np.ndarray  # (H, W) float32: distance to the nearest guide, in rows
    grad: np.ndarray  # (2, H, W) float32: d rows / dx, d rows / dy (per sample)
    flow: np.ndarray  # (2, H, W) float32: unit direction things run along (x, y)
    texture: np.ndarray  # (H, W) bool: where tiles follow the grain, not the outlines
    background: np.ndarray  # (H, W) bool: where rows run straight (not texture)
    straight: np.ndarray  # (H, W) float32: the straight rows' row field
    straight_grad: tuple[float, float]  # its gradient (constant), per sample
    spacing: float  # samples per row


@runtime_checkable
class FollowsStructure(Protocol):
    """A slicing operation that lays tiles by image structure, and can show it: the
    Slicing tab draws what structure() returns (ui/widgets/structure_overlay.py)."""

    def structure(self, region, ctx) -> tuple[StructureField, np.ndarray, float]:
        """(field, visible samples, samples per mosaic unit) in the region's frame."""


def find_edges(luminance: np.ndarray, smoothness: float, strength: float,
               min_length: float) -> np.ndarray:  # fmt: skip
    """(H, W) bool thin edges of the blurred luminance (see the module docstring)."""
    blurred = ndimage.gaussian_filter(np.asarray(luminance, np.float32), max(smoothness, 0.0))
    gy = ndimage.sobel(blurred, axis=0)
    gx = ndimage.sobel(blurred, axis=1)
    magnitude = np.hypot(gx, gy)
    strong = float(np.percentile(magnitude, STRONG_PERCENTILE))
    if strong <= FLAT:
        return np.zeros(magnitude.shape, bool)
    # Non-maximum suppression across the edge, the direction in 4 bins.
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    bin_ = np.round(angle / (np.pi / 4)).astype(np.int64) % 4  # 0: -, 1: \\, 2: |, 3: /
    padded = np.pad(magnitude, 1, mode="edge")
    h, w = magnitude.shape
    offsets = ((0, 1), (1, 1), (1, 0), (1, -1))  # (dy, dx) across the edge, per bin
    peak = np.zeros((h, w), bool)
    for b, (dy, dx) in enumerate(offsets):
        ahead = padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        behind = padded[1 - dy : 1 - dy + h, 1 - dx : 1 - dx + w]
        peak |= (bin_ == b) & (magnitude >= ahead) & (magnitude > behind)
    edges = peak & (magnitude >= strength * strong) & (magnitude > FLAT * strong)
    if min_length > 1 and edges.any():
        labels, count = ndimage.label(edges, structure=np.ones((3, 3), bool))
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        sizes[0] = 0
        edges = sizes[labels] >= min_length
    return edges


@dataclass(frozen=True)
class Grain:
    cos2: np.ndarray  # (H, W) cos 2a of the direction a things run along
    sin2: np.ndarray  # (H, W) sin 2a
    clarity: np.ndarray  # (H, W) coherence: 0 no direction, 1 one clear direction
    strength: np.ndarray  # (H, W) RMS gradient magnitude around each sample
    strong: float  # a strong gradient magnitude in the image (every sample counts)


def grain(luminance: np.ndarray, blur: float, spread: float,
          weight: np.ndarray | None = None) -> Grain:  # fmt: skip
    """The direction along the image's edges: the structure tensor of the luminance
    blurred by `blur` samples, averaged over `spread` samples (weighted by `weight`,
    0..1 per sample: samples weighted 0 don't count)."""
    blurred = ndimage.gaussian_filter(np.asarray(luminance, np.float32), max(blur, 0.0))
    gy = ndimage.sobel(blurred, axis=0)
    gx = ndimage.sobel(blurred, axis=1)
    w = np.ones(blurred.shape, np.float32) if weight is None else weight.astype(np.float32)
    total = np.maximum(ndimage.gaussian_filter(w, spread), 1e-6)
    jxx = ndimage.gaussian_filter(w * gx * gx, spread) / total
    jyy = ndimage.gaussian_filter(w * gy * gy, spread) / total
    jxy = ndimage.gaussian_filter(w * gx * gy, spread) / total
    # The gradient's doubled angle; edges run across it (a quarter turn: negate).
    c, s = jxx - jyy, 2.0 * jxy
    size = np.hypot(c, s)
    energy = np.maximum(jxx + jyy, 0.0)
    norm = np.maximum(size, 1e-12)
    strong = strong_gradient(blurred, 0.0)
    return Grain(-c / norm, -s / norm, size / np.maximum(energy, 1e-12), np.sqrt(energy), strong)


def strong_gradient(luminance: np.ndarray, blur: float) -> float:
    """A strong gradient magnitude in an image (its STRONG_PERCENTILE)."""
    blurred = ndimage.gaussian_filter(np.asarray(luminance, np.float32), max(blur, 0.0))
    return float(
        np.percentile(
            np.hypot(ndimage.sobel(blurred, axis=1), ndimage.sobel(blurred, axis=0)),
            STRONG_PERCENTILE,
        )
    )


def texture_zones(grain_field: Grain, rows: np.ndarray, spacing: float, strength: float,
                  visible: np.ndarray | None) -> np.ndarray:  # fmt: skip
    """(H, W) bool: where tiles follow the grain (see the module docstring)."""
    strong = grain_field.strong
    if strong <= FLAT:
        return np.zeros(rows.shape, bool)
    zones = (
        (grain_field.strength >= strength * strong)
        & (grain_field.clarity >= TEXTURE_CLARITY)
        & (rows >= 1.0)
    )
    if visible is not None:
        zones &= visible
    if zones.any():
        labels, count = ndimage.label(zones)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        sizes[0] = 0
        zones = sizes[labels] >= MIN_TEXTURE * spacing * spacing
    return zones


def straight_rows(shape, spacing: float, angle: float) -> tuple[np.ndarray, tuple]:
    """(row field, its gradient) of straight rows at angle (radians, clockwise; 0: rows
    across), one row every `spacing` samples, counted from the patch's top left."""
    h, w = shape
    nx, ny = -np.sin(angle) / spacing, np.cos(angle) / spacing  # across the rows
    y, x = np.mgrid[:h, :w].astype(np.float32) + 0.5
    field = nx * x + ny * y
    field -= np.floor(field.min())  # rows from 0 at the top left corner
    return field.astype(np.float32), (float(nx), float(ny))


def blend_flow(grad: np.ndarray, spacing: float, grain_field: Grain, texture: np.ndarray,
               spread: float) -> np.ndarray:  # fmt: skip
    """(2, H, W) unit flow: the grain in texture; elsewhere the rows' direction where
    it's clear, the grain where it isn't (see the module docstring)."""
    gx, gy = grad
    slope = np.hypot(gx, gy) * spacing  # 1 where rows are clean, ~0 where they meet
    angle = np.arctan2(gx, -gy)  # along the rows: the gradient turned a quarter
    rows_weight = np.where(texture, 0.0, np.clip(slope, 0.0, 1.0) ** 2)
    weight = (1.0 - rows_weight) * grain_field.clarity
    c = ndimage.gaussian_filter(rows_weight * np.cos(2 * angle) + weight * grain_field.cos2, spread)
    s = ndimage.gaussian_filter(rows_weight * np.sin(2 * angle) + weight * grain_field.sin2, spread)
    half = 0.5 * np.arctan2(s, c)
    return np.stack([np.cos(half), np.sin(half)]).astype(np.float32)


def mask_edge(visible: np.ndarray) -> np.ndarray:
    """(H, W) bool: visible samples next to (8-connected) hidden ones."""
    hidden = ~np.asarray(visible, bool)
    if not hidden.any():
        return np.zeros(hidden.shape, bool)
    return ~hidden & ndimage.binary_dilation(hidden, structure=np.ones((3, 3), bool))


def structure_field(luminance: np.ndarray, spacing: float, *, smoothness: float,
                    strength: float, min_length: float, visible: np.ndarray | None = None,
                    follow_mask: bool = True, follow_texture: bool = True,
                    reference: np.ndarray | None = None, outline_rows: int = 0,
                    background_angle: float = 0.0) -> StructureField:  # fmt: skip
    """The guides and row field of a patch (see the module docstring)."""
    edges = find_edges(luminance, smoothness, strength, min_length)
    if visible is not None:
        edges &= visible
    guides = edges.copy()
    if visible is not None and follow_mask:
        guides |= mask_edge(visible)
    spacing = max(spacing, 1e-9)
    straight, straight_grad = straight_rows(guides.shape, spacing, background_angle)
    if guides.any():
        distance = ndimage.distance_transform_edt(~guides).astype(np.float32)
        rows = distance / np.float32(spacing)
        background = (rows >= outline_rows) if outline_rows > 0 else np.zeros(rows.shape, bool)
    else:  # nothing to follow: straight rows everywhere
        rows, background = straight, np.ones(straight.shape, bool)
    gy, gx = np.gradient(rows)
    grad = np.stack([gx, gy]).astype(np.float32)
    fine = grain(luminance, GRAIN_BLUR * spacing, spacing, weight=rows >= 0.5)
    if reference is not None:
        fine = replace(fine, strong=strong_gradient(reference, GRAIN_BLUR * spacing))
    texture = (texture_zones(fine, rows, spacing, strength, visible) if follow_texture
               else np.zeros(rows.shape, bool))  # fmt: skip
    background &= ~texture
    along = np.where(background, np.float32(straight_grad[0]), grad[0])  # rows' direction
    across = np.where(background, np.float32(straight_grad[1]), grad[1])
    flow = blend_flow(np.stack([along, across]), spacing, fine, texture, spacing)
    return StructureField(edges, guides, rows, grad, flow, texture, background, straight,
                          straight_grad, spacing)  # fmt: skip

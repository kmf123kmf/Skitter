"""Contour rows: structure analysis, row placement, and the slicing operation."""

import math

import numpy as np
import pytest

from skitter.core.slicing import (
    MosaicLayout,
    SliceContext,
    SlicingPlan,
    Stage,
    operation_types,
    summarize,
)
from skitter.core.slicing.operations import ContourSlicer, PileSlicer
from skitter.core.slicing.rows import place_rows
from skitter.core.slicing.structure import find_edges, mask_edge, structure_field


def disk(size=120, radius=40.0, inside=220, outside=40) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    r = np.hypot(x + 0.5 - size / 2, y + 0.5 - size / 2)
    return np.where(r < radius, inside, outside).astype(np.float32)


SQUARE = math.pi / 2  # a square tile looks the same turned a quarter turn


def angle_gap(a, b, period=math.pi):
    """Difference of two directions (mod period: a half turn for a row's direction or
    a rectangle, a quarter turn for a square), in radians."""
    d = np.mod(np.asarray(a) - np.asarray(b), period)
    return np.minimum(d, period - d)


# Structure


def test_edges_are_thin_and_on_the_outline():
    edges = find_edges(disk(), smoothness=1.5, strength=0.3, min_length=0)
    y, x = np.nonzero(edges)
    r = np.hypot(x + 0.5 - 60, y + 0.5 - 60)
    assert len(r) > 2 * math.pi * 40 * 0.8  # most of the way round
    assert np.all(np.abs(r - 40) < 2.0)
    assert not find_edges(np.full((50, 50), 90.0), 1.0, 0.3, 0).any()  # flat: none


def test_short_edges_are_dropped():
    image = np.full((100, 100), 50.0, np.float32)
    image[20:24, 10:14] = 200  # a speck: its outline is short
    image[60:, :] = 200  # a long straight edge
    edges = find_edges(image, smoothness=1.0, strength=0.3, min_length=30)
    assert edges[55:65].any() and not edges[10:30, 0:30].any()


def test_mask_edge_is_the_visible_rim():
    visible = np.zeros((20, 20), bool)
    visible[5:15, 5:15] = True
    rim = mask_edge(visible)
    assert rim[5, 5:15].all() and rim[5:15, 5].all() and not rim[6:14, 6:14].any()
    assert not rim[~visible].any()
    assert not mask_edge(np.ones((5, 5), bool)).any()


def test_rows_count_distance_to_the_guides_in_row_heights():
    field = structure_field(disk(160), 8.0, smoothness=1.5, strength=0.3, min_length=0)
    r = field.rows[80, 80 + 40 + 24]  # 3 rows of 8 samples out from the outline
    assert r == pytest.approx(3.0, abs=0.3)
    flow_x, flow_y = field.flow[:, 80, 80 + 40 + 12]  # right of the disk: rows run up
    assert abs(flow_x) < 0.2 and abs(flow_y) > 0.95


def test_no_guides_means_straight_rows_across():
    field = structure_field(np.full((40, 60), 100.0), 8.0, smoothness=1, strength=0.3,
                            min_length=0)  # fmt: skip
    np.testing.assert_allclose(field.rows[:, 30], (np.arange(40) + 0.5) / 8.0)


# Placement


def largest_gap(count: np.ndarray) -> int:
    """Samples in the largest uncovered patch (8-connected)."""
    from scipy import ndimage

    labels, n = ndimage.label(count == 0, structure=np.ones((3, 3), bool))
    return int(np.bincount(labels.ravel())[1:].max()) if n else 0


def covered(centers, angles, length, height, shape) -> np.ndarray:
    """(H, W) count of tiles over each sample."""
    h, w = shape
    y, x = np.mgrid[:h, :w] + 0.5
    count = np.zeros(shape, np.int64)
    for (cx, cy), a in zip(centers, angles, strict=True):
        c, s = math.cos(a), math.sin(a)
        lx, ly = c * (x - cx) + s * (y - cy), -s * (x - cx) + c * (y - cy)
        count += (np.abs(lx) <= length / 2) & (np.abs(ly) <= height / 2)
    return count


def test_straight_rows_lie_on_their_centerlines_and_cover_everything():
    field = structure_field(np.full((64, 96), 100.0), 8.0, smoothness=1, strength=0.3,
                            min_length=0)  # fmt: skip
    visible = np.ones(field.rows.shape, bool)
    centers, angles, z = place_rows(field, visible, 8.0, 8.0, 0.2)
    rows = z > z.min()
    assert np.all(angle_gap(angles[rows], 0.0) < 1e-3)  # along the rows
    np.testing.assert_allclose(np.mod(centers[rows, 1], 8.0), 4.0, atol=1e-3)  # centerlines
    assert covered(centers, angles, 8.0, 8.0, visible.shape).min() >= 1  # rows meet exactly


def test_rows_follow_an_outline_and_keep_to_the_overlap_limit():
    image = disk(160, 50.0)
    field = structure_field(image, 8.0, smoothness=1.5, strength=0.3, min_length=0)
    visible = np.ones(image.shape, bool)
    centers, angles, z = place_rows(field, visible, 8.0, 8.0, 0.2)
    first = z == 0  # the rows hugging the outline, both sides of it
    out = centers[first] - 80
    tangent = np.arctan2(out[:, 0], -out[:, 1])  # perpendicular to the radius
    assert first.sum() > 40
    assert np.all(angle_gap(angles[first], tangent) < math.radians(15))
    # Rows hold to the limit among themselves (fillers lie under them anyway).
    rows = np.flatnonzero(z > z.min())
    later = covered(centers[rows], angles[rows], 8.0, 8.0, image.shape)
    assert (later > 1).mean() < 0.2 * (later > 0).mean() + 0.02
    count = covered(centers, angles, 8.0, 8.0, image.shape)
    assert (count == 0).mean() < 0.02  # only slivers of grout are left
    assert largest_gap(count) < 0.1 * 64  # none of them close to a tile's worth


def test_rows_stay_off_hidden_samples():
    field = structure_field(np.full((64, 64), 100.0), 8.0, smoothness=1, strength=0.3,
                            min_length=0)  # fmt: skip
    visible = np.zeros(field.rows.shape, bool)
    visible[:, :32] = True
    centers, _, _ = place_rows(field, visible, 8.0, 8.0, 0.2)
    assert np.all(centers[:, 0] < 32 + 4)


# The operation


def context(image, columns=12, aspect=1.0) -> SliceContext:
    return SliceContext(image, MosaicLayout(columns=columns, tile_aspect=aspect))


def parts(op: ContourSlicer, ctx: SliceContext):
    """The operation's own tiles over the canvas, with its z (rows by distance from
    the guides, 0 nearest; fillers lowest), before a plan restacks them."""
    return op.subdivide(ctx.canvas()[0], ctx)


def rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[..., None], 3, axis=2)


def test_contour_rows_is_registered():
    assert ContourSlicer in operation_types()


@pytest.mark.parametrize("aspect", [1.0, 1.5, 2 / 3])
def test_tiles_keep_the_base_shape_long_side_along_the_row(aspect):
    ctx = context(rgb(np.full((120, 180), 100.0)), columns=12, aspect=aspect)
    regions = parts(ContourSlicer(), ctx)
    np.testing.assert_allclose(regions.size, np.broadcast_to(ctx.tile_size, regions.size.shape))
    # Straight rows across: the long side lies horizontal.
    long_axis = np.where(regions.size[:, 0] >= regions.size[:, 1], 0.0, math.pi / 2)
    rows = regions.z > regions.z.min()
    assert np.all(angle_gap(regions.rotation[rows] + long_axis[rows], 0.0) < 1e-3)


def test_slicing_is_deterministic_and_covers_the_picture():
    ctx = context(rgb(disk(240, 80.0)), columns=24)
    plan = SlicingPlan([Stage(ContourSlicer())])
    a, b = plan.regions(ctx), plan.regions(ctx)
    np.testing.assert_array_equal(a.data, b.data)
    assert summarize(a, ctx).coverage > 0.98  # slivers of grout aside
    assert len(a) < 1.5 * 24 * 24  # not wildly more tiles than a grid


def test_rows_follow_the_mask_edge():
    image = np.zeros((240, 240, 4), np.uint8)
    image[..., :3] = 120  # flat: the only structure is the oval's edge
    y, x = np.mgrid[:240, :240]
    image[..., 3] = np.where(np.hypot(x + 0.5 - 120, y + 0.5 - 120) <= 100, 255, 0)
    ctx = context(image, columns=24)
    regions = parts(ContourSlicer(), ctx)
    first = regions.z == 0  # the row along the edge
    out = regions.center[first] / ctx.scale - 120
    tangent = np.arctan2(out[:, 0], -out[:, 1])
    assert first.sum() > 30
    assert np.all(angle_gap(regions.rotation[first], tangent, SQUARE) < math.radians(15))
    off = parts(ContourSlicer(follow_mask=False), ctx)
    straight = off.z > off.z.min()
    assert np.all(angle_gap(off.rotation[straight], 0.0, SQUARE) < 1e-3)  # rows across instead


def test_works_inside_rotated_regions():
    ctx = context(rgb(disk(240, 80.0)), columns=6)
    plan = SlicingPlan([Stage(PileSlicer(photo_size=3.0, rotation=30.0, seed=4)),
                        Stage(ContourSlicer(tile_size=0.5))])  # fmt: skip
    regions = plan.regions(ctx)
    assert len(regions)
    np.testing.assert_allclose(regions.size, 0.5 * np.asarray(ctx.tile_size)[None, :].repeat(
        len(regions), axis=0))  # fmt: skip


def test_no_edges_outside_the_picture():
    # Hidden pixels hold a hard edge (as a fill's seams would): it guides nothing.
    image = np.full((80, 120), 60.0, np.float32)
    image[:, 90:] = 230
    visible = np.zeros(image.shape, bool)
    visible[:, :70] = True
    field = structure_field(image, 8.0, smoothness=1.0, strength=0.3, min_length=0,
                            visible=visible)  # fmt: skip
    assert not field.edges.any()
    assert field.guides[:, 69].all() and not field.guides[:, 70:].any()  # the mask's edge


# Texture


def striped(size=240, period=10.0, degrees=30.0) -> np.ndarray:
    """Flat gray with a square of fine stripes (thinner than a tile) running at an angle."""
    angle = math.radians(degrees)
    image = np.full((size, size), 110.0, np.float32)
    y, x = np.mgrid[:size, :size] + 0.5
    across = -math.sin(angle) * x + math.cos(angle) * y  # distance across the stripes
    stripes = np.where(np.sin(2 * math.pi * across / period) > 0, 200.0, 30.0)
    inner = slice(size // 6, size - size // 6)
    image[inner, inner] = stripes[inner, inner]
    return image


def test_fine_stripes_are_texture_and_tiles_follow_them():
    angle = math.radians(30)
    ctx = context(rgb(striped(degrees=30.0)), columns=12)  # 20 px tiles, 5 px stripes
    field, _, scale = ContourSlicer().structure(ctx.canvas()[0], ctx)
    middle = field.texture[field.texture.shape[0] // 2 - 10 : field.texture.shape[0] // 2 + 10]
    assert middle.mean() > 0.3  # the striped square's middle is texture
    finer = context(rgb(striped(period=4.0)), columns=12)  # finer than the analysis sees:
    assert not ContourSlicer().structure(finer.canvas()[0], finer)[0].texture.any()  # no moire
    tiles = parts(ContourSlicer(), ctx)
    center = tiles.center / ctx.scale
    inside = np.all(np.abs(center - 120) < 40, axis=1)
    along = angle_gap(tiles.rotation[inside], angle) < math.radians(15)
    assert along.mean() > 0.8  # tiles in the stripes run along them
    across = parts(ContourSlicer(follow_texture=False), ctx)
    inside_off = np.all(np.abs(across.center / ctx.scale - 120) < 40, axis=1)
    assert (angle_gap(across.rotation[inside_off], angle) < math.radians(15)).mean() < 0.6


def test_no_texture_beside_a_plain_outline():
    # A strong outline's own gradient must not read as texture beside it.
    ctx = context(rgb(disk(240, 80.0)), columns=24)
    field, _, _ = ContourSlicer().structure(ctx.canvas()[0], ctx)
    assert not field.texture.any()


# Background


def background_tiles(op: ContourSlicer, ctx: SliceContext):
    """(tiles, mask of those centered on the background) for an operation over the canvas."""
    field, _, scale = op.structure(ctx.canvas()[0], ctx)
    tiles = parts(op, ctx)
    h, w = field.background.shape
    i = np.clip((tiles.center[:, 0] * scale).astype(int), 0, w - 1)
    j = np.clip((tiles.center[:, 1] * scale).astype(int), 0, h - 1)
    return tiles, field.background[j, i]


@pytest.mark.parametrize("degrees", [0.0, 45.0, -30.0])
def test_past_the_outline_rows_the_background_runs_straight(degrees):
    ctx = context(rgb(disk(240, 50.0)), columns=24)
    op = ContourSlicer(outline_rows=2, background_angle=degrees)
    tiles, back = background_tiles(op, ctx)
    rows = tiles.z == np.floor(tiles.z)  # rows along level sets (not flow rows, fillers)
    rows &= tiles.z > tiles.z.min()
    assert (back & rows).sum() > 100
    assert np.all(angle_gap(tiles.rotation[back & rows], math.radians(degrees), SQUARE) < 1e-3)
    # Near the outline, rows still follow it.
    out = tiles.center[~back & rows] / ctx.scale - 120
    radius = np.hypot(*out.T)
    near = np.abs(radius - 50) < 15
    tangent = np.arctan2(out[near, 0], -out[near, 1])
    assert np.all(angle_gap(tiles.rotation[~back & rows][near], tangent, SQUARE) < math.radians(15))


def test_outline_rows_zero_ripples_everywhere():
    ctx = context(rgb(disk(240, 50.0)), columns=24)
    field, _, _ = ContourSlicer(outline_rows=0).structure(ctx.canvas()[0], ctx)
    assert not field.background.any()
    field, _, _ = ContourSlicer(outline_rows=3).structure(ctx.canvas()[0], ctx)
    np.testing.assert_array_equal(field.background, field.rows >= 3)  # (no texture here)


def test_no_outlines_at_all_means_straight_rows_at_the_background_angle():
    ctx = context(rgb(np.full((120, 180), 100.0)), columns=12)
    tiles, back = background_tiles(ContourSlicer(background_angle=30.0), ctx)
    assert back.all()
    rows = (tiles.z == np.floor(tiles.z)) & (tiles.z > tiles.z.min())
    assert np.all(angle_gap(tiles.rotation[rows], math.radians(30.0)) < 1e-3)


@pytest.mark.parametrize("aspect", [1.0, 1.5, 2 / 3])
def test_tiles_face_as_upright_as_their_rows_allow(aspect):
    # Rows run every way around a disk; no tile may end up upside down.
    ctx = context(rgb(disk(240, 60.0)), columns=24, aspect=aspect)
    tiles = parts(ContourSlicer(outline_rows=0), ctx)
    limit = math.pi / 4 if aspect == 1.0 else math.pi / 2
    assert np.all(np.abs(tiles.rotation) <= limit + 1e-9)
    assert np.abs(tiles.rotation).max() > 0.8 * limit  # rows do turn all the way round

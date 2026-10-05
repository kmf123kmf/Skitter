"""Transparent sources: alpha as a solid mask, its edge treated like the border."""

import math

import numpy as np
import pytest
from PIL import Image

from skitter.core.edits import Crop, FlipHorizontal, Rotate90, apply_edits
from skitter.core.imaging import fill_hidden, load_image, visible_mask
from skitter.core.slicing import (
    MosaicLayout,
    RegionSet,
    SliceContext,
    SlicingPlan,
    Stage,
    coverage,
    mask_regions,
    summarize,
    touching,
)
from skitter.core.slicing.operations import GridSlicer, PileSlicer


def rgba(width=100, height=60, alpha=255, seed=0) -> np.ndarray:
    image = np.random.default_rng(seed).integers(0, 256, (height, width, 4), dtype=np.uint8)
    image[..., 3] = alpha
    return image


def oval(width=100, height=60) -> np.ndarray:
    """An RGBA image visible inside the ellipse inscribed in it."""
    image = rgba(width, height)
    y, x = np.mgrid[:height, :width]
    inside = ((x + 0.5 - width / 2) / (width / 2)) ** 2 + (
        (y + 0.5 - height / 2) / (height / 2)
    ) ** 2
    image[..., 3] = np.where(inside <= 1, 255, 0)
    return image


def covers_visible(regions: RegionSet, ctx: SliceContext, samples: int = 64) -> np.ndarray:
    """Brute force: (R,) whether a dense grid of points in each region hits a visible pixel
    (an edge on a pixel boundary doesn't reach the pixel beyond it)."""
    h, w = ctx.visible.shape
    u = np.linspace(-0.5, 0.5, samples)
    inset = 2e-6 * ctx.scale
    hit = np.zeros(len(regions), bool)
    for i, region in enumerate(regions):
        lx, ly = np.meshgrid(u * (region.width - inset), u * (region.height - inset))
        c, s = math.cos(region.rotation), math.sin(region.rotation)
        x = (region.cx + c * lx - s * ly) / ctx.scale
        y = (region.cy + s * lx + c * ly) / ctx.scale
        xi = np.clip(np.floor(x), 0, w - 1).astype(int)
        yi = np.clip(np.floor(y), 0, h - 1).astype(int)
        hit[i] = ctx.visible[yi, xi].any()
    return hit


# Loading and the mask


def test_sources_load_as_rgba_keeping_transparency(tmp_path):
    image = rgba(alpha=0)
    image[:, 50:, 3] = 200
    path = tmp_path / "cutout.png"
    Image.fromarray(image, "RGBA").save(path)
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded, image)
    Image.fromarray(image[..., :3]).save(tmp_path / "photo.png")
    assert (load_image(tmp_path / "photo.png")[..., 3] == 255).all()  # opaque


def test_alpha_is_a_solid_mask_at_half():
    image = rgba(4, 1)
    image[0, :, 3] = [0, 127, 128, 255]
    assert visible_mask(image).tolist() == [[False, False, True, True]]
    assert visible_mask(rgba()) is None  # all opaque: no mask
    assert visible_mask(rgba()[..., :3]) is None


def test_hidden_pixels_take_the_nearest_visible_color():
    rgb = np.zeros((1, 6, 3), np.uint8)
    rgb[0, 1] = (10, 20, 30)
    rgb[0, 4] = (200, 100, 50)
    visible = np.array([[False, True, False, False, True, False]])
    filled = fill_hidden(rgb, visible)
    assert filled[0, :, 0].tolist() == [10, 10, 10, 200, 200, 200]
    np.testing.assert_array_equal(fill_hidden(rgb, np.ones((1, 6), bool)), rgb)


def test_edits_carry_the_mask_with_the_picture():
    image = rgba(alpha=0)
    image[:10, :20, 3] = 255  # visible top left
    # Flipped: top right; turned clockwise: bottom right (rows 80..99, columns 50..59).
    edited = apply_edits(image, [FlipHorizontal(), Rotate90(1), Crop(30, 60, 30, 40)])
    expected = np.rot90(image[:, ::-1], k=-1)[60:, 30:]
    np.testing.assert_array_equal(edited, expected)
    assert (edited[20:, 20:, 3] == 255).all() and not edited[:20, :, 3].any()


def test_context_reads_hidden_pixels_as_past_the_border():
    image = rgba(alpha=0)
    image[:, 60:, 3] = 255
    ctx = SliceContext(image)
    assert ctx.image.shape == (60, 100, 3)
    np.testing.assert_array_equal(ctx.image[:, 60:], image[:, 60:, :3])
    np.testing.assert_array_equal(ctx.image[:, 0], image[:, 60, :3])  # the nearest edge
    opaque = SliceContext(rgba())
    assert opaque.visible is None
    np.testing.assert_array_equal(opaque.image, rgba()[..., :3])


# Slicing


def test_grid_keeps_tiles_touching_the_picture_whole():
    image = rgba(alpha=0)
    image[:, 55:, 3] = 255  # visible from x = 55: the column of tiles at 50..60 touches it
    ctx = SliceContext(image, MosaicLayout(columns=10, tile_aspect=1.0), tile_width=10)
    regions = SlicingPlan.default().regions(ctx)
    left = regions.center[:, 0] - regions.size[:, 0] / 2
    assert sorted(set(left.round(6).tolist())) == [50, 60, 70, 80, 90]
    np.testing.assert_allclose(regions.size, 10)  # whole, overhanging the mask's edge
    assert len(regions) == 5 * 6


def test_rotated_tiles_are_kept_exactly_when_they_touch_a_visible_pixel():
    ctx = SliceContext(oval(), MosaicLayout(columns=12, tile_aspect=1.0), tile_width=10)
    plan = SlicingPlan([Stage(PileSlicer(rotation=30.0, seed=3))])
    every = plan.evaluate(ctx)[-1].regions
    keep = touching(every, ctx)
    assert 0 < keep.sum() < len(every)
    np.testing.assert_array_equal(keep, covers_visible(every, ctx))
    kept = plan.regions(ctx)
    assert len(kept) == keep.sum()
    np.testing.assert_array_equal(kept.z, every.z[keep])  # stacking kept


def test_an_oval_mask_drops_the_corners():
    ctx = SliceContext(oval(200, 120), MosaicLayout(columns=20, tile_aspect=1.0), tile_width=10)
    regions = SlicingPlan.default().regions(ctx)
    assert len(regions) < 20 * 12
    corners = [(5, 5), (195, 5), (5, 115), (195, 115)]
    for x, y in corners:
        assert not regions.contains(x, y).any()
    assert regions.contains(100, 60).any()


def test_without_a_mask_nothing_is_dropped():
    ctx = SliceContext(rgba(), MosaicLayout(columns=10, tile_aspect=1.0), tile_width=10)
    regions = SlicingPlan.default().evaluate(ctx)[-1].regions
    assert mask_regions(regions, ctx) is regions


def test_tiles_past_the_border_read_its_edge():
    image = rgba(alpha=0)
    image[:, -1, 3] = 255  # only the last column is visible
    ctx = SliceContext(image)
    outside = RegionSet.from_rects([105, -20], [10, 10], 4, 4)  # right of it; left of it
    assert touching(outside, ctx).tolist() == [True, False]


def test_coverage_and_density_count_the_visible_part_only():
    image = rgba(alpha=0)
    image[:, 50:, 3] = 255
    ctx = SliceContext(image, MosaicLayout(columns=10, tile_aspect=1.0), tile_width=10)
    regions = SlicingPlan.default().regions(ctx)
    assert coverage(regions, ctx.width, ctx.height, 1.0, ctx=ctx)[0] == pytest.approx(1.0)
    assert coverage(regions, ctx.width, ctx.height, 1.0)[0] == pytest.approx(0.5, abs=0.02)
    summary = summarize(regions, ctx)
    assert summary.coverage == pytest.approx(1.0)
    assert summary.density == pytest.approx(1.0)  # 30 tiles on 30 tiles' worth of picture


def test_grid_slicer_unchanged_by_an_opaque_alpha_channel():
    layout = MosaicLayout(columns=8, tile_aspect=1.0)
    plan = SlicingPlan([Stage(GridSlicer())])
    a = plan.regions(SliceContext(rgba()[..., :3], layout, tile_width=10))
    b = plan.regions(SliceContext(rgba(), layout, tile_width=10))
    np.testing.assert_array_equal(a.data, b.data)


# Shapes


def ring(width=300, height=200, inner=45, outer=90) -> np.ndarray:
    """Alpha of a donut centered in the image (radii in pixels)."""
    y, x = np.mgrid[:height, :width]
    r = np.hypot(x + 0.5 - width / 2, y + 0.5 - height / 2)
    return np.where((r >= inner) & (r <= outer), 255, 0).astype(np.uint8)


def masked(alpha: np.ndarray, columns=20) -> SliceContext:
    image = rgba(alpha.shape[1], alpha.shape[0])
    image[..., 3] = alpha
    return SliceContext(image, MosaicLayout(columns=columns, tile_aspect=1.0), tile_width=10)


def pixel_exact(regions: RegionSet, ctx: SliceContext) -> np.ndarray:
    """(R,) for upright regions: whether a visible pixel overlaps the region's inside."""
    h, w = ctx.visible.shape
    hit = np.zeros(len(regions), bool)
    for i, region in enumerate(regions):
        x0, x1 = (
            (region.cx - region.width / 2) / ctx.scale,
            (region.cx + region.width / 2) / ctx.scale,
        )
        y0, y1 = (
            (region.cy - region.height / 2) / ctx.scale,
            (region.cy + region.height / 2) / ctx.scale,
        )
        i0, i1 = math.floor(x0 + 1e-6), math.ceil(x1 - 1e-6)
        j0, j1 = math.floor(y0 + 1e-6), math.ceil(y1 - 1e-6)
        hit[i] = ctx.visible[max(j0, 0) : min(j1, h), max(i0, 0) : min(i1, w)].any()
    return hit


@pytest.mark.parametrize("plan", ["grid", "pile"])
def test_a_donut_leaves_its_hole_empty(plan):
    ctx = masked(ring())
    stage = GridSlicer() if plan == "grid" else PileSlicer(rotation=30.0, seed=2)
    every = SlicingPlan([Stage(stage)]).evaluate(ctx)[-1].regions
    keep = touching(every, ctx)
    kept = every[np.flatnonzero(keep)]
    assert not kept.contains(ctx.width / 2, ctx.height / 2).any()  # the hole
    assert kept.contains(ctx.width / 2 + 67 * ctx.scale, ctx.height / 2).any()  # the ring
    truth = pixel_exact(every, ctx) if plan == "grid" else covers_visible(every, ctx, 96)
    np.testing.assert_array_equal(keep, truth)


def test_tile_edges_on_pixel_boundaries_dont_reach_the_next_pixel():
    # 15 px tiles: every edge lies on a pixel boundary (give or take rounding).
    alpha = np.zeros((200, 300), np.uint8)
    alpha[:, 150] = 255  # the first column of the tiles at 150..165
    ctx = masked(alpha)
    regions = SlicingPlan.default().regions(ctx)
    left = (regions.center[:, 0] - regions.size[:, 0] / 2) / ctx.scale
    np.testing.assert_allclose(left, 150.0)  # not the column of tiles at 135..150


def test_holes_smaller_than_a_tile_are_covered():
    # As at the border, a tile touching the picture stays whole: a hole within one
    # tile's reach is covered (and matched to the colors around it).
    alpha = np.full((200, 300), 255, np.uint8)
    alpha[95:105, 145:155] = 0  # 10 px; tiles are 15 px
    ctx = masked(alpha)
    assert len(SlicingPlan.default().regions(ctx)) == len(
        SlicingPlan.default().evaluate(ctx)[-1].regions
    )


def test_islands_get_tiles_and_the_space_between_none():
    alpha = np.zeros((200, 300), np.uint8)
    alpha[40:80, 30:90] = 255
    alpha[120:170, 200:280] = 255
    alpha[190, 10] = 255  # one stray visible pixel: it gets a tile too
    ctx = masked(alpha)
    regions = SlicingPlan.default().regions(ctx)
    at = lambda x, y: regions.contains((x + 0.5) * ctx.scale, (y + 0.5) * ctx.scale).any()  # noqa: E731
    assert at(60, 60) and at(240, 145) and at(10, 190)
    assert not at(150, 100) and not at(280, 20)


def test_thin_lines_under_large_rotated_tiles_are_not_missed():
    for column in (101, 150, 203):
        alpha = np.zeros((300, 300), np.uint8)
        alpha[:, column] = 255
        ctx = masked(alpha, columns=3)  # 100 px tiles
        every = (
            SlicingPlan([Stage(PileSlicer(rotation=40.0, seed=column))]).evaluate(ctx)[-1].regions
        )
        keep = touching(every, ctx)
        assert keep.any()
        np.testing.assert_array_equal(keep, covers_visible(every, ctx, 600))

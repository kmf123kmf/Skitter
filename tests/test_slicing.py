import math

import numpy as np
import pytest

from skitter.core.slicing import (
    MAX_REGIONS,
    TILE_UNIT,
    BoolParam,
    FloatParam,
    IntParam,
    MosaicLayout,
    Region,
    RegionSet,
    SliceContext,
    SlicingError,
    SlicingOperation,
    SlicingPlan,
    Stage,
    Subdivider,
    coverage,
    get_operation_type,
    operation_from_dict,
    operation_types,
    register_operation,
    summarize,
)
from skitter.core.slicing.base import _registry
from skitter.core.slicing.operations import (
    BondSlicer,
    GridSlicer,
    JitterAdjust,
    PatternSlicer,
    QuadtreeSlicer,
    SplitSlicer,
    StackingAdjust,
)


def blank(width=100, height=60):
    """One mosaic pixel per source pixel, 1 px base tiles."""
    return SliceContext(np.zeros((height, width, 3), np.uint8))


def tiled(width, height, tile, aspect=1.0):
    """Canvas the same size as the image, with tile x tile/aspect base tiles."""
    layout = MosaicLayout(tile_aspect=aspect, columns=width // tile)
    return SliceContext(np.zeros((height, width, 3), np.uint8), layout, tile_width=tile)


# Regions


def test_grid_cells_cover_area_row_major():
    cells = RegionSet.grid(100, 60, 4, 3)
    assert len(cells) == 12
    np.testing.assert_allclose(cells.size, [[25, 20]] * 12)
    np.testing.assert_allclose(cells.center[:2], [[12.5, 10], [37.5, 10]])
    assert cells.area().sum() == pytest.approx(6000)


def test_local_world_roundtrip_with_rotation():
    region = Region(50, 40, 20, 10, rotation=math.pi / 2)
    # Rotated 90° clockwise: the local top-left corner lands at top-right on screen.
    np.testing.assert_allclose(region.local_to_world([0, 0]), [55, 30])
    points = np.array([[3.0, 4.0], [20.0, 10.0]])
    np.testing.assert_allclose(region.world_to_local(region.local_to_world(points)), points)


def test_to_world_composes_rotation_and_position():
    parent = Region(50, 40, 20, 10, rotation=math.pi / 2)
    local = RegionSet.grid(20, 10, 2, 1)  # two 10x10 halves
    world = local.to_world(parent)
    np.testing.assert_allclose(world.center, [[50, 35], [50, 45]])
    np.testing.assert_allclose(world.rotation, [math.pi / 2] * 2)


def test_corners_bounds_and_hit_test():
    regions = RegionSet.from_arrays([[10, 10], [12, 10]], (4, 2), [0, math.pi / 2])
    np.testing.assert_allclose(regions.corners()[0], [[8, 9], [12, 9], [12, 11], [8, 11]])
    np.testing.assert_allclose(regions.bounds()[1], [11, 8, 13, 12])
    assert regions.hit_test(12, 11.5) == 1  # only the rotated (tall) region reaches here
    assert regions.hit_test(11.5, 10) == 1  # both contain it; the last one is on top
    assert regions.hit_test(0, 0) == -1


def test_region_sets_are_immutable():
    regions = RegionSet.covering(10, 10)
    with pytest.raises(ValueError):
        regions.center[0] = 5
    moved = regions.replace(center=[[1, 1]])
    assert regions[0].cx == 5 and moved[0].cx == 1


def test_indexing_and_concat():
    cells = RegionSet.grid(10, 10, 2, 2)
    assert isinstance(cells[0], Region)
    assert len(cells[1:3]) == 2
    assert len(cells[cells.center[:, 0] < 5]) == 2
    assert len(RegionSet.concat([cells, cells])) == 8
    assert len(RegionSet.concat([])) == 0


# Parameters and registry


class _Demo(SlicingOperation):
    id = "test-demo"
    name = "Demo"
    count = IntParam(3, min=1, max=10)
    ratio = FloatParam(0.5, min=0, max=1)
    flag = BoolParam(False)
    extra = IntParam(1, when=lambda op: op.flag)

    def apply(self, regions, ctx):
        return regions


def test_params_defaults_order_and_labels():
    op = _Demo()
    assert [p.name for p in op.params()] == ["count", "ratio", "flag", "extra"]
    assert op.values() == {"count": 3, "ratio": 0.5, "flag": False, "extra": 1}
    assert _Demo.count.label == "Count"


def test_params_validate():
    op = _Demo(count=4)
    assert op.count == 4
    with pytest.raises(ValueError):
        op.count = 11
    with pytest.raises(ValueError):
        op.count = 2.5
    with pytest.raises(ValueError):
        op.flag = 1
    with pytest.raises(KeyError):
        op.update(nope=1)


def test_conditional_params():
    op = _Demo()
    assert not _Demo.extra.is_active(op)
    op.flag = True
    assert _Demo.extra.is_active(op)


def test_registry_lists_builtins_in_category_order():
    ids = [cls.id for cls in operation_types()]
    assert ids.index("grid") < ids.index("jitter")
    assert {"grid", "quadtree", "jitter", "stacking"} <= set(ids)
    assert get_operation_type("grid") is GridSlicer


def test_register_rejects_duplicate_ids():
    class Clash(_Demo):
        id = "grid"

    with pytest.raises(ValueError):
        register_operation(Clash)


def test_register_custom_operation_roundtrips():
    @register_operation
    class Halve(Subdivider):
        id = "test-halve"
        name = "Halve"

        def subdivide(self, region, ctx):
            return RegionSet.grid(region.width, region.height, 2, 1)

    try:
        op = operation_from_dict({"type": "test-halve", "params": {}})
        plan = SlicingPlan([Stage(op), Stage(Halve())])
        assert len(plan.regions(blank())) == 4
    finally:
        del _registry["test-halve"]


# Plans


def test_default_plan_is_grid_of_base_tiles():
    regions = SlicingPlan.default().regions(tiled(240, 120, tile=10))
    assert len(regions) == 24 * 12
    np.testing.assert_allclose(regions.size[0], [10, 10])


def test_empty_plan_yields_whole_image():
    regions = SlicingPlan().regions(blank(100, 60))
    assert len(regions) == 1 and regions[0].width == 100


def test_disabled_stage_passes_through():
    plan = SlicingPlan(
        [
            Stage(SplitSlicer(across=4)),
            Stage(SplitSlicer(across=2), enabled=False),
        ]
    )
    assert len(plan.regions(blank(100, 100))) == 16


def test_evaluate_reuses_unchanged_prefix():
    calls = []

    class Counting(SplitSlicer):
        id = "test-counting"

        def apply(self, regions, ctx):
            calls.append(self.across)
            return super().apply(regions, ctx)

    plan = SlicingPlan([Stage(Counting(across=2)), Stage(Counting(across=3))])
    ctx = blank(60, 60)
    cache = plan.evaluate(ctx)
    plan.stages[1].operation.across = 4
    cache = plan.evaluate(ctx, cache)
    assert calls == [2, 3, 4]  # the first stage was not re-run
    assert len(cache[-1].regions) == 4 * 16


def test_too_many_regions_raises():
    grid = SplitSlicer(across=1000, keep_shape=False, down=1000)
    plan = SlicingPlan([Stage(grid)])
    with pytest.raises(SlicingError, match=f"{MAX_REGIONS:,}"):
        plan.evaluate(blank())


def test_too_many_regions_across_parents_fails_before_finishing():
    calls = []

    class Counting(SplitSlicer):
        def subdivide(self, region, ctx):
            calls.append(region)
            return super().subdivide(region, ctx)

    plan = SlicingPlan(
        [
            Stage(SplitSlicer(across=4, keep_shape=False, down=1)),
            Stage(Counting(across=400, keep_shape=False, down=400)),
        ]
    )
    with pytest.raises(SlicingError, match=f"{MAX_REGIONS:,}"):
        plan.evaluate(blank())
    assert len(calls) == 2  # stopped once the running total passed the limit


def test_plan_serialization_roundtrip():
    plan = SlicingPlan(
        [
            Stage(GridSlicer(cell_size=2)),
            Stage(JitterAdjust(seed=7), enabled=False),
            Stage(StackingAdjust(order="reverse")),
        ]
    )
    data = plan.to_dict()
    assert data["stages"][1] == {
        "enabled": False,
        "type": "jitter",
        "params": JitterAdjust(seed=7).values(),
    }
    assert SlicingPlan.from_dict(data).to_dict() == data


# Built-in operations


def test_grid_has_only_tile_settings():
    assert [p.name for p in GridSlicer.params()] == ["cell_size", "anchor"]


def test_split_keeps_the_tile_shape():
    op = SplitSlicer(across=10)
    assert op.down_for(100, 50, tile_aspect=1.0) == 5  # 10 x 10 pieces
    assert op.down_for(100, 50, tile_aspect=2.0) == 10  # 10 x 5 pieces, 2:1 like the tile
    assert op.down_for(100, 1, tile_aspect=1.0) == 1  # never fewer than one
    op.keep_shape = False
    op.down = 3
    assert op.down_for(100, 50, tile_aspect=1.0) == 3
    ctx = tiled(120, 60, tile=30, aspect=1.5)  # 30 x 20 tiles
    pieces = SplitSlicer(across=4).apply(ctx.canvas(), ctx)
    np.testing.assert_allclose(pieces.size[0], [30, 20])  # exactly the tile shape
    assert len(pieces) == 4 * 3


def test_split_inside_rotated_region_stays_inside():
    parent = RegionSet.from_arrays([[50, 50]], (40, 20), math.pi / 6)
    grid = SplitSlicer(across=4, keep_shape=False, down=2)
    cells = grid.apply(parent, blank())
    assert len(cells) == 8
    for corner in cells.corners().reshape(-1, 2):
        assert parent.contains(*(corner * 0.999 + parent.center[0] * 0.001))[0]


def test_quadtree_splits_only_where_detail_is():
    image = np.zeros((64, 64, 3), np.uint8)
    image[:32, 32:] = (np.indices((32, 32)).sum(0) % 2 * 255)[..., None]  # busy top-right
    regions = QuadtreeSlicer(threshold=5, min_size=4, max_depth=3).apply(
        RegionSet.covering(64, 64), SliceContext(image)
    )
    sizes = sorted({tuple(s) for s in regions.size.tolist()})
    assert sizes == [(8.0, 8.0), (32.0, 32.0)]
    assert len(regions) == 3 + 16  # three flat quarters + busy quarter split twice


def test_quadtree_flat_image_stays_whole():
    regions = QuadtreeSlicer().apply(RegionSet.covering(64, 64), blank(64, 64))
    assert len(regions) == 1


def test_patch_samples_in_region_frame():
    image = np.zeros((10, 10, 3), np.uint8)
    image[2, 7] = 255  # one bright pixel
    ctx = SliceContext(image)
    patch, scale = ctx.patch(Region(7.5, 2.5, 1, 1), source="rgb")
    assert scale == 1 and patch.shape == (1, 1, 3) and patch[0, 0, 0] == 255
    # A 4x2 region rotated 90°: its local top-left maps to the top-right on screen.
    rotated = Region(5, 5, 4, 2, rotation=math.pi / 2)
    patch, _ = ctx.patch(rotated)
    assert patch.shape == (2, 4)


def test_jitter_is_deterministic_and_zero_is_identity():
    cells = RegionSet.grid(100, 100, 5, 5)
    a = JitterAdjust(seed=3).apply(cells, blank())
    b = JitterAdjust(seed=3).apply(cells, blank())
    np.testing.assert_array_equal(a.data, b.data)
    still = JitterAdjust(offset=0, rotation=0, scale=0).apply(cells, blank())
    np.testing.assert_array_equal(still.data, cells.data)


# Stacking (z-order)


def test_default_stacking_is_array_order_and_z_sorts():
    regions = RegionSet.from_arrays([[0, 0]] * 4, (1, 1), z=[0, 2, 0, 1])
    assert regions.stacking_order().tolist() == [0, 2, 3, 1]  # ties keep array order
    assert regions.stacking_rank().tolist() == [0, 3, 1, 2]
    assert regions[1].z == 2


def test_hit_test_returns_topmost_by_z():
    regions = RegionSet.from_arrays([[10, 10], [10, 10]], (4, 4), z=[5, 1])
    assert regions.hit_test(10, 10) == 0


def test_replace_keeps_z_and_restacked_normalizes():
    regions = RegionSet.from_arrays([[0, 0], [1, 1], [2, 2]], (1, 1), z=[10, -3, 7])
    assert regions.replace(size=(2, 2)).z.tolist() == [10, -3, 7]
    restacked = regions.restacked([1, 1, 0])  # first two tie: keep their current order
    assert restacked.z.tolist() == [2, 1, 0]


def test_subdivider_keeps_parent_stack_positions():
    class Overlapping(Subdivider):
        id = "test-overlapping"
        name = "Overlapping"

        def subdivide(self, region, ctx):
            # Two parts; the first is given the higher local z (on top).
            return RegionSet.from_rects([0, 0], [0, 0], region.width, region.height, z=[1, 0])

    # Parent 0 lies above parent 1.
    parents = RegionSet.from_arrays([[5, 5], [6, 6]], (4, 4), z=[1, 0])
    parts = Overlapping().apply(parents, blank())
    # Parts 0,1 come from parent 0 and must both lie above parts 2,3.
    assert parts.z.tolist() == [3, 2, 1, 0]


def test_jitter_preserves_stacking():
    regions = RegionSet.from_arrays([[0, 0], [5, 5]], (2, 2), z=[1, 0])
    assert JitterAdjust().apply(regions, blank()).z.tolist() == [1, 0]
    assert "gap" not in [cls.id for cls in operation_types()]


def test_pile_covers_every_point_with_random_stacking():
    from skitter.core.slicing.operations import PileSlicer

    ctx = tiled(600, 400, tile=120, aspect=1.5)
    pile = PileSlicer(rotation=30, spread=1.0, portrait=0.5)  # 1.0: coverage guaranteed
    regions = pile.apply(ctx.canvas(), ctx)
    assert {tuple(size) for size in regions.size.tolist()} == {(120.0, 80.0), (80.0, 120.0)}
    assert sorted(regions.z.tolist()) == list(range(len(regions)))
    assert np.abs(regions.rotation).max() <= math.radians(30) + 1e-9
    ys, xs = np.mgrid[0:400:5, 0:600:5]
    for x, y in zip(xs.ravel() + 0.5, ys.ravel() + 0.5, strict=True):
        assert regions.contains(x, y).any(), (x, y)


def test_pile_is_deterministic_per_seed():
    from skitter.core.slicing.operations import PileSlicer

    ctx = tiled(500, 500, tile=50)
    a = PileSlicer(seed=4).apply(ctx.canvas(), ctx)
    b = PileSlicer(seed=4).apply(ctx.canvas(), ctx)
    c = PileSlicer(seed=5).apply(ctx.canvas(), ctx)
    np.testing.assert_array_equal(a.data, b.data)
    assert not np.array_equal(a.data, c.data)


def test_pile_handles_regions_off_the_top_left():
    from skitter.core.slicing.operations import PileSlicer

    ctx = tiled(500, 500, tile=50)
    regions = RegionSet.from_arrays([[-30.0, -40.0], [100.0, 100.0]], (100, 100))
    assert len(PileSlicer().apply(regions, ctx)) > 2


def test_stacking_order_operation():
    from skitter.core.slicing.operations import StackingAdjust

    regions = RegionSet.from_arrays([[0, 0]] * 3, [[3, 3], [1, 1], [2, 2]])
    small_on_top = StackingAdjust(order="small_on_top").apply(regions, blank())
    assert small_on_top.stacking_order().tolist() == [0, 2, 1]  # largest at the bottom
    large_on_top = StackingAdjust(order="large_on_top").apply(regions, blank())
    assert large_on_top.stacking_order().tolist() == [1, 2, 0]
    reverse = StackingAdjust(order="reverse").apply(regions, blank())
    assert reverse.stacking_order().tolist() == [2, 1, 0]
    shuffled = StackingAdjust(order="random", seed=3).apply(regions, blank())
    assert sorted(shuffled.z.tolist()) == [0, 1, 2]
    assert not StackingAdjust.seed.is_active(StackingAdjust(order="reverse"))


# Mosaic layout, mosaic-unit slicing, analysis


def test_layout_has_no_pixel_size():
    layout = MosaicLayout(tile_aspect=4 / 3, columns=40)
    assert layout.rows(3000, 2000) == pytest.approx(35.555, abs=1e-3)
    assert layout.whole_rows(3000, 2000) == 36
    assert layout.source_per_tile(3000) == 75
    assert MosaicLayout.from_dict(layout.to_dict()) == layout
    assert layout.to_dict() == {"tile_aspect": 4 / 3, "columns": 40}
    with pytest.raises(ValueError):
        MosaicLayout(columns=0)


def test_context_sizes_the_canvas_in_mosaic_units():
    image = np.zeros((2000, 3000, 3), np.uint8)
    ctx = SliceContext(image, MosaicLayout(tile_aspect=4 / 3, columns=40))
    assert ctx.tile_size == (TILE_UNIT, TILE_UNIT * 3 / 4)
    assert (ctx.width, ctx.height) == (40 * TILE_UNIT, pytest.approx(40 * TILE_UNIT * 2 / 3))
    assert ctx.scale == pytest.approx(40 * TILE_UNIT / 3000)
    assert ctx.height / ctx.tile_size[1] == pytest.approx(ctx.layout.rows(3000, 2000))


def test_slicing_does_not_depend_on_the_unit_size():
    image = np.random.default_rng(0).integers(0, 256, (90, 120, 3), dtype=np.uint8)
    plan = SlicingPlan([Stage(QuadtreeSlicer(min_size=0.25)), Stage(JitterAdjust())])
    layout = MosaicLayout(tile_aspect=1.5, columns=6)
    small = plan.regions(SliceContext(image, layout, tile_width=10))
    large = plan.regions(SliceContext(image, layout, tile_width=250))
    assert len(small) == len(large)
    np.testing.assert_allclose(small.center * 25, large.center, rtol=1e-9)
    np.testing.assert_allclose(small.size * 25, large.size, rtol=1e-9)
    np.testing.assert_allclose(small.rotation, large.rotation)


def test_tile_grid_is_centered_with_overhang():
    ctx = tiled(100, 65, tile=10)  # 6.5 rows of tiles
    regions = GridSlicer().apply(ctx.canvas(), ctx)
    assert len(regions) == 10 * 7
    assert regions.bounds()[:, 1].min() == pytest.approx(-2.5)  # overhang split top/bottom
    assert regions.bounds()[:, 3].max() == pytest.approx(67.5)
    top_left = GridSlicer(anchor="top_left").apply(ctx.canvas(), ctx)
    assert top_left.bounds()[:, 1].min() == 0 and top_left.bounds()[:, 3].max() == 70


def test_tile_grid_follows_tile_shape_and_cell_size():
    ctx = tiled(120, 120, tile=30, aspect=1.5)  # 30 x 20 tiles
    regions = GridSlicer(cell_size=2).apply(ctx.canvas(), ctx)
    np.testing.assert_allclose(regions.size[0], [60, 40])
    assert len(regions) == 2 * 3


def test_context_works_in_mosaic_units():
    image = np.zeros((10, 10, 3), np.uint8)
    image[2, 7] = 255
    ctx = SliceContext(image, MosaicLayout(columns=4), tile_width=5)  # 20 x 20 canvas
    assert (ctx.width, ctx.height, ctx.scale) == (20, 20, 2)
    patch, per_px = ctx.patch(Region(15, 5, 2, 2), source="rgb")
    assert per_px == 0.5 and patch.shape == (1, 1, 3) and patch[0, 0, 0] == 255


def test_quadtree_minimum_is_in_tiles():
    image = (np.indices((64, 64)).sum(0) % 2 * 255).astype(np.uint8)[..., None].repeat(3, -1)
    ctx = SliceContext(image, MosaicLayout(columns=4), tile_width=16)  # busy everywhere
    regions = QuadtreeSlicer(min_size=0.5, max_depth=8).apply(ctx.canvas(), ctx)
    assert regions.size.min() == 8  # half a 16 px tile, never smaller


def test_coverage_estimates():
    full = RegionSet.covering(100, 50)
    assert coverage(full, 100, 50, spacing=5)[0] == 1.0
    left_half = RegionSet.from_rects(0, 0, 50, 50)
    assert coverage(left_half, 100, 50, spacing=5)[0] == pytest.approx(0.5, abs=0.02)
    rotated = RegionSet.from_arrays([[50, 25]], (40, 40), math.pi / 4)  # diamond, area 1600
    assert coverage(rotated, 100, 50, spacing=1)[0] == pytest.approx(
        (1600 - 2 * 0.5 * 3.28**2 * 4) / 5000, abs=0.02
    )  # the diamond pokes past the canvas top and bottom
    assert coverage(RegionSet(), 10, 10, spacing=1)[0] == 0.0


def test_summary_reports_density_coverage_and_sizes():
    ctx = SliceContext(np.zeros((60, 100, 3), np.uint8), MosaicLayout(columns=10), tile_width=20)
    grid = SlicingPlan.default().regions(ctx)  # 200 x 120 canvas, 20 px tiles
    summary = summarize(grid, ctx)
    assert summary.count == 60 and summary.density == pytest.approx(1.0)
    assert summary.coverage == 1.0
    assert summary.median == (20.0, 20.0)
    assert summary.source_px_per_tile == pytest.approx(10.0)
    gapped = summarize(grid.replace(size=grid.size * 0.8), ctx)
    assert gapped.coverage == pytest.approx(0.64, abs=0.03)  # 16 x 16 of each 20 x 20 tile


def test_tile_size_param_defaults():
    from skitter.core.slicing import TileSizeParam

    assert GridSlicer.cell_size.suffix == " × tile"
    assert isinstance(QuadtreeSlicer.min_size, TileSizeParam)


# Brick bonds and patterns


def cover_counts(regions, width, height, samples=3000):
    """How many regions cover each of some random points on the canvas."""
    points = np.random.default_rng(0).random((samples, 2)) * [width, height]
    return np.array([regions.contains(x, y).sum() for x, y in points])


def test_running_bond_shifts_alternate_rows_by_half_a_brick():
    ctx = tiled(100, 40, tile=20, aspect=2.0)  # 20 x 10 bricks, 4 courses
    regions = BondSlicer(anchor="top_left").apply(ctx.canvas(), ctx)
    np.testing.assert_allclose(regions.size, [[20, 10]] * len(regions))
    lefts = regions.bounds()[:, 0]
    rows = np.round(regions.center[:, 1] / 10 - 0.5).astype(int)
    np.testing.assert_allclose(lefts[rows % 2 == 0] % 20, 0, atol=1e-9)
    np.testing.assert_allclose(lefts[rows % 2 == 1] % 20, 10, atol=1e-9)
    assert lefts.min() == pytest.approx(-10)  # shifted rows overhang by half a brick
    assert np.all(cover_counts(regions, 100, 40) == 1)


def test_vertical_bond_shifts_columns():
    ctx = tiled(60, 90, tile=20, aspect=2.0)
    regions = BondSlicer(orientation="vertical", bond="third").apply(ctx.canvas(), ctx)
    np.testing.assert_allclose(regions.size[0], [20, 10])  # bricks keep the tile shape
    tops = {round(x, 6): set(np.round(regions.bounds()[regions.center[:, 0] == x, 1] % 10, 6))
            for x in regions.center[:, 0]}  # fmt: skip
    assert len(tops) == 3  # columns of 20 px
    assert {len(t) for t in tops.values()} == {1}
    assert len({next(iter(t)) for t in tops.values()}) == 3  # each column shifted differently
    assert np.all(cover_counts(regions, 60, 90) == 1)


def test_bond_with_zero_shift_is_the_grid_and_random_is_seeded():
    ctx = tiled(100, 65, tile=10)
    grid = GridSlicer().apply(ctx.canvas(), ctx)
    stack = BondSlicer(bond="custom", step=0.0).apply(ctx.canvas(), ctx)
    np.testing.assert_allclose(stack.center, grid.center)
    a, b, c = (BondSlicer(bond="random", seed=s).apply(ctx.canvas(), ctx) for s in (1, 1, 2))
    np.testing.assert_array_equal(a.center, b.center)
    assert not np.array_equal(a.center, c.center) or len(a) != len(c)
    assert np.all(cover_counts(a, 100, 65) == 1)


def test_random_bond_differs_between_parent_regions():
    ctx = tiled(400, 400, tile=10)
    parents = GridSlicer(cell_size=20).apply(ctx.canvas(), ctx)  # 2 x 2 equal parents
    bond = BondSlicer(bond="random")
    first, *others = (bond.subdivide(parent, ctx).center for parent in parents)
    assert not any(np.array_equal(first, other) for other in others)


@pytest.mark.parametrize("aspect", [2.0, 1.5, 1.0, 0.5])
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"orientation": "vertical"},
        {"angle": 45.0},
        {"anchor": "top_left"},
        {"pattern": "basketweave"},
        {"pattern": "basketweave", "weave_auto": False, "weave": 3, "angle": 30.0},
    ],
)
def test_patterns_cover_the_canvas_exactly_once(aspect, settings):
    ctx = tiled(200, 150, tile=20, aspect=aspect)
    regions = PatternSlicer(**settings).apply(ctx.canvas(), ctx)
    assert np.all(cover_counts(regions, 200, 150) == 1)
    assert np.all((regions.rotation > -math.pi / 2) & (regions.rotation <= math.pi / 2 + 1e-9))


def test_herringbone_uses_tile_shaped_bricks_in_two_directions():
    ctx = tiled(200, 150, tile=30, aspect=1.5)  # 30 x 20 tiles
    regions = PatternSlicer().apply(ctx.canvas(), ctx)
    np.testing.assert_allclose(regions.size, [[30, 20]] * len(regions))
    assert set(np.round(regions.rotation, 9)) == {0.0, round(math.pi / 2, 9)}
    portrait = tiled(200, 150, tile=20, aspect=2 / 3)  # 20 x 30 tiles
    np.testing.assert_allclose(PatternSlicer().apply(portrait.canvas(), portrait).size[0], [20, 30])


def test_basketweave_blocks_match_a_two_to_one_tile():
    ctx = tiled(160, 160, tile=40, aspect=2.0)
    regions = PatternSlicer(pattern="basketweave", anchor="top_left").apply(ctx.canvas(), ctx)
    np.testing.assert_allclose(regions.size, [[40, 20]] * len(regions))
    assert len(regions) == 16 * 2  # 16 square blocks of 2 bricks, no overhang
    assert PatternSlicer.weave.is_active(PatternSlicer(pattern="basketweave", weave_auto=False))
    assert not PatternSlicer.weave.is_active(PatternSlicer())


def test_custom_pattern_appears_in_choices_and_slices():
    from skitter.core.slicing import patterns as pattern_list
    from skitter.core.slicing import register_pattern
    from skitter.core.slicing.patterns import _patterns

    @register_pattern("test_stack", "Test stack")
    def stack(b):
        b.add(b.length / 2, b.thickness / 2)
        return b.unit((b.length, 0), (0, b.thickness), corner=(0, 0))

    try:
        assert ("test_stack", "Test stack") in PatternSlicer.pattern.choices
        assert any(p.id == "test_stack" for p in pattern_list())
        ctx = tiled(100, 65, tile=10)
        regions = PatternSlicer(pattern="test_stack", anchor="top_left").apply(ctx.canvas(), ctx)
        top_left = GridSlicer(anchor="top_left").apply(ctx.canvas(), ctx)
        np.testing.assert_allclose(regions.center, top_left.center)
    finally:
        del _patterns["test_stack"]


def test_pattern_inside_rotated_region_and_too_fine_patterns():
    ctx = tiled(200, 150, tile=20, aspect=2.0)
    parent = RegionSet.from_arrays([[100, 75]], [[80, 60]], rotation=0.3)
    regions = PatternSlicer().apply(parent, ctx)
    np.testing.assert_allclose(regions.rotation % (math.pi / 2), 0.3, atol=1e-9)
    huge = tiled(4000, 4000, tile=1)
    with pytest.raises(SlicingError):
        PatternSlicer(cell_size=0.05).apply(huge.canvas(), huge)

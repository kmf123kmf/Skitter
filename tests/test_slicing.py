import math

import numpy as np
import pytest

from skitter.core.slicing import (
    MAX_REGIONS,
    BoolParam,
    FloatParam,
    IntParam,
    Region,
    RegionSet,
    SliceContext,
    SlicingError,
    SlicingOperation,
    SlicingPlan,
    Stage,
    Subdivider,
    get_operation_type,
    operation_from_dict,
    operation_types,
    register_operation,
)
from skitter.core.slicing.base import _registry
from skitter.core.slicing.operations import GapAdjust, GridSlicer, JitterAdjust, QuadtreeSlicer


def blank(width=100, height=60):
    return SliceContext(np.zeros((height, width, 3), np.uint8))


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
    assert {"grid", "quadtree", "jitter", "gap"} <= set(ids)
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


def test_default_plan_is_square_grid():
    regions = SlicingPlan.default().regions(blank(240, 120))
    assert len(regions) == 24 * 12
    np.testing.assert_allclose(regions.size[0], [10, 10])


def test_empty_plan_yields_whole_image():
    regions = SlicingPlan().regions(blank(100, 60))
    assert len(regions) == 1 and regions[0].width == 100


def test_disabled_stage_passes_through():
    plan = SlicingPlan([Stage(GridSlicer(columns=4)), Stage(GridSlicer(columns=2), enabled=False)])
    assert len(plan.regions(blank(100, 100))) == 16


def test_evaluate_reuses_unchanged_prefix():
    calls = []

    class Counting(GridSlicer):
        id = "test-counting"

        def apply(self, regions, ctx):
            calls.append(self.columns)
            return super().apply(regions, ctx)

    plan = SlicingPlan([Stage(Counting(columns=2)), Stage(Counting(columns=3))])
    ctx = blank(60, 60)
    cache = plan.evaluate(ctx)
    plan.stages[1].operation.columns = 4
    cache = plan.evaluate(ctx, cache)
    assert calls == [2, 3, 4]  # the first stage was not re-run
    assert len(cache[-1].regions) == 4 * 16


def test_too_many_regions_raises():
    plan = SlicingPlan([Stage(GridSlicer(columns=1000, square_cells=False, rows=1000))])
    with pytest.raises(SlicingError, match=f"{MAX_REGIONS:,}"):
        plan.evaluate(blank())


def test_plan_serialization_roundtrip():
    plan = SlicingPlan(
        [
            Stage(GridSlicer(columns=8)),
            Stage(JitterAdjust(seed=7), enabled=False),
            Stage(GapAdjust()),
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


def test_grid_square_cells_follow_aspect():
    op = GridSlicer(columns=10)
    assert op.rows_for(100, 50) == 5
    op.square_cells = False
    op.rows = 3
    assert op.rows_for(100, 50) == 3


def test_grid_inside_rotated_region_stays_inside():
    parent = RegionSet.from_arrays([[50, 50]], (40, 20), math.pi / 6)
    cells = GridSlicer(columns=4, square_cells=False, rows=2).apply(parent, blank())
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


def test_gap_shrinks_sizes():
    cells = GapAdjust(gap=4).apply(RegionSet.grid(100, 100, 2, 2), blank())
    np.testing.assert_allclose(cells.size, [[46, 46]] * 4)


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


def test_jitter_and_gap_preserve_stacking():
    regions = RegionSet.from_arrays([[0, 0], [5, 5]], (2, 2), z=[1, 0])
    for op in (JitterAdjust(), GapAdjust()):
        assert op.apply(regions, blank()).z.tolist() == [1, 0]


def test_pile_covers_every_point_with_random_stacking():
    from skitter.core.slicing.operations import PileSlicer

    ctx = blank(600, 400)
    pile = PileSlicer(photo_size=120, rotation=30, overlap=0.0, portrait=0.5)
    regions = pile.apply(RegionSet.covering(600, 400), ctx)
    assert sorted(regions.z.tolist()) == list(range(len(regions)))
    assert np.abs(regions.rotation).max() <= math.radians(30) + 1e-9
    ys, xs = np.mgrid[0:400:5, 0:600:5]
    for x, y in zip(xs.ravel() + 0.5, ys.ravel() + 0.5, strict=True):
        assert regions.contains(x, y).any(), (x, y)


def test_pile_is_deterministic_per_seed():
    from skitter.core.slicing.operations import PileSlicer

    a = PileSlicer(seed=4).apply(RegionSet.covering(500, 500), blank(500, 500))
    b = PileSlicer(seed=4).apply(RegionSet.covering(500, 500), blank(500, 500))
    c = PileSlicer(seed=5).apply(RegionSet.covering(500, 500), blank(500, 500))
    np.testing.assert_array_equal(a.data, b.data)
    assert not np.array_equal(a.data, c.data)


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

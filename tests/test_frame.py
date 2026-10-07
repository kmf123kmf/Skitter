"""Pinned frames (frame.py) and the settings slicers share (common.py)."""

import math

import numpy as np

from skitter.core.slicing.common import anchor_param, rotation_param, seed_param
from skitter.core.slicing.frame import CENTER, TOP_LEFT, PinnedFrame, span
from skitter.core.slicing.operations import BondSlicer, GridSlicer, PatternSlicer


def test_span_centers_on_the_pin_or_starts_at_it():
    origin, count, first = span(-25.0, 25.0, 10.0, CENTER)
    assert (origin, count, first) == (-25.0, 5, -2)  # 5 tiles, the middle one numbered 0
    origin, count, first = span(-20.0, 20.0, 10.0, CENTER)
    assert (origin, count, first) == (-20.0, 4, -2)
    origin, count, first = span(-15.0, 32.0, 10.0, TOP_LEFT)
    assert (origin, count, first) == (-20.0, 6, -2)  # tile 0 starts at the pin
    assert span(0.0, 30.0, 10.0, TOP_LEFT) == (0.0, 3, 0)


def test_unturned_frame_is_the_region():
    frame = PinnedFrame(100.0, 60.0, CENTER)
    lo, hi = frame.bounds()
    np.testing.assert_allclose([lo, hi], [[-50, -30], [50, 30]])
    lo, hi = PinnedFrame(100.0, 60.0, TOP_LEFT).bounds()
    np.testing.assert_allclose([lo, hi], [[0, 0], [100, 60]])


def test_place_turns_about_the_pin_faces_up_and_keeps_overlapping_tiles_in_order():
    frame = PinnedFrame(100.0, 60.0, CENTER, math.radians(90.0))
    centers = [[10.0, 0.0], [0.0, 0.0], [500.0, 0.0], [-10.0, 0.0]]
    placed = frame.place(centers, (8.0, 4.0), rotations=[0.0, 0.0, 0.0, math.pi])
    # (10, 0) turned a quarter turn clockwise about (50, 30) lands at (50, 40); the far
    # tile misses the region; the half-turned one faces up again.
    np.testing.assert_allclose(placed.center, [[50, 40], [50, 30], [50, 20]], atol=1e-9)
    np.testing.assert_allclose(placed.rotation, math.pi / 2)


def test_shared_settings_are_separate_params_with_shared_wording():
    a, b = anchor_param("grid"), anchor_param("pattern")
    assert a is not b and a.choices == b.choices and "grid" in a.help
    assert rotation_param("pattern", "45° x").max == 90.0
    assert seed_param(when=lambda op: False).when is not None
    for cls in (GridSlicer, BondSlicer, PatternSlicer):
        assert cls.anchor.label == "Anchor" and cls.angle.label == "Rotation"
        assert cls.anchor is not GridSlicer.anchor or cls is GridSlicer
    # Settings forms keep their order: inherited ones first, then as declared.
    assert [p.name for p in GridSlicer.params()] == ["cell_size", "angle", "anchor"]
    assert [p.name for p in BondSlicer.params()] == [
        "orientation",
        "bond",
        "step",
        "seed",
        "cell_size",
        "angle",
        "anchor",
    ]


def test_tile_dims_scale_the_base_tile():
    from skitter.core.slicing import MosaicLayout, SliceContext

    ctx = SliceContext(np.zeros((20, 30, 3), np.uint8), MosaicLayout(columns=3, tile_aspect=1.5))
    w, h = ctx.tile_size
    assert ctx.tile_dims() == (w, h) and ctx.tile_dims(2.5) == (2.5 * w, 2.5 * h)

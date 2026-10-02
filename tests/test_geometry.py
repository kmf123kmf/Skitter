import pytest

from skitter.core.geometry import fit_aspect, move_rect, resize_rect, round_rect

BOUNDS = (0, 0, 100, 80)


def approx(rect):
    return pytest.approx(rect, abs=1e-9)


def test_move_is_clamped_to_bounds():
    assert move_rect((10, 10, 30, 30), 100, -100, BOUNDS) == (80, 0, 100, 20)


def test_corner_drag_keeps_opposite_corner():
    assert resize_rect((10, 10, 50, 50), "br", (60, 70), BOUNDS) == approx((10, 10, 60, 70))
    assert resize_rect((10, 10, 50, 50), "tl", (20, 5), BOUNDS) == approx((20, 5, 50, 50))


def test_drag_past_anchor_stops_at_min_size():
    assert resize_rect((10, 10, 50, 50), "r", (0, 0), BOUNDS) == approx((10, 10, 11, 50))


def test_drag_is_clamped_to_bounds():
    assert resize_rect((10, 10, 50, 50), "br", (500, 500), BOUNDS) == approx((10, 10, 100, 80))


def test_corner_drag_with_aspect_follows_larger_pull():
    rect = resize_rect((0, 0, 20, 20), "br", (40, 25), BOUNDS, aspect=1.0)
    assert rect == approx((0, 0, 40, 40))


def test_aspect_drag_shrinks_to_fit_bounds():
    rect = resize_rect((0, 0, 20, 20), "br", (100, 30), BOUNDS, aspect=1.0)
    assert rect == approx((0, 0, 80, 80))


def test_edge_drag_with_aspect_grows_about_center():
    rect = resize_rect((40, 30, 60, 50), "r", (70, 40), BOUNDS, aspect=1.0)
    assert rect == approx((40, 25, 70, 55))


def test_fit_aspect_centers_inside():
    assert fit_aspect((0, 0, 100, 50), 1.0) == approx((25, 0, 75, 50))
    assert fit_aspect((0, 0, 40, 80), 2.0) == approx((0, 30, 40, 50))


def test_round_rect_snaps_and_stays_valid():
    assert round_rect((-3.2, 10.6, 120, 10.7), BOUNDS) == (0, 11, 100, 12)

import math

import numpy as np
import pytest

from skitter.core.dive_path import CENTER_FIRST, STRAIGHT, FlowPath, View, flow_rate

VIEWPORT = (1200.0, 800.0)
CANVAS = (4000.0, 4000 * 2 / 3)  # a 3:2 mosaic, filling the 3:2 view


def leg(target, rotation=0.0, path=STRAIGHT, tile=100.0, seconds=6.0):
    zoom0 = VIEWPORT[0] / CANVAS[0]
    start = View((CANVAS[0] / 2, CANVAS[1] / 2), zoom0)
    end_zoom = VIEWPORT[0] / tile  # a 3:2 tile tile units wide fills the view
    return FlowPath(start, target, rotation, end_zoom, VIEWPORT, flow_rate(40, seconds), path)


def screen_offset(view: View, point) -> np.ndarray:
    """Where point shows on screen, from the middle (px), for an upright view."""
    return (np.asarray(point) - np.asarray(view.center)) * view.zoom


@pytest.mark.parametrize("path", [STRAIGHT, CENTER_FIRST])
@pytest.mark.parametrize("target", [(2000, 1333), (3700, 300), (150, 2500), (2600, 1500)])
def test_a_leg_zooms_evenly_onto_its_tile_and_never_past_the_start_view(path, target):
    flow = leg(target, path=path)
    start, end = flow.at(0.0), flow.at(1.0)
    assert start.center == pytest.approx(flow.start.center) and start.zoom == flow.z0
    assert end.center == pytest.approx(target) and end.zoom == pytest.approx(flow.z1)

    u = np.linspace(0, 1, 401)
    views = [flow.at(x) for x in u]
    logs = np.log([v.zoom for v in views])
    np.testing.assert_allclose(np.diff(logs), np.diff(logs)[0], rtol=1e-9)  # an even zoom

    # Never beyond the start view (the whole mosaic).
    half0 = np.asarray(VIEWPORT) / (2 * flow.z0)
    for view in views:
        half = np.asarray(VIEWPORT) / (2 * view.zoom)
        lo, hi = np.asarray(view.center) - half, np.asarray(view.center) + half
        assert np.all(lo >= flow.c0 - half0 - 1e-6) and np.all(hi <= flow.c0 + half0 + 1e-6)

    # The tile's offset on screen starts and ends at rest: no jolt where legs meet.
    offsets = np.array([screen_offset(v, target) for v in views])
    speed = np.linalg.norm(np.diff(offsets, axis=0), axis=1)
    assert speed[0] < 0.05 * speed.max() + 1e-3 and speed[-1] < 0.05 * speed.max() + 1e-3


def test_center_first_brings_the_tile_to_the_middle_sooner():
    target = (2600, 1500)  # near enough the middle for the view to center it early
    straight, first = leg(target), leg(target, path=CENTER_FIRST)
    at = 0.4
    off_straight = np.linalg.norm(screen_offset(straight.at(at), target))
    off_first = np.linalg.norm(screen_offset(first.at(at), target))
    assert off_first < 0.05 * off_straight


def test_a_leg_turns_with_its_tile_and_lasts_by_its_zoom():
    flow = leg((2000, 1333), rotation=math.radians(350))  # -10 degrees, the short way
    assert flow.at(0).rotation == 0.0
    assert flow.at(1).rotation == pytest.approx(math.radians(-10))
    assert flow.at(0.5).rotation == pytest.approx(math.radians(-5))
    # 40 columns of base tiles in 6 s: a base tile's leg takes 6 s, a 4x wider tile's less.
    assert leg((2000, 1333), tile=CANVAS[0] / 40).seconds == pytest.approx(6.0)
    assert leg((2000, 1333), tile=4 * CANVAS[0] / 40).seconds == pytest.approx(
        6.0 * math.log(10) / math.log(40)
    )
    assert flow.tile_share(1.0) == pytest.approx(1.0)
    assert flow.tile_share(0.0) == pytest.approx(flow.z0 / flow.z1)

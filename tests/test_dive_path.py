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
def test_a_leg_ends_on_its_tile_at_rest(path, target):
    flow = leg(target, path=path)
    start, end = flow.at(0.0), flow.at(1.0)
    assert start.center == pytest.approx(flow.start.center) and start.zoom == flow.z0
    assert end.center == pytest.approx(target) and end.zoom == pytest.approx(flow.z1)

    # The tile's offset on screen starts and ends at rest: no jolt where legs meet (finely
    # sampled: a tile near the middle pans briefly).
    views = [flow.at(x) for x in np.linspace(0, 1, 4001)]
    offsets = np.array([screen_offset(v, target) for v in views])
    speed = np.linalg.norm(np.diff(offsets, axis=0), axis=1)
    rest = 0.05 * speed.max() + 0.5  # px a step (a tile in the middle pans under 1 px)
    assert speed[0] < rest and speed[-1] < rest


@pytest.mark.parametrize("target", [(2000, 1333), (3700, 300), (150, 2500), (2600, 1500)])
def test_straight_in_zooms_evenly_and_never_past_the_start_view(target):
    flow = leg(target)
    views = [flow.at(x) for x in np.linspace(0, 1, 401)]
    logs = np.log([v.zoom for v in views])
    np.testing.assert_allclose(np.diff(logs), np.diff(logs)[0], rtol=1e-9)  # an even zoom
    half0 = np.asarray(VIEWPORT) / (2 * flow.z0)
    for view in views:  # never beyond the start view (the whole mosaic)
        half = np.asarray(VIEWPORT) / (2 * view.zoom)
        lo, hi = np.asarray(view.center) - half, np.asarray(view.center) + half
        assert np.all(lo >= flow.c0 - half0 - 1e-6) and np.all(hi <= flow.c0 + half0 + 1e-6)


@pytest.mark.parametrize("target", [(3700, 300), (2600, 1500)])
def test_center_first_pans_without_zooming_then_zooms_on_the_middle(target):
    flow = leg(target, path=CENTER_FIRST)
    u = np.linspace(0, 1, 801)
    views = [flow.at(x) for x in u]
    zooms = np.array([v.zoom for v in views])
    panning = u <= flow.pan_share
    assert 0 < flow.pan_share < 1
    np.testing.assert_allclose(zooms[panning], flow.z0)  # the pan: no zoom
    offsets = np.array([screen_offset(v, target) for v in views])
    np.testing.assert_allclose(offsets[~panning], 0.0, atol=1e-6)  # the zoom: on the middle
    logs = np.diff(np.log(zooms[~panning]))
    assert logs.min() >= 0 and logs[-1] < 0.05 * logs.max()  # coming to rest at the end
    # A tile farther off pans longer; the leg lasts its zoom's time plus the pan's.
    assert (
        leg((3700, 300), path=CENTER_FIRST).pan_share
        > leg((2600, 1500), path=CENTER_FIRST).pan_share
    )
    assert flow.seconds == pytest.approx(leg(target).seconds / (1 - flow.pan_share))


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
    for share in (0.1, 0.35, 1.0):
        assert flow.tile_share(flow.u_at_share(share)) == pytest.approx(share)
    assert flow.u_at_share(1e-6) == 0.0  # already larger at the start
    first = leg((3000, 600), path=CENTER_FIRST)  # its zoom eased, after the pan
    for share in (0.1, 0.35, 1.0):
        assert first.u_at_share(share) > first.pan_share
        assert first.tile_share(first.u_at_share(share)) == pytest.approx(share)

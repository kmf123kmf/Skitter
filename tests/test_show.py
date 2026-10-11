"""Show choreographies (core/animation/show.py) and lifting piles (lift.py)."""

import numpy as np
import pytest
from test_animation import grid_scene, make_scene, pile_scene

from skitter.core.animation.lift import Lift, moving_tiles, slope_tint
from skitter.core.animation.look import AnimationLook
from skitter.core.animation.show import (
    COMPASS,
    RippleChoreography,
    WaveChoreography,
    wave_pulse,
)
from skitter.core.matching.raster import visible_share
from skitter.core.slicing import MosaicLayout, RegionSet, SliceContext


def tiled_scene(columns=40, rows=30):
    """A grid of 10-unit base tiles (grid_scene squeezes its tiles into 60 x 40)."""
    ctx = SliceContext(np.zeros((rows, columns, 3), np.uint8), MosaicLayout(columns=columns),
                       tile_width=10)  # fmt: skip
    regions = RegionSet.grid(ctx.width, ctx.height, columns, rows)
    return make_scene(regions, np.arange(columns * rows) % 5, ctx=ctx)


# Visible share


def test_visible_share_of_uncovered_and_covered_tiles():
    assert np.allclose(grid_scene().visible_share, 1.0)  # a grid hides nothing
    # Three 10 x 10 squares, each shifted 5 to the right of the one under it.
    regions = RegionSet.from_rects([0, 5, 10], 0, 10, 10, z=[0, 1, 2])
    scene = make_scene(regions, [1, 2, 3])
    np.testing.assert_allclose(scene.visible_share, [0.5, 0.5, 1.0], atol=0.02)
    # One square exactly on another hides it completely.
    hidden = visible_share([[5, 5], [5, 5]], [[10, 10], [10, 10]], [0.0, 0.0])
    np.testing.assert_allclose(hidden, [0.0, 1.0])


def test_visible_share_of_rotated_tiles_on_a_pile():
    scene = pile_scene()
    share = scene.visible_share
    assert share[-1] == pytest.approx(1.0, abs=0.02)  # the top tile is uncovered
    assert (share >= 0).all() and (share <= 1).all()
    assert share.min() < 0.25  # a pile has tiles that are nearly hidden


# Which tiles move


@pytest.mark.parametrize("still_below", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_nothing_still_lies_on_a_moving_tile(still_below):
    scene = pile_scene()
    moving = moving_tiles(scene, still_below)
    lower, upper = scene.overlaps.T
    assert not (moving[lower] & ~moving[upper]).any()
    assert moving[scene.visible_share >= still_below].all()
    if still_below == 0:
        assert moving.all()


def test_mostly_hidden_tiles_stay_still():
    scene = pile_scene()
    moving = moving_tiles(scene, 0.25)
    # Still tiles are those showing little of themselves (others ride on moving tiles).
    assert (~moving).any()
    assert (scene.visible_share[~moving] < 0.25).all()


# Lift


def test_tiles_settle_with_the_tiles_under_them():
    # A chain: each square lies on the one before. Lifting only the bottom one lifts
    # every tile on it, the chain meeting at a height weighted by what the camera sees:
    # the uncovered top tile has the most say.
    regions = RegionSet.from_rects([0, 5, 10], 0, 10, 10, z=[0, 1, 2])
    scene = make_scene(regions, [1, 2, 3])
    lift = Lift(scene, still_below=0.0)
    surface = np.zeros((3, 5))
    surface[0] = 4.0
    frame = lift.frame(surface)
    weight = scene.visible_share + 0.02
    np.testing.assert_allclose(frame.heights(), [4.0 * weight[0] / weight.sum()] * 3, rtol=1e-3)
    # Lifting the top one alone leaves the others on the table.
    surface = np.zeros((3, 5))
    surface[2] = 4.0
    frame = lift.frame(surface)
    np.testing.assert_allclose(frame.heights(), [0.0, 0.0, 4.0])
    assert frame.at_rest().tolist() == [True, True, False]


def test_a_tile_rests_on_its_highest_point():
    scene = grid_scene()
    lift = Lift(scene)
    surface = np.zeros((len(scene), 5))
    surface[3, 2] = 2.5  # under one corner of tile 3
    np.testing.assert_allclose(lift.frame(surface).heights()[3], 2.5)


def test_nothing_lifted_is_exactly_the_finished_mosaic():
    scene = pile_scene()
    lift = Lift(scene)
    assert lift.frame(np.zeros((len(scene), 5))) is lift.final


def test_tiles_draw_low_to_high_then_in_stacking_order():
    scene = grid_scene()
    lift = Lift(scene)
    surface = np.zeros((len(scene), 5))
    surface[[2, 5, 9], :] = [[3.0], [1.0], [3.0]]
    order = lift.frame(surface).draw_order().tolist()
    resting = [i for i in range(len(scene)) if i not in (2, 5, 9)]
    assert order == [*resting, 5, 2, 9]


def test_slope_shading_lights_the_side_facing_the_light():
    fall = np.array([1.0, 0.0])  # light from the left: shadows fall right
    tint = slope_tint(np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0], [0.0, 0.5]]), fall)
    rising_to_the_right, falling, flat, across = tint
    assert rising_to_the_right[:3].tolist() == [1, 1, 1] and rising_to_the_right[3] > 0
    assert falling[:3].tolist() == [0, 0, 0] and falling[3] > 0
    # Alike on both sides (lightening a little gentler: white washes photos out).
    assert rising_to_the_right[3] == pytest.approx(0.7 * falling[3])
    assert flat[3] == 0 and across[3] == 0  # tilted across the light: unshaded
    steep = slope_tint(np.array([[10.0, 0.0], [-10.0, 0.0]]), fall)
    assert (steep[:, 3] <= 0.5).all()


def test_slope_shading_follows_the_light_direction():
    scene = grid_scene()
    surface = np.zeros((len(scene), 5))
    surface[0] = 1.0
    slope = np.zeros((len(scene), 2))
    slope[0] = [0.5, 0.0]  # rising toward the right
    from_left = Lift(scene, look=AnimationLook(light_direction=270.0)).frame(surface, slope)
    from_right = Lift(scene, look=AnimationLook(light_direction=90.0)).frame(surface, slope)
    assert from_left.tint[0, :3].tolist() == [1, 1, 1]  # faces the light: lighter
    assert from_right.tint[0, :3].tolist() == [0, 0, 0]  # faces away: darker
    off = Lift(scene, shading=False).frame(surface, slope)
    assert not off.tint.any()


# Ripple


def test_wave_pulse_rises_from_flat_and_settles_back():
    tau = np.linspace(-1.0, 4.0, 501)
    height, rate = wave_pulse(tau, period=1.0, count=3)
    assert (height >= 0).all() and height.max() == pytest.approx(1.0)
    assert not height[(tau < 0) | (tau >= 3)].any() and not rate[(tau < 0) | (tau >= 3)].any()
    peaks = [height[(tau >= k) & (tau < k + 1)].max() for k in range(3)]
    np.testing.assert_allclose(peaks, [1.0, 0.6, 0.36], atol=1e-3)
    # The rate is the height's derivative.
    np.testing.assert_allclose(np.gradient(height, tau)[1:-1], rate[1:-1], atol=0.05)


def test_ripple_spreads_from_the_drop_point():
    scene = tiled_scene(24, 16)
    ripple = RippleChoreography(center_x=0.0, center_y=0.0, wavelength=2.0, rings=1)
    timeline = ripple.timeline(scene)
    first = np.full(len(scene), np.inf)  # when each tile first lifts
    for t in np.linspace(0, timeline.duration, 401):
        lifted = timeline.frame(t).heights() > 0
        first[lifted & np.isinf(first)] = t
    assert np.isfinite(first).all()  # every tile lifts at some point
    distance = np.hypot(*scene.center.T)  # from the top-left corner
    assert np.corrcoef(first, distance)[0, 1] > 0.95


def test_ripple_lifts_a_ring_and_leaves_the_rest_on_the_table():
    scene = tiled_scene()
    timeline = RippleChoreography(rings=1, wavelength=3.0).timeline(scene)
    frame = timeline.frame(timeline.duration / 2)
    height = frame.heights()
    lifted = height > 0
    assert 0 < lifted.sum() < len(scene) / 2
    # Tiles off the ring are exactly at rest, untinted, with nothing changed.
    rest = ~lifted
    assert frame.at_rest()[rest].all() and not frame.tint[rest].any()
    np.testing.assert_array_equal(frame.center, scene.center)
    # The ring: lifted tiles all lie at about the same distance from the drop.
    distance = np.hypot(*(scene.center - np.asarray(scene.canvas) / 2).T)
    assert np.ptp(distance[lifted]) < 3.0 * scene.tile_size[0] + 2 * np.hypot(*scene.tile_size)


@pytest.mark.parametrize("cls", [RippleChoreography, WaveChoreography])
@pytest.mark.parametrize("still_below", [0.0, 25.0, 60.0])
def test_lifting_shows_on_a_pile_never_break_the_pile(cls, still_below):
    scene = pile_scene()
    timeline = cls(still_below=still_below).timeline(scene)
    moving = timeline.lift.moving
    lower, upper = scene.overlaps.T
    for t in np.linspace(0, timeline.duration, 121):
        frame = timeline.frame(t)
        height = frame.heights()
        assert not height[~moving].any()  # still tiles stay on the table
        assert (height[upper] >= height[lower]).all()  # nothing passes through what's on it
        rank = np.empty(len(scene), np.int64)
        rank[frame.draw_order()] = np.arange(len(scene))
        assert (rank[upper] > rank[lower]).all()  # and draws under it
        assert (frame.at_rest() == (height == 0)).all()


def test_ripple_height_is_kept_well_below_the_camera():
    scene = grid_scene()
    look = AnimationLook(camera_height=0.5)
    timeline = RippleChoreography(amplitude=10.0).timeline(scene, look)
    extent = max(np.ptp(scene.corners[..., 0]), np.ptp(scene.corners[..., 1]))
    top = max(timeline.frame(t).heights().max() for t in np.linspace(0, timeline.duration, 41))
    assert top <= 0.25 * 0.5 * extent + 1e-9


def test_ripple_without_shading_keeps_tints():
    scene = grid_scene()
    timeline = RippleChoreography(shading=False).timeline(scene)
    for t in np.linspace(0, timeline.duration, 21):
        assert not timeline.frame(t).tint.any()


def test_ripple_on_a_big_pile_keeps_what_the_camera_sees_near_the_surface():
    """Flat tiles can't follow the wave exactly where they overlap; settling keeps the
    visible ones close (carrying tiles up chains of overlaps put 30% of the visible
    area more than a quarter of the peak off; settling, about 8%)."""
    from skitter.core.slicing.operations import PileSlicer

    ctx = SliceContext(np.zeros((30, 40, 3), np.uint8), MosaicLayout(columns=40), tile_width=10)
    regions = PileSlicer(seed=3).apply(ctx.canvas(), ctx)
    scene = make_scene(regions, np.arange(len(regions)) % 7, ctx=ctx)
    timeline = RippleChoreography().timeline(scene)
    moving, visible = timeline.lift.moving, scene.visible_share
    peak = timeline.amplitude.max()
    off = seen = 0.0
    for t in np.linspace(0, timeline.duration, 41)[1:-1]:
        distance = timeline.front.distance
        pulse, _ = wave_pulse(t - distance / timeline.speed, timeline.period, timeline.count)
        own = np.where(moving, (timeline.amplitude * pulse).max(axis=1), 0.0)
        height = timeline.frame(t).heights()
        active = (height > 0) | (own > 0)
        off += visible[active & (np.abs(height - own) > 0.25 * peak)].sum()
        seen += visible[active].sum()
    assert off / seen < 0.12


# Wave


def first_lifted(timeline, n: int, steps: int = 401) -> np.ndarray:
    """(N,) when each tile first leaves the table (inf: never)."""
    first = np.full(n, np.inf)
    for t in np.linspace(0, timeline.duration, steps):
        lifted = timeline.frame(t).heights() > 0
        first[lifted & np.isinf(first)] = t
    return first


@pytest.mark.parametrize("direction", list(COMPASS))
def test_wave_travels_its_direction(direction):
    scene = tiled_scene(24, 16)
    wave = WaveChoreography(direction=direction, wavelength=2.0, count=1)
    first = first_lifted(wave.timeline(scene), len(scene))
    assert np.isfinite(first).all()  # it crosses every tile
    a = np.radians(COMPASS[direction])
    along = scene.center @ np.array([np.sin(a), -np.cos(a)])  # y runs down
    assert np.corrcoef(first, along)[0, 1] > 0.95


def test_wave_custom_angle_is_used_only_when_chosen():
    assert WaveChoreography(direction="custom", angle=30.0).degrees == 30.0
    assert WaveChoreography(direction="down", angle=30.0).degrees == 180.0
    angle = WaveChoreography.angle
    assert not angle.is_active(WaveChoreography()) and angle.is_active(
        WaveChoreography(direction="custom")
    )


def test_wave_lifts_a_band_across_the_picture():
    scene = tiled_scene()
    timeline = WaveChoreography(direction="right", count=1, wavelength=3.0).timeline(scene)
    height = timeline.frame(timeline.duration / 2).heights()
    lifted = height > 0
    assert 0 < lifted.sum() < len(scene) / 2
    x = scene.center[:, 0]
    assert np.ptp(x[lifted]) < 3.0 * scene.tile_size[0] + 2 * scene.tile_size[0]
    # A straight wave without fading lifts a whole column alike, top to bottom.
    column = np.isclose(x, x[lifted][0])
    assert lifted[column].all()
    np.testing.assert_allclose(height[column], height[column][0])


def test_wave_shading_follows_its_direction():
    # A wave travelling right, lit from the left: its trailing flank rises toward the
    # right, facing the light (lighter); its leading flank falls away from it (darker).
    scene = tiled_scene()
    look = AnimationLook(light_direction=270.0)  # from the left: shadows fall right
    timeline = WaveChoreography(direction="right", count=1).timeline(scene, look)
    frame = timeline.frame(timeline.duration / 2)
    lifted = frame.heights() > 0
    x = scene.center[lifted, 0]
    shaded = frame.tint[lifted, 3] > 0
    lighter = frame.tint[lifted, 0] == 1.0
    lead = x > np.median(x)
    assert lighter[~lead & shaded].mean() > 0.9 and lighter[lead & shaded].mean() < 0.1

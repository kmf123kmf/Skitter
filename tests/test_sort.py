"""The Sort build (core/animation/sort.py) and its algorithms and paths (sorting.py)."""

import math

import numpy as np
import pytest
from test_animation import pile_scene
from test_show import tiled_scene

from skitter.core.animation.sort import ALGORITHMS, PATHS, SortChoreography
from skitter.core.animation.sorting import hilbert_index, path_order, sort_states

NAMES = [a for a, _ in ALGORITHMS]

# Algorithms


@pytest.mark.parametrize("algorithm", NAMES)
@pytest.mark.parametrize("n", [1, 2, 3, 7, 64, 100, 513])
def test_every_algorithm_sorts_any_length(algorithm, n):
    a = np.random.default_rng(n).permutation(n)
    states, cleanup = sort_states(a, algorithm, 1, 10_000, {"columns": 9})
    assert not cleanup  # it sorted on its own (Bogosort, by its lucky last shuffle)
    np.testing.assert_array_equal(states[0], a)
    np.testing.assert_array_equal(states[-1], np.arange(n))
    for state in states:
        assert sorted(state.tolist()) == list(range(n))  # every photo somewhere, once
    assert not any(np.array_equal(s, t) for s, t in zip(states[:-1], states[1:], strict=True))


def test_rounds_beyond_the_limit_are_grouped():
    a = np.random.default_rng(0).permutation(300)
    full, _ = sort_states(a, "odd_even", 1, 10_000)
    shown, _ = sort_states(a, "odd_even", 1, 40)
    assert len(shown) - 1 <= 40 < len(full) - 1
    group = math.ceil((len(full) - 1) / 40)
    np.testing.assert_array_equal(
        shown[1:-1], full[group : len(full) - 1 : group][: len(shown) - 2]
    )
    np.testing.assert_array_equal(shown[-1], full[-1])


def test_each_algorithm_sorts_its_own_way():
    n = 200
    a = np.random.default_rng(1).permutation(n)

    def rounds(name, **info):
        return sort_states(a, name, 1, 10_000, info)[0]

    assert len(rounds("merge")) - 1 == math.ceil(math.log2(n))  # a pass per doubling
    assert len(rounds("radix")) - 1 == math.ceil(math.log(n, 4))  # a pass per base-4 digit
    for s, t in zip(*(lambda r: (r[:-1], r[1:]))(rounds("selection")), strict=True):
        assert (s != t).sum() == 2  # one swap at a time
    for s, t in zip(*(lambda r: (r[:-1], r[1:]))(rounds("odd_even")), strict=True):
        moved = np.flatnonzero(s != t)
        assert (np.abs(np.diff(moved.reshape(-1, 2), axis=1)) == 1).all()  # neighbors swap
    assert len(rounds("bitonic")) - 1 <= 2 * math.ceil(math.log2(n)) ** 2  # a network's stages
    # Radix: after its first pass every photo lies in its own quarter.
    first = rounds("radix")[1]
    digits = math.ceil(math.log(n, 4))
    np.testing.assert_array_equal(np.sort(first // 4 ** (digits - 1)), first // 4 ** (digits - 1))


# Paths


def test_hilbert_index_walks_the_curve():
    side = 2
    cells = [(0, 0), (0, 1), (1, 1), (1, 0)]  # the order a 2 x 2 Hilbert curve visits
    x, y = np.array(cells).T
    assert hilbert_index(side, x, y).tolist() == [0, 1, 2, 3]
    side = 8
    x, y = np.meshgrid(np.arange(side), np.arange(side))
    d = hilbert_index(side, x.ravel(), y.ravel())
    assert sorted(d.tolist()) == list(range(side * side))
    walk = np.argsort(d)
    steps = np.abs(np.diff(x.ravel()[walk])) + np.abs(np.diff(y.ravel()[walk]))
    assert (steps == 1).all()  # each step to a neighboring cell


@pytest.mark.parametrize("path", [p for p, _ in PATHS])
def test_paths_visit_every_spot_once(path):
    for scene in (tiled_scene(16, 12), pile_scene()):
        order = path_order(scene, path)
        assert sorted(order.tolist()) == list(range(len(scene)))


def test_paths_have_their_shapes():
    scene = tiled_scene(16, 12)
    tile = scene.tile_size[0]

    def steps(path):
        c = scene.center[path_order(scene, path)]
        return np.hypot(*np.diff(c, axis=0).T) / tile

    np.testing.assert_array_equal(path_order(scene, "reading"), np.arange(len(scene)))
    assert steps("snake").max() == pytest.approx(1.0)  # always to a neighbor
    # A Hilbert curve fills a square: on a wider picture it leaves and comes back, so a
    # few steps jump; the rest go to a neighbor.
    assert np.mean(np.isclose(steps("hilbert"), 1.0)) > 0.95
    spiral = path_order(scene, "spiral")
    distance = np.hypot(*(scene.center - np.asarray(scene.canvas) / 2).T)
    assert distance[spiral[0]] == distance.min() and distance[spiral[-1]] > 0.9 * distance.max()


# Sort


@pytest.mark.parametrize("algorithm", NAMES)
@pytest.mark.parametrize("make", [tiled_scene, pile_scene])
def test_sort_starts_scrambled_and_ends_on_the_mosaic(algorithm, make):
    scene = make()
    timeline = SortChoreography(algorithm=algorithm, path="hilbert").timeline(scene)
    start = timeline.frame(0.0)
    assert start.at_rest().all() and not start.tint.any()
    place = timeline.states[0]
    assert (place != np.arange(len(scene))).mean() > 0.8
    # Photos take on the shape of the spot they lie in.
    np.testing.assert_allclose(start.center, scene.center[place])
    np.testing.assert_allclose(start.size, scene.size[place])
    np.testing.assert_allclose(start.rotation, scene.rotation[place])
    np.testing.assert_array_equal(timeline.states[-1], np.arange(len(scene)))
    assert timeline.frame(timeline.duration) is timeline.final


def test_sort_shows_every_step_at_its_step_time():
    scene = tiled_scene(24, 16)
    a = SortChoreography(algorithm="odd_even", step_time=0.2).timeline(scene)
    values = np.argsort(a.states[0])  # reading order on a grid: positions are spots
    full, _ = sort_states(values, "odd_even", 1, 100_000)  # the same scramble
    assert a.rounds == len(full) - 1  # every step, none grouped
    assert a.interval == pytest.approx(0.2) and a.travel == pytest.approx(0.17)
    assert a.duration == pytest.approx((a.rounds - 1) * 0.2 + 0.17)  # as long as it needs
    slow = SortChoreography(algorithm="odd_even", step_time=0.4).timeline(scene)
    assert slow.duration == pytest.approx(2 * a.duration)  # paced, not squeezed
    # Mid-step photos move; after a step's moves, all lie still until the next.
    assert not a.frame(0.1).at_rest().all()
    assert a.frame(0.19).at_rest().all()
    quick = SortChoreography(algorithm="radix", step_time=0.2).timeline(scene)
    assert quick.duration < a.duration / 10  # Radix needs a handful of steps


def test_every_algorithm_glides_photos_straight():
    scene = tiled_scene(8, 6)
    for algo in ("pancake", "selection"):
        timeline = SortChoreography(algorithm=algo).timeline(scene)
        here, there = timeline.states[0], timeline.states[1]
        mid = timeline.frame(timeline.travel / 2)
        np.testing.assert_allclose(mid.center, (scene.center[here] + scene.center[there]) / 2)
        assert (mid.facings() == 1).all()


def test_sort_highlights_moving_photos():
    scene = tiled_scene(24, 16)
    timeline = SortChoreography(algorithm="radix").timeline(scene)
    frame = timeline.frame(timeline.travel / 2)
    moving = ~frame.at_rest()
    assert moving.any()
    assert np.allclose(frame.tint[moving, :3], [1.0, 0xB0 / 255, 0x20 / 255])
    assert (frame.tint[moving, 3] > 0.3).all() and not frame.tint[~moving].any()
    plain = SortChoreography(algorithm="radix", highlight=False).timeline(scene)
    assert not plain.frame(plain.travel / 2).tint.any()


def test_sort_with_nothing_scrambled_holds_still():
    scene = tiled_scene(12, 8)
    timeline = SortChoreography(scrambled=0.0).timeline(scene)
    assert timeline.rounds == 0 and timeline.duration > 0  # a still step
    for t in np.linspace(0, timeline.duration, 9):
        np.testing.assert_allclose(timeline.frame(t).center, scene.center)


def test_sort_resting_photos_stack_as_their_spots():
    scene = pile_scene()
    timeline = SortChoreography(algorithm="quick", highlight=False).timeline(scene)
    for t in np.linspace(0, timeline.duration, 31):
        frame = timeline.frame(t)
        order = frame.draw_order()
        rest = frame.at_rest()
        resting = order[rest[order]]
        r = min(int(t / timeline.interval), timeline.rounds - 1) if timeline.interval else 0
        spots = np.where(np.isin(np.arange(len(scene)), resting), timeline.states[r], -1)
        # At rest, drawn in the order of the spots they lie in (the pile's stacking order).
        place = np.array([np.argmin(np.hypot(*(scene.center - c).T)) for c in frame.center])
        assert (np.diff(place[resting]) > 0).all() or not len(resting), spots
        assert rest[order[: rest.sum()]].all()  # moving photos draw above


def test_the_slow_sorts_have_their_patterns():
    n = 120
    a = np.random.default_rng(2).permutation(n)

    def rounds(name):
        return sort_states(a, name, 1, 100_000)[0]

    for k, state in enumerate(rounds("bubble")[1:], 1):  # the end sorted, k more per pass
        np.testing.assert_array_equal(state[n - k :], np.arange(n - k, n))
    shaker = rounds("cocktail")
    for k, state in enumerate(shaker[1:], 1):  # both ends, taking turns
        top, bottom = (k + 1) // 2, k // 2
        np.testing.assert_array_equal(state[n - top :], np.arange(n - top, n))
        np.testing.assert_array_equal(state[:bottom], np.arange(bottom))
    for state in rounds("insertion"):  # a sorted start, growing
        start = np.flatnonzero(np.diff(state) < 0)
        assert len(start) == 0 or (np.diff(state[: start[0] + 1]) > 0).all()
    believe = rounds("cant_believe")
    for state in believe[1:-1]:  # sorted up to the largest photo, which rides at its front
        top = int(np.argmax(state))
        assert (np.diff(state[: top + 1]) > 0).all()
    for s, t in zip(rounds("pancake")[:-1], rounds("pancake")[1:], strict=True):
        k = np.flatnonzero(s != t).max() + 1  # every step flips a stretch from the start over
        np.testing.assert_array_equal(t[:k], s[:k][::-1])

"""Animation phases (core/animation/phases.py): build, show, clear in sequence."""

import numpy as np
import pytest
from test_animation import CONFIGS, assert_frame_equal, grid_scene, overlapping, pile_scene

from skitter.core.animation import (
    PHASES,
    ChainTimeline,
    Choreography,
    PhasedTimeline,
    PhaseHolds,
    TileFrame,
    Timeline,
    VideoClock,
    can_chain,
    choreography_types,
    get_choreography,
    plan_chain,
    plan_phases,
    register_choreography,
    unregister_choreography,
)
from skitter.core.animation.camera import Shot
from skitter.core.animation.keyframes import CameraKey, CameraTrack, KeyTime


class FadeTimeline(Timeline):
    """Every tile fades out over `duration` (or, as a show, out and back in)."""

    def __init__(self, scene, duration, back=False):
        self.final, self.duration, self.back = TileFrame.final(scene), duration, back

    def frame(self, t):
        u = min(max(t / self.duration, 0.0), 1.0)
        alpha = abs(1.0 - 2.0 * u) if self.back else 1.0 - u
        return self.final.replace(alpha=np.full(len(self.final), alpha))


@pytest.fixture
def fade():
    """A clear choreography and a show, registered for the test only."""

    @register_choreography
    class Fade(Choreography):
        id, name, phase, description = "test_fade", "Fade", "clear", "Tiles fade out."

        def timeline(self, scene, look=None):
            return FadeTimeline(scene, 1.0)

    @register_choreography
    class Blink(Choreography):
        id, name, phase, description = "test_blink", "Blink", "show", "Out and back in."

        def timeline(self, scene, look=None):
            return FadeTimeline(scene, 2.0, back=True)

    yield Fade(), Blink()
    unregister_choreography("test_fade")
    unregister_choreography("test_blink")


def holds(before=0.0, after=0.0) -> PhaseHolds:
    return PhaseHolds(hold_before=before, hold_after=after)


def same(a: TileFrame, b: TileFrame) -> bool:
    return all(np.allclose(getattr(a, f)(), getattr(b, f)()) for f in ("heights", "facings")) and \
        all(np.allclose(getattr(a, f), getattr(b, f))
            for f in ("center", "size", "rotation", "alpha"))  # fmt: skip


def test_every_choreography_keeps_its_phase_boundaries(fade):
    """Any build, show and clear combine: each meets the finished mosaic at the
    boundaries of its phase (a new choreography is checked here too)."""
    scene = grid_scene()
    final = TileFrame.final(scene)
    kinds = {phase: [cls.id for cls in choreography_types(phase)] for phase in PHASES}
    assert kinds["build"] == ["assemble", "deal", "sort", "unscramble"]
    for cls in choreography_types():
        timeline = cls().timeline(scene)
        if cls.phase in ("show", "clear"):
            assert same(timeline.frame(0.0), final), cls.id
        if cls.phase in ("build", "show"):
            assert same(timeline.frame(timeline.duration), final), cls.id


def test_phases_play_one_after_another_between_their_holds(fade):
    clear, blink = fade
    scene = grid_scene()
    build = get_choreography("assemble")(duration=3.0)
    timeline = plan_phases({"build": build, "show": blink, "clear": clear}, scene,
                           holds={"build": holds(1.0, 2.0), "show": holds(0.0, 0.5),
                                  "clear": holds(0.5, 1.0)})  # fmt: skip
    b, s, c = timeline.spans
    assert (b.phase, s.phase, c.phase) == PHASES
    assert (b.start, b.motion, b.end) == (0.0, (1.0, pytest.approx(4.0)), pytest.approx(6.0))
    assert s.length == 2.5 and s.motion == (pytest.approx(6.0), pytest.approx(8.0))
    assert c.length == 2.5 and timeline.duration == pytest.approx(11.0)
    final = TileFrame.final(scene)
    # Holds stand still: the build's first frame before its motion, the finished
    # mosaic after it; the show plays between its own holds.
    assert same(timeline.frame(0.0), timeline.frame(0.99))
    assert same(timeline.frame(4.0), final) and same(timeline.frame(5.9), final)
    assert timeline.frame(7.0).alpha == pytest.approx(0.0)  # the show, half way
    assert same(timeline.frame(8.2), final)  # the show's hold after
    assert timeline.frame(8.6).alpha == pytest.approx(1.0)  # the clear's hold before
    assert timeline.frame(9.5).alpha == pytest.approx(0.5)  # clearing
    assert timeline.frame(10.5).alpha.max() == 0  # gone, and held so
    assert [timeline.phase_at(t) for t in (1.0, b.end, 7.0, 9.5)] == [
        "build",
        "build",
        "show",
        "clear",
    ]  # a boundary belongs to the earlier

    # A phase left out is an empty span where it would be, holds and all; the rest
    # close up.
    no_show = plan_phases({"build": build, "clear": clear}, scene,
                          holds={"show": holds(5.0, 5.0)})  # fmt: skip
    assert no_show.span("show").start == no_show.span("show").end == pytest.approx(3.0)
    assert no_show.span("clear").start == pytest.approx(3.0)
    with pytest.raises(ValueError):
        PhasedTimeline({"show": blink.timeline(scene)})  # a build is required
    with pytest.raises(ValueError):
        plan_phases({"build": build, "show": clear}, scene)  # a clear isn't a show


def test_camera_keys_stay_with_their_phase(fade):
    clear, blink = fade
    scene = grid_scene()

    def clock(build_seconds, show_hold=None):
        build = get_choreography("assemble")(duration=build_seconds)
        show = blink if show_hold is not None else None
        timeline = plan_phases({"build": build, "show": show, "clear": clear}, scene,
                               holds={"show": holds(after=show_hold or 0.0)})  # fmt: skip
        return timeline.clock

    long = clock(4.0, 0.0)
    t = long.phase_span("show")[0] + 1.0  # halfway through the show
    when = KeyTime.at(t, long)
    assert when == KeyTime("show", pytest.approx(0.5))
    track = CameraTrack.of([CameraKey(when, Shot((0, 0)))])
    # A longer build moves the show and its key along; a longer show keeps the share.
    other = clock(8.0, 2.0)
    assert track.times(other)[0] == pytest.approx(other.phase_span("show")[0] + 2.0)
    # The show left out: the key sits where it would be (the build's end).
    gone = clock(4.0)
    assert track.times(gone)[0] == pytest.approx(gone.phase_span("build")[1])
    # A boundary belongs to the phase that ends there; inside a phase, to that phase.
    assert KeyTime.at(long.phase_span("build")[1], long).part == "build"
    assert KeyTime.at(long.phase_span("clear")[1] - 0.25, long).part == "clear"


def test_clocks_without_phases_are_all_build():
    clock = VideoClock(4.0)
    assert clock.phase_span("build") == (0.0, 4.0) and clock.phase_span("show") == (4.0, 4.0)
    assert clock.phase_span("clear") == (4.0, 4.0)


# Chains: lead-ins landing where the next build starts (phases.py, plan_chain).

LEADS = [(cls, kw) for cls, kw in CONFIGS if cls.fits_any_layout]
LED = [cls for cls in choreography_types("build") if cls.rests_on_table]
CHAIN_IDS = [f"{cls.id}{'-' + '-'.join(map(str, kw.values())) if kw else ''}" for cls, kw in LEADS]


def test_chains_join_builds_that_land_anywhere_to_builds_that_start_on_the_table():
    builds = {cls.id: cls for cls in choreography_types("build")}
    assert {cls.id for cls in LED} == {"sort", "unscramble"}
    assert {cls.id for cls, _ in LEADS} == {"assemble", "deal"}
    assert can_chain(builds["assemble"], builds["unscramble"])
    assert not can_chain(builds["unscramble"], builds["assemble"])  # Assemble starts empty
    assert not can_chain(builds["sort"], builds["unscramble"])  # Sort only ends at home
    assert not can_chain(builds["deal"], builds["deal"])
    with pytest.raises(ValueError):
        plan_chain([builds["unscramble"](), builds["assemble"]()], grid_scene())


@pytest.mark.parametrize("led", LED, ids=[cls.id for cls in LED])
@pytest.mark.parametrize(("cls", "settings"), LEADS, ids=CHAIN_IDS)
def test_a_lead_in_lands_exactly_where_the_next_build_starts(cls, settings, led):
    """On a pile: the lead-in ends on the next build's first frame (its stacking order
    too), keeping overlapping tiles in that layout's stacking order all the way (see
    test_every_choreography_keeps_overlapping_tiles_in_stacking_order)."""
    scene = pile_scene()
    chain = plan_chain([cls(**settings), led()], scene, pause=0.5)
    assert isinstance(chain, ChainTimeline)
    lead, main = chain.links
    start = main.frame(0.0)
    assert_frame_equal(lead.frame(lead.duration), start)
    np.testing.assert_array_equal(lead.frame(lead.duration).draw_order(), start.draw_order())
    # Played in turn, holding still between them; the chain ends on the finished mosaic.
    assert chain.duration == pytest.approx(lead.duration + 0.5 + main.duration)
    assert_frame_equal(chain.frame(lead.duration + 0.25), start)
    assert_frame_equal(chain.frame(chain.duration), TileFrame.final(scene))
    layout = scene.laid_out(start)
    lower, upper = layout.overlaps.T
    for t in np.linspace(0, lead.duration, 61):
        frame = lead.frame(t)
        rank = np.empty(len(scene), np.int64)
        rank[frame.draw_order()] = np.arange(len(scene))
        visible = (frame.alpha[lower] > 0) & (frame.alpha[upper] > 0)
        visible &= overlapping(frame.center[lower], frame.size[lower], frame.rotation[lower],
                               frame.center[upper], frame.size[upper],
                               frame.rotation[upper])  # fmt: skip
        nearer = frame.heights()[lower] > frame.heights()[upper]
        wrong = visible & (rank[upper] < rank[lower]) & ~nearer
        assert not wrong.any(), f"{wrong.sum()} lower tiles drawn on top at {t:.2f} s"


def test_a_laid_out_scene_stacks_as_its_layout():
    scene = pile_scene()
    start = get_choreography("unscramble")(duration=4.0).timeline(scene).frame(0.0)
    layout = scene.laid_out(start)
    np.testing.assert_allclose(layout.center, start.center)
    np.testing.assert_array_equal(TileFrame.final(layout).draw_order(), start.draw_order())
    lower, upper = layout.overlaps.T
    assert len(lower) and (layout.levels[lower] < layout.levels[upper]).all()
    assert scene.laid_out(TileFrame.final(scene)).stack is None  # its own order: as is


def test_chains_play_in_their_phase_between_its_holds():
    scene = grid_scene()
    lead = get_choreography("assemble")(duration=2.0)
    main = get_choreography("unscramble")(duration=3.0)
    build_holds = PhaseHolds(hold_before=1.0, hold_between=0.5, hold_after=1.0)
    timeline = plan_phases({"build": [lead, main]}, scene, holds={"build": build_holds})
    part = timeline.parts["build"]
    assert isinstance(part, ChainTimeline) and part.starts[1] == pytest.approx(
        part.links[0].duration + 0.5)  # fmt: skip
    assert timeline.span("build").length == pytest.approx(1.0 + part.duration + 1.0)
    assert_frame_equal(timeline.frame(timeline.duration), TileFrame.final(scene))
    # One choreography alone plans as before (no chain).
    alone = plan_phases({"build": [main]}, scene).parts["build"]
    assert not isinstance(alone, ChainTimeline)

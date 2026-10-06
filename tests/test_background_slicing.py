"""Slicing in the background: progress, cancellation, and only the latest run counts."""

import threading
import time

import numpy as np
import pytest

from skitter.core.slicing import (
    MosaicLayout,
    Progress,
    SliceContext,
    SlicingCancelled,
    SlicingPlan,
    Stage,
)
from skitter.core.slicing.operations import GridSlicer, SplitSlicer

# Progress and plans (core)


def test_progress_maps_parts_reports_in_steps_and_cancels():
    heard, stop = [], threading.Event()
    progress = Progress(heard.append, stop.is_set)
    half = progress.part(0.5, 1.0)
    for k in range(101):
        half(k / 100)
    assert heard[0] == pytest.approx(0.5) and heard[-1] == pytest.approx(1.0)
    assert len(heard) <= 52 and heard == sorted(heard)  # every 1% at most
    stop.set()
    with pytest.raises(SlicingCancelled):
        half(1.0)


def test_plan_reports_each_stage_and_can_be_cancelled():
    ctx = SliceContext(np.zeros((40, 60, 3), np.uint8), MosaicLayout(columns=6))
    plan = SlicingPlan([Stage(GridSlicer()), Stage(SplitSlicer(across=2))])
    heard = []
    plan.evaluate(ctx, progress=lambda message, f: heard.append((message, f)))
    assert heard[0] == ("Grid (stage 1 of 2)", 0.0)
    assert heard[-1] == ("Split (stage 2 of 2)", 1.0)
    assert [f for _, f in heard] == sorted(f for _, f in heard)

    calls = []

    def cancelled() -> bool:  # once the second stage is under way
        calls.append(1)
        return any("stage 2" in message for message, _ in heard2)

    heard2 = []
    with pytest.raises(SlicingCancelled):
        plan.evaluate(ctx, progress=lambda m, f: heard2.append((m, f)), cancelled=cancelled)
    assert calls


def test_plan_copy_is_independent():
    plan = SlicingPlan([Stage(GridSlicer())])
    copy = plan.copy()
    copy.stages[0].operation.cell_size = 2.0
    copy.stages.append(Stage(SplitSlicer()))
    assert plan.stages[0].operation.cell_size == 1.0 and len(plan.stages) == 1


# The session (worker thread)


@pytest.fixture
def session(qapp, background_slicing):
    from skitter.ui.session import Session

    s = Session()
    s.set_layout(MosaicLayout(columns=6))  # 6 x 4 tiles
    s.set_source("photo.png", np.random.default_rng(0).integers(0, 256, (60, 90, 3), np.uint8))
    yield s
    s.shutdown()


class Slow(GridSlicer):
    """A grid that takes a while per region, reporting as it goes (not registered;
    its own id, so a plan doesn't take it for a Grid stage it has cached)."""

    id = "slow-grid"
    delay = 0.005

    def subdivide(self, region, ctx, progress=None):
        for k in range(5):
            if progress is not None:
                progress(k / 5)
            time.sleep(self.delay)
        return super().subdivide(region, ctx)


def test_commit_slices_in_the_background(session):
    changed, started = [], []
    session.slicing_changed.connect(lambda: changed.append(session.project.regions))
    session.slicing_started.connect(lambda: started.append(True))
    session.commit_source()
    assert started and session.slicing_running
    assert session.project.regions is None  # the new image's regions aren't in yet
    session.wait_for_slicing()
    assert not session.slicing_running
    assert changed[-1] is session.project.regions and len(session.project.regions) > 0
    assert session.slicing_summary is not None and session.slicing_error is None


def test_only_the_latest_run_counts(session):
    session.commit_source()
    session.wait_for_slicing()
    grid = session.project.slicing_plan.stages[0].operation
    installed = []
    session.slicing_changed.connect(lambda: installed.append(len(session.project.regions)))
    for size in (2.0, 3.0, 0.5):  # three quick edits: each run replaces the last
        grid.cell_size = size
        session.slicing_edited()
    session.wait_for_slicing()
    expected = len(SlicingPlan([Stage(GridSlicer(cell_size=0.5))]).regions(session.slice_context))
    assert installed == [expected]


def test_regions_stay_shown_while_a_plan_edit_runs(session, monkeypatch):
    session.commit_source()
    session.wait_for_slicing()
    before = session.project.regions
    session.project.slicing_plan.stages[0] = Stage(Slow())
    session.slicing_edited()
    assert session.slicing_running
    assert session.project.regions is before  # still shown, but out of date:
    assert not session.can_match
    session.wait_for_slicing()
    assert session.project.regions is not before


def test_editing_the_plan_during_a_run_doesnt_reach_it(session):
    session.commit_source()
    session.wait_for_slicing()
    session.project.slicing_plan.stages[0] = Stage(Slow())
    session.slicing_edited()
    session.project.slicing_plan.stages[0].operation.cell_size = 3.0  # not re-sliced (yet)
    session.wait_for_slicing()
    expected = len(SlicingPlan([Stage(GridSlicer())]).regions(session.slice_context))
    assert len(session.project.regions) == expected


def test_a_replaced_run_stops_early(session, monkeypatch):
    session.commit_source()
    session.wait_for_slicing()
    monkeypatch.setattr(Slow, "delay", 0.2)
    plan = session.project.slicing_plan
    plan.stages[0] = Stage(Slow())  # 24 regions x 1 s if it ran to the end
    session.slicing_edited()
    first = session._slicing_job
    time.sleep(0.1)
    plan.stages[0] = Stage(GridSlicer(cell_size=2.0))
    session.slicing_edited()
    start = time.perf_counter()
    first.wait(5)
    assert time.perf_counter() - start < 1.0  # it noticed the cancel between steps
    session.wait_for_slicing()
    assert session.slicing_error is None


def test_failures_in_the_background_are_reported(session, monkeypatch):
    session.commit_source()
    session.wait_for_slicing()

    def fail(self, regions, ctx, progress):
        raise RuntimeError("boom")

    monkeypatch.setattr(GridSlicer, "apply", fail)
    session.project.slicing_plan.stages[0].operation.cell_size = 2.0
    session.slicing_edited()
    session.wait_for_slicing()
    assert session.slicing_error == "Slicing failed: RuntimeError: boom"
    assert session.project.regions is None


def test_the_slicing_step_waits_for_its_run(qapp, background_slicing, tmp_path):
    from skitter.core.imaging import save_image
    from skitter.ui.main_window import MainWindow
    from skitter.ui.steps.slicing import SlicingStep
    from skitter.ui.steps.source import SourceStep

    path = tmp_path / "photo.png"
    save_image(np.random.default_rng(1).integers(0, 256, (30, 40, 3), np.uint8), path)
    window = MainWindow()
    try:
        window.step(SourceStep).load_file(path)
        window.next_button.click()
        slicing = window.step(SlicingStep)
        assert window.tabs.currentWidget() is slicing
        assert not slicing.can_advance() and not slicing._running.isHidden()
        window.session.wait_for_slicing()
        assert slicing.can_advance() and slicing._running.isHidden()
    finally:
        window.session.shutdown()
        window.deleteLater()


def test_a_new_layout_clears_the_old_regions_at_once(session):
    session.commit_source()
    session.wait_for_slicing()
    shown = []
    session.slicing_changed.connect(lambda: shown.append(session.project.regions))
    session.set_layout(MosaicLayout(columns=12))
    assert shown == [None] and session.slicing_running  # nothing out of shape on screen
    session.wait_for_slicing()
    assert len(shown) == 2 and len(shown[-1]) == 12 * 8

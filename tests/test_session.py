import numpy as np
import pytest

from skitter.core.edits import Crop, FlipHorizontal, Rotate90


@pytest.fixture
def session(qapp):
    from skitter.ui.session import Session

    s = Session()
    s.set_source("photo.png", np.arange(24, dtype=np.uint8).reshape(2, 4, 3))
    return s


def record(signal):
    calls = []
    signal.connect(lambda *args: calls.append(args))
    return calls


def test_apply_undo_redo(session):
    edited = record(session.source_edited)
    session.apply_edit(Rotate90(1))
    session.apply_edit(Crop(0, 0, 1, 3))
    assert session.project.source_image.shape == (3, 1, 3)

    session.undo()
    assert session.project.source_image.shape == (4, 2, 3)
    assert session.next_redo == Crop(0, 0, 1, 3)

    session.redo()
    assert session.project.source_image.shape == (3, 1, 3)
    assert edited == [
        (Rotate90(1), False),
        (Crop(0, 0, 1, 3), False),
        (Crop(0, 0, 1, 3), True),
        (Crop(0, 0, 1, 3), False),
    ]


def test_new_edit_clears_redo(session):
    session.apply_edit(FlipHorizontal())
    session.undo()
    session.apply_edit(Rotate90(1))
    assert not session.can_redo


def test_revert_then_redo_replays_in_order(session):
    original = session.project.source_original
    session.apply_edit(Rotate90(1))
    session.apply_edit(FlipHorizontal())
    session.revert_edits()
    assert session.project.source_image is original or np.array_equal(
        session.project.source_image, original
    )
    assert session.next_redo == Rotate90(1)
    session.redo()
    session.redo()
    assert session.project.source_edits == [Rotate90(1), FlipHorizontal()]


def test_new_source_resets_edits(session):
    session.apply_edit(FlipHorizontal())
    session.set_source("other.png", np.zeros((5, 5, 3), np.uint8))
    assert session.project.source_edits == []
    assert not session.can_undo and not session.can_redo


def test_commit_snapshots_read_only_copy(session):
    committed = record(session.source_committed)
    assert not session.source_is_committed
    assert session.commit_source()
    final = session.project.source_final
    assert np.array_equal(final, session.project.source_image)
    assert final is not session.project.source_image
    assert not final.flags.writeable
    assert session.source_is_committed and len(committed) == 1


def test_commit_unchanged_is_noop(session):
    session.commit_source()
    committed = record(session.source_committed)
    assert not session.commit_source()
    assert committed == []


def test_edits_invalidate_commit_until_undone(session):
    session.commit_source()
    session.apply_edit(FlipHorizontal())
    assert not session.source_is_committed
    session.undo()
    assert session.source_is_committed  # back to exactly what was committed


def test_reloading_source_invalidates_commit(session):
    session.commit_source()
    session.set_source("photo.png", session.project.source_original)
    assert not session.source_is_committed

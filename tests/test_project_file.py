"""Project files (core/project_file.py), saved mosaics (matching/saved.py) and the
File menu round trip in the app."""

import json
import zipfile

import numpy as np
import pytest
from test_manual_picks import library, matched  # noqa: F401 (fixtures)
from test_matching_steps import (  # noqa: F401 (fixtures)
    build_library,
    matching_step,
    photos,
    sliced,
    window,
)

from skitter.core.animation.camera import Shot
from skitter.core.animation.keyframes import CameraKey, CameraTrack, KeyTime
from skitter.core.edits import Crop, FlipHorizontal, Rotate90, apply_edits
from skitter.core.matching.saved import SavedMosaic, restore, saved_mosaic
from skitter.core.project import Project
from skitter.core.project_file import (
    ProjectFileError,
    document,
    load_project,
    save_project,
)
from skitter.core.slicing import MosaicLayout, SliceContext, SlicingPlan, Stage
from skitter.core.slicing.operations import GridSlicer, QuadtreeSlicer

# The file format


def sample_project(alpha=True) -> Project:
    """A committed project with every kind of setting changed from its default."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (60, 80, 4), dtype=np.uint8)
    image[..., 3] = 255
    if alpha:
        image[:10, :10, 3] = 0  # a transparent corner
    project = Project()
    project.source_path = None
    project.source_original = image
    project.source_edits = [Crop(2, 3, 70, 50), Rotate90(1), FlipHorizontal()]
    project.source_image = apply_edits(image, project.source_edits)
    project.source_final = project.source_image
    project.layout = MosaicLayout(tile_aspect=1.5, columns=7)
    stages = [Stage(GridSlicer(angle=30.0)), Stage(QuadtreeSlicer(splits=1), enabled=False)]
    project.slicing_plan = SlicingPlan(stages)
    ctx = SliceContext(project.source_final, project.layout)
    project.regions = project.slicing_plan.regions(ctx)
    project.match_settings.update(max_uses=7, tint="custom", tint_strength=0.4)
    project.export_settings.update(format="jpeg", tile_px=55)
    project.animation_look.update(background="#123456")
    project.video_settings.update(frame_rate="24")
    project.choreography_id = "deal"
    project.choreographies["deal"].update(decks=3)
    project.camera_moves["follow"].update(count=12, room=2.0)
    project.camera_track = CameraTrack((
        CameraKey(KeyTime("lead", 0.5), Shot((10.0, 20.0), 3.0, 0.25)),
        CameraKey(KeyTime("body", 0.4), Shot((30.0, 25.0), 1.5), stop=False, motion="linear"),
    ), stretch=False)  # fmt: skip
    return project


def test_a_project_survives_a_round_trip(tmp_path):
    project = sample_project()
    path = tmp_path / "test.skitter"
    save_project(path, project, committed=True, mosaic=None)
    loaded = load_project(path)
    assert loaded.committed and loaded.mosaic is None and loaded.problems == []
    again = loaded.project
    assert document(again, True) == document(project, True)  # every setting and edit
    np.testing.assert_array_equal(again.source_original, project.source_original)
    np.testing.assert_array_equal(again.source_image, project.source_image)
    np.testing.assert_array_equal(again.source_final, project.source_final)
    assert not again.source_final.flags.writeable
    for name in ("center", "size", "rotation", "z"):
        np.testing.assert_array_equal(getattr(again.regions, name), getattr(project.regions, name))
    assert again.choreography.id == "deal" and again.choreographies["deal"].decks == 3
    assert again.camera_move.id == "follow" and again.camera_move.count == 12
    assert again.camera_track == project.camera_track
    assert not again.slicing_plan.stages[1].enabled
    assert not (tmp_path / "test.skitter.part").exists()


def test_view_state_is_kept_as_is_and_optional(tmp_path):
    project = sample_project()
    save_project(tmp_path / "a.skitter", project, True, None, view={"step": "animate", "x": [1]})
    assert load_project(tmp_path / "a.skitter").view == {"step": "animate", "x": [1]}
    save_project(tmp_path / "b.skitter", project, True, None)
    assert load_project(tmp_path / "b.skitter").view == {}
    rewrite(tmp_path / "b.skitter", lambda doc: doc.update(view="not a dict"))
    assert load_project(tmp_path / "b.skitter").view == {}


def test_uncommitted_projects_keep_settings_but_not_regions(tmp_path):
    project = sample_project(alpha=False)
    save_project(tmp_path / "p.skitter", project, committed=False, mosaic=None)
    loaded = load_project(tmp_path / "p.skitter")
    assert not loaded.committed
    assert loaded.project.regions is None and loaded.project.source_final is None
    with zipfile.ZipFile(tmp_path / "p.skitter") as archive:
        assert "regions.npz" not in archive.namelist()
        png = archive.read("source.png")
    assert png[25] == 2  # PNG color type 2: RGB, as the image is opaque
    assert loaded.project.source_original.shape[-1] == 4  # but loaded as RGBA, like any source


def rewrite(path, change):
    """Edit project.json inside a project file."""
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    doc = json.loads(files["project.json"])
    change(doc)
    files["project.json"] = json.dumps(doc).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def test_loading_skips_what_this_version_does_not_know(tmp_path):
    path = tmp_path / "p.skitter"
    save_project(path, sample_project(), committed=True, mosaic=None)

    def change(doc):
        doc["match_settings"]["retired_setting"] = 1
        doc["match_settings"]["max_uses"] = -5  # out of range
        doc["slicing_plan"]["stages"].insert(0, {"type": "hexagons", "params": {}})
        doc["choreographies"]["deal"]["decks"] = "many"
        doc["layout"]["columns"] = 0

    rewrite(path, change)
    loaded = load_project(path)
    project, problems = loaded.project, " | ".join(loaded.problems)
    assert "retired_setting" in problems and "max_uses" in problems and "hexagons" in problems
    assert "decks" in problems and "layout" in problems.lower()
    assert project.match_settings.max_uses == 3 and project.match_settings.tint_strength == 0.4
    assert [s.operation.id for s in project.slicing_plan.stages] == ["grid", "quadtree"]
    assert project.layout == MosaicLayout()


def test_files_that_are_not_projects_are_refused(tmp_path):
    with pytest.raises(ProjectFileError):
        load_project(tmp_path / "missing.skitter")
    (tmp_path / "junk.skitter").write_bytes(b"not a zip")
    with pytest.raises(ProjectFileError):
        load_project(tmp_path / "junk.skitter")
    path = tmp_path / "newer.skitter"
    save_project(path, sample_project(), committed=True, mosaic=None)
    rewrite(path, lambda doc: doc.update(format=99))
    with pytest.raises(ProjectFileError, match="newer version"):
        load_project(path)
    with pytest.raises(ValueError):
        save_project(tmp_path / "empty.skitter", Project(), committed=False, mosaic=None)


# Saved mosaics


def test_a_saved_mosaic_restores_the_same_tiles(matched, library):  # noqa: F811
    ctx, result = matched
    saved = saved_mosaic(result, library)
    assert len(saved.paths) == len(np.unique(result.tile[result.tile >= 0]))
    again = SavedMosaic.from_parts(saved.arrays(), json.loads(json.dumps(saved.meta())))
    restored, missing = restore(again, library, result.regions, ctx)
    assert missing == 0 and restored.candidates is None
    np.testing.assert_array_equal(restored.tile, result.tile)
    np.testing.assert_array_equal(restored.mirrored, result.mirrored)
    np.testing.assert_allclose(restored.rect, result.rect)
    np.testing.assert_allclose(restored.tile_mean, result.tile_mean, atol=1e-6)
    assert restored.quality.score == pytest.approx(result.quality.score, abs=1e-4)
    assert restored.settings.key() == result.settings.key()
    assert restored.stats["unique_tiles"] == result.stats["unique_tiles"]


def test_photos_gone_from_the_library_leave_their_regions_empty(matched, library):  # noqa: F811
    ctx, result = matched
    saved = saved_mosaic(result, library)
    gone = saved.paths[0]
    moved = SavedMosaic(**{**saved.__dict__, "paths": ["elsewhere.png", *saved.paths[1:]]})
    restored, missing = restore(moved, library, result.regions, ctx)
    lost = saved.photo == 0
    assert missing == lost.sum() > 0 and (restored.tile[lost] == -1).all()
    assert (restored.tile[~lost] == result.tile[~lost]).all() and gone
    with pytest.raises(ValueError):
        restore(saved, library, result.regions[:5], ctx)


# In the app


def test_save_new_and_open_restore_the_whole_session(sliced, photos, tmp_path):  # noqa: F811
    from skitter.ui.steps.animate import AnimateStep
    from skitter.ui.steps.matching import MatchingStep
    from skitter.ui.steps.source import SourceStep

    app = sliced
    session = app.session
    app.next_button.click()  # to Tiles
    build_library(app, photos)
    app.next_button.click()  # to Matching
    session.project.match_settings.update(refine_seconds=0.2, adaptive_rounds=0)
    matching_step(app).run_matching()
    session.wait_for_job()
    region = int(np.flatnonzero(session.project.matches.tile >= 0)[0])
    session.pick_tile(region, int(session.project.matches.candidates.ranked(region)[0][1]))
    assert session.manual_picks == 1 and app.isWindowModified()
    tiles = session.project.matches.tile.copy()
    session.project.choreographies["deal"].update(decks=4)

    path = tmp_path / "work.skitter"
    assert app._save_to(path)
    assert path.exists() and not session.modified and not app.isWindowModified()
    assert app.windowTitle() == "Skitter — work.skitter[*]"
    # Save is there only for unsaved changes (Save As always), even ones nothing redraws for.
    assert not app.save_action.isEnabled() and app.save_as_action.isEnabled()
    animate = app.step(AnimateStep)
    animate.camera_form.editor("count").widget.setValue(11)
    assert app.save_action.isEnabled() and app.isWindowModified()
    assert app._save_to(path) and not app.save_action.isEnabled()

    assert app.new_project()  # nothing unsaved: no question asked
    assert not session.project.has_source and session.project.matches is None
    assert app.tabs.currentWidget() is app.step(SourceStep)
    assert not app.tabs.isTabEnabled(1)

    from PySide6.QtWidgets import QProgressDialog

    shown = []  # a bouncing bar shows while the project opens, and is gone after
    session.matching_changed.connect(
        lambda: shown.extend(
            (d.isVisible(), d.maximum(), d.isModal()) for d in app.findChildren(QProgressDialog)
        )
    )
    assert app.open_project(path)
    assert (True, 0, True) in shown
    assert not [d for d in app.findChildren(QProgressDialog) if d.isVisible()]
    project = session.project
    assert app.tabs.currentWidget() is app.step(MatchingStep)
    assert session.textures is not None and not session.textures.loading
    assert session.source_is_committed and session.mosaic_is_valid and session.matching_is_current
    np.testing.assert_array_equal(project.matches.tile, tiles)
    assert session.manual_picks == 1 and session.missing_tiles == 0
    assert project.choreographies["deal"].decks == 4
    step = matching_step(app)
    assert step.form._target is project.match_settings  # forms follow the new project
    assert not step.edit.can_edit() and "run matching again" in step.edit.picker.hint.text()
    assert not session.modified and app.windowTitle() == "Skitter — work.skitter[*]"

    # Projects reopen on the tab they were saved on (by step id); switching tabs isn't
    # an unsaved change. An id this version doesn't have falls back to the furthest tab.
    app.tabs.setCurrentWidget(app.step(AnimateStep))
    assert not session.modified
    assert app._save_to(path) and app.open_project(path)
    assert app.tabs.currentWidget() is app.step(AnimateStep)
    session.save_project(path, {"step": "retired-tab"})
    assert app.open_project(path) and app.tabs.currentWidget() is app.step(MatchingStep)

    # Matching again keeps the pick and brings back the candidates for picking by hand.
    session.start_matching(keep_picks=True)
    session.wait_for_job()
    assert session.manual_picks == 1 and session.project.matches.tile[region] == tiles[region]
    assert session.project.matches.candidates is not None and session.modified


def test_steps_have_stable_unique_ids():
    from skitter.ui.steps import STEPS

    ids = [step.id for step in STEPS]
    assert all(ids) and len(set(ids)) == len(ids)
    assert ids == ["source", "slicing", "tiles", "matching", "animate"]  # saved in projects


def test_projects_that_ran_a_camera_move_get_its_keys(sliced, photos, tmp_path):  # noqa: F811
    """Before moves wrote keys, a project's camera ran its chosen move live: opening one
    writes that move's keys, so it plays as it did."""
    app = sliced
    session = app.session
    app.next_button.click()  # to Tiles
    build_library(app, photos)
    session.project.match_settings.update(refine_seconds=0.2, adaptive_rounds=0)
    matching_step(app).run_matching()
    session.wait_for_job()
    session.project.camera_moves["follow"].update(count=5)
    path = tmp_path / "old.skitter"
    assert app._save_to(path)

    def old_style(doc):  # what such a project held: a chosen move and no keys
        doc["camera_move"] = "follow"
        doc.pop("camera_track")

    rewrite(path, old_style)
    assert load_project(path).camera_from_move == "follow"
    assert app.open_project(path)
    assert len(session.project.camera_track.keys) == 5  # the move, as keys
    # A project with keys, or whose move is gone (Static, the simple moves), is left alone.
    for gone in ("static", "pull_back"):
        rewrite(path, lambda doc, gone=gone: doc.update(camera_move=gone))
        assert load_project(path).camera_from_move is None

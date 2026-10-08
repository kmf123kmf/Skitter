"""Project files: a whole project in one .skitter file.

A project file is a zip archive:

- project.json: the format version and every step's settings, the source's
  name and edits, whether the source was committed, and the matched
  mosaic's non-array data (see matching/saved.py). Its "view" part holds how
  the app was showing the project (such as the tab it was on): the app's
  business, not compared for unsaved changes, and safe to ignore.
- source.png: a copy of the source image as loaded (lossless; with alpha
  only when the image has transparency), so the project doesn't depend on
  the original file.
- regions.npz: the regions slicing made, when the source was committed (they
  are kept rather than sliced again, so the mosaic stays exactly as it was).
- mosaic.npz: the matched mosaic's per-region arrays, if there is one.

Loading is forgiving: settings or operations this version doesn't know (saved
by an older or newer one) keep their defaults or are left out, and each is
reported in `ProjectFile.problems`. Older animation settings are carried
over (`_old_animation`): the video's start and end holds become the build's
hold before and the last phase's hold after, the Still show becomes the
build's hold after, and "deconstruct" (the clear phase's first name) loads
as clear. Files of format 1 had a base tile 100 mosaic units wide (the
canvas grew with the columns); their regions and camera keys are scaled to
today's canvas, CANVAS_WIDTH across (`_old_units`). Files are written to a
temporary name and then renamed, so a failed save never damages an
existing project.

`document` is the JSON part alone; the app compares it to tell whether a
project has unsaved changes.
"""

import io
import json
import os
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from PIL import Image

from skitter.core.animation import PHASE_NAMES, PHASES, choreography_types
from skitter.core.animation.keyframes import CameraTrack
from skitter.core.animation.look import AnimationLook
from skitter.core.animation.phases import OPTIONAL, PhaseHolds
from skitter.core.animation.video import VideoSettings
from skitter.core.assembly import ExportSettings
from skitter.core.edits import apply_edits, edit_from_dict, edit_to_dict
from skitter.core.imaging import load_image
from skitter.core.matching.saved import SavedMosaic
from skitter.core.matching.settings import MatchSettings
from skitter.core.project import Project
from skitter.core.slicing import MosaicLayout, RegionSet, SlicingPlan

FORMAT = 2  # raise when a change would mislead older versions (2: a fixed canvas width)
OLD_TILE_WIDTH = 100.0  # format 1: a base tile's width in mosaic units
EXTENSION = ".skitter"
REGION_FIELDS = ("center", "size", "rotation", "z")


class ProjectFileError(Exception):
    """A file that isn't a readable project (the message says why)."""


@dataclass
class ProjectFile:
    """What a project file holds, ready for the app to use."""

    project: Project  # source_final and regions are set only if committed
    committed: bool  # the source was committed: regions (and a mosaic) belong to it
    mosaic: SavedMosaic | None = None  # restore with matching.saved.restore
    problems: list[str] = field(default_factory=list)  # what was skipped while loading
    view: dict = field(default_factory=dict)  # how the app showed it (see save_project)


def _plain(value):
    """JSON fallback for numpy values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def document(project: Project, committed: bool) -> dict:
    """The project's settings and choices as plain data (no images or arrays)."""
    return {
        "format": FORMAT,
        "source": {
            "name": project.source_path.name if project.source_path else None,
            "path": str(project.source_path) if project.source_path else None,
            "edits": [edit_to_dict(edit) for edit in project.source_edits],
            "committed": bool(committed and project.has_source),
        },
        "layout": project.layout.to_dict(),
        "slicing_plan": project.slicing_plan.to_dict(),
        "match_settings": project.match_settings.values(),
        "export_settings": project.export_settings.values(),
        "choreographies": {key: c.values() for key, c in project.choreographies.items()},
        "choreography": project.choreography_id,  # the build's (the only phase, once)
        "phases": {phase: project.phase_choices.get(phase) for phase in OPTIONAL},
        "phase_holds": {phase: holds.values() for phase, holds in project.phase_holds.items()},
        "camera_track": project.camera_track.to_dict(),
        "animation_look": project.animation_look.values(),
        "video_settings": project.video_settings.values(),
    }


def save_project(
    path, project: Project, committed: bool, mosaic: SavedMosaic | None, view: dict | None = None
) -> None:
    """Write project to path (a .skitter file). committed: project.source_final is the
    current source with its edits (regions and mosaic are saved only then). view: how
    the app shows the project (plain data, kept as is)."""
    if not project.has_source:
        raise ValueError("a project needs a source image to be saved")
    path = Path(path)
    doc = document(project, committed)
    committed = doc["source"]["committed"]
    regions = project.regions if committed else None
    mosaic = mosaic if regions is not None else None
    if mosaic is not None:
        doc["mosaic"] = mosaic.meta()
    if view:
        doc["view"] = view
    part = path.with_name(path.name + ".part")
    try:
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("project.json", json.dumps(doc, indent=1, default=_plain))
            archive.writestr("source.png", _png(project.source_original), zipfile.ZIP_STORED)
            if regions is not None:
                archive.writestr(
                    "regions.npz", _npz({f: getattr(regions, f) for f in REGION_FIELDS})
                )
            if mosaic is not None:
                archive.writestr("mosaic.npz", _npz(mosaic.arrays()))
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def load_project(path) -> ProjectFile:
    """Read a .skitter file (ProjectFileError if it isn't a readable project)."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            doc = json.loads(archive.read("project.json"))
            source = load_image(io.BytesIO(archive.read("source.png")))
            regions = _arrays(archive.read("regions.npz")) if "regions.npz" in names else None
            mosaic = _arrays(archive.read("mosaic.npz")) if "mosaic.npz" in names else None
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise ProjectFileError(f"{path.name} is not a readable Skitter project ({exc})") from exc
    if not isinstance(doc, dict) or doc.get("format", 0) > FORMAT:
        raise ProjectFileError(f"{path.name} was saved by a newer version of Skitter")

    problems: list[str] = []
    project = _project(doc, source, problems)
    committed = bool(doc.get("source", {}).get("committed"))
    if committed:
        final = np.array(project.source_image, dtype=np.uint8, order="C", copy=True)
        final.setflags(write=False)
        project.source_final = final
        if regions is not None:
            project.regions = RegionSet.from_arrays(*(regions[f] for f in REGION_FIELDS))
    if doc.get("format", 0) < 2:
        _old_units(project, project.layout.tile_width / OLD_TILE_WIDTH)
    saved = None
    if mosaic is not None and project.regions is not None and "mosaic" in doc:
        try:
            saved = SavedMosaic.from_parts(mosaic, doc["mosaic"], problems)
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"The matched mosaic could not be read ({exc}); match again")
    view = doc.get("view")
    return ProjectFile(project, committed, saved, problems,
                       view=view if isinstance(view, dict) else {})  # fmt: skip


def _project(doc: dict, source: np.ndarray, problems: list[str]) -> Project:
    project = Project()
    info = doc.get("source", {})
    if info.get("path"):
        project.source_path = Path(info["path"])
    project.source_original = source
    for item in info.get("edits", []):
        try:
            project.source_edits.append(edit_from_dict(item))
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"Source edit left out ({exc}); later edits may not line up")
    try:
        project.source_image = apply_edits(source, project.source_edits)
    except ValueError as exc:  # an edit that doesn't fit the image
        problems.append(f"Source edits dropped ({exc})")
        project.source_edits = []
        project.source_image = source

    if "layout" in doc:
        try:
            project.layout = MosaicLayout.from_dict(doc["layout"])
        except (TypeError, ValueError) as exc:
            problems.append(f"Mosaic layout ignored ({exc})")
    if "slicing_plan" in doc:
        project.slicing_plan = SlicingPlan.from_dict(doc["slicing_plan"], problems)
    project.match_settings = MatchSettings.from_values(doc.get("match_settings", {}), problems,
                                                       "Matching")  # fmt: skip
    project.export_settings = ExportSettings.from_values(doc.get("export_settings", {}), problems,
                                                         "Export")  # fmt: skip
    project.animation_look = AnimationLook.from_values(doc.get("animation_look", {}), problems,
                                                       "Animation look")  # fmt: skip
    video = dict(doc.get("video_settings", {}))
    old_holds = (video.pop("hold_start", None), video.pop("hold_end", None))
    project.video_settings = VideoSettings.from_values(video, problems, "Video")
    saved = doc.get("choreographies", {})
    for cls in choreography_types():
        if cls.id in saved:
            project.choreographies[cls.id] = cls.from_values(saved[cls.id], problems, cls.name)
    choices = {"build": doc.get("choreography"), **(doc.get("phases") or {})}
    if "deconstruct" in choices:
        choices.setdefault("clear", choices.pop("deconstruct"))
    still = choices.get("show") == "still"  # the Still show: a pause, now the build's hold
    if still:
        choices["show"] = None
    for phase, holds in (doc.get("phase_holds") or {}).items():
        if phase in project.phase_holds and isinstance(holds, dict):
            project.phase_holds[phase] = PhaseHolds.from_values(
                holds, problems, f"{PHASE_NAMES[phase]} holds")  # fmt: skip
    for phase, choice in choices.items():
        known = project.choreographies.get(choice) if isinstance(choice, str) else None
        if phase not in PHASES:
            continue
        if known is not None and known.phase == phase:
            project.phase_choices[phase] = choice
        elif choice is not None:
            problems.append(f"{PHASE_NAMES[phase]}: {choice!r} isn't available; "
                            f"{'the default' if phase == 'build' else 'none'} is used")  # fmt: skip
    if "phase_holds" not in doc:
        _old_animation(project, old_holds, saved.get("still") if still else None)
    if isinstance(doc.get("camera_track"), dict):
        project.camera_track = CameraTrack.from_dict(doc["camera_track"], problems)
    return project


def _old_animation(project: Project, holds, still) -> None:
    """Holds of a file from before phases had their own: the video's start hold before the
    build, its end hold (and a Still show's time) after the last phase played."""
    start, end = holds
    build = project.phase_holds["build"]
    build.update(hold_before=_seconds(start, 0.0))
    played = [phase for phase in PHASES if project.phase_choices.get(phase) is not None]
    for phase in PHASES:
        project.phase_holds[phase].update(hold_after=0.0)
    last = project.phase_holds[played[-1]]
    pause = _seconds((still or {}).get("duration"), 2.0) if still is not None else 0.0
    last.update(hold_after=min(_seconds(end, 2.0) + pause, 60.0))


def _old_units(project: Project, factor: float) -> None:
    """Positions of a format 1 file (mosaic units of 100 per base tile) scaled by factor to
    today's units (the canvas CANVAS_WIDTH across)."""
    if factor == 1.0:
        return
    regions = project.regions
    if regions is not None:
        project.regions = RegionSet.from_arrays(regions.center * factor, regions.size * factor,
                                                regions.rotation, regions.z)  # fmt: skip
    track = project.camera_track
    keys = []
    for key in track.keys:
        x, y = key.shot.center
        keys.append(replace(key, shot=replace(key.shot, center=(x * factor, y * factor))))
    project.camera_track = replace(track, keys=tuple(keys))


def _seconds(value, default: float) -> float:
    try:
        return min(max(float(value), 0.0), 60.0)
    except (TypeError, ValueError):
        return default


def _png(image: np.ndarray) -> bytes:
    """Lossless PNG; RGB when every pixel is opaque."""
    if image.shape[-1] == 4 and (image[..., 3] == 255).all():
        image = image[..., :3]
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buffer, "PNG", compress_level=3)
    return buffer.getvalue()


def _npz(arrays: dict) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **{name: np.asarray(a) for name, a in arrays.items()})
    return buffer.getvalue()


def _arrays(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        return {name: npz[name] for name in npz.files}

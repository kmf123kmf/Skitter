"""Observable wrapper around the Project shared by all step pages."""

import logging
import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from skitter.core.animation.video import check_video, plan_video, sync_size
from skitter.core.assembly import (
    ExportCancelled,
    ExportReport,
    ExportSettings,
    TileFiles,
    render_mosaic,
    save_mosaic,
)
from skitter.core.edits import Edit, apply_edits
from skitter.core.matching import edit as picks
from skitter.core.matching.candidates import Pins
from skitter.core.matching.matcher import MatchCancelled, Matcher, MatchResult
from skitter.core.project import Project
from skitter.core.scene import MosaicScene
from skitter.core.slicing import (
    MosaicLayout,
    SliceContext,
    SliceSummary,
    SlicingError,
    StageResult,
    summarize,
)
from skitter.core.tiles.library import OK, TileLibrary, UpdateReport, default_library_folder
from skitter.ui.jobs import Job
from skitter.ui.render.tile_textures import DetailRequest, TileTextures
from skitter.ui.render.video_export import VideoJob, run_video_job

logger = logging.getLogger(__name__)


class Session(QObject):
    """Holds the current Project and announces changes to it.

    Step pages modify the project only through Session methods so that every
    page interested in a change gets notified.
    """

    source_changed = Signal()  # a new source image was loaded
    source_edited = Signal(object, bool)  # (edit, undone); edit is None after revert
    source_committed = Signal()  # project.source_final changed; later steps must refresh
    layout_changed = Signal()  # project.layout changed (tile size, aspect, columns)
    slicing_changed = Signal()  # project.regions recomputed (see slicing_error, slicing_summary)
    library_changed = Signal()  # tile library opened, updated, or its folders changed
    library_progress = Signal(str, float)  # (message, fraction; -1 unknown) while updating
    matching_changed = Signal()  # project.matches replaced, or matching failed or stopped
    # (scene and textures always match project.matches when it is emitted)
    matching_progress = Signal(str, float)
    # A look at the run in progress (core.matching.matcher.MatchPreview), from the worker
    # thread (queued); a few may still arrive just after the run ends.
    matching_preview = Signal(object)
    # Progress within a long matching step (indexing, searching, widening): (message, 0..1).
    matching_detail = Signal(str, float)
    # Manual picks changed tiles of these regions (np.ndarray); project.matches, scene and
    # textures were updated in place of a full replacement (matching_changed isn't emitted).
    mosaic_edited = Signal(object)
    export_progress = Signal(str, float)
    export_finished = Signal(
        str, object, object
    )  # (path, ExportReport, error); cancelled: both None
    busy_changed = Signal()  # a background job started or stopped
    animation_changed = Signal()  # choreography, look or video settings edited
    video_progress = Signal(str, float)
    video_finished = Signal(str, object, object)  # (path, VideoReport, error); cancelled: both None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = Project()
        self._redo: list[Edit] = []
        self._source_loads = 0  # identifies the loaded original
        self._committed_key: tuple | None = None
        self._slice_context: SliceContext | None = None
        self._slicing_cache: list[StageResult] = []
        self.slicing_error: str | None = None
        self.slicing_summary: SliceSummary | None = None

        self.library: TileLibrary | None = None
        self.matcher: Matcher | None = None
        self.library_report: UpdateReport | None = None
        self.library_error: str | None = None
        self.match_error: str | None = None
        self._match_key: tuple | None = None  # inputs project.matches was computed from
        self._match_tiles: tuple | None = None  # the library tiles it uses (see mosaic_is_valid)
        self.scene: MosaicScene | None = None  # project.matches as a scene (core/scene.py)
        self.textures: TileTextures | None = None  # the scene's tile textures, shared by views
        # Manual picks: each entry maps regions to their (choice, manual) before a step.
        self._pick_undo: list[dict[int, tuple[int, bool]]] = []
        self._pick_redo: list[dict[int, tuple[int, bool]]] = []
        self.dropped_picks = 0  # picks the last matching run couldn't keep
        self._job: Job | None = None
        self._job_kind: str | None = None

    def set_source(self, path: Path, image: np.ndarray) -> None:
        project = self.project
        project.source_path = Path(path)
        project.source_original = image
        project.source_edits = []
        project.source_image = image
        self._redo.clear()
        self._source_loads += 1
        self.source_changed.emit()

    # Committing the source

    def _source_key(self) -> tuple:
        return (self._source_loads, tuple(self.project.source_edits))

    @property
    def source_is_committed(self) -> bool:
        """Whether source_final matches the current loaded image and edits."""
        return self.project.has_source and self._committed_key == self._source_key()

    def commit_source(self) -> bool:
        """Snapshot the edited source as project.source_final.

        Returns True if the final image changed (and source_committed was
        emitted); committing an unchanged image leaves later steps' work intact.
        """
        if not self.project.has_source or self.source_is_committed:
            return False
        final = np.array(self.project.source_image, dtype=np.uint8, order="C", copy=True)
        final.setflags(write=False)
        self.project.source_final = final
        self._committed_key = self._source_key()
        self._rebuild_slice_context()
        self._evaluate_slicing()
        self.source_committed.emit()
        return True

    # Mosaic layout

    def set_layout(self, layout: MosaicLayout) -> None:
        """Change the base tile or column count; re-slices the final image."""
        if layout == self.project.layout:
            return
        self.project.layout = layout
        self._rebuild_slice_context()  # first, so listeners see the new mosaic size
        self.layout_changed.emit()
        self._evaluate_slicing()

    def mosaic_size(self) -> tuple[float, float] | None:
        """Canvas size in mosaic units, once the source is committed."""
        ctx = self._slice_context
        return None if ctx is None else (ctx.width, ctx.height)

    @property
    def slice_context(self) -> SliceContext | None:
        return self._slice_context

    # Slicing

    def slicing_edited(self) -> None:
        """Re-run slicing after project.slicing_plan (stages or parameters) changed."""
        self._evaluate_slicing()

    def _rebuild_slice_context(self) -> None:
        """New final image or layout: new context, no cached stage results."""
        final = self.project.source_final
        if final is not None:
            self._slice_context = SliceContext(final, self.project.layout)
        self._slicing_cache = []

    def _evaluate_slicing(self) -> None:
        project = self.project
        self.slicing_summary = None
        if self._slice_context is None:
            project.regions = None
        else:
            try:
                self._slicing_cache = project.slicing_plan.evaluate(
                    self._slice_context, self._slicing_cache
                )
            except Exception as exc:
                # Any failure must reach the UI; otherwise the preview silently goes stale.
                if isinstance(exc, SlicingError):
                    self.slicing_error = str(exc)
                else:
                    logger.exception("slicing plan failed")
                    self.slicing_error = f"Slicing failed: {type(exc).__name__}: {exc}"
                self._slicing_cache = []
                project.regions = None
            else:
                self.slicing_error = None
                ctx = self._slice_context
                project.regions = (
                    self._slicing_cache[-1].regions if self._slicing_cache else ctx.canvas()
                )
                if project.regions:
                    self.slicing_summary = summarize(project.regions, ctx)
        self._drop_stale_matches()
        self.slicing_changed.emit()

    # Source edits

    @property
    def can_undo(self) -> bool:
        return bool(self.project.source_edits)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def next_redo(self) -> Edit | None:
        return self._redo[-1] if self._redo else None

    def apply_edit(self, edit: Edit) -> None:
        self.project.source_edits.append(edit)
        self._redo.clear()
        self._rebuild_source()
        self.source_edited.emit(edit, False)

    def undo(self) -> None:
        if not self.project.source_edits:
            return
        edit = self.project.source_edits.pop()
        self._redo.append(edit)
        self._rebuild_source()
        self.source_edited.emit(edit, True)

    def redo(self) -> None:
        if not self._redo:
            return
        edit = self._redo.pop()
        self.project.source_edits.append(edit)
        self._rebuild_source()
        self.source_edited.emit(edit, False)

    def revert_edits(self) -> None:
        """Drop all edits; they stay available to redo one at a time."""
        edits = self.project.source_edits
        if not edits:
            return
        self._redo.extend(reversed(edits))
        edits.clear()
        self._rebuild_source()
        self.source_edited.emit(None, True)

    def _rebuild_source(self) -> None:
        project = self.project
        project.source_image = apply_edits(project.source_original, project.source_edits)

    # Background jobs (one at a time)

    @property
    def busy(self) -> str | None:
        """The running job's kind ("library", "matching", "export" or "video"), else None."""
        return self._job_kind if self._job is not None and self._job.running else None

    def cancel_job(self) -> None:
        if self._job is not None:
            self._job.cancel()

    def wait_for_job(self, timeout: float | None = None) -> None:
        if self._job is not None:
            self._job.wait(timeout)

    def shutdown(self, timeout: float = 10) -> None:
        """Stop all background work (the app is closing)."""
        self.cancel_job()
        if self.textures is not None:
            self.textures.cancel()
            self.textures.wait(timeout)
        self.wait_for_job(timeout)

    def _start_job(self, kind: str, work, progress_signal, on_done, cancel_errors=()) -> Job:
        if self.busy:
            raise RuntimeError(f"{self.busy} job already running")
        job = Job(work, cancel_errors, self)
        job.progress.connect(progress_signal)
        job.finished.connect(lambda result: on_done(result, None))
        job.failed.connect(lambda message: on_done(None, message))
        job.cancelled.connect(lambda: on_done(None, None))
        job.stopped.connect(self._job_stopped)
        self._job, self._job_kind = job, kind
        job.start()
        self.busy_changed.emit()
        return job

    def _job_stopped(self) -> None:
        self._job = self._job_kind = None
        self.busy_changed.emit()

    # Tile library

    def open_library(self, folder: str | Path | None = None) -> TileLibrary:
        """Open (or create) the tile library cache; default: the per-user location."""
        if self.library is not None:
            self.library.close()
        if self.project.matches is not None:  # its tiles refer to the old library
            self._set_matches(None)
            self.matching_changed.emit()
        self.library = TileLibrary(folder or default_library_folder())
        self.matcher = Matcher(self.library)
        self.library_report = self.library_error = None
        self.library_changed.emit()
        return self.library

    def set_library_folders(self, folders) -> None:
        self.library.set_roots(folders)
        self.library_changed.emit()

    @property
    def library_ready(self) -> bool:
        return self.library is not None and len(self.library) > 0 and self.busy != "library"

    def update_library(self, workers: int | None = None) -> Job:
        """Read new and changed images from the library's folders in the background."""
        library = self.library

        def work(progress, cancelled):
            def report(phase, done, total):
                progress(f"{phase}: {done:,}" + (f" of {total:,}" if total else ""),
                         done / total if total else None)  # fmt: skip

            return library.update(progress=report, cancelled=cancelled, workers=workers)

        def done(report, error):
            self.library_report, self.library_error = report, error
            self.library_changed.emit()

        return self._start_job("library", work, self.library_progress, done)

    # Matching

    def _match_inputs(self) -> tuple:
        library = self.library
        return (
            self.project.regions,
            self._slice_context,
            library.version if library is not None else None,
            self.project.match_settings.key(),
        )

    @property
    def matching_is_current(self) -> bool:
        """Whether project.matches was computed from the current regions, library and settings."""
        if self.project.matches is None or self._match_key is None:
            return False
        now = self._match_inputs()
        then = self._match_key
        return self._same_regions(then) and now[2:] == then[2:]

    def _same_regions(self, key: tuple) -> bool:
        """Whether key (from _match_inputs) has the current regions and slice context."""
        return key[0] is self.project.regions and key[1] is self._slice_context

    def _drop_stale_matches(self) -> None:
        """Forget matches made for other regions: they no longer fit the mosaic.

        (A library or settings change only makes them out of date.)
        """
        if self.project.matches is not None and not self._same_regions(self._match_key):
            self._set_matches(None)
            self.matching_changed.emit()

    @property
    def can_match(self) -> bool:
        return (
            bool(self.project.regions) and self._slice_context is not None and (self.library_ready)
        )

    def start_matching(self, keep_picks: bool = False) -> Job:
        """Match tiles to the current regions in the background.

        keep_picks: keep the current mosaic's manual picks (pinned in place).
        """
        if not self.can_match:
            raise RuntimeError("matching needs regions and a tile library")
        key = self._match_inputs()
        regions, ctx, _, _ = key
        settings = self.project.match_settings.copy()
        matcher = self.matcher
        kept = self._pins() if keep_picks else None
        self.dropped_picks = 0 if kept is None else self.manual_picks - len(kept)

        def work(progress, cancelled):
            return matcher.run(regions, ctx, settings, progress, cancelled, pins=kept,
                               preview=self.matching_preview.emit,
                               detail=self.matching_detail.emit)  # fmt: skip

        def done(result: MatchResult | None, error: str | None):
            self.match_error = error
            if result is not None and self._same_regions(key):  # else re-sliced meanwhile
                self._set_matches(result, key)
            self.matching_changed.emit()

        self.match_error = None
        return self._start_job("matching", work, self.matching_progress, done, (MatchCancelled,))

    # Manual picks

    @property
    def manual_picks(self) -> int:
        """Regions of the current mosaic whose tiles were picked by hand."""
        matches = self.project.matches
        return 0 if matches is None or not self.mosaic_is_valid else matches.manual_count

    def can_pick(self, region: int) -> bool:
        return (
            self.mosaic_is_valid
            and not self._picks_blocked
            and picks.editable(self.project.matches, region)
        )

    @property
    def _picks_blocked(self) -> bool:
        """Matching or a library update is running (exports copy what they render)."""
        return self.busy in ("matching", "library")

    def pick_tile(self, region: int, ref: int) -> None:
        """Show candidate `ref` (see Candidates) in a region, picked by hand."""
        self._edit_picks({int(region): (int(ref), True)})

    def revert_picks(self, regions=None) -> None:
        """Give regions (default: every manual pick) back the matcher's choice."""
        matches = self.project.matches
        if regions is None:
            regions = np.flatnonzero(matches.manual)
        auto = matches.candidates.auto
        self._edit_picks({int(r): (int(auto[r]), False) for r in regions})

    def find_more_candidates(self, region: int) -> int:
        """Search every tile for one region; returns how many candidates were added."""
        return picks.find_more(self.project.matches, self._slice_context, int(region))

    @property
    def can_undo_pick(self) -> bool:
        return bool(self._pick_undo) and self.mosaic_is_valid and not self._picks_blocked

    @property
    def can_redo_pick(self) -> bool:
        return bool(self._pick_redo) and self.mosaic_is_valid and not self._picks_blocked

    def undo_pick(self) -> None:
        if self.can_undo_pick:
            self._edit_picks(self._pick_undo.pop(), self._pick_redo)

    def redo_pick(self) -> None:
        if self.can_redo_pick:
            self._edit_picks(self._pick_redo.pop(), self._pick_undo)

    def _edit_picks(self, changes: dict[int, tuple[int, bool]], history=None) -> None:
        """Apply picks; the state they replace goes on `history` (default: undo, and the
        redo history is dropped)."""
        matches = self.project.matches
        if matches is None or not self.mosaic_is_valid or self._picks_blocked:
            return
        before = {r: (int(matches.choice[r]), bool(matches.manual[r])) for r in changes}
        edited = picks.apply_choices(matches, self._slice_context, changes)
        if edited is matches:
            return
        if history is None:
            history = self._pick_undo
            self._pick_redo.clear()
        history.append(before)
        changed = np.fromiter(changes, np.int64)
        self.project.matches = edited
        self._match_tiles = self._tile_snapshot(edited.tile)
        self.scene = self.scene.with_result(edited, self._slice_context)
        self.textures.set_scene(self.scene, self.library)
        self.mosaic_edited.emit(changed)

    def _pins(self) -> Pins | None:
        """The current picks, less those whose photos are no longer usable."""
        matches = self.project.matches
        if matches is None or not matches.manual_count or not self.mosaic_is_valid:
            return None
        kept = picks.pins(matches)
        status = self.library.status
        ok = kept.tile < len(status)
        ok[ok] = status[kept.tile[ok]] == OK
        return kept.subset(ok)

    def match_settings_edited(self) -> None:
        """project.match_settings changed (results become out of date)."""
        self.matching_changed.emit()

    # Export

    def _set_matches(self, result: MatchResult | None, key: tuple | None = None) -> None:
        """Replace project.matches, with its scene and textures (callers emit matching_changed)."""
        if self.textures is not None:
            self.textures.cancel()
        self.project.matches, self._match_key = result, key
        self._pick_undo.clear()
        self._pick_redo.clear()
        if result is None:
            self._match_tiles = self.scene = self.textures = None
            return
        self._match_tiles = self._tile_snapshot(result.tile)
        self.scene = MosaicScene.from_result(result, self._slice_context)
        self.textures = TileTextures(self.scene, self.library, self)

    def _tile_snapshot(self, tiles) -> tuple:
        """The library and the state of the tiles a mosaic uses."""
        library = self.library
        slots = np.unique(tiles[tiles >= 0])
        return library, slots, library.paths(slots), library.width[slots], library.height[slots]

    @property
    def mosaic_is_valid(self) -> bool:
        """Whether project.matches still fits the committed source, the regions and the tiles.

        Unlike matching_is_current, changed match settings or new library
        photos don't matter; a used tile that changed, vanished or failed does.
        """
        if (
            self.project.matches is None
            or self._match_key is None
            or not self._same_regions(self._match_key)
            or not self.source_is_committed
            or self._match_tiles is None
        ):
            return False
        library, slots, paths, width, height = self._match_tiles
        if library is not self.library:
            return False
        return (
            bool(np.all(library.status[slots] == OK))
            and library.paths(slots) == paths
            and np.array_equal(library.width[slots], width)
            and np.array_equal(library.height[slots], height)
        )

    @property
    def can_export(self) -> bool:
        """Whether there is a valid mosaic to export (and the library is not being updated)."""
        return self.busy != "library" and self.mosaic_is_valid

    # Animation

    def animation_edited(self) -> None:
        """The choreography, its settings, the look or the video settings changed."""
        sync_size(self.project.video_settings, self.scene)
        self.animation_changed.emit()

    def video_problems(self, path: str | Path | None = None) -> list[str]:
        """Why a video export can't start now (empty: it can)."""
        problems = []
        if not self.can_export:
            problems.append("Match tiles first: there is no up-to-date mosaic to animate.")
            return problems
        project = self.project
        return check_video(self.scene, project.video_settings, project.animation_look.background,
                           None if path is None else Path(path))  # fmt: skip

    def start_video_export(self, path: str | Path) -> Job:
        """Render the current animation to a video file, in the background."""
        problems = self.video_problems(path)
        if problems:
            raise RuntimeError(problems[0])
        project, scene = self.project, self.scene
        choreography = project.choreography.copy()
        settings = project.video_settings.copy()
        background = project.animation_look.background
        look = project.animation_look.copy()
        timeline = choreography.timeline(scene, look)
        plan = plan_video(scene, timeline.duration, settings, background)
        job = VideoJob(
            path=Path(path), scene=scene, choreography=choreography, settings=settings,
            background=background, look=look, plan=plan,
            # Read here: the library isn't threadsafe.
            request=DetailRequest.for_scene(scene, TileFiles.read(self.library, scene.slot),
                                            plan.scale),
        )  # fmt: skip

        def work(progress, cancelled):
            return run_video_job(job, progress, cancelled)

        def done(report, error):
            self.video_finished.emit(str(path), report, error)

        return self._start_job("video", work, self.video_progress, done)

    def export_signals(self) -> tuple:
        """Signals after which can_export may have changed."""
        return (
            self.source_changed, self.source_edited, self.source_committed, self.slicing_changed,
            self.library_changed, self.matching_changed, self.busy_changed,
        )  # fmt: skip

    def start_export(self, path: str | Path, settings: ExportSettings) -> Job:
        """Render the current mosaic at full detail and save it, in the background."""
        if not self.can_export:
            raise RuntimeError("export needs a matched mosaic")
        path = Path(path)
        scene = self.scene
        files = TileFiles.read(self.library, scene.slot)  # here: the library isn't threadsafe
        settings = settings.copy()

        def work(progress, cancelled) -> ExportReport:
            image, report = render_mosaic(scene, files, settings, progress, cancelled)
            progress(f"Saving {path.name}…", None)
            part = path.with_name(path.name + ".part")  # never leave a half-written image
            try:
                save_mosaic(image, part, settings)
                os.replace(part, path)
            finally:
                part.unlink(missing_ok=True)
            return report

        def done(report: ExportReport | None, error: str | None):
            self.export_finished.emit(str(path), report, error)

        return self._start_job("export", work, self.export_progress, done, (ExportCancelled,))

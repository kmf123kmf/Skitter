"""The Infinite Mosaic window: dive into any tile, and its photo becomes the next mosaic.

The side panel holds the settings (core/infinite.py's InfiniteSettings and
the chosen slicer's own) and Start. Starting fixes them for the session and
shows its start (Start with):

- Random photo: a library photo shows, the first level is matched (it also
  builds the search index, the long part), and the photo dissolves into its
  mosaic.
- Random mosaic: the view fills at once with random photos in the layout
  (`InfiniteSession.random_mosaic`), while the search index is built in the
  background (`prepare`). A dive made before it is ready plays as usual and
  its matching waits for it (the photo holds meanwhile).

New Start shows another start of the same kind, with the same settings.
Pan and zoom as usual; double-click a tile to dive into it:

- At once, the tile's photo is read at its own resolution and laid over
  the tile (the photo layer), and the next level starts matching in the
  background (`InfiniteSession.level`).
- Over the dive, the tile straightens and loses its tint while the view
  zooms until it fills it, and every other tile flies outward, out of view
  (`_dive_frame`).
- Once both are done the window moves to the new level: the photo now
  covers the new canvas exactly as it covered the tile, so the view is
  refitted with nothing changing on screen (`_arrive`), and the photo fades
  into the new mosaic below it. If matching takes longer than the dive, the
  photo holds until it is in.

Each level gets its own world (mosaic units of its own canvas), so diving
never runs out of floating-point precision. Tiles show from thumbnails at
once and full-size crops when read (TileTextures, through a process pool
kept running for the session); zooming in sharpens them (ZoomDetail).
Pan and zoom are locked during a dive and the fade. F11 toggles full
screen (the panel hides).
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from skitter.core.easing import ease_in_out_cubic, ease_out_cubic
from skitter.core.infinite import (
    DIVE,
    DIVE_THREADS,
    MATCHING,
    MOSAIC,
    InfiniteSession,
    InfiniteSettings,
    Level,
    TileView,
)
from skitter.core.matching.mosaic import MosaicMethod
from skitter.core.matching.result import MatchCancelled
from skitter.core.slicing import SlicingCancelled, SlicingPlan
from skitter.core.slicing.layout import CANVAS_WIDTH
from skitter.ui import preferences
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.jobs import Job
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.render.tile_textures import TileTextures
from skitter.ui.render.zoom_detail import DETAIL_PAGES, DETAIL_SIDE, ZoomDetail
from skitter.ui.steps.base import side_panel
from skitter.ui.style import muted, progress_bar, show_progress
from skitter.ui.widgets.param_form import ParamForm

SETTINGS_KEY = "infinite/settings"  # preferences: the last session's settings (JSON)
PLAN_KEY = "infinite/plan"  # and its slicers' settings
FLY_OUT = 3.0  # how far the other tiles fly, in their distances from the dived-into tile
PHOTO_IN = 0.2  # share of the dive over which the photo fades in over its tile
PIXELATE_ABOVE = 4.0  # texels this large show as squares (not while diving: tiles blow up)
START_TIP = "Start a session from a random photo of the library."
LOCKED_EVENTS = (
    QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease, QEvent.Type.MouseMove,
    QEvent.Type.MouseButtonDblClick, QEvent.Type.Wheel,
)  # fmt: skip


def load_settings() -> tuple[InfiniteSettings, SlicingPlan]:
    """The settings the last session started with (defaults if none, or unreadable)."""
    prefs = preferences.settings()
    try:
        settings = InfiniteSettings.from_values(json.loads(prefs.value(SETTINGS_KEY, "{}")), [])
    except (TypeError, ValueError):
        settings = InfiniteSettings()
    try:
        plan = SlicingPlan.from_dict(json.loads(prefs.value(PLAN_KEY, "{}")), [])
    except (TypeError, ValueError, KeyError):
        plan = SlicingPlan()
    return settings, plan


def save_settings(settings: InfiniteSettings, plan: SlicingPlan) -> None:
    prefs = preferences.settings()
    prefs.setValue(SETTINGS_KEY, json.dumps(settings.values()))
    prefs.setValue(PLAN_KEY, json.dumps(plan.to_dict()))


class _Dive:
    """What a dive into tile `index` starts from."""

    def __init__(self, index: int, layer: SpriteLayer, canvas: MosaicCanvas, end_zoom: float):
        tiles = layer.instances
        self.index = index
        self.pos = tiles["pos"].astype(np.float64)
        self.center = self.pos[index].copy()
        self.rotation = float(tiles["rotation"][index])
        self.offset = tiles["offset"][index].copy()
        cam = canvas.camera
        self.zoom0, self.zoom1 = float(cam.zoom), float(end_zoom)
        self.screen0 = (self.center - cam.center) * cam.zoom  # the tile's offset on screen
        self.matched: Level | None = None  # the next level, once matched
        self.done = False  # the animation has finished


class InfiniteWindow(QWidget):
    """Infinite Mosaic, on the app session's library (and its matcher's indexes)."""

    detail_reported = Signal(str, float)  # progress within a step (from the job's thread)

    def __init__(self, session, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Infinite Mosaic")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.app_session = session
        self.settings, self.plan = load_settings()
        self.infinite: InfiniteSession | None = None
        self.level: Level | None = None
        self.textures: TileTextures | None = None
        self.tile_layer: SpriteLayer | None = None
        self.photo_layer: SpriteLayer | None = None
        self.dive: _Dive | None = None
        self._job: Job | None = None
        self._job_kind = ""  # "first" (a random photo's level), "prepare" or "dive"
        self._job_started = 0.0
        self._queued: np.ndarray | None = None  # a dive's picture, matched once prepared
        self._prepared = False  # the search index is built (a random mosaic's session)
        self._pool: ProcessPoolExecutor | None = None
        self._locked = False  # pan and zoom off (diving, fading)
        self._fade_token: object | None = None

        self.canvas = MosaicCanvas()
        self.canvas.fit_margin = 1.0  # a picture fills the view, as a tile dived into does
        self.canvas.pixelate_above = PIXELATE_ABOVE
        self.canvas.double_clicked.connect(self._on_double_click)
        self.canvas.installEventFilter(self)
        self.zoom_detail = ZoomDetail(self.canvas, self)
        self.detail_reported.connect(self._on_detail)

        self.panel = side_panel(
            self._build_session_group(), *self._build_settings_groups()
        )  # fmt: skip
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.panel)
        QShortcut(QKeySequence("F11"), self, self.toggle_full_screen)
        QShortcut(QKeySequence("Esc"), self, self._leave_full_screen)
        self.resize(1400, 900)
        self._refresh()

    # Construction

    def _build_settings_groups(self) -> list[QGroupBox]:
        """Mosaic (with the slicer's own settings), Matching and Dive."""
        self.forms = {name: ParamForm() for name in ("mosaic", "matching", "dive")}
        for form in self.forms.values():
            form.changed.connect(self._on_setting_changed)
        self.slicer_form = ParamForm()
        self.settings_note = muted(QLabel("Settings stay as they are for the whole session."))
        self._target_forms()
        groups = []
        for title, name in (("Mosaic", "mosaic"), ("Matching", "matching"), ("Dive", "dive")):
            group = QGroupBox(title)
            layout = QVBoxLayout(group)
            layout.addWidget(self.forms[name])
            if name == "mosaic":
                layout.addWidget(self.slicer_form)
            groups.append(group)
        groups[0].layout().addWidget(self.settings_note)
        return groups

    def _target_forms(self) -> None:
        """Show self.settings (and its slicer's settings) in the forms."""
        for name, only in (("mosaic", MOSAIC), ("matching", MATCHING), ("dive", DIVE)):
            self.forms[name].set_target(self.settings, only)
        self.slicer_form.set_target(self.plan.slicers[self.settings.slicer])

    def set_settings(self, settings: InfiniteSettings) -> None:
        """Edit these settings from now on (the next session starts with them)."""
        self.settings = settings
        self._target_forms()

    def _build_session_group(self) -> QGroupBox:
        self.start_button = QPushButton("Start")
        self.start_button.setToolTip(START_TIP)
        self.start_button.clicked.connect(self.toggle_session)
        self.new_start_button = QPushButton("New Start")
        self.new_start_button.setToolTip(
            "Start again from another random photo or mosaic, with the same settings."
        )
        self.new_start_button.clicked.connect(self.new_start)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.progress = progress_bar()
        self.sub_status = muted(QLabel())
        self.sub_progress = progress_bar()
        self.sub_progress.setMaximumHeight(6)
        self.level_label = QLabel()
        self.level_label.setWordWrap(True)
        help_text = muted(QLabel(
            "Double-click a tile to dive into it: its photo becomes the next mosaic. Drag to "
            "pan, use the wheel to zoom. F11: full screen."
        ))  # fmt: skip
        group = QGroupBox("Session")
        layout = QVBoxLayout(group)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.new_start_button)
        layout.addLayout(buttons)
        for widget in (self.status, self.progress, self.sub_status, self.sub_progress,
                       self.level_label, help_text):  # fmt: skip
            layout.addWidget(widget)
        return group

    def _on_setting_changed(self, name: str) -> None:
        if name == "slicer":
            self.slicer_form.set_target(self.plan.slicers[self.settings.slicer])
        else:  # settings in one form may show or hide others in another (`when`)
            for form in self.forms.values():
                form.refresh()

    # Session

    @property
    def running(self) -> bool:
        return self.infinite is not None

    @property
    def busy(self) -> bool:
        """Matching a level, diving or fading: no new dive yet (preparing allows one)."""
        matching = self._job is not None and self._job_kind != "prepare"
        return matching or self.dive is not None or self._fade_token is not None

    @property
    def can_start_anew(self) -> bool:
        """New Start: running and not busy (a random mosaic may be preparing)."""
        return self.running and not self.busy and self._queued is None

    def toggle_session(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        """A new session with the current settings, from its start (Start with)."""
        app = self.app_session
        library = app.library if app.library is not None else app.open_library()
        if not len(library):
            self.status.setText("The tile library has no photos: add some on the Tiles tab.")
            return
        save_settings(self.settings, self.plan)
        base = next((m for m in app.project.match_methods.values()
                     if isinstance(m, MosaicMethod)), None)  # fmt: skip
        self.infinite = InfiniteSession(library, self.settings, self.plan, base, app.matcher)
        self._pool = ProcessPoolExecutor(os.cpu_count())
        for _ in range(os.cpu_count() or 1):  # start the workers while the first level matches
            self._pool.submit(int)
        self._prepared = False
        self.new_start()

    def new_start(self) -> None:
        """Show the session's start: another random photo or mosaic."""
        if not self.can_start_anew:
            return
        if self.settings.start == "mosaic":
            self._start_mosaic()
        else:
            self._start_photo()
        self._refresh()

    def _start_photo(self) -> None:
        self._clear_level()
        self.level = None
        picture = self.infinite.picture(self.infinite.random_tile())
        width, height = CANVAS_WIDTH, CANVAS_WIDTH * picture.shape[0] / picture.shape[1]
        self._show_photo(picture, (width / 2, height / 2), (width, height), 0.0, 1.0)
        self.canvas.fit_to(0, 0, width, height)
        self._match(picture, "first", "Preparing tiles (the first level takes longest)…")

    def _start_mosaic(self) -> None:
        if self.photo_layer is not None:
            self.canvas.remove_layer(self.photo_layer)
            self.photo_layer = None
        width, height = self.canvas.width(), self.canvas.height()
        level = self.infinite.random_mosaic(width / height if width > 0 and height > 0 else 1.5)
        self._show_level(level)
        self.level_label.setText(f"Random mosaic: {len(level.scene):,} tiles")
        if not self._prepared and self._job is None:
            self._prepare()
        elif self._prepared:
            self.status.setText("Ready.")

    def stop(self) -> None:
        """End the session: stop its work and clear the view."""
        if self._job is not None:
            self._job.cancel()
            self._job.wait()
            self._job = None
        self.dive = None
        self._fade_token = None
        self._queued = None
        self._locked = False
        self._clear_level()
        if self.photo_layer is not None:
            self.canvas.remove_layer(self.photo_layer)
            self.photo_layer = None
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        self.infinite = self.level = None
        self.status.setText("")
        self.level_label.setText("")
        self._refresh()

    def _match(self, picture: np.ndarray, kind: str, message: str) -> None:
        """Match a level in the background: kind "first" (a random photo: every core, it
        may build the index) or "dive" (during the animation: fewer search threads)."""
        infinite = self.infinite
        threads = None if kind == "first" else DIVE_THREADS

        def work(progress, cancelled):
            return infinite.level(picture, progress, cancelled, self.detail_reported.emit,
                                  threads)  # fmt: skip

        self._start_job(work, kind, message, self._on_matched)

    def _prepare(self) -> None:
        """Build the search index in the background (a random mosaic shows meanwhile)."""
        infinite = self.infinite

        def work(progress, cancelled):
            infinite.prepare(progress, cancelled, self.detail_reported.emit)

        self._start_job(work, "prepare", "Preparing tiles: dives match once ready…",
                        self._on_prepared)  # fmt: skip

    def _start_job(self, work, kind: str, message: str, on_done) -> None:
        job = Job(work, (MatchCancelled, SlicingCancelled), self)
        job.progress.connect(self._on_progress)
        job.finished.connect(on_done)
        job.failed.connect(self._on_failed)
        self._job, self._job_kind, self._job_started = job, kind, time.perf_counter()
        self.status.setText(message)
        show_progress(self.progress, -1)
        job.start()
        self._refresh()

    @property
    def _followed(self) -> bool:
        """The running job takes long enough to show its progress (not a dive's)."""
        return self._job is not None and self._job_kind != "dive"

    def _on_progress(self, message: str, fraction: float) -> None:
        if self._followed:
            if self._job_kind == "first":
                self.status.setText(message)
            show_progress(self.progress, fraction)
        self._show_sub(False)

    def _on_detail(self, message: str, fraction: float) -> None:
        if not self._followed:
            return
        self.sub_status.setText(message)
        show_progress(self.sub_progress, min(max(fraction, 0.0), 1.0))
        self._show_sub(True)

    def _show_sub(self, shown: bool) -> None:
        self.sub_status.setVisible(shown)
        self.sub_progress.setVisible(shown)

    def _on_prepared(self, _result) -> None:
        self._job, self._prepared = None, True
        self.status.setText("Ready." if self.dive is None else "Matching the next level…")
        queued, self._queued = self._queued, None
        if queued is not None:
            self._match(queued, "dive", "Matching the next level…")
        self._refresh()

    def _on_matched(self, level: Level) -> None:
        self._job = None
        seconds = time.perf_counter() - self._job_started
        self.level_label.setText(f"Level {level.number}: {len(level.scene):,} tiles, matched "
                                 f"in {seconds:.2f} s")  # fmt: skip
        if self.dive is None:  # the first level
            self._arrive(level)
        elif self.dive.done:
            self._arrive(level)
        else:
            self.dive.matched = level
        self._refresh()

    def _on_failed(self, message: str) -> None:
        self._job = None
        self.stop()
        self.status.setText(f"Matching failed: {message}")

    # Showing a level

    def _show_photo(self, picture, center, size, rotation: float, alpha: float) -> None:
        """Lay the picture over everything (one sprite)."""
        if self.photo_layer is not None:
            self.canvas.remove_layer(self.photo_layer)
        photo = make_instances(1)
        photo["pos"], photo["size"] = center, size
        photo["rotation"], photo["alpha"] = rotation, alpha
        self.photo_layer = self.canvas.add_layer(SpriteLayer(picture, photo))

    def _clear_level(self) -> None:
        self.zoom_detail.show(None, None, None)
        if self.tile_layer is not None:
            self.canvas.remove_layer(self.tile_layer)
            self.tile_layer = None
        if self.textures is not None:
            self.textures.cancel()
            self.textures.changed.disconnect(self._on_textures_changed)
            self.textures = None

    def _arrive(self, level: Level) -> None:
        """Move to a level: the photo covers its canvas as it covered the tile (or the
        view, at the start), the view refits to it, and it fades into the mosaic."""
        self.dive = None
        width, height = float(level.ctx.width), float(level.ctx.height)
        photo = self.photo_layer.instances
        photo["pos"], photo["size"], photo["rotation"] = (width / 2, height / 2), (width, height), 0
        self.photo_layer.mark_dirty()
        self._show_level(level)
        self.status.setText("Ready.")
        show_progress(self.progress, 1.0)
        self._show_sub(False)
        self._fade_photo()

    def _show_level(self, level: Level) -> None:
        """Show a level's tiles (under the photo, if any), fitted to the view."""
        self._clear_level()
        self.level = level
        self.canvas.fit_to(0, 0, float(level.ctx.width), float(level.ctx.height))
        self.canvas.pixelate_above = PIXELATE_ABOVE
        self.textures = TileTextures(level.scene, self.infinite.library, self, pool=self._pool)
        self.textures.changed.connect(self._on_textures_changed)
        self._new_tile_layer()

    def _new_tile_layer(self) -> None:
        """The level's tiles below the photo, from the best textures there are."""
        if self.tile_layer is not None:
            self.canvas.remove_layer(self.tile_layer)
        layer = SpriteLayer(self.textures.pages, self.textures.instances(),
                            detail_pages=DETAIL_PAGES, detail_side=DETAIL_SIDE)  # fmt: skip
        self.tile_layer = self.canvas.add_layer(layer, 0)
        self.zoom_detail.show(layer, self.level.scene, self.infinite.library)

    def _on_textures_changed(self) -> None:
        """Full-size tiles arrived: swap them in (unless a dive is moving them)."""
        if self.tile_layer is not None and self.dive is None:
            self._new_tile_layer()

    def _fade_photo(self) -> None:
        token = self._fade_token = object()
        duration = float(self.settings.fade_seconds)
        self._locked = True
        layer = self.photo_layer

        def step(t: float) -> bool:
            if self._fade_token is not token:
                return False
            e = 1.0 if duration <= 0 else min(t / duration, 1.0)
            layer.instances["alpha"] = 1.0 - e
            layer.mark_dirty()
            if e < 1.0:
                return True
            layer.visible = False
            self._fade_token = None
            self._locked = False
            self._refresh()
            return False

        self.canvas.add_animation(step)
        self._refresh()

    # Diving

    def tile_at(self, x: float, y: float) -> int:
        """The level's topmost tile at world (x, y) (index into its scene), or -1."""
        scene = self.level.scene
        region = scene.result.regions.hit_test(x, y)
        if region < 0:
            return -1
        hits = np.flatnonzero(scene.region == region)
        return int(hits[0]) if len(hits) else -1

    def _on_double_click(self, sx: float, sy: float) -> None:
        if self.level is None or self.busy:
            return
        index = self.tile_at(*self.canvas.camera.screen_to_world(sx, sy))
        if index >= 0:
            self.dive_into(index)

    def dive_into(self, index: int) -> None:
        """Dive into tile index of the level: its photo becomes the next one."""
        scene, infinite = self.level.scene, self.infinite
        tile = TileView(int(scene.slot[index]), tuple(float(v) for v in scene.rect[index]),
                        bool(scene.mirrored[index]))  # fmt: skip
        picture = infinite.picture(tile)
        self.zoom_detail.show(None, None, None)  # tiles move: nothing to sharpen
        self.canvas.pixelate_above = None  # tiles grow far past their texels: smooth them
        tiles = self.tile_layer.instances
        size = tiles["size"][index].astype(np.float64)
        _, end_zoom = self.canvas.camera.fit_params(0, 0, *size, margin=self.canvas.fit_margin)
        self.dive = _Dive(index, self.tile_layer, self.canvas, end_zoom)
        self._show_photo(picture, tiles["pos"][index], size, tiles["rotation"][index], 0.0)
        self._locked = True
        if self._job is not None:  # preparing (a random mosaic's start): match once ready
            self._queued = picture
            self.status.setText("Preparing tiles: this dive matches once they are ready…")
        else:
            self._match(picture, "dive", "Matching the next level…")
        dive, duration = self.dive, float(self.settings.dive_seconds)

        def step(t: float) -> bool:
            if self.dive is not dive:
                return False
            e = min(t / duration, 1.0)
            self._dive_frame(e)
            if e < 1.0:
                return True
            dive.done = True
            if dive.matched is not None:
                self._arrive(dive.matched)
            elif self._queued is None:  # else still preparing (and saying so)
                self.status.setText("Matching the next level…")
            return False

        self.canvas.add_animation(step)
        self._refresh()

    def _dive_frame(self, e: float) -> None:
        """The dive at e (0 to 1): the tile straightens, untints and fills the view while
        the other tiles fly outward."""
        dive, canvas = self.dive, self.canvas
        m = float(ease_in_out_cubic(e))
        tiles = self.tile_layer.instances
        fly = FLY_OUT * float(ease_out_cubic(e))
        tiles["pos"] = dive.pos + (dive.pos - dive.center) * fly
        i = dive.index
        tiles["pos"][i] = dive.center
        tiles["rotation"][i] = dive.rotation * (1 - m)
        tiles["offset"][i] = dive.offset * (1 - m)
        self.tile_layer.mark_dirty()
        photo = self.photo_layer.instances
        photo["rotation"] = dive.rotation * (1 - m)
        photo["alpha"] = min(e / PHOTO_IN, 1.0)
        self.photo_layer.mark_dirty()
        zoom = dive.zoom0 * (dive.zoom1 / dive.zoom0) ** m  # geometric: even zoom speed
        # The tile moves on screen from where it was to the middle as the view zooms.
        canvas.set_view(dive.center - dive.screen0 * (1 - m) / zoom, zoom)

    # Window

    def _refresh(self) -> None:
        running = self.running
        self.start_button.setText("Stop" if running else "Start")
        self.new_start_button.setVisible(running)
        self.new_start_button.setEnabled(self.can_start_anew)
        self.start_button.setToolTip("End this session." if running else START_TIP)
        for form in (*self.forms.values(), self.slicer_form):
            form.setEnabled(not running)
        self.settings_note.setVisible(running)
        self.progress.setVisible(self._followed)
        if not self._followed:
            self._show_sub(False)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.canvas and self._locked and event.type() in LOCKED_EVENTS:
            return True  # no panning or zooming while diving or fading
        return super().eventFilter(watched, event)

    def toggle_full_screen(self) -> None:
        if self.isFullScreen():
            self._leave_full_screen()
        else:
            self.panel.hide()
            self.showFullScreen()

    def _leave_full_screen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self.panel.show()

    def shutdown(self) -> None:
        """Stop all background work (the window is closing)."""
        self.stop()
        self.zoom_detail.shutdown()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)

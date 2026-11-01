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
  zooms until it fills it, and every other tile leaves the view as the
  Tiles leave by setting has it (core/dive_styles.py; `_dive_frame`).
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

The panel has the Session controls on top and two tabs below: Settings and
Library (ui/library_panel.py's LibraryPanel, the same as on the main
window's Tiles tab, on the app session's library). While a session runs the
library can't be changed: its search index holds the photos it started with.

Auto-dive (the Auto-Dive button, A) dives on its own until stopped (the
button, A, Esc, or a click or the wheel on the mosaic): each mosaic shows
whole for the Dwell time, then a tile in view is picked (`pick_tile`), the
view moves to it and zooms in (`_approach`, the tile outlined), rests on it
for a moment, and dives. Its settings and the dive's apply as they change,
even while a session runs: they only pace it.

With Auto-dive moves set to Continuously, it never stops (`_start_flow`):
the view zooms evenly from each mosaic onto a tile picked as the mosaic
appears (core/dive_path.py's FlowPath: one leg per mosaic), while that
tile's photo is read and its mosaic made in the background (`_flow_match`),
its layers built and staged to the GPU ahead (`_flow_prepare`,
`_build_layer`). As the tile grows its photo fades in over it, then into
its own mosaic, drawn inside it (its tiles placed on the tile, `_moved`),
done when the tile fills Mosaic appears at of the view. When the tile
fills the view that mosaic takes over in its own world, looking just the
same (`_flow_arrive`); the last one goes into that world too, under it,
and fades out. A mosaic not made in time slows the zoom to a stop where
its photo would start to show, until it is. Loops are made in steps only
(for now).

With Loop on, auto-dive comes back to where it started (core/infinite.py's
Loop): the first tile it dives into is home; from the bottom of the range
on, while each mosaic dwells, several tiles' mosaics are made ahead
(`_look_ahead`) and it dives into the one home fits best, already matched;
once a level holds home it heads there, and that dive lands on the loop's
first level. From then on every pass replays the same levels and tiles, so
one pass is a seamless loop. A manual dive, Stop, New Start or turning Loop
off drops the loop.
"""

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from skitter.core.dive_path import CENTER_FIRST, MIN_SECONDS, FlowPath, View, flow_rate
from skitter.core.dive_styles import DiveStyle, DiveTiles, dive_seed, pick_style
from skitter.core.easing import ease_in_out_cubic
from skitter.core.infinite import (
    AUTO,
    DIVE,
    DIVE_THREADS,
    LIVE,
    MATCHING,
    MOSAIC,
    InfiniteSession,
    InfiniteSettings,
    Level,
    Loop,
    Lure,
    TileView,
    display_copy,
    pick_weights,
)
from skitter.core.matching.mosaic import MosaicMethod
from skitter.core.matching.result import MatchCancelled
from skitter.core.slicing import SlicingCancelled, SlicingPlan
from skitter.core.slicing.layout import CANVAS_WIDTH
from skitter.ui import preferences
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.jobs import Job
from skitter.ui.library_panel import LibraryPanel
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.render.tile_textures import TileTextures, detail_request, thumbnail_atlas
from skitter.ui.render.zoom_detail import DETAIL_PAGES, DETAIL_SIDE, ZoomDetail
from skitter.ui.steps.base import side_panel, view_and_panel
from skitter.ui.steps.tile_picker import upper_closure
from skitter.ui.style import TAB_STYLE, muted, progress_bar, show_progress
from skitter.ui.widgets.param_form import ParamForm

SETTINGS_KEY = "infinite/settings"  # preferences: the last session's settings (JSON)
PLAN_KEY = "infinite/plan"  # and its slicers' settings
PHOTO_IN = 0.2  # share of the dive over which the photo fades in over its tile
WHOLE = 0.98  # a tile showing this much of itself lies on top (auto-dive prefers those)
PIXELATE_ABOVE = 4.0  # texels this large show as squares (not while diving: tiles blow up)
LIBRARY_TAB = 1  # the panel's tabs: Settings, Library
LIBRARY_LOCKED = "A session searches the photos it started with: stop it to change them."
START_TIP = "Start a session from a random photo of the library."
# Continuous auto-dive. As the tile it heads for grows, the tile's photo fades in over it,
# then fades into its own mosaic, drawn inside it: each over FADE_ZOOM of zoom, done when
# the tile fills Mosaic appears at of the view. After the switch, the last mosaic (around
# and under the new one) stays until the new one covers the view, then fades out of its gaps
# over LAST_OUT_ZOOM. A leg
# slows to a stop (over EASE_SECONDS) where the photo would start to show if the next
# mosaic isn't made by then. Continuously by hand, the view creeps in between legs, at the
# same rate, slowing to a stop CREEP_ROOM of zoom before a base tile would start to show
# its photo, so a leg into one has time to make its mosaic.
FADE_ZOOM = 1.8
LAST_OUT_ZOOM = 1.6
EASE_SECONDS = 1.0
CREEP_ROOM = FADE_ZOOM
CREEP_STATUS = "Double-click a tile to head into it (drag to steer)."
PHOTO_DISPLAY = 2.0  # a photo is shown at up to this many times the view's pixels
MAX_FRAME_STEP = 0.1  # seconds the zoom moves on in one frame, at most
HIGHLIGHT_COLOR = (1.0, 1.0, 1.0)  # the outline of the tile auto-dive heads for
HIGHLIGHT_PX = 3.0
AUTO_TIP = "Dive on and on by itself until stopped (A; Esc, a click or the wheel stop it too)."
# Input on the mosaic that stops auto-dive (the viewer takes over).
STOPS_AUTO = (
    QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick, QEvent.Type.Wheel,
)  # fmt: skip
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

    def __init__(self, index: int, layer: SpriteLayer, canvas: MosaicCanvas, end_zoom: float,
                 style: DiveStyle, seed: int):  # fmt: skip
        tiles = layer.instances
        self.index = index
        self.pos = tiles["pos"].astype(np.float64)
        self.center = self.pos[index].copy()
        self.rotation = float(tiles["rotation"][index])
        self.offset = tiles["offset"][index].copy()
        self.size = tiles["size"][index].copy()
        cam = canvas.camera
        self.zoom0, self.zoom1 = float(cam.zoom), float(end_zoom)
        self.screen0 = (self.center - cam.center) * cam.zoom  # the tile's offset on screen
        self.style = style  # how the other tiles leave
        self.tiles = DiveTiles(self.pos, tiles["size"], tiles["rotation"], tiles["alpha"], index,
                               cam.viewport / self.zoom0, cam.viewport / self.zoom1, seed,
                               cam.turn((0.0, 1.0)))  # fmt: skip
        self.matched: Level | None = None  # the next level, once matched
        self.done = False  # the animation has finished
        # The next level's, made while the dive goes on (see _dive_matched).
        self.textures: TileTextures | None = None
        self.layer: SpriteLayer | None = None  # staged
        self.layer_pages: np.ndarray | None = None  # the textures' pages it was built from


class _Flow:
    """Continuous auto-dive's leg through the shown level, onto tile index."""

    def __init__(self, level: Level, index: int, tile: TileView, path: FlowPath):
        self.level, self.index, self.tile, self.path = level, index, tile, path
        self.u = 0.0  # progress along the path
        self.speed = 1.0  # eased down to 0 (and up again) while the next level is not made
        self.hold = 1.0  # where it waits if the next level isn't ready (see _flow_advance)
        self.next: Level | None = None  # the tile's mosaic, once made
        self.picture: np.ndarray | None = None  # the tile's photo, as shown (display size)
        self.seconds = 0.0  # how long making it took
        self.waiting = False  # its making waits for the search index (a random mosaic's start)
        # Made ahead, and staged, so nothing stalls a frame: the photo's layer, the next
        # level's textures, its tile layer from their thumbnails (shown inside the tile:
        # its tiles are small there) and, once its full-size crops are in, from those.
        self.photo: SpriteLayer | None = None
        self.textures: TileTextures | None = None
        self.layer: SpriteLayer | None = None  # the one shown inside the tile
        self.full: SpriteLayer | None = None  # full-size, until it takes layer's place
        self.placed = False  # layer is on the canvas, inside the tile
        # This level's tiles lying on the tile (a pile's): they fade away as its photo
        # fades in, so it doesn't rise through them.
        self.over = upper_closure(level.scene, index)[1:]
        self.inside: np.ndarray | None = None  # the next level's tiles within its canvas


def _fade(x: float) -> float:
    """0 to 1, eased, for x from 0 to 1 (clamped)."""
    return float(ease_in_out_cubic(min(max(x, 0.0), 1.0)))


def _moved(instances: np.ndarray, origin, to, scale: float, turn: float) -> np.ndarray:
    """Sprite instances moved from one world to another: what was at origin goes to `to`,
    scaled and turned (radians) about it."""
    moved = instances.copy()
    c, s = math.cos(turn), math.sin(turn)
    offset = (instances["pos"].astype(np.float64) - origin) * scale
    moved["pos"] = np.asarray(to) + np.stack(
        [c * offset[:, 0] - s * offset[:, 1], s * offset[:, 0] + c * offset[:, 1]], axis=1
    )
    moved["size"] = instances["size"] * scale
    moved["rotation"] = instances["rotation"] + turn
    return moved


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
        # A dive's (picture, lure, tile), matched once prepared
        self._queued: tuple[np.ndarray, Lure | None, TileView] | None = None
        self.loop: Loop | None = None  # auto-dive's loop, recording or replaying
        self._loop_pos = -1  # the shown level's index in loop.levels (-1: not in it)
        self._loop_next = -1  # where the dive under way lands in it (-1: a new level)
        self._ahead: list[tuple[int, Level]] | None = None  # look-ahead's, for self.level
        # Jobs have no parent (it would keep each, and its inputs, for the window's life):
        # the last two are kept here, so each goes once a later one has started.
        self._recent_jobs: list[Job] = []
        self._pool: ProcessPoolExecutor | None = None
        self._locked = False  # pan and zoom off (diving, fading)
        self._fade_token: object | None = None
        self.auto = False  # auto-dive is on
        self._start_when_open = False  # Start was pressed while the library opened
        self._auto_token: object | None = None  # the auto-dive step under way
        self._highlight: SpriteLayer | None = None  # the outline of auto-dive's next tile
        self._flow: _Flow | None = None  # continuous dive's leg under way
        self._flow_token: object | None = None  # its animation (and by hand, the creep's)
        self._flow_last: float | None = None  # the animation's time at the previous frame
        self._creep_speed = 0.0  # by hand, between legs: the creep's, as a share of the rate
        self._creep_left = 0.0  # its log zoom to go, as of the last frame
        self._last_layer: SpriteLayer | None = None  # the last level's, fading after a switch
        self._last_alpha: np.ndarray | None = None  # its tiles' opacity as the switch left it
        self._last_zoom: float | None = None  # the view's zoom once the new level covered it
        self._last_clock = 0.0  # seconds since
        self._tile_swap: SpriteLayer | None = None  # full-size tiles, staging, to swap in
        self._layer_jobs: list[Job] = []  # building tile layers in the background
        self._approaching = False  # auto-dive's view moving in on a tile
        self._swap_waiting = False  # full-size tiles came in meanwhile: swap them in after
        self._rng = np.random.default_rng()

        self.canvas = MosaicCanvas()
        self.canvas.fit_margin = 1.0  # a picture fills the view, as a tile dived into does
        self.canvas.pixelate_above = PIXELATE_ABOVE
        self.canvas.double_clicked.connect(self._on_double_click)
        self.canvas.installEventFilter(self)
        self.zoom_detail = ZoomDetail(self.canvas, self)
        self.detail_reported.connect(self._on_detail)

        # The panel: Session on top, then the Settings and Library tabs.
        self.library = LibraryPanel(session, lambda: float(self.settings.tile_aspect))
        # A tab bar over a stack, as Animate's phases: a QTabWidget whose bar has a style
        # sheet keeps its pages from following the window's font.
        self.tab_bar = QTabBar()
        self.tab_bar.setExpanding(True)
        self.tab_bar.setDocumentMode(True)
        self.tab_bar.setStyleSheet(TAB_STYLE)
        self.pages = QStackedWidget()
        settings = side_panel(*self._build_settings_groups(), resizable=True)
        library = side_panel(self.library, resizable=True)
        for title, page in (("Settings", settings), ("Library", library)):
            self.tab_bar.addTab(title)
            self.pages.addWidget(page)
        self.tab_bar.currentChanged.connect(self._on_tab)
        self.panel = QWidget()
        column = QVBoxLayout(self.panel)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self._build_session_group())
        column.addWidget(self.tab_bar)
        column.addWidget(self.pages, stretch=1)
        self.splitter = view_and_panel(self.canvas, self.panel, "infinite")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.splitter)
        session.library_changed.connect(self._on_library_changed)
        session.busy_changed.connect(self._refresh)
        QShortcut(QKeySequence("F11"), self, self.toggle_full_screen)
        QShortcut(QKeySequence("Esc"), self, self._on_escape)
        QShortcut(QKeySequence("A"), self, self.toggle_auto)
        self.resize(1400, 900)
        self._refresh()

    # Construction

    def _build_settings_groups(self) -> list[QGroupBox]:
        """Mosaic (with the slicer's own settings), Matching, Dive and Auto-dive."""
        self.forms = {name: ParamForm() for name in ("mosaic", "matching", "dive", "auto")}
        for form in self.forms.values():
            form.changed.connect(self._on_setting_changed)
        self.slicer_form = ParamForm()
        self.settings_note = muted(QLabel(
            "Mosaic and matching settings stay as they are for the whole session; dive and "
            "auto-dive settings apply as they change."
        ))  # fmt: skip
        self._target_forms()
        groups = []
        for title, name in (("Mosaic", "mosaic"), ("Matching", "matching"), ("Dive", "dive"),
                            ("Auto-dive", "auto")):  # fmt: skip
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
        for name, only in (("mosaic", MOSAIC), ("matching", MATCHING), ("dive", DIVE),
                           ("auto", AUTO)):  # fmt: skip
            self.forms[name].set_target(self.settings, only)
        self.slicer_form.set_target(self.plan.slicers[self.settings.slicer])

    def set_settings(self, settings: InfiniteSettings) -> None:
        """Edit these settings from now on (the next session starts with them)."""
        self.settings = settings
        self._target_forms()
        self.library.refresh_stats()  # its tile shape

    def _build_session_group(self) -> QGroupBox:
        self.start_button = QPushButton("Start")
        self.start_button.setToolTip(START_TIP)
        self.start_button.clicked.connect(self.toggle_session)
        self.new_start_button = QPushButton("New Start")
        self.new_start_button.setToolTip(
            "Start again from another random photo or mosaic, with the same settings."
        )
        self.new_start_button.clicked.connect(self.new_start)
        self.auto_button = QPushButton("Auto-Dive")
        self.auto_button.setCheckable(True)
        self.auto_button.setToolTip(AUTO_TIP)
        self.auto_button.toggled.connect(self.set_auto)
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
            "pan, use the wheel to zoom (diving continuously: the view zooms in by itself). "
            "A: auto-dive. F11: full screen."
        ))  # fmt: skip
        group = QGroupBox("Session")
        layout = QVBoxLayout(group)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.new_start_button)
        layout.addLayout(buttons)
        layout.addWidget(self.auto_button)
        for widget in (self.status, self.progress, self.sub_status, self.sub_progress,
                       self.level_label, help_text):  # fmt: skip
            layout.addWidget(widget)
        return group

    def _on_setting_changed(self, name: str) -> None:
        if name in LIVE:  # a running session follows them: keep them for the next one too
            save_settings(self.settings, self.plan)
        if name == "loop" and not self.settings.loop:
            self._drop_loop()
        if name == "motion" and self.running:  # the other way of moving, from where it is
            self._cancel_auto_step()
            if self.level is not None and not self.busy:  # at rest (a creep's may not be)
                self.canvas.pixelate_above = PIXELATE_ABOVE
            self._auto_next()
        if name == "tile_aspect":
            self.library.refresh_stats()
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
        """Matching a level, diving or fading: no new dive yet (preparing or looking
        ahead allow one)."""
        matching = self._job is not None and self._job_kind in ("first", "dive")
        moving = self.dive is not None or self._flow is not None
        return matching or moving or self._fade_token is not None

    @property
    def _continuous(self) -> bool:
        """Dives move continuously in this session."""
        return self.running and self.settings.motion == "flow"

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
        """A new session with the current settings, from its start (Start with). With the
        library not open yet, once it is (it opens in the background)."""
        app = self.app_session
        library = app.library
        if library is None:
            app.open_library_in_background()
            self._start_when_open = True
            self.status.setText("Opening the library: the session starts once it's open…")
            self._refresh()
            return
        self._start_when_open = False
        if not len(library):
            self.status.setText("The tile library has no photos: add some on the Library tab.")
            self.tab_bar.setCurrentIndex(LIBRARY_TAB)
            return
        save_settings(self.settings, self.plan)
        base = next((m for m in app.project.match_methods.values()
                     if isinstance(m, MosaicMethod)), None)  # fmt: skip
        self.infinite = InfiniteSession(library, self.settings, self.plan, base, app.matcher)
        self._pool = ProcessPoolExecutor(os.cpu_count())
        for _ in range(os.cpu_count() or 1):  # start the workers while the first level matches
            self._pool.submit(int)
        self.new_start()

    def new_start(self) -> None:
        """Show the session's start: another random photo or mosaic."""
        if not self.can_start_anew:
            return
        self._cancel_auto_step()
        self._drop_loop()
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
        if self._job is None:  # once prepared, only its variety is new (the rest is cached)
            self._prepare()
        self._auto_next()

    def stop(self) -> None:
        """End the session: stop its work and clear the view."""
        self.set_auto(False)
        self._stop_flow()
        if self._job is not None:
            self._job.cancel()
            self._job.wait()
            self._job = None
        self._drop_loop()
        if self.dive is not None:
            self._discard_dive(self.dive)
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

    def _match(self, picture: np.ndarray, kind: str, message: str,
               lure: Lure | None = None, source: TileView | None = None) -> None:  # fmt: skip
        """Match a level in the background: kind "first" (a random photo: every core, it
        may build the index) or "dive" (during the animation: fewer search threads)."""
        infinite = self.infinite
        threads = None if kind == "first" else DIVE_THREADS

        def work(progress, cancelled):
            return infinite.level(picture, progress, cancelled, self.detail_reported.emit,
                                  threads, lure, source)  # fmt: skip

        self._start_job(work, kind, message, self._on_matched)

    def _prepare(self) -> None:
        """Build the search index in the background (a random mosaic shows meanwhile)."""
        infinite, start = self.infinite, self.level  # the random mosaic: its variety too

        def work(progress, cancelled):
            infinite.prepare(progress, cancelled, self.detail_reported.emit, start)

        self._start_job(work, "prepare", "Preparing tiles: dives match once ready…",
                        self._on_prepared)  # fmt: skip

    def _start_job(self, work, kind: str, message: str, on_done) -> None:
        job = Job(work, (MatchCancelled, SlicingCancelled))
        for old in self._recent_jobs[:-1]:
            old.join()  # long stopped; its thread too, to be sure
        self._recent_jobs = [*self._recent_jobs[-1:], job]
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
        return self._job is not None and self._job_kind not in ("dive", "flow")

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
        self._job = None
        self.status.setText("Ready." if self.dive is None else "Matching the next level…")
        queued, self._queued = self._queued, None
        if queued is not None:
            picture, lure, tile = queued
            self._match(picture, "dive", "Matching the next level…", lure, tile)
        elif self._flow is not None and self._flow.waiting:
            self._flow_match()
        elif self.level is not None and self.level.variety is None and self.dive is None:
            self._prepare()  # New Start meanwhile: the mosaic shown now needs its variety
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
            self._dive_matched(level)
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
        self._swap_waiting = False  # that level's tiles
        if self._tile_swap is not None:
            self.canvas.unstage(self._tile_swap)
            self._tile_swap = None
        self.zoom_detail.show(None, None, None)
        if self.tile_layer is not None:
            self.canvas.remove_layer(self.tile_layer)
            self.tile_layer = None
        if self.textures is not None:
            self.textures.changed.disconnect(self._on_textures_changed)
            self.textures.close()
            self.textures = None

    def _arrive(self, level: Level) -> None:
        """Move to a level: the photo covers its canvas as it covered the tile (or the
        view, at the start), the view refits to it, and it fades into the mosaic. Its tile
        layer: the one the dive made for it, if it did."""
        dive, self.dive = self.dive, None
        width, height = float(level.ctx.width), float(level.ctx.height)
        photo = self.photo_layer.instances
        photo["pos"], photo["size"], photo["rotation"] = (width / 2, height / 2), (width, height), 0
        self.photo_layer.mark_dirty()
        textures = layer = None
        if dive is not None and dive.textures is not None:
            textures, layer = dive.textures, dive.layer  # still staging: the rest goes up now
            textures.changed.disconnect(self._dive_textures_ready)
        self._show_level(level, textures, layer)
        if layer is not None and dive.layer_pages is not textures.pages:
            self._on_textures_changed()  # its full-size tiles came while it was built
        self._loop_arrived(level)
        self.status.setText("Ready.")
        show_progress(self.progress, 1.0)
        self._show_sub(False)
        self._fade_photo()

    def _show_level(self, level: Level, textures: TileTextures | None = None,
                    layer: SpriteLayer | None = None) -> None:  # fmt: skip
        """Show a level's tiles (under the photo, if any), fitted to the view. textures,
        layer: its own, made ahead (continuous auto-dive), else made now."""
        self._clear_level()
        self.level = level
        self.canvas.fit_to(0, 0, float(level.ctx.width), float(level.ctx.height))
        self.canvas.pixelate_above = PIXELATE_ABOVE
        if textures is None:
            textures = TileTextures(level.scene, self.infinite.library, self, pool=self._pool)
        self.textures = textures
        self.textures.changed.connect(self._on_textures_changed)
        self._new_tile_layer(layer)

    def _new_tile_layer(self, layer: SpriteLayer | None = None) -> None:
        """The level's tiles below the photo, from the best textures there are (layer:
        made from them already)."""
        index = 0
        if self.tile_layer is not None:
            index = self.canvas.layers.index(self.tile_layer)
            self.canvas.remove_layer(self.tile_layer)
        if layer is None:
            layer = SpriteLayer(self.textures.pages, self.textures.instances(),
                                detail_pages=DETAIL_PAGES, detail_side=DETAIL_SIDE)  # fmt: skip
        self.tile_layer = self.canvas.add_layer(layer, index)
        self.zoom_detail.show(layer, self.level.scene, self.infinite.library)

    def _on_textures_changed(self) -> None:
        """Full-size tiles arrived: swap them in (unless a dive is moving them; while
        auto-dive zooms in, once it rests: uploading them stalls a frame)."""
        if self._approaching:
            self._swap_waiting = True
        elif self.tile_layer is not None and (self._flow_token is not None or self.dive is None):
            self._build_layer(self.textures, None)  # built aside and staged, then swapped

    def _end_approach(self) -> None:
        self._approaching = False
        if self._swap_waiting:
            self._swap_waiting = False
            self._on_textures_changed()

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
            self._auto_next()
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
            self._drop_loop()  # loops are auto-dive's: a dive by hand leaves it
            if self.settings.motion == "flow":
                self._flow_into(index)
            else:
                self.dive_into(index)

    def tile_view(self, index: int) -> TileView:
        """Tile index of the shown level, as a TileView."""
        scene = self.level.scene
        return TileView(int(scene.slot[index]), tuple(float(v) for v in scene.rect[index]),
                        bool(scene.mirrored[index]))  # fmt: skip

    def dive_into(self, index: int, next_level: Level | None = None,
                  lure: Lure | None = None) -> None:  # fmt: skip
        """Dive into tile index of the level: its photo becomes the next one. next_level:
        already made (a loop's, a look-ahead's): shown when the dive ends, nothing is
        matched. Else it is matched meanwhile, with lure (a loop's home) if given."""
        infinite = self.infinite
        tile = self.tile_view(index)
        picture = infinite.picture(tile)
        self._remove_highlight()
        self.zoom_detail.show(None, None, None)  # tiles move: nothing to sharpen
        self.canvas.pixelate_above = None  # tiles grow far past their texels: smooth them
        tiles = self.tile_layer.instances
        size = tiles["size"][index].astype(np.float64)
        _, end_zoom = self.canvas.camera.fit_params(0, 0, *size, margin=self.canvas.fit_margin)
        seed = dive_seed(index, tile.slot, len(tiles))
        style = pick_style(str(self.settings.dive_style), seed)
        self.dive = _Dive(index, self.tile_layer, self.canvas, end_zoom, style, seed)
        self._show_photo(picture, tiles["pos"][index], size, tiles["rotation"][index], 0.0)
        self._locked = True
        if next_level is not None:
            self._dive_matched(next_level)
        elif self._job is not None:  # preparing (a random mosaic's start): match once ready
            self._queued = picture, lure, tile
            self.status.setText("Preparing tiles: this dive matches once they are ready…")
        else:
            self._match(picture, "dive", "Matching the next level…", lure, tile)
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
        the other tiles leave the view (Tiles leave by: the dive's style)."""
        dive, canvas = self.dive, self.canvas
        m = float(ease_in_out_cubic(e))
        tiles = self.tile_layer.instances
        frame = dive.style.frame(dive.tiles, e)
        tiles["pos"], tiles["size"] = frame.pos, frame.size
        tiles["rotation"], tiles["alpha"] = frame.rotation, frame.alpha
        if frame.shade is not None:
            tiles["tint"][:, :3] = 0.0
            tiles["tint"][:, 3] = frame.shade
            tiles["tint"][dive.index] = 0.0
        i = dive.index
        tiles["pos"][i], tiles["size"][i] = dive.center, dive.size
        tiles["alpha"][i] = dive.tiles.alpha[i]
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

    def _dive_matched(self, level: Level) -> None:
        """The dive's next level is made: while the dive goes on, its thumbnails are packed
        and its tile layer built (in the background) and staged, so arriving only swaps it
        in (made then, it stalled a frame about 50 ms)."""
        dive, library = self.dive, self.infinite.library
        dive.matched = level

        def work(progress, cancelled):
            return thumbnail_atlas(level.scene, library), detail_request(level.scene, library)

        def packed(made) -> None:
            if self.dive is not dive:
                return  # arrived (its tiles made then) or stopped meanwhile
            atlas, request = made
            dive.textures = TileTextures(level.scene, library, self, pool=self._pool,
                                         atlas=atlas, request=request)  # fmt: skip
            dive.textures.changed.connect(self._dive_textures_ready)
            self._build_layer(dive.textures, None, dive)

        self._layer_job(work, packed)

    def _dive_textures_ready(self) -> None:
        """The dive's next level's full-size crops are in: build its tile layer from them."""
        dive = self.dive
        if dive is not None and dive.textures is not None:
            self._build_layer(dive.textures, None, dive)

    def _discard_dive(self, dive: _Dive) -> None:
        """Let go of what a dive made for its next level, not to be shown."""
        if dive.layer is not None:
            self.canvas.unstage(dive.layer)
        if dive.textures is not None:
            dive.textures.changed.disconnect(self._dive_textures_ready)
            dive.textures.close()
        dive.textures = dive.layer = None

    def _swap_when_staged(self, layer: SpriteLayer) -> None:
        """Stage the shown level's new tile layer and swap it in once it is up (and no dive
        or zoom-in moves the tiles)."""
        self._tile_swap = layer
        self.canvas.stage(layer)

        def step(t: float) -> bool:
            if self._tile_swap is not layer:
                return False  # swapped or dropped meanwhile
            if not self.canvas.staged(layer) or self.dive is not None or self._approaching:
                return True
            self._tile_swap = None
            self._new_tile_layer(layer)
            return False

        self.canvas.add_animation(step)

    # Auto-dive

    def toggle_auto(self) -> None:
        self.set_auto(not self.auto)

    def set_auto(self, on: bool) -> None:
        """Turn auto-dive on (it starts from the current mosaic, once it is shown) or off
        (what it was doing stops; a dive under way finishes)."""
        on = bool(on) and self.running
        if on == self.auto:
            self._refresh()
            return
        self.auto = on
        if on:
            self._auto_next()
        elif self._flow is not None:  # a leg on its way: it goes on to its mosaic, then creeps
            self.status.setText("Auto-dive stops once this mosaic shows.")
        else:
            self._cancel_auto_step()
            if self.running and not self.busy:
                self.status.setText("Ready.")
        self._refresh()

    def _cancel_auto_step(self) -> None:
        self._auto_token = None
        self._stop_flow()
        self._remove_highlight()
        if self._approaching:
            self._end_approach()

    def _auto_next(self) -> None:
        """Once the mosaic shows (no dive or fade under way): auto-dive's next round (dwell,
        then head for a tile); continuously, its next leg, or by hand the creep."""
        if not self.running or self.level is None or self.busy:
            return
        if self.settings.motion == "flow":
            if self._flow_token is None:
                self._start_flow()
            elif self.auto:  # from the creep, at its speed
                self._flow_leg(self.level, speed=self._creep_speed)
                self._refresh()
            return
        if not self.auto:
            return
        token = self._auto_token = object()
        self._look_ahead()
        dwell = float(self.settings.dwell_seconds)
        self.status.setText(f"Auto-dive: next dive in {dwell:g} s." if dwell else "Auto-dive.")
        QTimer.singleShot(round(dwell * 1000), self, lambda: self._approach(token))

    def pick_tile(self) -> int:
        """Auto-dive's next tile: at random among the tiles in view (Next tile: anywhere,
        or among the tenth nearest the middle) that lie on top (a pile's: nothing on them;
        else those mostly shown), varied ones likelier (Favor varied tiles: see
        core/infinite.py's Looks)."""
        scene, cam = self.level.scene, self.canvas.camera
        pos = scene.center
        half = cam.viewport / (2 * cam.zoom)
        in_view = np.all(np.abs(pos - cam.center) <= half, axis=1)
        share = scene.visible_share
        whole, shown = share >= WHOLE, share >= 0.5
        for candidates in (in_view & whole, in_view & shown, in_view, shown):
            pool = np.flatnonzero(candidates)
            if len(pool):
                break
        else:
            pool = np.arange(len(scene))
        if self.settings.next_tile == "middle" and len(pool) > 1:
            distance = np.linalg.norm(pos[pool] - cam.center, axis=1)
            pool = pool[np.argsort(distance)[: max(1, len(pool) // 10)]]
        variety = self.level.variety
        weights = pick_weights(None if variety is None else variety[pool],
                               float(self.settings.variety) / 100)  # fmt: skip
        return int(self._rng.choice(pool, p=weights))

    def _approach(self, token: object) -> None:
        """Move to a tile and zoom in on it (to Zoom in to tiles across, over Zoom-in
        time), rest on it (Pause on tile), then dive."""
        if token is not self._auto_token or self.busy or self.level is None:
            return
        if self._job is not None and self._job_kind == "ahead":  # wait for it
            QTimer.singleShot(100, self, lambda: self._approach(token))
            return
        index = self._next_tile()
        settings, canvas = self.settings, self.canvas
        cam = canvas.camera
        center = self.level.scene.center[index].astype(np.float64)
        tile_width = float(self.level.ctx.tile_size[0])
        across = cam.viewport[0] / (float(settings.approach_tiles) * tile_width)
        zoom0 = float(cam.zoom)
        zoom1 = float(np.clip(max(across, zoom0), cam.min_zoom, cam.max_zoom))  # never out
        screen0 = (center - cam.center) * zoom0
        duration = float(settings.approach_seconds)
        if settings.highlight:
            self._show_highlight(index)
        self.status.setText("Auto-dive: heading for a tile…")
        self._approaching = True

        def step(t: float) -> bool:
            if token is not self._auto_token:
                return False  # stopped: the view stays where it got to
            e = 1.0 if duration <= 0 else min(t / duration, 1.0)
            m = float(ease_in_out_cubic(e))
            zoom = zoom0 * (zoom1 / zoom0) ** m
            canvas.set_view(center - screen0 * (1 - m) / zoom, zoom)
            if self._highlight is not None:
                self._highlight.instances["alpha"] = min(1.0, 2 * e)
                self._highlight.mark_dirty()
            if e < 1.0:
                return True
            self._end_approach()
            pause = round(float(settings.pause_seconds) * 1000)
            QTimer.singleShot(pause, self, lambda: self._auto_dive(token, index))
            return False

        canvas.add_animation(step)

    def _auto_dive(self, token: object, index: int) -> None:
        if token is not self._auto_token or self.busy or self.level is None:
            return
        self._auto_token = None
        if self.settings.loop:
            self._loop_dive(index)
        else:
            self.dive_into(index)

    # Continuous auto-dive

    def _start_flow(self, index: int | None = None, speed: float = 1.0) -> None:
        """Start the never-ending zoom from the view as it is (see core/dive_path.py):
        auto-dive's legs, a leg onto tile index (a double-click), or by hand the creep
        (from rest: speed is a leg's, as a share of the rate)."""
        self._drop_loop()  # loops are made in steps (for now)
        token = self._flow_token = object()
        self._flow_last = None
        self.canvas.pixelate_above = None  # tiles grow far past their texels: smooth them
        if self.auto or index is not None:
            self._flow_leg(self.level, index, speed)
        else:
            self._creep_speed = 0.0
            self.status.setText(CREEP_STATUS)

        def step(t: float) -> bool:
            if self._flow_token is not token:
                return False
            dt = 0.0 if self._flow_last is None else t - self._flow_last
            self._flow_last = t
            dt = min(max(dt, 0.0), MAX_FRAME_STEP)  # a stalled frame doesn't leap ahead
            self._flow_advance(dt)
            if self._flow_token is token and self._flow is None and self._creep_done():
                self._flow_token = None  # at rest, nothing left to show: no more frames
            return self._flow_token is token

        self.canvas.add_animation(step)
        self._refresh()

    def _stop_flow(self) -> None:
        """Stop where the view is: the next mosaic's making stops, what was made ahead for
        it goes, the last level's leftovers too (a leg's, or the creep's)."""
        moving = self._flow_token is not None
        flow, self._flow, self._flow_token = self._flow, None, None
        self._creep_speed = 0.0
        if flow is None and not moving and self._last_layer is None and self._tile_swap is None:
            return  # nothing continuous under way (a step dive may be)
        for job in self._layer_jobs:
            job.cancel()
        if flow is not None:
            if self._job is not None and self._job_kind == "flow":
                self._job.cancel()
                self._job.wait()
                self._job = None
            if len(flow.over) and self.tile_layer is not None and flow.level is self.level:
                self.tile_layer.instances["alpha"][flow.over] = 1.0  # back on the tile
                self.tile_layer.mark_dirty()
            self._flow_discard(flow)
        if self._last_layer is not None:
            self.canvas.remove_layer(self._last_layer)
            self._last_layer = None
        if self._tile_swap is not None:  # full-size tiles staged: in now, as they'd be at rest
            swap, self._tile_swap = self._tile_swap, None
            self._new_tile_layer(swap)
        if self.photo_layer is not None:  # a step dive's photo, faded
            self.photo_layer.visible = False
        if self.level is not None:
            self.canvas.pixelate_above = PIXELATE_ABOVE
        self._refresh()

    def _flow_discard(self, flow: _Flow) -> None:
        """Let go of what a leg made ahead for its next level, not to be shown now."""
        for layer in (flow.photo, flow.layer, flow.full):
            if layer is None:
                continue
            if layer in self.canvas.layers:
                self.canvas.remove_layer(layer)
            else:
                self.canvas.unstage(layer)
        if flow.textures is not None:
            flow.textures.changed.disconnect(self._flow_textures_ready)
            flow.textures.close()
        flow.photo = flow.layer = flow.full = flow.textures = None

    def _flow_leg(self, level: Level, index: int | None = None, speed: float = 1.0) -> None:
        """A leg through level, from the view as it is (moving at speed) onto tile index (by
        hand: double-clicked), or one picked now."""
        if index is None:
            index = self.pick_tile()
        scene, cam, settings = level.scene, self.canvas.camera, self.settings
        size = scene.size[index].astype(np.float64)
        _, end_zoom = cam.fit_params(0, 0, *size, margin=self.canvas.fit_margin)
        start = View((float(cam.center[0]), float(cam.center[1])), float(cam.zoom),
                     float(cam.rotation))  # fmt: skip
        rate = flow_rate(int(settings.columns), float(settings.flow_seconds))
        path = FlowPath(start, scene.center[index], float(scene.rotation[index]), end_zoom,
                        cam.viewport, rate, str(settings.flow_path))  # fmt: skip
        self._flow = _Flow(level, index, self.tile_view(index), path)
        self._flow.speed = speed
        if self._job is not None:  # preparing (a random mosaic's start): once it's done
            self._flow.waiting = True
            self.status.setText("The next mosaic is made once tiles are ready…")
        else:
            self._flow_match()

    def _flow_match(self) -> None:
        """Make the leg's next level in the background: the tile's photo, its mosaic and
        that mosaic's thumbnails, packed."""
        flow, infinite = self._flow, self.infinite
        flow.waiting = False
        tile = flow.tile
        path = infinite.library.paths([tile.slot])[0]  # here: the database is this thread's
        ratio = self.canvas.devicePixelRatioF()
        longest = round(PHOTO_DISPLAY * max(self.canvas.camera.viewport) * ratio)

        def work(progress, cancelled):
            picture = infinite.picture(tile, path)
            level = infinite.level(picture, progress, cancelled, self.detail_reported.emit,
                                   DIVE_THREADS, None, tile)  # fmt: skip
            atlas = thumbnail_atlas(level.scene, infinite.library)
            request = detail_request(level.scene, infinite.library)
            return level, display_copy(picture, longest), (atlas, request)

        def done(result) -> None:
            self._job = None
            if self._flow is flow:
                flow.next, flow.picture, atlas = result
                flow.seconds = time.perf_counter() - self._job_started
                self._flow_prepare(flow, *atlas)
                self.status.setText("Auto-dive." if self.auto else "Diving.")
            self._refresh()

        self._start_job(work, "flow", "Making the next mosaic…", done)

    def _flow_prepare(self, flow: _Flow, atlas, request) -> None:
        """The next level is made: its photo's layer and its textures (reading full-size
        crops), its tile layer from the thumbnails (built in the background), all staged."""
        scene, i = flow.level.scene, flow.index
        photo = make_instances(1)
        photo["pos"], photo["size"] = scene.center[i], scene.size[i]
        photo["rotation"], photo["alpha"] = scene.rotation[i], 0.0
        flow.photo = SpriteLayer(flow.picture, photo)
        self.canvas.stage(flow.photo)
        flow.textures = TileTextures(flow.next.scene, self.infinite.library, self,
                                     pool=self._pool, atlas=atlas, request=request)  # fmt: skip
        flow.textures.changed.connect(self._flow_textures_ready)
        self._build_layer(flow.textures, flow)

    def _flow_textures_ready(self) -> None:
        """The next level's full-size crops are in: build its tile layer from them too."""
        flow = self._flow
        if flow is not None and flow.textures is not None:
            self._build_layer(flow.textures, flow)

    def _build_layer(self, textures: TileTextures, flow: _Flow | None,
                     dive: _Dive | None = None) -> None:  # fmt: skip
        """Build a tile layer from the textures' pages as they are, in the background (the
        pages are copied out to RGBA there), then stage it: for the leg's next level (the
        first: shown inside the tile; full-size ones swap in), the dive's (dive) or the
        shown one (flow None, or the leg moved on meanwhile: it swaps in once staged)."""
        pages, instances = textures.pages, textures.instances()

        def work(progress, cancelled):
            return SpriteLayer(pages, instances, detail_pages=DETAIL_PAGES,
                               detail_side=DETAIL_SIDE)  # fmt: skip

        def built(layer: SpriteLayer) -> None:
            if textures.pages is not pages:  # newer pages: their own layer follows
                return
            if dive is not None:
                if self.dive is not dive or dive.textures is not textures:
                    return  # arrived meanwhile (its tiles made then) or stopped
                if dive.layer is not None:
                    self.canvas.unstage(dive.layer)
                dive.layer, dive.layer_pages = layer, pages
            elif flow is not None and self._flow is flow and flow.textures is textures:
                if flow.layer is None:
                    flow.layer = layer
                else:
                    flow.full = layer
            elif self.textures is textures and self._flow_token is not None:
                self._tile_swap = layer
            elif self.textures is textures and self.tile_layer is not None:
                if self._tile_swap is not None:
                    self.canvas.unstage(self._tile_swap)
                self._swap_when_staged(layer)  # in steps, or the creep at rest meanwhile
                return
            else:
                return
            self.canvas.stage(layer)

        self._layer_job(work, built)

    def _layer_job(self, work, done) -> None:
        """Run work(progress, cancelled) in the background, then done(its result) here."""
        job = Job(work)
        job.finished.connect(done)

        def stopped() -> None:
            if job in self._layer_jobs:
                self._layer_jobs.remove(job)
            QTimer.singleShot(0, self, job.forget)  # its slots capture this window

        job.stopped.connect(stopped)
        self._layer_jobs.append(job)
        job.start()

    def _flow_advance(self, dt: float) -> None:
        """Move the leg on by dt seconds: slowing to a stop where the photo is to show if
        the next level isn't ready yet; on to the next leg once the tile fills the view."""
        self._last_clock += dt
        flow = self._flow
        if flow is None:
            self._creep(dt)
            return
        rate = flow_rate(int(self.settings.columns), float(self.settings.flow_seconds))
        per_second = 1.0 / flow.path.duration(rate)
        first, _ = self._flow_shares(flow.path)
        flow.hold = flow.path.u_at_share(first)
        ready = flow.photo is not None and flow.layer is not None
        target = 1.0
        if not ready:  # stop at the hold: brake once the distance it takes is left
            braking = 0.5 * flow.speed * flow.speed * EASE_SECONDS * per_second
            if flow.u + braking >= flow.hold:
                target = 0.0
        step = dt / EASE_SECONDS
        if flow.speed < target:
            flow.speed = min(flow.speed + step, target)
        else:
            flow.speed = max(flow.speed - step, target)
        flow.u += flow.speed * dt * per_second
        if not ready:
            flow.u = min(flow.u, flow.hold)
        elif flow.u >= 1.0:
            over = (flow.u - 1.0) / per_second  # seconds past the end: on into the next leg
            self._flow_arrive(over)
            return
        self._flow_frame()

    def _flow_shares(self, path: FlowPath) -> tuple[float, float]:
        """The tile's shares of the view where its photo starts to fade in, and where its
        mosaic has taken over (Mosaic appears at). A tile already larger than the first at
        the leg's start (few columns, a large tile) starts there, its mosaic later."""
        last = float(self.settings.flow_mosaic) / 100
        first = last / FADE_ZOOM**2
        start = path.tile_share(0.0)
        if start > first:
            first, last = start, min(max(last, start * FADE_ZOOM**2), 1.0)
            first = min(first, last / 1.01)  # some zoom for the fades, however little
        return first, last

    def _creep(self, dt: float) -> None:
        """By hand, between legs: the view zooms on in about its middle (dragging steers
        it), at the legs' rate, easing to a stop CREEP_ROOM short of where a base tile
        would start to show its photo; the last level fades meanwhile."""
        canvas, cam, settings = self.canvas, self.canvas.camera, self.settings
        rate = flow_rate(int(settings.columns), float(settings.flow_seconds))
        width, height = (float(v) for v in self.level.ctx.tile_size)
        _, tile_zoom = cam.fit_params(0, 0, width, height, margin=canvas.fit_margin)
        first = float(settings.flow_mosaic) / 100 / FADE_ZOOM**2
        limit = tile_zoom * first / CREEP_ROOM
        left = math.log(max(limit / cam.zoom, 1.0))  # log zoom to go
        self._creep_left = left
        braking = 0.5 * self._creep_speed**2 * EASE_SECONDS * rate
        target = 1.0 if left > braking + 1e-4 else 0.0
        step = dt / EASE_SECONDS
        if self._creep_speed < target:
            self._creep_speed = min(self._creep_speed + step, target)
        else:
            self._creep_speed = max(self._creep_speed - step, target)
        if left > 0:
            zoom = min(cam.zoom * math.exp(self._creep_speed * rate * dt), limit)
            canvas.set_view(cam.center, zoom)
        self._flow_leftovers(cam.zoom)

    def _creep_done(self) -> bool:
        """The creep is at rest where it stops, with nothing left to fade or swap in."""
        return (self._creep_speed == 0 and self._creep_left <= 1e-4
                and self._last_layer is None and self._tile_swap is None)  # fmt: skip

    def _flow_into(self, index: int) -> None:
        """By hand: head on into tile index (double-clicked) from the creep, as a leg."""
        if self._flow_token is None:  # the creep at rest
            self._start_flow(index, speed=0.0)
        else:
            self._flow_leg(self.level, index, self._creep_speed)
        self._refresh()

    def _flow_leftovers(self, zoom: float) -> None:
        """The last level fading once this one covers the view (until then its tiles show
        around it: the view's margins, or beyond its edge as Center first pans), by the
        view's zoom since, or as if it had zoomed on at the rate if slower (by hand the
        creep may stop); full-size tiles swapped in once staged."""
        canvas, last = self.canvas, self._last_layer
        if last is not None and self._last_zoom is None and self._level_covers_view():
            self._last_zoom, self._last_clock = zoom, 0.0
        if last is not None and self._last_zoom is not None:  # under it: its gaps
            rate = flow_rate(int(self.settings.columns), float(self.settings.flow_seconds))
            zoomed = max(math.log(max(zoom / self._last_zoom, 1.0)), rate * self._last_clock)
            gone = zoomed / math.log(LAST_OUT_ZOOM)
            last.instances["alpha"] = self._last_alpha * (1.0 - _fade(gone))
            last.mark_dirty()
            if gone >= 1.0:
                canvas.remove_layer(last)
                self._last_layer = None
        swap = self._tile_swap
        if swap is not None and canvas.staged(swap):
            self._tile_swap = None
            self._new_tile_layer(swap)

    def _level_covers_view(self) -> bool:
        """Every corner of the view lies on the level's canvas."""
        cam, ctx = self.canvas.camera, self.level.ctx
        w, h = cam.viewport
        corners = np.array([cam.screen_to_world(x, y) for x, y in ((0, 0), (w, 0), (0, h), (w, h))])
        return bool(
            np.all(corners >= -1e-6) and np.all(corners <= (ctx.width + 1e-6, ctx.height + 1e-6))
        )

    def _flow_frame(self) -> None:
        """Show the leg at its progress: the view; the last level fading; the tile's photo
        fading in, then into its mosaic, inside it; full-size tiles staged, swapped in."""
        flow, canvas = self._flow, self.canvas
        path, u = flow.path, flow.u
        view = path.at(u)
        canvas.camera.rotation = view.rotation
        canvas.set_view(view.center, view.zoom)
        canvas.update()
        self._flow_leftovers(view.zoom)
        if flow.full is not None and canvas.staged(flow.full):
            self._flow_full_in(flow)
        if flow.photo is None or flow.layer is None:
            return
        first, done = self._flow_shares(path)
        share = math.log(path.tile_share(u))
        step = math.log(done / first) / 2  # each fade's zoom (FADE_ZOOM, if there's room)
        photo_in = _fade((share - math.log(first)) / step)
        mosaic_in = _fade((share - math.log(first) - step) / step)
        if photo_in <= 0:
            return
        if not flow.placed:
            self._flow_place(flow)
        # Its tiles show once the photo covers the tile (those over its edges fade in with
        # them: nothing covers those): never through the photo as it fades in.
        inside = 1.0 if photo_in >= 1.0 else 0.0
        flow.layer.instances["alpha"] = np.where(flow.inside, inside, mosaic_in)
        flow.layer.mark_dirty()
        flow.photo.instances["alpha"] = photo_in * (1.0 - mosaic_in)
        flow.photo.mark_dirty()
        if len(flow.over):  # what lies on the tile lifts away as its photo comes
            self.tile_layer.instances["alpha"][flow.over] = 1.0 - photo_in
            self.tile_layer.mark_dirty()

    def _flow_place(self, flow: _Flow) -> None:
        """Put the next level's tiles inside the tile (in this level's world), just above
        this level's, and the photo over everything."""
        scene, i, canvas = flow.level.scene, flow.index, self.canvas
        ctx = flow.next.ctx
        scale = float(scene.size[i][0]) / float(ctx.width)
        corners = flow.next.scene.corners
        lo, hi = corners.min(axis=1), corners.max(axis=1)
        flow.inside = np.all(lo >= -1e-6, axis=1) & (hi[:, 0] <= ctx.width + 1e-6) & (
            hi[:, 1] <= ctx.height + 1e-6)  # fmt: skip
        layer = flow.layer
        layer.instances = _moved(layer.instances, (ctx.width / 2, ctx.height / 2),
                                 scene.center[i], scale, float(scene.rotation[i]))  # fmt: skip
        layer.instances["alpha"] = 0.0
        layer.mark_dirty()
        canvas.add_layer(layer, canvas.layers.index(self.tile_layer) + 1)
        canvas.add_layer(flow.photo)
        flow.placed = True

    def _flow_full_in(self, flow: _Flow) -> None:
        """Full-size tiles for the next level, staged: they take the thumbnails' place (in
        the tile, as those are placed, if they show)."""
        full, old = flow.full, flow.layer
        flow.full = None
        if flow.placed:
            for field in ("pos", "size", "rotation", "alpha"):
                full.instances[field] = old.instances[field]
            full.mark_dirty()
            index = self.canvas.layers.index(old)
            self.canvas.remove_layer(old)
            self.canvas.add_layer(full, index)
        else:
            self.canvas.unstage(old)
        flow.layer = full

    def _flow_arrive(self, over: float) -> None:
        """The tile fills the view: its level takes over, in its own world. Its tiles, shown
        inside the tile, are now where they belong, which looks the same; this level's go
        into that world (under them, fading out); the next leg starts, over seconds into
        it."""
        flow = self._flow
        if not flow.placed or flow.photo.instances["alpha"][0] > 0:  # a leg too short
            flow.u = 1.0
            self._flow_frame()  # its mosaic all in
        level, canvas = flow.next, self.canvas
        scene, i = flow.level.scene, flow.index
        ctx = level.ctx
        center, scale = (ctx.width / 2, ctx.height / 2), float(ctx.width) / float(scene.size[i][0])
        turn = float(scene.rotation[i])
        canvas.remove_layer(flow.photo)
        if self._last_layer is not None:
            canvas.remove_layer(self._last_layer)
        # This level's tiles go into the next one's world, around and under its tiles.
        self.zoom_detail.show(None, None, None)
        last = self.tile_layer
        last.instances = _moved(last.instances, scene.center[i], center, scale, -turn)
        last.mark_dirty()
        self._last_layer, self._last_alpha = last, last.instances["alpha"].copy()
        if self.textures is not None:
            self.textures.changed.disconnect(self._on_textures_changed)
            self.textures.close()
        if self._tile_swap is not None:  # this level's full-size tiles: too late now
            canvas.unstage(self._tile_swap)
            self._tile_swap = None
        # The next level's, where they belong.
        layer, textures = flow.layer, flow.textures
        textures.changed.disconnect(self._flow_textures_ready)
        native = textures.instances()
        for field in ("pos", "size", "rotation"):
            layer.instances[field] = native[field]
        layer.instances["alpha"] = native["alpha"]
        layer.mark_dirty()
        self.level, self.textures, self.tile_layer = level, textures, layer
        textures.changed.connect(self._on_textures_changed)
        if flow.full is not None:  # its full-size tiles, staging: swapped in once up
            self._tile_swap = flow.full
        flow.photo = flow.layer = flow.full = flow.textures = None
        canvas.fit_to(0, 0, float(ctx.width), float(ctx.height))  # the view: as the tile was
        canvas.camera.rotation = 0.0  # upright, as the tile was on screen
        canvas.pixelate_above = None
        self.zoom_detail.show(layer, level.scene, self.infinite.library)
        self.level_label.setText(f"Level {level.number}: {len(level.scene):,} tiles, made in "
                                 f"{flow.seconds:.2f} s")  # fmt: skip
        self._last_zoom, self._last_clock = None, over  # fading once the new one covers
        if not self.auto:  # by hand: the view creeps on, for the next tile to be picked
            # Center first's zoom is at rest as it arrives: the creep starts from rest too.
            speed = 0.0 if flow.path.path == CENTER_FIRST else flow.speed
            self._flow = None
            self._creep_speed = speed
            self._creep(over)
            self.status.setText(CREEP_STATUS)
            self._refresh()
            return
        self._flow_leg(level)
        self._flow.u = over / max(self._flow.path.seconds, MIN_SECONDS)
        self._flow_frame()

    # Loops

    def _recording(self) -> bool:
        """A loop is being recorded from the shown level (its latest)."""
        loop = self.loop
        return (loop is not None and not loop.closed
                and self._loop_pos == len(loop.levels) - 1 >= 0)  # fmt: skip

    def _next_tile(self) -> int:
        """Auto-dive's next tile: a loop's (replayed, home, or the look-ahead's best), or
        one picked in view."""
        loop = self.loop
        if self.settings.loop and loop is not None and self._loop_pos >= 0:
            if loop.closed:
                return loop.picks[self._loop_pos]
            if self.level.home >= 0:
                return self.level.home  # home is back: diving there closes the loop
            if self._ahead is not None:
                return self._ahead[0][0]
        return self.pick_tile()

    def _look_ahead(self) -> None:
        """While recording a loop in its range: match several candidate tiles' mosaics
        (in the background) to dive into the one home fits best."""
        loop = self.loop
        if (not self.settings.loop or not self._recording() or self._ahead is not None
                or self.level.home >= 0 or self._job is not None):  # fmt: skip
            return
        lure = loop.lure(self.settings)
        if lure is None:  # before the range: dives are as usual
            return
        picks: list[int] = []
        for _ in range(8 * int(self.settings.look_ahead)):
            index = self.pick_tile()
            if index not in picks:
                picks.append(index)
            if len(picks) >= int(self.settings.look_ahead):
                break
        library = self.infinite.library
        candidates = [(i, self.tile_view(i), path)
                      for i, path in zip(picks, library.paths(self.level.scene.slot[picks]),
                                         strict=True)]  # fmt: skip
        infinite, level = self.infinite, self.level

        def work(progress, cancelled):
            return infinite.look_ahead(candidates, lure, progress, cancelled, DIVE_THREADS)

        def done(results) -> None:
            self._job = None
            if self.level is level and self.loop is loop:
                self._ahead = results
            self._refresh()

        self._start_job(work, "ahead", "Auto-dive: looking ahead…", done)

    def _loop_dive(self, index: int) -> None:
        """Auto-dive's dive with Loop on: start the loop, record it, close it, or replay."""
        loop = self.loop
        if loop is None or self._loop_pos < 0:  # it starts: this tile is home
            self.loop = loop = Loop(self.tile_view(index))
            self._loop_pos, self._loop_next = -1, -1
            self.dive_into(index, lure=loop.lure(self.settings))  # level 1 (1: itself)
            return
        pos = self._loop_pos
        if loop.closed:  # replaying
            self._loop_next = (pos + 1) % len(loop.levels)
            self.dive_into(index, next_level=loop.levels[self._loop_next])
            return
        loop.picks.append(index)
        if index == self.level.home:  # home: back to the loop's first level
            loop.closed = True
            self._loop_next = 0
            self.dive_into(index, next_level=loop.levels[0])
            return
        ahead = dict(self._ahead or ())
        self._loop_next = -1
        if index in ahead:
            self.dive_into(index, next_level=self.infinite.number(ahead[index]))
        else:
            self.dive_into(index, lure=loop.lure(self.settings))

    def _loop_arrived(self, level: Level) -> None:
        """Keep the loop's books on arriving at a level (and say where it is)."""
        self._ahead = None
        loop = self.loop
        if loop is None:
            return
        if self._loop_next >= 0:  # a level of the loop, again
            self._loop_pos = self._loop_next
            if self._loop_pos == 0:
                loop.passes += 1
        else:  # recorded now
            loop.levels.append(level)
            self._loop_pos = len(loop.levels) - 1
        self._loop_next = -1
        n = len(loop.levels)
        if loop.closed:
            text = f"Loop of {n} dives: level {self._loop_pos + 1} (pass {loop.passes + 1})"
        else:
            text = f"Loop: level {n}, recording"
            if level.home >= 0:
                text += f"; home is back (fits {level.home_fit - 1:+.0%} on its spot's best)"
        self.level_label.setText(text)

    def _drop_loop(self) -> None:
        if self._job is not None and self._job_kind == "ahead":
            self._job.cancel()
            self._job.wait()
            self._job = None
        self.loop = None
        self._loop_pos = self._loop_next = -1
        self._ahead = None

    def _show_highlight(self, index: int) -> None:
        self._remove_highlight()
        scene = self.level.scene
        outline = make_instances(1)
        outline["pos"], outline["size"] = scene.center[index], scene.size[index]
        outline["rotation"], outline["alpha"] = scene.rotation[index], 0.0
        outline["tint"][:, :3], outline["tint"][:, 3] = HIGHLIGHT_COLOR, 1.0
        self._highlight = self.canvas.add_layer(
            SpriteLayer(None, outline, outline_px=HIGHLIGHT_PX, edge_px=1.0)
        )

    def _remove_highlight(self) -> None:
        if self._highlight is not None:
            self.canvas.remove_layer(self._highlight)
            self._highlight = None

    def _on_escape(self) -> None:
        self.set_auto(False)
        self._leave_full_screen()

    # Window

    def _on_library_changed(self) -> None:
        app = self.app_session
        if self._start_when_open and not app.library_opening:
            self._start_when_open = False
            if app.library is not None:
                self.start()
            else:  # it couldn't be opened (the Library tab says why)
                self.status.setText(app.library_error or "The library could not be opened.")
        self._refresh()

    def _on_tab(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        if index == LIBRARY_TAB:
            self.library.open_library()

    def _refresh(self) -> None:
        running = self.running
        self.start_button.setText("Stop" if running else "Start")
        self.start_button.setEnabled(running or (self.app_session.busy != "library"
                                                 and not self._start_when_open))  # fmt: skip
        self.library.set_locked(LIBRARY_LOCKED if running else None)
        self.new_start_button.setVisible(running)
        self.new_start_button.setEnabled(self.can_start_anew)
        self.start_button.setToolTip("End this session." if running else START_TIP)
        for form in (self.forms["mosaic"], self.forms["matching"], self.slicer_form):
            form.setEnabled(not running)
        self.auto_button.setVisible(running)
        self.auto_button.blockSignals(True)
        self.auto_button.setChecked(self.auto)
        self.auto_button.blockSignals(False)
        self.auto_button.setText("Stop Auto-Dive" if self.auto else "Auto-Dive")
        self.settings_note.setVisible(running)
        self.progress.setVisible(self._followed)
        if not self._followed:
            self._show_sub(False)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.canvas and self.auto and event.type() in STOPS_AUTO:
            self.set_auto(False)  # the viewer takes over
        if watched is self.canvas and event.type() in LOCKED_EVENTS:
            if self._locked or self._flow is not None:
                return True  # no panning or zooming while diving or fading (a leg's path)
            if event.type() == QEvent.Type.Wheel and self._continuous:
                return True  # it zooms by itself
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
        for job in list(self._layer_jobs):  # their signals come while this window is here
            job.cancel()
            job.wait()
        self.zoom_detail.shutdown()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)

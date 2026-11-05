"""The Infinite Mosaic window: dive into any tile, and its photo becomes the next mosaic.

The side panel shows one stage at a time. Setup is what comes before a session,
in order: the library (Manage shows its panel), the mosaic's look (core/infinite.py's
InfiniteSettings and the chosen slicer's own), and how to start. Three buttons
show a start in the view (`pick`; pressed again, another), following the look as it
changes, and Start goes from what shows, fixing the look for the session (`begin`):

- Random photo: a library photo, with a crop box locked to the tile shape. On
  Start, the crop is matched as the first level (it also builds the search
  index, the long part) and dissolves into its mosaic, with home set to it.
- Chosen image: an image file (ui/image_choice.py), the same way; its crop is a
  photo of the session's alone (`InfiniteSession.chosen_tile`).
- Random mosaic: random photos in the layout (`InfiniteSession.random_mosaic`).
  On Start the session is on at once, while the search index is built in the
  background (`prepare`). A dive made before it is ready plays as usual and
  its matching waits for it (the photo holds meanwhile).

End Session goes back to setup.
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

During a session the panel shows only what can be used then: Session on top
(what is being done: Explore or Auto-dive, Steps or Continuous, said in one line
that also shows briefly over the view in full screen), then Loop, Recording and
the settings that apply as they change. The library isn't there: its search
index holds the photos the session started with.

Auto-dive (the Auto-Dive button, A) dives on its own until stopped (the
button, A, Esc, or a click or the wheel on the mosaic): each mosaic shows
whole for the Dwell time, then a tile in view is picked (`pick_tile`), the
view moves to it and zooms in (`_approach`, the tile outlined), rests on it
for a moment, and dives. Its settings and the dive's apply as they change,
even while a session runs: they only pace it.

With Motion set to Continuous, it never stops (`_start_flow`):
the view zooms evenly from each mosaic onto a tile picked as the mosaic
appears (core/dive_path.py's FlowPath: one leg per mosaic), while that
tile's photo is read and its mosaic made in the background (`_flow_match`),
its layers built and staged to the GPU ahead (`_flow_prepare`,
`_build_layer`). As the tile grows its photo fades in over it, then into
its own mosaic, drawn inside it (its tiles placed on the tile, `moved`),
done when the tile fills Mosaic appears at of the view. When the tile
fills the view that mosaic takes over in its own world, looking just the
same (`_flow_arrive`); the last one goes into that world too, under it,
and fades out. A mosaic not made in time slows the zoom to a stop where
its photo would start to show, until it is. Once a leg's mosaic is made, the
next leg's tile is picked and the view carries on toward it through the switch
(`_flow_aim`, core/dive_path.py's FlowPath.leave).

Three things can be going on, one at a time, chosen on the panel: Explore (the
viewer chooses the dives), Auto-Dive (the window chooses new ones; it never plays
a recording) and Replay (the recording plays). Leaving Replay for either of the
others goes on from the mosaic shown, the recording kept. New dives that would
cut into the recording ask first.

Every session keeps its dives (`record`, core/infinite.py's DiveRecord), by
hand and auto-dive's alike. A dive the record has, made by hand, plays its level
again from the record (`_dive`); any other is kept from there on. Ending the
session and New Start begin a new record.

Recording (R) marks what Replay plays (and, later, a video): from the mosaic
shown when it starts to the last reached while it is on (DiveRecord's
recording; the record keeps every level either way, for loops). What would lose
any of it asks first (a dive elsewhere from a mosaic it went on from, Auto-Dive
there, ending the session, a new start, a new recording over one with dives).

Replay (P) plays the recording with the dive settings as they are then, once, or
holding a closed loop, round it until stopped. In steps its timeline
(core/dive_motion.py's DiveTimeline) drives the view, the tiles, the photo and the
outline frame by frame (`_replay_frame`). Continuously, the live legs play it
(`_replay_flow`): each leg heads for the tile the record went into, and its level
is made again from the record. Each level is made again from the record in the
background, with its tile layer and photo staged, while the one before it plays;
one not made in time holds the replay before its dive. Esc, a click or the wheel
stop it, as Explore does; stopped or done, the session goes on from the mosaic
shown, in Explore.

Set Home (H) makes the mosaic shown the one to come back to: home is the tile
it is the mosaic of. From then on each new level lures home (by hand every one;
auto-dive within Close within) and where a level holds it, it is outlined once
the mosaic shows. Close Loop (C) or a double-click on it dives in, by hand or
not, in steps or continuously, and that lands on the loop's first level. The
closed loop is then what Replay plays, and every pass goes the same way, so one
pass is a seamless loop. Auto-dive heads there by itself and stops there; in steps
it also makes several tiles' mosaics ahead while each mosaic holds (`_look_ahead`)
and dives into the one home fits best, already matched. Diving elsewhere opens a
closed loop again from there; Clear forgets it (its levels stay recorded).
"""

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from skitter.core.dive_motion import (
    FADE_ZOOM,
    DiveMotion,
    DiveTimeline,
    approach_zoom,
    fade,
    fade_out,
    flow_fades,
    flow_shares,
    glide,
    last_fade,
    moved,
    rest_tiles,
)
from skitter.core.dive_path import CENTER_FIRST, MIN_SECONDS, STRAIGHT, FlowPath, View, flow_rate
from skitter.core.dive_styles import DiveStyle, dive_seed, pick_style
from skitter.core.easing import ease_in_out_cubic
from skitter.core.infinite import (
    AUTO,
    AUTO_LOOP,
    DIVE,
    DIVE_THREADS,
    LIVE,
    LOOP,
    MATCHING,
    MOSAIC,
    DiveRecord,
    InfiniteSession,
    InfiniteSettings,
    Level,
    Lure,
    RecordedLevel,
    TileView,
    display_copy,
    find_tile,
    pick_weights,
)
from skitter.core.matching.mosaic import MosaicMethod
from skitter.core.matching.result import MatchCancelled
from skitter.core.slicing import SlicingCancelled, SlicingPlan
from skitter.core.slicing.layout import CANVAS_WIDTH
from skitter.ui import icons, preferences
from skitter.ui.actions import make_action, tool_button
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.image_choice import choose_image
from skitter.ui.jobs import Job
from skitter.ui.library_panel import LibraryPanel
from skitter.ui.render.atlas import PAGE as ATLAS_PAGE
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.render.tile_textures import (
    DETAIL_PAGE,
    TileTextures,
    detail_request,
    thumbnail_atlas,
)
from skitter.ui.render.zoom_detail import DETAIL_PAGES, DETAIL_SIDE, ZoomDetail
from skitter.ui.steps.base import side_panel, view_and_panel
from skitter.ui.steps.tile_picker import upper_closure
from skitter.ui.style import muted, progress_bar, show_progress
from skitter.ui.widgets.crop_overlay import CropOverlay
from skitter.ui.widgets.param_form import ParamForm

SETTINGS_KEY = "infinite/settings"  # preferences: the last session's settings (JSON)
PLAN_KEY = "infinite/plan"  # and its slicers' settings
WHOLE = 0.98  # a tile showing this much of itself lies on top (auto-dive prefers those)
PIXELATE_ABOVE = 4.0  # texels this large show as squares (not while diving: tiles blow up)
SETUP, LIBRARY, SESSION = 0, 1, 2  # the panel's stages
PICK_SIDE = 4096  # a picked photo's longest side as shown, at most (a texture's)
PICK_ROOM = 0.03  # around a picked photo, of its longest side: the crop box's handles
STARTS = (("photo", "Random Photo", "Show a random photo of the library; again for another."),
          ("mosaic", "Random Mosaic", "Show random photos in the layout; again for others."),
          ("image", "Choose Image…", "Start from an image of yours, cropped to the tile shape: "
                                     "it is also the home a loop comes back to."))  # fmt: skip
LIBRARY_LOCKED = "A session searches the photos it started with: stop it to change them."
# Continuous auto-dive (the fades: core/dive_motion.py). After the switch, the last mosaic
# (around and under the new one) stays until the new one covers the view, then fades out of
# its gaps. A leg slows to a stop (over EASE_SECONDS) where the photo would start to show if
# the next mosaic isn't made by then. Continuously by hand, the view creeps in between legs,
# at the same rate, slowing to a stop CREEP_ROOM of zoom before a base tile would start to
# show its photo, so a leg into one has time to make its mosaic.
EASE_SECONDS = 1.0
CREEP_ROOM = FADE_ZOOM
CREEP_STATUS = "Double-click a tile to head into it (drag to steer)."
PHOTO_DISPLAY = 2.0  # a photo is shown at up to this many times the view's pixels
MAX_FRAME_STEP = 0.1  # seconds the zoom moves on in one frame, at most
HIGHLIGHT_COLOR = (1.0, 1.0, 1.0)  # the outline of the tile auto-dive heads for
HIGHLIGHT_PX = 3.0
EXPLORE_TIP = "Dive by hand from the mosaic shown: double-click a tile. Pan and zoom as you like."
AUTO_TIP = "Make new dives on and on until stopped (A; Esc, a click or the wheel stop it too)."
STEPS_TIP = "Each mosaic shows whole until a tile is dived into."
FLOW_TIP = "One never-ending zoom: the view moves in by itself (drag to steer)."
SET_HOME_TIP = ("Make the mosaic shown the one to come back to (H): its picture is home, "
                "brought back where it fits.")  # fmt: skip
NO_HOME_TIP = "This mosaic isn't a tile's: dive into a tile, then set home."
CLOSE_LOOP_TIP = "Dive into home (C): back to where the loop began, closing it."
CLEAR_HOME_TIP = "Forget home (and the loop; its dives stay recorded)."
TOAST_SECONDS = 2.5
RECORD_TIP = ("Start recording from the mosaic shown (R): the dives from here are what "
              "Replay plays.")  # fmt: skip
STOP_RECORDING_TIP = "Stop recording (R): the recording keeps what it holds."
REPLAY_TIP = ("Play the recording (P), with the dive settings as they are now: once, or a "
              "closed loop round and round until stopped.")  # fmt: skip
REPLAYING = "Replaying the recording (P or Esc stops)."
NO_REPLAY_TIP = "Nothing to play yet: record some dives (R), or close a loop."
# Input on the mosaic that stops auto-dive (the viewer takes over).
STOPS_AUTO = (
    QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick, QEvent.Type.Wheel,
)  # fmt: skip
LOCKED_EVENTS = (
    QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease, QEvent.Type.MouseMove,
    QEvent.Type.MouseButtonDblClick, QEvent.Type.Wheel,
)  # fmt: skip


FORMS = (("mosaic", MOSAIC), ("matching", MATCHING), ("auto", AUTO), ("dive", DIVE),
         ("loop", LOOP), ("loop_auto", AUTO_LOOP))  # fmt: skip


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


class _Dive(DiveMotion):
    """A dive into tile `index` (its motion: core/dive_motion.py), and what it makes."""

    def __init__(self, index: int, layer: SpriteLayer, canvas: MosaicCanvas, end_zoom: float,
                 style: DiveStyle, seed: int):  # fmt: skip
        cam = canvas.camera
        super().__init__(layer.instances, index, cam.center, cam.zoom, cam.viewport,
                         cam.turn((0.0, 1.0)), end_zoom, style, seed)  # fmt: skip
        self.matched: Level | None = None  # the next level, once matched
        self.done = False  # the animation has finished
        # The next level's, made while the dive goes on (see _dive_matched).
        self.textures: TileTextures | None = None
        self.layer: SpriteLayer | None = None  # staged
        self.layer_pages: np.ndarray | None = None  # the textures' pages it was built from


class _Made:
    """A recorded level made again for a replay: the level, its textures, its tile layer
    and its photo (staged; None: not yet, or no photo: a blank canvas's)."""

    def __init__(self):
        self.level: Level | None = None
        self.textures: TileTextures | None = None
        self.layer: SpriteLayer | None = None
        self.layer_pages: np.ndarray | None = None  # the textures' pages it was built from
        self.building: np.ndarray | None = None  # the pages its last layer build began with
        self.photo: SpriteLayer | None = None
        self.listening = False  # its textures' changed signal is connected (for the layer)

    @property
    def ready(self) -> bool:
        level = self.level
        return (level is not None and self.layer is not None
                and (self.photo is not None or level.source is None))  # fmt: skip


class _Replay:
    """A replay under way: its timeline, its clock, and the levels made for it (by record
    index: the one shown, the next). A continuous one (live legs) has no timeline: span,
    the recording's (first, last) level, says where it starts, and legs how many there
    are to go (a closed loop's way back too; one that repeats has no end of them)."""

    def __init__(self, timeline: DiveTimeline | None, span: tuple[int, int] = (0, 0),
                 legs: float = 0):  # fmt: skip
        self.timeline = timeline
        self.span = span
        self.legs = legs
        self.t = 0.0
        self.made: dict[int, _Made] = {}
        self.shown = -1  # the record index on screen (-1: the session's own view still)
        self.diving = False  # the frame shown is a dive's
        self.waiting = False  # holding for a level not made yet


@dataclass
class _Pick:
    """The start shown, for Start to go from."""

    image: np.ndarray | None  # the whole photo (None: the random mosaic shown)
    slot: int  # its library slot (-1: not the library's)


class _Flow:
    """Continuous auto-dive's leg through the shown level, onto tile index."""

    def __init__(self, level: Level, index: int, tile: TileView, path: FlowPath):
        self.level, self.index, self.tile, self.path = level, index, tile, path
        self.u = 0.0  # progress along the path
        self.speed = 1.0  # eased down to 0 (and up again) while the next level is not made
        self.hold = 1.0  # where it waits if the next level isn't ready (see _flow_advance)
        self.next: Level | None = None  # the tile's mosaic, once made
        self.after: int | None = None  # the tile the leg after this one heads for, if known
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
        self.record = DiveRecord()  # the session's dives (see the module docstring)
        self._replay: _Replay | None = None  # a replay under way
        self._pos = -1  # the shown level's index in record.levels (-1: none yet)
        self._next = -1  # where the dive under way lands in it (-1: a new level)
        self._ahead: list[tuple[int, Level]] | None = None  # look-ahead's, for self.level
        # Jobs have no parent (it would keep each, and its inputs, for the window's life):
        # the last two are kept here, so each goes once a later one has started.
        self._recent_jobs: list[Job] = []
        self._pool: ProcessPoolExecutor | None = None
        self._locked = False  # pan and zoom off (diving, fading)
        self._fade_token: object | None = None
        self.auto = False  # auto-dive is on
        self._pick: _Pick | None = None  # the start shown, for Start to go from
        self._start_when_open: str | None = None  # the pick asked for while the library opened
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
        self.crop = CropOverlay(self.canvas)
        self.detail_reported.connect(self._on_detail)

        # The panel: a stage at a time (see the module docstring).
        self.library = LibraryPanel(session, lambda: float(self.settings.tile_aspect))
        self._build_forms()
        self.panel = QStackedWidget()
        for page in (self._build_setup(), self._build_library_page(), self._build_session()):
            self.panel.addWidget(page)
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
        QShortcut(QKeySequence("R"), self, self.toggle_recording)
        QShortcut(QKeySequence("P"), self, self.toggle_replay)
        QShortcut(QKeySequence("H"), self, self.set_home)
        QShortcut(QKeySequence("C"), self, self.close_loop)
        self._build_toast()
        self.prompt = QLabel(self.canvas)  # what to do, in the empty view
        self.prompt.setStyleSheet("color: rgba(255, 255, 255, 150);")
        self.prompt.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.resize(1400, 900)
        self._refresh()

    # Construction

    def _build_forms(self) -> None:
        self.forms = {name: ParamForm() for name, _ in FORMS}
        for form in self.forms.values():
            form.changed.connect(self._on_setting_changed)
        self.slicer_form = ParamForm()
        self.slicer_form.changed.connect(lambda _: self._repick())
        self._target_forms()

    def _build_setup(self) -> QWidget:
        """The library, the look, and how to start (always in view)."""
        self.library_label = QLabel()
        self.library_label.setWordWrap(True)
        manage = QPushButton("Manage…")
        manage.setToolTip("Choose the folders of photos the mosaics are made from.")
        manage.clicked.connect(self._show_library)
        row = QHBoxLayout()
        row.addWidget(self.library_label, stretch=1)
        row.addWidget(manage)
        library = QGroupBox("1. Library")
        library.setLayout(row)
        look = QGroupBox("2. Look")
        layout = QVBoxLayout(look)
        for form in (self.forms["mosaic"], self.slicer_form, self.forms["matching"]):
            layout.addWidget(form)
        self.start_buttons = {}
        start = QGroupBox("3. Start")
        layout = QVBoxLayout(start)
        for kind, label, tip in STARTS:
            button = self.start_buttons[kind] = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, kind=kind: self.pick(kind))
            layout.addWidget(button)
        self.start_button = QPushButton("Start")
        self.start_button.setToolTip("Start the session from what shows.")
        self.start_button.clicked.connect(self.begin)
        layout.addWidget(self.start_button)
        self.setup_status = QLabel()
        self.setup_status.setWordWrap(True)
        layout.addWidget(self.setup_status)
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(side_panel(library, look, resizable=True), stretch=1)
        column.addWidget(start)
        return page

    def _build_library_page(self) -> QWidget:
        back = QPushButton("Back to Setup")
        back.clicked.connect(lambda: self.panel.setCurrentIndex(SETUP))
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(back)
        column.addWidget(side_panel(self.library, resizable=True), stretch=1)
        return page

    def _build_session(self) -> QWidget:
        """Session (always in view), then Loop, Recording and what applies as it changes."""
        self.groups = {}
        for title, name in (("Auto-dive", "auto"), ("Dive", "dive")):
            group = self.groups[name] = QGroupBox(title)
            QVBoxLayout(group).addWidget(self.forms[name])
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self._build_session_group())
        column.addWidget(side_panel(self._build_loop_group(), self._build_recording_group(),
                                    *self.groups.values(), resizable=True), stretch=1)  # fmt: skip
        return page

    def _show_library(self) -> None:
        self.panel.setCurrentIndex(LIBRARY)
        self.library.open_library()

    def _target_forms(self) -> None:
        """Show self.settings (and its slicer's settings) in the forms."""
        for name, only in FORMS:
            self.forms[name].set_target(self.settings, only)
        self.slicer_form.set_target(self.plan.slicers[self.settings.slicer])

    def set_settings(self, settings: InfiniteSettings) -> None:
        """Edit these settings from now on (the next session starts with them)."""
        self.settings = settings
        self._target_forms()
        self.library.refresh_stats()  # its tile shape
        self._repick()

    def _build_session_group(self) -> QGroupBox:
        self.end_button = QPushButton("End Session…")
        self.end_button.setToolTip("Back to setup (asked first if a recording would be lost).")
        self.end_button.clicked.connect(self.ask_end_session)
        self.explore_button = QPushButton("Explore")
        self.explore_button.setToolTip(EXPLORE_TIP)
        self.auto_button = QPushButton("Auto-Dive")
        self.auto_button.setToolTip(AUTO_TIP)
        self.replay_button = QPushButton("Replay")
        self.steps_button = QPushButton("Steps")
        self.steps_button.setToolTip(STEPS_TIP)
        self.flow_button = QPushButton("Continuous")
        self.flow_button.setToolTip(FLOW_TIP)
        self.mode_label = QLabel()
        self.mode_label.setWordWrap(True)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.progress = progress_bar()
        self.sub_status = muted(QLabel())
        self.sub_progress = progress_bar()
        self.sub_progress.setMaximumHeight(6)
        self.level_label = muted(QLabel())
        help_text = muted(QLabel(
            "Double-click a tile to dive into it. A: auto-dive. H: set home. C: close the "
            "loop. R: record. P: replay. F11: full screen."
        ))  # fmt: skip
        group = QGroupBox("Session")
        layout = QVBoxLayout(group)
        layout.addWidget(self.end_button)
        layout.addLayout(self._choice(self._set_doing, self.explore_button, self.auto_button,
                                      self.replay_button))  # fmt: skip
        layout.addLayout(self._choice(self._set_motion, self.steps_button, self.flow_button))
        for widget in (self.mode_label, self.status, self.progress, self.sub_status,
                       self.sub_progress, self.level_label, help_text):  # fmt: skip
            layout.addWidget(widget)
        return group

    def _choice(self, clicked, *buttons: QPushButton) -> QHBoxLayout:
        """Buttons of which one is on; clicked gets its index (a click only: _refresh sets
        them from the state)."""
        group = QButtonGroup(self)
        row = QHBoxLayout()
        row.setSpacing(0)
        for index, button in enumerate(buttons):
            button.setCheckable(True)
            group.addButton(button, index)
            row.addWidget(button)
        group.idClicked.connect(clicked)
        return row

    def _set_doing(self, index: int) -> None:
        """Explore, Auto-Dive or Replay, in place of what is going on."""
        if index == 2:
            self.set_replay(True)
        else:
            self._end_replay("Replay stopped.")
            self.set_auto(index == 1)
        self._refresh()

    def _build_loop_group(self) -> QGroupBox:
        """Set Home, Close Loop and Clear, and where the loop has got to."""
        self.home_button = QPushButton("Set Home")
        self.home_button.clicked.connect(self.set_home)
        self.close_button = QPushButton("Close Loop")
        self.close_button.setToolTip(CLOSE_LOOP_TIP)
        self.close_button.clicked.connect(self.close_loop)
        self.clear_home_button = QPushButton("Clear")
        self.clear_home_button.setToolTip(CLEAR_HOME_TIP)
        self.clear_home_button.clicked.connect(self.clear_home)
        self.loop_label = QLabel()
        self.loop_label.setWordWrap(True)
        row = QHBoxLayout()
        for button in (self.home_button, self.close_button, self.clear_home_button):
            row.addWidget(button)
        group = QGroupBox("Loop")
        layout = QVBoxLayout(group)
        layout.addLayout(row)
        layout.addWidget(self.loop_label)
        layout.addWidget(self.forms["loop"])
        layout.addWidget(self.forms["loop_auto"])
        self.loop_group = group
        return group

    def _build_toast(self) -> None:
        """What is being done, over the view: full screen hides the panel that says it."""
        self.toast = QLabel(self.canvas)
        self.toast.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.toast.hide()
        self._toast_fade = QVariantAnimation(self)
        self._toast_fade.setDuration(round(TOAST_SECONDS * 1000))
        self._toast_fade.setKeyValueAt(0.0, 1.0)
        self._toast_fade.setKeyValueAt(0.7, 1.0)
        self._toast_fade.setKeyValueAt(1.0, 0.0)
        self._toast_fade.valueChanged.connect(self._toast_opacity)
        self._toast_fade.finished.connect(self.toast.hide)
        self._mode = ""  # what the panel last said is being done

    def _toast_opacity(self, opacity: float) -> None:
        self.toast.setStyleSheet(
            f"background: rgba(0, 0, 0, {round(170 * opacity)}); "
            f"color: rgba(255, 255, 255, {round(255 * opacity)}); "
            "padding: 6px 10px; border-radius: 4px;"
        )

    def _show_toast(self) -> None:
        self.toast.setText(self._mode)
        self._toast_opacity(1.0)
        self.toast.adjustSize()
        self.toast.move(16, 16)
        self.toast.show()
        self._toast_fade.stop()
        self._toast_fade.start()

    def _build_recording_group(self) -> QGroupBox:
        """Record and stop (icons), what the recording holds, and how far Replay is."""
        self.record_action = make_action(self, "Start Recording", self.start_recording,
                                         icon=icons.record(), tooltip=RECORD_TIP)  # fmt: skip
        self.stop_recording_action = make_action(self, "Stop Recording", self.stop_recording,
                                                 icon=icons.stop(),
                                                 tooltip=STOP_RECORDING_TIP)  # fmt: skip
        self.recording_label = QLabel()
        self.recording_label.setWordWrap(True)
        self.replay_progress = progress_bar()
        self.replay_label = muted(QLabel())
        row = QHBoxLayout()
        for action in (self.record_action, self.stop_recording_action):
            row.addWidget(tool_button(action, 24))
        row.addWidget(self.recording_label, stretch=1)
        group = QGroupBox("Recording")
        layout = QVBoxLayout(group)
        layout.addLayout(row)
        layout.addWidget(self.replay_progress)
        layout.addWidget(self.replay_label)
        self.recording_group = group
        return group

    def _on_setting_changed(self, name: str) -> None:
        if name in LIVE:  # a running session follows them: keep them for the next one too
            save_settings(self.settings, self.plan)
        if name == "motion" and self.running:  # the other way of moving, from where it is
            self._cancel_ahead()  # steps' only
            self._cancel_auto_step()
            if self.level is not None and not self.busy:  # at rest (a creep's may not be)
                self.canvas.pixelate_above = PIXELATE_ABOVE
            self._auto_next()
            self._mark_home()
        if name == "tile_aspect":
            self.library.refresh_stats()
        if name == "slicer":
            self.slicer_form.set_target(self.plan.slicers[self.settings.slicer])
        else:  # settings in one form may show or hide others in another (`when`)
            for form in self.forms.values():
                form.refresh()
        if name not in LIVE:
            self._repick()

    def _set_motion(self, index: int) -> None:
        motion = ("steps", "flow")[index]
        if motion != self.settings.motion:
            self.settings.motion = motion
            self._on_setting_changed("motion")
        self._refresh()

    # Session

    @property
    def running(self) -> bool:
        return self.infinite is not None and self._pick is None

    @property
    def busy(self) -> bool:
        """Matching a level, diving or fading: no new dive yet (preparing or looking
        ahead allow one)."""
        matching = self._job is not None and self._job_kind in ("first", "dive")
        moving = self.dive is not None or self._flow is not None or self._replay is not None
        return matching or moving or self._fade_token is not None

    @property
    def _continuous(self) -> bool:
        """Dives move continuously in this session."""
        return self.running and self.settings.motion == "flow"

    def ask_end_session(self) -> None:
        if self._may_lose_recording("End Session", "End this session?"):
            self.stop()

    def _may_lose_recording(self, action: str, question: str, dives: int | None = None) -> bool:
        """Whether an action that loses the recording's dives (by default all of them) may
        go ahead: asked, if there are any."""
        dives = self.record.recorded_dives() if dives is None else dives
        if dives <= 0:
            return True
        text = f"{question} The recording's {dives:,} dive{'s' if dives != 1 else ''} will be lost."
        return self._confirm(text, action)

    def _confirm(self, text: str, action: str) -> bool:
        """Ask: the action's button, or Cancel (the default)."""
        box = QMessageBox(QMessageBox.Icon.Question, "Infinite Mosaic", text, parent=self)
        go = box.addButton(action, QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        return box.clickedButton() is go

    def pick(self, kind: str = "photo") -> None:
        """Show a start ("photo", "mosaic" or "image") for Start to go from, in place of any
        shown. With the library not open yet, once it is (it opens in the background)."""
        app = self.app_session
        library = app.library
        if library is None:
            app.open_library_in_background()
            self._start_when_open = kind
            self.setup_status.setText("Opening the library: this shows once it's open…")
            self._refresh()
            return
        self._start_when_open = None
        self.setup_status.setText("")
        if not len(library):
            self._show_library()
            self._refresh()
            return
        image, slot = None, -1
        if kind == "image":
            image = choose_image(self)
            if image is None:
                return
        self._open_session()
        if kind == "photo":
            slot = self.infinite.random_tile().slot
            image = self.infinite.picture(TileView(slot, (0.0, 0.0, 1.0, 1.0)))
        self._pick = _Pick(image, slot)
        self._show_pick()
        self._refresh()

    def _open_session(self) -> None:
        """A session of the current settings, for what's picked and then for Start."""
        app = self.app_session
        base = next((m for m in app.project.match_methods.values()
                     if isinstance(m, MosaicMethod)), None)  # fmt: skip
        self.infinite = InfiniteSession(app.library, self.settings, self.plan, base, app.matcher)
        if self._pool is not None:
            return
        self._pool = ProcessPoolExecutor(os.cpu_count())
        for _ in range(os.cpu_count() or 1):  # start the workers while the first level matches
            self._pool.submit(int)
        # Detail pages for the levels' tile layers, readied before anything moves: one made
        # mid-dive stalled a frame about 45 ms (see sprites.SPARE_DETAILS).
        self.canvas.reserve_details(DETAIL_SIDE, DETAIL_PAGES, 3)
        # Likewise the stacks of levels' tile layers: two for full-size tiles (1 or 2 pages:
        # a 1-page layer takes a 2-page stack), one for thumbnails (see sprites.SPARE_TEXTURES).
        full, thumbs = (DETAIL_PAGE, DETAIL_PAGE, 2), (ATLAS_PAGE, ATLAS_PAGE, 1)
        self.canvas.reserve_textures([full, full, thumbs])

    def _show_pick(self) -> None:
        """What's picked in the view: a random mosaic, or the photo with a crop box of the
        tile shape, the largest that fits."""
        image = self._pick.image
        if image is None:
            self.crop.stop()
            if self.photo_layer is not None:
                self.canvas.remove_layer(self.photo_layer)
                self.photo_layer = None
            width, height = self.canvas.width(), self.canvas.height()
            aspect = width / height if width > 0 and height > 0 else 1.5
            self._show_level(self.infinite.random_mosaic(aspect))
            self.setup_status.setText("This is the first mosaic.")
            return
        self._clear_level()
        self.level = None
        height, width = image.shape[:2]
        self._show_photo(display_copy(image, PICK_SIDE), (width / 2, height / 2), (width, height),
                         0.0, 1.0)  # fmt: skip
        room = PICK_ROOM * max(width, height)
        self.canvas.fit_to(-room, -room, width + room, height + room)
        self.crop.start(width, height)
        self.crop.set_aspect(float(self.settings.tile_aspect))
        self.setup_status.setText("Drag the box: what's inside becomes the first picture.")

    def _repick(self) -> None:
        """The look changed: what's picked, in it (a crop stays if its shape does)."""
        if self._pick is None:
            return
        self._open_session()
        if self._pick.image is None or self.crop.aspect != float(self.settings.tile_aspect):
            self._show_pick()

    def _picked_tile(self) -> TileView:
        """The crop of the picked photo, as a tile."""
        image, slot = self._pick.image, self._pick.slot
        left, top, width, height = self.crop.crop_box()
        if slot < 0:
            return self.infinite.chosen_tile(image[top : top + height, left : left + width])
        high, wide = image.shape[:2]
        rect = left / wide, top / high, (left + width) / wide, (top + height) / high
        return TileView(slot, rect)

    def begin(self) -> None:
        """Start the session from what's picked."""
        if self._pick is None:
            return
        tile = None if self._pick.image is None else self._picked_tile()
        self._pick = None
        self.crop.stop()
        self.setup_status.setText("")
        save_settings(self.settings, self.plan)
        if tile is None:
            self._record_arrived(self.level)
            self.level_label.setText(f"Random mosaic: {len(self.level.scene):,} tiles")
            self._prepare()
            self._auto_next()
        else:
            self.record.start_loop(tile, 0)  # coming back to it is what it was picked for
            self._start_photo(tile)
        self._refresh()

    def _start_photo(self, tile: TileView) -> None:
        self._clear_level()
        self.level = None
        picture = self.infinite.picture(tile)
        width, height = CANVAS_WIDTH, CANVAS_WIDTH * picture.shape[0] / picture.shape[1]
        self._show_photo(picture, (width / 2, height / 2), (width, height), 0.0, 1.0)
        self.canvas.fit_to(0, 0, width, height)
        self._match(picture, "first", "Preparing tiles (the first level takes longest)…",
                    self.record.lure(self.settings, not self.auto), tile)  # fmt: skip

    def stop(self) -> None:
        """End the session (or drop what's picked): stop its work and clear the view."""
        self._pick = None
        self.crop.stop()
        self.setup_status.setText("")
        self._end_replay("")
        self.set_auto(False)
        self._stop_flow()
        if self._job is not None:
            self._job.cancel()
            self._job.wait()
            self._job = None
        self._new_record()
        self._remove_highlight()
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
        self._refresh()

    def _on_matched(self, level: Level) -> None:
        self._job = None
        seconds = time.perf_counter() - self._job_started
        self.level_label.setText(f"{len(level.scene):,} tiles, matched "
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
        self._record_arrived(level)
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
            alpha = fade_out(t, duration)
            layer.instances["alpha"] = alpha
            layer.mark_dirty()
            if alpha > 0:
                return True
            layer.visible = False
            self._fade_token = None
            self._locked = False
            self._refresh()
            self._mark_home()
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
        if not self.running or self.level is None or self.busy:
            return
        index = self.tile_at(*self.canvas.camera.screen_to_world(sx, sy))
        if index >= 0:
            self._go_into(index)

    def _go_into(self, index: int) -> None:
        """Dive into tile index, by hand (asked first if that cuts into the recording)."""
        cut = self.record.cut_by(self._pos, index)
        if cut and not self._may_lose_recording(
                "Dive", "This dive leaves the recorded way.", cut):  # fmt: skip
            return
        if self.settings.motion == "flow":
            self._flow_into(index)
        else:
            self._dive(index)

    def tile_view(self, index: int) -> TileView:
        """Tile index of the shown level, as a TileView."""
        scene = self.level.scene
        return TileView(int(scene.slot[index]), tuple(float(v) for v in scene.rect[index]),
                        bool(scene.mirrored[index]))  # fmt: skip

    def dive_into(self, index: int, next_level: Level | None = None, lure: Lure | None = None,
                  recorded: RecordedLevel | None = None) -> None:  # fmt: skip
        """Dive into tile index of the level: its photo becomes the next one. next_level:
        already made (a look-ahead's): shown when the dive ends, nothing is matched;
        recorded: made again meanwhile from the record (replaying), nothing matched. Else it
        is matched meanwhile, with lure (a loop's home) if given. The record is _dive's."""
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
        elif recorded is not None:
            self._rebuild(recorded)
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
        view = self.dive.frame(e, self.tile_layer.instances)
        self.tile_layer.mark_dirty()
        photo = self.photo_layer.instances
        photo["rotation"], photo["alpha"] = view.photo_rotation, view.photo_alpha
        self.photo_layer.mark_dirty()
        self.canvas.set_view(view.center, view.zoom)

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
            moving = self._approaching or self._replay is not None and self._replay.diving
            if not self.canvas.staged(layer) or self.dive is not None or moving:
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
        if on and not self.auto:
            self._end_replay("Replay stopped.")
            cut = self.record.cut_by(self._pos, -1) if self._pos >= 0 else 0
            on = self._may_lose_recording("Auto-Dive", "Auto-dive makes new dives from here.", cut)
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
            self._mark_home()
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

    def _whole_view(self, level: Level) -> tuple[np.ndarray, float]:
        """The center and zoom the view has on level's whole mosaic."""
        ctx = level.ctx
        return self.canvas.camera.fit_params(0, 0, float(ctx.width), float(ctx.height),
                                             margin=self.canvas.fit_margin)  # fmt: skip

    def pick_tile(self, level: Level | None = None) -> int:
        """Auto-dive's next tile: at random among the tiles in view (Next tile: anywhere,
        or among the tenth nearest the middle) that lie on top (a pile's: nothing on them;
        else those mostly shown), varied ones likelier (Favor varied tiles: see
        core/infinite.py's Looks). level: of this mosaic yet to show, whole in view."""
        cam = self.canvas.camera
        center, zoom = cam.center, cam.zoom
        if level is None:
            level = self.level
        else:
            center, zoom = self._whole_view(level)
        scene = level.scene
        pos = scene.center
        half = cam.viewport / (2 * zoom)
        in_view = np.all(np.abs(pos - center) <= half, axis=1)
        share = scene.visible_share
        whole, shown = share >= WHOLE, share >= 0.5
        for candidates in (in_view & whole, in_view & shown, in_view, shown):
            pool = np.flatnonzero(candidates)
            if len(pool):
                break
        else:
            pool = np.arange(len(scene))
        if self.settings.next_tile == "middle" and len(pool) > 1:
            distance = np.linalg.norm(pos[pool] - center, axis=1)
            pool = pool[np.argsort(distance)[: max(1, len(pool) // 10)]]
        variety = level.variety
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
        zoom0 = float(cam.zoom)
        zoom1 = approach_zoom(float(cam.viewport[0]), tile_width, float(settings.approach_tiles),
                              zoom0)  # fmt: skip
        zoom1 = float(np.clip(zoom1, cam.min_zoom, cam.max_zoom))
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
            canvas.set_view(*glide(center, screen0, zoom0, zoom1, float(ease_in_out_cubic(e))))
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
        self._dive(index)

    # Continuous auto-dive

    def _start_flow(self, index: int | None = None, speed: float = 1.0) -> None:
        """Start the never-ending zoom from the view as it is (see core/dive_path.py):
        auto-dive's legs, a leg onto tile index (a double-click), or by hand the creep
        (from rest: speed is a leg's, as a share of the rate)."""
        token = self._flow_token = object()
        self._flow_last = None
        self.canvas.pixelate_above = None  # tiles grow far past their texels: smooth them
        if self._legs_go_on or index is not None:
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
        hand: double-clicked), or auto-dive's next (the record's, where it goes on from
        here). Along the recorded way its level is made again; else it is recorded."""
        if index is None:
            index = self._next_tile()
        record = self.record
        self._next = record.leads_to(self._pos, index)
        if self._next < 0:
            self._branch(index)
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
        recorded = self.record.levels[self._next] if self._next >= 0 else None
        lure = self.record.lure(self.settings, not self.auto)

        def work(progress, cancelled):
            if recorded is not None:  # along the recorded way: made again, nothing matched
                level = infinite.rebuild(recorded, path)
                picture = level.picture
            else:
                picture = infinite.picture(tile, path)
                level = infinite.level(picture, progress, cancelled, self.detail_reported.emit,
                                       DIVE_THREADS, lure, tile)  # fmt: skip
            atlas = thumbnail_atlas(level.scene, infinite.library)
            request = detail_request(level.scene, infinite.library)
            return level, display_copy(picture, longest), (atlas, request)

        def done(result) -> None:
            self._job = None
            if self._flow is flow:
                flow.next, flow.picture, atlas = result
                flow.seconds = time.perf_counter() - self._job_started
                self._flow_prepare(flow, *atlas)
                self._flow_aim(flow)
                doing = "Auto-dive." if self.auto else "Diving."
                self.status.setText(REPLAYING if self._replay is not None else doing)
            self._refresh()

        message = REPLAYING if self._replay is not None else "Making the next mosaic…"
        self._start_job(work, "flow", message, done)

    def _flow_aim(self, flow: _Flow) -> None:
        """Its mosaic made, pick the leg after this one's tile, if legs go on Straight in,
        for the view to carry on toward it where they meet, rather than stop on this one."""
        replay = self._replay
        if not self._legs_go_on or self.settings.flow_path != STRAIGHT:
            return
        if self.record.closed if replay is None else replay.legs <= 1:
            return  # the last leg, auto-dive's into home or a replay's
        level = flow.next
        pos = self._next if self._next >= 0 else len(self.record.levels)
        flow.after = self._next_tile(level, pos)
        center, zoom = self._whole_view(level)
        flow.path.leave((level.scene.center[flow.after] - center) * zoom, flow.u)

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
                     dive: _Dive | None = None, into=None) -> None:  # fmt: skip
        """Build a tile layer from the textures' pages as they are, in the background (the
        pages are copied out to RGBA there), then stage it: for the leg's next level (the
        first: shown inside the tile; full-size ones swap in), the dive's (dive) or the
        shown one (flow None, or the leg moved on meanwhile: it swaps in once staged).
        into(layer, pages): given, it takes the layer instead (a replay's)."""
        pages, instances = textures.pages, textures.instances()

        def work(progress, cancelled):
            return SpriteLayer(pages, instances, detail_pages=DETAIL_PAGES,
                               detail_side=DETAIL_SIDE)  # fmt: skip

        def built(layer: SpriteLayer) -> None:
            if textures.pages is not pages:  # newer pages: their own layer follows
                return
            if into is not None:
                into(layer, pages)
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
        """core/dive_motion.py's flow_shares at this session's Mosaic appears at."""
        return flow_shares(path, float(self.settings.flow_mosaic) / 100)

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
            gone = last_fade(zoom / self._last_zoom, self._last_clock, rate)
            last.instances["alpha"] = self._last_alpha * (1.0 - fade(gone))
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
        photo_in, mosaic_in = flow_fades(path, u, float(self.settings.flow_mosaic) / 100)
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
        layer.instances = moved(layer.instances, (ctx.width / 2, ctx.height / 2),
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
        self._remove_highlight()  # home's, in the world being left
        center, scale = (ctx.width / 2, ctx.height / 2), float(ctx.width) / float(scene.size[i][0])
        turn = float(scene.rotation[i])
        canvas.remove_layer(flow.photo)
        if self._last_layer is not None:
            canvas.remove_layer(self._last_layer)
        # This level's tiles go into the next one's world, around and under its tiles.
        self.zoom_detail.show(None, None, None)
        last = self.tile_layer
        last.instances = moved(last.instances, scene.center[i], center, scale, -turn)
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
        self._record_arrived(level)
        textures.changed.connect(self._on_textures_changed)
        if flow.full is not None:  # its full-size tiles, staging: swapped in once up
            self._tile_swap = flow.full
        flow.photo = flow.layer = flow.full = flow.textures = None
        canvas.fit_to(0, 0, float(ctx.width), float(ctx.height))  # the view: as the tile was
        canvas.camera.rotation = 0.0  # upright, as the tile was on screen
        canvas.pixelate_above = None
        self.zoom_detail.show(layer, level.scene, self.infinite.library)
        self._mark_home()
        self.level_label.setText(f"{len(level.scene):,} tiles, made in "
                                 f"{flow.seconds:.2f} s")  # fmt: skip
        self._last_zoom, self._last_clock = None, over  # fading once the new one covers
        replay = self._replay
        if replay is not None and replay.timeline is None:
            replay.legs -= 1
            if replay.legs <= 0:
                self._end_replay("Replay done: dive on from here.")  # its last level: at rest
                return
        if not self._legs_go_on:  # by hand: the view creeps on, for the next tile to be picked
            # Center first's zoom is at rest as it arrives: the creep starts from rest too.
            speed = 0.0 if flow.path.path == CENTER_FIRST else flow.speed
            self._flow = None
            self._creep_speed = speed
            self._creep(over)
            self.status.setText(CREEP_STATUS)
            self._refresh()
            return
        self._flow_leg(level, flow.after)
        self._flow.u = over / max(self._flow.path.seconds, MIN_SECONDS)
        self._flow_frame()

    # The record and loops

    def _rebuild(self, entry: RecordedLevel) -> None:
        """Make a recorded level again in the background (replaying a dive: nothing is
        matched; its photo is read again)."""
        infinite = self.infinite
        path = None if entry.source is None else infinite.library.paths([entry.source.slot])[0]

        def work(progress, cancelled):
            return infinite.rebuild(entry, path)

        self._start_job(work, "dive", "Replaying the next level…", self._on_matched)

    def _dive(self, index: int) -> None:
        """Dive (in steps) into tile index of the shown level, as the record has it: a dive
        it recorded is replayed; any other is recorded from here on (what followed goes)."""
        record, pos = self.record, self._pos
        then = record.leads_to(pos, index)
        if then >= 0:
            self._next = then
            self.dive_into(index, recorded=record.levels[then])
            return
        if self._branch(index):
            self.dive_into(index, recorded=record.levels[self._next])
            return
        ahead = dict(self._ahead or ())
        if index in ahead:
            self.dive_into(index, next_level=self.infinite.number(ahead[index]))
        else:
            self.dive_into(index, lure=record.lure(self.settings, not self.auto))

    def _branch(self, index: int) -> bool:
        """Record a new dive into tile index of the shown level; whether it closes the loop
        (home: then it leads back to the loop's first level, `_next`)."""
        closing = self._loop_open() and index == self._home_tile()
        self.record.branch(self._pos, index)
        self._next = -1
        if closing:
            self.record.close_loop()
            self._next = self.record.loop_start
        return closing

    def _home_tile(self, level: Level | None = None) -> int:
        """Home's index in level (default: the mosaic shown; -1: not there). Looked for, not
        the level's `home`: that is the home it was matched for, which may have changed.
        One that matching left under other tiles doesn't count: it can't be seen."""
        home, level = self.record.home, self.level if level is None else level
        if home is None or level is None:
            return -1
        index = find_tile(level.scene, home)
        if index < 0 or index == level.home or level.scene.visible_share[index] >= WHOLE:
            return index
        return -1

    def _loop_open(self) -> bool:
        """A loop is under way, and the shown level is its latest. Not at its first: home's
        photo may turn up in its own mosaic, and a loop has to go somewhere."""
        record = self.record
        return (record.home is not None and not record.closed
                and record.loop_start < self._pos == len(record.levels) - 1)  # fmt: skip

    def _next_tile(self, level: Level | None = None, pos: int | None = None) -> int:
        """The next leg's or auto-dive's tile from level, the record's pos (default: the
        mosaic shown): a replay's, the record's; a loop's (home, or the look-ahead's best);
        or one picked in view."""
        record = self.record
        pos = self._pos if pos is None else pos
        if self._replay is not None:
            return record.picks[pos]
        if record.home is not None and pos >= record.loop_start and not record.closed:
            # Home is back: diving there closes the loop (not short of the range, should
            # matching have put it here by itself).
            home = self._home_tile(level)
            if home >= 0 and pos - record.loop_start + 1 >= self.settings.loop_dives[0]:
                return home
            if self._ahead is not None and level is None:
                return self._ahead[0][0]
        return self.pick_tile(level)

    def _look_ahead(self) -> None:
        """While a loop is open, in its range: match several candidate tiles' mosaics (in
        the background) to dive into the one home fits best."""
        record = self.record
        if (not self._loop_open() or self._ahead is not None or self._home_tile() >= 0
                or self._job is not None):  # fmt: skip
            return
        lure = record.lure(self.settings)
        if lure is None or not lure.place:  # before the range: dives are as usual
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
            if self.level is level and self.record is record:
                self._ahead = results
            self._refresh()

        self._start_job(work, "ahead", "Auto-dive: looking ahead…", done)

    def _record_arrived(self, level: Level) -> None:
        """Keep the record's books on arriving at a level (and say where a loop is)."""
        self._ahead = None
        record = self.record
        if self._next >= 0:  # a recorded level, again
            if record.closed and self._next == record.loop_start and self._next <= self._pos:
                record.passes += 1  # come round
            self._pos = self._next
        else:  # a new one
            self._pos = record.add(level)
        self._next = -1
        record.reached(self._pos)
        replay = self._replay
        if replay is not None and replay.timeline is None:  # a continuous replay's progress
            first, last = replay.span
            show_progress(self.replay_progress, (self._pos - first) / max(last - first, 1))
            self.replay_label.setText(f"Replay: mosaic {self._pos - first + 1} of "
                                      f"{last - first + 1}")  # fmt: skip
        self._show_recording()
        if self.auto and record.closed and self._replay is None:
            self.set_auto(False)  # it made the loop; Replay plays it

    def _mark_home(self) -> None:
        """Outline home where the shown level holds it while a loop is open."""
        if self._loop_open() and self._home_tile() >= 0 and self._highlight is None:
            self._show_highlight(self._home_tile())
            self._highlight.instances["alpha"] = 1.0
            self._highlight.mark_dirty()

    def _cancel_ahead(self) -> None:
        if self._job is not None and self._job_kind == "ahead":
            self._job.cancel()
            self._job.wait()
            self._job = None
        self._ahead = None

    def _drop_loop(self) -> None:
        """Forget the loop (its levels stay recorded)."""
        self._cancel_ahead()
        self.record.end_loop()

    @property
    def can_set_home(self) -> bool:
        """Set Home: a mosaic of a tile shown (one being left continuously still is)."""
        return (self.level is not None and self.level.source is not None and self._pos >= 0
                and self._replay is None and self.dive is None)  # fmt: skip

    @property
    def can_close_loop(self) -> bool:
        """Close Loop: home is in the mosaic shown, the loop open, nothing under way."""
        return self._loop_open() and self._home_tile() >= 0 and not self.busy

    def _may_forget_loop(self) -> bool:
        return not self.record.closed or self._confirm(
            "Forget the closed loop? Its dives stay recorded, but no longer lead back.", "Forget"
        )

    def set_home(self) -> None:
        """Come back to the mosaic shown from now on, in place of any home there is."""
        if self.can_set_home and self._may_forget_loop():
            self._drop_loop()
            self._remove_highlight()
            self.record.start_loop(self.level.source, self._pos)
            self._mark_home()
            self._show_recording()

    def clear_home(self) -> None:
        if self.record.home is not None and self._replay is None and self._may_forget_loop():
            self._drop_loop()
            self._remove_highlight()
            self._show_recording()

    def close_loop(self) -> None:
        """Dive into home, whatever auto-dive was up to."""
        if self.can_close_loop:
            self._auto_token = None
            if self._approaching:
                self._end_approach()
            self._go_into(self._home_tile())

    def _loop_text(self) -> str:
        record, level = self.record, self.level
        if record.home is None:
            return "No home: dive into a tile first." if self._no_source else "No home set."
        count = len(record.levels) - record.loop_start
        if record.closed:
            dives = f"{count} dive{'s' if count != 1 else ''}"
            return f"Closed loop of {dives}: Replay plays it."
        if self._loop_open() and self._home_tile() >= 0:
            if level.home != self._home_tile() or math.isnan(level.home_fit):
                return "Home is here."
            off = level.home_fit - 1
            return (
                f"Home is here: {abs(off):.0%} {'worse' if off > 0 else 'better'} than "
                "the best tile for its spot."
            )
        out = max(self._pos - record.loop_start, 0)
        return f"Home set: {out} dive{'s' if out != 1 else ''} out."

    @property
    def _no_source(self) -> bool:
        """The mosaic shown isn't a tile's (a random mosaic's): it can't be home."""
        return self.level is not None and self.level.source is None

    def _new_record(self) -> None:
        """Begin a new record (a new start, or the session's end)."""
        self._drop_loop()
        self.record = DiveRecord()
        self._pos = self._next = -1

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
        self._end_replay("Replay stopped.")
        self.set_auto(False)
        self._leave_full_screen()

    # Recording

    def toggle_recording(self) -> None:
        if self.record.recording_on:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        """Record from the mosaic shown (in place of a recording there is, once asked)."""
        if not self.running or self._pos < 0 or self._replay is not None:
            return
        if self._may_lose_recording("Record", "Start a new recording from here?"):
            self.record.start_recording(self._pos)
        self._show_recording()

    def stop_recording(self) -> None:
        self.record.stop_recording()
        self._show_recording()

    def _show_recording(self) -> None:
        """Say what the recording holds (and refresh its buttons)."""
        record = self.record
        dives = record.recorded_dives()
        count = f"{dives:,} dive{'s' if dives != 1 else ''}"
        if record.recording_on:
            text = f"<b style='color:#d33'>●</b> Recording: {count}"
        elif record.recording is not None:
            text = f"Recorded: {count}"
        else:
            text = "Not recording."
        if (
            record.recording is not None
            and record.closed
            and record.loop_start >= record.recording[0]
        ):
            text += " (a closed loop)"
        self.recording_label.setText(text)
        self._refresh()

    # Replay

    @property
    def can_replay(self) -> bool:
        """Replay: a recording with a dive, at rest (or replaying: to stop it)."""
        if self._replay is not None:
            return True
        return self.running and self.record.recorded_dives() >= 1 and not self.busy

    def toggle_replay(self) -> None:
        self.set_replay(self._replay is None)

    def set_replay(self, on: bool) -> None:
        if on and self._replay is None and self.can_replay:
            self.replay()
        elif not on:
            self._end_replay("Replay stopped.")
        self._refresh()

    def replay(self) -> None:
        """Play the recording (see the module docstring); recording stops."""
        self.set_auto(False)
        self._cancel_auto_step()  # a creep's too
        self.record.stop_recording()
        if self.settings.motion == "flow":
            self._replay_flow()
            return
        settings = self.settings.copy()  # as they are now: changes apply to the next replay
        timeline = DiveTimeline(self.record, settings, self.canvas.camera.viewport.copy(),
                                span=self.record.recording, repeat=self.record.repeats)  # fmt: skip
        replay = self._replay = _Replay(timeline)
        self._remove_highlight()
        if self.photo_layer is not None:
            self.canvas.remove_layer(self.photo_layer)
            self.photo_layer = None
        first = timeline.steps[0]
        self._make_for_replay(first.position)
        self._make_for_replay(first.next)
        self.status.setText("Replay: making the first mosaic…")
        self.replay_label.setText(f"Replay: mosaic 1 of {len(timeline.steps)}")
        show_progress(self.replay_progress, 0.0)
        last = {"t": None}

        def step(t: float) -> bool:
            if self._replay is not replay:
                return False
            dt = 0.0 if last["t"] is None else min(max(t - last["t"], 0.0), MAX_FRAME_STEP)
            last["t"] = t
            return self._replay_frame(replay, dt)

        self.canvas.add_animation(step)
        self._refresh()

    @property
    def _legs_go_on(self) -> bool:
        """Continuously, each leg's end starts the next: auto-dive, or a replay."""
        return self.auto or self._replay is not None and self._replay.timeline is None

    def _replay_flow(self) -> None:
        """Replay the recording continuously: its first level made again and shown, then
        the live legs along the recorded way (each next tile the record's) to its last."""
        record = self.record
        first, last = record.recording  # closed: then the way back
        legs = math.inf if record.repeats else record.recorded_dives()
        replay = self._replay = _Replay(None, (first, last), legs)
        self._remove_highlight()
        self.status.setText("Replay: making the first mosaic…")
        self.replay_label.setText(f"Replay: mosaic 1 of {last - first + 1}")
        show_progress(self.replay_progress, 0.0)
        infinite, entry = self.infinite, record.levels[first]
        path = None if entry.source is None else infinite.library.paths([entry.source.slot])[0]

        def done(level: Level) -> None:
            if self._replay is not replay:
                return  # stopped meanwhile
            if self.photo_layer is not None:
                self.canvas.remove_layer(self.photo_layer)
                self.photo_layer = None
            self._show_level(level)
            self._pos, self._next = first, -1
            self.status.setText(REPLAYING)
            self._start_flow()

        self._layer_job(lambda progress, cancelled: infinite.rebuild(entry, path), done)
        self._refresh()

    def _replay_frame(self, replay: _Replay, dt: float) -> bool:
        """Move the replay on by dt seconds and show it (an animation: False once done).
        It holds (shows the same) while a level it needs isn't made yet."""
        timeline = replay.timeline
        t = replay.t + dt if timeline.loop else min(replay.t + dt, timeline.duration)
        step, phase = timeline.phase_at(t)
        for pos in (step.position, step.next):
            self._make_for_replay(pos)
        keep = {replay.shown, step.position, step.next}
        for pos in [p for p in replay.made if p not in keep]:
            self._discard_made(replay.made.pop(pos))
        need = [step.position] + ([step.next] if phase == "dive" else [])
        if not all(replay.made[pos].ready for pos in need):
            if not replay.waiting:
                replay.waiting = True
                self.status.setText("Replay: making the next mosaic…")
            return True
        if replay.waiting or replay.shown < 0:
            replay.waiting = False
            self.status.setText(REPLAYING)
        replay.t = t
        self._show_replay_frame(replay, timeline.frame(t, replay.made[step.position].level))
        at = timeline.wrapped(t)
        show_progress(self.replay_progress, at / timeline.duration if timeline.duration else 1.0)
        if t >= timeline.duration and not timeline.loop:
            self._end_replay("Replay done: dive on from here.")
            return False
        return True

    def _show_replay_frame(self, replay: _Replay, frame) -> None:
        """Show a timeline frame: its level (switching to it), tiles, view, photo, outline."""
        canvas = self.canvas
        if replay.shown != frame.position:
            self._replay_switch(replay, frame.position)
        if frame.diving and not replay.diving:  # tiles move: nothing to sharpen
            self.zoom_detail.show(None, None, None)
            canvas.pixelate_above = None  # they grow far past their texels: smooth them
        replay.diving = frame.diving
        tiles = self.tile_layer.instances
        for name, values in frame.tiles.items():
            tiles[name] = values
        self.tile_layer.mark_dirty()
        canvas.camera.rotation = 0.0
        canvas.set_view(frame.center, frame.zoom)
        photo = frame.photo
        if photo is None:
            if self.photo_layer is not None:
                self.photo_layer.visible = False
        else:
            layer = replay.made[photo.level].photo
            if self.photo_layer is not layer:
                if self.photo_layer is not None and self.photo_layer in canvas.layers:
                    canvas.remove_layer(self.photo_layer)
                self.photo_layer = canvas.add_layer(layer)  # over the tiles
            inst = layer.instances
            inst["pos"], inst["size"] = photo.pos, photo.size
            inst["rotation"], inst["alpha"] = photo.rotation, photo.alpha
            layer.visible = True
            layer.mark_dirty()
        if frame.highlight >= 0:
            if self._highlight is None:
                self._show_highlight(frame.highlight)
            self._highlight.instances["alpha"] = frame.highlight_alpha
            self._highlight.mark_dirty()
        elif self._highlight is not None:
            self._remove_highlight()
        canvas.update()

    def _replay_switch(self, replay: _Replay, pos: int) -> None:
        """Show the replay's level pos (made, its tile layer staged) in the last one's place."""
        made = replay.made[pos]
        if made.listening:
            made.textures.changed.disconnect(self._replay_textures_ready)
            made.listening = False
        old = replay.made.pop(replay.shown, None)
        self._show_level(made.level, made.textures, made.layer)  # the last one's go
        if made.layer_pages is not made.textures.pages:
            self._on_textures_changed()  # its full-size tiles came while it was built
        if old is not None:  # its textures and tile layer went with it: its photo too
            old.textures = old.layer = None
            self._discard_made(old)
        replay.shown, replay.diving = pos, False
        replay.made[pos] = made
        steps = replay.timeline.steps  # a closed loop's first shows twice: by time, not index
        k = steps.index(replay.timeline.phase_at(replay.t)[0])
        self.replay_label.setText(f"Replay: mosaic {k + 1} of {len(steps)}")
        self.level_label.setText(f"{len(made.level.scene):,} tiles")

    def _make_for_replay(self, pos: int) -> None:
        """Make the record's level pos again in the background, then its textures, tile
        layer and photo (staged), for the replay to show."""
        replay = self._replay
        if pos < 0 or replay is None or pos in replay.made:
            return
        made = replay.made[pos] = _Made()
        infinite = self.infinite
        library = infinite.library
        entry = self.record.levels[pos]
        path = None if entry.source is None else library.paths([entry.source.slot])[0]
        ratio = self.canvas.devicePixelRatioF()
        longest = round(PHOTO_DISPLAY * max(self.canvas.camera.viewport) * ratio)

        def work(progress, cancelled):
            level = infinite.rebuild(entry, path)
            picture = None if level.source is None else display_copy(level.picture, longest)
            return (level, thumbnail_atlas(level.scene, library),
                    detail_request(level.scene, library), picture)  # fmt: skip

        def done(result) -> None:
            if self._replay is not replay or replay.made.get(pos) is not made:
                return  # done or moved on meanwhile
            level, atlas, request, picture = result
            made.level = level
            made.textures = TileTextures(level.scene, library, self, pool=self._pool,
                                         atlas=atlas, request=request)  # fmt: skip
            made.textures.changed.connect(self._replay_textures_ready)
            made.listening = True
            if picture is not None:
                photo = make_instances(1)
                photo["alpha"] = 0.0
                made.photo = SpriteLayer(picture, photo)
                self.canvas.stage(made.photo)
            self._build_made_layer(replay, pos, made)

        self._layer_job(work, done)

    def _build_made_layer(self, replay: _Replay, pos: int, made: _Made) -> None:
        def into(layer: SpriteLayer, pages) -> None:
            if self._replay is not replay or replay.made.get(pos) is not made:
                return  # done or moved on meanwhile
            if replay.shown == pos:
                return  # shown: its full-size tiles swap in as the session's do
            if made.layer is not None:
                self.canvas.unstage(made.layer)
            made.layer, made.layer_pages = layer, pages
            self.canvas.stage(layer)

        made.building = made.textures.pages
        self._build_layer(made.textures, None, into=into)

    def _replay_textures_ready(self) -> None:
        """Full-size crops came for a level the replay has made, not shown yet: its tile
        layer from them."""
        replay = self._replay
        if replay is None:
            return
        for pos, made in replay.made.items():
            # A build begun with older pages is dropped when it ends: build from these.
            if pos != replay.shown and made.listening and made.building is not made.textures.pages:
                self._build_made_layer(replay, pos, made)

    def _discard_made(self, made: _Made) -> None:
        """Let go of what a replay made for a level, not to be shown."""
        canvas = self.canvas
        for layer in (made.layer, made.photo):
            if layer is None:
                continue
            if layer is self.photo_layer:
                self.photo_layer = None
            if layer in canvas.layers:
                canvas.remove_layer(layer)
            else:
                canvas.unstage(layer)
        if made.textures is not None:
            if made.listening:
                made.textures.changed.disconnect(self._replay_textures_ready)
            made.textures.close()
        made.textures = made.layer = made.photo = None

    def _end_replay(self, message: str) -> None:
        """Stop a replay (if one is under way): at rest on the mosaic it shows, which the
        session goes on from."""
        replay = self._replay
        if replay is None:
            return
        self._replay = None
        if replay.timeline is None:  # continuous: the legs stop, at rest on the mosaic shown
            self._stop_flow()
            level = self.level
            if level is not None and self._pos >= replay.span[0]:
                self.canvas.camera.rotation = 0.0
                self.canvas.fit_to(0, 0, float(level.ctx.width), float(level.ctx.height))
                self.canvas.pixelate_above = PIXELATE_ABOVE
            self.replay_label.setText("")
            self._show_recording()
            self._auto_next()  # by hand, the creep (it says how to go on)
            creep = self.status.text() == CREEP_STATUS
            self.status.setText(f"{message} {CREEP_STATUS}" if creep else message)
            return
        for pos in [p for p in replay.made if p != replay.shown]:
            self._discard_made(replay.made.pop(pos))
        self._remove_highlight()
        if self.photo_layer is not None:
            self.photo_layer.visible = False
        if replay.shown >= 0:
            level, canvas = self.level, self.canvas
            tiles = self.tile_layer.instances
            for name, values in rest_tiles(level.scene).items():
                tiles[name] = values
            self.tile_layer.mark_dirty()
            canvas.camera.rotation = 0.0
            canvas.fit_to(0, 0, float(level.ctx.width), float(level.ctx.height))
            canvas.pixelate_above = PIXELATE_ABOVE
            if replay.diving:  # it stopped mid-dive: tiles at rest, sharpened again
                self.zoom_detail.show(self.tile_layer, level.scene, self.infinite.library)
            self._pos, self._next = replay.shown, -1
            self.level_label.setText(f"{len(level.scene):,} tiles")
        self.replay_label.setText("")
        self.status.setText(message)
        self._show_recording()
        self._auto_next()  # continuously by hand, the creep

    # Window

    def _on_library_changed(self) -> None:
        app = self.app_session
        if self._pick is not None:  # it may not be there any more
            self.stop()
        if self._start_when_open is not None and not app.library_opening:
            kind, self._start_when_open = self._start_when_open, None
            self.setup_status.setText("")
            if app.library is not None:
                self.pick(kind)
        self._refresh()

    def _library_text(self) -> str:
        app = self.app_session
        if app.busy == "library":
            return "Updating the library…"
        if app.library is None:
            if app.library_opening:
                return "Opening the library…"
            return app.library_error or "The library isn't open yet."
        count = len(app.library)
        if not count:
            return "No photos yet: add a folder to begin."
        return f"{count:,} photo{'s' if count != 1 else ''}"

    def _show_setup(self) -> None:
        """What can be started, said in the panel and the empty view."""
        app = self.app_session
        empty = app.library is not None and not len(app.library)
        waiting = app.busy == "library" or self._start_when_open is not None
        for button in self.start_buttons.values():
            button.setEnabled(not empty and not waiting)
        self.start_button.setEnabled(self._pick is not None and not waiting)
        self.library_label.setText(self._library_text())
        self.setup_status.setVisible(bool(self.setup_status.text()))
        self.prompt.setText("Add photos to the library to begin." if empty
                            else "Set up on the right, then choose how to start.")  # fmt: skip
        self._place_prompt()

    def _place_prompt(self) -> None:
        self.prompt.adjustSize()
        self.prompt.move((self.canvas.width() - self.prompt.width()) // 2,
                         (self.canvas.height() - self.prompt.height()) // 2)  # fmt: skip

    def _refresh(self) -> None:
        running = self.running
        if running:
            self.panel.setCurrentIndex(SESSION)
        elif self.panel.currentIndex() == SESSION:
            self.panel.setCurrentIndex(SETUP)
        self.prompt.setVisible(self.infinite is None)
        self._show_setup()
        self.library.set_locked(LIBRARY_LOCKED if running else None)
        replaying, record = self._replay is not None, self.record
        doing = self.replay_button if replaying else self.auto_button if self.auto else None
        (doing or self.explore_button).setChecked(True)
        (self.flow_button if self.settings.motion == "flow" else self.steps_button).setChecked(True)
        self.replay_button.setEnabled(self.can_replay)
        self.replay_button.setToolTip(REPLAY_TIP if record.recorded_dives() else NO_REPLAY_TIP)
        for button in (self.steps_button, self.flow_button):
            button.setEnabled(not replaying)
        self.groups["auto"].setVisible(self.auto)
        self.forms["loop_auto"].setVisible(self.auto)
        self._show_mode(replaying)
        self.loop_group.setVisible(running)
        self.home_button.setEnabled(self.can_set_home)
        self.home_button.setToolTip(NO_HOME_TIP if self._no_source else SET_HOME_TIP)
        self.close_button.setEnabled(self.can_close_loop)
        self.clear_home_button.setEnabled(record.home is not None and not replaying)
        self.loop_label.setText(self._loop_text())
        self.recording_group.setVisible(running)
        self.record_action.setEnabled(running and not record.recording_on and self._pos >= 0
                                      and not replaying)  # fmt: skip
        self.stop_recording_action.setEnabled(record.recording_on)
        self.replay_progress.setVisible(replaying)
        self.replay_label.setVisible(replaying)
        self.progress.setVisible(self._followed)
        if not self._followed:
            self._show_sub(False)

    def _show_mode(self, replaying: bool) -> None:
        """Say what is being done, in the panel and, full screen, over the view if it changed."""
        doing = "Replay" if replaying else "Auto-dive" if self.auto else "Explore"
        motion = "Continuous" if self.settings.motion == "flow" else "Steps"
        mode = f"{doing} · {motion}"
        mosaic = f" · Mosaic {self._pos + 1}" if self.running and self._pos >= 0 else ""
        self.mode_label.setText(f"<b>{doing}</b> · {motion}{mosaic}")
        if mode != self._mode:
            self._mode = mode
            if self.isFullScreen():
                self._show_toast()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.canvas and event.type() == QEvent.Type.Resize:
            self._place_prompt()
        if watched is self.canvas and event.type() in STOPS_AUTO:
            self._end_replay("Replay stopped.")  # the viewer takes over
            if self.auto:
                self.set_auto(False)
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
            self._show_toast()

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
        if self.running and not self._may_lose_recording("Close", "Close the Infinite Mosaic?"):
            event.ignore()
            return
        self.shutdown()
        super().closeEvent(event)

"""Step 5: animate the construction of the finished mosaic.

Pick a choreography for each phase of the animation (core/animation,
phases.py: Build, then optionally Show and Clear, chosen on the Animation
group's tabs, each with its own holds before and after), edit their settings
and the look, then play or scrub the timeline (its strip shows the phases and
their holds), and export it as a video (Export Animation): the video is
exactly this animation. The choreographies, the camera's keys, the look and
the video settings live in the project, so the export renders exactly what
this tab shows.

The camera is its keyframes (camera_keys.py: the Look through camera
viewfinder, keys on the timeline, the Keyframes group). The Video group
edits the settings that decide what the video shows (size and framing; the
Export Animation window edits the same settings, and both stay in step);
"Show export frame" outlines what the video shows at the current moment and
dims the rest, and "Follow camera" keeps the view on it (panning or zooming
by hand stops following). A new mosaic opens on its finished state; Play
runs from the start. Playback controls sit under the view
(ui/widgets/transport.py).
"""

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from skitter.core.animation import PHASE_NAMES, PHASES, PhasedTimeline
from skitter.core.animation.camera import CameraPath, Shot, home_shot
from skitter.core.animation.look import TableCamera
from skitter.core.animation.phases import OPTIONAL
from skitter.core.animation.video import output_size, view_rect
from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import ColorParam
from skitter.ui import preferences
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.render.player import TimelinePlayer
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.render.tile_textures import TileTextures
from skitter.ui.steps.base import StepPage, side_panel
from skitter.ui.steps.camera_keys import CameraKeys
from skitter.ui.style import muted
from skitter.ui.widgets.param_form import ParamForm
from skitter.ui.widgets.transport import TransportBar

MARGIN = 0.05  # view margin around the mosaic, share of its size
FRAME_DIM = 0.6  # opacity of the shade outside the export frame
FAR = 1e7  # mosaic units: "everywhere" for the shade
FRAME_SETTINGS = ("resolution", "width", "height", "framing", "margin")  # what the video shows


class AnimateStep(StepPage):
    id = "animate"
    title = "Animate"

    export_requested = Signal()  # the user pressed Export Animation…

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.scene: MosaicScene | None = None
        self.textures: TileTextures | None = None
        self.camera_path: CameraPath | None = None  # the camera over the video
        self._stale = False  # manual picks changed the mosaic while this tab was hidden

        self.canvas = MosaicCanvas()
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)  # the playback keys work
        self.player = TimelinePlayer(self.canvas, self)
        self.player.time_changed.connect(self._show_shot)
        self.canvas.view_changed.connect(self._on_view_changed)
        self.transport = TransportBar(self.player)
        self.transport.status.setText("No mosaic: match tiles first.")
        self.canvas.fps_changed.connect(self.transport.set_fps)
        self.keys = CameraKeys(self)
        self._shade = self.canvas.add_layer(SpriteLayer(None, make_instances(0)))
        self._frame_line = self.canvas.add_layer(
            SpriteLayer(None, make_instances(0), outline_px=1.5, edge_px=1.0)
        )

        view = QWidget()
        column = QVBoxLayout(view)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.canvas, stretch=1)
        column.addWidget(self.transport)
        self.transport.install_shortcuts(view)
        self.keys.install_shortcuts(self)  # the side panel too: after editing a key's fields

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(view, stretch=1)
        layout.addWidget(
            side_panel(
                self.keys.build_group(),  # first: it works with the strip under the view
                self._build_settings_group(),
                self._build_look_group(),
                self._build_video_group(),
            )  # fmt: skip
        )

        session.project_replaced.connect(self._on_project_replaced)
        session.matching_changed.connect(self._load_scene)
        session.mosaic_edited.connect(self._on_mosaic_edited)
        session.animation_changed.connect(self._on_animation_changed)
        session.camera_changed.connect(self._on_camera_changed)
        for signal in session.export_signals():
            signal.connect(self._refresh_export)
        self._apply_look()
        self._load_scene()
        self._refresh_export()

    @property
    def project(self):
        return self.session.project

    # Construction

    def _build_settings_group(self) -> QGroupBox:
        """The phases (Build, Show, Clear) as tabs, each with its choreography and holds."""
        self.phase_tabs = QTabBar()
        self.phase_tabs.setExpanding(True)
        self.phase_tabs.setDocumentMode(True)
        # The style draws the chosen tab much like the others: mark it plainly.
        self.phase_tabs.setStyleSheet(
            "QTabBar::tab { padding: 4px 6px; border-bottom: 2px solid transparent; }"
            "QTabBar::tab:selected { font-weight: bold;"
            " border-bottom: 2px solid palette(highlight); }"
        )
        for phase in PHASES:
            self.phase_tabs.addTab(PHASE_NAMES[phase])
        self.phase_tabs.setTabToolTip(0, "Putting the mosaic together.")
        self.phase_tabs.setTabToolTip(1, "Something happening on the finished mosaic "
                                         "(optional).")  # fmt: skip
        self.phase_tabs.setTabToolTip(2, "Clearing the mosaic away (optional).")
        self.phase_tabs.currentChanged.connect(lambda _: self._show_phase())
        self.choreography_box = QComboBox()
        self.choreography_box.currentIndexChanged.connect(self._on_choreography)
        self.description = muted(QLabel())
        self.description.setWordWrap(True)
        self.form = ParamForm()
        self.form.changed.connect(lambda _: self.session.animation_edited())
        self.holds_form = ParamForm()  # the phase's holds (phases.PhaseHolds)
        self.holds_form.changed.connect(lambda _: self.session.animation_edited())
        group = QGroupBox("Animation")
        layout = QVBoxLayout(group)
        layout.addWidget(self.phase_tabs)
        top = QFormLayout()
        top.addRow("Choreography:", self.choreography_box)
        layout.addLayout(top)
        layout.addWidget(self.description)
        layout.addWidget(self.holds_form)  # the phase's, so above its choreography's settings
        layout.addWidget(self.form)
        self._show_phase()
        return group

    def _build_look_group(self) -> QGroupBox:
        self.look_form = ParamForm()
        self.look_form.set_target(self.project.animation_look)
        self.look_form.changed.connect(lambda _: self.session.animation_edited())
        group = QGroupBox("Look")
        QVBoxLayout(group).addWidget(self.look_form)
        return group

    @property
    def phase(self) -> str:
        """The phase whose choreography the Animation group shows."""
        return PHASES[max(self.phase_tabs.currentIndex(), 0)]

    def show_phase(self, phase: str) -> None:
        self.phase_tabs.setCurrentIndex(PHASES.index(phase))

    def _build_video_group(self) -> QGroupBox:
        self.show_frame = QCheckBox("Show export frame")
        self.show_frame.setToolTip(
            "Outline what the exported video shows (its size and framing) and dim the rest."
        )
        self.show_frame.setChecked(
            preferences.settings().value("animate/show_frame", "false") == "true"
        )
        self.show_frame.toggled.connect(self._on_show_frame)
        self.follow = QCheckBox("Follow camera")
        self.follow.setToolTip(
            "Keep the view on what the video shows as the camera moves. Panning or zooming "
            "by hand stops following."
        )
        self.follow.setChecked(
            preferences.settings().value("animate/follow_camera", "true") == "true"
        )
        self.follow.toggled.connect(self._on_follow)
        self.follow.setEnabled(self.show_frame.isChecked())
        self.video_form = ParamForm()
        self.video_form.set_target(self.project.video_settings, only=FRAME_SETTINGS)
        self.video_form.changed.connect(lambda _: self.session.animation_edited())
        self.export_button = QPushButton("Export Animation…")
        self.export_button.setToolTip("Format, frame rate, quality and more, then export.")
        self.export_button.clicked.connect(self.export_requested)
        group = QGroupBox("Video")
        layout = QVBoxLayout(group)
        layout.addWidget(self.show_frame)
        layout.addWidget(self.follow)
        layout.addWidget(self.video_form)
        layout.addWidget(self.export_button)
        return group

    # Step

    def is_complete(self) -> bool:
        return self.scene is not None

    def on_enter(self) -> None:
        if self._stale:
            self._load_scene()

    def on_leave(self) -> None:
        self.player.pause()

    def _on_mosaic_edited(self, regions) -> None:
        if self.isVisible():
            self._load_scene()
        else:
            self._stale = True

    # Scene

    def _load_scene(self) -> None:
        scene = self.session.scene
        self._stale = False
        if scene is self.scene:
            return
        self.player.pause()
        if self.textures is not None:
            self.textures.changed.disconnect(self._on_textures)
            self.textures.patched.disconnect(self._on_textures)
            self.textures.status_changed.disconnect(self.transport.status.setText)
        self.scene, self.textures = scene, self.session.textures
        if scene is None or self.textures is None or not len(scene):
            self.scene = self.textures = None
            self.camera_path = None
            self.player.set_timeline(None)
            self.player.set_content(None, None)
            self.transport.refresh()
            self.transport.status.setText("No mosaic: match tiles first.")
            self._show_export_frame()
            self.state_changed.emit()
            return
        self.textures.changed.connect(self._on_textures)
        self.textures.patched.connect(self._on_textures)  # crops for picks read meanwhile
        self.textures.status_changed.connect(self.transport.status.setText)
        self.transport.status.setText(self.textures.status)
        timeline = self.project.animation(scene)
        self.player.set_timeline(timeline, time=timeline.duration)  # open on the finished mosaic
        self._plan_camera()
        self.player.set_camera(TableCamera.for_scene(scene, self.project.animation_look))
        self.player.set_content(self.textures.pages, self.textures.instances())
        self._sync_transport()
        self.keys.refresh()
        self._show_export_frame()
        self._fit()
        self.state_changed.emit()

    def _on_textures(self, *_) -> None:
        if not self.isVisible() and self._stale:
            return  # reloaded on entering
        self.player.set_content(self.textures.pages, self.textures.instances())

    def _on_project_replaced(self) -> None:
        self.look_form.set_target(self.project.animation_look)
        self.video_form.set_target(self.project.video_settings, only=FRAME_SETTINGS)
        self._show_phase()
        self.keys.selected = -1
        self.keys.refresh()

    def _show_phase(self) -> None:
        """The current phase tab's choices (None for an optional phase), and the chosen one."""
        phase, box = self.phase, self.choreography_box
        box.blockSignals(True)
        box.clear()
        if phase in OPTIONAL:
            box.addItem("None", None)
            box.setItemData(0, f"No {PHASE_NAMES[phase].lower()} phase.",
                            Qt.ItemDataRole.ToolTipRole)  # fmt: skip
        for choreography in self.project.choreographies.values():
            if choreography.phase == phase:
                box.addItem(choreography.name, choreography.id)
                box.setItemData(box.count() - 1, choreography.description,
                                Qt.ItemDataRole.ToolTipRole)  # fmt: skip
        box.setCurrentIndex(max(box.findData(self.project.phase_choices.get(phase)), 0))
        box.blockSignals(False)
        self._show_choreography()

    def _show_choreography(self) -> None:
        chosen = self.project.phase_choreography(self.phase)
        if chosen is not None:
            self.description.setText(chosen.description)
        elif self.choreography_box.count() > 1:
            self.description.setText(f"No {PHASE_NAMES[self.phase].lower()} phase: choose one "
                                     "to add it.")  # fmt: skip
        else:
            self.description.setText(f"No {PHASE_NAMES[self.phase].lower()} choreographies "
                                     "yet.")  # fmt: skip
        self.form.set_target(chosen)
        # A phase left out has no holds.
        self.holds_form.set_target(None if chosen is None else self.project.phase_holds[self.phase])

    def _on_choreography(self) -> None:
        self.project.phase_choices[self.phase] = self.choreography_box.currentData()
        self._show_choreography()
        self.session.animation_edited()

    def _on_animation_changed(self) -> None:
        """Choreography, camera, look or video settings edited (here or in Export Animation)."""
        self.video_form.refresh()
        self.look_form.refresh()
        self.holds_form.refresh()
        self._apply_look()
        self._replan()
        self._sync_transport()
        self.keys.refresh()
        self._show_export_frame()
        if self._following():
            self._fit()  # keep the whole frame in view as its shape changes

    def _on_camera_changed(self) -> None:
        """The camera's keys changed (edited, undone, redone): only the camera is planned
        again, not the tiles, so editing keys live through the camera stays smooth."""
        self._plan_camera()
        self.keys.refresh()
        self._show_export_frame()
        if self._following() and not self.keys.editing_live:
            self._fit()  # (live, the view already is the shot)

    def _replan(self) -> None:
        """New timeline for the current settings: at the same moment, or still at the end."""
        if self.scene is None:
            return
        at_end = self.player.time >= self.player.duration
        timeline = self.project.animation(self.scene)
        self._plan_camera(timeline)
        self.player.set_timeline(timeline, timeline.duration if at_end else None)

    def _sync_transport(self) -> None:
        """Frame steps match the export's frames; controls follow whether there's a timeline;
        the strip shows the phases and their holds."""
        self.transport.frame_step = 1.0 / float(self.project.video_settings.fps)
        timeline = self.player.timeline
        if timeline is not None:
            self.transport.set_sections(timeline.spans)
        self.transport.refresh()

    def _plan_camera(self, timeline: PhasedTimeline | None = None) -> None:
        """The camera over the video, framed like the video."""
        timeline = timeline or self.player.timeline
        if self.scene is None or timeline is None:
            self.camera_path = None
            return
        base, _ = self._frame_rect()
        self.camera_path = self.project.camera_track.path(timeline.clock, base)

    # Look and export frame

    def _apply_look(self) -> None:
        look = self.project.animation_look
        if self.scene is not None:
            self.player.set_camera(TableCamera.for_scene(self.scene, look))
        background = look.background
        rgb = ColorParam.rgb(background)
        self.canvas.checkerboard = rgb is None
        if rgb is not None:
            self.canvas.background = rgb
        self.canvas.update()

    def _on_show_frame(self, checked: bool) -> None:
        preferences.settings().setValue("animate/show_frame", "true" if checked else "false")
        self.follow.setEnabled(checked)
        self._show_export_frame()
        self._fit()

    def _on_follow(self, checked: bool) -> None:
        preferences.settings().setValue("animate/follow_camera", "true" if checked else "false")
        if checked:
            self._fit()

    def _on_view_changed(self) -> None:
        """Panning or zooming by hand (the canvas leaves fit mode): through the camera, it
        frames a new shot; otherwise it stops following."""
        if self.canvas.fit_mode:
            return
        if self.keys.viewfinder:
            self.keys.view_moved()
            self._show_export_frame()
        elif self.follow.isChecked() and self.show_frame.isChecked():
            self.follow.setChecked(False)

    def viewfinder_changed(self) -> None:
        """Looking through the camera shows the export frame and follows it."""
        on = self.keys.viewfinder
        for box in (self.show_frame, self.follow):
            box.setEnabled(not on and (box is self.show_frame or self.show_frame.isChecked()))
        self._show_export_frame()
        self._fit()

    def _framing(self) -> bool:
        """The export frame shows (asked for, or looking through the camera)."""
        return self.keys.viewfinder or self.show_frame.isChecked()

    def _following(self) -> bool:
        """The view keeps to the camera's shot."""
        return self.keys.viewfinder or (self.show_frame.isChecked() and self.follow.isChecked())

    def frame_rect(self):
        """The video's framing (what the camera shows at zoom 1), or None."""
        rect = self._frame_rect()
        return None if rect is None else rect[0]

    def _frame_rect(self):
        """(the video's framing, its pixel size): what the camera shows at zoom 1."""
        if self.scene is None:
            return None
        width, height = output_size(self.project.video_settings, self.scene)
        return view_rect(self.scene, width, height, self.project.video_settings), (width, height)

    def shot(self) -> Shot:
        """What the video shows at the player's current time: the camera's shot, or a
        framing made through the viewfinder and not keyed yet."""
        if self.keys.pending is not None:
            return self.keys.pending
        base, _ = self._frame_rect()
        if self.camera_path is None:
            return home_shot(base)
        return self.camera_path.shot(self.player.time)

    def _show_shot(self, _t: float = 0.0) -> None:
        """The player moved in time: the export frame (and a following view) moves along."""
        self.keys.time_moved()
        if self.scene is None or not self._framing():
            return
        self._show_export_frame()
        if self._following():
            self._fit()

    def _show_export_frame(self) -> None:
        shown = self._framing() and self.scene is not None
        shade = make_instances(4 if shown else 0)
        line = make_instances(1 if shown else 0)
        if shown:
            shot = self.shot()
            w, h = shot.size(self._frame_rect()[0])
            # Four bands around the frame, in its own (turned) frame: above, below, left,
            # right; centers relative to the frame's.
            boxes = [
                (0.0, -(h + FAR) / 2, w + 2 * FAR, FAR),
                (0.0, (h + FAR) / 2, w + 2 * FAR, FAR),
                (-(w + FAR) / 2, 0.0, FAR, h),
                ((w + FAR) / 2, 0.0, FAR, h),
            ]
            c, s = math.cos(shot.rotation), math.sin(shot.rotation)
            for i, (bx, by, bw, bh) in enumerate(boxes):
                shade[i]["pos"] = (shot.center[0] + c * bx - s * by,
                                   shot.center[1] + s * bx + c * by)  # fmt: skip
                shade[i]["size"] = (bw, bh)
            shade["rotation"] = shot.rotation
            shade["tint"] = (0.0, 0.0, 0.0, 1.0)
            shade["alpha"] = FRAME_DIM
            line["pos"] = shot.center
            line["size"] = (w, h)
            line["rotation"] = shot.rotation
            line["tint"] = (1.0, 1.0, 1.0, 1.0)
        self._shade.instances, self._frame_line.instances = shade, line
        self._shade.mark_dirty()
        self._frame_line.mark_dirty()
        self.canvas.update()

    def _fit(self) -> None:
        if self.scene is None:
            return
        rotation, shot = 0.0, None
        base, _ = self._frame_rect()
        if self._framing():
            shot = self.shot() if self._following() else home_shot(base)
            (cx, cy), (w, h), rotation = shot.center, shot.size(base), shot.rotation
            x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            x0, y0, x1, y1 = self.scene.bounds
        mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        self.canvas.fit_to(x0 - mx, y0 - my, x1 - x0 + 2 * mx, y1 - y0 + 2 * my,
                           rotation=rotation)  # fmt: skip
        if shot is not None:
            self.keys.fitted(shot, base)  # the frame's size on screen, for the viewfinder

    def _refresh_export(self) -> None:
        self.export_button.setEnabled(self.session.can_export and self.scene is not None)

    # Playback

    def toggle_play(self) -> None:
        self.transport.toggle_play()

    def restart(self) -> None:
        self.transport.to_start()
        self.player.play()

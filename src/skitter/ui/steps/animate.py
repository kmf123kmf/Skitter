"""Step 5: animate the construction of the finished mosaic.

Pick a choreography (core/animation), edit its settings and the background,
then play or scrub its timeline, and export it as a video (Export
Animation). The choreographies, the camera moves, the look and the video
settings live in the project, so the export renders exactly what this tab
shows. The Video group edits the settings that decide what the video shows
(size and framing; the Export Animation window edits the same settings, and
both stay in step); "Show export frame" outlines what the video shows at the
current moment (the camera move's shot) and dims the rest, and "Follow
camera" keeps the view on it (panning or zooming by hand stops following).
A new mosaic opens on its finished state; Play runs from the start. Playback
controls sit under the view (ui/widgets/transport.py).
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
    QVBoxLayout,
    QWidget,
)

from skitter.core.animation.camera import CameraPath, Shot, home_shot
from skitter.core.animation.look import TableCamera
from skitter.core.animation.video import output_size, view_rect
from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import ColorParam
from skitter.ui import preferences
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.render.player import TimelinePlayer
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.render.tile_textures import TileTextures
from skitter.ui.steps.base import StepPage, side_panel
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
        self.camera_path: CameraPath | None = None  # the camera move over the timeline
        self._stale = False  # manual picks changed the mosaic while this tab was hidden

        self.canvas = MosaicCanvas()
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)  # the playback keys work
        self.player = TimelinePlayer(self.canvas, self)
        self.player.time_changed.connect(self._show_shot)
        self.canvas.view_changed.connect(self._on_view_changed)
        self.transport = TransportBar(self.player)
        self.transport.status.setText("No mosaic: match tiles first.")
        self.canvas.fps_changed.connect(self.transport.set_fps)
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

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(view, stretch=1)
        layout.addWidget(
            side_panel(
                self._build_settings_group(),
                self._build_camera_group(),
                self._build_video_group(),
            )  # fmt: skip
        )

        session.project_replaced.connect(self._on_project_replaced)
        session.matching_changed.connect(self._load_scene)
        session.mosaic_edited.connect(self._on_mosaic_edited)
        session.animation_changed.connect(self._on_animation_changed)
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
        self.look_form = ParamForm()
        self.look_form.set_target(self.project.animation_look)
        self.look_form.changed.connect(lambda _: self.session.animation_edited())
        self.choreography_box = QComboBox()
        for choreography in self.project.choreographies.values():
            self.choreography_box.addItem(choreography.name, choreography.id)
            index = self.choreography_box.count() - 1
            self.choreography_box.setItemData(
                index, choreography.description, Qt.ItemDataRole.ToolTipRole
            )
        self.choreography_box.setCurrentIndex(
            self.choreography_box.findData(self.project.choreography_id)
        )
        self.choreography_box.currentIndexChanged.connect(self._on_choreography)
        self.description = muted(QLabel())
        self.form = ParamForm()
        self.form.changed.connect(lambda _: self.session.animation_edited())
        group = QGroupBox("Animation")
        layout = QVBoxLayout(group)
        layout.addWidget(self.look_form)
        top = QFormLayout()
        top.addRow("Choreography:", self.choreography_box)
        layout.addLayout(top)
        layout.addWidget(self.description)
        layout.addWidget(self.form)
        self._show_choreography()
        return group

    def _build_camera_group(self) -> QGroupBox:
        self.camera_box = QComboBox()
        for move in self.project.camera_moves.values():
            self.camera_box.addItem(move.name, move.id)
            index = self.camera_box.count() - 1
            self.camera_box.setItemData(index, move.description, Qt.ItemDataRole.ToolTipRole)
        self.camera_box.setCurrentIndex(self.camera_box.findData(self.project.camera_move_id))
        self.camera_box.currentIndexChanged.connect(self._on_camera_move)
        self.camera_description = muted(QLabel())
        self.camera_description.setWordWrap(True)
        self.camera_form = ParamForm()
        self.camera_form.changed.connect(lambda _: self.session.animation_edited())
        group = QGroupBox("Camera")
        layout = QVBoxLayout(group)
        top = QFormLayout()
        top.addRow("Move:", self.camera_box)
        layout.addLayout(top)
        layout.addWidget(self.camera_description)
        layout.addWidget(self.camera_form)
        self._show_camera_move()
        return group

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

    @property
    def choreography(self):
        return self.project.choreography

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
        timeline = self.choreography.timeline(scene, self.project.animation_look)
        self.player.set_timeline(timeline, time=timeline.duration)  # open on the finished mosaic
        self._plan_camera()
        self.player.set_camera(TableCamera.for_scene(scene, self.project.animation_look))
        self.player.set_content(self.textures.pages, self.textures.instances())
        self._sync_transport()
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
        self.choreography_box.blockSignals(True)
        self.choreography_box.setCurrentIndex(
            self.choreography_box.findData(self.project.choreography_id)
        )
        self.choreography_box.blockSignals(False)
        self._show_choreography()
        self.camera_box.blockSignals(True)
        self.camera_box.setCurrentIndex(self.camera_box.findData(self.project.camera_move_id))
        self.camera_box.blockSignals(False)
        self._show_camera_move()

    def _show_choreography(self) -> None:
        self.description.setText(self.choreography.description)
        self.form.set_target(self.choreography)

    def _on_choreography(self) -> None:
        self.project.choreography_id = self.choreography_box.currentData()
        self._show_choreography()
        self.session.animation_edited()

    def _show_camera_move(self) -> None:
        self.camera_description.setText(self.project.camera_move.description)
        self.camera_form.set_target(self.project.camera_move)

    def _on_camera_move(self) -> None:
        self.project.camera_move_id = self.camera_box.currentData()
        self._show_camera_move()
        self.session.animation_edited()

    def _on_animation_changed(self) -> None:
        """Choreography, camera, look or video settings edited (here or in Export Animation)."""
        self.video_form.refresh()
        self.look_form.refresh()
        self.camera_form.refresh()
        self._apply_look()
        self._replan()
        self._sync_transport()
        self._show_export_frame()
        if self.show_frame.isChecked() and self.follow.isChecked():
            self._fit()  # keep the whole frame in view as its shape changes

    def _replan(self) -> None:
        """New timeline for the current settings: at the same moment, or still at the end."""
        if self.scene is None:
            return
        at_end = self.player.time >= self.player.duration
        timeline = self.choreography.timeline(self.scene, self.project.animation_look)
        self._plan_camera(timeline)
        self.player.set_timeline(timeline, timeline.duration if at_end else None)

    def _sync_transport(self) -> None:
        """Frame steps match the export's frames; controls follow whether there's a timeline."""
        self.transport.frame_step = 1.0 / float(self.project.video_settings.fps)
        self.transport.refresh()

    def _plan_camera(self, timeline=None) -> None:
        """The camera move over the timeline, framed like the video."""
        timeline = timeline or self.player.timeline
        if self.scene is None or timeline is None:
            self.camera_path = None
            return
        base, _ = self._frame_rect()
        self.camera_path = self.project.camera_move.path(self.scene, timeline, base)

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
        """Panning or zooming by hand (the canvas leaves fit mode) stops following."""
        if self.follow.isChecked() and self.show_frame.isChecked() and not self.canvas.fit_mode:
            self.follow.setChecked(False)

    def _frame_rect(self):
        """(the video's framing, its pixel size): what the camera shows at zoom 1."""
        if self.scene is None:
            return None
        width, height = output_size(self.project.video_settings, self.scene)
        return view_rect(self.scene, width, height, self.project.video_settings), (width, height)

    def shot(self) -> Shot:
        """What the video shows at the player's current time (the camera move's shot)."""
        base, _ = self._frame_rect()
        if self.camera_path is None:
            return home_shot(base)
        return self.camera_path.shot(self.player.time)

    def _show_shot(self, _t: float = 0.0) -> None:
        """The player moved in time: the export frame (and a following view) moves along."""
        if self.scene is None or not self.show_frame.isChecked():
            return
        self._show_export_frame()
        if self.follow.isChecked():
            self._fit()

    def _show_export_frame(self) -> None:
        shown = self.show_frame.isChecked() and self.scene is not None
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
        rotation = 0.0
        if self.show_frame.isChecked():
            base, _ = self._frame_rect()
            shot = self.shot() if self.follow.isChecked() else home_shot(base)
            (cx, cy), (w, h), rotation = shot.center, shot.size(base), shot.rotation
            x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            x0, y0, x1, y1 = self.scene.bounds
        mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        self.canvas.fit_to(x0 - mx, y0 - my, x1 - x0 + 2 * mx, y1 - y0 + 2 * my,
                           rotation=rotation)  # fmt: skip

    def _refresh_export(self) -> None:
        self.export_button.setEnabled(self.session.can_export and self.scene is not None)

    # Playback

    def toggle_play(self) -> None:
        self.transport.toggle_play()

    def restart(self) -> None:
        self.transport.to_start()
        self.player.play()

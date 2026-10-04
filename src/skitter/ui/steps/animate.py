"""Step 5: animate the construction of the finished mosaic.

Pick a choreography (core/animation), edit its settings and the background,
then play or scrub its timeline, and export it as a video (Export
Animation). The choreographies, the look and the video settings live in the
project, so the export renders exactly what this tab shows. The Video group
edits the settings that decide what the video shows (size and framing; the
Export Animation window edits the same settings, and both stay in step);
"Show export frame" outlines that area and dims the rest. A new mosaic opens
on its finished state; Play runs from the start.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

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

SLIDER_STEPS = 1000
MARGIN = 0.05  # view margin around the mosaic, share of its size
FRAME_DIM = 0.6  # opacity of the shade outside the export frame
FAR = 1e7  # mosaic units: "everywhere" for the shade
FRAME_SETTINGS = ("resolution", "width", "height", "framing", "margin")  # what the video shows


class AnimateStep(StepPage):
    title = "Animate"

    export_requested = Signal()  # the user pressed Export Animation…

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.scene: MosaicScene | None = None
        self.textures: TileTextures | None = None
        self._stale = False  # manual picks changed the mosaic while this tab was hidden

        self.canvas = MosaicCanvas()
        self.player = TimelinePlayer(self.canvas, self)
        self.player.time_changed.connect(self._show_time)
        self.player.playing_changed.connect(self._show_playing)
        self._shade = self.canvas.add_layer(SpriteLayer(None, make_instances(0)))
        self._frame_line = self.canvas.add_layer(
            SpriteLayer(None, make_instances(0), outline_px=1.5, edge_px=1.0)
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(
            side_panel(
                self._build_playback_group(),
                self._build_settings_group(),
                self._build_video_group(),
            )  # fmt: skip
        )

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

    def _build_playback_group(self) -> QGroupBox:
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        restart = QPushButton("Restart")
        restart.clicked.connect(self.restart)
        buttons = QHBoxLayout()
        buttons.addWidget(self.play_button)
        buttons.addWidget(restart)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.sliderMoved.connect(self._on_slider)
        self.time_label = QLabel("—")
        self.fps_label = muted(QLabel())
        self.canvas.fps_changed.connect(
            lambda fps: self.fps_label.setText(f"{fps:.0f} fps" if fps else "")
        )
        self.status = muted(QLabel("No mosaic: match tiles first."))
        group = QGroupBox("Playback")
        layout = QVBoxLayout(group)
        layout.addLayout(buttons)
        layout.addWidget(self.slider)
        times = QHBoxLayout()
        times.addWidget(self.time_label, stretch=1)
        times.addWidget(self.fps_label)
        layout.addLayout(times)
        layout.addWidget(self.status)
        return group

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

    def _build_video_group(self) -> QGroupBox:
        self.show_frame = QCheckBox("Show export frame")
        self.show_frame.setToolTip(
            "Outline what the exported video shows (its size and framing) and dim the rest."
        )
        self.show_frame.setChecked(
            preferences.settings().value("animate/show_frame", "false") == "true"
        )
        self.show_frame.toggled.connect(self._on_show_frame)
        self.video_form = ParamForm()
        self.video_form.set_target(self.project.video_settings, only=FRAME_SETTINGS)
        self.video_form.changed.connect(lambda _: self.session.animation_edited())
        self.export_button = QPushButton("Export Animation…")
        self.export_button.setToolTip("Format, frame rate, quality and more, then export.")
        self.export_button.clicked.connect(self.export_requested)
        group = QGroupBox("Video")
        layout = QVBoxLayout(group)
        layout.addWidget(self.show_frame)
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
            self.textures.status_changed.disconnect(self.status.setText)
        self.scene, self.textures = scene, self.session.textures
        if scene is None or self.textures is None or not len(scene):
            self.scene = self.textures = None
            self.player.set_timeline(None)
            self.player.set_content(None, None)
            self.status.setText("No mosaic: match tiles first.")
            self._show_export_frame()
            self.state_changed.emit()
            return
        self.textures.changed.connect(self._on_textures)
        self.textures.patched.connect(self._on_textures)  # crops for picks read meanwhile
        self.textures.status_changed.connect(self.status.setText)
        self.status.setText(self.textures.status)
        timeline = self.choreography.timeline(scene, self.project.animation_look)
        self.player.set_timeline(timeline, time=timeline.duration)  # open on the finished mosaic
        self.player.set_camera(TableCamera.for_scene(scene, self.project.animation_look))
        self.player.set_content(self.textures.pages, self.textures.instances())
        self._show_export_frame()
        self._fit()
        self.state_changed.emit()

    def _on_textures(self, *_) -> None:
        if not self.isVisible() and self._stale:
            return  # reloaded on entering
        self.player.set_content(self.textures.pages, self.textures.instances())

    def _show_choreography(self) -> None:
        self.description.setText(self.choreography.description)
        self.form.set_target(self.choreography)

    def _on_choreography(self) -> None:
        self.project.choreography_id = self.choreography_box.currentData()
        self._show_choreography()
        self.session.animation_edited()

    def _on_animation_changed(self) -> None:
        """Choreography, look or video settings edited (here or in Export Animation)."""
        self.video_form.refresh()
        self.look_form.refresh()
        self._apply_look()
        self._replan()
        self._show_export_frame()
        if self.show_frame.isChecked():
            self._fit()  # keep the whole frame in view as its shape changes

    def _replan(self) -> None:
        """New timeline for the current settings: at the same moment, or still at the end."""
        if self.scene is None:
            return
        at_end = self.player.time >= self.player.duration
        timeline = self.choreography.timeline(self.scene, self.project.animation_look)
        self.player.set_timeline(timeline, timeline.duration if at_end else None)

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
        self._show_export_frame()
        self._fit()

    def _frame_rect(self):
        if self.scene is None:
            return None
        width, height = output_size(self.project.video_settings, self.scene)
        return view_rect(self.scene, width, height, self.project.video_settings), (width, height)

    def _show_export_frame(self) -> None:
        shown = self.show_frame.isChecked() and self.scene is not None
        shade = make_instances(4 if shown else 0)
        line = make_instances(1 if shown else 0)
        if shown:
            (x, y, w, h), _ = self._frame_rect()
            # Four bands around the frame: above, below, left, right.
            boxes = [
                (x - FAR, y - FAR, w + 2 * FAR, FAR),
                (x - FAR, y + h, w + 2 * FAR, FAR),
                (x - FAR, y, FAR, h),
                (x + w, y, FAR, h),
            ]
            for i, (bx, by, bw, bh) in enumerate(boxes):
                shade[i]["pos"] = (bx + bw / 2, by + bh / 2)
                shade[i]["size"] = (bw, bh)
            shade["tint"] = (0.0, 0.0, 0.0, 1.0)
            shade["alpha"] = FRAME_DIM
            line["pos"] = (x + w / 2, y + h / 2)
            line["size"] = (w, h)
            line["tint"] = (1.0, 1.0, 1.0, 1.0)
        self._shade.instances, self._frame_line.instances = shade, line
        self._shade.mark_dirty()
        self._frame_line.mark_dirty()
        self.canvas.update()

    def _fit(self) -> None:
        if self.scene is None:
            return
        if self.show_frame.isChecked():
            (x, y, w, h), _ = self._frame_rect()
            x0, y0, x1, y1 = x, y, x + w, y + h
        else:
            x0, y0, x1, y1 = self.scene.bounds
        mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        self.canvas.fit_to(x0 - mx, y0 - my, x1 - x0 + 2 * mx, y1 - y0 + 2 * my)

    def _refresh_export(self) -> None:
        self.export_button.setEnabled(self.session.can_export and self.scene is not None)

    # Playback

    def toggle_play(self) -> None:
        if self.player.playing:
            self.player.pause()
        else:
            self.player.play()

    def restart(self) -> None:
        self.player.pause()
        self.player.seek(0.0)
        self.player.play()

    def _on_slider(self, value: int) -> None:
        self.player.pause()
        self.player.seek(value / SLIDER_STEPS * self.player.duration)

    def _show_time(self, t: float) -> None:
        duration = self.player.duration
        self.time_label.setText(f"{t:.2f} s of {duration:.2f} s" if duration else "—")
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(round(t / duration * SLIDER_STEPS) if duration else 0)
            self.slider.blockSignals(False)

    def _show_playing(self, playing: bool) -> None:
        self.play_button.setText("Pause" if playing else "Play")

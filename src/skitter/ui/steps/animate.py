"""Step 5: animate the construction of the finished mosaic.

Pick a choreography (core/animation), edit its settings, then play or scrub
its timeline. The canvas shows the session's scene with its shared tile
textures (full-size tiles once loaded). A new mosaic opens on its finished
state; Play runs the animation from the start.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from skitter.core.animation import Choreography, choreography_types
from skitter.core.scene import MosaicScene
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.render.player import TimelinePlayer
from skitter.ui.render.tile_textures import TileTextures
from skitter.ui.steps.base import StepPage, side_panel
from skitter.ui.widgets.param_form import ParamForm

SLIDER_STEPS = 1000
MARGIN = 0.05  # view margin around the mosaic, share of its size


def _muted(label: QLabel) -> QLabel:
    label.setWordWrap(True)
    label.setStyleSheet("color: palette(placeholder-text);")
    return label


class AnimateStep(StepPage):
    title = "Animate"

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.scene: MosaicScene | None = None
        self.textures: TileTextures | None = None
        self.choreographies: dict[str, Choreography] = {
            cls.id: cls() for cls in choreography_types()
        }

        self.canvas = MosaicCanvas()
        self.player = TimelinePlayer(self.canvas, self)
        self.player.time_changed.connect(self._show_time)
        self.player.playing_changed.connect(self._show_playing)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(side_panel(self._build_playback_group(), self._build_settings_group()))

        session.matching_changed.connect(self._load_scene)
        self._load_scene()

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
        self.fps_label = _muted(QLabel())
        self.canvas.fps_changed.connect(
            lambda fps: self.fps_label.setText(f"{fps:.0f} fps" if fps else "")
        )
        self.status = _muted(QLabel("No mosaic: match tiles first."))
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
        self.choreography_box = QComboBox()
        for choreography in self.choreographies.values():
            self.choreography_box.addItem(choreography.name, choreography.id)
            index = self.choreography_box.count() - 1
            self.choreography_box.setItemData(
                index, choreography.description, Qt.ItemDataRole.ToolTipRole
            )
        self.choreography_box.currentIndexChanged.connect(self._on_choreography)
        self.description = _muted(QLabel())
        self.form = ParamForm()
        self.form.changed.connect(lambda _: self._replan())
        group = QGroupBox("Animation")
        layout = QVBoxLayout(group)
        top = QFormLayout()
        top.addRow("Choreography:", self.choreography_box)
        layout.addLayout(top)
        layout.addWidget(self.description)
        layout.addWidget(self.form)
        self._show_choreography()
        return group

    @property
    def choreography(self) -> Choreography:
        return self.choreographies[self.choreography_box.currentData()]

    # Step

    def is_complete(self) -> bool:
        return self.scene is not None

    def on_leave(self) -> None:
        self.player.pause()

    # Scene

    def _load_scene(self) -> None:
        scene = self.session.scene
        if scene is self.scene:
            return
        self.player.pause()
        if self.textures is not None:
            self.textures.changed.disconnect(self._on_textures)
            self.textures.status_changed.disconnect(self.status.setText)
        self.scene, self.textures = scene, self.session.textures
        if scene is None or self.textures is None or not len(scene):
            self.scene = self.textures = None
            self.player.set_timeline(None)
            self.player.set_content(None, None)
            self.status.setText("No mosaic: match tiles first.")
            self.state_changed.emit()
            return
        self.textures.changed.connect(self._on_textures)
        self.textures.status_changed.connect(self.status.setText)
        self.status.setText(self.textures.status)
        timeline = self.choreography.timeline(scene)
        self.player.set_timeline(timeline, time=timeline.duration)  # open on the finished mosaic
        self.player.set_content(self.textures.pages, self.textures.instances())
        x0, y0, x1, y1 = scene.bounds
        mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        self.canvas.fit_to(x0 - mx, y0 - my, x1 - x0 + 2 * mx, y1 - y0 + 2 * my)
        self.state_changed.emit()

    def _on_textures(self) -> None:
        self.player.set_content(self.textures.pages, self.textures.instances())

    def _show_choreography(self) -> None:
        self.description.setText(self.choreography.description)
        self.form.set_target(self.choreography)

    def _on_choreography(self) -> None:
        self._show_choreography()
        self._replan()

    def _replan(self) -> None:
        """New timeline for the current settings: at the same moment, or still at the end."""
        if self.scene is None:
            return
        at_end = self.player.time >= self.player.duration
        timeline = self.choreography.timeline(self.scene)
        self.player.set_timeline(timeline, timeline.duration if at_end else None)

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

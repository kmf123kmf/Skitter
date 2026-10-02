"""Build Animation Preview: try choreographies on the current mosaic.

A temporary home for animated construction (Demo menu) until it gets its
own workflow step: pick a choreography, edit its settings, then play or
scrub the timeline. It shows the session's scene with its shared textures.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from skitter.core.animation import Choreography, choreography_types
from skitter.core.scene import MosaicScene
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.render.player import TimelinePlayer
from skitter.ui.render.tile_textures import TileTextures
from skitter.ui.steps.base import side_panel
from skitter.ui.widgets.param_form import ParamForm

SLIDER_STEPS = 1000
MARGIN = 0.05  # view margin around the mosaic, share of its size


class BuildPreviewWindow(QMainWindow):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Build Animation Preview")
        self.resize(1280, 800)
        self.scene: MosaicScene | None = None
        self.textures: TileTextures | None = None
        self.choreographies: dict[str, Choreography] = {
            cls.id: cls() for cls in choreography_types()
        }

        self.canvas = MosaicCanvas()
        self.player = TimelinePlayer(self.canvas, self)
        self.player.time_changed.connect(self._show_time)
        self.player.playing_changed.connect(self._show_playing)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(side_panel(self._build_playback_group(), self._build_settings_group()))
        self.setCentralWidget(central)
        fps = QLabel()
        self.statusBar().addPermanentWidget(fps)
        self.canvas.fps_changed.connect(lambda value: fps.setText(f"{value:.0f} fps"))

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
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: palette(placeholder-text);")
        group = QGroupBox("Playback")
        form = QVBoxLayout(group)
        form.addLayout(buttons)
        form.addWidget(self.slider)
        form.addWidget(self.time_label)
        form.addWidget(self.status)
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
        self.form = ParamForm()
        self.form.changed.connect(lambda _: self._replan())
        group = QGroupBox("Animation")
        layout = QVBoxLayout(group)
        top = QFormLayout()
        top.addRow("Choreography:", self.choreography_box)
        layout.addLayout(top)
        layout.addWidget(self.form)
        self.form.set_target(self.choreography)
        return group

    @property
    def choreography(self) -> Choreography:
        return self.choreographies[self.choreography_box.currentData()]

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
            return
        self.textures.changed.connect(self._on_textures)
        self.textures.status_changed.connect(self.status.setText)
        self.status.setText(self.textures.status)
        self.player.set_timeline(self.choreography.timeline(scene), time=0.0)
        self.player.set_content(self.textures.pages, self.textures.instances())
        x0, y0, x1, y1 = scene.bounds
        mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        self.canvas.fit_to(x0 - mx, y0 - my, x1 - x0 + 2 * mx, y1 - y0 + 2 * my)

    def _on_textures(self) -> None:
        self.player.set_content(self.textures.pages, self.textures.instances())

    def _on_choreography(self) -> None:
        self.form.set_target(self.choreography)
        self._replan(time=0.0)

    def _replan(self, time: float | None = None) -> None:
        if self.scene is not None:
            self.player.set_timeline(self.choreography.timeline(self.scene), time)

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

    def closeEvent(self, event) -> None:
        self.player.pause()
        super().closeEvent(event)

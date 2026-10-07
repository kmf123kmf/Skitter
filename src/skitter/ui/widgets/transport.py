"""Playback controls for a TimelinePlayer, laid out under its canvas.

A full-width scrubber over one row: the time on the left; to start, previous
frame, play / pause, next frame and to end in the middle; loop, speed, a
status line and the frame rate on the right.

Keys (while the bar's page has focus, see `install_shortcuts`): Space plays
or pauses, Home / End jump to the start / end, Left / Right step one frame
(`frame_step`, the export's frame time, so stepping lands on exported
frames) and Shift+Left / Right one second.
"""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QFontDatabase, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from skitter.ui import icons
from skitter.ui.render.player import TimelinePlayer
from skitter.ui.style import muted

SLIDER_STEPS = 10_000
SPEEDS = (0.25, 0.5, 1.0, 2.0)
ICON = QSize(20, 20)


class TransportBar(QWidget):
    def __init__(self, player: TimelinePlayer, parent=None):
        super().__init__(parent)
        self.player = player
        self.frame_step = 1 / 30  # seconds: one frame of the video (set by the page)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setToolTip("Drag to scrub through the animation.")
        self.slider.sliderMoved.connect(self.scrub)
        self.slider.sliderPressed.connect(lambda: self.scrub(self.slider.value()))

        self.time_label = QLabel("—")
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSizeF(self.font().pointSizeF())  # digits that don't jiggle, at text size
        self.time_label.setFont(mono)

        self.start_action = self._action(icons.to_start(), "To start", self.to_start, "Home")
        self.back_action = self._action(icons.step_back(), "Previous frame",
                                        lambda: self.step(-1), "Left")  # fmt: skip
        self.play_action = self._action(icons.play(), "Play", self.toggle_play, "Space")
        self.forward_action = self._action(icons.step_forward(), "Next frame",
                                           lambda: self.step(1), "Right")  # fmt: skip
        self.end_action = self._action(icons.to_end(), "To end", self.to_end, "End")
        self.second_back = self._action(None, "Back a second", lambda: self.jump(-1.0),
                                        "Shift+Left")  # fmt: skip
        self.second_forward = self._action(None, "Forward a second", lambda: self.jump(1.0),
                                           "Shift+Right")  # fmt: skip
        self.loop_action = QAction(icons.loop(), "Loop", self)
        self.loop_action.setToolTip("Loop: play on from the start after the end.")
        self.loop_action.setCheckable(True)
        self.loop_action.toggled.connect(self._on_loop)

        self.play_button = self._button(self.play_action)
        self.play_button.setIconSize(QSize(26, 26))
        self.speed = QComboBox()
        for speed in SPEEDS:
            self.speed.addItem(f"{speed:g}×", speed)
        self.speed.setCurrentIndex(SPEEDS.index(1.0))
        self.speed.setToolTip("Playback speed (the export always plays at 1×).")
        self.speed.currentIndexChanged.connect(self._on_speed)
        self.status = muted(QLabel())
        self.fps_label = muted(QLabel())

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self._button(self.start_action))
        controls.addWidget(self._button(self.back_action))
        controls.addWidget(self.play_button)
        controls.addWidget(self._button(self.forward_action))
        controls.addWidget(self._button(self.end_action))
        right = QHBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.addStretch()
        right.addWidget(self.status)
        right.addSpacing(8)
        right.addWidget(self.fps_label)
        right.addSpacing(8)
        right.addWidget(self._button(self.loop_action))
        right.addWidget(self.speed)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        # Equal stretch either side keeps the buttons centered under the view.
        left = QHBoxLayout()
        left.addWidget(self.time_label)
        left.addStretch()
        row.addLayout(left, stretch=1)
        row.addLayout(controls)
        row.addLayout(right, stretch=1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 6)
        layout.setSpacing(2)
        layout.addWidget(self.slider)
        layout.addLayout(row)

        player.time_changed.connect(self._show_time)
        player.playing_changed.connect(self._show_playing)
        self._show_time(player.time)
        self.refresh()

    # Construction

    def _action(self, icon, text: str, slot, key: str | None = None) -> QAction:
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(icon)
        if key:
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            native = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
            action.setToolTip(f"{text} ({native})")
        action.triggered.connect(slot)
        return action

    def _button(self, action: QAction) -> QToolButton:
        button = QToolButton()
        button.setDefaultAction(action)
        button.setAutoRaise(True)
        button.setIconSize(ICON)
        return button

    def install_shortcuts(self, widget: QWidget) -> None:
        """Make the keys work anywhere in widget (the page the bar is on)."""
        for action in (self.start_action, self.back_action, self.play_action,
                       self.forward_action, self.end_action, self.second_back,
                       self.second_forward):  # fmt: skip
            widget.addAction(action)

    # Actions

    def toggle_play(self) -> None:
        if self.player.playing:
            self.player.pause()
        else:
            self.player.play()

    def to_start(self) -> None:
        self.player.pause()
        self.player.seek(0.0)

    def to_end(self) -> None:
        self.player.pause()
        self.player.seek(self.player.duration)

    def step(self, frames: int) -> None:
        """Pause and move by whole frames, onto the frame grid."""
        player = self.player
        player.pause()
        index = round(player.time / self.frame_step) + frames
        player.seek(index * self.frame_step)

    def jump(self, seconds: float) -> None:
        self.player.pause()
        self.player.seek(self.player.time + seconds)

    def scrub(self, value: int) -> None:
        self.player.pause()
        self.player.seek(value / SLIDER_STEPS * self.player.duration)

    def _on_loop(self, checked: bool) -> None:
        self.player.loop = checked

    def _on_speed(self) -> None:
        player = self.player
        playing = player.playing
        if playing:  # restart the clock at the new speed from where it is
            player.pause()
        player.speed = float(self.speed.currentData())
        if playing:
            player.play()

    # Display

    def refresh(self) -> None:
        """Enable the controls while there is something to play."""
        has = self.player.timeline is not None
        for widget in (self.slider, self.speed):
            widget.setEnabled(has)
        for action in (self.start_action, self.back_action, self.play_action,
                       self.forward_action, self.end_action, self.second_back,
                       self.second_forward, self.loop_action):  # fmt: skip
            action.setEnabled(has)
        self._show_time(self.player.time)

    def set_fps(self, fps: float) -> None:
        self.fps_label.setText(f"{fps:.0f} fps" if fps else "")

    def _show_time(self, t: float) -> None:
        duration = self.player.duration
        self.time_label.setText(f"{t:6.2f} s / {duration:.2f} s" if duration else "—")
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(round(t / duration * SLIDER_STEPS) if duration else 0)
            self.slider.blockSignals(False)

    def _show_playing(self, playing: bool) -> None:
        self.play_action.setIcon(icons.pause() if playing else icons.play())
        self.play_action.setText("Pause" if playing else "Play")
        key = self.play_action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
        self.play_action.setToolTip(f"{self.play_action.text()} ({key})")

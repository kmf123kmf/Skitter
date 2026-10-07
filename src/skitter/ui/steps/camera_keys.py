"""Camera keyframes on the Animate tab: the viewfinder, the keys and their editing.

`CameraKeys` works for an AnimateStep (`step`):

- **Look through camera** (C): the view becomes the camera. The export frame
  stays put in the view, and panning, zooming (wheel) and turning
  (Shift+wheel) the mosaic under it frames a shot; Add Key (K) keeps it at
  the current time (replacing a key there). Moving in time drops a framing
  that wasn't keyed.
- Keys show on the transport's timeline strip: click one to select it and go
  there, drag it to retime it, right-click it for Stop / Pass through, how
  the camera moves on, and Delete. [ and ] jump to the previous / next key.
- The Keyframes group shows the selected key's exact values (time, where the
  frame's middle is across and down the mosaic, zoom, turn, stop, then) for
  editing, and whether keys keep their share of the animation when its
  length changes.

Every change replaces the project's CameraTrack through the session
(`set_camera_track`), so the preview, the export and unsaved-change tracking
all see it.
"""

import math
from dataclasses import replace

from PySide6.QtCore import QObject, QSize, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QMenu,
    QToolButton,
    QVBoxLayout,
)

from skitter.core.animation.camera import Shot
from skitter.core.animation.keyframes import MOTIONS, CameraKey, CameraTrack, KeyTime
from skitter.core.animation.video import content_rect
from skitter.core.slicing.params import BoolParam, ChoiceParam, Configurable, FloatParam
from skitter.ui import icons
from skitter.ui.style import muted
from skitter.ui.widgets.param_form import ParamForm


class KeyFields(Configurable):
    """The selected key's values, as the Keyframes group edits them."""

    time = FloatParam(0.0, "Time", min=0.0, max=3600.0, step=0.1, decimals=2, suffix=" s")
    across = FloatParam(50.0, "Across", min=-100.0, max=200.0, step=1.0, decimals=1,
                        suffix="%", help="The frame's middle, from the mosaic's left edge "
                                         "(0%) to its right (100%).")  # fmt: skip
    down = FloatParam(50.0, "Down", min=-100.0, max=200.0, step=1.0, decimals=1, suffix="%",
                      help="The frame's middle, from the mosaic's top (0%) to its "
                           "bottom (100%).")  # fmt: skip
    zoom = FloatParam(1.0, "Zoom", min=0.1, max=50.0, step=0.1, decimals=2, suffix="×",
                      help="Compared with the video's framing (1: the framing "
                           "itself).")  # fmt: skip
    turn = FloatParam(0.0, "Turn", min=-3600.0, max=3600.0, step=5.0, decimals=1, suffix="°",
                      help="The camera's turn; the picture turns the other way. Keys can "
                           "be more than a turn apart to spin.")  # fmt: skip
    stop = BoolParam(True, "Comes to rest",
                     help="The camera eases to a stop here; off: it moves on through "
                          "without slowing.")  # fmt: skip
    motion = ChoiceParam("smooth", "Then", choices=MOTIONS,
                         help="How the camera moves on to the next key: smoothly, at a "
                              "steady pace, or holding here and then cutting.")  # fmt: skip


class CameraKeys(QObject):
    def __init__(self, step):
        super().__init__(step)
        self.step = step
        self.session = step.session
        self.selected = -1  # index into the track's keys
        self.pending: Shot | None = None  # a viewfinder framing not keyed yet
        self._frame_px: tuple[float, float] | None = None  # the frame on screen, viewfinder
        self._fitting = False

        self.viewfinder_action = self._action("Look through camera", self.set_viewfinder, "C")
        self.viewfinder_action.setCheckable(True)
        self.viewfinder_action.setToolTip(
            "Look through camera (C): pan, zoom (wheel) and turn (Shift+wheel) the mosaic "
            "under the export frame to frame a shot, then Add Key (K)."
        )
        self.add_action = self._action("Add key", self.add_key, "K")
        self.delete_action = self._action("Delete key", self.delete_key, "Delete")
        self.previous_action = self._action("Previous key", lambda: self.jump(-1), "[")
        self.next_action = self._action("Next key", lambda: self.jump(1), "]")

        strip = step.transport.timeline
        strip.key_clicked.connect(self.select_and_go)
        strip.key_moved.connect(self.retime)
        strip.key_menu.connect(self.show_menu)
        for action, icon in ((self.viewfinder_action, icons.viewfinder()),
                             (self.previous_action, icons.previous_key()),
                             (self.add_action, icons.add_key()),
                             (self.next_action, icons.next_key()),
                             (self.delete_action, icons.delete_key())):  # fmt: skip
            action.setIcon(icon)
            button = QToolButton()
            button.setDefaultAction(action)
            button.setAutoRaise(True)
            button.setIconSize(QSize(20, 20))
            step.transport.left_tools.addWidget(button)

    def _action(self, text: str, slot, key: str) -> QAction:
        action = QAction(text, self.step)
        action.setShortcut(QKeySequence(key))
        action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        native = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
        action.setToolTip(f"{text} ({native})")
        action.triggered.connect(slot)
        return action

    def install_shortcuts(self, widget) -> None:
        for action in (self.viewfinder_action, self.add_action, self.delete_action,
                       self.previous_action, self.next_action):  # fmt: skip
            widget.addAction(action)

    def build_group(self) -> QGroupBox:
        self.count = QLabel()
        self.hint = muted(QLabel())
        self.hint.setWordWrap(True)
        self.stretch = QCheckBox("Keep keys in step with the animation")
        self.stretch.setToolTip(
            "On: keys during the animation keep their share of it when its length changes. "
            "Off: they keep their seconds from its start."
        )
        self.stretch.toggled.connect(self._on_stretch)
        self.fields = KeyFields()
        self.form = ParamForm()
        self.form.set_target(self.fields)
        self.form.changed.connect(self._on_field)
        group = QGroupBox("Keyframes")
        layout = QVBoxLayout(group)
        layout.addWidget(self.count)
        layout.addWidget(self.stretch)
        layout.addWidget(self.form)
        layout.addWidget(self.hint)
        self.refresh()
        return group

    # The track

    @property
    def track(self) -> CameraTrack:
        return self.session.project.camera_track

    def clock(self):
        timeline = self.step.player.timeline
        return None if timeline is None else timeline.clock

    def _set(self, track: CameraTrack, select: CameraKey | None = None) -> None:
        self.pending = None
        self.session.set_camera_track(track)  # replans and redraws (animation_changed)
        self.selected = track.keys.index(select) if select in track.keys else -1
        self.refresh()

    def _key_index_at(self, t: float, tolerance: float = 1e-3) -> int:
        clock = self.clock()
        for i, time in enumerate(self.track.times(clock)):
            if abs(time - t) <= tolerance:
                return i
        return -1

    # Actions

    def add_key(self) -> None:
        """Key what the camera shows now (the viewfinder's framing, if any) at this time."""
        clock = self.clock()
        if clock is None:
            return
        t = self.step.player.time
        shot = self.pending or self.step.shot()
        at = self._key_index_at(t)
        old = self.track.keys[at] if at >= 0 else CameraKey(KeyTime("lead", 0.0), shot)
        key = CameraKey(KeyTime.at(t, clock, self.track.stretch), shot, old.stop, old.motion)
        self._set(self.track.with_key(key, clock), select=key)

    def delete_key(self) -> None:
        index = self.selected if self.selected >= 0 else self._key_index_at(self.step.player.time)
        if index >= 0:
            self._set(self.track.without(index))

    def jump(self, direction: int) -> None:
        clock = self.clock()
        if clock is None or not self.track.keys:
            return
        t = self.step.player.time
        times = self.track.times(clock)
        later = [i for i, k in enumerate(times) if k > t + 1e-6]
        earlier = [i for i, k in enumerate(times) if k < t - 1e-6]
        targets = later[:1] if direction > 0 else earlier[-1:]
        if targets:
            self.select_and_go(targets[0])

    def select_and_go(self, index: int) -> None:
        self.selected = index
        self.step.transport.seek_to(float(self.track.times(self.clock())[index]))
        self.refresh()

    def retime(self, index: int, t: float) -> None:
        clock = self.clock()
        key = self.track.keys[index]
        moved = replace(key, time=KeyTime.at(t, clock, self.track.stretch))
        track = self.track.without(index).with_key(moved, clock)
        self._set(track, select=moved)

    def show_menu(self, index: int, position) -> None:
        self.selected = index
        self.refresh()
        key = self.track.keys[index]
        menu = QMenu(self.step)
        stop = menu.addAction("Comes to rest")
        stop.setCheckable(True)
        stop.setChecked(key.stop)
        stop.triggered.connect(lambda on: self._edit(index, stop=on))
        then = menu.addMenu("Then")
        for value, label in MOTIONS:
            action = then.addAction(label)
            action.setCheckable(True)
            action.setChecked(key.motion == value)
            action.triggered.connect(lambda _=False, v=value: self._edit(index, motion=v))
        menu.addSeparator()
        menu.addAction("Delete key").triggered.connect(lambda: self._set(self.track.without(index)))
        menu.popup(position)

    def _edit(self, index: int, **changes) -> None:
        key = replace(self.track.keys[index], **changes)
        self._set(self.track.replaced(index, key), select=key)

    def _on_stretch(self, on: bool) -> None:
        clock = self.clock()
        if clock is not None and on != self.track.stretch:
            selected = self.selected
            self._set(self.track.restretched(on, clock))
            self.selected = selected
            self.refresh()

    def _on_field(self, name: str) -> None:
        """The Keyframes form edited the selected key."""
        clock = self.clock()
        if self.selected < 0 or clock is None:
            return
        f, key = self.fields, self.track.keys[self.selected]
        x0, y0, x1, y1 = content_rect(self.step.scene)
        shot = Shot((x0 + f.across / 100 * (x1 - x0), y0 + f.down / 100 * (y1 - y0)),
                    f.zoom, math.radians(f.turn))  # fmt: skip
        time = KeyTime.at(f.time, clock, self.track.stretch) if name == "time" else key.time
        edited = CameraKey(time, shot, f.stop, f.motion)
        track = self.track.replaced(self.selected, edited)
        if name == "time":  # it may now share a moment with another key: that one goes
            track = self.track.without(self.selected).with_key(edited, clock)
        self._set(track, select=edited)

    # The viewfinder

    @property
    def viewfinder(self) -> bool:
        return self.viewfinder_action.isChecked()

    def set_viewfinder(self, on: bool) -> None:
        self.viewfinder_action.setChecked(on)
        self.pending = None
        self.step.canvas.rotatable = on
        self.step.viewfinder_changed()
        self.refresh()

    def fitted(self, shot: Shot, base) -> None:
        """The step framed this shot in the view: remember the frame's size on screen."""
        zoom = self.step.canvas.camera.zoom
        w, h = shot.size(base)
        self._frame_px = (w * zoom, h * zoom)

    def view_moved(self) -> None:
        """The view was panned, zoomed or turned by hand while looking through the camera:
        the frame's content is a new framing."""
        cam, base = self.step.canvas.camera, self.step.frame_rect()
        if self._frame_px is None or base is None:
            return
        zoom = base[2] * cam.zoom / self._frame_px[0]
        self.pending = Shot((float(cam.center[0]), float(cam.center[1])), float(zoom),
                            float(cam.rotation))  # fmt: skip
        self.refresh()

    def time_moved(self) -> None:
        """Moving in time drops a framing that wasn't keyed."""
        if self.pending is not None:
            self.pending = None
            self.refresh()

    # Display

    def refresh(self) -> None:
        clock = self.clock()
        track = self.track
        has = clock is not None
        keys = track.keys
        if self.selected >= len(keys):
            self.selected = -1
        strip = self.step.transport.timeline
        if has:
            times = track.times(clock)
            strip.set_keys([(float(t), k.stop, k.motion) for t, k in zip(times, keys,
                                                                          strict=True)],
                           self.selected)  # fmt: skip
        else:
            strip.set_keys([], -1)
        for action in (self.viewfinder_action, self.add_action):
            action.setEnabled(has)
        self.delete_action.setEnabled(has and bool(keys))
        self.previous_action.setEnabled(has and bool(keys))
        self.next_action.setEnabled(has and bool(keys))
        if not hasattr(self, "count"):
            return
        noun = "key" if len(keys) == 1 else "keys"
        self.count.setText(f"{len(keys)} {noun}" + ("" if keys else
                           ": the camera move below runs (keys replace it)."))  # fmt: skip
        self.stretch.blockSignals(True)
        self.stretch.setChecked(track.stretch)
        self.stretch.blockSignals(False)
        self.form.setVisible(has and self.selected >= 0)
        if has and self.selected >= 0:
            self._show_key(keys[self.selected], float(track.times(clock)[self.selected]))
        if self.pending is not None:
            hint = "Framing changed: Add Key (K) keeps it here; moving in time drops it."
        elif self.viewfinder:
            hint = ("Looking through the camera: drag to pan, wheel to zoom, Shift+wheel to "
                    "turn, then Add Key (K).")  # fmt: skip
        elif not keys:
            hint = ("Look through camera (C) to frame a shot, or Add Key (K) to key what the "
                    "camera shows now.")  # fmt: skip
        else:
            hint = "Click a key on the timeline to select it; drag it to retime it."
        self.hint.setText(hint)

    def _show_key(self, key: CameraKey, t: float) -> None:
        x0, y0, x1, y1 = content_rect(self.step.scene)
        values = dict(
            time=t,
            across=(key.shot.center[0] - x0) / max(x1 - x0, 1e-9) * 100,
            down=(key.shot.center[1] - y0) / max(y1 - y0, 1e-9) * 100,
            zoom=key.shot.zoom, turn=math.degrees(key.shot.rotation),
        )  # fmt: skip
        params = {p.name: p for p in KeyFields.params()}
        for name, value in values.items():  # shown within each field's range
            values[name] = min(max(float(value), params[name].min), params[name].max)
        self.fields.update(**values, stop=key.stop, motion=key.motion)
        self.form.refresh()

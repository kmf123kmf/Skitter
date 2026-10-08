"""Camera keyframes on the Animate tab: the viewfinder, the keys and their editing.

`CameraKeys` works for an AnimateStep (`step`):

- **Look through camera** (C): the view becomes the camera. The export frame
  stays put in the view, and panning, zooming (wheel) and turning
  (Shift+wheel) the mosaic under it frames a shot. With a key selected and
  the playhead on it, that edits the key live; otherwise the framing waits
  for Add Key (K), which keeps it at the current time (replacing a key
  there). Moving in time drops a framing that wasn't keyed.
- Keys show on the transport's timeline strip: click one to select it and go
  there, drag it to retime it, right-click it for Stop / Pass through, how
  the camera moves on, and Delete. [ and ] jump to the previous / next key.
- The Keyframes group shows the selected key's exact values (time, where the
  frame's middle is across and down the mosaic, zoom, turn, stop, then) for
  editing, and whether keys keep their share of their phase when its
  length changes.

Every change replaces the project's CameraTrack through the session
(`set_camera_track`), so the preview, the export and unsaved-change tracking
all see it, and each is an undo step (Ctrl+Z, Ctrl+Y or Ctrl+Shift+Z; a live
drag or a held-down field is one step).
"""

import math
from dataclasses import replace

from PySide6.QtCore import QEvent, QObject, QSize, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
)
from shiboken6 import isValid as shiboken_alive

from skitter.core.animation.camera import Shot
from skitter.core.animation.keyframes import MOTIONS, CameraKey, CameraTrack, KeyTime
from skitter.core.animation.video import content_rect
from skitter.core.slicing.params import BoolParam, ChoiceParam, Configurable, FloatParam
from skitter.ui import icons
from skitter.ui.style import muted
from skitter.ui.widgets.param_form import ParamForm

# Keys of the camera shortcuts a number field has no use for: they pass it by. (A field
# applies each change at once, so its own text undo has nothing to offer: Ctrl+Z, Ctrl+Y
# and Ctrl+Shift+Z undo and redo key edits instead.)
NUMBER_FIELD_PASSES = {Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight, Qt.Key.Key_K,
                       Qt.Key.Key_C}  # fmt: skip
UNDO_KEYS = {Qt.Key.Key_Z, Qt.Key.Key_Y}


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
        self.deselect_action = self._action("Deselect key", self.deselect, "Escape")
        self.undo_action = self._action("Undo key edit", self.undo, "Ctrl+Z")
        self.redo_action = self._action("Redo key edit", self.redo, "Ctrl+Y")
        self.redo_action.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        self.undo_action.setIcon(icons.undo())
        self.redo_action.setIcon(icons.redo())
        self.editing_live = False  # a viewfinder move is editing the selected key

        strip = step.transport.timeline
        strip.key_clicked.connect(self.select_and_go)
        strip.key_moved.connect(self.retime)
        strip.key_menu.connect(self.show_menu)
        strip.ground_clicked.connect(self.deselect)
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
        """Make the keys work anywhere in widget (the whole page). A text field with the
        focus still gets the letters it types (Qt lets it override shortcuts); a number
        field can't use these letters, so they pass it by (see eventFilter)."""
        for action in (self.viewfinder_action, self.add_action, self.delete_action,
                       self.previous_action, self.next_action, self.deselect_action,
                       self.undo_action, self.redo_action):  # fmt: skip
            widget.addAction(action)
        self._shortcut_scope = widget
        QApplication.instance().focusChanged.connect(self._watch_focus)

    def _watch_focus(self, old, new) -> None:
        """Watch the number field on the page that has the focus (see eventFilter)."""
        for box, on in ((old, False), (new, True)):
            if (isinstance(box, QAbstractSpinBox) and shiboken_alive(box)
                    and self._shortcut_scope.isAncestorOf(box)):  # fmt: skip
                if on:
                    box.installEventFilter(self)
                else:
                    box.removeEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        """Let [ ] K C, and undo / redo, reach the shortcuts while a number field on the page
        has focus."""
        if event.type() != QEvent.Type.ShortcutOverride:
            return False
        ctrl, alt = Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.AltModifier
        mods = event.modifiers()
        plain = event.key() in NUMBER_FIELD_PASSES and not mods & (ctrl | alt)
        undo = event.key() in UNDO_KEYS and mods & ctrl and not mods & alt
        if plain or undo:
            event.ignore()  # not the field's: the shortcut runs
            return True
        return False

    def build_group(self) -> QGroupBox:
        self.count = QLabel()
        self.count.setWordWrap(True)  # beside a button: long counts wrap, never widen the panel
        self.clear_button = QPushButton("Clear All…")
        self.clear_button.setToolTip("Remove every key: the camera shows the video's framing.")
        self.clear_button.clicked.connect(lambda: self.clear_keys())
        top = QHBoxLayout()
        top.addWidget(self.count, stretch=1)
        for action in (self.undo_action, self.redo_action):
            button = QToolButton()
            button.setDefaultAction(action)
            button.setAutoRaise(True)
            top.addWidget(button)
        top.addWidget(self.clear_button)
        self.hint = muted(QLabel())
        self.hint.setWordWrap(True)
        self.stretch = QCheckBox("Keep keys in step with the animation")
        self.stretch.setToolTip(
            "On: each key keeps its share of its phase (Build, Show or Clear, holds "
            "included) when the phase's length changes. Off: its seconds from the phase's "
            "start."
        )
        self.stretch.toggled.connect(self._on_stretch)
        self.fields = KeyFields()
        self.form = ParamForm()
        self.form.set_target(self.fields)
        self.form.changed.connect(self._on_field)
        group = QGroupBox("Keyframes")
        layout = QVBoxLayout(group)
        layout.addLayout(top)
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

    def _set(self, track: CameraTrack, select: CameraKey | None = None, merge=None) -> None:
        """Make track the camera's (one undo step; merge: see Session.set_camera_track)
        and select the key `select` in it."""
        self.pending = None
        self.selected = track.keys.index(select) if select in track.keys else -1
        self.session.set_camera_track(track, merge)  # the camera is planned and drawn again
        self.refresh()

    def undo(self) -> None:
        if self.session.undo_camera():
            self._after_history()

    def redo(self) -> None:
        if self.session.redo_camera():
            self._after_history()

    def _after_history(self) -> None:
        """An undo or redo replaced the keys: no framing waits; the selection stays if its
        key is still there."""
        self.pending = None
        if self.selected >= len(self.track.keys):
            self.selected = -1
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
        old = self.track.keys[at] if at >= 0 else CameraKey(KeyTime("build", 0.0), shot)
        key = CameraKey(KeyTime.at(t, clock, self.track.stretch), shot, old.stop, old.motion)
        self._set(self.track.with_key(key, clock), select=key)

    def delete_key(self) -> None:
        index = self.selected if self.selected >= 0 else self._key_index_at(self.step.player.time)
        if index >= 0:
            self._set(self.track.without(index))

    def clear_keys(self, confirm: bool = True) -> bool:
        """Remove every key (asking first, if confirm)."""
        keys = self.track.keys
        if not keys:
            return False
        if confirm:
            answer = QMessageBox.question(self.step, "Camera Keys", f"Remove all {len(keys)} keys?")
            if answer != QMessageBox.StandardButton.Yes:
                return False
        self._set(CameraTrack((), self.track.stretch))
        return True

    def deselect(self) -> None:
        """No key selected: the key fields hide, so nothing gets edited by accident."""
        if self.selected >= 0:
            self.selected = -1
            self.refresh()

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
        # A field held down (or stepped quickly) is one undo step.
        self._set(track, select=edited, merge=("field", self.selected, name))

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
        the frame's content is a new framing. On the selected key's moment it is that
        key's new shot at once (live; one undo step per gesture); elsewhere it waits for
        Add Key."""
        cam, base = self.step.canvas.camera, self.step.frame_rect()
        if self._frame_px is None or base is None:
            return
        zoom = base[2] * cam.zoom / self._frame_px[0]
        shot = Shot((float(cam.center[0]), float(cam.center[1])), float(zoom),
                    float(cam.rotation))  # fmt: skip
        if self.on_selected_key():
            index = self.selected
            key = replace(self.track.keys[index], shot=shot)
            self.editing_live = True
            try:
                self._set(self.track.replaced(index, key), select=key, merge=("live", index))
            finally:
                self.editing_live = False
            return
        self.pending = shot
        self.refresh()

    def on_selected_key(self) -> bool:
        """A key is selected and the playhead is on its moment (through the camera, the
        view then edits it)."""
        clock = self.clock()
        if clock is None or self.selected < 0 or self.selected >= len(self.track.keys):
            return False
        return abs(self.track.times(clock)[self.selected] - self.step.player.time) < 1e-6

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
        self.undo_action.setEnabled(self.session.can_undo_camera)
        self.redo_action.setEnabled(self.session.can_redo_camera)
        self.previous_action.setEnabled(has and bool(keys))
        self.next_action.setEnabled(has and bool(keys))
        if not hasattr(self, "count"):
            return
        self.clear_button.setEnabled(bool(keys))
        noun = "key" if len(keys) == 1 else "keys"
        self.count.setText(f"{len(keys)} {noun}" + ("" if keys else
                           ": the camera shows the video's framing."))  # fmt: skip
        self.stretch.blockSignals(True)
        self.stretch.setChecked(track.stretch)
        self.stretch.blockSignals(False)
        self.form.setVisible(has and self.selected >= 0)
        if has and self.selected >= 0:
            self._show_key(keys[self.selected], float(track.times(clock)[self.selected]))
        if self.pending is not None:
            hint = "Framing changed: Add Key (K) keeps it here; moving in time drops it."
        elif self.viewfinder and self.on_selected_key():
            hint = (f"Editing key {self.selected + 1} through the camera: dragging, the wheel "
                    "and Shift+wheel change it directly (Ctrl+Z undoes).")  # fmt: skip
        elif self.viewfinder:
            hint = ("Looking through the camera: drag to pan, wheel to zoom, Shift+wheel to "
                    "turn, then Add Key (K).")  # fmt: skip
        elif not keys:
            hint = "Look through camera (C) to frame a shot, then Add Key (K)."
        elif self.selected >= 0:
            hint = ("Editing the selected key. Esc, or a click on empty timeline, deselects "
                    "it; [ and ] go to the previous and next key.")  # fmt: skip
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

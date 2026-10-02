"""Step 1: choose the source image the mosaic will reproduce, with basic edits."""

from pathlib import Path

from PIL import Image
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from skitter.core.edits import Crop, Edit, FlipHorizontal, FlipVertical, Rotate90
from skitter.core.imaging import IMAGE_EXTENSIONS, load_image
from skitter.ui import icons
from skitter.ui.steps.base import StepPage, side_panel
from skitter.ui.widgets.crop_overlay import CropOverlay
from skitter.ui.widgets.image_viewer import ImageViewer

IMAGE_FILTER = "Images (" + " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS)) + ")"
ORIGINAL_ASPECT = "original"
ASPECT_CHOICES = (
    ("Free", None),
    ("Original", ORIGINAL_ASPECT),
    ("Square (1:1)", 1.0),
    ("4:3", 4 / 3),
    ("3:4", 3 / 4),
    ("3:2", 3 / 2),
    ("2:3", 2 / 3),
    ("16:9", 16 / 9),
    ("9:16", 9 / 16),
)


def _format_bytes(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    raise AssertionError("unreachable")


def _format_size(image) -> str:
    h, w = image.shape[:2]
    return f"{w:,} × {h:,} px"


class SourceStep(StepPage):
    title = "Source"

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.setAcceptDrops(True)

        self.viewer = ImageViewer()
        self.crop_overlay = CropOverlay(self.viewer.canvas)
        self.crop_overlay.box_changed.connect(self._sync_crop_fields)

        self._empty = QLabel("Drop an image here\nor click Open")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setObjectName("emptyState")
        self._empty.setStyleSheet(
            "#emptyState { border: 2px dashed palette(mid); border-radius: 12px;"
            " color: palette(placeholder-text); font-size: 16pt; margin: 24px; }"
        )
        self._view = QStackedWidget()
        self._view.addWidget(self._empty)
        self._view.addWidget(self.viewer)

        self._build_actions()
        body = QHBoxLayout()
        body.setSpacing(0)
        body.addWidget(self._view, stretch=1)
        body.addWidget(self._build_panel())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_toolbar())
        layout.addLayout(body)

        session.source_changed.connect(self._on_source_changed)
        session.source_edited.connect(self._on_source_edited)
        self._update_actions()

    # Construction

    def _action(self, icon, text, slot, *shortcuts) -> QAction:
        action = QAction(icon, text, self)
        action.triggered.connect(slot)
        if shortcuts:
            action.setShortcuts([QKeySequence(s) for s in shortcuts])
        self._set_tooltip(action, text)
        return action

    @staticmethod
    def _set_tooltip(action: QAction, text: str) -> None:
        keys = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
        action.setToolTip(f"{text} ({keys})" if keys else text)

    def _build_actions(self) -> None:
        session = self.session
        self.open_action = self._action(icons.open_folder(), "Open", self.open_dialog)
        self.undo_action = self._action(
            icons.undo(), "Undo", session.undo, QKeySequence.StandardKey.Undo
        )
        self.redo_action = self._action(
            icons.redo(), "Redo", session.redo, QKeySequence.StandardKey.Redo, "Ctrl+Shift+Z"
        )
        self.rotate_left_action = self._action(
            icons.rotate_left(), "Rotate Left", lambda: self._edit(Rotate90(-1)), "Ctrl+Shift+R"
        )
        self.rotate_right_action = self._action(
            icons.rotate_right(), "Rotate Right", lambda: self._edit(Rotate90(1)), "Ctrl+R"
        )
        self.flip_h_action = self._action(
            icons.flip_horizontal(), "Flip Horizontal", lambda: self._edit(FlipHorizontal())
        )
        self.flip_v_action = self._action(
            icons.flip_vertical(), "Flip Vertical", lambda: self._edit(FlipVertical())
        )
        self.crop_action = QAction(icons.crop(), "Crop", self)
        self.crop_action.setCheckable(True)
        self.crop_action.setShortcut(QKeySequence("C"))
        self._set_tooltip(self.crop_action, "Crop")
        self.crop_action.toggled.connect(self._set_cropping)
        self.revert_action = self._action(icons.revert(), "Revert", session.revert_edits)
        self.revert_action.setToolTip("Revert to the original image")

        # Crop mode keys live on the page so they work wherever focus is.
        self.apply_crop_action = QAction("Apply Crop", self)
        self.apply_crop_action.setShortcuts([QKeySequence("Return"), QKeySequence("Enter")])
        self.apply_crop_action.triggered.connect(self.apply_crop)
        self.cancel_crop_action = QAction("Cancel Crop", self)
        self.cancel_crop_action.setShortcut(QKeySequence("Escape"))
        self.cancel_crop_action.triggered.connect(self.cancel_crop)
        self.addActions([self.apply_crop_action, self.cancel_crop_action])

    def _build_toolbar(self) -> QToolBar:
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        groups = (
            (self.open_action,),
            (self.undo_action, self.redo_action),
            (
                self.rotate_left_action,
                self.rotate_right_action,
                self.flip_h_action,
                self.flip_v_action,
                self.crop_action,
            ),
            (self.revert_action,),
        )
        for i, group in enumerate(groups):
            if i:
                toolbar.addSeparator()
            toolbar.addActions(group)
        return toolbar

    def _build_panel(self) -> QWidget:
        return side_panel(
            self._build_info_group(),
            self._build_crop_group(),
            self._build_history_group(),
            stretch_last=True,
        )

    def _build_info_group(self) -> QGroupBox:
        self._name = QLabel("—")
        self._original_size = QLabel("—")
        self._size = QLabel("—")
        self._megapixels = QLabel("—")
        self._file_size = QLabel("—")
        group = QGroupBox("Image")
        form = QFormLayout(group)
        for label, widget in (
            ("File:", self._name),
            ("Original:", self._original_size),
            ("Current:", self._size),
            ("Megapixels:", self._megapixels),
            ("File size:", self._file_size),
        ):
            widget.setTextFormat(Qt.TextFormat.PlainText)
            widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(label, widget)
        self._name.setWordWrap(True)
        return group

    def _build_crop_group(self) -> QGroupBox:
        self._crop_aspect = QComboBox()
        for label, value in ASPECT_CHOICES:
            self._crop_aspect.addItem(label, value)
        self._crop_aspect.currentIndexChanged.connect(
            lambda: self.crop_overlay.set_aspect(self._crop_aspect_value())
        )

        self._crop_fields: dict[str, QSpinBox] = {}
        fields = QFormLayout()
        fields.addRow("Aspect:", self._crop_aspect)
        for key, label in (("x", "X:"), ("y", "Y:"), ("w", "Width:"), ("h", "Height:")):
            spin = QSpinBox()
            spin.setSuffix(" px")
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(lambda _, k=key: self._on_crop_field(k))
            self._crop_fields[key] = spin
            fields.addRow(label, spin)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_crop)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.cancel_crop)
        buttons = QHBoxLayout()
        buttons.addWidget(apply_button)
        buttons.addWidget(cancel_button)

        hint = QLabel("Drag the box or its handles, or drag outside it to draw a new box.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text);")

        self._crop_group = QGroupBox("Crop")
        layout = QVBoxLayout(self._crop_group)
        layout.addLayout(fields)
        layout.addWidget(hint)
        layout.addLayout(buttons)
        self._crop_group.hide()
        return self._crop_group

    def _build_history_group(self) -> QGroupBox:
        self._history = QListWidget()
        self._history.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._history.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        group = QGroupBox("Edits")
        QVBoxLayout(group).addWidget(self._history)
        return group

    # Loading

    def open_dialog(self) -> None:
        self.cancel_crop()
        start_dir = ""
        if self.session.project.source_path:
            start_dir = str(self.session.project.source_path.parent)
        path, _ = QFileDialog.getOpenFileName(self, "Open Source Image", start_dir, IMAGE_FILTER)
        if path:
            self.load_file(Path(path))

    def load_file(self, path: Path) -> bool:
        try:
            image = load_image(path)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            QMessageBox.warning(self, "Skitter", f"Could not open image:\n{exc}")
            return False
        self.session.set_source(path, image)
        return True

    def is_complete(self) -> bool:
        return self.session.source_is_committed

    def can_advance(self) -> bool:
        return self.session.project.has_source and not self.is_cropping()

    def advance(self) -> bool:
        """Finish the Source step: freeze the edited image for later steps."""
        if not self.can_advance():
            return False
        if self.session.commit_source():
            h, w = self.session.project.source_final.shape[:2]
            self.status_message.emit(f"Source image ready: {w:,} × {h:,} px")
        return True

    def on_leave(self) -> None:
        self.cancel_crop()

    # Editing

    def _edit(self, edit: Edit) -> None:
        if self.session.project.has_source and not self.is_cropping():
            self.session.apply_edit(edit)

    def is_cropping(self) -> bool:
        return self.crop_action.isChecked()

    def _set_cropping(self, cropping: bool) -> None:
        image = self.session.project.source_image
        if cropping and image is None:
            self.crop_action.setChecked(False)
            return
        if cropping:
            h, w = image.shape[:2]
            self._configure_crop_fields(w, h)
            self._crop_aspect.setCurrentIndex(0)
            self.crop_overlay.start(w, h)
            self._crop_group.show()
            self.status_message.emit("Crop: Enter to apply, Esc to cancel")
        else:
            self.crop_overlay.stop()
            self._crop_group.hide()
        self._update_actions()
        self.state_changed.emit()  # Next is unavailable while cropping

    def apply_crop(self) -> None:
        if not self.is_cropping():
            return
        box = self.crop_overlay.crop_box()
        h, w = self.session.project.source_image.shape[:2]
        self.crop_action.setChecked(False)
        if box != (0, 0, w, h):
            self.session.apply_edit(Crop(*box))

    def cancel_crop(self) -> None:
        self.crop_action.setChecked(False)

    def _crop_aspect_value(self) -> float | None:
        value = self._crop_aspect.currentData()
        if value == ORIGINAL_ASPECT:
            h, w = self.session.project.source_image.shape[:2]
            return w / h
        return value

    def _configure_crop_fields(self, w: int, h: int) -> None:
        limits = {"x": (0, w - 1), "y": (0, h - 1), "w": (1, w), "h": (1, h)}
        for key, spin in self._crop_fields.items():
            spin.blockSignals(True)
            spin.setRange(*limits[key])
            spin.blockSignals(False)

    def _sync_crop_fields(self) -> None:
        for key, value in zip("xywh", self.crop_overlay.crop_box(), strict=True):
            spin = self._crop_fields[key]
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _on_crop_field(self, key: str) -> None:
        x, y, w, h = (self._crop_fields[k].value() for k in "xywh")
        _, _, max_w, max_h = self.crop_overlay.bounds
        aspect = self._crop_aspect_value()
        if aspect and key == "w":
            h = round(w / aspect)
        elif aspect and key == "h":
            w = round(h * aspect)
        w, h = min(max(w, 1), max_w), min(max(h, 1), max_h)
        if aspect:
            # Clamping one side can break the ratio; shrink the other side to match.
            w, h = min(w, max(1, round(h * aspect))), min(h, max(1, round(w / aspect)))
        x, y = min(max(x, 0), max_w - w), min(max(y, 0), max_h - h)
        self.crop_overlay.set_box((x, y, x + w, y + h))

    # Session updates

    def _on_source_changed(self) -> None:
        self.cancel_crop()
        project = self.session.project
        self._view.setCurrentWidget(self.viewer)
        self.viewer.show_image(project.source_image)
        self._refresh_info()
        self._update_actions()
        self.status_message.emit(f"Source image: {project.source_path.name}")
        self.state_changed.emit()

    def _on_source_edited(self, edit: Edit | None, undone: bool) -> None:
        image = self.session.project.source_image
        viewer = self.viewer
        if edit is None:
            viewer.replace_image(image)
        elif isinstance(edit, FlipHorizontal | FlipVertical):
            viewer.flip(image, horizontal=isinstance(edit, FlipHorizontal))
        elif isinstance(edit, Rotate90):
            viewer.rotate(image, -edit.turns if undone else edit.turns)
        elif isinstance(edit, Crop):
            offset = (edit.left, edit.top)
            if undone:
                viewer.uncrop(image, offset)
            else:
                viewer.crop(image, offset)

        self._refresh_info()
        self._update_actions()
        if edit is None:
            self.status_message.emit("Reverted to the original image")
        else:
            self.status_message.emit(f"{'Undo: ' if undone else ''}{edit.describe()}")
        self.state_changed.emit()

    def _refresh_info(self) -> None:
        project = self.session.project
        path = project.source_path
        self._name.setText(path.name)
        self._name.setToolTip(str(path))
        self._original_size.setText(_format_size(project.source_original))
        self._size.setText(_format_size(project.source_image))
        h, w = project.source_image.shape[:2]
        self._megapixels.setText(f"{w * h / 1e6:.1f}")
        self._file_size.setText(_format_bytes(path.stat().st_size))

        self._history.clear()
        if project.source_edits:
            self._history.addItems([edit.describe() for edit in project.source_edits])
        else:
            self._history.addItem("No edits")
            self._history.item(0).setForeground(self.palette().placeholderText())

    def _update_actions(self) -> None:
        session = self.session
        has_image = session.project.has_source
        cropping = self.is_cropping()
        editable = has_image and not cropping

        self.open_action.setEnabled(not cropping)
        for action in (
            self.rotate_left_action,
            self.rotate_right_action,
            self.flip_h_action,
            self.flip_v_action,
        ):
            action.setEnabled(editable)
        self.crop_action.setEnabled(has_image)
        self.undo_action.setEnabled(editable and session.can_undo)
        self.redo_action.setEnabled(editable and session.can_redo)
        self.revert_action.setEnabled(editable and session.can_undo)
        self.apply_crop_action.setEnabled(cropping)
        self.cancel_crop_action.setEnabled(cropping)

        edits = session.project.source_edits
        self._set_tooltip(self.undo_action, f"Undo {edits[-1].describe()}" if edits else "Undo")
        redo = session.next_redo
        self._set_tooltip(self.redo_action, f"Redo {redo.describe()}" if redo else "Redo")

    # Drag and drop

    @staticmethod
    def _dropped_image(event: QDragEnterEvent | QDropEvent) -> Path | None:
        urls = event.mimeData().urls()
        if len(urls) != 1 or not urls[0].isLocalFile():
            return None
        path = Path(urls[0].toLocalFile())
        return path if path.suffix.lower() in IMAGE_EXTENSIONS else None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._dropped_image(event) and not self.is_cropping():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        path = self._dropped_image(event)
        if path and self.load_file(path):
            event.acceptProposedAction()

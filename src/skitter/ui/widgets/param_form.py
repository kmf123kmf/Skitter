"""Editor form generated from an operation's declared parameters.

Each Param type maps to an editor factory. Register a factory for a new
Param type with `@register_editor(MyParam)`; parameter subclasses without
their own factory use their base class's.
"""

from collections.abc import Callable
from dataclasses import dataclass
from html import escape

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QTextDocument
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QWidget,
)

from skitter.core.slicing import (
    BoolParam,
    ChoiceParam,
    ColorParam,
    FloatParam,
    IntParam,
    Param,
    RangeParam,
    Section,
)
from skitter.ui import preferences
from skitter.ui.style import MUTED_STYLE, WARNING_STYLE, muted

RANGE_BOX_MIN = 48  # px: the least each box of a range keeps (the row shares the rest)

COLOR_PRESETS = (
    ("#000000", "Black"), ("#1e1e1e", "Dark gray"), ("#808080", "Gray"), ("#ffffff", "White"),
)  # fmt: skip


@dataclass
class Editor:
    widget: QWidget
    set_value: Callable[[object], None]
    # Connects a callback that receives the new value when the user edits it.
    on_change: Callable[[Callable[[object], None]], None]
    # Receives the target after any edit, to update what the editor offers.
    refresh: Callable[[object], None] | None = None


EditorFactory = Callable[[Param], Editor]
_factories: dict[type[Param], EditorFactory] = {}


def register_editor(param_type: type[Param]):
    def decorator(factory: EditorFactory) -> EditorFactory:
        _factories[param_type] = factory
        return factory

    return decorator


def create_editor(param: Param) -> Editor:
    for klass in type(param).__mro__:
        if klass in _factories:
            return _factories[klass](param)
    raise TypeError(f"no editor registered for {type(param).__name__}")


@register_editor(IntParam)
def _int_editor(param: IntParam) -> Editor:
    spin = QSpinBox()
    spin.setRange(param.min, param.max)
    spin.setSingleStep(param.step)
    spin.setSuffix(param.suffix)
    spin.setKeyboardTracking(False)
    return Editor(spin, spin.setValue, spin.valueChanged.connect)


@register_editor(FloatParam)
def _float_editor(param: FloatParam) -> Editor:
    spin = QDoubleSpinBox()
    spin.setDecimals(param.decimals)
    spin.setRange(param.min, param.max)
    spin.setSingleStep(param.step)
    spin.setSuffix(param.suffix)
    spin.setKeyboardTracking(False)
    return Editor(spin, spin.setValue, spin.valueChanged.connect)


@register_editor(RangeParam)
def _range_editor(param: RangeParam) -> Editor:
    """Two boxes in one row, "low – high" (the unit only after the second, no arrow
    buttons); raising the low end above the high one moves both, and the other way
    around."""
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    boxes = []
    for i in range(2):
        box = QSpinBox() if param.whole else QDoubleSpinBox()
        if not param.whole:
            box.setDecimals(param.decimals)
        box.setRange(param.min, param.max)
        box.setSingleStep(param.step)
        box.setKeyboardTracking(False)
        if i == 1:
            box.setSuffix(param.suffix)
        # Spin boxes ask for room for their widest possible value ("20.00 turns"); two
        # side by side would widen the whole form. They share the row instead.
        box.setMinimumWidth(RANGE_BOX_MIN)
        box.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)  # by stretch
        # No arrow buttons: two pairs leave too little room for the numbers. Typing, the
        # arrow keys and the mouse wheel (once focused) still step the value.
        box.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        boxes.append(box)
    low, high = boxes
    layout.addWidget(low, stretch=2)
    layout.addWidget(QLabel("–"))
    layout.addWidget(high, stretch=3)
    callbacks = []

    def set_value(value) -> None:
        for box, v in zip(boxes, value, strict=True):
            box.blockSignals(True)
            box.setValue(v)
            box.blockSignals(False)

    def edited(which: int) -> None:
        a, b = low.value(), high.value()
        if a > b:  # keep low <= high: the other end follows
            a, b = (a, a) if which == 0 else (b, b)
            set_value((a, b))
        value = (int(a), int(b)) if param.whole else (a, b)
        for callback in callbacks:
            callback(value)

    low.valueChanged.connect(lambda _: edited(0))
    high.valueChanged.connect(lambda _: edited(1))
    return Editor(widget, set_value, callbacks.append)


def color_swatch(value: str, size: int = 16) -> QIcon:
    """A square of the color, or a checkerboard for transparent."""
    pixmap = QPixmap(size, size)
    if value == ColorParam.TRANSPARENT:
        pixmap.fill(QColor("#ffffff"))
        painter = QPainter(pixmap)
        half = size // 2
        painter.fillRect(0, 0, half, half, QColor("#bdbdbd"))
        painter.fillRect(half, half, size - half, size - half, QColor("#bdbdbd"))
        painter.end()
    else:
        pixmap.fill(QColor(value))
    return QIcon(pixmap)


def color_name(value: str) -> str:
    if value == ColorParam.TRANSPARENT:
        return "Transparent"
    return dict(COLOR_PRESETS).get(value, value.upper())


@register_editor(ColorParam)
def _color_editor(param: ColorParam) -> Editor:
    """One button showing the color; its menu offers presets, Transparent and Custom."""
    button = QToolButton()
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = QMenu(button)
    button.setMenu(menu)
    callbacks = []
    state = {"value": param.default}

    def set_value(value) -> None:
        state["value"] = value
        button.setIcon(color_swatch(value))
        button.setText(color_name(value))

    def choose(value) -> None:
        if value != state["value"]:
            set_value(value)
            for callback in callbacks:
                callback(value)

    def custom() -> None:
        start = QColor(state["value"]) if state["value"] != ColorParam.TRANSPARENT else QColor()
        color = QColorDialog.getColor(start, button, param.label or "Color")
        if color.isValid():
            choose(color.name().lower())

    for value, name in COLOR_PRESETS:
        menu.addAction(color_swatch(value), name, lambda v=value: choose(v))
    if param.allow_transparent:
        menu.addAction(color_swatch(ColorParam.TRANSPARENT), "Transparent",
                       lambda: choose(ColorParam.TRANSPARENT))  # fmt: skip
    menu.addSeparator()
    menu.addAction("Custom…", custom)
    set_value(param.default)
    return Editor(button, set_value, callbacks.append)


@register_editor(BoolParam)
def _bool_editor(param: BoolParam) -> Editor:
    check = QCheckBox()
    return Editor(check, check.setChecked, check.toggled.connect)


@register_editor(ChoiceParam)
def _choice_editor(param: ChoiceParam) -> Editor:
    combo = QComboBox()
    for value, label in param.choices:
        combo.addItem(label, value)

    def set_value(value) -> None:
        combo.setCurrentIndex(combo.findData(value))

    def on_change(callback) -> None:
        combo.currentIndexChanged.connect(lambda _: callback(combo.currentData()))

    def refresh(target) -> None:
        model = combo.model()
        for i in range(combo.count()):
            model.item(i).setEnabled(param.is_available(target, combo.itemData(i)))

    return Editor(combo, set_value, on_change, refresh=refresh)


INDENT = 14  # px: how far a setting sits in from the one it belongs with (Param.under)


@dataclass
class _Heading:
    section: Section
    button: QToolButton
    note: QLabel | None  # the section's help, shown under its title
    expanded: bool


@dataclass
class _Row:
    param: Param
    editor: Editor
    label: QLabel
    parent: "_Row | None"  # the row of the setting it belongs with (Param.under)
    heading: _Heading | None
    depth: int = 0  # how many settings it sits under
    note: QLabel | None = None  # what the setting came to (set_notes), right under it


class ParamForm(QWidget):
    """Edits the parameters of one object that declares Params (an operation).

    Edits are written straight to the object; `changed` then reports the
    parameter name. Several forms may edit one object (each showing some of
    its parameters); call `refresh()` on the others after it changes.

    Settings that don't apply (Param.when) are hidden; one that belongs with
    another (Param.under) sits indented below it. Sections (params.Section) get
    a heading that folds them away; whether each is folded is remembered. Notes
    (`set_notes`) show what settings came to, each under its setting.
    """

    changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._target = None
        self._rows: list[_Row] = []
        self._headings: list[_Heading] = []

    def set_target(self, target, only=None) -> None:
        """Show editors for target's parameters (None clears the form).

        only: names of the parameters to show (default: all), in declaration order.
        """
        while self._layout.rowCount():
            self._layout.removeRow(0)
        self._rows.clear()
        self._headings.clear()
        self._target = target
        if target is None:
            return
        groups = target.sections() if hasattr(target, "sections") else [(None, target.params())]
        rows: dict[str, _Row] = {}
        for section, params in groups:
            params = [p for p in params if only is None or p.name in only]
            if not params:
                continue
            heading = None
            if section is not None and section.title:
                heading = self._add_heading(type(target), section)
            elif section is not None and self._rows:  # the end of the sections
                self._layout.addRow(_line(top=6))
            for param in params:
                editor = create_editor(param)
                editor.set_value(getattr(target, param.name))
                editor.widget.setToolTip(param.help)
                editor.on_change(lambda value, p=param: self._on_edit(p, value))
                label = QLabel(f"{param.label}:")
                label.setToolTip(param.help)
                parent = rows.get(param.under) if param.under else None
                depth, up = 0, parent
                while up is not None:
                    depth, up = depth + 1, up.parent
                label.setContentsMargins(INDENT * depth, 0, 0, 0)
                self._layout.addRow(label, editor.widget)
                rows[param.name] = row = _Row(param, editor, label, parent, heading, depth)
                self._rows.append(row)
        self._update_active()

    def _add_heading(self, owner: type, section: Section) -> _Heading:
        key = f"forms/{owner.__name__}/{section.name}/folded"
        button = QToolButton()
        button.setText(section.title)
        button.setToolTip(f"Show or hide the {section.title.lower()} settings.")
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setAutoRaise(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        font = button.font()
        font.setBold(True)
        button.setFont(font)
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 6 if self._rows else 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(button)
        layout.addWidget(_line(), stretch=1)
        self._layout.addRow(bar)
        note = None
        if section.help:
            note = muted(QLabel(section.help))
            self._layout.addRow(note)
        heading = _Heading(section, button, note, not _folded(key))

        def show_arrow() -> None:
            button.setArrowType(
                Qt.ArrowType.DownArrow if heading.expanded else Qt.ArrowType.RightArrow
            )

        def toggle() -> None:
            heading.expanded = not heading.expanded
            show_arrow()
            preferences.settings().setValue(key, not heading.expanded)
            self._update_visible()

        show_arrow()
        button.clicked.connect(toggle)
        self._headings.append(heading)
        return heading

    def set_notes(self, notes) -> None:
        """Show notes under settings, replacing any shown: (name, text, warning) each
        (warning: in the warning color). Several notes on one setting share its row;
        names the form doesn't show are left out."""
        texts: dict[str, list[tuple[str, bool]]] = {}
        for name, text, warning in notes:
            texts.setdefault(name, []).append((text, warning))
        for row in self._rows:
            lines = texts.get(row.param.name, [])
            if not lines and row.note is None:
                continue
            if row.note is None:
                row.note = QLabel()
                row.note.setWordWrap(True)
                row.note.setTextFormat(Qt.TextFormat.RichText)
                row.note.setContentsMargins(INDENT * (row.depth + 1), 0, 0, 2)
                index, _ = self._layout.getWidgetPosition(row.editor.widget)
                self._layout.insertRow(index + 1, row.note)
            row.note.setText("<br>".join(
                f'<span style="{WARNING_STYLE if warning else MUTED_STYLE}">{escape(text)}</span>'
                for text, warning in lines
            ))  # fmt: skip
        self._update_visible()

    def note(self, name: str) -> str:
        """The note under a setting, as plain text (lines joined by newlines; "": none)."""
        label = self._row(name).note
        if label is None:
            return ""
        doc = QTextDocument()
        doc.setHtml(label.text())
        return doc.toPlainText()

    def refresh(self) -> None:
        """Show the target's current values (it was edited elsewhere), without reporting edits."""
        for row in self._rows:
            row.editor.widget.blockSignals(True)
            try:
                row.editor.set_value(getattr(self._target, row.param.name))
            finally:
                row.editor.widget.blockSignals(False)
        self._update_active()

    def names(self) -> list[str]:
        """The settings the form edits, in order (shown or not)."""
        return [row.param.name for row in self._rows]

    def editor(self, name: str) -> Editor:
        return self._row(name).editor

    def is_shown(self, name: str) -> bool:
        """Whether the setting's row shows: it applies, and its section is open."""
        return self._shown(self._row(name))

    def headings(self) -> list[QToolButton]:
        """The section headings' buttons (a click folds or opens one), in order."""
        return [heading.button for heading in self._headings]

    def is_open(self, title: str) -> bool:
        """Whether the section with this title is open (not folded away)."""
        return next(h.expanded for h in self._headings if h.section.title == title)

    def _row(self, name: str) -> _Row:
        return next(row for row in self._rows if row.param.name == name)

    def _on_edit(self, param: Param, value) -> None:
        setattr(self._target, param.name, value)
        self._update_active()
        self.changed.emit(param.name)

    def _applies(self, row: _Row | None) -> bool:
        """It applies, and so does the setting it belongs with."""
        while row is not None:
            if not row.param.is_active(self._target):
                return False
            row = row.parent
        return True

    def _shown(self, row: _Row) -> bool:
        return (row.heading is None or row.heading.expanded) and self._applies(row)

    def _update_visible(self) -> None:
        for row in self._rows:
            shown = self._shown(row)
            self._layout.setRowVisible(row.editor.widget, shown)
            if row.note is not None:
                self._layout.setRowVisible(row.note, shown and bool(row.note.text()))
        for heading in self._headings:
            if heading.note is not None:
                self._layout.setRowVisible(heading.note, heading.expanded)

    def _update_active(self) -> None:
        target = self._target
        self._update_visible()
        for row in self._rows:
            if row.editor.refresh is not None:
                row.editor.refresh(target)
        for row in self._rows:
            param, editor = row.param, row.editor
            if isinstance(param, ChoiceParam) and not param.is_available(
                target, getattr(target, param.name)
            ):
                fallback = next(
                    (v for v, _ in param.choices if param.is_available(target, v)), None
                )
                if fallback is not None:
                    editor.set_value(fallback)  # the editor reports it back through _on_edit
                    return


def _line(top: int = 0) -> QWidget:
    """A thin rule across the form (above: `top` px of space)."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    if not top:
        return line
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, top, 0, 0)
    layout.addWidget(line)
    return box


def _folded(key: str) -> bool:
    value = preferences.settings().value(key, False)
    return value in (True, "true", "1", 1)

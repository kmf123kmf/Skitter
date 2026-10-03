"""Editor form generated from an operation's declared parameters.

Each Param type maps to an editor factory. Register a factory for a new
Param type with `@register_editor(MyParam)`; parameter subclasses without
their own factory use their base class's.
"""

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMenu,
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
)

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


class ParamForm(QWidget):
    """Edits the parameters of one object that declares Params (an operation).

    Edits are written straight to the object; `changed` then reports the
    parameter name. Several forms may edit one object (each showing some of
    its parameters); call `refresh()` on the others after it changes.
    """

    changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._target = None
        self._rows: list[tuple[Param, Editor, QLabel]] = []

    def set_target(self, target, only=None) -> None:
        """Show editors for target's parameters (None clears the form).

        only: names of the parameters to show (default: all), in declaration order.
        """
        while self._layout.rowCount():
            self._layout.removeRow(0)
        self._rows.clear()
        self._target = target
        if target is None:
            return
        for param in target.params():
            if only is not None and param.name not in only:
                continue
            editor = create_editor(param)
            editor.set_value(getattr(target, param.name))
            editor.widget.setToolTip(param.help)
            editor.on_change(lambda value, p=param: self._on_edit(p, value))
            label = QLabel(f"{param.label}:")
            label.setToolTip(param.help)
            self._layout.addRow(label, editor.widget)
            self._rows.append((param, editor, label))
        self._update_active()

    def refresh(self) -> None:
        """Show the target's current values (it was edited elsewhere), without reporting edits."""
        for param, editor, _ in self._rows:
            editor.widget.blockSignals(True)
            try:
                editor.set_value(getattr(self._target, param.name))
            finally:
                editor.widget.blockSignals(False)
        self._update_active()

    def editor(self, name: str) -> Editor:
        return next(editor for param, editor, _ in self._rows if param.name == name)

    def _on_edit(self, param: Param, value) -> None:
        setattr(self._target, param.name, value)
        self._update_active()
        self.changed.emit(param.name)

    def _update_active(self) -> None:
        target = self._target
        for param, editor, label in self._rows:
            active = param.is_active(target)
            editor.widget.setEnabled(active)
            label.setEnabled(active)
            if editor.refresh is not None:
                editor.refresh(target)
        for param, editor, _ in self._rows:
            if isinstance(param, ChoiceParam) and not param.is_available(
                target, getattr(target, param.name)
            ):
                fallback = next(
                    (v for v, _ in param.choices if param.is_available(target, v)), None
                )
                if fallback is not None:
                    editor.set_value(fallback)  # the editor reports it back through _on_edit
                    return

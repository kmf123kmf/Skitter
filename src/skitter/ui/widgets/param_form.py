"""Editor form generated from an operation's declared parameters.

Each Param type maps to an editor factory. Register a factory for a new
Param type with `@register_editor(MyParam)`; parameter subclasses without
their own factory use their base class's.
"""

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from skitter.core.slicing import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    IntParam,
    Param,
    TileSizeParam,
)


@dataclass
class Editor:
    widget: QWidget
    set_value: Callable[[object], None]
    # Connects a callback that receives the new value when the user edits it.
    on_change: Callable[[Callable[[object], None]], None]
    # Receives the form's context (e.g. {"tile_size": (w, h)}) when it changes.
    set_context: Callable[[dict], None] | None = None
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


@register_editor(TileSizeParam)
def _tile_size_editor(param: TileSizeParam) -> Editor:
    """Base-tile multiple, with the size in mosaic pixels shown beside it."""
    spin = QDoubleSpinBox()
    spin.setDecimals(param.decimals)
    spin.setRange(param.min, param.max)
    spin.setSingleStep(param.step)
    spin.setSuffix(param.suffix)
    spin.setKeyboardTracking(False)
    pixels = QLabel()
    pixels.setStyleSheet("color: palette(placeholder-text);")
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(spin, stretch=1)
    layout.addWidget(pixels)
    context = {"tile_size": None}

    def refresh() -> None:
        tile = context["tile_size"]
        if tile is None:
            pixels.clear()
            return
        w, h = spin.value() * tile[0], spin.value() * tile[1]
        pixels.setText(f"{w:,.0f} px" if abs(w - h) < 0.5 else f"{w:,.0f} × {h:,.0f} px")

    def set_context(new: dict) -> None:
        context.update(new)
        refresh()

    spin.valueChanged.connect(lambda _: refresh())
    return Editor(widget, spin.setValue, spin.valueChanged.connect, set_context)


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
    parameter name.
    """

    changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._target = None
        self._rows: list[tuple[Param, Editor, QLabel]] = []
        self._context: dict = {}

    def set_context(self, **context) -> None:
        """Information editors may display, e.g. tile_size=(w, h) in mosaic px."""
        self._context.update(context)
        for _, editor, _ in self._rows:
            if editor.set_context is not None:
                editor.set_context(self._context)

    def set_target(self, target) -> None:
        """Show editors for target's parameters (None clears the form)."""
        while self._layout.rowCount():
            self._layout.removeRow(0)
        self._rows.clear()
        self._target = target
        if target is None:
            return
        for param in target.params():
            editor = create_editor(param)
            editor.set_value(getattr(target, param.name))
            if editor.set_context is not None:
                editor.set_context(self._context)
            editor.widget.setToolTip(param.help)
            editor.on_change(lambda value, p=param: self._on_edit(p, value))
            label = QLabel(f"{param.label}:")
            label.setToolTip(param.help)
            self._layout.addRow(label, editor.widget)
            self._rows.append((param, editor, label))
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

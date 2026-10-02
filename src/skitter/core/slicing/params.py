"""Declarative parameters for slicing operations.

An operation declares its settings as class attributes:

    class GridSlicer(Subdivider):
        columns = IntParam(24, "Columns", min=1, max=1000)

Reading `op.columns` returns the instance's current value and assigning
validates it. The UI builds an editor for each parameter from these
declarations, and plans save and load the values, so a new operation needs
no UI or serialization code of its own.
"""

from collections.abc import Callable, Sequence
from typing import Any


class Param:
    """Base descriptor. Subclasses add validation and editor hints."""

    def __init__(
        self,
        default: Any,
        label: str = "",
        *,
        help: str = "",
        when: Callable[[Any], bool] | None = None,
    ):
        self.default = default
        self.label = label
        self.help = help
        self.when = when  # when it returns False the parameter is inactive (greyed out)
        self.name = ""

    def __set_name__(self, owner, name: str) -> None:
        self.name = name
        if not self.label:
            self.label = name.replace("_", " ").capitalize()
        self.default = self.validate(self.default)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj._values[self.name]

    def __set__(self, obj, value) -> None:
        obj._values[self.name] = self.validate(value)

    def validate(self, value):
        return value

    def is_active(self, obj) -> bool:
        return self.when is None or bool(self.when(obj))

    def _error(self, message: str) -> ValueError:
        return ValueError(f"{self.label}: {message}")


class IntParam(Param):
    def __init__(self, default: int, label: str = "", *, min: int = 0, max: int = 1_000_000,
                 step: int = 1, suffix: str = "", **kwargs):  # fmt: skip
        self.min, self.max, self.step, self.suffix = min, max, step, suffix
        super().__init__(default, label, **kwargs)

    def validate(self, value) -> int:
        if isinstance(value, bool) or not float(value).is_integer():
            raise self._error(f"expected a whole number, got {value!r}")
        value = int(value)
        if not self.min <= value <= self.max:
            raise self._error(f"{value} is outside {self.min}..{self.max}")
        return value


class FloatParam(Param):
    def __init__(self, default: float, label: str = "", *, min: float = 0.0, max: float = 1e9,
                 step: float = 0.1, decimals: int = 2, suffix: str = "", **kwargs):  # fmt: skip
        self.min, self.max, self.step = min, max, step
        self.decimals, self.suffix = decimals, suffix
        super().__init__(default, label, **kwargs)

    def validate(self, value) -> float:
        if isinstance(value, bool):
            raise self._error(f"expected a number, got {value!r}")
        value = float(value)
        if not self.min <= value <= self.max:
            raise self._error(f"{value} is outside {self.min}..{self.max}")
        return value


class TileSizeParam(FloatParam):
    """A length measured in base tiles: 1.0 is one base tile (see MosaicLayout).

    Sizes given this way follow the user's tile size, so plans keep their
    character when the tile size changes. Operations convert to mosaic
    pixels with `ctx.tile_size`; the UI shows the pixel size alongside.
    """

    def __init__(self, default: float, label: str = "", *, min: float = 0.05,
                 max: float = 100.0, step: float = 0.05, decimals: int = 2, **kwargs):  # fmt: skip
        kwargs.setdefault("suffix", " × tile")
        super().__init__(default, label, min=min, max=max, step=step, decimals=decimals, **kwargs)


class BoolParam(Param):
    def validate(self, value) -> bool:
        if not isinstance(value, bool):
            raise self._error(f"expected true or false, got {value!r}")
        return value


class ChoiceParam(Param):
    """One of a set of values, each shown with a label.

    choices is a list of (value, label) pairs, or a function returning one,
    for choices drawn from a registry that may grow (like brick patterns).
    """

    def __init__(
        self,
        default,
        label: str = "",
        *,
        choices: Sequence[tuple[Any, str]] | Callable[[], Sequence[tuple[Any, str]]],
        **kwargs,
    ):
        self._choices = choices if callable(choices) else list(choices)
        super().__init__(default, label, **kwargs)

    @property
    def choices(self) -> list[tuple[Any, str]]:
        return list(self._choices()) if callable(self._choices) else self._choices

    def validate(self, value):
        if value not in [v for v, _ in self.choices]:
            raise self._error(f"{value!r} is not one of the choices")
        return value

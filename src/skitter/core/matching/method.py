"""Matching methods: how the sliced regions get what they show.

A method turns the regions into a MatchResult. "Mosaic" matches library
photos to the picture (the pipeline in run.py); others may place something
else entirely ("Jigsaw Puzzle" shows the picture itself, cut along the
regions). The Matching tab offers every registered method, and the project
keeps each one's settings.

To add a method: subclass `MatchMethod`, set `id`, `name` and
`description`, declare parameters (Params, see slicing/params.py),
implement `match`, and decorate the class with `@register_method`. A method
that places no library photos sets `needs_library = False` (it then runs
without a library; `matcher.library` may be None).

What a result promises everything downstream (scene, textures, export,
animations) is in result.py. Picking tiles by hand needs
`result.candidates`; a method that leaves it None has no picks.

`match` runs in the background: report through the Reporter (each step
and detail checks for cancelling) and preview through it when there is
something to show.
"""

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from skitter.core.matching.result import MatchCancelled, MatchPreview, MatchResult, Progress
from skitter.core.slicing import RegionSet, SliceContext
from skitter.core.slicing.params import Configurable

if TYPE_CHECKING:
    from skitter.core.matching.candidates import Pins
    from skitter.core.matching.matcher import Matcher
    from skitter.core.matching.saved import SavedMosaic
    from skitter.core.tiles.library import TileLibrary


class Reporter:
    """Progress, cancelling and previews for one run."""

    def __init__(
        self,
        progress: Progress,
        cancelled: Callable[[], bool],
        detail: Progress | None,
        preview: Callable[[MatchPreview], None] | None,
    ):
        self._progress, self._cancelled = progress, cancelled
        self._detail, self._preview = detail, preview
        self.timings: dict[str, float] = {}
        self._clock = time.perf_counter()

    @property
    def previewing(self) -> bool:
        return self._preview is not None

    def step(self, message: str, fraction: float | None = None, timing: str | None = None):
        """A new step; the time since the last one is added to timings[timing]."""
        self.check()
        now = time.perf_counter()
        if timing:
            self.timings[timing] = self.timings.get(timing, 0.0) + now - self._clock
        self._clock = now
        self._progress(message, fraction)

    def detail(self, message: str, fraction: float) -> None:
        """Progress within a long step."""
        self.check()
        if self._detail is not None:
            self._detail(message, fraction)

    def counter(self, message: str, total: int) -> Callable:
        """A found(rows) callback reporting how many of total regions are done."""
        done = [0]

        def found(rows) -> None:
            done[0] += len(rows)
            self.detail(f"{message}: {done[0]:,} of {total:,} regions", done[0] / max(total, 1))

        return found

    def preview(self, item: MatchPreview) -> None:
        if self._preview is not None:
            self._preview(item)

    def check(self) -> None:
        if self._cancelled():
            raise MatchCancelled()


class MatchMethod(Configurable):
    """A way of matching; its settings are its Params."""

    id: ClassVar[str] = ""
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    needs_library: ClassVar[bool] = True  # places library photos

    def match(
        self,
        matcher: "Matcher",
        regions: RegionSet,
        ctx: SliceContext,
        pins: "Pins | None",
        report: Reporter,
    ) -> MatchResult:
        """What each region shows. matcher holds the library and caches shared across
        runs; pins are manual picks to keep in place."""
        raise NotImplementedError

    def restore(
        self, saved: "SavedMosaic", library: "TileLibrary | None", regions: RegionSet,
        ctx: SliceContext,
    ) -> "tuple[MatchResult, int] | None":  # fmt: skip
        """(result, missing) rebuilt from a project file's saved mosaic, or None to restore
        the photos it lists (see saved.py)."""
        return None

    def key(self) -> tuple:
        return (self.id, *super().key())


_registry: dict[str, type[MatchMethod]] = {}


def register_method(cls: type[MatchMethod]) -> type[MatchMethod]:
    """Class decorator making a matching method available to the UI."""
    if not cls.id or not cls.name:
        raise TypeError(f"{cls.__name__} must define id and name")
    existing = _registry.get(cls.id)
    if existing is not None and existing is not cls:
        raise ValueError(f"matching method id {cls.id!r} is already used by {existing.__name__}")
    _registry[cls.id] = cls
    return cls


def method_types() -> list[type[MatchMethod]]:
    """Registered methods, the default (Mosaic) first, then by name."""
    return sorted(_registry.values(), key=lambda cls: (cls.id != DEFAULT_METHOD, cls.name))


def unregister_method(type_id: str) -> None:
    """Take a method out of the registry (tests registering their own)."""
    _registry.pop(type_id, None)


def get_method(type_id: str) -> type[MatchMethod]:
    try:
        return _registry[type_id]
    except KeyError:
        raise KeyError(f"unknown matching method {type_id!r}") from None


DEFAULT_METHOD = "mosaic"

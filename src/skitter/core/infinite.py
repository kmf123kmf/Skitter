"""Infinite Mosaic: dive into any tile, and its photo becomes the next mosaic.

A session keeps its settings (`InfiniteSettings`: tile shape, columns,
slicer, a few matching rules) from start to end. It starts from a random
library photo (matched as the first level), or from a random mosaic: random
photos in the layout, over a canvas of the view's shape (`random_mosaic`,
level 0, matching nothing; `prepare` builds the search index meanwhile).
Each level is a picture sliced and matched; diving into a
tile makes that tile's photo, as the tile shows it (its crop, mirrored,
untinted), the next level's picture (`InfiniteSession.picture`). Every
picture has the base tile's shape, so every level's canvas does too, and a
tile seen up close becomes its own mosaic seamlessly.

Levels must be quick, so they are matched without what a single mosaic
spends extra time on (`InfiniteSession.level`): no refinement or adaptive
passes, the search effort calibrated on the first level only (run.MatchRun:
`efforts`), no quality check, and no candidates for picking by hand
(`result(lean=True)`). Small photos are enlarged to SOURCE_PX_PER_TILE
pixels per tile, so every region has enough pixels for full detail and is
searched in the same index (targets.detail_levels): the first level builds
it, later levels only search. Measured on 200,000 photos (2026-10-30): the
first level takes about 25 s (describing the tiles, building the index);
later ones well under the dive at 40 columns.

A level matched while the UI animates (a dive) should search with fewer
threads (`level(threads=...)`, see DIVE_THREADS): faiss's threads wait for
each other at every step, so one of them sharing a core with the UI thread
slows them all. Measured in the app on 16 cores, 40 columns: searching took
1.35 s with 16 threads, 0.4 s with 8, 0.29 s with 4 (64 columns: 0.75 s with
4, 1.0 s with 2 or 8).
"""

import os
from dataclasses import dataclass, field

import faiss
import numpy as np

from skitter.core.matching.matcher import Matcher
from skitter.core.matching.method import Reporter
from skitter.core.matching.mosaic import MosaicMethod
from skitter.core.matching.result import MatchResult, Progress
from skitter.core.matching.run import MatchRun
from skitter.core.matching.targets import FULL_DETAIL, level_weights
from skitter.core.scene import MosaicScene
from skitter.core.slicing import (
    TILE_ASPECTS,
    MosaicLayout,
    SliceContext,
    SlicingPlan,
    operation_types,
)
from skitter.core.slicing.params import (
    BoolParam,
    ChoiceParam,
    Configurable,
    FloatParam,
    IntParam,
)
from skitter.core.tiles.crops import aspect_classes
from skitter.core.tiles.library import TileLibrary
from skitter.core.tiles.render import render_crops_or_thumbs

MOSAIC = ("start", "tile_aspect", "columns", "slicer")  # InfiniteSettings by group
MATCHING = ("tint", "tint_strength", "max_uses", "min_spacing", "mirror")
DIVE = ("dive_seconds", "fade_seconds")
SOURCE_PX_PER_TILE = 16  # least picture pixels across a base tile (4 per cell: full detail)
MAX_SOURCE_PX = 4096  # longest side a level's picture is read at
DIVE_THREADS = max(2, (os.cpu_count() or 4) // 4)  # search threads while the UI animates


class InfiniteSettings(Configurable):
    """What every level of a session is made with (fixed while the session runs). The
    window shows them in three groups: MOSAIC, MATCHING and DIVE."""

    start = ChoiceParam(
        "photo", "Start with", choices=[("photo", "Random photo"), ("mosaic", "Random mosaic")],
        help="Random photo: a library photo, matched as the first mosaic (the first level "
             "takes longest). Random mosaic: the view filled with random photos in the "
             "layout, at once; dive into one to begin.",
    )  # fmt: skip
    tile_aspect = ChoiceParam(
        1.0, "Tile shape", choices=list(TILE_ASPECTS),
        help="Shape of every tile, and so of every mosaic: a tile you dive into becomes "
             "the next picture.",
    )  # fmt: skip
    columns = IntParam(
        40, "Columns", min=4, max=200,
        help="Tiles across each mosaic. More look finer, but each level takes longer to "
             "match and to load.",
    )  # fmt: skip
    slicer = ChoiceParam(
        "grid", "Slicer", choices=[(cls.id, cls.name) for cls in operation_types()],
        help="How each picture is cut into tiles (its own settings are below).",
    )  # fmt: skip

    tint = ChoiceParam(
        "subtle", "Tinting",
        choices=[("none", "None (purist)"), ("subtle", "Subtle"), ("custom", "Custom")],
        help="Shift each tile's average color toward its region's. A tile you dive into "
             "loses its tint on the way in: its photo becomes the next picture as it is.",
    )  # fmt: skip
    tint_strength = FloatParam(
        0.35, "Tint strength", min=0.0, max=1.0, step=0.05,
        under="tint", when=lambda s: s.tint == "custom",
        help="0 leaves tiles untouched; higher values let structure matter more than color.",
    )  # fmt: skip
    max_uses = IntParam(
        3, "Max uses per tile", min=0, max=100_000,
        help="How many times one photo may appear in a mosaic (0: no limit).",
    )  # fmt: skip
    min_spacing = FloatParam(
        4.0, "Repeat spacing", min=0.0, max=1000.0, step=0.5, decimals=1, suffix=" × tile",
        help="Minimum distance between two uses of the same photo, in tiles (0: off).",
    )  # fmt: skip
    mirror = BoolParam(False, "Mirrored tiles", help="Also try every crop flipped left to right.")

    dive_seconds = FloatParam(
        1.5, "Dive time", min=0.3, max=10.0, step=0.1, decimals=1, suffix=" s",
        help="How long a dive takes: the tile straightens and fills the view while the "
             "others fly out of it.",
    )  # fmt: skip
    fade_seconds = FloatParam(
        0.6, "Fade time", min=0.0, max=5.0, step=0.1, decimals=1, suffix=" s",
        help="How long the photo takes to dissolve into its own mosaic once it is matched.",
    )  # fmt: skip

    @property
    def layout(self) -> MosaicLayout:
        return MosaicLayout(float(self.tile_aspect), int(self.columns))

    def method(self, base: MosaicMethod | None = None) -> MosaicMethod:
        """The Mosaic settings levels are matched with: base (default: Mosaic's
        defaults) with this session's rules, and no refinement or adaptive passes."""
        method = MosaicMethod() if base is None else base.copy()
        method.update(tint=self.tint, tint_strength=self.tint_strength, max_uses=self.max_uses,
                      min_spacing=self.min_spacing, mirror=self.mirror, refine_seconds=0.0,
                      adaptive_rounds=0, error_diffusion=False)  # fmt: skip
        return method


@dataclass(frozen=True)
class TileView:
    """A crop of a library photo, as a tile shows it."""

    slot: int
    rect: tuple[float, float, float, float]  # fractions of the upright photo
    mirrored: bool = False


@dataclass
class Level:
    """One mosaic of the session."""

    number: int  # 1 for the session's first
    picture: np.ndarray  # (H, W, 4) uint8, the base tile's shape
    ctx: SliceContext
    result: MatchResult
    scene: MosaicScene
    timings: dict = field(default_factory=dict)  # seconds per step


def centered_crops(widths, heights, aspects) -> np.ndarray:
    """(N, 4) the largest centered crop of each widths x heights photo with its aspect
    (w / h), as fractions of the photo (the whole photo when its size is unknown)."""
    widths, heights = np.asarray(widths, np.float64), np.asarray(heights, np.float64)
    aspects = np.broadcast_to(np.asarray(aspects, np.float64), widths.shape)
    known = (widths > 0) & (heights > 0)
    photo = np.where(known, widths / np.where(known, heights, 1.0), aspects)
    wide = photo > aspects
    across = np.where(wide, aspects / photo, 1.0)
    down = np.where(wide, 1.0, photo / aspects)
    return np.stack([(1 - across) / 2, (1 - down) / 2, (1 + across) / 2, (1 + down) / 2], axis=1)


def centered_crop(width: int, height: int, aspect: float) -> tuple[float, float, float, float]:
    """centered_crops of one photo."""
    return tuple(float(v) for v in centered_crops([width], [height], [aspect])[0])


def blank_picture(aspect: float, width: int) -> np.ndarray:
    """(H, W, 4) an opaque black picture of this shape: a canvas to slice with nothing
    to follow."""
    picture = np.zeros((max(1, round(width / aspect)), width, 4), np.uint8)
    picture[..., 3] = 255
    return picture


def picture_size(native_width: float, aspect: float, columns: int) -> tuple[int, int]:
    """(w, h) pixels to read a picture of this aspect at: its own width, but at least
    SOURCE_PX_PER_TILE per column and at most MAX_SOURCE_PX on its longest side."""
    width = max(float(native_width), columns * SOURCE_PX_PER_TILE)
    width = min(width, MAX_SOURCE_PX, MAX_SOURCE_PX * aspect)
    return max(1, round(width)), max(1, round(width / aspect))


class InfiniteSession:
    """A run of levels with fixed settings, sharing a matcher (and its indexes).

    `picture` reads the library (call it on the library's thread); `level`
    may run in the background.
    """

    def __init__(
        self,
        library: TileLibrary,
        settings: InfiniteSettings,
        plan: SlicingPlan | None = None,
        base: MosaicMethod | None = None,
        matcher: Matcher | None = None,
        seed: int | None = None,
    ):
        """plan: slicer settings (its `settings.slicer` is used); base: Mosaic settings
        to start from (see InfiniteSettings.method)."""
        self.library = library
        self.settings = settings.copy()
        plan = SlicingPlan() if plan is None else plan
        self.plan = SlicingPlan(plan.slicers[self.settings.slicer].copy())
        self.method = self.settings.method(base)
        self.matcher = Matcher(library) if matcher is None else matcher
        self.layout = self.settings.layout
        self.efforts: dict = {}  # calibrated search efforts, kept for every later level
        self.rng = np.random.default_rng(seed)
        self.levels = 0  # levels made so far

    def random_tile(self) -> TileView:
        """A random photo of the library, cropped to the tile shape: the first picture."""
        library = self.library
        ids = library.ids
        if not len(ids):
            raise ValueError("the library has no photos in use")
        slot = int(self.rng.choice(ids))
        crop = centered_crop(int(library.width[slot]), int(library.height[slot]),
                             self.layout.tile_aspect)  # fmt: skip
        return TileView(slot, crop)

    def random_mosaic(self, aspect: float) -> Level:
        """Level 0: random photos (each cropped to its region's shape) in the session's
        layout, over a canvas of this shape (w / h: the view's, which it fills). Nothing is
        matched, so it is instant."""
        ctx = SliceContext(blank_picture(aspect, 256), self.layout)
        regions = self.plan.evaluate(ctx)[-1].regions
        n, library = len(regions), self.library
        ids = library.ids
        if not len(ids):
            raise ValueError("the library has no photos in use")
        slots = self.rng.choice(ids, n, replace=len(ids) < n).astype(np.int64)
        rect = centered_crops(library.width[slots], library.height[slots],
                              regions.size[:, 0] / regions.size[:, 1])  # fmt: skip
        zeros = np.zeros((n, 3), np.float32)
        result = MatchResult(
            regions=regions, tile=slots, rect=rect.astype(np.float32),
            mirrored=np.zeros(n, bool), cost=np.full(n, np.nan, np.float32),
            tile_mean=zeros, tint_target=zeros, tint=0.0, quality=None, regret=[],
        )  # fmt: skip
        return Level(0, ctx.picture, ctx, result, MosaicScene.from_result(result, ctx))

    def prepare(
        self,
        progress: Progress = lambda message, fraction: None,
        cancelled=lambda: False,
        detail: Progress | None = None,
    ) -> None:
        """Build what matching a level needs and the first level would otherwise build:
        candidates and search indexes, for the shapes of the layout's regions (a start
        that matches nothing, the random mosaic, runs it meanwhile)."""
        report = Reporter(progress, cancelled, detail, None)
        report.step("Slicing", 0.0)
        width = self.layout.columns * SOURCE_PX_PER_TILE
        ctx = SliceContext(blank_picture(self.layout.tile_aspect, width), self.layout)
        regions = self.plan.evaluate(ctx, cancelled=cancelled)[-1].regions
        classes, _ = aspect_classes(regions.size[:, 0] / regions.size[:, 1])
        weights = level_weights(self.method.weights(), FULL_DETAIL)
        for aspect in classes:
            report.step(f"Indexing tiles for {aspect:.3g}:1 regions", None)
            key, cands = self.matcher.candidates(aspect, self.method, report.detail)
            if len(cands):
                self.matcher.index(key, cands, self.method, report.detail, weights)
        report.step("Ready", 1.0)

    def picture(self, tile: TileView) -> np.ndarray:
        """(H, W, 4) uint8: the photo as the tile shows it (untinted), at about its own
        resolution (see picture_size). Its thumbnail stands in if the file can't be read."""
        library, slot = self.library, tile.slot
        span = (tile.rect[2] - tile.rect[0]) * float(library.width[slot])
        size = picture_size(span, self.layout.tile_aspect, self.layout.columns)
        images, _ = render_crops_or_thumbs(
            library.paths([slot]), np.array([tile.rect]), np.array([size]),
            library.thumbs[[slot]], library.thumb_size[[slot]], workers=0,
        )  # fmt: skip
        image = images[0][:, ::-1] if tile.mirrored else images[0]
        alpha = np.full((*image.shape[:2], 1), 255, np.uint8)
        return np.ascontiguousarray(np.concatenate([image, alpha], axis=2))

    def level(
        self,
        picture: np.ndarray,
        progress: Progress = lambda message, fraction: None,
        cancelled=lambda: False,
        detail: Progress | None = None,
        threads: int | None = None,
    ) -> Level:
        """Slice and match a picture (see the module docstring for what is left out).

        threads: search threads (faiss's, for the calling thread; None: as many as it
        has), fewer while the UI animates (DIVE_THREADS).
        """
        if threads is not None:
            faiss.omp_set_num_threads(int(threads))
        timings = {}
        report = Reporter(progress, cancelled, detail, None)
        report.step("Slicing", 0.0)
        ctx = SliceContext(picture, self.layout)
        regions = self.plan.evaluate(ctx, cancelled=cancelled)[-1].regions
        report.step("Analyzing regions", 0.05, "slicing")
        run = MatchRun(self.matcher, regions, ctx, self.method, None, report, self.efforts)
        run.analyze()
        run.search()
        run.pin()
        run.assign()
        report.step("Done", 1.0, "assign")
        result = run.result(lean=True)
        timings.update(report.timings)
        scene = MosaicScene.from_result(result, ctx)
        self.levels += 1
        return Level(self.levels, picture, ctx, result, scene, timings)


__all__ = [
    "DIVE",
    "DIVE_THREADS",
    "MATCHING",
    "MOSAIC",
    "MAX_SOURCE_PX",
    "SOURCE_PX_PER_TILE",
    "InfiniteSession",
    "InfiniteSettings",
    "Level",
    "TileView",
    "blank_picture",
    "centered_crop",
    "centered_crops",
    "picture_size",
]

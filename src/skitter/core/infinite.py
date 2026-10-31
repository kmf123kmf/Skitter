"""Infinite Mosaic: dive into any tile, and its photo becomes the next mosaic.

A session keeps its settings (`InfiniteSettings`: tile shape, columns,
slicer, a few matching rules) from start to end. It starts from a random
library photo (matched as the first level), or from a random mosaic: random
photos in the layout, over a canvas of the view's shape (`random_mosaic`,
level 0, matching nothing; `prepare` builds the search index meanwhile).
Each level is a picture sliced and matched; diving into a
tile makes that tile's photo, as the tile shows it (its crop, mirrored,
untinted), the next level's picture (`InfiniteSession.picture`). Every
picture has the shape of the tile it is of (the base tile's, or a slicer's
turned one: Photo Pile's landscape photos in a portrait layout), so every
level's canvas does too, and a tile seen up close becomes its own mosaic
seamlessly.

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

Loops (auto-dive's Loop setting): a dive returns, within a range of dives,
to the picture it started from (`Loop`). Home is the first tile dived into.
Each level from the bottom of the range on is matched with a lure
(`Lure`): after the search, home's exact crop is costed against every
region that shows whole, inside the canvas; where it fits at most Closing
fit worse than the region's own best, it is pinned there (so it looks as if
it had been matched), and at the top of the range at its best region
regardless (`level`, `Level.home`). Diving into it closes the loop: the
next level is the loop's first, which every pass then replays as recorded
(its levels and the tiles dived into), so one pass can be rendered as a
seamless video.

Home gets its own tint (`Lure.tint`, Home tint): matching weighs color by
what tinting leaves of it, so a full tint lets home go where its structure
fits, its color pulled to the spot's. In the range, auto-dive looks ahead
(`look_ahead`): it matches several candidate tiles' pictures, lure and all,
and dives into the one where home fits best, whose level is then ready.
Measured on 200,000 photos (2026-10-31), home's error at its best spot over
the spot's best: median 2.0 untinted, 1.5 fully tinted; within 1.2 one draw
in ten, 1.3 one in five. So a natural close takes several draws, which
look-ahead provides (4 candidates on each of 4 levels: 16).

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
import numba
import numpy as np
from PIL import Image

from skitter.core import dive_styles
from skitter.core.dive_path import CENTER_FIRST, STRAIGHT
from skitter.core.matching.candidates import Pins
from skitter.core.matching.index import rerank
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
    RangeParam,
)
from skitter.core.tiles.crops import aspect_classes
from skitter.core.tiles.descriptors import DIM, LEVEL2, MEAN, TEXTURE, tile_descriptors
from skitter.core.tiles.library import TileLibrary
from skitter.core.tiles.render import render_crops_or_thumbs

MOSAIC = ("start", "tile_aspect", "columns", "slicer")  # InfiniteSettings by group
MATCHING = ("tint", "tint_strength", "max_uses", "min_spacing", "mirror")
DIVE = ("dive_seconds", "dive_style", "fade_seconds")
AUTO = ("motion", "flow_seconds", "flow_path", "flow_mosaic", "dwell_seconds", "next_tile",
        "variety", "approach_seconds", "approach_tiles", "pause_seconds", "highlight", "loop",
        "loop_dives", "loop_fit", "loop_tint", "look_ahead")  # fmt: skip
LIVE = DIVE + AUTO  # settings a running session follows as they change (pacing only)
SOURCE_PX_PER_TILE = 16  # least picture pixels across a base tile (4 per cell: full detail)
MAX_SOURCE_PX = 4096  # longest side a level's picture is read at
DIVE_THREADS = max(2, (os.cpu_count() or 4) // 4)  # search threads while the UI animates
LOOK_SAMPLE = 4096  # library photos a session's typical look is taken from (`Looks`)
VARIETY_POWER = 4.0  # Favor varied tiles at 100%: picks weighted by variety to this power


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
             "others leave the view.",
    )  # fmt: skip
    dive_style = ChoiceParam(
        "fly_out", "Tiles leave by", choices=dive_styles.CHOICES,
        help="How the other tiles clear the view during a dive. Fly out: straight away "
             "from the tile. Vortex: swirling around it. Ripple: shrinking away, nearest "
             "first. Fade away: dissolving where they lie. Fall away: dropping off the "
             "bottom. Card flip: turning edge-on in a wave. Wind: blown off one side. "
             "Random: a different one each dive (a loop replays the same ones).",
    )  # fmt: skip
    fade_seconds = FloatParam(
        0.6, "Fade time", min=0.0, max=5.0, step=0.1, decimals=1, suffix=" s",
        help="How long the photo takes to dissolve into its own mosaic once it is matched.",
    )  # fmt: skip

    motion = ChoiceParam(
        "steps", "Auto-dive moves",
        choices=[("steps", "In steps"), ("flow", "Continuously")],
        help="In steps: each mosaic shows whole for a while, the view zooms in on a tile and "
             "dives into it. Continuously: one never-ending zoom; each mosaic's next tile is "
             "picked as it appears and its mosaic made while the view heads for it.",
    )  # fmt: skip
    flow_seconds = FloatParam(
        6.5, "Time per mosaic", min=2.0, max=120.0, step=0.5, decimals=1, suffix=" s",
        under="motion", when=lambda s: s.motion == "flow",
        help="Continuously: about how long the zoom takes from a whole mosaic onto one of "
             "its tiles (a base tile; larger tiles take less). Slower is calmer.",
    )  # fmt: skip
    flow_path = ChoiceParam(
        "straight", "Path", choices=[(STRAIGHT, "Straight in"), (CENTER_FIRST, "Center first")],
        under="motion", when=lambda s: s.motion == "flow",
        help="Continuously: Straight in moves toward the tile while zooming all the way; "
             "Center first brings it to the middle early, then zooms straight in. Either way "
             "the view stays on the mosaic and only waits to center a tile near its edge.",
    )  # fmt: skip
    flow_mosaic = FloatParam(
        35.0, "Mosaic appears at", min=5.0, max=100.0, step=5.0, decimals=0,
        suffix="% of the view", under="motion", when=lambda s: s.motion == "flow",
        help="Continuously: how much of the view the tile fills by the time its photo has "
             "turned into its own mosaic, shown inside it as the view heads on in. Lower "
             "shows the next mosaic sooner; 100%: as the tile fills the view.",
    )  # fmt: skip
    dwell_seconds = FloatParam(
        3.0, "Dwell", min=0.0, max=600.0, step=0.5, decimals=1, suffix=" s",
        when=lambda s: s.motion == "steps",
        help="Auto-dive: how long each new mosaic shows whole before the view heads for "
             "the next tile.",
    )  # fmt: skip
    next_tile = ChoiceParam(
        "anywhere", "Next tile",
        choices=[("anywhere", "Anywhere in view"), ("middle", "Near the middle")],
        help="Auto-dive: where the next tile is picked, at random among the tiles in view "
             "(those mostly hidden under others left out).",
    )  # fmt: skip
    variety = FloatParam(
        70.0, "Favor varied tiles", min=0.0, max=100.0, step=10.0, decimals=0, suffix="%",
        help="Auto-dive: how strongly the pick favors tiles with something in them (light "
             "and dark, detail) and of the library's usual brightness and color. Without it "
             "a dive into a dark or plain tile tends to lead to more of the same, mosaic "
             "after mosaic. 0: every tile in view is as likely.",
    )  # fmt: skip
    approach_seconds = FloatParam(
        2.0, "Zoom-in time", min=0.0, max=60.0, step=0.5, decimals=1, suffix=" s",
        when=lambda s: s.motion == "steps",
        help="Auto-dive: how long the view takes to move to the tile and zoom in on it "
             "before diving (0: dive straight from where the view is).",
    )  # fmt: skip
    approach_tiles = FloatParam(
        6.0, "Zoom in to", min=1.0, max=500.0, step=1.0, decimals=1, suffix=" tiles across",
        when=lambda s: s.motion == "steps",
        help="Auto-dive: how close the view gets before diving, as the number of tiles "
             "across it. With the zoom-in time this sets how fast the view zooms.",
    )  # fmt: skip
    pause_seconds = FloatParam(
        0.5, "Pause on tile", min=0.0, max=30.0, step=0.1, decimals=1, suffix=" s",
        when=lambda s: s.motion == "steps",
        help="Auto-dive: how long the view rests on the chosen tile before diving into it.",
    )  # fmt: skip
    highlight = BoolParam(
        True, "Highlight the tile", when=lambda s: s.motion == "steps",
        help="Auto-dive: outline the chosen tile while the view zooms in on it.",
    )  # fmt: skip
    loop = BoolParam(
        False, "Loop", when=lambda s: s.motion == "steps",
        help="Auto-dive comes back to where it started: the first tile it dives into turns "
             "up again, placed where it fits, within the range of dives below. After that "
             "it replays the same dives, again and again.",
    )  # fmt: skip
    loop_dives = RangeParam(
        (3, 6), "Loop within", min=1, max=50, step=1, decimals=0, suffix=" dives", whole=True,
        under="loop", when=lambda s: s.loop and s.motion == "steps",
        help="How many dives one pass of the loop takes: it closes as soon as the first tile "
             "fits naturally (see Closing fit) once the low end is reached, and at the high "
             "end regardless. 1: the picture contains itself.",
    )  # fmt: skip
    loop_fit = FloatParam(
        30.0, "Closing fit", min=0.0, max=500.0, step=5.0, decimals=0, suffix="%",
        under="loop_dives", when=lambda s: s.loop and s.motion == "steps",
        help="How much more error than the best tile for its spot the first tile may have "
             "there for the loop to close early (0: only where it is the best). Larger "
             "closes sooner but can look placed.",
    )  # fmt: skip
    loop_tint = FloatParam(
        100.0, "Home tint", min=0.0, max=100.0, step=10.0, decimals=0, suffix="%",
        under="loop_dives", when=lambda s: s.loop and s.motion == "steps",
        help="How far the first tile's color is pulled to its spot's when it comes back. "
             "Stronger lets it go where its shapes fit and blend in by color (it loses the "
             "tint as the view dives in).",
    )  # fmt: skip
    look_ahead = IntParam(
        4, "Look ahead", min=1, max=16, suffix=" tiles",
        under="loop_dives", when=lambda s: s.loop and s.motion == "steps",
        help="In the loop's range, how many tiles' mosaics are made before each dive, to "
             "dive into the one the first tile fits best (in the background, while the "
             "mosaic dwells). More closes the loop sooner and more naturally, but takes "
             "longer: about half a second each at 40 columns.",
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
    picture: np.ndarray  # (H, W, 4) uint8, the shape of the tile it is of
    ctx: SliceContext
    result: MatchResult
    scene: MosaicScene
    timings: dict = field(default_factory=dict)  # seconds per step
    source: "TileView | None" = None  # the tile it is the mosaic of (None: not a tile's)
    home: int = -1  # scene index of the lure's tile, placed (see Lure; -1: none)
    home_fit: float = float("nan")  # its error there over the spot's best's (1: the best)
    variety: np.ndarray | None = None  # (n,) each tile's, scene order (see Looks.variety)


@dataclass(frozen=True)
class Looks:
    """What a library's photos usually look like, as tiles: the reference auto-dive's
    picks are weighed against (`variety`).

    Each crop is summed up by its average OKLab color and its liveliness: the spread of
    lightness across its 4 x 4 cells (contrast) with the spread within them (detail)."""

    mean: np.ndarray  # (3,) the median photo's average color
    spread: np.ndarray  # (3,) how far averages usually are from it, per channel
    lively: float  # the median liveliness

    @staticmethod
    def of(desc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) average colors and (N,) liveliness of crops' descriptors."""
        desc = np.asarray(desc, np.float64)
        lightness = desc[:, LEVEL2][:, 0::3]  # the cells' lightness, less the average
        contrast = np.sqrt(np.mean(lightness**2, axis=1))
        detail = desc[:, TEXTURE].mean(axis=1)
        return desc[:, MEAN], np.hypot(contrast, detail)

    @classmethod
    def sample(cls, library: TileLibrary, aspect: float, rng, count: int = LOOK_SAMPLE):
        """The looks of up to count random library photos, centered crops of this shape."""
        ids = library.ids
        slots = np.sort(rng.choice(ids, min(count, len(ids)), replace=False))
        rects = centered_crops(library.width[slots], library.height[slots], aspect)
        mean, lively = cls.of(describe_crops(library, slots, rects))
        q1, median, q3 = np.percentile(mean, [25, 50, 75], axis=0)
        spread = np.maximum((q3 - q1) / 1.35, 1e-3)  # a robust standard deviation
        return cls(median, spread, max(float(np.median(lively)), 1e-6))

    def variety(self, desc: np.ndarray) -> np.ndarray:
        """(N,) in [0, 1] for crops' descriptors: 1 for a crop at least as lively as the
        median photo and of about its color; toward 0 for flat ones, or ones far darker,
        brighter or more colored than photos usually are."""
        mean, lively = self.of(desc)
        livelier = np.clip(lively / self.lively, 0.0, 1.0)
        off = (mean - self.mean) / self.spread
        usual = np.exp(-0.5 * np.sum(off**2, axis=1) / len(self.mean))
        return livelier * usual


def describe_crops(library: TileLibrary, slots, rects) -> np.ndarray:
    """(N, DIM) descriptors of crops (rects: fractions) of library photos, from their
    thumbnails: only those photos' are read."""
    slots = np.asarray(slots, np.int64)
    rects = np.asarray(rects, np.float32).reshape(-1, 4)
    if not len(slots):
        return np.zeros((0, DIM), np.float32)
    photos, which = np.unique(slots, return_inverse=True)
    order = np.argsort(which, kind="stable")  # grouped by photo, as tile_descriptors wants
    desc = np.empty((len(slots), DIM), np.float32)
    desc[order] = tile_descriptors(library.thumbs[photos], library.thumb_size[photos],
                                   which[order], rects[order])  # fmt: skip
    return desc


def pick_weights(variety: np.ndarray | None, strength: float) -> np.ndarray | None:
    """Probabilities of picking tiles of this variety (Favor varied tiles: strength 0 to
    1), or None for every tile alike."""
    if variety is None or strength <= 0 or not len(variety):
        return None
    weights = np.maximum(np.asarray(variety, np.float64), 1e-6) ** (VARIETY_POWER * strength)
    return weights / weights.sum()


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


def display_copy(picture: np.ndarray, longest: int) -> np.ndarray:
    """picture, scaled down (if need be) so its longest side is at most longest pixels:
    as much as a view shows of it (a full-size one would stall a frame to upload)."""
    height, width = picture.shape[:2]
    scale = longest / max(height, width)
    if scale >= 1:
        return picture
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(Image.fromarray(picture).resize(size, Image.Resampling.LANCZOS))


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
        self._looks: Looks | None = None

    @property
    def looks(self) -> Looks:
        """The library's usual looks (sampled once, at the tile shape)."""
        if self._looks is None:
            self._looks = Looks.sample(self.library, self.layout.tile_aspect,
                                       np.random.default_rng(0))  # fmt: skip
        return self._looks

    def variety(self, scene: MosaicScene) -> np.ndarray:
        """(n,) each tile's variety (Looks.variety), in scene order."""
        return self.looks.variety(describe_crops(self.library, scene.slot, scene.rect))

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
        scene = MosaicScene.from_result(result, ctx)
        scene.visible_share  # noqa: B018 (auto-dive asks; at the start, not mid-animation)
        return Level(0, ctx.picture, ctx, result, scene)  # its variety: in `prepare`

    def prepare(
        self,
        progress: Progress = lambda message, fraction: None,
        cancelled=lambda: False,
        detail: Progress | None = None,
        start: Level | None = None,
    ) -> None:
        """Build what matching a level needs and the first level would otherwise build:
        candidates and search indexes, for the shapes of the layout's regions (a start
        that matches nothing, the random mosaic, runs it meanwhile), and the library's
        usual looks. start: a level to give its variety (the random mosaic's: reading
        its thumbnails would stall the UI thread)."""
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
        report.step("Sampling the library's looks", None)
        if start is not None:
            start.variety = self.variety(start.scene)
        else:
            self.looks  # noqa: B018 (sampled here, in the background)
        report.step("Ready", 1.0)

    def picture(self, tile: TileView, path: str | None = None) -> np.ndarray:
        """(H, W, 4) uint8: the photo as the tile shows it (untinted), in its crop's shape,
        at about its own resolution (see picture_size). Its thumbnail stands in if the file
        can't be read. path: the photo's (given, it may run off the library's thread: no
        database)."""
        library, slot = self.library, tile.slot
        x0, y0, x1, y1 = tile.rect
        span = (x1 - x0) * float(library.width[slot])
        tall = (y1 - y0) * float(library.height[slot])
        # The crop's own shape: a turned tile's is not the base tile's (unknown size: that).
        aspect = span / tall if span > 0 and tall > 0 else self.layout.tile_aspect
        size = picture_size(span, aspect, self.layout.columns)
        images, _ = render_crops_or_thumbs(
            library.paths([slot]) if path is None else [path], np.array([tile.rect]),
            np.array([size]),
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
        lure: "Lure | None" = None,
        source: TileView | None = None,
        count: bool = True,
    ) -> Level:
        """Slice and match a picture (see the module docstring for what is left out).

        threads: search threads (faiss's, for the calling thread; None: as many as it
        has), fewer while the UI animates (DIVE_THREADS). lure: a tile to place where it
        fits well enough (a loop's home; see Lure). source: the tile the picture is of.
        count: number it as the session's next level (else 0: a look-ahead's, numbered
        if it is shown).
        """
        if threads is not None:
            # numba's first: its parallel kernels (they run before the search) set OpenMP's
            # count, faiss's too, to numba's for the calling thread.
            numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))
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
        region, fit = (-1, float("nan")) if lure is None else self._place(run, lure)
        run.pin()
        run.assign()
        report.step("Done", 1.0, "assign")
        result = run.result(lean=True)
        timings.update(report.timings)
        scene = MosaicScene.from_result(result, ctx)
        scene.visible_share  # noqa: B018 (auto-dive asks: here, not on the UI thread)
        scene.overlaps  # noqa: B018 (and continuous auto-dive: what lies on its tile)
        number = 0
        if count:
            self.levels += 1
            number = self.levels
        home = np.flatnonzero(scene.region == region) if region >= 0 else ()
        home = int(home[0]) if len(home) else -1
        if home >= 0 and lure.tint is not None:  # its own tint (the arrays are its scene's)
            mean = result.tile_mean[region]
            scene.tint_shift[home] = lure.tint * (scene.target_color[home] - mean)
            scene.tile_color[home] = mean + scene.tint_shift[home]
        variety = self.variety(scene)  # auto-dive's weights: here, not on the UI thread
        return Level(number, picture, ctx, result, scene, timings, source, home, fit, variety)

    def look_ahead(
        self,
        candidates: list[tuple[int, TileView, str]],
        lure: "Lure",
        progress: Progress = lambda message, fraction: None,
        cancelled=lambda: False,
        threads: int | None = None,
    ) -> list[tuple[int, Level]]:
        """Match each candidate tile's picture (index, tile, its photo's path) with the
        lure: [(index, level)], best first (home placed, then by how home fits). The
        levels are numbered 0 (see `level`'s count)."""
        done = []
        for k, (index, tile, path) in enumerate(candidates):
            progress(f"Looking ahead: {k + 1} of {len(candidates)}", k / len(candidates))
            picture = self.picture(tile, path)
            level = self.level(picture, cancelled=cancelled, threads=threads, lure=lure,
                               source=tile, count=False)  # fmt: skip
            done.append((index, level))
        done.sort(key=lambda item: (item[1].home < 0, item[1].home_fit))
        return done

    def number(self, level: Level) -> Level:
        """Number a look-ahead's level as the session's next (it is being shown)."""
        if level.number == 0:
            self.levels += 1
            level.number = self.levels
        return level

    def _place(self, run: MatchRun, lure: "Lure") -> tuple[int, float]:
        """(region, fit): where the lure's tile goes (pinned on the run; -1: nowhere) and
        how it fits there (its cost over the region's best). Only regions that show whole,
        inside the canvas, and of the tile's shape are considered."""
        tile = lure.tile
        regions, ctx = run.regions, run.ctx
        lo, hi = _extent(regions)
        inside = (lo[:, 0] >= 0) & (lo[:, 1] >= 0) & (hi[:, 0] <= ctx.width) & (
            hi[:, 1] <= ctx.height)  # fmt: skip
        whole = run.targets.visible >= 0.999
        best_region, best_fit = -1, np.inf
        for cands, regs in run.shapes.values():
            regs = regs[inside[regs] & whole[regs]]
            if not len(regs):
                continue
            cands, ref = cands.with_crops(self.library, [tile.slot], [tile.rect],
                                          [tile.mirrored])  # fmt: skip
            weights = run.weights[regs].copy()
            if lure.tint is not None and run.tint < 1:  # color as left by its own tint
                weights[:, MEAN] *= ((1 - lure.tint) / (1 - run.tint)) ** 2
            _, cost = rerank(cands, np.full((len(regs), 1), ref[0]), run.targets.desc[regs],
                             weights, run.settings.crop_penalty)  # fmt: skip
            best = run.costs[regs, 0]
            best = np.where(np.isfinite(best), best, cost[:, 0])
            fit = np.sqrt(cost[:, 0] / np.maximum(best, 1e-12))  # error over the best's
            k = int(np.argmin(fit))
            if fit[k] < best_fit:
                best_region, best_fit = int(regs[k]), float(fit[k])
        if best_region < 0 or (best_fit > 1 + lure.tolerance and not lure.force):
            return -1, best_fit
        run.pins = Pins(np.array([best_region]), np.array([tile.slot], np.int64),
                        np.array([tile.rect], np.float32), np.array([tile.mirrored]))  # fmt: skip
        return best_region, best_fit


def _extent(regions) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) (R, 2) corners of each region's axis-aligned bounding box."""
    half = np.abs(regions.size) / 2
    c, s = np.abs(np.cos(regions.rotation)), np.abs(np.sin(regions.rotation))
    reach = np.stack([half[:, 0] * c + half[:, 1] * s, half[:, 0] * s + half[:, 1] * c], axis=1)
    return regions.center - reach, regions.center + reach


@dataclass(frozen=True)
class Lure:
    """A tile for a level's matching to place where it fits (see the module docstring)."""

    tile: TileView
    tolerance: float  # how much worse than a spot's best it may fit there (0.1: 10%)
    force: bool = False  # place it at its best spot however it fits
    # Its own tint strength (None: the level's). Matching weighs color by how much is
    # left after tinting, so a stronger tint lets it go where its structure fits, its
    # color pulled to the spot's (the eye judges a tile at mosaic scale by its color).
    tint: float | None = None


@dataclass
class Loop:
    """A cycle of levels coming back to its first (see the module docstring).

    levels[i] is the loop's level i + 1 (levels[0]: the mosaic of home); picks[i] the
    tile dived into from it. While recording, each new level is matched with lure();
    once a level holds home (Level.home) diving into it closes the loop, and from then
    on every pass replays levels and picks as recorded.
    """

    home: TileView
    levels: list[Level] = field(default_factory=list)
    picks: list[int] = field(default_factory=list)
    closed: bool = False
    passes: int = 0  # times the loop has come back to its first level

    def lure(self, settings: "InfiniteSettings") -> Lure | None:
        """The lure for matching the loop's next level (None: before the range)."""
        number = len(self.levels) + 1
        low, high = settings.loop_dives
        if number < low:
            return None
        return Lure(self.home, settings.loop_fit / 100.0, force=number >= high,
                    tint=settings.loop_tint / 100.0)  # fmt: skip

    @property
    def closing(self) -> int:
        """Index of the level holding home (where the loop closes), or -1."""
        last = self.levels[-1] if self.levels else None
        return len(self.levels) - 1 if last is not None and last.home >= 0 else -1


__all__ = [
    "AUTO",
    "DIVE",
    "DIVE_THREADS",
    "LIVE",
    "Loop",
    "Lure",
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

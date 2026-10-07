"""A matched mosaic reduced to what a project file keeps, and rebuilt from it.

Per region a project keeps the photo shown (by file path, so the mosaic
survives library updates that renumber or add photos), its crop window, its
mirroring and whether it was picked by hand, plus the tint targets, the
costs and the run's settings and statistics.

Candidate lists for picking by hand are not kept: their indices only mean
something to the run that made them. A restored result has none; matching
again with the picks kept (Pins) rebuilds them.

`restore` finds each photo by path in the current library (regions whose
photo is gone get no tile and are counted), describes the crops again from
the library thumbnails (as matching did), and evaluates their quality anew.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from skitter.core.matching.edit import usage_stats
from skitter.core.matching.quality import evaluate
from skitter.core.matching.result import MatchResult, RegretStats
from skitter.core.matching.settings import MatchSettings
from skitter.core.slicing import RegionSet, SliceContext
from skitter.core.tiles.descriptors import DIM, MEAN, cell_colors, mirror, tile_descriptors
from skitter.core.tiles.library import TileLibrary

ARRAYS = ("photo", "rect", "mirrored", "manual", "cost", "tint_target")


@dataclass(frozen=True)
class SavedMosaic:
    paths: list[str]  # the distinct photos shown, by file path
    photo: np.ndarray  # (R,) int32 index into paths (-1: no tile)
    rect: np.ndarray  # (R, 4) float32 crop window, fractions of the photo
    mirrored: np.ndarray  # (R,) bool
    manual: np.ndarray  # (R,) bool: picked by hand
    cost: np.ndarray  # (R,) float32 matching cost
    tint_target: np.ndarray  # (R, 3) float32 OKLab color each tile is tinted toward
    tint: float
    settings: MatchSettings  # the settings it was matched with
    library_version: int  # of the library it was matched against
    regret: list[RegretStats] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def arrays(self) -> dict[str, np.ndarray]:
        """The per-region arrays, by name."""
        return {name: getattr(self, name) for name in ARRAYS}

    def meta(self) -> dict:
        """Everything else, as plain data (JSON)."""
        return {
            "paths": self.paths,
            "tint": self.tint,
            "settings": self.settings.values(),
            "library_version": self.library_version,
            "regret": [asdict(r) for r in self.regret],
            "stats": self.stats,
        }

    @classmethod
    def from_parts(cls, arrays: dict, meta: dict, problems: list[str] | None = None):
        regret = []
        for item in meta.get("regret", []):
            try:
                regret.append(RegretStats(**item))
            except TypeError:
                pass  # statistics only: a changed format just drops them
        return cls(
            paths=list(meta["paths"]),
            photo=np.asarray(arrays["photo"], np.int32),
            rect=np.asarray(arrays["rect"], np.float32),
            mirrored=np.asarray(arrays["mirrored"], bool),
            manual=np.asarray(arrays["manual"], bool),
            cost=np.asarray(arrays["cost"], np.float32),
            tint_target=np.asarray(arrays["tint_target"], np.float32),
            tint=float(meta["tint"]),
            settings=MatchSettings.from_values(
                meta.get("settings", {}), problems, "Matched with"
            ),  # fmt: skip
            library_version=int(meta.get("library_version", -1)),
            regret=regret,
            stats=dict(meta.get("stats", {})),
        )


def saved_mosaic(result: MatchResult, library: TileLibrary) -> SavedMosaic:
    """What a project file keeps of a result matched against library."""
    placed = result.tile >= 0
    slots, photo = np.unique(result.tile[placed], return_inverse=True)
    index = np.full(len(result.tile), -1, np.int32)
    index[placed] = photo
    manual = result.manual if result.manual is not None else np.zeros(len(result.tile), bool)
    return SavedMosaic(
        paths=library.paths(slots),
        photo=index,
        rect=np.asarray(result.rect, np.float32),
        mirrored=np.asarray(result.mirrored, bool),
        manual=np.asarray(manual, bool),
        cost=np.asarray(result.cost, np.float32),
        tint_target=np.asarray(result.tint_target, np.float32),
        tint=float(result.tint),
        settings=(result.settings or MatchSettings()).copy(),
        library_version=library.version,
        regret=list(result.regret),
        stats=dict(result.stats),
    )


def restore(
    saved: SavedMosaic, library: TileLibrary, regions: RegionSet, ctx: SliceContext
) -> tuple[MatchResult, int]:
    """(result, missing): the saved mosaic over regions, its photos found by path in
    library; missing counts regions whose photo isn't there (they get no tile)."""
    if len(saved.photo) != len(regions):
        raise ValueError(f"the mosaic has {len(saved.photo)} regions, the plan {len(regions)}")
    ids = library.ids
    slot_of = dict(zip(library.paths(ids), ids.tolist(), strict=True))
    photo_slot = np.array([slot_of.get(p, -1) for p in saved.paths] + [-1], np.int64)
    tile = photo_slot[np.where(saved.photo >= 0, saved.photo, len(saved.paths))]
    missing = int(np.count_nonzero((saved.photo >= 0) & (tile < 0)))
    placed = tile >= 0

    # The crops described as matching described them (float16 candidates).
    desc = np.zeros((len(tile), DIM), np.float32)
    order = np.flatnonzero(placed)
    order = order[np.argsort(tile[order], kind="stable")]
    if len(order):
        found = tile_descriptors(library.thumbs, library.thumb_size, tile[order], saved.rect[order])
        desc[order] = found.astype(np.float16).astype(np.float32)
    desc[saved.mirrored] = mirror(desc[saved.mirrored])
    tile_mean = desc[:, MEAN].copy()
    shift = saved.tint * (saved.tint_target - tile_mean)
    quality = evaluate(regions, ctx, cell_colors(desc) + shift[:, None, None, :], placed)

    result = MatchResult(
        regions=regions, tile=tile, rect=saved.rect.copy(), mirrored=saved.mirrored.copy(),
        cost=np.where(placed, saved.cost, np.inf).astype(np.float32), tile_mean=tile_mean,
        tint_target=saved.tint_target.copy(), tint=saved.tint, quality=quality,
        regret=list(saved.regret), stats=dict(saved.stats), manual=saved.manual & placed,
        settings=saved.settings.copy(),
    )  # fmt: skip
    result.stats.update(usage_stats(result, ctx))
    return result, missing

"""The Jigsaw Puzzle method: the picture itself, cut along the regions.

Nothing is matched. Every region that shows (not wholly covered by regions
above it, judged conservatively: raster.may_show) becomes a piece: the
picture seen through that region, wherever the piece goes. Pieces are never
cut out as separate images; whatever draws one samples the picture at the
piece's home footprint (scene.picture_uv, assembly.render_picture), so the
pieces at home are the picture exactly, and a project stores nothing but
the method.

Piece shapes are the slicer's (rectangles, for now).
"""

import numpy as np
from PIL import Image

from skitter.core.color import rgb8_to_oklab
from skitter.core.matching.method import MatchMethod, register_method
from skitter.core.matching.raster import may_show
from skitter.core.matching.result import MatchResult
from skitter.core.slicing import RegionSet, SliceContext

COLOR_SIDE = 1024  # piece colors are averaged on the picture reduced to this size


@register_method
class JigsawMethod(MatchMethod):
    id = "jigsaw"
    name = "Jigsaw Puzzle"
    description = ("No matching: the picture itself, cut along the regions. Each piece shows "
                   "its own part of the picture, so the finished puzzle is the picture "
                   "exactly; the animations do the rest.")  # fmt: skip
    needs_library = False

    def match(self, matcher, regions, ctx, pins, report):
        report.step("Cutting the picture into pieces", 0.0)
        result = cut_picture(regions, ctx, self)
        report.step("Done", 1.0, "cut")
        result.stats["timings"] = report.timings
        return result

    def restore(self, saved, library, regions, ctx):
        return cut_picture(regions, ctx, saved.settings.copy()), 0


def cut_picture(regions: RegionSet, ctx: SliceContext, method: MatchMethod) -> MatchResult:
    """The picture as pieces: one per region that shows."""
    n = len(regions)
    # Only pieces certainly hidden are left out: one dropped while a sliver of it shows
    # would leave a gap in the picture.
    order = regions.stacking_order()
    shown = np.zeros(n, bool)
    shown[order] = may_show(regions.center[order], regions.size[order], regions.rotation[order])
    tile = np.where(shown, np.arange(n), -1).astype(np.int64)
    half = regions.size / 2
    canvas = np.array([ctx.width, ctx.height])
    rect = np.concatenate([(regions.center - half) / canvas, (regions.center + half) / canvas],
                          axis=1).astype(np.float32)  # fmt: skip
    mean = piece_colors(regions, ctx)
    return MatchResult(
        regions=regions, tile=tile, rect=rect, mirrored=np.zeros(n, bool),
        cost=np.where(shown, 0.0, np.inf).astype(np.float32), tile_mean=mean,
        tint_target=mean.copy(), tint=0.0, quality=None, regret=[],
        stats=dict(regions=n, placed=int(shown.sum()), hidden=int(n - shown.sum()), timings={}),
        settings=method.copy(), picture=True,
    )  # fmt: skip


def piece_colors(regions: RegionSet, ctx: SliceContext) -> np.ndarray:
    """(R, 3) float32 OKLab average of the visible picture inside each region's bounding
    box (for animations that order or group pieces by color)."""
    h, w = ctx.source_height, ctx.source_width
    f = min(1.0, COLOR_SIDE / max(h, w))
    sw, sh = max(1, round(w * f)), max(1, round(h * f))
    small = np.asarray(Image.fromarray(ctx.image).resize((sw, sh), Image.Resampling.BOX))
    lab = rgb8_to_oklab(small).astype(np.float64)
    if ctx.visible is None:
        seen = np.ones((sh, sw))
    else:
        mask = Image.fromarray(ctx.visible.astype(np.uint8) * 255)
        seen = np.asarray(mask.resize((sw, sh), Image.Resampling.BOX), np.float64) / 255
    sums = _table(np.concatenate([lab * seen[..., None], seen[..., None]], axis=2))
    box = regions.bounds() * np.array([sw / ctx.width, sh / ctx.height] * 2)
    x0 = np.clip(np.floor(box[:, 0]), 0, sw - 1).astype(np.int64)
    y0 = np.clip(np.floor(box[:, 1]), 0, sh - 1).astype(np.int64)
    x1 = np.clip(np.ceil(box[:, 2]), x0 + 1, sw).astype(np.int64)
    y1 = np.clip(np.ceil(box[:, 3]), y0 + 1, sh).astype(np.int64)
    total = sums[y1, x1] - sums[y0, x1] - sums[y1, x0] + sums[y0, x0]
    count = total[:, 3:]
    plain = lab[(y0 + y1) // 2, (x0 + x1) // 2]  # no visible pixel: the color filled in
    mean = np.where(count > 1e-9, total[:, :3] / np.maximum(count, 1e-9), plain)
    return mean.astype(np.float32)


def _table(values: np.ndarray) -> np.ndarray:
    """(H + 1, W + 1, C) summed-area table."""
    table = np.zeros((values.shape[0] + 1, values.shape[1] + 1, values.shape[2]))
    table[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
    return table

"""How the other tiles leave the view while the Infinite Mosaic dives into one.

A dive (ui/infinite_window.py) zooms the view from where it is onto one
tile; meanwhile a style moves, shrinks, turns or fades every other tile so
none is left in view when the dive ends (`test_every_dive_style_clears_the_view`
checks every registered style). A style is a function of the tiles' start
(`DiveTiles`) and the dive's progress e (0 to 1), returning their `DiveFrame`;
the window then puts the dived-into tile back where it belongs. Register new
ones in STYLES.

Per-tile randomness (timing, tumbling, the wind's direction) comes from
`DiveTiles.rand`, seeded from the dive (see `dive_seed`), so a loop's
replay dives exactly as its first pass did. So does Random's pick.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from skitter.core.easing import ease_in_cubic, ease_in_out_cubic, ease_out_cubic

FLY_OUT = 3.0  # how far Fly out moves the tiles, in their distances from the dived-into tile
RANDOM = "random"  # the style id that picks one of STYLES at random for each dive


@dataclass
class DiveFrame:
    """Every tile's look at one moment of a dive (the dived-into one is the window's)."""

    pos: np.ndarray  # (n, 2) world centers
    size: np.ndarray  # (n, 2)
    rotation: np.ndarray  # (n,) radians
    alpha: np.ndarray  # (n,)
    shade: np.ndarray | None = None  # (n,) darkening in [0, 1], or None for none


class DiveTiles:
    """The tiles as a dive into tile `index` starts, and the view's course.

    view0 and view1: the world width and height the view shows at the start
    and at the end; between them it zooms geometrically over the eased dive
    (`view_at`, as the window does). down: the world direction that shows as
    down on screen."""

    def __init__(self, pos, size, rotation, alpha, index: int, view0, view1,
                 seed: int = 0, down=(0.0, 1.0)):  # fmt: skip
        self.pos = np.asarray(pos, dtype=np.float64)
        self.size = np.asarray(size, dtype=np.float64)
        self.rotation = np.asarray(rotation, dtype=np.float64)
        self.alpha = np.asarray(alpha, dtype=np.float64)
        self.index = index
        self.center = self.pos[index].copy()
        self.view0 = np.asarray(view0, dtype=np.float64)
        self.view1 = np.asarray(view1, dtype=np.float64)
        self.down = np.asarray(down, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.rand = self.rng.random((len(self.pos), 4))  # per-tile draws in [0, 1)
        self.heading = 2 * np.pi * float(self.rng.random())  # a direction for the dive
        self.offset = self.pos - self.center  # from the dived-into tile
        self.distance = np.linalg.norm(self.offset, axis=1)
        far = float(self.distance.max()) if len(self.pos) else 0.0
        self.far = far if far > 0 else 1.0
        # Tiles right on the center (stacked) leave in a direction of their own.
        angle = 2 * np.pi * self.rand[:, 3]
        lone = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        safe = np.maximum(self.distance, 1e-9)[:, None]
        self.direction = np.where(self.distance[:, None] > 1e-6, self.offset / safe, lone)

    def __len__(self) -> int:
        return len(self.pos)

    def view_at(self, e) -> np.ndarray:
        """The view's world width and height at e (shape (..., 2) for an array e)."""
        m = np.asarray(ease_in_out_cubic(e), dtype=np.float64)[..., None]
        return self.view0 * (self.view1 / self.view0) ** m

    def frame(self, pos=None, size=None, rotation=None, alpha=None, shade=None) -> DiveFrame:
        """A frame, the start's values where none is given."""
        return DiveFrame(self.pos.copy() if pos is None else pos,
                         self.size.copy() if size is None else size,
                         self.rotation.copy() if rotation is None else rotation,
                         self.alpha.copy() if alpha is None else alpha, shade)  # fmt: skip


@dataclass(frozen=True)
class DiveStyle:
    id: str
    name: str
    frame: Callable[[DiveTiles, float], DiveFrame]


def _local(e: float, start: np.ndarray, span) -> np.ndarray:
    """Each tile's own progress (0 to 1), starting at start and lasting span."""
    return np.clip((e - start) / span, 0.0, 1.0)


def _fade_late(s: np.ndarray, begin: float = 0.7) -> np.ndarray:
    """Opacity that falls from 1 to 0 over the end of a progress s (from begin on)."""
    return 1.0 - ease_in_out_cubic(np.clip((s - begin) / (1 - begin), 0.0, 1.0))


def _turn(v: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], axis=1)


def fly_out(tiles: DiveTiles, e: float) -> DiveFrame:
    """Every tile flies straight away from the dived-into one."""
    return tiles.frame(pos=tiles.pos + tiles.offset * (FLY_OUT * float(ease_out_cubic(e))))


VORTEX_TURNS = 0.6  # turns the nearest tiles make around the dived-into one


def vortex(tiles: DiveTiles, e: float) -> DiveFrame:
    """The tiles swirl around the dived-into one as they spread out, like water down a
    drain: the nearer, the faster, each spinning as it goes."""
    near = 1.0 - tiles.distance / tiles.far
    angle = 2 * np.pi * VORTEX_TURNS * float(ease_in_cubic(e)) ** 0.7 * (0.35 + 0.65 * near)
    spread = 1.0 + 4.0 * e * e
    pos = tiles.center + _turn(tiles.offset, angle) * spread
    alpha = tiles.alpha * _fade_late(np.full(len(tiles), e), 0.8)
    return tiles.frame(pos=pos, rotation=tiles.rotation + angle, alpha=alpha)


def ripple(tiles: DiveTiles, e: float) -> DiveFrame:
    """Nothing moves: a ring spreads from the dived-into tile, and each tile it reaches
    shrinks away to nothing."""
    s = ease_in_out_cubic(_local(e, 0.7 * tiles.distance / tiles.far, 0.3))
    keep = 1.0 - s
    return tiles.frame(size=tiles.size * keep[:, None], alpha=tiles.alpha * keep)


def fade(tiles: DiveTiles, e: float) -> DiveFrame:
    """Nothing moves: each tile fades away where it lies, at a moment of its own (nearer
    ones sooner), so the mosaic dissolves around the dived-into tile."""
    start = 0.35 * tiles.distance / tiles.far + 0.35 * tiles.rand[:, 0]
    s = ease_in_out_cubic(_local(e, start, 0.3))
    return tiles.frame(alpha=tiles.alpha * (1.0 - s))


def fall(tiles: DiveTiles, e: float) -> DiveFrame:
    """The table falls away: the tiles let go, nearest first, and drop off the bottom of
    the view, tumbling."""
    r = tiles.rand
    start = 0.45 * tiles.distance / tiles.far + 0.1 * r[:, 0]
    s = _local(e, start, 0.45)
    drop = 1.6 * tiles.view_at(start)[:, 1] * (0.8 + 0.4 * r[:, 1]) * s * s
    side = np.array([-tiles.down[1], tiles.down[0]])
    drift = 0.08 * tiles.view_at(start)[:, 0] * (r[:, 2] - 0.5) * s
    pos = tiles.pos + drop[:, None] * tiles.down + drift[:, None] * side
    rotation = tiles.rotation + 2.4 * (r[:, 3] - 0.5) * s * s
    return tiles.frame(pos=pos, rotation=rotation, alpha=tiles.alpha * _fade_late(s))


def flip(tiles: DiveTiles, e: float) -> DiveFrame:
    """A wave spreads from the dived-into tile and turns each tile it reaches edge-on, like
    a card turning over (it darkens as it turns, lifting a little)."""
    s = ease_in_cubic(_local(e, 0.7 * tiles.distance / tiles.far, 0.3))
    # Each turns about the axis across the wave: it narrows along the wave's direction.
    local = _turn(tiles.direction, -tiles.rotation)
    along_x = np.abs(local[:, 0]) >= np.abs(local[:, 1])
    narrow = np.cos(0.5 * np.pi * s)
    lift = 1.0 + 0.12 * np.sin(np.pi * s)
    size = tiles.size * lift[:, None]
    size[:, 0] *= np.where(along_x, narrow, 1.0)
    size[:, 1] *= np.where(along_x, 1.0, narrow)
    alpha = tiles.alpha * np.where(s < 1.0, 1.0, 0.0)
    return tiles.frame(size=size, alpha=alpha, shade=0.6 * np.sin(0.5 * np.pi * s))


def wind(tiles: DiveTiles, e: float) -> DiveFrame:
    """A gust from one side (a random compass direction for each dive) blows the tiles
    off the other side, the upwind ones first, fluttering."""
    blow = np.array([np.cos(tiles.heading), np.sin(tiles.heading)])
    across = np.array([-blow[1], blow[0]])
    q = tiles.offset @ blow
    q = (q - q.min()) / max(float(np.ptp(q)), 1e-9)
    r = tiles.rand
    start = 0.5 * q + 0.1 * r[:, 0]
    s = _local(e, start, 0.4)
    view = tiles.view_at(start).max(axis=1)
    push = 1.8 * view * (0.7 + 0.6 * r[:, 1]) * s * s
    flutter = 0.05 * view * np.sin(2 * np.pi * (1.5 * s + r[:, 2])) * s
    pos = tiles.pos + push[:, None] * blow + flutter[:, None] * across
    rotation = tiles.rotation + 3.0 * (r[:, 3] - 0.5) * s
    return tiles.frame(pos=pos, rotation=rotation, alpha=tiles.alpha * _fade_late(s))


STYLES = {
    style.id: style
    for style in (
        DiveStyle("fly_out", "Fly out", fly_out),
        DiveStyle("vortex", "Vortex", vortex),
        DiveStyle("ripple", "Ripple", ripple),
        DiveStyle("fade", "Fade away", fade),
        DiveStyle("fall", "Fall away", fall),
        DiveStyle("flip", "Card flip", flip),
        DiveStyle("wind", "Wind", wind),
    )
}
CHOICES = [(style.id, style.name) for style in STYLES.values()] + [(RANDOM, "Random")]


def dive_seed(index: int, slot: int, count: int) -> int:
    """A dive's seed: the same dive (tile index of a level of count tiles, showing library
    photo slot) always dives the same way, as a loop's replay must."""
    return int(np.random.SeedSequence([index, slot, count]).generate_state(1)[0])


def pick_style(style_id: str, seed: int) -> DiveStyle:
    """The style a dive uses: style_id's, or for Random one picked by seed (unknown ids:
    the first)."""
    if style_id == RANDOM:
        styles = list(STYLES.values())
        return styles[int(np.random.default_rng([seed, 1]).integers(len(styles)))]
    return STYLES.get(style_id, next(iter(STYLES.values())))

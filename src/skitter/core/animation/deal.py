"""Built-in choreography: tiles dealt like cards from decks on the table."""

import math
from dataclasses import dataclass

import numpy as np

from skitter.core.animation.assemble import ORDERS, RANDOM_ORDERS, landing_ranks, order_key
from skitter.core.animation.base import Choreography, TileFrame, Timeline, register_choreography
from skitter.core.animation.landing import landing_times
from skitter.core.animation.look import AnimationLook, scene_extent
from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    IntParam,
    RangeParam,
)

EDGES = {"bottom": (0, 1), "top": (0, -1), "left": (-1, 0), "right": (1, 0)}
CORNERS = {"bottom_left": (-1, 1), "bottom_right": (1, 1), "top_left": (-1, -1),
           "top_right": (1, -1)}  # fmt: skip
POSITIONS = (
    ("bottom", "Bottom"),
    ("top", "Top"),
    ("left", "Left"),
    ("right", "Right"),
    ("bottom_left", "Bottom left"),
    ("bottom_right", "Bottom right"),
    ("top_left", "Top left"),
    ("top_right", "Top right"),
    ("center", "Center"),
    ("orbit", "Orbit"),
)
DEAL_ORDERS = (("near", "Nearest first"), ("far", "Farthest first"), *ORDERS)
DECK_HEIGHT = 1.0  # a full deck stands at most this many tile sizes tall
CARD = 0.01  # a card's thickness, in tile sizes (decks of up to 100 cards)
FLIP_TIME = 0.3  # face down: seconds a card takes to turn face up as it leaves the deck
FLIP_SHARE = 0.4  # ... at most this share of its slide
SETTLE_PASSES = 6  # orbiting decks: refinements of which deck deals each tile, and when


@dataclass(frozen=True)
class DeckPath:
    """Where each deck is, and how it is turned, at any time.

    Fixed decks stay at `spots`, turned `angle`. Orbiting ones (`middle` set)
    circle `middle` at `radius`: deck d is at `start[d] + rate t` radians,
    clockwise from straight up, and turns with its travel (as the moon does,
    its bottom always toward the middle), plus `angle`.
    """

    spots: np.ndarray  # (decks, 2) where the decks are at t = 0
    angle: float  # radians clockwise
    middle: np.ndarray | None = None  # orbit: (2,) its center
    radius: float = 0.0
    start: np.ndarray | None = None  # (decks,) radians clockwise from up, at t = 0
    rate: float = 0.0  # radians per second, + clockwise

    @property
    def orbit(self) -> bool:
        return self.middle is not None

    def at(self, deck, t) -> tuple[np.ndarray, np.ndarray]:
        """(centers (..., 2), rotations (...)) of these decks at these times (broadcast)."""
        deck, t = np.broadcast_arrays(np.asarray(deck), np.asarray(t, np.float64))
        if not self.orbit:
            return self.spots[deck], np.full(deck.shape, self.angle)
        a = self.start[deck] + self.rate * t
        way = np.stack([np.sin(a), -np.cos(a)], axis=-1)
        return self.middle + self.radius * way, self.angle + a

    def all_at(self, t) -> np.ndarray:
        """(M, decks, 2) every deck's center at each of these M times."""
        t = np.asarray(t, np.float64)
        center, _ = self.at(np.arange(len(self.spots))[None, :], t[:, None])
        return center


class DealTimeline(Timeline):
    """Tiles dealt off the top of decks, sliding across the table to their places.

    Tile i lies in deck[i], which moves along `path` (a DeckPath), offset by
    local[i] in the deck's own frame and turned skew[i] more than it (a deck
    not quite square), at deck_height[i] (its place in the stack; heights
    keep a deck drawing top card last). At depart[i] it slides off from
    where it is then (deck_center[i], turned deck_rotation[i]), decelerating evenly (like a card
    on felt: s = 1 - (1 - p)², p its share of the slide) over slide[i]
    seconds, turning to its final rotation (by about `turn` radians). It keeps
    its height until clear of its deck (a tile diagonal away), so it never
    passes under the cards it lay on, then comes down onto the table. Face
    down (face_down), a card shows its back in the deck and is turned face
    up as it leaves, as a dealer does: about its own `flip_axis` over the
    first FLIP_TIME seconds of its slide (at most FLIP_SHARE of it), lifting
    by half its size at mid-turn, then slides face up. It is at rest once in place.
    """

    def __init__(self, scene: MosaicScene, path: DeckPath, deck, local, skew, deck_height,
                 depart, slide, turn, face_down: bool = False, flip_axis=None):  # fmt: skip
        n = len(scene)
        self.final = TileFrame.final(scene)
        self.path = path
        self.deck = np.asarray(deck, np.int64)
        self.local = np.asarray(local, np.float64).reshape(n, 2)
        self.skew = np.asarray(skew, np.float64)
        self.deck_height = np.asarray(deck_height, np.float64)
        self.depart = np.asarray(depart, np.float64)
        self.deck_center, self.deck_rotation = self.in_deck(self.depart)
        self.slide = np.maximum(np.asarray(slide, np.float64), 1e-6)
        # To the final rotation, turning by as near `turn` as ends on it (within half a
        # turn, its way): a tile never snaps to its angle on arrival.
        delta = (self.final.rotation - self.deck_rotation + math.pi) % math.tau - math.pi
        laps = np.round((np.asarray(turn, np.float64) - delta) / math.tau)
        self.end_rotation = self.deck_rotation + delta + laps * math.tau
        self.face_down = bool(face_down)
        self.flip_axis = np.zeros(n, np.int64) if flip_axis is None else np.asarray(flip_axis)
        self.flip_time = np.minimum(FLIP_TIME, FLIP_SHARE * self.slide)
        self.duration = float((self.depart + self.slide).max()) if n else 0.0
        way = np.linalg.norm(self.final.center - self.deck_center, axis=1)
        diagonal = np.hypot(self.final.size[:, 0], self.final.size[:, 1])
        cleared = np.minimum(diagonal / np.maximum(way, 1e-12), 1.0)  # share of the way
        self.clear = 1.0 - np.sqrt(1.0 - cleared)  # share of the slide, on top of the deck

    def in_deck(self, t) -> tuple[np.ndarray, np.ndarray]:
        """(centers, rotations) every tile has while in its deck at time t (one time,
        or one per tile)."""
        center, rotation = self.path.at(self.deck, t)
        c, s = np.cos(rotation)[:, None], np.sin(rotation)[:, None]
        x, y = self.local[:, :1], self.local[:, 1:]
        return center + np.hstack([c * x - s * y, s * x + c * y]), rotation + self.skew

    def frame(self, t: float) -> TileFrame:
        t = min(max(float(t), 0.0), self.duration)
        if t >= self.duration:
            return self.final  # exactly, whatever the rounding of the times
        b = self.final
        p = np.clip((t - self.depart) / self.slide, 0.0, 1.0)
        e = 1.0 - (1.0 - p) ** 2  # evenly slowing down
        placed = p >= 1.0
        center = self.deck_center + (b.center - self.deck_center) * e[:, None]
        rotation = self.deck_rotation + (self.end_rotation - self.deck_rotation) * e
        waiting = t < self.depart
        if self.path.orbit and waiting.any():  # still in a moving deck: going with it
            here, turned = self.in_deck(t)
            center = np.where(waiting[:, None], here, center)
            rotation = np.where(waiting, turned, rotation)
        down = np.clip((p - self.clear) / np.maximum(1.0 - self.clear, 1e-12), 0.0, 1.0)
        height = self.deck_height * (1.0 - down**2)  # clear of the deck, down onto the table
        size, facing = b.size.copy(), None
        if self.face_down:
            f = np.clip((t - self.depart) / self.flip_time, 0.0, 1.0)  # 0: back up, 1: up
            facing = -np.cos(math.pi * f)
            rows = np.arange(len(p))
            along = size[rows, 1 - self.flip_axis].copy()  # the side that foreshortens
            size[rows, 1 - self.flip_axis] *= np.maximum(np.abs(facing), 0.02)
            height = height + 0.5 * along * np.sin(math.pi * f)
            facing = np.where(placed, 1.0, facing)
        height = np.where(placed, 0.0, height)
        # At rest in stacking order, then everything else from low to high (decks
        # draw bottom card first; ties in stacking order).
        order = np.lexsort((np.arange(len(p)), np.where(placed, 0.0, height), ~placed))
        return TileFrame(
            center=np.where(placed[:, None], b.center, center),
            size=size,
            rotation=np.where(placed, b.rotation, rotation),
            alpha=np.ones(len(p)),
            tint=b.tint.copy(),
            order=None if placed.all() else order,
            height=height,
            rest=None if placed.all() else placed,
            facing=facing,
        )


@register_choreography
class DealChoreography(Choreography):
    id = "deal"
    name = "Deal"
    description = (
        "Tiles are dealt like cards from one or more decks on the table: each slides "
        "off the top of its deck and across the table to its place, slowing to a stop. "
        "Overlapping tiles always land bottom first."
    )

    position = ChoiceParam(
        "bottom", "Deck position", choices=POSITIONS,
        help="Where the deck sits, beside the mosaic (or in its middle). With several "
             "decks, where the first one sits. Orbit: the decks circle the mosaic as "
             "they deal.",
    )  # fmt: skip
    offset = FloatParam(
        0.1, "Offset", min=-0.5, max=3.0, step=0.05, suffix=" × mosaic",
        when=lambda c: c.position not in ("center", "orbit"),
        help="How far out from the mosaic's edge the deck sits (0: on the edge; "
             "less: over the mosaic). Decks off the picture deal tiles in from its edge.",
    )  # fmt: skip
    radius = FloatParam(
        0.75, "Radius", min=0.0, max=5.0, step=0.05, suffix=" × mosaic",
        when=lambda c: c.position == "orbit",
        help="How far from the mosaic's center the decks circle, in sizes of the mosaic "
             "(its longer side). Below 0.5 they pass over the mosaic.",
    )  # fmt: skip
    start_angle = FloatParam(
        180.0, "Starting angle", min=0.0, max=360.0, step=15.0, decimals=0, suffix="°",
        when=lambda c: c.position == "orbit",
        help="Where on the orbit the first deck starts: 0° above the mosaic, clockwise "
             "(180°: below it).",
    )  # fmt: skip
    orbit_turns = FloatParam(
        1.0, "Orbit speed", min=0.0, max=20.0, step=0.25, suffix=" turns",
        when=lambda c: c.position == "orbit",
        help="How many times the decks go round over the whole animation (0: they stay "
             "put on the orbit).",
    )  # fmt: skip
    orbit_direction = ChoiceParam(
        "clockwise", "Orbit direction",
        choices=[("clockwise", "Clockwise"), ("counter", "Counter-clockwise")],
        when=lambda c: c.position == "orbit",
    )  # fmt: skip
    decks = IntParam(
        1, "Decks", min=1, max=8,
        help="How many decks deal at once; each deals the tiles nearest to it. Orbiting "
             "decks are spaced evenly around the orbit.",
    )  # fmt: skip
    arrangement = ChoiceParam(
        "around", "Arrangement",
        choices=[("around", "Around the mosaic"), ("side", "Side by side")],
        available=lambda c, value: value == "around" or c.position not in CORNERS,
        when=lambda c: c.decks > 1 and c.position != "orbit",
        help="Around: spread evenly around the mosaic, starting at the deck position. "
             "Side by side: spread evenly along that edge (or across the middle).",
    )  # fmt: skip
    deck_angle = FloatParam(
        0.0, "Deck angle", min=-180.0, max=180.0, step=15.0, decimals=0, suffix="°",
        help="How the decks are turned on the table, clockwise. Orbiting decks turn "
             "as they go round (at 0°, upright at the top, bottom toward the middle).",
    )  # fmt: skip
    neatness = FloatParam(
        0.8, "Neatness", min=0.0, max=1.0, step=0.05,
        help="How squarely the cards are stacked in a deck (1: perfectly; less: a few "
             "cards stick out, slightly turned).",
    )  # fmt: skip
    order = ChoiceParam("near", "Order", choices=DEAL_ORDERS, help="Which tiles land first.")
    spread = FloatParam(
        0.15, "Spread", min=0.0, max=1.0, step=0.05,
        when=lambda c: c.order not in RANDOM_ORDERS,
        help="Randomness in the landing order, as a share of the whole sequence.",
    )  # fmt: skip
    duration = FloatParam(
        8.0, "Duration", min=0.5, max=600.0, step=0.5, decimals=1, suffix=" s",
        help="From the first tile leaving its deck to the last coming to rest.",
    )  # fmt: skip
    slide = FloatParam(
        1.0, "Slide time", min=0.05, max=60.0, step=0.1, decimals=2, suffix=" s",
        help="How long the farthest tile slides (at most the duration); nearer ones "
             "take less, as if all were pushed off just as hard.",
    )  # fmt: skip
    wind_down = FloatParam(
        0.15, "Wind-down", min=0.0, max=0.5, step=0.05, suffix=" × duration",
        help="Landings slow down over this last share of the duration, so the last "
             "few tiles land one by one instead of all in the same moment (0: an even "
             "pace to the end).",
    )  # fmt: skip
    last_gap = FloatParam(
        0.25, "Last gap", min=0.0, max=5.0, step=0.05, suffix=" s",
        when=lambda c: c.wind_down > 0,
        help="Time between the last two landings (at most half the wind-down).",
    )  # fmt: skip
    spin = RangeParam(
        (0.0, 0.5), "Spin", min=0.0, max=10.0, step=0.25, suffix=" turns",
        help="Extra turns each tile makes on its way, either way: each tile draws its "
             "own amount from this range (to within half a turn, so it comes to rest "
             "at its own angle).",
    )  # fmt: skip
    face_down = BoolParam(
        False, "Face down",
        help="Cards lie face down in the deck, showing their backs, and are turned "
             "face up as they leave it, then slide face up.",
    )  # fmt: skip
    seed = IntParam(1, "Seed", min=0, max=999_999)

    def timeline(self, scene: MosaicScene, look: AnimationLook | None = None) -> Timeline:
        n = len(scene)
        rng = np.random.default_rng(self.seed)
        params = type(self)
        tile = float(np.median(scene.size.max(axis=1))) if n else 1.0
        path = self.deck_path(scene)

        if self.order in RANDOM_ORDERS:
            key = None
        elif self.order in ("near", "far"):
            key = self._reach_key(scene, path)
        else:
            key = order_key(scene, self.order, rng)
        rank = landing_ranks(scene, key, self.spread, self.seed, rng)

        # In the deck: a little out of square, as neatness allows.
        mess = 1.0 - self.neatness
        local = rng.uniform(-1.0, 1.0, (n, 2)) * 0.15 * mess * tile
        skew = rng.uniform(-1.0, 1.0, n) * math.radians(10.0) * mess
        slide_time = min(self.slide, self.duration)

        # Each tile is dealt by the deck nearest it when it leaves. Where a deck is
        # depends on when, and when a tile leaves on how far it slides: orbiting decks
        # settle that over a few passes (fixed ones need one).
        depart = np.zeros(n)
        for _ in range(SETTLE_PASSES if path.orbit else 1):
            spots = path.all_at(depart) if n else np.zeros((0, len(path.spots), 2))
            deck = np.argmin(np.linalg.norm(scene.center[:, None] - spots, axis=2), axis=1)
            center, rotation = path.at(deck, depart)
            c, s = np.cos(rotation)[:, None], np.sin(rotation)[:, None]
            x, y = local[:, :1], local[:, 1:]
            start = center + np.hstack([c * x - s * y, s * x + c * y])
            distance = np.linalg.norm(scene.center - start, axis=1)
            # Pushed off equally hard, the farthest slides for `slide` seconds:
            # constant deceleration a, distance d = a T² / 2.
            far = float(distance.max()) if n else 0.0
            slide = slide_time * np.sqrt(distance / far) if far > 0 else np.full(n, slide_time)
            slide = np.maximum(slide, 1e-3)
            # Landings paced through the end; the first tile leaves at 0.
            lead = self._lead(n, rank, slide)
            land = lead + landing_times(n, self.duration - lead, self._tail(), self.last_gap)
            land = land[rank]
            depart = np.maximum(land - slide, 0.0)
        slide = land - depart

        # Stacked: the first to leave on top.
        height = np.zeros(n)
        spacing = tile * min(CARD, DECK_HEIGHT / max(int(np.bincount(deck).max()), 1)) if n else 0
        for d in np.unique(deck):
            mine = np.flatnonzero(deck == d)
            stack = mine[np.argsort(-depart[mine], kind="stable")]  # bottom to top
            height[stack] = spacing * np.arange(1, len(stack) + 1)

        turn = (rng.choice([-1.0, 1.0], n)
                * params.spin.draw(self.spin, rng, n) * math.tau)  # fmt: skip
        return DealTimeline(
            scene, path, deck, local, skew, height, depart, slide, turn,
            face_down=self.face_down, flip_axis=rng.integers(0, 2, n),
        )  # fmt: skip

    def _reach_key(self, scene: MosaicScene, path: DeckPath) -> np.ndarray:
        """Nearest (or farthest) first. Fixed decks: by distance to the nearest deck.
        Orbiting ones deal each tile as a deck passes it: by when the next deck
        comes by, near tiles (to the orbit) on the first passes, far ones on later
        ones, so the decks deal inward (outward) round after round."""
        center = scene.center
        if not path.orbit or self.orbit_turns <= 0:
            gaps = np.linalg.norm(center[:, None] - path.spots[None], axis=2).min(axis=1)
            return gaps if self.order == "near" else -gaps
        n, k = len(scene), len(path.spots)
        out = center - path.middle
        angle = np.arctan2(out[:, 0], -out[:, 1])  # clockwise from up
        ahead = (np.sign(path.rate) * (angle - path.start[0])) % (math.tau / k)
        phase = ahead / (math.tau / k)  # 0..1: share of the way to it for the next deck
        passes = self.orbit_turns * k  # decks going by any one place, over the animation
        gap = np.abs(np.hypot(out[:, 0], out[:, 1]) - path.radius)  # to the orbit
        closeness = np.empty(n)
        closeness[np.argsort(gap if self.order == "near" else -gap, kind="stable")] = np.arange(
            n
        ) / max(n, 1)
        lap = np.clip(np.round(closeness * passes - phase), 0, max(math.ceil(passes) - 1, 0))
        return lap + phase

    def _tail(self) -> float:
        return self.wind_down * self.duration

    def _lead(self, n: int, rank: np.ndarray, slide: np.ndarray) -> float:
        """When the first tile lands: the earliest that lets every tile leave its deck
        at 0 or later (landings paced over the rest of the duration)."""
        if n == 0:
            return 0.0

        def fits(lead: float) -> bool:  # every tile leaves at 0 or later
            times = landing_times(n, self.duration - lead, self._tail(), self.last_gap)
            return bool(np.all(lead + times[rank] >= slide - 1e-9))

        lo, hi = 0.0, float(slide.max())
        if fits(lo):
            return lo
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if fits(mid) else (mid, hi)
        return hi

    def deck_path(self, scene: MosaicScene) -> DeckPath:
        """Where the decks are, and how they turn, over the animation."""
        angle = math.radians(self.deck_angle)
        if self.position != "orbit":
            return DeckPath(self.deck_spots(scene), angle)
        x0, y0, x1, y1 = scene.bounds
        middle = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        way = 1.0 if self.orbit_direction == "clockwise" else -1.0
        start = math.radians(self.start_angle) + way * math.tau * np.arange(self.decks) / self.decks
        rate = way * math.tau * self.orbit_turns / self.duration
        radius = self.radius * scene_extent(scene)
        spots = middle + radius * np.stack([np.sin(start), -np.cos(start)], axis=1)
        return DeckPath(spots, angle, middle, radius, start, rate)

    def deck_spots(self, scene: MosaicScene) -> np.ndarray:
        """(decks, 2) where each deck sits on the table (at the start, if orbiting)."""
        if self.position == "orbit":
            return self.deck_path(scene).spots
        x0, y0, x1, y1 = scene.bounds
        middle = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        half = np.array([(x1 - x0) / 2, (y1 - y0) / 2])
        k = self.decks
        if self.position == "center":
            if k == 1:
                return middle[None, :]
            if self.arrangement == "side":
                x = x0 + (x1 - x0) * (np.arange(k) + 0.5) / k
                return np.stack([x, np.full(k, middle[1])], axis=1)
            angle = math.pi / 2 + math.tau * np.arange(k) / k  # from the bottom, around
            return middle + 0.5 * half * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        out = np.maximum(
            half + self.offset * scene_extent(scene), 0.0
        )  # the rectangle decks sit on
        if self.position in EDGES and self.arrangement == "side" and k > 1:
            sx, sy = EDGES[self.position]
            along = (np.arange(k) + 0.5) / k * 2.0 - 1.0  # -1..1 along the edge
            if sx:
                return middle + np.stack([np.full(k, sx * out[0]), along * half[1]], axis=1)
            return middle + np.stack([along * half[0], np.full(k, sy * out[1])], axis=1)
        sx, sy = {**EDGES, **CORNERS}[self.position]
        start = math.atan2(sy * out[1], sx * out[0])
        angle = start + math.tau * np.arange(k) / k
        way = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        # Where each way out from the middle meets the rectangle.
        with np.errstate(divide="ignore"):
            reach = np.minimum(out[0] / np.abs(way[:, 0]), out[1] / np.abs(way[:, 1]))
        return middle + reach[:, None] * way

    def summary(self) -> str:
        decks = f"{self.decks} decks" if self.decks > 1 else "1 deck"
        if self.position == "orbit":
            decks += " orbiting"
        return f"{dict(DEAL_ORDERS)[self.order]}, {decks}, {self.duration:g} s"

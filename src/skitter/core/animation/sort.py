"""Built-in build: a sorting algorithm puts a scrambled mosaic in order.

The spots are lined up along a path, the scrambled photos form an unsorted
array, and the chosen algorithm sorts it round by round (sorting.py); each
round, every photo whose position changed glides to its new spot. This is a
sorting visualizer with photos for bars, not physics: photos glide flat over
everything, taking on the shape of the spot they go to, and moving photos
can flash a highlight color (as visualizers mark the elements they touch).

Paced like a visualizer, by step: every round (an algorithm's step) takes
`step_time`, its moves most of that, so the build lasts as long as the
algorithm needs (a few seconds for Radix, many minutes for Bubble sort).
Every step is shown as the algorithm made it. Rounds are stored as each
photo's spot after each (`states`): only beyond STORED_SPOTS in all are
steps grouped, to keep memory bounded.

Three motions: photos glide to their new spots, the spots flip, or (none)
photos jump to their new spots as each step begins, then hold. Flipping,
a spot turns over in place like a card: its photo narrows to edge-on, and
there (unseen) gives way to the photo coming to that spot, which widens back
face up. Nothing crosses the mosaic, which suits algorithms moving many
photos far at once (Pancake, Radix, Merge): gliding, those cross
in a swarm, and a reversed stretch bunches up in its middle.
"""

import math

import numpy as np

from skitter.core.animation.base import (
    Choreography,
    Flips,
    TileFrame,
    Timeline,
    register_choreography,
)
from skitter.core.animation.look import AnimationLook
from skitter.core.animation.sorting import path_order, sort_states
from skitter.core.easing import ease_in_out_cubic
from skitter.core.scene import MosaicScene
from skitter.core.slicing.common import seed_param
from skitter.core.slicing.params import BoolParam, ChoiceParam, ColorParam, FloatParam

ALGORITHMS = (
    ("bitonic", "Bitonic network"), ("odd_even", "Odd-even (parallel bubble)"),
    ("merge", "Merge sort"), ("quick", "Quicksort"), ("radix", "Radix sort (MSD)"),
    ("heap", "Heap sort"), ("selection", "Selection sort"), ("shear", "Shear sort (2D)"),
    ("insertion", "Insertion sort"), ("bubble", "Bubble sort"),
    ("cocktail", "Cocktail shaker sort"), ("cant_believe", "I can't believe it can sort"),
    ("pancake", "Pancake sort"),
)  # fmt: skip
MOTIONS = (("glide", "Glide"), ("flip", "Flip"), ("none", "None"))
PATHS = (("reading", "Reading order"), ("snake", "Snake"), ("hilbert", "Hilbert curve"),
         ("spiral", "Spiral"))  # fmt: skip
STORED_SPOTS = 20_000_000  # rounds x photos kept at most (int32: 80 MB)
MOVE_SHARE = 0.85  # share of a step its moves take (the rest: still)
HOVER = 0.01  # moving photos' height, in base tiles: above the others, no visible shadow
HIGHLIGHT = 0.45  # strongest highlight tint


class SortTimeline(Timeline):
    """Photos moving between spots, round by round: `states` (S, N) the spot of each
    photo before the first round and after each; round r starts at r x `step` and its
    moves take MOVE_SHARE of it. `highlight` (rgb or None) tints moving photos.
    `motion`: "glide", "flip" (spots turn over in place, trading photos edge-on) or
    "none" (photos jump as each step begins; see the module docstring)."""

    def __init__(self, scene: MosaicScene, states: np.ndarray, step: float,
                 highlight: tuple[float, float, float] | None = None,
                 motion: str = "glide"):  # fmt: skip
        self.final = TileFrame.final(scene)
        self.states = np.asarray(states, np.int32)
        self.interval = max(float(step), 1e-6)
        self.travel = MOVE_SHARE * self.interval
        rounds = max(len(self.states) - 1, 0)
        # The last step's moves end the build (nothing to sort: a still step).
        self.duration = (rounds - 1) * self.interval + self.travel if rounds else self.interval
        self.highlight = None if highlight is None else np.asarray(highlight, np.float64)
        self.hover = HOVER * scene.tile_size[0]
        self.motion = motion

    @property
    def rounds(self) -> int:
        return len(self.states) - 1

    def frame(self, t: float) -> TileFrame:
        t = float(t)
        if t >= self.duration or not self.rounds:
            return self.final  # exactly, whatever the rounding of the times
        t = max(t, 0.0)
        b = self.final
        r = min(int(t / self.interval), self.rounds - 1)
        p = min(max((t - r * self.interval) / self.travel, 0.0), 1.0)
        here, there = self.states[r], self.states[r + 1]
        if self.motion == "none":  # jumped as the step began (t = its start: not yet)
            begun = t > r * self.interval
            place = there if begun else here
            tint = b.tint
            if self.highlight is not None and begun:  # what jumped flashes, fading
                tint = b.tint.copy()
                jumped = here != there
                tint[jumped, :3] = self.highlight
                tint[jumped, 3] = HIGHLIGHT * (1.0 - p)
            return b.replace(center=b.center[place], size=b.size[place],
                             rotation=b.rotation[place], tint=tint,
                             order=np.argsort(place, kind="stable"))  # fmt: skip
        moving = (here != there) & (0.0 < p < 1.0)
        place = there if p >= 1.0 else here  # where each photo lies (or is leaving)
        if not moving.any():
            return b.replace(center=b.center[place], size=b.size[place],
                             rotation=b.rotation[place],
                             order=np.argsort(place, kind="stable"))  # fmt: skip
        e = ease_in_out_cubic(p)
        if self.motion == "flip":  # the spot turns over: edge-on halfway, where the photos trade
            place = np.where(moving & (e >= 0.5), there, here)
            facing = np.where(moving, abs(math.cos(math.pi * e)), 1.0)
            size = b.size[place].copy()
            size[:, 0] *= np.maximum(facing, Flips.THINNEST)  # narrowing across
            return b.replace(center=b.center[place], size=size, rotation=b.rotation[place],
                             order=np.argsort(place, kind="stable"), facing=facing)  # fmt: skip
        center = b.center[here] + (b.center[there] - b.center[here]) * e
        size = b.size[here] + (b.size[there] - b.size[here]) * e
        turn = (b.rotation[there] - b.rotation[here] + math.pi) % math.tau - math.pi
        rotation = b.rotation[here] + turn * e
        height = np.where(moving, self.hover, 0.0)
        resting, flying = np.flatnonzero(~moving), np.flatnonzero(moving)
        order = np.concatenate([resting[np.argsort(place[resting], kind="stable")], flying])
        tint = b.tint
        if self.highlight is not None:
            tint = b.tint.copy()
            tint[moving, :3] = self.highlight
            tint[moving, 3] = HIGHLIGHT * math.sin(math.pi * p)
        return b.replace(center=center, size=size, rotation=rotation, tint=tint, order=order,
                         height=height, rest=~moving)  # fmt: skip


@register_choreography
class SortChoreography(Choreography):
    id = "sort"
    name = "Sort"
    phase = "build"
    photos_take_spots = True
    description = (
        "A sorting algorithm puts the scrambled mosaic in order, round by round, like a "
        "sorting visualizer with photos for bars: the spots lined up along a path, each "
        "photo's home its place in the sorted order."
    )

    algorithm = ChoiceParam(
        "bitonic", "Algorithm", choices=ALGORITHMS,
        help="How the photos get sorted. Bitonic: stages of swaps all at once, in "
             "butterfly patterns. Odd-even: neighbors swap, the picture seeping into "
             "place. Merge: sorted runs double each pass. Quicksort: stretches split "
             "around pivots, level by level. Radix: photos find their quarter, then "
             "sixteenth... Heap: a heap, then the picture fills in from the end. "
             "Selection: one photo after another. Shear: rows, then columns, on a grid. "
             "The slow ones, for their patterns: Insertion: photos slide back into a "
             "sorted start. Bubble: the sorted end grows backwards, a pass at a time. "
             "Cocktail shaker: both ends at once. I can't believe it can sort: it looks "
             "backwards, and works. Pancake: only flipping the start over (about two "
             "flips per photo: best on small mosaics, with Motion: Flip).",
    )  # fmt: skip
    motion = ChoiceParam(
        "glide", "Motion", choices=MOTIONS,
        help="Glide: photos slide to their new spots (best when moves are short, as in "
             "Odd-even, Bubble or Insertion). Flip: each spot turns over in place, like "
             "a card, and comes back showing its new photo; nothing crosses the mosaic "
             "(best when many photos move far, as in Pancake, Radix or Merge). "
             "Flipping, the turning shade marks what moves. None: photos "
             "jump to their new spots as each step begins (with Highlight, what jumped "
             "flashes and fades over the step).",
    )  # fmt: skip
    path = ChoiceParam(
        "reading", "Path", choices=PATHS,
        help="The order the sorted mosaic reads in, which shapes how the picture "
             "appears: reading order (rows, left to right), snake (rows, every other "
             "one backwards: moves stay short), Hilbert curve (nearby in the order is "
             "nearby on the picture: blocks form), or spiral (from the middle out).",
    )  # fmt: skip
    step_time = FloatParam(
        0.25, "Step time", min=0.02, max=30.0, step=0.05, decimals=2, suffix=" s",
        help="How long each step of the algorithm takes on screen. Every step is shown, "
             "so the build lasts as long as the algorithm needs: seconds for Radix or "
             "Merge, many minutes for Bubble or Insertion on a big mosaic (the timeline "
             "shows how long).",
    )  # fmt: skip
    scrambled = FloatParam(
        100.0, "Scrambled", min=0.0, max=100.0, step=5.0, decimals=0, suffix="%",
        help="Share of the photos that start out of place (shuffled among themselves).",
    )  # fmt: skip
    highlight = BoolParam(
        True, "Highlight",
        help="Moving photos flash a color, as sorting visualizers mark what they touch.",
    )  # fmt: skip
    highlight_color = ColorParam(
        "#ffb020", "Highlight color", under="highlight", when=lambda c: c.highlight,
        help="The color moving photos flash.",
    )  # fmt: skip
    seed = seed_param()

    def timeline(self, scene: MosaicScene, look: AnimationLook | None = None) -> Timeline:
        n = len(scene)
        rng = np.random.default_rng(self.seed)
        path = path_order(scene, self.path)
        values = np.arange(n)  # position -> value; sorted: every photo home
        chosen = rng.choice(n, size=int(round(n * self.scrambled / 100.0)), replace=False)
        values[chosen] = values[rng.permutation(chosen)]
        columns = max(1, round((scene.bounds[2] - scene.bounds[0]) / scene.tile_size[0]))
        if (values == np.arange(n)).all():  # nothing out of place: nothing to do
            arrays = values[None]
        else:
            arrays, _ = sort_states(
                values,
                self.algorithm,
                self.seed,
                max(1, STORED_SPOTS // max(n, 1)),
                {"columns": columns},
            )
        # Each state: the photo with value v (home spot path[v]) lies at spot path[p].
        states = np.empty((len(arrays), n), np.int32)
        for s, a in enumerate(arrays):
            states[s, path[a]] = path
        highlight = ColorParam.rgb(self.highlight_color) if self.highlight else None
        return SortTimeline(scene, states, self.step_time, highlight,
                            self.motion)  # fmt: skip

"""The Infinite Mosaic's motion as pure functions: what the live window shows, and what a
replay or a video renders from a record, from the same math (no Qt here).

In steps, a dive at a time:
- `glide`: the view moving a point on screen to the middle while zooming geometrically
  (even zoom speed), eased: auto-dive's approach onto a tile (`approach_zoom`), and a
  dive's camera.
- `DiveMotion`: a dive into a tile: the other tiles leave the view (its style), the tile
  straightens, loses its tint and fills the view, its photo fading in over it
  (PHOTO_IN of the dive). After it, the photo fades into the level's mosaic over Fade
  time (`fade_out`).

Continuously (core/dive_path.py's FlowPath moves the view):
- `flow_shares`, `flow_fades`: as the tile grows its photo fades in over it, then into
  its own mosaic drawn inside it, each over FADE_ZOOM of zoom, done when the tile fills
  Mosaic appears at of the view.
- `last_fade`: after the switch, the last level fades out of the new one's gaps over
  LAST_OUT_ZOOM, once the new one covers the view.
- `moved`: sprites moved from one level's world into the next one's.

`DiveTimeline` puts the steps ones together for a record (core/infinite.py's
DiveRecord): its dives as a pure function of time, for replaying it and rendering it
as a video. Continuous legs aren't on it yet.
"""

import math
from dataclasses import dataclass

import numpy as np

from skitter.core.dive_path import FlowPath
from skitter.core.dive_styles import DiveStyle, DiveTiles, dive_seed, pick_style
from skitter.core.easing import ease_in_out_cubic

PHOTO_IN = 0.2  # share of a dive over which the photo fades in over its tile
FADE_ZOOM = 1.8  # zoom each of a leg's fades takes (photo in, then into its mosaic)
LAST_OUT_ZOOM = 1.6  # zoom the last level takes to fade out of the new one's gaps


def fade(x: float) -> float:
    """0 to 1, eased, for x from 0 to 1 (clamped)."""
    return float(ease_in_out_cubic(min(max(x, 0.0), 1.0)))


def glide(center, screen0, zoom0: float, zoom1: float, m: float) -> tuple[np.ndarray, float]:
    """(view center, zoom) at m (0 to 1, eased) of a move bringing the world point center
    from screen0 (its offset on screen from the middle, px) to the middle, as the zoom goes
    geometrically from zoom0 to zoom1."""
    zoom = zoom0 * (zoom1 / zoom0) ** m
    return np.asarray(center, np.float64) - np.asarray(screen0) * (1 - m) / zoom, zoom


def approach_zoom(view_width: float, tile_width: float, tiles_across: float,
                  zoom0: float) -> float:  # fmt: skip
    """The zoom where tiles_across base tiles (tile_width units) span a view view_width px
    wide: auto-dive's approach zooms in to it (never out from zoom0)."""
    return max(view_width / (tiles_across * tile_width), zoom0)


@dataclass(frozen=True)
class DiveView:
    """What a dive frame shows besides its tiles."""

    center: np.ndarray  # the view's
    zoom: float
    photo_rotation: float  # the photo, over the tile dived into
    photo_alpha: float


class DiveMotion:
    """A dive into tile index of a level's tiles (sprite instances, as they are when it
    starts) from a view (center, zoom; viewport px; down: the screen's down in the world),
    onto the tile filling it at end_zoom."""

    def __init__(self, instances: np.ndarray, index: int, center, zoom: float, viewport,
                 down, end_zoom: float, style: DiveStyle, seed: int):  # fmt: skip
        self.index = index
        self.pos = instances["pos"].astype(np.float64)
        self.center = self.pos[index].copy()
        self.rotation = float(instances["rotation"][index])
        self.offset = instances["offset"][index].copy()
        self.size = instances["size"][index].copy()
        self.zoom0, self.zoom1 = float(zoom), float(end_zoom)
        self.screen0 = (self.center - np.asarray(center, np.float64)) * self.zoom0
        self.style = style  # how the other tiles leave
        viewport = np.asarray(viewport, np.float64)
        self.tiles = DiveTiles(self.pos, instances["size"], instances["rotation"],
                               instances["alpha"], index, viewport / self.zoom0,
                               viewport / self.zoom1, seed, down)  # fmt: skip

    def frame(self, e: float, tiles: np.ndarray) -> DiveView:
        """The dive at e (0 to 1): the tiles' instances written as they are then (in
        place), and the rest of what shows."""
        m = float(ease_in_out_cubic(e))
        frame = self.style.frame(self.tiles, e)
        tiles["pos"], tiles["size"] = frame.pos, frame.size
        tiles["rotation"], tiles["alpha"] = frame.rotation, frame.alpha
        if frame.shade is not None:
            tiles["tint"][:, :3] = 0.0
            tiles["tint"][:, 3] = frame.shade
            tiles["tint"][self.index] = 0.0
        i = self.index
        tiles["pos"][i], tiles["size"][i] = self.center, self.size
        tiles["alpha"][i] = self.tiles.alpha[i]
        tiles["rotation"][i] = self.rotation * (1 - m)
        tiles["offset"][i] = self.offset * (1 - m)
        # The tile moves on screen from where it was to the middle as the view zooms.
        center, zoom = glide(self.center, self.screen0, self.zoom0, self.zoom1, m)
        return DiveView(center, zoom, self.rotation * (1 - m), min(e / PHOTO_IN, 1.0))


def fade_out(t: float, seconds: float) -> float:
    """The photo's opacity t seconds into its fade into the mosaic (after a dive)."""
    return 1.0 - (1.0 if seconds <= 0 else min(t / seconds, 1.0))


def flow_shares(path: FlowPath, mosaic_at: float) -> tuple[float, float]:
    """The tile's shares of the view where its photo starts to fade in, and where its
    mosaic has taken over (mosaic_at: Mosaic appears at, 0 to 1). A tile already larger
    than the first at the leg's start (few columns, a large tile) starts there, its mosaic
    later."""
    first = mosaic_at / FADE_ZOOM**2
    start = path.tile_share(0.0)
    if start > first:
        first, mosaic_at = start, min(max(mosaic_at, start * FADE_ZOOM**2), 1.0)
        first = min(first, mosaic_at / 1.01)  # some zoom for the fades, however little
    return first, mosaic_at


def flow_fades(path: FlowPath, u: float, mosaic_at: float) -> tuple[float, float]:
    """(photo in, mosaic in), each 0 to 1, at u of the leg: the tile's photo fading in
    over it, then into its mosaic."""
    first, done = flow_shares(path, mosaic_at)
    share = math.log(path.tile_share(u))
    step = math.log(done / first) / 2  # each fade's zoom (FADE_ZOOM, if there's room)
    photo_in = fade((share - math.log(first)) / step)
    return photo_in, fade((share - math.log(first) - step) / step)


def last_fade(zoom_ratio: float, seconds: float, rate: float) -> float:
    """How far the last level has faded out (0 to 1 and on: gone) once the new one covers
    the view: by the zoom since (zoom_ratio), or as if it had zoomed on at rate (log zoom
    per second) for seconds if slower (by hand the creep may stop)."""
    zoomed = max(math.log(max(zoom_ratio, 1.0)), rate * seconds)
    return zoomed / math.log(LAST_OUT_ZOOM)


def moved(instances: np.ndarray, origin, to, scale: float, turn: float) -> np.ndarray:
    """Sprite instances moved from one world to another: what was at origin goes to `to`,
    scaled and turned (radians) about it."""
    out = instances.copy()
    c, s = math.cos(turn), math.sin(turn)
    offset = (instances["pos"].astype(np.float64) - origin) * scale
    out["pos"] = np.asarray(to) + np.stack(
        [c * offset[:, 0] - s * offset[:, 1], s * offset[:, 0] + c * offset[:, 1]], axis=1
    )
    out["size"] = instances["size"] * scale
    out["rotation"] = instances["rotation"] + turn
    return out


# A record's dives as a timeline (steps)

PHASES = ("fade", "dwell", "approach", "pause", "dive")  # each level's, in order


@dataclass(frozen=True)
class TimelineStep:
    """One level of a timeline: shown, then (pick >= 0) dived from into tile pick."""

    position: int  # its index in the record
    pick: int  # the tile dived into from it (-1: none, the timeline ends on it at rest)
    next: int  # the record index that dive leads to (-1: none)
    start: float  # seconds into the timeline


@dataclass(frozen=True)
class Photo:
    """The photo sprite: a level's picture (record index), where it lies, how opaque."""

    level: int
    pos: np.ndarray
    size: np.ndarray
    rotation: float
    alpha: float


@dataclass(frozen=True)
class TimelineFrame:
    """What a timeline shows at a moment: the tiles of the record's level `position`
    (their motion fields: pos, size, rotation, alpha, offset, tint; the rest are the
    tile layer's own) and the view on them, in that level's world; the photo; and the
    outline on tile `highlight` (-1: none)."""

    position: int
    phase: str  # one of PHASES
    tiles: dict
    center: np.ndarray
    zoom: float
    photo: Photo | None = None
    highlight: int = -1
    highlight_alpha: float = 0.0

    @property
    def diving(self) -> bool:
        """Tiles grow far past their texels: views sample them smoothly then."""
        return self.phase == "dive"


def rest_tiles(scene) -> dict:
    """The motion fields of a level's tiles at rest (as its tile layer shows them)."""
    n = len(scene)
    return {"pos": scene.center.astype(np.float64), "size": scene.size.astype(np.float64),
            "rotation": scene.rotation.astype(np.float64), "alpha": np.ones(n),
            "offset": np.array(scene.tint_shift, np.float64), "tint": np.zeros((n, 4))}  # fmt: skip


class DiveTimeline:
    """A record's dives (core/infinite.py's DiveRecord) as a pure function of time, in
    steps, as the live window plays them and from the same functions above: each level
    fades in from its photo (Fade time), dwells, the view approaches the tile picked
    from it (Zoom-in time, to Zoom in to tiles across; outlined with Highlight), pauses
    on it and dives (Dive time, Tiles leave by), arriving at the next level.

    The whole record from its first level, or span (first, last): its levels from first
    to last (ending at rest on last, or, a closed loop's way back from it in the span,
    once that dive has landed); or with loop, one pass of its closed loop:
    frame(duration) is then frame(0), for seamless video. With repeat, span's dives and
    then its closed loop's on and on (time has no end: past duration it goes round from
    the loop's first level). viewport: the view's pixels.
    """

    def __init__(self, record, settings, viewport, loop: bool = False, span=None,
                 repeat: bool = False):  # fmt: skip
        self.settings = settings
        self.viewport = np.asarray(viewport, np.float64)
        self.loop = loop or repeat
        if self.loop and not record.closed:
            raise ValueError("only a closed loop has a pass to play")
        if loop:
            first, last = record.loop_start, len(record.levels) - 1
        else:
            first, last = span if span is not None else (0, len(record.levels) - 1)
        steps, start = [], 0.0
        for p in range(first, last + 1):
            pick = record.picks[p] if p < len(record.picks) else -1
            then = record.leads_to(p, pick) if pick >= 0 else -1
            if not first <= then <= last:  # it leads out of the span: it ends at rest here
                pick, then = -1, -1
            steps.append(TimelineStep(p, pick, then, start))
            start += self._seconds(pick >= 0)
        if not self.loop and steps and steps[-1].next >= 0:  # a closed loop's way back, at rest
            steps.append(TimelineStep(steps[-1].next, -1, -1, start))
            start += self._seconds(False)
        self.steps, self.duration = steps, start
        self.loop_at = 0.0  # when the loop's first level starts (what going round returns to)
        if self.loop:
            self.loop_at = next(s.start for s in steps if s.position == record.loop_start)
        self._dives: dict = {}  # step -> its DiveMotion

    def _phases(self) -> list[float]:
        s = self.settings
        return [float(s.fade_seconds), float(s.dwell_seconds), float(s.approach_seconds),
                float(s.pause_seconds), float(s.dive_seconds)]  # fmt: skip

    def _seconds(self, dives: bool) -> float:
        phases = self._phases()
        return sum(phases) if dives else phases[0] + phases[1]

    def wrapped(self, t: float) -> float:
        """t within the timeline: looping, gone round as often as it takes."""
        if self.loop and t >= self.duration > self.loop_at:
            t = self.loop_at + (t - self.loop_at) % (self.duration - self.loop_at)
        return min(max(t, 0.0), self.duration)

    def _step_at(self, t: float) -> tuple[int, float]:
        """(step, seconds into it) at t (looping: around; else clamped to the ends)."""
        t = self.wrapped(t)
        starts = [step.start for step in self.steps]
        k = max(0, int(np.searchsorted(starts, t, side="right")) - 1)
        return k, t - self.steps[k].start

    def position_at(self, t: float) -> int:
        """The record index of the level whose tiles show at t (frame wants it made)."""
        return self.steps[self._step_at(t)[0]].position

    def phase_at(self, t: float) -> tuple[TimelineStep, str]:
        """The step at t and its phase there (one of PHASES), without the level."""
        k, local = self._step_at(t)
        step = self.steps[k]
        for name, seconds in zip(PHASES, self._phases(), strict=True):
            if name == "approach" and step.pick < 0:
                return step, "dwell"  # it ends at rest
            if local < seconds or name == "dive":
                return step, name
            local -= seconds
        return step, "dive"

    def frame(self, t: float, level) -> TimelineFrame:
        """The timeline at t seconds; level: the record's level position_at(t), made
        (InfiniteSession.rebuild, or the live one)."""
        k, local = self._step_at(t)
        step = self.steps[k]
        scene, ctx = level.scene, level.ctx
        canvas = np.array([ctx.width, ctx.height], np.float64)
        middle = canvas / 2
        fit = float(np.min(self.viewport / canvas))  # the view fits the canvas (margin 1)
        tiles = rest_tiles(scene)
        fade_s, dwell_s, approach_s, pause_s, dive_s = self._phases()
        if local < fade_s:  # the photo it is the mosaic of fades into it
            photo = None
            if level.source is not None:
                photo = Photo(step.position, middle, canvas, 0.0, fade_out(local, fade_s))
            return TimelineFrame(step.position, "fade", tiles, middle, fit, photo)
        local -= fade_s
        if local < dwell_s or step.pick < 0:
            return TimelineFrame(step.position, "dwell", tiles, middle, fit)
        local -= dwell_s
        i = step.pick
        target = scene.center[i].astype(np.float64)
        zoom1 = approach_zoom(float(self.viewport[0]), float(ctx.tile_size[0]),
                              float(self.settings.approach_tiles), fit)  # fmt: skip
        outline = i if self.settings.highlight else -1
        if local < approach_s:
            e = local / approach_s
            center, zoom = glide(target, (target - middle) * fit, fit, zoom1,
                                 float(ease_in_out_cubic(e)))  # fmt: skip
            return TimelineFrame(step.position, "approach", tiles, center, zoom, None, outline,
                                 min(1.0, 2 * e))  # fmt: skip
        local -= approach_s
        if local < pause_s:
            return TimelineFrame(step.position, "pause", tiles, target, zoom1, None, outline,
                                 1.0)  # fmt: skip
        local -= pause_s
        e = min(local / dive_s, 1.0) if dive_s > 0 else 1.0
        view = self._dive(k, level, i, target, zoom1).frame(e, tiles)
        photo = Photo(step.next, target, np.abs(scene.size[i]).astype(np.float64),
                      view.photo_rotation, view.photo_alpha)  # fmt: skip
        return TimelineFrame(step.position, "dive", tiles, view.center, view.zoom, photo)

    def _dive(self, k: int, level, index: int, center, zoom: float) -> DiveMotion:
        dive = self._dives.get(k)
        if dive is None:
            scene = level.scene
            end_zoom = float(np.min(self.viewport / np.abs(scene.size[index])))  # it fills
            seed = dive_seed(index, int(scene.slot[index]), len(scene))
            style = pick_style(str(self.settings.dive_style), seed)
            dive = self._dives[k] = DiveMotion(rest_tiles(scene), index, center, zoom,
                                               self.viewport, (0.0, 1.0), end_zoom, style,
                                               seed)  # fmt: skip
        return dive

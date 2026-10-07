"""Camera keyframes: shots at moments of the video, and the moves between them.

Keys
----
A `CameraKey` is a `Shot` (camera.py: the table point at the frame's middle,
a zoom relative to the video's framing, a turn) at a `KeyTime`, whether the
camera stops there, and how it moves on to the next key (`motion`).

Keys live on the video's clock (video.py: start hold, animation, end hold),
so the camera can move before the build starts and over the finished mosaic.
A `KeyTime` anchors a key to one part of the video, so it stays put relative
to that part when lengths change: "lead" (seconds from the start, during the
start hold), "body" (a share of the animation, or seconds from its start
when the track is pinned, see `CameraTrack.stretch`) or "tail" (seconds
after the animation ends). Times past a part's end are clamped to it.

Motion
------
A shot is a similarity map of the table onto the frame, p -> alpha p + beta
in complex numbers, with alpha = zoom e^(-i turn). Moving along a geodesic of
these maps scales and turns the frame about one fixed table point (the
spiral segments of camera.py), so nothing slides across the frame on its
own. Between keys the path is a cubic Bezier of such maps (de Casteljau with
geodesics), with Catmull-Rom tangents (spaced by the keys' times) through
keys that pass and none at keys that stop, where the camera eases to rest.
Keys on one spiral (all zooming about one point, say) give a path on that
spiral. Motion "linear" is the plain geodesic at a steady pace; "hold"
keeps the key's shot until the next key, then cuts. Turns are kept unwrapped,
so keys can turn the camera several times round.

Before the first key and after the last, the camera holds still; without
keys it shows the video's framing.
"""

import cmath
import math
from dataclasses import dataclass, field, replace

import numpy as np

from skitter.core.animation.camera import CameraMove, CameraPath, Shot, StillPath, home_shot
from skitter.core.animation.video import VideoClock

PARTS = ("lead", "body", "tail")
MOTIONS = (("smooth", "Smooth"), ("linear", "Steady"), ("hold", "Hold, then cut"))
PATH_SAMPLES = 256  # moments sampled for the path's closest zoom (texture detail)


@dataclass(frozen=True)
class KeyTime:
    """When a key is, anchored to a part of the video (see the module docstring)."""

    part: str  # "lead", "body" or "tail"
    value: float  # lead / tail: seconds; body: share of the animation, or seconds if pinned

    def seconds(self, clock: VideoClock, stretch: bool = True) -> float:
        """Video time of the key."""
        if self.part == "lead":
            return min(max(self.value, 0.0), clock.hold_start)
        if self.part == "tail":
            return clock.animation_end + min(max(self.value, 0.0), clock.hold_end)
        body = self.value * clock.duration if stretch else self.value
        return clock.hold_start + min(max(body, 0.0), clock.duration)

    @classmethod
    def at(cls, t: float, clock: VideoClock, stretch: bool = True) -> "KeyTime":
        """The anchored time of video time t (the animation itself takes its ends)."""
        t = min(max(float(t), 0.0), clock.total)
        if t < clock.hold_start - 1e-9:
            return cls("lead", t)
        if t > clock.animation_end + 1e-9:
            return cls("tail", t - clock.animation_end)
        body = t - clock.hold_start
        if stretch:
            return cls("body", body / clock.duration if clock.duration > 0 else 0.0)
        return cls("body", body)


@dataclass(frozen=True)
class CameraKey:
    time: KeyTime
    shot: Shot
    stop: bool = True  # the camera comes to rest here (else it moves on through)
    motion: str = "smooth"  # how it goes on to the next key (MOTIONS)


@dataclass(frozen=True)
class CameraTrack:
    """The camera's keys. stretch: keys during the animation keep their share of it
    when its length changes (else their seconds from its start)."""

    keys: tuple[CameraKey, ...] = ()  # in time order (see _in_order)
    stretch: bool = True

    def times(self, clock: VideoClock) -> np.ndarray:
        return np.array([k.time.seconds(clock, self.stretch) for k in self.keys])

    def path(self, clock: VideoClock, base) -> CameraPath:
        """The camera over the video; base: the video's framing (zoom 1)."""
        if not self.keys:
            return StillPath(home_shot(base))
        times = self.times(clock)
        order = np.argsort(times, kind="stable")
        return KeyframePath(times[order], tuple(self.keys[i] for i in order))

    # Editing (each returns a new track)

    def with_key(self, key: CameraKey, clock: VideoClock, tolerance: float = 1e-3):
        """With key added, or replacing a key at the same moment (within tolerance s)."""
        t = key.time.seconds(clock, self.stretch)
        kept = tuple(k for k in self.keys
                     if abs(k.time.seconds(clock, self.stretch) - t) > tolerance)  # fmt: skip
        return replace(self, keys=_in_order((*kept, key)))

    def without(self, index: int):
        return replace(self, keys=self.keys[:index] + self.keys[index + 1 :])

    def replaced(self, index: int, key: CameraKey):
        keys = list(self.keys)
        keys[index] = key
        return replace(self, keys=_in_order(keys))

    def restretched(self, stretch: bool, clock: VideoClock) -> "CameraTrack":
        """The same key moments, kept from now on by share (True) or by seconds."""
        keys = tuple(replace(k, time=KeyTime.at(k.time.seconds(clock, self.stretch), clock,
                                                stretch)) for k in self.keys)  # fmt: skip
        return CameraTrack(keys, stretch)

    # Project files

    def to_dict(self) -> dict:
        return {
            "stretch": self.stretch,
            "keys": [
                {
                    "part": k.time.part, "time": k.time.value,
                    "center": list(k.shot.center), "zoom": k.shot.zoom,
                    "rotation": k.shot.rotation, "stop": k.stop, "motion": k.motion,
                }
                for k in self.keys
            ],
        }  # fmt: skip

    @classmethod
    def from_dict(cls, data: dict, problems: list[str] | None = None) -> "CameraTrack":
        keys = []
        motions = {m for m, _ in MOTIONS}
        for item in data.get("keys", []):
            try:
                part = item["part"] if item["part"] in PARTS else "body"
                shot = Shot((float(item["center"][0]), float(item["center"][1])),
                            max(float(item["zoom"]), 1e-3), float(item["rotation"]))  # fmt: skip
                motion = item.get("motion", "smooth")
                keys.append(CameraKey(KeyTime(part, float(item["time"])), shot,
                                      bool(item.get("stop", True)),
                                      motion if motion in motions else "smooth"))  # fmt: skip
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                if problems is None:
                    raise
                problems.append(f"Camera key left out ({exc})")
        return cls(_in_order(keys), bool(data.get("stretch", True)))


def _in_order(keys) -> tuple[CameraKey, ...]:
    """Keys in time order. Anchors alone decide it (start hold, animation, end hold, each
    by value), so it holds for any lengths: tracks keep their keys in this order."""
    return tuple(sorted(keys, key=lambda k: (PARTS.index(k.time.part), k.time.value)))


# Similarity maps of the table onto the frame: p -> alpha p + beta, with
# alpha = exp(log_alpha); log_alpha = ln zoom - i turn, kept unwrapped.


@dataclass(frozen=True)
class _Map:
    log_alpha: complex
    beta: complex

    @property
    def alpha(self) -> complex:
        return cmath.exp(self.log_alpha)

    @classmethod
    def of(cls, shot: Shot) -> "_Map":
        log_alpha = complex(math.log(shot.zoom), -shot.rotation)
        return cls(log_alpha, -cmath.exp(log_alpha) * complex(*shot.center))

    def shot(self) -> Shot:
        center = -self.beta / self.alpha
        return Shot((float(center.real), float(center.imag)),
                    float(math.exp(self.log_alpha.real)), float(-self.log_alpha.imag))  # fmt: skip

    def then(self, other: "_Map") -> "_Map":
        """self applied after other (self o other)."""
        return _Map(self.log_alpha + other.log_alpha, self.alpha * other.beta + self.beta)

    def inverse(self) -> "_Map":
        return _Map(-self.log_alpha, -self.beta / self.alpha)


def _phi(x: complex) -> complex:
    """(e^x - 1) / x, smooth through 0."""
    return 1.0 + x / 2 if abs(x) < 1e-6 else (cmath.exp(x) - 1) / x


def _log(m: _Map) -> tuple[complex, complex]:
    """The generator (w, v) with exp(w, v) = m."""
    return m.log_alpha, m.beta / _phi(m.log_alpha)


def _exp(w: complex, v: complex) -> _Map:
    return _Map(w, v * _phi(w))


def _geodesic(a: _Map, b: _Map, u: float) -> _Map:
    """From a (u = 0) to b (u = 1) along the steady spiral between them."""
    w, v = _log(b.then(a.inverse()))
    return _exp(u * w, u * v).then(a)


@dataclass(frozen=True)
class KeyframePath(CameraPath):
    """The camera through keys at the given (sorted) video times."""

    times: np.ndarray
    keys: tuple[CameraKey, ...]

    def __post_init__(self):
        maps = [_Map.of(k.shot) for k in self.keys]
        n, t = len(maps), self.times
        tangents = []  # generators per second, applied on the left (frame coordinates)
        for i, key in enumerate(self.keys):
            lo, hi = max(i - 1, 0), min(i + 1, n - 1)
            if key.stop or hi == lo or t[hi] - t[lo] <= 1e-9:
                tangents.append((0j, 0j))
                continue
            w, v = _log(maps[hi].then(maps[lo].inverse()))
            span = float(t[hi] - t[lo])
            tangents.append((w / span, v / span))
        controls = []  # (after key i, before key i + 1)
        for i in range(n - 1):
            gap = float(t[i + 1] - t[i]) / 3
            (wa, va), (wb, vb) = tangents[i], tangents[i + 1]
            controls.append((_exp(wa * gap, va * gap).then(maps[i]),
                             _exp(-wb * gap, -vb * gap).then(maps[i + 1])))  # fmt: skip
        object.__setattr__(self, "_maps", maps)
        object.__setattr__(self, "_controls", controls)

    def shot(self, t: float) -> Shot:
        times, keys, maps = self.times, self.keys, self._maps
        if t <= times[0]:
            return keys[0].shot
        if t >= times[-1]:
            return keys[-1].shot
        i = int(np.searchsorted(times, t, side="right")) - 1
        i = min(max(i, 0), len(keys) - 2)
        span = times[i + 1] - times[i]
        if span <= 1e-12:
            return keys[i + 1].shot
        u = float((t - times[i]) / span)
        motion = keys[i].motion
        if motion == "hold":
            return keys[i].shot
        if motion == "linear":
            return _geodesic(maps[i], maps[i + 1], u).shot()
        c1, c2 = self._controls[i]
        p = [maps[i], c1, c2, maps[i + 1]]  # de Casteljau with geodesics
        while len(p) > 1:
            p = [_geodesic(a, b, u) for a, b in zip(p, p[1:], strict=False)]
        return p[0].shot()

    @property
    def max_zoom(self) -> float:
        times = np.linspace(self.times[0], self.times[-1], PATH_SAMPLES)
        return max([k.shot.zoom for k in self.keys] + [self.shot(float(t)).zoom for t in times])


@dataclass(frozen=True)
class AnimationTimePath(CameraPath):
    """A path planned on animation time, followed on the video clock (still during the
    holds): how the camera moves of camera.py run until they become keys."""

    path: CameraPath
    clock: VideoClock = field(default_factory=lambda: VideoClock(0.0, 0.0, 0.0))

    def shot(self, t: float) -> Shot:
        return self.path.shot(float(self.clock.animation_time(t)))

    @property
    def max_zoom(self) -> float:
        return self.path.max_zoom


def video_camera_path(track: CameraTrack, move: CameraMove, scene, timeline, base,
                      clock: VideoClock) -> CameraPath:  # fmt: skip
    """The camera over the whole video: the track's keys, or while it has none, the
    camera move (planned on animation time). base: the video's framing (zoom 1)."""
    if track.keys:
        return track.path(clock, base)
    return AnimationTimePath(move.path(scene, timeline, base), clock)

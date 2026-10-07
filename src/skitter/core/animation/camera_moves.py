"""Ready-made camera moves, written as keyframes.

A `CameraMove` is a configurable recipe that writes `CameraKey`s
(keyframes.py) for a scene, its timeline and the video's framing; they then
are ordinary keys to edit. Moves time their keys as shares of the animation,
so they stay in step when its length changes.

Built-in: Follow the action (keys that keep the landing spots of the tiles
in flight in view, then the whole view at the end): the one move that is
hard to key by hand, as it reads the choreography. Simple moves (pull back,
pan, ...) are a few keys made in the viewfinder.

Add a move by subclassing CameraMove (declare Params, write keys) and
decorating it with @register_camera_move.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from scipy.ndimage import gaussian_filter1d

from skitter.core import easing
from skitter.core.animation.base import Timeline
from skitter.core.animation.camera import Shot, WorldRect, home_shot, keep_on_mosaic
from skitter.core.animation.keyframes import CameraKey, KeyTime
from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import Configurable, FloatParam, IntParam


class CameraMove(Configurable, ABC):
    """A configurable way to move the camera; writes keys."""

    id: ClassVar[str] = ""
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    @abstractmethod
    def keys(self, scene: MosaicScene, timeline: Timeline, base: WorldRect) -> list[CameraKey]:
        """Keys for this scene and timeline (time order); base: the video's framing."""


_registry: dict[str, type[CameraMove]] = {}


def register_camera_move(cls: type[CameraMove]) -> type[CameraMove]:
    """Class decorator making a camera move available to the UI."""
    if not cls.id or not cls.name:
        raise TypeError(f"{cls.__name__} must define id and name")
    existing = _registry.get(cls.id)
    if existing is not None and existing is not cls:
        raise ValueError(f"camera move id {cls.id!r} is already used by {existing.__name__}")
    _registry[cls.id] = cls
    return cls


def camera_move_types() -> list[type[CameraMove]]:
    """Registered camera moves, in the order they were registered."""
    return list(_registry.values())


def get_camera_move(type_id: str) -> type[CameraMove]:
    try:
        return _registry[type_id]
    except KeyError:
        raise KeyError(f"unknown camera move {type_id!r}") from None


# Shared settings and helpers


def _zoom(default: float = 4.0, label: str = "Close-up", help: str = "") -> FloatParam:
    return FloatParam(
        default, label, min=1.0, max=20.0, step=0.5, decimals=1, suffix="×",
        help=help or "How close the close-up is, compared with the video's framing.",
    )  # fmt: skip


def _key(share: float, shot: Shot, stop: bool = True, motion: str = "smooth") -> CameraKey:
    """A key at a share (percent) of the animation."""
    return CameraKey(KeyTime("body", min(max(share, 0.0), 100.0) / 100.0), shot, stop, motion)


# Built-in moves


FOLLOW_SAMPLES = 96  # moments the timeline is looked at to find the action


def _nearest(found: np.ndarray) -> np.ndarray:
    """For each index, the nearest index where found is True (found has one at least)."""
    have = np.flatnonzero(found)
    index = np.arange(len(found))
    right = np.searchsorted(have, index).clip(0, len(have) - 1)
    left = (right - 1).clip(0, len(have) - 1)
    closer = np.abs(have[left] - index) <= np.abs(have[right] - index)
    return np.where(closer, have[left], have[right])


@register_camera_move
class FollowMove(CameraMove):
    id = "follow"
    name = "Follow the action"
    description = (
        "Keys that watch where the mosaic is being built: the frame keeps the landing "
        "spots of the tiles in flight in view, then shows the whole mosaic at the end."
    )

    count = IntParam(8, "Keys", min=3, max=40,
                     help="How many keys to write: more follow the action more "
                          "closely.")  # fmt: skip
    closest = _zoom(3.0, "Closest", help="The closest the camera gets, compared with the "
                                        "video's framing.")  # fmt: skip
    room = FloatParam(
        1.5, "Room", min=1.0, max=4.0, step=0.25, decimals=2, suffix="×",
        help="How much room the frame leaves around the action (1: just enough).",
    )  # fmt: skip
    smoothness = FloatParam(
        10.0, "Smoothness", min=0.0, max=50.0, step=2.5, decimals=1, suffix="%",
        help="How calmly the camera follows: it averages the action over this share of "
             "the animation.",
    )  # fmt: skip
    ending = FloatParam(
        20.0, "Pull back over", min=0.0, max=100.0, step=5.0, decimals=0, suffix="%",
        help="The last share of the animation, over which the camera eases back to the "
             "whole mosaic.",
    )  # fmt: skip

    def keys(self, scene, timeline, base) -> list[CameraKey]:
        duration = timeline.duration
        home = home_shot(base)
        if duration <= 0 or not len(scene):
            return []
        times, centers, zooms = self._follow(scene, timeline, base, home)
        if times is None:
            return []
        shares = np.linspace(0.0, 100.0, self.count)
        keys = []
        for i, share in enumerate(shares):
            t = duration * share / 100.0
            center = (float(np.interp(t, times, centers[:, 0])),
                      float(np.interp(t, times, centers[:, 1])))  # fmt: skip
            zoom = float(np.exp(np.interp(t, times, np.log(zooms))))
            ends = i in (0, len(shares) - 1)  # the camera starts and ends at rest
            keys.append(_key(float(share), Shot(center, zoom), stop=ends))
        return keys

    def _follow(self, scene, timeline, base, home):
        """(times, centers, zooms) of a frame keeping the action in view, smoothed and
        easing back to the whole view at the end; (None, None, None) without action."""
        duration = timeline.duration
        times = np.linspace(0.0, duration, FOLLOW_SAMPLES)
        spots = np.asarray(scene.center, dtype=np.float64)
        reach = 0.5 * np.asarray(scene.size, dtype=np.float64).max(axis=1)
        centers = np.full((len(times), 2), np.nan)
        zooms = np.full(len(times), np.nan)
        for i, t in enumerate(times):
            frame = timeline.frame(float(t))
            moving = frame.alpha > 0
            if frame.rest is not None:
                moving &= ~frame.rest
            else:
                moving &= np.abs(frame.center - spots).sum(axis=1) > 1e-6
            if not moving.any():
                continue
            where = spots[moving]
            lo = np.percentile(where, 10, axis=0) - reach[moving].mean()
            hi = np.percentile(where, 90, axis=0) + reach[moving].mean()
            centers[i] = (lo + hi) / 2
            need = np.maximum(hi - lo, 1e-9) * self.room
            zooms[i] = min(self.closest, base[2] / need[0], base[3] / need[1])
        found = ~np.isnan(zooms)
        if not found.any():
            return None, None, None
        # Moments without action hold the shot of the nearest moment with some.
        nearest = _nearest(found)
        centers, zooms = centers[nearest], np.maximum(zooms[nearest], 1.0)
        sigma = self.smoothness / 100.0 * (len(times) - 1)
        log_zoom = np.log(zooms)
        if sigma > 0:
            centers = gaussian_filter1d(centers, sigma, axis=0, mode="nearest")
            log_zoom = gaussian_filter1d(log_zoom, sigma, mode="nearest")
        zooms = np.exp(log_zoom)
        # Ease back to the whole mosaic over the last share.
        start = duration * (1 - self.ending / 100.0)
        u = np.clip((times - start) / max(duration - start, 1e-9), 0.0, 1.0)
        u = easing.ease_in_out_cubic(u) if self.ending > 0 else (times >= duration) * 1.0
        zooms = np.exp(np.log(zooms) * (1 - u))
        centers = centers + (np.asarray(home.center) - centers) * u[:, None]
        centers = np.array([keep_on_mosaic(scene, base, c, z)
                            for c, z in zip(centers, zooms, strict=True)])  # fmt: skip
        return times, centers, zooms

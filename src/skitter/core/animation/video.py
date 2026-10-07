"""Video export settings and planning: formats, sizes, framing, frame times, checks.

Nothing here renders or encodes; see ui/render/video_renderer.py (GPU
frames) and core/animation/encode.py (files). Everything an export needs is
decided here from a scene, the background and VideoSettings, so it can be
checked before any work starts.
"""

import math
import shutil
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import (
    BoolParam,
    ChoiceParam,
    ColorParam,
    Configurable,
    FloatParam,
    IntParam,
)

MAX_SIDE = 3840  # 4K: no side longer than this...
MAX_PIXELS = 3840 * 2160  # ...and no more pixels than UHD
MIN_SIDE = 16


@dataclass(frozen=True)
class VideoFormat:
    id: str
    name: str
    extension: str  # "" for an image sequence (a folder)
    codec: str  # FFmpeg encoder ("png" for a sequence)
    alpha: bool  # can keep a transparent background
    even: bool  # needs even width and height (4:2:0 chroma)
    max_fps: float | None = None
    note: str = ""

    @property
    def sequence(self) -> bool:
        return self.extension == ""


FORMATS = {
    f.id: f
    for f in (
        VideoFormat(
            "mp4",
            "MP4 (H.264)",
            ".mp4",
            "libx264",
            alpha=False,
            even=True,
            note="Plays everywhere: social media, phones, browsers.",
        ),
        VideoFormat(
            "mp4_hevc",
            "MP4 (H.265)",
            ".mp4",
            "libx265",
            alpha=False,
            even=True,
            note="Smaller files; Apple devices and YouTube. Few browsers play it.",
        ),
        VideoFormat(
            "mp4_av1",
            "MP4 (AV1)",
            ".mp4",
            "libsvtav1",
            alpha=False,
            even=True,
            note="Smallest files; modern browsers and YouTube. Slower to encode.",
        ),
        VideoFormat(
            "webm",
            "WebM (VP9)",
            ".webm",
            "libvpx-vp9",
            alpha=True,
            even=True,
            note="For web pages; keeps transparency.",
        ),
        VideoFormat(
            "webp",
            "Animated WebP",
            ".webp",
            "libwebp_anim",
            alpha=True,
            even=False,
            note="Short web loops, far better than GIF; keeps transparency.",
        ),
        VideoFormat(
            "gif",
            "GIF",
            ".gif",
            "gif",
            alpha=False,
            even=False,
            max_fps=50.0,
            note="Old loops: 256 colors, large files. Prefer WebP or MP4.",
        ),
        VideoFormat(
            "prores",
            "MOV (ProRes)",
            ".mov",
            "prores_ks",
            alpha=True,
            even=True,
            note="For video editors (ProRes 4444 with transparency, else 422 HQ). "
            "Very large files.",
        ),
        VideoFormat(
            "png",
            "PNG sequence",
            "",
            "png",
            alpha=True,
            even=False,
            note="One lossless PNG per frame in a new folder; keeps transparency.",
        ),
    )  # fmt: skip
}

RESOLUTIONS = (  # (id, label, (width, height) or None)
    ("1080p", "1080p (1920 × 1080)", (1920, 1080)),
    ("720p", "720p (1280 × 720)", (1280, 720)),
    ("1440p", "1440p (2560 × 1440)", (2560, 1440)),
    ("4k", "4K (3840 × 2160)", (3840, 2160)),
    ("square", "Square (1080 × 1080)", (1080, 1080)),
    ("portrait", "Portrait 4:5 (1080 × 1350)", (1080, 1350)),
    ("vertical", "Vertical 9:16 (1080 × 1920)", (1080, 1920)),
    ("mosaic", "Mosaic shape", None),
    ("custom", "Custom", None),
)
FRAME_RATES = (("30", "30 fps"), ("24", "24 fps"), ("25", "25 fps"), ("50", "50 fps"),
               ("60", "60 fps"), ("custom", "Custom"))  # fmt: skip
QUALITY = (("high", "High"), ("balanced", "Balanced"), ("small", "Small file"))
SPEED = (("slow", "Slow (smaller file)"), ("medium", "Medium"), ("fast", "Fast"))
SUPERSAMPLING = ((1, "Off"), (2, "2 × 2"), (3, "3 × 3"), (4, "4 × 4"))
MOTION_BLUR = ((0, "Off"), (4, "4 samples"), (8, "8 samples"), (16, "16 samples"))


class VideoSettings(Configurable):
    format = ChoiceParam(
        "mp4", "Format", choices=[(f.id, f.name) for f in FORMATS.values()],
        help="\n".join(f"{f.name}: {f.note}" for f in FORMATS.values()),
    )  # fmt: skip
    resolution = ChoiceParam("1080p", "Resolution", choices=[(i, n) for i, n, _ in RESOLUTIONS])
    width = IntParam(
        1920, "Width", min=MIN_SIDE, max=MAX_SIDE, suffix=" px",
        when=lambda s: s.resolution in ("custom", "mosaic"),
    )  # fmt: skip
    height = IntParam(
        1080, "Height", min=MIN_SIDE, max=MAX_SIDE, suffix=" px",
        when=lambda s: s.resolution == "custom",
    )  # fmt: skip
    frame_rate = ChoiceParam("30", "Frame rate", choices=FRAME_RATES)
    custom_fps = FloatParam(
        30.0, "Custom rate", min=1.0, max=120.0, step=1.0, decimals=3, suffix=" fps",
        when=lambda s: s.frame_rate == "custom",
    )  # fmt: skip
    framing = ChoiceParam(
        "fit", "Framing", choices=[("fit", "Fit (whole mosaic)"), ("fill", "Fill (crop)")],
        help="Fit shows the whole mosaic with the background around it; Fill crops it "
             "to fill the frame.",
    )  # fmt: skip
    margin = FloatParam(
        5.0, "Margin", min=0.0, max=50.0, step=1.0, decimals=1, suffix=" %",
        help="Space around the mosaic, as a share of the frame.",
    )  # fmt: skip
    hold_start = FloatParam(
        0.0, "Hold at start", min=0.0, max=60.0, step=0.5, decimals=1, suffix=" s",
        help="Show the empty first frame this long before tiles start moving.",
    )  # fmt: skip
    hold_end = FloatParam(
        2.0, "Hold at end", min=0.0, max=60.0, step=0.5, decimals=1, suffix=" s",
        help="Show the finished mosaic this long at the end.",
    )  # fmt: skip
    loop = BoolParam(
        True, "Loop forever", when=lambda s: s.format in ("gif", "webp"),
        help="GIF and WebP: play again and again (otherwise once).",
    )  # fmt: skip
    quality = ChoiceParam(
        "high", "Quality", choices=QUALITY,
        when=lambda s: s.format not in ("prores", "png"),
        help="Trades file size for detail. High suits upload to social media, which "
             "compresses again.",
    )  # fmt: skip
    speed = ChoiceParam(
        "medium", "Encoding speed", choices=SPEED, when=lambda s: s.format != "prores",
        help="Slower encoding makes smaller files at the same quality.",
    )  # fmt: skip
    supersampling = ChoiceParam(
        2, "Supersampling", choices=SUPERSAMPLING,
        help="Render each pixel from this many samples per side (on top of 8 × "
             "multisampling) for smooth edges and fine tile detail.",
    )  # fmt: skip
    motion_blur = ChoiceParam(
        8, "Motion blur", choices=MOTION_BLUR,
        help="Blend this many moments within each frame, so fast tiles streak smoothly "
             "instead of jumping.",
    )  # fmt: skip
    shutter = FloatParam(
        180.0, "Shutter angle", min=10.0, max=360.0, step=10.0, decimals=0, suffix="°",
        when=lambda s: s.motion_blur > 0,
        help="How much of each frame's time the blur covers (180° is the film look).",
    )  # fmt: skip

    @property
    def video_format(self) -> VideoFormat:
        return FORMATS[self.format]

    @property
    def fps(self) -> Fraction:
        """Exact frame rate (custom rates like 29.97 become 30000/1001)."""
        if self.frame_rate != "custom":
            return Fraction(int(self.frame_rate))
        for base in (24, 30, 60, 120):  # broadcast rates: 23.976 means 24000/1001, ...
            if abs(self.custom_fps - base * 1000 / 1001) < 0.005:
                return Fraction(base * 1000, 1001)
        return Fraction(self.custom_fps).limit_denominator(1000)


@dataclass(frozen=True)
class VideoClock:
    """The video's own time: a start hold, the animation, an end hold.

    Tiles follow animation time, which stands still during the holds; the
    camera (keyframes.py) follows video time, so it can move over the empty
    table before the build and over the finished mosaic after it.
    """

    hold_start: float
    duration: float  # the animation's
    hold_end: float

    @property
    def total(self) -> float:
        return self.hold_start + self.duration + self.hold_end

    @property
    def animation_end(self) -> float:
        """Video time at which the animation finishes."""
        return self.hold_start + self.duration

    def animation_time(self, video_time):
        """Animation time at video time(s): still during the holds."""
        return np.clip(np.asarray(video_time, dtype=np.float64) - self.hold_start,
                       0.0, self.duration)  # fmt: skip

    @classmethod
    def of(cls, settings: "VideoSettings", duration: float) -> "VideoClock":
        return cls(float(settings.hold_start), float(duration), float(settings.hold_end))


class ClockedTimeline:
    """A timeline seen on the video clock: frame(t) at video time t, over the holds
    too (still before and after the animation)."""

    def __init__(self, timeline, clock: VideoClock):
        self.timeline, self.clock = timeline, clock
        self.duration = clock.total

    def frame(self, t: float):
        return self.timeline.frame(float(self.clock.animation_time(t)))


@dataclass(frozen=True)
class VideoPlan:
    """Everything decided before rendering: size, view, frame times."""

    width: int
    height: int
    view: tuple[float, float, float, float]  # world rect (x, y, w, h) the frame shows
    fps: Fraction
    clock: np.ndarray  # (F,) video time of each frame, seconds
    hold_start: float  # video seconds before the animation starts
    duration: float  # animation length, seconds
    shutter: float  # seconds each frame's blur covers
    samples: int  # moments per frame (1: no motion blur)
    alpha: bool  # keep a transparent background

    @property
    def frames(self) -> int:
        return len(self.clock)

    @property
    def video_clock(self) -> VideoClock:
        total = float(self.clock[-1]) if len(self.clock) else 0.0
        return VideoClock(self.hold_start, self.duration,
                          max(0.0, total - self.hold_start - self.duration))  # fmt: skip

    @property
    def times(self) -> np.ndarray:
        """(F,) animation time of each frame (still during the holds)."""
        return np.clip(self.clock - self.hold_start, 0.0, self.duration)

    @property
    def seconds(self) -> float:
        return self.frames / float(self.fps)

    @property
    def scale(self) -> float:
        """Output pixels per mosaic unit."""
        return self.width / self.view[2]

    def video_moments(self, k: int) -> np.ndarray:
        """Video times blended into frame k: spread over the shutter around its time (the
        camera follows these). The last frame is exactly the end."""
        if self.samples <= 1 or self.shutter <= 0 or k == self.frames - 1:
            return self.clock[k : k + 1].astype(np.float64)
        offsets = (np.arange(self.samples) + 0.5) / self.samples - 0.5
        return np.clip(self.clock[k] + offsets * self.shutter, 0.0, float(self.clock[-1]))

    def moments(self, k: int) -> np.ndarray:
        """Animation times blended into frame k: video_moments mapped through the holds
        (so held frames stay perfectly still). The last frame is exactly the finished
        mosaic."""
        if k == self.frames - 1:
            return self.times[k : k + 1]
        return self.video_clock.animation_time(self.video_moments(k))


def output_size(settings: VideoSettings, scene: MosaicScene) -> tuple[int, int]:
    """Pixel size of the video, rounded to even numbers where the format needs it."""
    preset = dict((i, size) for i, _, size in RESOLUTIONS)[settings.resolution]
    if preset is not None:
        width, height = preset
    elif settings.resolution == "mosaic":
        x0, y0, x1, y1 = content_rect(scene)
        width = settings.width
        height = round(width * (y1 - y0) / max(x1 - x0, 1e-9))
    else:
        width, height = settings.width, settings.height
    if settings.video_format.even:
        width, height = 2 * round(width / 2), 2 * round(height / 2)
    return max(width, 2), max(height, 2)


def sync_size(settings: VideoSettings, scene: MosaicScene | None) -> None:
    """Show a preset's actual size in the (inactive) width and height, so switching to
    Custom starts from the size the video has now."""
    preset = dict((i, size) for i, _, size in RESOLUTIONS)[settings.resolution]
    if preset is not None:
        settings.width, settings.height = preset
    elif settings.resolution == "mosaic" and scene is not None and len(scene):
        height = output_size(settings, scene)[1]
        settings.height = min(max(height, MIN_SIDE), MAX_SIDE)


def content_rect(scene: MosaicScene) -> tuple[float, float, float, float]:
    """What the video frames: the image frame and any tile hanging past it."""
    x0, y0, x1, y1 = scene.bounds
    w, h = scene.canvas
    return min(x0, 0.0), min(y0, 0.0), max(x1, w), max(y1, h)


def view_rect(scene: MosaicScene, width: int, height: int, settings: VideoSettings):
    """World rect (x, y, w, h) shown by a width x height frame."""
    x0, y0, x1, y1 = content_rect(scene)
    cw, ch = x1 - x0, y1 - y0
    aspect = width / height
    keep = 1.0 - 2.0 * settings.margin / 100.0  # share of the frame the mosaic spans
    if settings.framing == "fit":
        vw = max(cw, ch * aspect) / max(keep, 1e-3)
    else:
        vw = min(cw, ch * aspect) / max(keep, 1e-3)
    vh = vw / aspect
    return ((x0 + x1) / 2 - vw / 2, (y0 + y1) / 2 - vh / 2, vw, vh)


def plan_video(
    scene: MosaicScene, duration: float, settings: VideoSettings, background: str
) -> VideoPlan:
    width, height = output_size(settings, scene)
    fps = settings.fps
    total = settings.hold_start + duration + settings.hold_end
    count = max(1, math.floor(total * fps) + 1)
    clock = np.arange(count) / float(fps)
    clock[-1] = total  # the last frame is exactly the end: the finished mosaic
    samples = int(settings.motion_blur)
    return VideoPlan(
        width=width,
        height=height,
        view=view_rect(scene, width, height, settings),
        fps=fps,
        clock=clock,
        hold_start=settings.hold_start,
        duration=duration,
        shutter=settings.shutter / 360.0 / float(fps) if samples else 0.0,
        samples=max(samples, 1),
        alpha=background == ColorParam.TRANSPARENT and settings.video_format.alpha,
    )


def check_video(
    scene: MosaicScene | None, settings: VideoSettings, background: str, path: Path | None = None
) -> list[str]:
    """Problems that stop an export (empty: ready)."""
    if scene is None or not len(scene):
        return ["There is no mosaic to animate: match tiles first."]
    problems = []
    fmt = settings.video_format
    width, height = output_size(settings, scene)
    if max(width, height) > MAX_SIDE or width * height > MAX_PIXELS:
        problems.append(f"{width:,} × {height:,} is larger than 4K; choose a smaller size.")
    if min(width, height) < MIN_SIDE:
        problems.append(f"{width:,} × {height:,} is too small.")
    if fmt.max_fps is not None and settings.fps > fmt.max_fps:
        problems.append(f"{fmt.name} plays at most {fmt.max_fps:g} frames per second.")
    if background == ColorParam.TRANSPARENT and not fmt.alpha:
        alpha = ", ".join(f.name for f in FORMATS.values() if f.alpha)
        problems.append(
            f"{fmt.name} can't keep a transparent background. Choose a background color on "
            f"the Animate tab, or a format with transparency: {alpha}."
        )
    if path is not None:
        problems += _check_path(path, fmt)
    return problems


def _check_path(path: Path, fmt: VideoFormat) -> list[str]:
    folder = path if fmt.sequence else path.parent
    parent = folder.parent if fmt.sequence else folder
    if not parent.is_dir():
        return [f"The folder {parent} does not exist."]
    if fmt.sequence and path.exists() and (not path.is_dir() or any(path.iterdir())):
        return [f"{path.name} already exists and is not an empty folder; choose a new name."]
    return []


def estimate_bytes(plan: VideoPlan, fmt: VideoFormat) -> int | None:
    """Rough upper estimate for formats whose size is predictable (else None)."""
    pixels = plan.width * plan.height * plan.frames
    if fmt.id == "png":
        return int(pixels * (4 if plan.alpha else 3) * 0.6)
    if fmt.id == "prores":
        return int(pixels * (1.6 if plan.alpha else 1.1))
    return None


def free_bytes(path: Path) -> int | None:
    folder = path if path.is_dir() else path.parent
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    try:
        return shutil.disk_usage(folder).free
    except OSError:
        return None

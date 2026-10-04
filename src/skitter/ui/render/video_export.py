"""The video export job: textures, frames, encoding (runs on a worker thread).

Everything it uses is a snapshot taken on the UI thread (`VideoJob`), so
editing the project during an export can't affect it. Frames stream from
the GPU renderer straight into the encoder; on any error or cancel the
partial output is deleted (see encode.VideoWriter).
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

from skitter.core.animation import Choreography
from skitter.core.animation.encode import VideoWriter
from skitter.core.animation.look import AnimationLook, TableCamera
from skitter.core.animation.video import VideoPlan, VideoSettings
from skitter.core.scene import MosaicScene
from skitter.core.slicing.params import ColorParam
from skitter.ui.jobs import JobCancelled
from skitter.ui.render.tile_textures import DetailRequest, scene_instances
from skitter.ui.render.video_renderer import VideoRenderer

os.environ.setdefault("SVT_LOG", "1")  # AV1 encoder: errors only, not its banner


@dataclass(frozen=True)
class VideoJob:
    path: Path
    scene: MosaicScene
    choreography: Choreography  # a copy
    settings: VideoSettings  # a copy
    background: str  # "#rrggbb" or "transparent"
    look: AnimationLook  # a copy: camera height, light and shadows
    plan: VideoPlan
    request: DetailRequest  # tile crops at the plan's scale


@dataclass(frozen=True)
class VideoReport:
    path: Path
    frames: int
    size: tuple[int, int]
    seconds: float  # video length
    bytes: int
    elapsed: float  # time the export took
    detail_scale: float  # tile texture resolution used (1.0: full)
    failed: int  # tile files that could not be read (thumbnails used)


def _size(n: int) -> str:
    return f"{n / 2**20:,.1f} MB" if n >= 2**20 else f"{n / 2**10:,.0f} KB"


def run_video_job(job: VideoJob, progress, cancelled) -> VideoReport:
    start = time.perf_counter()
    plan, settings = job.plan, job.settings
    fmt = settings.video_format
    timeline = job.choreography.timeline(job.scene, job.look)

    def report_textures(message, fraction):
        progress(message, None if fraction is None else 0.08 * fraction)

    detail = job.request.build(report_textures, cancelled)
    layer, uv = detail.atlas.locate(job.request.image, job.scene.mirrored)
    progress("Starting the renderer…", 0.08)
    renderer = VideoRenderer(
        plan, detail.atlas.pages, scene_instances(job.scene, layer, uv),
        ColorParam.rgb(job.background) if job.background != ColorParam.TRANSPARENT else None,
        settings.supersampling, TableCamera.for_scene(job.scene, job.look),
    )  # fmt: skip
    passes = 2 if fmt.id == "gif" else 1
    clock = {"start": time.perf_counter(), "done": 0}
    total = plan.frames * passes

    def frames(label: str):
        for k in range(plan.frames):
            if cancelled():
                raise JobCancelled
            yield renderer.render(timeline, k)
            clock["done"] += 1
            done = clock["done"]
            elapsed = time.perf_counter() - clock["start"]
            rate = done / elapsed if elapsed > 0 else 0.0
            left = (total - done) / rate if rate > 0 else 0.0
            written = writer.bytes_written()
            text = f"{label} {k + 1:,} of {plan.frames:,} · {rate:.0f} fps · {left:.0f} s left"
            if written:
                text += f" · {_size(written)}"
            progress(text, 0.08 + 0.92 * done / total)

    try:
        writer = VideoWriter(
            job.path, fmt, settings, plan.width, plan.height, plan.fps, plan.alpha,
            frames_again=lambda: frames("Mapping colors: frame"),
        )  # fmt: skip
        try:
            for frame in frames("Frame"):
                writer.write(frame)
            progress("Finishing the file…", None)
            writer.close()
        except BaseException:
            writer.abort()
            raise
    finally:
        renderer.release()
    return VideoReport(
        path=job.path,
        frames=plan.frames,
        size=(plan.width, plan.height),
        seconds=plan.seconds,
        bytes=_disk_size(job.path),
        elapsed=time.perf_counter() - start,
        detail_scale=detail.scale,
        failed=detail.failed,
    )


def _disk_size(path: Path) -> int:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.iterdir())
    return path.stat().st_size

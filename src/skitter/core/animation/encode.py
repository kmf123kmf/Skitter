"""Writing rendered frames to video files (PyAV, which bundles FFmpeg).

A VideoWriter takes (H, W, 4) uint8 RGBA frames (straight alpha, sRGB) one
at a time and streams them to the encoder, so memory stays flat however
long the video. Output goes to a temporary name (a ".part" file, or a
".part" folder for PNG sequences) that becomes the real one only when
`close()` succeeds; `abort()` (or any error) deletes it. Nothing half
written is ever left behind under the chosen name.

Colors: frames are converted to YUV with the BT.709 matrix in limited
range and the stream is tagged so (BT.709 primaries, sRGB-like transfer),
which is what players and social sites expect for HD video. GIF frames get
one palette optimized for the whole animation (two passes over the frames,
which the caller supplies again through `frames_again`).
"""

import os
import shutil
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from av.video.reformatter import ColorRange, Colorspace
from PIL import Image

from skitter.core.animation.video import VideoFormat, VideoSettings

# Encoder settings per quality ("high", "balanced", "small") and speed.
CRF = {
    "libx264": {"high": 16, "balanced": 20, "small": 26},
    "libx265": {"high": 18, "balanced": 23, "small": 28},
    "libsvtav1": {"high": 22, "balanced": 30, "small": 38},
    "libvpx-vp9": {"high": 18, "balanced": 28, "small": 36},
}
X26X_PRESET = {"slow": "slow", "medium": "medium", "fast": "veryfast"}
SVT_PRESET = {"slow": "4", "medium": "6", "fast": "9"}
VP9_CPU = {"slow": "1", "medium": "2", "fast": "4"}
WEBP_QUALITY = {"high": 92, "balanced": 82, "small": 70}
WEBP_EFFORT = {"slow": "6", "medium": "4", "fast": "1"}
PNG_LEVEL = {"slow": 9, "medium": 6, "fast": 1}
CONTAINER = {".mp4": "mp4", ".webm": "webm", ".mov": "mov", ".webp": "webp", ".gif": "gif"}


class VideoWriter:
    """Streams RGBA frames into a video file (or a folder of PNGs)."""

    def __init__(
        self,
        path: str | Path,
        fmt: VideoFormat,
        settings: VideoSettings,
        width: int,
        height: int,
        fps: Fraction,
        alpha: bool,
        frames_again: Callable[[], Iterator[np.ndarray]] | None = None,
    ):
        self.path = Path(path)
        self.fmt = fmt
        self.settings = settings
        self.size = (width, height)
        self.fps = fps
        self.alpha = alpha and fmt.alpha
        self.frames_again = frames_again  # GIF: the same frames again, for the second pass
        self.part = self.path.with_name(self.path.name + ".part")
        self.count = 0
        self._container = None
        self._stream = None
        self._palette_graph = None
        self._gif_pts = 0
        self._closed = False
        self._remove_part()
        try:
            self._open()
        except BaseException:
            self.abort()
            raise

    # Public

    def write(self, rgba: np.ndarray) -> None:
        """Add the next frame: (height, width, 4) uint8, straight alpha."""
        if rgba.shape != (self.size[1], self.size[0], 4) or rgba.dtype != np.uint8:
            raise ValueError(f"expected a {self.size[0]} x {self.size[1]} RGBA uint8 frame")
        try:
            if self.fmt.sequence:
                self._write_png(rgba)
            elif self.fmt.id == "gif":
                self._palette_graph.vpush(self._frame(rgba, "rgb24"))
            else:
                self._encode(self._convert(rgba))
            self.count += 1
        except BaseException:
            self.abort()
            raise

    def bytes_written(self) -> int:
        """Size of the output so far."""
        try:
            if self.fmt.sequence:
                return sum(f.stat().st_size for f in self.part.iterdir())
            return self.part.stat().st_size
        except OSError:
            return 0

    def close(self) -> Path:
        """Finish the file and give it its real name."""
        if self._closed:
            return self.path
        try:
            if self.fmt.id == "gif":
                self._finish_gif()
            if self._stream is not None:
                self._encode(None)  # flush the encoder
            if self._container is not None:
                self._container.close()
                self._container = None
            if self.fmt.sequence:
                if self.path.exists():
                    self.path.rmdir()  # an empty folder (checked before export)
                os.replace(self.part, self.path)
            else:
                os.replace(self.part, self.path)
        except BaseException:
            self.abort()
            raise
        self._closed = True
        return self.path

    def abort(self) -> None:
        """Stop and delete everything written so far."""
        self._closed = True
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
        self._remove_part()

    # Opening

    def _open(self) -> None:
        fmt, s = self.fmt, self.settings
        if fmt.sequence:
            self.part.mkdir(parents=False)
            return
        options = {}
        if fmt.extension in (".mp4", ".mov"):
            options["movflags"] = "+faststart"  # playable while still downloading
        if fmt.id in ("gif", "webp"):
            options["loop"] = "0" if s.loop else ("-1" if fmt.id == "gif" else "1")
        self._container = av.open(
            str(self.part), mode="w", format=CONTAINER[fmt.extension], options=options
        )
        if fmt.id == "gif":
            self._open_gif()
            return
        stream = self._container.add_stream(fmt.codec, rate=self.fps)
        stream.width, stream.height = self.size
        stream.time_base = 1 / self.fps
        pix_fmt, codec_options = self._codec_setup()
        stream.pix_fmt = pix_fmt
        stream.options = codec_options
        if fmt.codec == "libx265":
            stream.codec_tag = "hvc1"  # lets Apple devices play it
        if pix_fmt.startswith("yuv"):
            ctx = stream.codec_context
            ctx.color_range = ColorRange.MPEG.value
            ctx.colorspace = Colorspace.ITU709.value
            ctx.color_primaries = 1  # BT.709
            ctx.color_trc = 1  # BT.709
        self._stream = stream

    def _codec_setup(self) -> tuple[str, dict[str, str]]:
        fmt, s = self.fmt, self.settings
        codec = fmt.codec
        if codec in ("libx264", "libx265"):
            options = {"crf": str(CRF[codec][s.quality]), "preset": X26X_PRESET[s.speed]}
            if codec == "libx264":
                options["profile"] = "high"
            else:
                options["x265-params"] = "log-level=error"
            return "yuv420p", options
        if codec == "libsvtav1":
            return "yuv420p", {"crf": str(CRF[codec][s.quality]), "preset": SVT_PRESET[s.speed]}
        if codec == "libvpx-vp9":
            options = {"crf": str(CRF[codec][s.quality]), "b": "0", "row-mt": "1",
                       "deadline": "good", "cpu-used": VP9_CPU[s.speed]}  # fmt: skip
            if self.alpha:
                options["auto-alt-ref"] = "0"
            return ("yuva420p" if self.alpha else "yuv420p"), options
        if codec == "libwebp_anim":
            return "bgra", {"quality": str(WEBP_QUALITY[s.quality]),
                            "compression_level": WEBP_EFFORT[s.speed]}  # fmt: skip
        if codec == "prores_ks":
            if self.alpha:  # ProRes 4444 with alpha
                return "yuva444p10le", {"profile": "4", "vendor": "apl0", "alpha_bits": "16"}
            return "yuv422p10le", {"profile": "3", "vendor": "apl0"}  # ProRes 422 HQ
        raise ValueError(f"no encoder setup for {codec}")

    # Frames

    def _frame(self, rgba: np.ndarray, format: str) -> av.VideoFrame:
        if format == "rgb24":
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgba[..., :3]), format="rgb24")
        else:
            frame = av.VideoFrame.from_ndarray(rgba, format="rgba")
        frame.pts = self.count
        frame.time_base = 1 / self.fps
        return frame

    def _convert(self, rgba: np.ndarray) -> av.VideoFrame:
        pix_fmt = self._stream.pix_fmt
        source = rgba if self.alpha else _opaque(rgba)
        frame = self._frame(source, "rgba")
        if pix_fmt.startswith("yuv"):
            frame = frame.reformat(
                format=pix_fmt,
                src_colorspace=Colorspace.ITU709,
                dst_colorspace=Colorspace.ITU709,
                dst_color_range=ColorRange.MPEG,
            )
        else:
            frame = frame.reformat(format=pix_fmt)
        frame.pts = self.count
        return frame

    def _encode(self, frame: av.VideoFrame | None) -> None:
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def _write_png(self, rgba: np.ndarray) -> None:
        mode = "RGBA" if self.alpha else "RGB"
        image = Image.fromarray(rgba if self.alpha else np.ascontiguousarray(rgba[..., :3]), mode)
        name = f"{self.path.name}_{self.count:05d}.png"
        image.save(self.part / name, "PNG", compress_level=PNG_LEVEL[self.settings.speed])

    # GIF: pass 1 builds one palette from every frame; pass 2 maps frames to it.

    def _open_gif(self) -> None:
        self._palette_graph = _graph(
            self.size, self.fps, "palettegen=stats_mode=full:max_colors=256"
        )

    def _finish_gif(self) -> None:
        graph = self._palette_graph
        graph.vpush(None)
        palette = graph.vpull()
        if self.frames_again is None:
            raise ValueError("GIF needs the frames twice (frames_again)")
        stream = self._container.add_stream("gif", rate=self.fps)
        stream.width, stream.height = self.size
        stream.pix_fmt = "pal8"
        stream.time_base = 1 / self.fps
        self._stream = stream
        mapper = av.filter.Graph()
        frames_in = mapper.add_buffer(width=self.size[0], height=self.size[1], format="rgb24",
                                      time_base=1 / self.fps)  # fmt: skip
        palette_in = mapper.add_buffer(width=16, height=16, format=palette.format.name,
                                       time_base=1 / self.fps)  # fmt: skip
        use = mapper.add("paletteuse", "dither=sierra2_4a:diff_mode=rectangle")
        sink = mapper.add("buffersink")
        frames_in.link_to(use, 0, 0)
        palette_in.link_to(use, 0, 1)
        use.link_to(sink)
        mapper.configure()
        palette_in.push(palette)
        palette_in.push(None)
        for k, rgba in enumerate(self.frames_again()):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgba[..., :3]), format="rgb24")
            frame.pts = k
            frame.time_base = 1 / self.fps
            frames_in.push(frame)
            self._drain(sink)
        frames_in.push(None)
        self._drain(sink)

    def _drain(self, sink) -> None:
        while True:
            try:
                frame = sink.pull()
            except (av.BlockingIOError, av.EOFError):
                return
            frame.pts = self._gif_pts
            frame.time_base = 1 / self.fps
            self._gif_pts += 1
            self._encode(frame)

    def _remove_part(self) -> None:
        if self.part.is_dir():
            shutil.rmtree(self.part, ignore_errors=True)
        else:
            self.part.unlink(missing_ok=True)


def _opaque(rgba: np.ndarray) -> np.ndarray:
    if rgba[..., 3].min() == 255:
        return rgba
    out = rgba.copy()
    out[..., 3] = 255
    return out


def _graph(size, fps, description: str) -> av.filter.Graph:
    graph = av.filter.Graph()
    source = graph.add_buffer(width=size[0], height=size[1], format="rgb24", time_base=1 / fps)
    node = graph.add(*description.split("=", 1))
    sink = graph.add("buffersink")
    source.link_to(node)
    node.link_to(sink)
    graph.configure()
    return graph

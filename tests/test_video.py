"""Animation video export: planning, checks, encoding every format, GPU rendering."""

import math

import av
import numpy as np
import pytest
from PIL import Image
from test_animation import grid_scene  # noqa: E402  (tests directory is on sys.path)

from skitter.core.animation import PhasedTimeline
from skitter.core.animation.assemble import AssembleChoreography
from skitter.core.animation.encode import VideoWriter
from skitter.core.animation.video import (
    FORMATS,
    VideoSettings,
    check_video,
    output_size,
    plan_video,
    view_rect,
)
from skitter.core.slicing.params import ColorParam

# Planning


def test_output_sizes_follow_presets_and_round_for_chroma():
    scene = grid_scene()  # 60 x 40 canvas
    assert output_size(VideoSettings(resolution="vertical"), scene) == (1080, 1920)
    odd = dict(resolution="custom", width=1001, height=501)
    assert output_size(VideoSettings(format="mp4", **odd), scene) == (1000, 500)  # 4:2:0
    assert output_size(VideoSettings(format="webp", **odd), scene) == (1001, 501)
    assert output_size(VideoSettings(resolution="mosaic", width=600), scene) == (600, 400)


@pytest.mark.parametrize("framing", ["fit", "fill"])
def test_view_matches_the_frame_shape(framing):
    scene = grid_scene()  # 60 x 40
    settings = VideoSettings(framing=framing, margin=0.0)
    x, y, w, h = view_rect(scene, 1080, 1920, settings)  # tall frame
    assert w / h == pytest.approx(1080 / 1920)
    assert (x + w / 2, y + h / 2) == pytest.approx((30, 20))
    if framing == "fit":
        assert w == pytest.approx(60)  # whole width shows, background above and below
    else:
        assert h == pytest.approx(40)  # whole height shows, sides cropped
    margin = view_rect(scene, 1080, 1920, VideoSettings(framing=framing, margin=10.0))
    assert margin[2] == pytest.approx(w / 0.8)


def test_frame_times_end_exactly_on_the_last_moment():
    settings = VideoSettings(frame_rate="30", motion_blur=8)
    plan = plan_video(grid_scene(), 7.0, settings, "#000000")  # the video is the animation
    assert plan.frames == 7 * 30 + 1 and plan.fps == 30
    assert plan.clock[30] == pytest.approx(1.0) and plan.clock[-1] == 7.0
    blur = plan.moments(75)
    assert len(blur) == 8 and blur.max() - blur.min() == pytest.approx(0.5 / 30 * 7 / 8)
    assert plan.moments(plan.frames - 1).tolist() == [7.0]  # exactly the last moment
    assert plan.video_clock.phase_span("build") == (0.0, 7.0)  # no phases given: all build
    custom = VideoSettings(frame_rate="custom", custom_fps=29.97)
    assert plan_video(grid_scene(), 1.0, custom, "#000000").fps == pytest.approx(30000 / 1001)


def test_checks_explain_what_blocks_an_export(tmp_path):
    scene = grid_scene()
    assert check_video(scene, VideoSettings(), "#000000", tmp_path / "a.mp4") == []
    assert check_video(None, VideoSettings(), "#000000")
    transparent = check_video(scene, VideoSettings(format="mp4"), ColorParam.TRANSPARENT)
    assert "transparent" in transparent[0] and "WebM" in transparent[0]
    assert check_video(scene, VideoSettings(format="webm"), ColorParam.TRANSPARENT) == []
    gif = VideoSettings(format="gif", frame_rate="60")
    assert "50 frames" in check_video(scene, gif, "#000000")[0]
    big = VideoSettings(resolution="custom", width=3840, height=3840)
    assert "4K" in check_video(scene, big, "#000000")[0]
    assert "does not exist" in check_video(scene, VideoSettings(), "#000000",
                                           tmp_path / "nope" / "a.mp4")[0]  # fmt: skip
    folder = tmp_path / "frames"
    folder.mkdir()
    (folder / "x.txt").write_text("x")
    png = VideoSettings(format="png")
    assert "not an empty folder" in check_video(scene, png, "#000000", folder)[0]


def test_color_param():
    p = ColorParam("#112233", "C", allow_transparent=True)
    assert p.validate("#AABBCC") == "#aabbcc" and p.validate("transparent") == "transparent"
    for bad in ("red", "#12345", 5):
        with pytest.raises(ValueError):
            p.validate(bad)
    with pytest.raises(ValueError):
        ColorParam("#000000", "C").validate("transparent")
    assert ColorParam.rgb("#ff0000") == (1.0, 0.0, 0.0) and ColorParam.rgb("transparent") is None


# Encoding


W, H, N = 96, 64, 12


def frames():
    for k in range(N):
        f = np.zeros((H, W, 4), np.uint8)
        f[..., 0] = np.linspace(0, 255, W)[None, :]
        f[..., 1] = 90
        f[..., 2] = 180
        f[..., 3] = 255
        f[20:40, 4 + 6 * k : 24 + 6 * k] = (250, 250, 40, 255)
        f[:10, :, 3] = 0  # a transparent band at the top
        yield f


def write_all(path, fid, alpha=True, **settings):
    fmt = FORMATS[fid]
    s = VideoSettings(format=fid, **settings)
    writer = VideoWriter(path, fmt, s, W, H, s.fps, alpha=alpha, frames_again=frames)
    for f in frames():
        writer.write(f)
    return writer.close()


def decode(path, fid):
    """(frame count, first frame RGBA, frame rate) read back from the file."""
    if fid in ("webp", "gif"):  # FFmpeg can't read animated WebP; Pillow reads both
        with Image.open(path) as im:
            im.seek(0)
            return im.n_frames, np.asarray(im.convert("RGBA")), None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = av.CodecContext.create("libvpx-vp9", "r") if fid == "webm" else None
        decoded = []
        for packet in container.demux(stream):
            decoded += list(codec.decode(packet) if codec else packet.decode())
        return len(decoded), decoded[0].to_ndarray(format="rgba"), stream.average_rate


@pytest.mark.parametrize("fid", [f for f in FORMATS if f != "png"])
def test_every_video_format_round_trips(tmp_path, fid):
    fmt = FORMATS[fid]
    path = write_all(tmp_path / f"out{fmt.extension}", fid)
    assert path.exists() and not (tmp_path / f"out{fmt.extension}.part").exists()
    count, first, rate = decode(path, fid)
    assert count == N and first.shape == (H, W, 4)
    if rate is not None:
        assert rate == 30
    # Colors survive (lossy formats within a few levels; GIF within its palette).
    tolerance = 40 if fid == "gif" else 12
    assert np.abs(first[50, 48, :3].astype(int) - (round(48 / 95 * 255), 90, 180)).max() < tolerance
    if fmt.alpha:
        assert first[2, 48, 3] < 10 and first[50, 48, 3] > 245  # transparency kept
    else:
        assert first[2, 48, 3] == 255


def test_png_sequence_writes_a_new_folder(tmp_path):
    path = write_all(tmp_path / "frames", "png")
    files = sorted(path.iterdir())
    assert len(files) == N and files[0].name == "frames_00000.png"
    assert not (tmp_path / "frames.part").exists()
    with Image.open(files[0]) as im:
        assert im.mode == "RGBA" and im.getpixel((48, 2))[3] == 0
    opaque = write_all(tmp_path / "opaque", "png", alpha=False)
    with Image.open(next(opaque.iterdir())) as im:
        assert im.mode == "RGB"


def test_mp4_is_tagged_bt709_and_streamable(tmp_path):
    path = write_all(tmp_path / "out.mp4", "mp4")
    with av.open(str(path)) as container:
        ctx = container.streams.video[0].codec_context
        assert (ctx.color_primaries, ctx.color_trc, ctx.colorspace) == (1, 1, 1)
    data = path.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")  # fast start: index first


def test_abort_and_errors_leave_nothing(tmp_path):
    fmt = FORMATS["mp4"]
    s = VideoSettings()
    writer = VideoWriter(tmp_path / "a.mp4", fmt, s, W, H, s.fps, alpha=False)
    writer.write(next(frames()))
    writer.abort()
    assert list(tmp_path.iterdir()) == []
    writer = VideoWriter(tmp_path / "b.mp4", fmt, s, W, H, s.fps, alpha=False)
    with pytest.raises(ValueError):
        writer.write(np.zeros((10, 10, 4), np.uint8))  # wrong size
    assert list(tmp_path.iterdir()) == []


# GPU rendering


@pytest.fixture(scope="module")
def gpu():
    import moderngl

    try:
        moderngl.create_standalone_context(require=330).release()
    except Exception as exc:  # no OpenGL 3.3 here (headless CI)
        pytest.skip(f"no GPU context: {exc}")


def render_setup(tmp_path, background="#000000", holds=(0.0, 0.0), **settings):
    """A 2 x 2 mosaic built in 1 s (holds: the build's, before and after), its plan,
    timeline and textures."""
    from test_assembly import context, result_for, tile_files

    from skitter.core.scene import MosaicScene
    from skitter.core.slicing import RegionSet
    from skitter.ui.render.tile_textures import DetailRequest, scene_instances

    tiles = []
    for i in range(4):
        y, x = np.mgrid[0:40, 0:40]
        tiles.append(np.stack([x * 6, y * 6, np.full_like(x, 60 * i)], -1).astype(np.uint8))
    files = tile_files(tmp_path, tiles)
    regions = RegionSet.grid(80, 80, 2, 2).replace(rotation=[0, 0.3, 0, 0])
    result = result_for(regions, [0, 1, 2, 3], shift=[[0.05, 0.02, 0]] * 4)
    scene = MosaicScene.from_result(result, context(80, 80))
    defaults = dict(resolution="custom", width=160, height=160, margin=0.0)
    video = VideoSettings(**{**defaults, **settings})
    build = AssembleChoreography(duration=1.0, travel=0.5).timeline(scene)
    timeline = PhasedTimeline({"build": build}, {"build": holds})
    plan = plan_video(scene, timeline.duration, video, background, timeline.spans)
    request = DetailRequest.for_scene(scene, files, scale=plan.scale)
    detail = request.build(lambda m, f: None, lambda: False)
    layer, uv = detail.atlas.locate(request.image, scene.mirrored)
    return scene, files, plan, timeline, detail.atlas.pages, scene_instances(scene, layer, uv)


def test_final_frame_matches_the_still_export(tmp_path, gpu):
    from skitter.core.assembly import ExportSettings, render_mosaic
    from skitter.ui.render.video_renderer import VideoRenderer

    scene, files, plan, timeline, pages, base = render_setup(tmp_path, motion_blur=0)
    still, _ = render_mosaic(scene, files, ExportSettings(framing="tiles", size_by="width",
                             width_px=160, background="black"), workers=0)  # fmt: skip
    renderer = VideoRenderer(plan, pages, base, (0, 0, 0), supersampling=2)
    try:
        last = renderer.render(timeline, plan.frames - 1)
        again = renderer.render(timeline, plan.frames - 1)
        first = renderer.render(timeline, 0)
    finally:
        renderer.release()
    diff = np.abs(last[..., :3].astype(int) - still.astype(int))
    assert np.median(diff) == 0 and diff.mean() < 2  # edges differ slightly (linear blending)
    assert np.array_equal(last, again)  # deterministic
    assert first[..., :3].max() == 0  # nothing has arrived yet: background only


def test_transparent_and_colored_backgrounds(tmp_path, gpu):
    from skitter.ui.render.video_renderer import VideoRenderer

    scene, _, plan, timeline, pages, base = render_setup(tmp_path, ColorParam.TRANSPARENT,
                                                         format="webm", motion_blur=0)  # fmt: skip
    assert plan.alpha
    renderer = VideoRenderer(plan, pages, base, None, supersampling=2)
    try:
        frame = renderer.render(timeline, plan.frames - 1)
    finally:
        renderer.release()
    assert frame[2, 2, 3] == 0  # the corner the rotated tile leaves empty
    assert frame[120, 40, 3] == 255  # inside a tile
    _, _, plan, timeline, pages, base = render_setup(tmp_path, "#ff0000", motion_blur=0)
    renderer = VideoRenderer(plan, pages, base, (1.0, 0.0, 0.0), supersampling=1)
    try:
        frame = renderer.render(timeline, 0)
    finally:
        renderer.release()
    assert tuple(frame[80, 80]) == (255, 0, 0, 255)


def test_motion_blur_only_blurs_motion(tmp_path, gpu):
    from skitter.ui.render.video_renderer import VideoRenderer

    _, _, sharp_plan, timeline, pages, base = render_setup(
        tmp_path, motion_blur=0, holds=(0.0, 0.5)
    )
    _, _, blur_plan, _, _, _ = render_setup(tmp_path, motion_blur=8, holds=(0.0, 0.5))
    sharp = VideoRenderer(sharp_plan, pages, base, (0, 0, 0), supersampling=1)
    blur = VideoRenderer(blur_plan, pages, base, (0, 0, 0), supersampling=1)
    try:
        last = sharp_plan.frames - 1
        assert np.array_equal(sharp.render(timeline, last), blur.render(timeline, last))
        mid = math.floor(sharp_plan.frames * 0.4)
        assert not np.array_equal(sharp.render(timeline, mid), blur.render(timeline, mid))
    finally:
        sharp.release()
        blur.release()


def test_tossed_tiles_draw_in_three_groups_with_shadows():
    from test_animation import pile_scene

    from skitter.core.animation.look import AnimationLook, TableCamera
    from skitter.ui.render.player import frame_layers
    from skitter.ui.render.sprites import make_instances

    scene = pile_scene()
    timeline = AssembleChoreography(duration=4.0, travel=1.0).timeline(scene)  # Toss
    camera = TableCamera.for_scene(scene, AnimationLook(shadow_strength=0.6))
    base = make_instances(len(scene))
    ground, shadows, air = frame_layers(base, timeline.frame(2.0), camera)
    assert len(ground) + len(air) == len(scene)
    assert len(air) and len(shadows)  # some tiles in the air cast shadows
    assert shadows["blur"].min() > 0 and shadows["alpha"].max() <= 0.6
    final = frame_layers(base, timeline.frame(timeline.duration), camera)
    assert len(final[0]) == len(scene) and len(final[1]) == len(final[2]) == 0


def test_gpu_draws_soft_shadows_under_tossed_tiles(tmp_path, gpu):
    from skitter.core.animation.look import AnimationLook, TableCamera
    from skitter.ui.render.video_renderer import VideoRenderer

    scene, _, plan, _, pages, base = render_setup(tmp_path, "#ffffff", motion_blur=0)
    timeline = AssembleChoreography(duration=1.0, travel=0.8, arc=0.6, distance=0.3).timeline(scene)
    assert timeline.duration == pytest.approx(plan.duration)  # fits the requested duration
    frames = {}
    for strength in (0.0, 0.8):
        camera = TableCamera.for_scene(scene, AnimationLook(shadow_strength=strength))
        renderer = VideoRenderer(plan, pages, base, (1.0, 1.0, 1.0), 1, camera)
        try:
            frames[strength] = renderer.render(timeline, plan.frames // 3)
            frames["final", strength] = renderer.render(timeline, plan.frames - 1)
        finally:
            renderer.release()
    darker = frames[0.0][..., :3].astype(int) - frames[0.8][..., :3].astype(int)
    assert darker.max() > 40 and darker.min() >= -2  # shadows only darken
    assert np.array_equal(frames["final", 0.0], frames["final", 0.8])  # at rest: no shadows

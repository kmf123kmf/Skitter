"""Matching methods (the registry, what every method promises) and the Jigsaw Puzzle method."""

import json

import numpy as np
import pytest
from PIL import Image
from test_matching_steps import matching_step, sliced, window  # noqa: F401 (fixtures)

from skitter.core.assembly import ExportSettings, render_picture
from skitter.core.color import rgb8_to_oklab
from skitter.core.matching.jigsaw import JigsawMethod
from skitter.core.matching.matcher import Matcher
from skitter.core.matching.method import (
    MatchMethod,
    get_method,
    method_types,
    register_method,
    unregister_method,
)
from skitter.core.matching.mosaic import MosaicMethod
from skitter.core.matching.saved import SavedMosaic, restore, saved_mosaic
from skitter.core.scene import MosaicScene
from skitter.core.slicing import MosaicLayout, RegionSet, SliceContext
from skitter.core.slicing.operations import GridSlicer
from skitter.core.tiles.library import TileLibrary


def noise(h, w, seed=0, block=1):
    """(h, w, 3) uint8 random colors in block x block squares (every misplaced pixel shows)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (-(-h // block), -(-w // block), 3), dtype=np.uint8)
    return np.repeat(np.repeat(small, block, 0), block, 1)[:h, :w].copy()


def cut(image, columns=8, regions=None):
    """(ctx, regions, result): image cut into a grid of columns (or the given regions)."""
    ctx = SliceContext(image, MosaicLayout(columns=columns))
    regions = GridSlicer().apply(ctx.canvas(), ctx) if regions is None else regions
    return ctx, regions, Matcher(None).run(regions, ctx, JigsawMethod())


def export(result, ctx, **settings):
    scene = MosaicScene.from_result(result, ctx)
    image, report = render_picture(scene, ctx.picture, ExportSettings(**settings))
    return image, report


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiles")
    rng = np.random.default_rng(5)
    for i, color in enumerate(rng.integers(0, 256, (40, 3))):
        size = [(60, 60), (90, 60), (60, 90)][i % 3]
        Image.new("RGB", size, tuple(int(c) for c in color)).save(root / f"t{i}.png")
    lib = TileLibrary(tmp_path_factory.mktemp("cache"))
    lib.set_roots([root])
    lib.update(workers=0)
    yield lib
    lib.close()


# The registry


def test_methods_are_registered_the_default_first():
    types = method_types()
    assert types[0] is MosaicMethod and JigsawMethod in types
    assert len({cls.id for cls in types}) == len({cls.name for cls in types}) == len(types)
    assert get_method("jigsaw") is JigsawMethod and all(cls.description for cls in types)
    with pytest.raises(KeyError):
        get_method("telepathy")


def test_a_method_id_is_taken_once():
    class Impostor(MatchMethod):
        id, name = "mosaic", "Impostor"

    with pytest.raises(ValueError):
        register_method(Impostor)

    class Spare(MatchMethod):
        id, name = "spare", "Spare"

    register_method(Spare)
    try:
        assert Spare in method_types()
    finally:
        unregister_method("spare")
    assert Spare not in method_types()


def test_settings_keys_tell_methods_apart():
    assert MosaicMethod().key()[0] == "mosaic" and JigsawMethod().key() == ("jigsaw",)
    assert MosaicMethod().key() != MosaicMethod(max_uses=9).key()


def test_a_method_that_needs_the_library_refuses_to_run_without_one():
    ctx = SliceContext(noise(30, 40), MosaicLayout(columns=4))
    with pytest.raises(ValueError):
        Matcher(None).run(ctx.canvas(), ctx, MosaicMethod())


@pytest.mark.parametrize("cls", method_types(), ids=lambda cls: cls.id)
def test_every_method_fills_the_regions_and_round_trips(cls, library):
    """What every method promises: each region that shows gets something, the result
    makes a scene, and a project file brings it back."""
    image = noise(60, 80, block=10)
    ctx = SliceContext(image, MosaicLayout(columns=8))
    grid = GridSlicer().apply(ctx.canvas(), ctx)  # base tiles 500 units across
    hidden = RegionSet.from_arrays([[250, 250]], [[400, 400]], [0.0], [-1.0])  # under one
    regions = RegionSet.concat([grid, hidden])
    settings = cls() if cls is not MosaicMethod else cls(refine_seconds=0.1, adaptive_rounds=0)
    result = Matcher(library).run(regions, ctx, settings)
    assert result.settings.key() == settings.key()
    assert (result.tile[:-1] >= 0).all() and result.tile[-1] == -1
    assert result.tile_mean.shape == (len(regions), 3) and result.rect.shape == (len(regions), 4)
    scene = MosaicScene.from_result(result, ctx)
    assert len(scene) == len(regions) - 1 and scene.picture == result.picture

    saved = saved_mosaic(result, library)
    again = SavedMosaic.from_parts(saved.arrays(), json.loads(json.dumps(saved.meta())))
    assert type(again.settings) is cls
    restored, missing = restore(again, library, regions, ctx)
    assert missing == 0 and restored.picture == result.picture
    np.testing.assert_array_equal(restored.tile, result.tile)
    np.testing.assert_allclose(restored.rect, result.rect)


# Jigsaw Puzzle


def test_pieces_are_the_regions_that_show_and_need_no_library():
    image = noise(60, 80)
    ctx, regions, result = cut(image)
    assert result.picture and result.candidates is None and result.quality is None
    np.testing.assert_array_equal(result.tile, np.arange(len(regions)))
    assert result.tint == 0 and np.isfinite(result.cost).all()
    assert result.stats["placed"] == len(regions) == 48
    np.testing.assert_allclose(result.rect[0], [0, 0, 1 / 8, 1 / 6], atol=1e-6)


def test_a_piece_showing_only_a_sliver_is_kept():
    from skitter.core.matching.raster import may_show

    # Bottom to top: a sliver 0.1 units wide of the first shows past the second (far
    # thinner than a raster cell); the third is wholly under the second.
    center = [[50.0, 50.0], [50.1, 50.0], [50.0, 50.0]]
    size = [[100.0, 100.0], [100.0, 100.0], [60.0, 60.0]]
    rotation = [0.0, 0.0, 0.0]
    assert may_show(center[:2], size[:2], rotation[:2]).tolist() == [True, True]
    order = [2, 0, 1]  # the small one at the bottom
    shows = may_show(np.take(center, order, 0), np.take(size, order, 0), [0.0] * 3)
    assert shows.tolist() == [False, True, True]


def test_a_photo_pile_leaves_no_gaps_in_the_picture():
    from skitter.core.slicing.operations.pile import PileSlicer

    image = noise(120, 180, block=6)
    ctx = SliceContext(image, MosaicLayout(columns=12))
    regions = PileSlicer(spread=0.8).apply(ctx.canvas(), ctx)
    ctx, regions, result = cut(image, regions=regions)
    assert (result.tile < 0).any()  # some pieces lie wholly under others
    out, _ = export(result, ctx, size_by="width", width_px=180, background="transparent")
    assert (out[..., 3] == 255).all()  # pieces showing only slivers are kept: no gaps
    np.testing.assert_array_equal(out[..., :3], image)


def test_piece_colors_are_the_pictures():
    image = np.zeros((60, 80, 3), np.uint8)
    image[:, :40] = (200, 30, 30)
    image[:, 40:] = (20, 90, 220)
    ctx, regions, result = cut(image)
    left = regions.center[:, 0] < ctx.width / 2
    np.testing.assert_allclose(result.tile_mean[left], rgb8_to_oklab(image[0, 0])[None].repeat(
        left.sum(), 0), atol=1e-3)  # fmt: skip
    np.testing.assert_allclose(result.tile_mean[~left], rgb8_to_oklab(image[0, -1])[None].repeat(
        (~left).sum(), 0), atol=1e-3)  # fmt: skip


def test_the_finished_puzzle_exports_as_the_picture_exactly():
    image = noise(60, 80)
    ctx, _, result = cut(image)
    out, report = export(result, ctx, size_by="width", width_px=80, background="black")
    assert report.tiles == 48
    np.testing.assert_array_equal(out, image)
    clear, _ = export(result, ctx, size_by="width", width_px=80, background="transparent")
    np.testing.assert_array_equal(clear[..., :3], image)
    assert (clear[..., 3] == 255).all()
    # Any other size: the picture resampled once, as a plain resize would.
    big, _ = export(result, ctx, size_by="width", width_px=200, background="black")
    plain = np.asarray(Image.fromarray(image).resize((200, 150), Image.Resampling.LANCZOS))
    np.testing.assert_array_equal(big, plain)


def test_strips_export_the_same_picture_to_a_level(monkeypatch):
    import skitter.core.assembly as assembly

    image = noise(60, 80)
    ctx, _, result = cut(image)
    whole, _ = export(result, ctx, size_by="width", width_px=240)
    monkeypatch.setattr(assembly, "PICTURE_STRIP", 240 * 16)  # strips of 16 rows
    parts, _ = export(result, ctx, size_by="width", width_px=240)
    # Each strip's resize places its rows a hair differently: now and then a level apart.
    diff = np.abs(parts.astype(int) - whole.astype(int))
    assert diff.max() <= 1 and (diff > 0).mean() < 1e-3


def test_turned_and_uneven_pieces_show_the_picture_wherever_they_cover_it():
    image = noise(60, 80, seed=3)
    grid = GridSlicer().apply(SliceContext(image, MosaicLayout(columns=8)).canvas(),
                              SliceContext(image, MosaicLayout(columns=8)))  # fmt: skip
    turned = np.where(np.arange(len(grid)) % 3 == 0, 0.4, 0.0)
    regions = grid.replace(rotation=turned, size=grid.size * 0.9)
    ctx, _, result = cut(image, regions=regions)
    out, _ = export(result, ctx, size_by="width", width_px=80, background="transparent")
    full = out[..., 3] == 255
    assert 0.5 < full.mean() < 0.95 and (out[..., 3] == 0).any()  # gaps between pieces
    np.testing.assert_array_equal(out[full][:, :3], image[full])


def test_what_hangs_past_the_picture_shows_nothing():
    image = noise(55, 80)  # 5.5 rows of base tiles: the grid hangs a quarter tile over
    ctx, regions, result = cut(image)  # top and bottom (2.5 px of the export each)
    out, _ = export(result, ctx, framing="tiles", size_by="width", width_px=80,
                    background="transparent")  # fmt: skip
    assert out.shape[0] == 60
    # Rows wholly on the picture show it (half a pixel down); rows past it show nothing,
    # nor do the two half on it (the picture's edge falls mid-pixel).
    inside = np.asarray(Image.fromarray(image).resize((80, 54), Image.Resampling.LANCZOS,
                                                      box=(0, 0.5, 80, 54.5)))  # fmt: skip
    np.testing.assert_array_equal(out[3:57, :, :3], inside)
    assert (out[3:57, :, 3] == 255).all() and (out[:3, :, 3] == 0).all()
    assert (out[57:, :, 3] == 0).all()


def test_masked_pixels_are_transparent_pieces():
    image = np.concatenate([noise(60, 80), np.full((60, 80, 1), 255, np.uint8)], axis=2)
    image[:30, :40, 3] = 0  # a hidden corner
    ctx, regions, result = cut(image)
    assert (ctx.picture[:30, :40, 3] == 0).all() and (ctx.picture[30:, :, 3] == 255).all()
    out, _ = export(result, ctx, size_by="width", width_px=80, background="transparent")
    np.testing.assert_array_equal(out[..., 3], ctx.picture[..., 3])
    np.testing.assert_array_equal(out[30:, :, :3], image[30:, :, :3])


def test_each_piece_shows_its_own_footprint_on_the_picture():
    image = noise(60, 80)
    grid = GridSlicer().apply(SliceContext(image, MosaicLayout(columns=4)).canvas(),
                              SliceContext(image, MosaicLayout(columns=4)))  # fmt: skip
    regions = grid.replace(rotation=np.linspace(-1, 2, len(grid)))
    ctx, _, result = cut(image, regions=regions)
    scene = MosaicScene.from_result(result, ctx)
    uv, skew = scene.picture_uv()
    canvas = np.array(scene.canvas)
    for corner, (s, t) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):  # as scene.corners
        u = uv[:, 0] + s * (uv[:, 2] - uv[:, 0]) + t * skew[:, 0]
        v = uv[:, 1] + t * (uv[:, 3] - uv[:, 1]) + s * skew[:, 1]
        np.testing.assert_allclose(np.column_stack([u, v]), scene.corners[:, corner] / canvas,
                                   atol=1e-5)  # fmt: skip


# On the GPU


@pytest.fixture(scope="module")
def gpu():
    import moderngl

    try:
        moderngl.create_standalone_context(require=330).release()
    except Exception as exc:  # no OpenGL 3.3 here (headless CI)
        pytest.skip(f"no GPU context: {exc}")


def gpu_frame(image, regions=None, columns=10):
    """The last frame of a 1 s build of image cut into pieces, rendered at its size."""
    from skitter.core.animation import PhasedTimeline
    from skitter.core.animation.assemble import AssembleChoreography
    from skitter.core.animation.video import VideoSettings, plan_video
    from skitter.ui.render.tile_textures import PictureRequest
    from skitter.ui.render.video_renderer import VideoRenderer

    ctx, regions, result = cut(image, columns, regions)
    scene = MosaicScene.from_result(result, ctx)
    h, w = image.shape[:2]
    video = VideoSettings(resolution="custom", width=w, height=h, margin=0.0, motion_blur=0)
    timeline = PhasedTimeline({"build": AssembleChoreography(duration=1.0).timeline(scene)},
                              {"build": (0.0, 0.0)})  # fmt: skip
    plan = plan_video(scene, timeline.duration, video, "#000000", timeline.spans)
    pages, base, scale, failed = PictureRequest(ctx.picture).content(scene, None, None)
    assert scale == 1 and failed == 0 and pages.shape == (1, h, w, 4)
    renderer = VideoRenderer(plan, pages, base, (0, 0, 0), supersampling=2)
    try:
        return renderer.render(timeline, plan.frames - 1)[..., :3], regions
    finally:
        renderer.release()


def test_the_gpu_draws_the_finished_puzzle_as_the_picture(gpu):
    image = noise(64, 80, seed=1, block=8)
    frame, _ = gpu_frame(image)  # 8 px pieces: 10 x 8, filling the picture
    diff = np.abs(frame.astype(int) - image.astype(int)).max(axis=2)
    # In place, to the pixel; supersampled filtering blends a quarter texel across
    # each edge of the 8 px blocks (as for photos), and the rest is exact.
    inner = np.ones(8, bool)
    inner[[0, -1]] = False
    inside = np.outer(np.tile(inner, 8), np.tile(inner, 10))
    assert diff[inside].max() <= 1 and 0 < diff[~inside].mean() < 40


def test_the_gpu_draws_turned_pieces_where_they_lie(gpu):
    y, x = np.mgrid[0:64, 0:80]
    image = np.stack([x * 3, y * 3, np.full_like(x, 128)], -1).astype(np.uint8)  # filtering
    ctx = SliceContext(image, MosaicLayout(columns=10))  # keeps a gradient as it is
    grid = GridSlicer().apply(ctx.canvas(), ctx)
    turned = grid.replace(rotation=np.full(len(grid), 0.5), size=grid.size * 0.7)
    frame, _ = gpu_frame(image, turned)
    covered = frame[..., 2] > 64  # blue 128 on the pieces, black between them
    interior = covered.copy()
    for axis in (0, 1):
        for step in (-1, 1):
            interior &= np.roll(covered, step, axis)
    assert 0.2 < interior.mean() < covered.mean() < 0.75
    diff = np.abs(frame.astype(int) - image.astype(int)).max(axis=2)
    assert diff[interior].max() <= 2  # a piece showing the picture anywhere else would not


# In the app


def test_the_jigsaw_runs_exports_and_saves_without_a_library(sliced, tmp_path, qapp):  # noqa: F811
    from skitter.ui.render.tile_textures import PictureTextures

    app = sliced
    session = app.session
    app.next_button.click()  # to Tiles (the library is empty)
    app.next_button.click()  # to Matching
    step = matching_step(app)
    assert not session.can_match  # Mosaic needs photos
    step.method_box.setCurrentIndex(step.method_box.findData("jigsaw"))
    assert session.project.match_method_id == "jigsaw" and session.can_match
    assert "picture" in step.method_description.text()
    assert step.settings_group.isHidden()  # the Jigsaw has no settings
    step.run_matching()
    session.wait_for_job()
    qapp.processEvents()
    assert session.mosaic_is_valid and session.matching_is_current and session.can_export
    assert isinstance(session.textures, PictureTextures)
    assert step._labels["tiles"].text() == f"{len(session.project.regions):,} pieces of the picture"
    assert not session.can_pick(0)

    path = tmp_path / "out.png"
    session.start_export(path, ExportSettings(size_by="width", width_px=40))
    session.wait_for_job()
    picture = session.slice_context.picture[..., :3]
    np.testing.assert_array_equal(np.asarray(Image.open(path)), picture)

    session.project.video_settings.update(resolution="custom", width=80, height=60,
                                          frame_rate="custom", custom_fps=10, motion_blur=0,
                                          format="webm")  # fmt: skip
    session.project.choreography.update(duration=1.0, travel=0.5)
    session.animation_edited()
    finished = []
    session.video_finished.connect(lambda *args: finished.append(args))
    session.start_video_export(tmp_path / "puzzle.webm")
    session.wait_for_job()
    assert finished and finished[0][2] is None and (tmp_path / "puzzle.webm").exists()

    project = tmp_path / "puzzle.skitter"
    assert app._save_to(project)
    assert app.new_project()
    session.open_project(project)
    qapp.processEvents()
    assert session.project.match_method_id == "jigsaw" and session.project.matches.picture
    assert session.mosaic_is_valid and session.matching_is_current
    assert step.method_box.currentData() == "jigsaw"

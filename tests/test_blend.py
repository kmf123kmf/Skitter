"""Blending the picture into the matched tiles (core/blend.py), shown and exported."""

from dataclasses import replace

import numpy as np
import pytest
from test_assembly import result_for, tile_files
from test_matching_steps import photos, sliced, window  # noqa: F401 (fixtures)

from skitter.core.assembly import ExportSettings, render_mosaic
from skitter.core.blend import BlendSettings, tile_blend
from skitter.core.color import oklab_to_rgb8, rgb8_to_oklab
from skitter.core.scene import MosaicScene
from skitter.core.slicing import MosaicLayout, RegionSet, SliceContext

TILE = np.array([40, 90, 200], np.uint8)
PICTURE = np.array([230, 180, 30], np.uint8)


def test_weak_matches_blend_just_enough_to_reach_the_target():
    error = np.array([3.0, 6.0, 12.0, 60.0, np.nan])
    weak = BlendSettings(amount=50.0, target=6.0)
    # Within the target: pure; 12 needs half to come down to 6; 60 would need 90%: capped.
    np.testing.assert_allclose(tile_blend(error, weak), [0, 0, 0.5, 0.5, 0])
    np.testing.assert_allclose(tile_blend(error, BlendSettings(amount=100.0, target=6.0)),
                               [0, 0, 0.5, 0.9, 0])  # fmt: skip
    everywhere = BlendSettings(amount=30.0, where="all")
    np.testing.assert_allclose(tile_blend(error, everywhere), 0.3)
    assert not tile_blend(error, BlendSettings()).any()  # 0%: off
    assert not tile_blend(error, None).any()


def flat_scene(tmp_path, blend: BlendSettings, picture=PICTURE, size=16):
    """A one-tile mosaic of a flat TILE color over a flat picture."""
    image = np.tile(picture, (size, size, 1))
    ctx = SliceContext(image, MosaicLayout(1.0, size), 1.0)
    regions = RegionSet.grid(size, size, 1, 1)
    files = tile_files(tmp_path, [np.tile(TILE, (size, size, 1))])
    scene = MosaicScene.from_result(result_for(regions, [0]), ctx, blend)
    return scene, files, ctx


def export(scene, files, ctx, **settings):
    settings.setdefault("tile_px", 1)
    image, _ = render_mosaic(scene, files, ExportSettings(**settings), workers=0,
                             picture=ctx.picture)  # fmt: skip
    return image


def test_export_mixes_in_the_picture_in_oklab(tmp_path):
    scene, files, ctx = flat_scene(tmp_path, BlendSettings(amount=100.0, where="all"))
    assert scene.blended
    assert np.abs(export(scene, files, ctx).astype(int) - PICTURE).max() <= 1  # all picture
    scene = scene.with_blend(BlendSettings(amount=40.0, where="all"))
    want = oklab_to_rgb8(0.6 * rgb8_to_oklab(TILE) + 0.4 * rgb8_to_oklab(PICTURE))
    assert np.abs(export(scene, files, ctx).astype(int) - want).max() <= 1
    scene = scene.with_blend(BlendSettings())  # off: the photo alone
    assert not scene.blended
    assert (export(scene, files, ctx) == TILE).all()
    # Without the picture nothing blends (as for callers that have none).
    image, _ = render_mosaic(scene.with_blend(BlendSettings(amount=100.0, where="all")), files,
                             ExportSettings(tile_px=1), workers=0)  # fmt: skip
    assert (image == TILE).all()


def test_tiles_hanging_past_the_picture_blend_only_over_it(tmp_path):
    image = np.tile(PICTURE, (16, 16, 1))
    ctx = SliceContext(image, MosaicLayout(1.0, 16), 1.0)
    regions = RegionSet.from_rects([8.0], [0.0], [16.0], [16.0])  # half past the right edge
    files = tile_files(tmp_path, [np.tile(TILE, (16, 16, 1))])
    scene = MosaicScene.from_result(result_for(regions, [0]), ctx,
                                    BlendSettings(amount=100.0, where="all"))  # fmt: skip
    out = export(scene, files, ctx, framing="tiles")  # x 8 to 24 of the mosaic
    assert out.shape[:2] == (16, 16)
    assert np.abs(out[:, 1:7].astype(int) - PICTURE).max() <= 1  # over the picture
    assert (out[:, 9:] == TILE).all()  # past it: the photo


def test_blending_keeps_the_scene_and_never_touches_pieces_of_the_picture(tmp_path):
    scene, _, _ = flat_scene(tmp_path, BlendSettings())
    corners = scene.corners  # cached
    again = scene.with_blend(BlendSettings(amount=50.0, where="all"))
    assert again.corners is corners and again.result is scene.result
    np.testing.assert_allclose(again.blends, 0.5)
    ctx = SliceContext(np.zeros((16, 16, 3), np.uint8), MosaicLayout(1.0, 16), 1.0)
    picture = MosaicScene.from_result(replace(scene.result, picture=True), ctx,
                                      BlendSettings(amount=50.0, where="all"))  # fmt: skip
    assert not picture.blended and not picture.with_blend(BlendSettings(amount=80.0)).blended


@pytest.fixture(scope="module")
def gpu():
    import moderngl

    try:
        moderngl.create_standalone_context(require=330).release()
    except Exception as exc:  # no OpenGL 3.3 here (headless CI)
        pytest.skip(f"no GPU context: {exc}")


def gpu_frame(scene, ctx):
    """The finished mosaic drawn by the video renderer at one pixel per unit."""
    from skitter.core.animation import PhasedTimeline
    from skitter.core.animation.assemble import AssembleChoreography
    from skitter.core.animation.video import VideoSettings, plan_video
    from skitter.ui.render.tile_textures import blend_page, scene_instances
    from skitter.ui.render.video_renderer import VideoRenderer

    w, h = (int(v) for v in scene.canvas)
    pages = np.tile(np.append(TILE, 255).astype(np.uint8), (1, 8, 8, 1))
    base = scene_instances(scene, np.zeros(len(scene)), np.tile([0, 0, 1, 1], (len(scene), 1)))
    video = VideoSettings(resolution="custom", width=w, height=h, margin=0.0, motion_blur=0)
    timeline = PhasedTimeline({"build": AssembleChoreography(duration=1.0).timeline(scene)},
                              {"build": (0.0, 0.0)})  # fmt: skip
    plan = plan_video(scene, timeline.duration, video, "#000000", timeline.spans)
    renderer = VideoRenderer(plan, pages, base, (0, 0, 0), supersampling=1,
                             picture=blend_page(scene, ctx.picture))  # fmt: skip
    try:
        return renderer.render(timeline, plan.frames - 1)[..., :3]
    finally:
        renderer.release()


def test_the_gpu_blends_like_the_export(gpu, tmp_path):
    scene, files, ctx = flat_scene(tmp_path, BlendSettings(amount=40.0, where="all"), size=32)
    frame = gpu_frame(scene, ctx)
    want = export(scene, files, ctx)
    assert np.abs(frame[2:-2, 2:-2].astype(int) - want[2:-2, 2:-2]).max() <= 2
    assert np.abs(frame[2:-2, 2:-2].astype(int) - TILE).min() > 10  # visibly blended


def test_the_blend_travels_with_the_tile():
    """Two tiles over a picture half red, half blue: each mixes in its own home half,
    wherever an animation puts it."""
    from skitter.core.animation import TileFrame
    from skitter.ui.render.player import frame_layers
    from skitter.ui.render.tile_textures import scene_instances

    image = np.zeros((16, 32, 3), np.uint8)
    image[:, :16] = (255, 0, 0)
    image[:, 16:] = (0, 0, 255)
    ctx = SliceContext(image, MosaicLayout(1.0, 32), 1.0)
    regions = RegionSet.grid(32, 16, 2, 1)
    scene = MosaicScene.from_result(result_for(regions, [0, 0]), ctx,
                                    BlendSettings(amount=100.0, where="all"))  # fmt: skip
    base = scene_instances(scene, np.zeros(2), np.tile([0, 0, 1, 1], (2, 1)))
    assert base["blend"].tolist() == [1.0, 1.0]
    left, right = base["picture_uv"]
    assert left[0] == 0.0 and left[2] == pytest.approx(0.5) and right[0] == pytest.approx(0.5)
    final = TileFrame.final(scene)
    swapped = final.replace(center=final.center[::-1].copy())  # each where the other belongs
    tiles = frame_layers(base, swapped, None)[0]
    np.testing.assert_array_equal(tiles["pos"], scene.center[::-1])
    np.testing.assert_array_equal(tiles["picture_uv"], base["picture_uv"])
    np.testing.assert_array_equal(tiles["blend"], base["blend"])


def test_blend_settings_round_trip_in_project_files(tmp_path):
    from test_project_file import sample_project

    from skitter.core.project_file import document, load_project, save_project

    project = sample_project()
    project.blend.update(amount=35.0, where="weak", target=4.5)
    path = tmp_path / "blend.skitter"
    save_project(path, project, committed=True, mosaic=None)
    loaded = load_project(path)
    assert loaded.problems == [] and document(loaded.project, True) == document(project, True)
    assert loaded.project.blend.values() == {"amount": 35.0, "where": "weak", "target": 4.5}


def test_matching_tab_blends_the_mosaic_without_matching_again(sliced, photos, qapp):  # noqa: F811
    from test_matching_steps import matched

    session = sliced.session
    step = matched(sliced, photos, qapp)
    tiles = session.project.matches.tile.copy()
    assert step._tile_layer.picture is None and not step._blend_summary.isVisible()
    changes = []
    session.blend_changed.connect(lambda: changes.append(session.scene))
    step.blend_form.editor("where").widget.setCurrentIndex(1)  # every tile
    assert not session.scene.blended  # still 0%
    step.blend_form.editor("amount").widget.setValue(30)
    assert changes and session.scene.blended and session.matching_is_current
    np.testing.assert_array_equal(session.project.matches.tile, tiles)  # the same tiles
    np.testing.assert_allclose(session.scene.blends, 0.3)
    layer = step._tile_layer
    assert layer.picture is session.textures.picture is not None
    np.testing.assert_allclose(layer.instances["blend"], 0.3)
    assert "tiles blend" in step._blend_summary.text() and session.modified
    step.blend_form.editor("amount").widget.setValue(0)
    assert not session.scene.blended and layer.picture is None
    assert not layer.instances["blend"].any()

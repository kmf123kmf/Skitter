"""Zoom detail: tiles read again at the screen's resolution where a view zooms in."""

import math

import numpy as np
import pytest
from PIL import Image
from test_matching_steps import matched, sliced, window  # noqa: F401 (fixtures)

from skitter.ui.render import zoom_detail
from skitter.ui.render.atlas import texel_uv
from skitter.ui.render.zoom_detail import (
    DENSITY_STEP,
    crop_pixels,
    detail_densities,
    step_up,
)


def test_densities_step_up_from_the_zoom_up_to_the_photos_own():
    assert step_up(1.0) == pytest.approx(1.0)
    assert step_up(1.1) == pytest.approx(DENSITY_STEP)
    assert step_up(3.0) == pytest.approx(DENSITY_STEP**4)
    sizes = np.full((3, 2), 100.0)
    main = np.array([1.0, 1.0, 4.0])  # the last already has 4 texels per unit
    native = np.array([10.0, 1.5, 10.0])  # the second's photo has only 1.5
    density = detail_densities(3.0, main, native, sizes, budget=1e9)
    np.testing.assert_allclose(density, [4.0, 1.5, 0.0])
    # Zoomed out, the usual textures are enough.
    assert not detail_densities(0.5, main, native, sizes, budget=1e9).any()


def test_densities_step_down_alike_to_fit_the_budget():
    sizes = np.full((10, 2), 100.0)
    main, native = np.ones(10), np.full(10, 100.0)
    density = detail_densities(8.0, main, native, sizes, budget=10 * 300**2)
    assert len(set(density.tolist())) == 1 and density[0] < 8.0
    assert np.prod(crop_pixels(sizes, density), axis=1).sum() <= 10 * 300**2
    steps = math.log(density[0], DENSITY_STEP)
    assert steps == pytest.approx(round(steps))  # a step of the zoom's
    # Crops never exceed a page.
    assert crop_pixels([[100.0, 50.0]], [100.0], side=1024).tolist() == [[1024, 512]]


def test_the_reader_drops_what_leaves_the_view_before_reading_it(qapp, monkeypatch):
    import threading

    started, release, read = threading.Event(), threading.Event(), []

    def slow(paths, rects, sizes, **kwargs):
        started.set()
        release.wait(5)
        read.append(paths[0])
        return [np.zeros((2, 2, 3), np.uint8)]

    monkeypatch.setattr(zoom_detail, "render_crops", slow)
    reader = zoom_detail._Reader(threads=1)
    arrived = []
    reader.read.connect(lambda key, image: arrived.append(key))
    item = lambda name: ((name,), name, (0, 0, 1, 1), (2, 2))  # noqa: E731
    try:
        reader.ask([item("a"), item("b"), item("c")])  # most urgent first
        assert started.wait(5)
        reader.ask([item("a"), item("d")])  # the view moved: a is being read already
        release.set()
        reader.wait(5)
        qapp.processEvents()
    finally:
        reader.close()
    assert read == ["a", "d"] and arrived == [("a",), ("d",)]


@pytest.fixture
def big_photos(tmp_path):
    """Photos with detail everywhere, large enough to show far beyond a tile's texture."""
    folder = tmp_path / "big"
    folder.mkdir()
    rng = np.random.default_rng(3)
    for i in range(6):
        image = rng.integers(0, 256, (1000, 1500, 3), np.uint8)
        image[..., i % 3] //= 2  # colors apart, so different tiles show
        Image.fromarray(image).save(folder / f"b{i}.jpg", quality=90)
    return folder


@pytest.fixture
def reads(monkeypatch):
    """Every crop the zoom detail reads, as (path, (w, h))."""
    calls = []
    original = zoom_detail.render_crops

    def counted(paths, rects, sizes, **kwargs):
        calls.extend((p, tuple(int(v) for v in s)) for p, s in zip(paths, sizes, strict=True))
        return original(paths, rects, sizes, **kwargs)

    monkeypatch.setattr(zoom_detail, "render_crops", counted)
    return calls


def look(step, center, zoom: float) -> None:
    """Point the (unshown) canvas at a world point, at zoom device pixels per unit."""
    canvas = step.viewer.canvas
    canvas.camera.viewport[:] = (800, 600)
    canvas.set_view(center, zoom / canvas.devicePixelRatioF())


def assert_consistent(step) -> None:
    """Each sharpened tile shows its crop where the crop lies on the pages, and crops on
    a page never overlap."""
    zoom, layer = step.zoom_detail, step._tile_layer
    crops = {c.id: c for c in zoom._crops.values()}
    for tile in np.flatnonzero(zoom._crop_of >= 0):
        crop = crops[int(zoom._crop_of[tile])]
        assert crop.page >= 0
        assert layer.instances["layer"][tile] == layer.first_detail + crop.page
        h, w = crop.image.shape[:2]
        uv = texel_uv(np.array([[crop.x, crop.y]], float), np.array([[w, h]], float),
                      layer.detail_side, step._shown.mirrored[[tile]])  # fmt: skip
        np.testing.assert_allclose(layer.instances["uv"][tile], uv[0])
    placed = [c for c in zoom._crops.values() if c.page >= 0]
    for i, a in enumerate(placed):
        for b in placed[i + 1 :]:
            if a.page == b.page:
                (ah, aw), (bh, bw) = a.image.shape[:2], b.image.shape[:2]
                apart = (a.x + aw <= b.x or b.x + bw <= a.x or a.y + ah <= b.y
                         or b.y + bh <= a.y)  # fmt: skip
                assert apart


def test_zooming_in_reads_tiles_at_the_screens_resolution(sliced, big_photos, qapp, reads):  # noqa: F811
    from skitter.core.tiles.render import render_crops

    step = matched(sliced, big_photos, qapp)
    zoom, layer, scene = step.zoom_detail, step._tile_layer, step._shown
    usual = layer.instances.copy()
    assert zoom.sharpened == 0 and layer.detail_pages
    reads.clear()

    # 2 device pixels per unit: a 500-unit tile wants 1000 texels across; its photo
    # (a 1000 px square cut from 1500 x 1000) has just that.
    tile = 9
    assert scene.size[tile].tolist() == [500.0, 500.0]
    look(step, scene.center[tile], 2.0)
    zoom.wait()
    assert zoom.sharpened >= 1 and layer.instances["layer"][tile] >= layer.first_detail
    uv = layer.instances["uv"][tile]
    np.testing.assert_allclose(np.abs(uv[2:] - uv[:2]) * layer.detail_side + 1, [1000, 1000])
    assert all(size == (1000, 1000) for _, size in reads)
    assert len(reads) == zoom.sharpened <= 4  # the tiles in view (or nearly), each once
    assert_consistent(step)
    # What goes onto the pages is the photo's crop at that size.
    path = sliced.session.library.paths([scene.slot[tile]])[0]
    crop = next(c for c in zoom._crops.values() if c.id == zoom._crop_of[tile])
    key = zoom._key_of_id[crop.id]  # the tile's rect (to 5 decimals)
    np.testing.assert_allclose(key[1:5], scene.rect[tile], atol=1e-5)
    want = render_crops([path], [key[1:5]], [(1000, 1000)], workers=0)[0]
    written = [image for page, x, y, image in layer.detail_writes
               if (page, x, y) == (crop.page, crop.x, crop.y)]  # fmt: skip
    np.testing.assert_array_equal(written[0], want)

    # Zooming out drops it at once (the usual texture is enough again), before the
    # view is looked at again.
    look(step, scene.center[tile], 0.5)
    assert zoom.sharpened == 0
    np.testing.assert_array_equal(layer.instances["layer"], usual["layer"])
    np.testing.assert_array_equal(layer.instances["uv"], usual["uv"])

    # Zooming back in reads nothing again; nor does zooming past the photo's pixels.
    reads.clear()
    look(step, scene.center[tile], 2.0)
    zoom.wait()
    assert layer.instances["layer"][tile] >= layer.first_detail
    look(step, scene.center[tile], 6.0)
    zoom.wait()
    assert reads == [] and layer.instances["layer"][tile] >= layer.first_detail
    assert_consistent(step)

    # Zooming in and out quickly: only the view it ends on is read.
    for factor in (1.3, 2.7, 0.4, 3.1, 0.2, 1.7):
        look(step, scene.center[0], factor)
    look(step, scene.center[0], 2.0)
    zoom.wait()
    assert 1 <= len(reads) <= 4 and all(size == (1000, 1000) for _, size in reads)
    assert layer.instances["layer"][0] >= layer.first_detail
    assert_consistent(step)


def test_zoom_detail_survives_new_layers_blending_and_full_pages(
    sliced,  # noqa: F811
    big_photos,
    qapp,
    reads,
    monkeypatch,
):
    from skitter.ui.steps import matching

    monkeypatch.setattr(matching, "DETAIL_SIDE", 1024)  # one crop per page
    step = matched(sliced, big_photos, qapp)
    zoom, scene = step.zoom_detail, step._shown
    # Panning across the mosaic, close up: each view gets its tiles, pages are freed
    # for them as needed, and tiles whose crop leaves the pages show their usual texture.
    for tile in range(0, len(scene), 5):
        look(step, scene.center[tile], 2.0)
        zoom.wait()
        layer = step._tile_layer
        assert layer.instances["layer"][tile] >= layer.first_detail, tile
        assert sum(c.page >= 0 for c in zoom._crops.values()) <= layer.detail_pages
        assert_consistent(step)
    uv = step._tile_layer.instances["uv"][tile]
    assert np.abs(uv[2] - uv[0]) * 1024 + 1 == pytest.approx(1000)

    # The pages are 1024 texels: a closer look can't have more than a page each.
    look(step, scene.center[tile], 3.0)
    zoom.wait()
    assert_consistent(step)

    # A new layer (as when textures are repacked) starts with empty pages: the tiles in
    # view are put back from memory, without reading their photos again.
    reads.clear()
    step._on_textures_changed()
    zoom.wait()
    layer = step._tile_layer
    assert layer.instances["layer"][tile] >= layer.first_detail and reads == []
    assert_consistent(step)

    # Blending replaces the instances: tiles stay sharp.
    sliced.session.project.blend.update(amount=30.0, where="all")
    sliced.session.blend_edited()
    qapp.processEvents()
    assert step._tile_layer.instances["layer"][tile] >= layer.first_detail
    assert step._tile_layer.instances["blend"].max() > 0
    assert_consistent(step)


@pytest.fixture(scope="module")
def gl():
    import moderngl

    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as exc:  # no OpenGL 3.3 here (headless CI)
        pytest.skip(f"no GPU context: {exc}")
    yield ctx
    ctx.release()


def draw(ctx, layer, pixelate: float) -> np.ndarray:
    """The middle row of a 64 px view of a layer, 8 px per unit: (64, 3) uint8."""
    from skitter.ui.render.camera import Camera2D
    from skitter.ui.render.sprites import SpriteRenderer

    fbo = ctx.simple_framebuffer((64, 64))
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    camera = Camera2D()
    camera.viewport[:] = (64, 64)
    camera.zoom = 8.0
    renderer = SpriteRenderer(ctx)
    renderer.set_camera(camera)
    renderer.render(layer, pixelate)
    row = np.frombuffer(fbo.read(components=3), np.uint8).reshape(64, 64, 3)[32]
    renderer.release(layer)
    fbo.release()
    return row


def halves(width: int) -> np.ndarray:
    """(1, 1, width, 3) a texture page, black on the left half, white on the right."""
    page = np.zeros((1, 1, width, 3), np.uint8)
    page[:, :, width // 2 :] = 255
    return page


def test_texels_turn_square_by_their_own_magnification(gl):
    from skitter.ui.render.sprites import SpriteLayer, make_instances

    tile = make_instances(1)
    tile["size"] = (8.0, 8.0)  # 64 px across, at 8 px per unit
    # 2 texels across: each covers 32 px, so they show as sharp squares...
    coarse = draw(gl, SpriteLayer(halves(2), tile.copy()), pixelate=4.0)
    assert set(coarse[:, 0].tolist()) == {0, 255}
    # ...unless pixelation is off: then they blend into each other.
    smooth = draw(gl, SpriteLayer(halves(2), tile.copy()), pixelate=0.0)
    assert len(set(smooth[:, 0].tolist())) > 20
    # 32 texels across cover 2 px each: filtered smoothly, though a mosaic unit covers 8.
    fine = draw(gl, SpriteLayer(halves(32), tile.copy()), pixelate=4.0)
    assert 0 < fine[31, 0] < 255 or 0 < fine[32, 0] < 255

    # Detail pages follow the same rule.
    detail = SpriteLayer(halves(2) // 2, tile.copy(), detail_pages=1, detail_side=2)
    detail.write_detail(0, 0, 0, halves(2)[0])
    detail.instances["layer"] = detail.first_detail
    assert set(draw(gl, detail, pixelate=4.0)[:, 0].tolist()) == {0, 255}  # not the 127s

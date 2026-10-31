"""Infinite Mosaic: sessions of levels (core/infinite.py) and its window (headless)."""

import threading
import time

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from skitter.core.infinite import (
    SOURCE_PX_PER_TILE,
    InfiniteSession,
    InfiniteSettings,
    Loop,
    Lure,
    TileView,
    centered_crop,
    centered_crops,
    picture_size,
)
from skitter.core.slicing import SlicingPlan
from skitter.core.tiles.library import TileLibrary


@pytest.fixture(scope="module")
def photos(tmp_path_factory):
    """Photos of a few shapes: two-color halves (so tiles have structure), some solid."""
    root = tmp_path_factory.mktemp("photos")
    rng = np.random.default_rng(5)
    for i in range(90):
        size = [(96, 96), (144, 96), (96, 144)][i % 3]
        image = np.zeros((size[1], size[0], 3), np.uint8)
        a, b = rng.integers(0, 256, (2, 3))
        image[:] = a
        if i % 2:
            image[:, size[0] // 2 :] = b
        Image.fromarray(image).save(root / f"p{i}.png")
    return root


@pytest.fixture(scope="module")
def library(photos, tmp_path_factory):
    lib = TileLibrary(tmp_path_factory.mktemp("cache"))
    lib.set_roots([photos])
    lib.update(workers=0)
    yield lib
    lib.close()


def test_centered_crop_keeps_the_tile_shape_in_the_middle():
    assert centered_crop(200, 100, 1.0) == pytest.approx((0.25, 0.0, 0.75, 1.0))
    assert centered_crop(100, 200, 1.0) == pytest.approx((0.0, 0.25, 1.0, 0.75))
    x0, y0, x1, y1 = centered_crop(300, 300, 3 / 2)
    assert (x1 - x0) * 300 / ((y1 - y0) * 300) == pytest.approx(3 / 2)
    assert centered_crop(0, 0, 4 / 3) == pytest.approx((0.0, 0.0, 1.0, 1.0))  # unknown size


def test_pictures_have_enough_pixels_per_tile_and_no_more_than_the_cap():
    assert picture_size(900, 1.0, 40) == (900, 900)  # its own pixels
    assert picture_size(100, 1.0, 40) == (40 * SOURCE_PX_PER_TILE,) * 2  # enlarged
    w, h = picture_size(10_000, 2 / 3, 40)  # capped on its longest side (the height)
    assert h == 4096 and w == pytest.approx(4096 * 2 / 3, abs=1)


def crop_descriptors(lightness, cells=None, detail=0.0, color=(0.0, 0.0)):
    """A crop's descriptor: its average lightness and color, its 4 x 4 cells' lightness
    (default: all alike) and the spread within them."""
    from skitter.core.tiles.descriptors import DIM, LEVEL2, MEAN, TEXTURE

    desc = np.zeros(DIM, np.float32)
    desc[MEAN] = (lightness, *color)
    if cells is not None:
        level2 = desc[LEVEL2].reshape(16, 3)
        level2[:, 0] = np.asarray(cells, np.float32) - np.mean(cells)
        desc[LEVEL2] = level2.ravel()
    desc[TEXTURE] = detail
    return desc


def test_variety_favors_lively_tiles_of_the_usual_brightness_and_color():
    from skitter.core.infinite import Looks, pick_weights

    looks = Looks(np.array([0.6, 0.0, 0.0]), np.array([0.12, 0.03, 0.03]), 0.1)
    lively = np.tile([0.4, 0.8], 8)  # half the cells dark, half light: contrast 0.2
    desc = np.stack([
        crop_descriptors(0.6, lively, 0.05),  # usual brightness, lively
        crop_descriptors(0.05),  # black, flat
        crop_descriptors(0.97),  # white, flat
        crop_descriptors(0.6),  # usual brightness, flat
        crop_descriptors(0.08, np.tile([0.0, 0.16], 8), 0.02),  # dark, a little detail
        crop_descriptors(0.6, lively, 0.05, color=(0.2, 0.0)),  # a strong color cast
    ])  # fmt: skip
    variety = looks.variety(desc)
    assert variety[0] == pytest.approx(1.0)
    assert np.all(variety[1:] < 0.2) and np.all(variety >= 0)
    assert variety[4] > variety[1]  # some detail beats none

    assert pick_weights(variety, 0.0) is None  # every tile alike
    assert pick_weights(None, 0.7) is None  # no variety known (a level without it)
    weights = pick_weights(variety, 0.7)
    assert weights.sum() == pytest.approx(1.0) and weights.argmax() == 0
    assert weights[0] > 0.9  # the lively tile is far likelier than the five others together
    assert np.all(np.diff(pick_weights(np.array([0.2, 0.5, 1.0]), 1.0)) > 0)


def test_levels_know_their_tiles_variety(library):
    from skitter.core.infinite import describe_crops
    from skitter.core.tiles.descriptors import tile_descriptors

    session = InfiniteSession(library, InfiniteSettings(columns=6, max_uses=0), seed=2)
    looks = session.looks
    assert looks.lively > 0 and np.all(looks.spread > 0)
    level = session.level(session.picture(session.random_tile()))
    scene = level.scene
    assert level.variety.shape == (len(scene),)
    assert np.all((level.variety >= 0) & (level.variety <= 1))
    # Reading only the scene's photos describes them as reading them all would.
    order = np.argsort(scene.slot, kind="stable")
    full = np.empty((len(scene), 67), np.float32)
    full[order] = tile_descriptors(library.thumbs, library.thumb_size, scene.slot[order],
                                   scene.rect[order])  # fmt: skip
    np.testing.assert_allclose(describe_crops(library, scene.slot, scene.rect), full, atol=1e-6)
    start = session.random_mosaic(1.5)  # instant: its variety comes with prepare
    assert start.variety is None
    session.prepare(start=start)
    assert start.variety.shape == (len(start.scene),)


def test_a_turned_tile_becomes_a_picture_and_level_of_its_own_shape(library):
    """Photo Pile turns some photos a quarter turn: in a portrait layout they are landscape,
    and so are their pictures and levels (not squeezed into the base tile's shape)."""
    plan = SlicingPlan()
    plan.slicers["pile"].update(portrait=0.5, rotation=0.0)
    settings = InfiniteSettings(tile_aspect=2 / 3, columns=6, slicer="pile", max_uses=0)
    session = InfiniteSession(library, settings, plan, seed=3)
    level = session.level(session.picture(session.random_tile()))
    scene = level.scene
    shapes = scene.size[:, 0] / scene.size[:, 1]
    for aspect in (3 / 2, 2 / 3):  # a turned photo, and one as laid out
        i = int(np.flatnonzero(np.isclose(shapes, aspect, rtol=0.02))[0])
        tile = TileView(int(scene.slot[i]), tuple(float(v) for v in scene.rect[i]))
        picture = session.picture(tile)
        assert picture.shape[1] / picture.shape[0] == pytest.approx(aspect, rel=0.02)
        dived = session.level(picture, source=tile)
        assert dived.ctx.width / dived.ctx.height == pytest.approx(aspect, rel=0.02)


def test_settings_match_without_the_slow_extras():
    settings = InfiniteSettings(tint="none", max_uses=2, mirror=True)
    method = settings.method()
    assert method.refine_seconds == 0 and method.adaptive_rounds == 0
    assert (method.tint, method.max_uses, method.mirror) == ("none", 2, True)
    layout = InfiniteSettings(columns=24, tile_aspect=4 / 3).layout
    assert (layout.columns, layout.tile_aspect) == (24, pytest.approx(4 / 3))
    copy = InfiniteSettings.from_values(settings.values())
    assert copy.values() == settings.values()


def test_a_session_dives_from_level_to_level(library, monkeypatch):
    from skitter.core.matching import run as run_module

    calibrations = []
    calibrate = run_module.MatchRun._calibrate

    def counted(self, *args):
        calibrations.append(1)
        return calibrate(self, *args)

    monkeypatch.setattr(run_module.MatchRun, "_calibrate", counted)
    settings = InfiniteSettings(columns=8, tile_aspect=4 / 3, max_uses=0)
    session = InfiniteSession(library, settings, seed=1)

    tile = session.random_tile()
    assert tile.slot in library.ids
    picture = session.picture(tile)
    h, w = picture.shape[:2]
    assert picture.shape[2] == 4 and w / h == pytest.approx(4 / 3, rel=0.01)
    assert w >= 8 * SOURCE_PX_PER_TILE  # small photos are enlarged

    level = session.level(picture)
    assert level.number == 1 and len(level.scene) == len(level.result.regions)  # all placed
    assert level.result.candidates is None and level.result.quality is None  # lean
    assert level.ctx.width / level.ctx.height == pytest.approx(4 / 3, rel=0.01)
    first = len(calibrations)
    assert first >= 1

    # Diving into a tile: its photo, as the tile shows it, becomes the next picture.
    scene = level.scene
    i = int(np.argmax(scene.rect[:, 2] - scene.rect[:, 0] < 1))  # a cropped one, if any
    view = TileView(int(scene.slot[i]), tuple(float(v) for v in scene.rect[i]), True)
    mirrored = session.picture(view)
    plain = session.picture(TileView(view.slot, view.rect, False))
    np.testing.assert_array_equal(mirrored, plain[:, ::-1])
    second = session.level(mirrored)
    assert second.number == 2 and len(second.scene) == len(second.result.regions)
    assert len(calibrations) == first  # search efforts calibrated on the first level only


def test_centered_crops_of_many_photos_at_once():
    rects = centered_crops([200, 100, 0], [100, 200, 0], [1.0, 1.0, 2.0])
    np.testing.assert_allclose(rects, [[0.25, 0, 0.75, 1], [0, 0.25, 1, 0.75], [0, 0, 1, 1]])


def test_a_random_mosaic_fills_a_canvas_of_the_views_shape(library, monkeypatch):
    from skitter.core.matching import matcher as matcher_module

    settings = InfiniteSettings(columns=10, tile_aspect=4 / 3, start="mosaic")
    session = InfiniteSession(library, settings, seed=6)
    level = session.random_mosaic(16 / 9)
    assert level.number == 0 and session.levels == 0  # dives number from 1
    assert level.ctx.width / level.ctx.height == pytest.approx(16 / 9, rel=0.01)
    scene = level.scene
    assert len(scene) == len(level.result.regions) > 10 * 5
    assert np.isin(scene.slot, library.ids).all()
    assert np.allclose(scene.tint_shift, 0)  # photos as they are
    # Each crop has its tile's shape.
    photo = np.stack([library.width[scene.slot], library.height[scene.slot]], axis=1)
    span = (scene.rect[:, 2:] - scene.rect[:, :2]) * photo
    np.testing.assert_allclose(span[:, 0] / span[:, 1], 4 / 3, rtol=0.02)
    other = session.random_mosaic(16 / 9)  # a new start: other photos
    assert not np.array_equal(other.scene.slot, scene.slot)

    # Preparing builds the index, so the first level only searches.
    built = []
    index_type = matcher_module.SearchIndex

    def counted(*args, **kwargs):
        built.append(1)
        return index_type(*args, **kwargs)

    monkeypatch.setattr(matcher_module, "SearchIndex", counted)
    session.prepare()
    assert built == [1]
    session.level(session.picture(TileView(int(scene.slot[0]), tuple(scene.rect[0]))))
    assert built == [1]


def test_a_loop_lures_home_back_within_its_range(library):
    settings = InfiniteSettings(columns=8, max_uses=0, loop=True, loop_dives=(2, 4),
                                loop_fit=10.0, loop_tint=60.0)  # fmt: skip
    session = InfiniteSession(library, settings, seed=7)
    home = session.random_tile()
    loop = Loop(home)
    assert loop.lure(settings) is None  # level 1: before the range
    first = session.level(session.picture(home), source=home)
    loop.levels.append(first)
    lure = loop.lure(settings)  # level 2: in the range, not forced
    assert (lure.tile, lure.tolerance, lure.force, lure.tint) == (home, 0.1, False, 0.6)
    loop.levels += [first, first]
    assert loop.lure(settings).force  # level 4: the top, home placed whatever it takes

    # Forced, home is placed with its exact crop, tinted by its own strength.
    scene = first.scene
    tile = TileView(int(scene.slot[3]), tuple(float(v) for v in scene.rect[3]))
    picture = session.picture(tile)
    level = session.level(picture, lure=Lure(home, 0.0, force=True, tint=0.6), source=tile)
    assert level.home >= 0 and level.home_fit >= 1.0 and level.source == tile
    placed = level.scene
    assert placed.slot[level.home] == home.slot
    np.testing.assert_allclose(placed.rect[level.home], home.rect, atol=1e-6)
    region = placed.region[level.home]
    shift = 0.6 * (placed.target_color[level.home] - level.result.tile_mean[region])
    np.testing.assert_allclose(placed.tint_shift[level.home], shift, atol=1e-6)
    # Not forced, only where it fits within the tolerance.
    loose = session.level(picture, lure=Lure(home, float("inf")), source=tile)
    strict = session.level(picture, lure=Lure(home, 0.0), source=tile)
    assert loose.home >= 0 and (strict.home < 0) == (strict.home_fit > 1.0)

    # Looking ahead: candidates' levels, home placed first, then by fit; numbered 0 until
    # shown.
    paths = library.paths(scene.slot[[1, 2, 5]])
    candidates = [(i, TileView(int(scene.slot[i]), tuple(float(v) for v in scene.rect[i])), path)
                  for i, path in zip([1, 2, 5], paths, strict=True)]  # fmt: skip
    ahead = session.look_ahead(candidates, Lure(home, 0.05, tint=1.0))
    assert sorted(i for i, _ in ahead) == [1, 2, 5]
    keys = [(lv.home < 0, lv.home_fit) for _, lv in ahead]
    assert keys == sorted(keys) and all(lv.number == 0 for _, lv in ahead)
    numbered = session.number(ahead[0][1])
    assert numbered.number == session.levels


def test_levels_can_search_with_fewer_threads(library):
    """While the UI animates, faiss searches with fewer threads (set for the calling thread)."""
    import threading

    import faiss

    seen = []

    def work():
        session = InfiniteSession(library, InfiniteSettings(columns=4, max_uses=0), seed=4)
        session.level(session.picture(session.random_tile()), threads=2)
        seen.append(faiss.omp_get_max_threads())

    thread = threading.Thread(target=work)  # as the window's jobs: other threads keep theirs
    thread.start()
    thread.join()
    assert seen == [2]


def test_every_registered_slicer_makes_levels(library):
    from skitter.core.slicing import operation_types

    for cls in operation_types():
        settings = InfiniteSettings(columns=6, slicer=cls.id, max_uses=0)
        session = InfiniteSession(library, settings, seed=2)
        level = session.level(session.picture(session.random_tile()))
        assert len(level.scene) > 0, cls.id


# The window


def run_animations(canvas, seconds: float, step: float = 1 / 60) -> None:
    """Advance the canvas's animations as frames would, up to seconds (or until done)."""
    start = time.perf_counter()
    canvas._run_animations(start)
    t = 0.0
    while canvas._animations and t < seconds:
        t += step
        canvas._run_animations(start + t)


@pytest.fixture
def window(qapp, photos, tmp_path):
    from skitter.ui.infinite_window import InfiniteWindow
    from skitter.ui.session import Session

    session = Session()
    library = session.open_library(tmp_path / "cache")  # never the user's real library
    library.set_roots([photos])
    library.update(workers=0)
    win = InfiniteWindow(session)
    win.settings.update(columns=6, max_uses=0, dive_seconds=0.5, fade_seconds=0.2)
    win.set_settings(win.settings)
    win.resize(1000, 700)
    win.show()
    qapp.processEvents()
    yield win
    win.close()  # shuts it down; an open application-modal window blocks other tests' input
    session.shutdown()
    library.close()
    win.deleteLater()


def test_closing_the_window_drops_what_it_queued(qapp, photos, tmp_path):
    """Closing deletes the window (as the app opens it) before calls it queued run: the
    level's textures letting go of it, auto-dive's next step. They must be dropped, not
    run on deleted objects."""
    from PySide6.QtCore import QEvent

    from skitter.ui.infinite_window import InfiniteWindow
    from skitter.ui.session import Session

    session = Session()
    library = session.open_library(tmp_path / "cache")
    library.set_roots([photos])
    library.update(workers=0)
    win = InfiniteWindow(session)
    win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    win.settings.update(columns=6, max_uses=0, fade_seconds=0.1, dwell_seconds=0.05)
    win.set_settings(win.settings)
    win.show()
    win.start_button.click()
    wait_for_level(win, qapp)
    win.textures.wait()
    run_animations(win.canvas, 1.0)
    win.auto_button.click()  # a dwell under way: auto-dive's next step is queued
    win.close()  # stops: the textures' release is queued too
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)  # deleted first
    deadline = time.perf_counter() + 0.3
    while time.perf_counter() < deadline:  # the dwell's timer comes due
        qapp.processEvents()
    session.shutdown()
    library.close()


def wait_for_level(window, qapp) -> None:
    job = window._job
    if job is not None:
        job.wait()
    qapp.processEvents()


def test_window_starts_dives_and_stops(window, qapp):
    win = window
    assert win.start_button.text() == "Start" and win.forms["mosaic"].isEnabled()
    win.start_button.click()
    assert win.running and win.start_button.text() == "Stop"
    assert not win.forms["mosaic"].isEnabled() and not win.forms["matching"].isEnabled()
    assert win.forms["dive"].isEnabled() and win.forms["auto"].isEnabled()  # pacing only
    assert not win.slicer_form.isEnabled()
    assert win.photo_layer is not None and win.photo_layer.instances["alpha"][0] == 1
    wait_for_level(win, qapp)

    # The first level arrives under the photo, which fades into it; pan and zoom wait.
    level = win.level
    assert level is not None and level.number == 1
    assert win.canvas.layers.index(win.tile_layer) < win.canvas.layers.index(win.photo_layer)
    assert len(win.tile_layer.instances) == len(level.scene)
    assert win._locked and win.busy
    run_animations(win.canvas, 1.0)
    assert not win._locked and not win.busy and not win.photo_layer.visible
    assert "Level 1" in win.level_label.text()

    # A double-click on a tile dives into it.
    scene = level.scene
    i = len(scene) // 2
    sx, sy = win.canvas.camera.world_to_screen(*scene.center[i])
    assert win.tile_at(*scene.center[i]) == i
    win._on_double_click(float(sx), float(sy))
    assert win.dive is not None and win.dive.index == i and win._locked
    assert win.canvas.pixelate_above is None  # tiles blown up far past their texels: smooth
    photo = win.photo_layer.instances
    np.testing.assert_allclose(photo["pos"][0], scene.center[i], atol=1e-3)
    win._on_double_click(float(sx), float(sy))  # ignored while diving
    assert win.dive.index == i

    run_animations(win.canvas, 0.25)  # halfway: others fly outward, the tile stays
    tiles = win.tile_layer.instances
    np.testing.assert_allclose(tiles["pos"][i], scene.center[i], atol=1e-3)
    moved = np.linalg.norm(tiles["pos"] - scene.center[i], axis=1)
    before = np.linalg.norm(scene.center - scene.center[i], axis=1)
    others = np.arange(len(scene)) != i
    assert np.all(moved[others] > before[others])

    wait_for_level(win, qapp)  # matched before the dive ends: it waits for the dive
    run_animations(win.canvas, 2.0)
    assert win.level is not level and win.level.number == 2 and win.dive is None
    width, height = win.level.ctx.width, win.level.ctx.height
    np.testing.assert_allclose(win.photo_layer.instances["size"][0], (width, height), rtol=1e-6)
    assert win.canvas.bounds == (0, 0, width, height)
    assert win.canvas.pixelate_above == 4.0
    assert not win.busy and not win.photo_layer.visible

    win.start_button.click()
    assert not win.running and win.tile_layer is None and win.photo_layer is None
    assert win.forms["mosaic"].isEnabled() and win.start_button.text() == "Start"


def test_the_library_tab_is_the_main_apps_and_locks_while_a_session_runs(window, qapp,
                                                                         tmp_path):  # fmt: skip
    from PySide6.QtCore import Qt

    from skitter.ui.infinite_window import LIBRARY_TAB
    from skitter.ui.library_panel import LibraryPanel

    win, session = window, window.app_session
    panel = win.library
    other = LibraryPanel(session, lambda: 1.0)  # as the main window's Tiles tab has
    win.tab_bar.setCurrentIndex(LIBRARY_TAB)
    assert panel.folders.topLevelItemCount() == other.folders.topLevelItemCount() == 1
    assert panel._tiles.text() == other._tiles.text() == f"{len(session.library):,}"

    # An edit in one shows in the other: they are one library.
    more = tmp_path / "more"
    more.mkdir()
    assert panel.add_folder(more)
    assert other.folders.topLevelItemCount() == 2
    other.folders.setCurrentItem(other.folders.topLevelItem(1))
    assert other.forget_folder(confirm=False)
    assert panel.folders.topLevelItemCount() == 1

    # A running session searches the photos it started with: the library is locked.
    win.start_button.click()
    assert panel.locked and not panel.lock_note.isHidden()
    root = panel.folders.topLevelItem(0)
    assert not root.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert not panel.add_button.isEnabled() and not panel.update_button.isEnabled()
    assert not panel.add_folder(more) and len(session.library.roots) == 1
    assert not other.locked  # only the window whose session runs
    wait_for_level(win, qapp)
    run_animations(win.canvas, 1.0)
    win.start_button.click()  # stopped: it can be changed again
    assert not panel.locked and panel.lock_note.isHidden()
    assert panel.folders.topLevelItem(0).flags() & Qt.ItemFlag.ItemIsUserCheckable

    # While the library updates, no session starts.
    release = threading.Event()
    update = session.library.update
    session.library.update = lambda **kw: (release.wait(10), update(**kw))[1]
    assert panel.update_library()
    assert session.busy == "library" and not win.start_button.isEnabled()
    release.set()
    session.wait_for_job()
    qapp.processEvents()
    assert win.start_button.isEnabled()
    del session.library.update
    other.deleteLater()


def test_the_infinite_panel_fits_its_width(windows_font, window, qapp):
    """Side panels never scroll sideways (see AGENTS.md): both tabs, locked or not. (The
    font is installed before the window is made: one set later doesn't reach the panel.)"""

    def overflowing() -> list[str]:
        too_wide = []
        for index in range(win.tab_bar.count()):
            win.tab_bar.setCurrentIndex(index)
            qapp.processEvents()
            page = win.pages.widget(index)
            need, have = page.widget().minimumSizeHint().width(), page.viewport().width()
            if need > have:
                too_wide.append(f"{win.tab_bar.tabText(index)}: needs {need} px, has {have}")
        session = win.panel.layout().itemAt(0).widget()
        if session.minimumSizeHint().width() > win.panel.width():
            too_wide.append("Session")
        return too_wide

    win = window
    win.setFont(windows_font)
    qapp.processEvents()
    assert overflowing() == []
    win.start_button.click()  # running: the lock note, Stop, New Start, Auto-Dive
    wait_for_level(win, qapp)
    assert win.library.locked and overflowing() == []


def test_every_dive_style_moves_the_others_and_keeps_the_tile(window, qapp):
    from skitter.core.dive_styles import STYLES

    win = window
    win.settings.update(dive_style="random")
    win.start_button.click()
    wait_for_level(win, qapp)
    run_animations(win.canvas, 1.0)
    scene = win.level.scene
    i = len(scene) // 2
    win.dive_into(i)
    assert win.dive.style.id in STYLES
    others = np.arange(len(scene)) != i
    for style in STYLES.values():
        win.dive.style = style
        win._dive_frame(0.0)
        tiles = win.tile_layer.instances
        np.testing.assert_allclose(tiles["pos"], scene.center, atol=1e-3)  # no jump
        win._dive_frame(0.6)
        np.testing.assert_allclose(tiles["pos"][i], scene.center[i], atol=1e-3)
        np.testing.assert_allclose(tiles["size"][i], scene.size[i], atol=1e-3)
        assert tiles["alpha"][i] == 1.0 and tiles["tint"][i][3] == 0.0
        moved = np.linalg.norm(tiles["pos"] - scene.center, axis=1) > 1e-3
        resized = np.abs(tiles["size"] - scene.size).max(axis=1) > 1e-3
        changed = moved | resized | (tiles["alpha"] < 1)
        assert np.any(changed[others]), style.id
    wait_for_level(win, qapp)
    run_animations(win.canvas, 2.0)
    assert win.level.number == 2


def test_a_dive_waits_for_matching_on_the_photo(window, qapp):
    win = window
    win.start_button.click()
    wait_for_level(win, qapp)
    run_animations(win.canvas, 1.0)
    level = win.level
    i = 0
    sx, sy = win.canvas.camera.world_to_screen(*level.scene.center[i])
    job_started = []
    original = win._match

    def held(picture, kind, message, *rest):  # the match "takes longer than the dive"
        job_started.append(picture)

    win._match = held
    win._on_double_click(float(sx), float(sy))
    win._match = original
    run_animations(win.canvas, 2.0)
    assert win.dive is not None and win.dive.done and win.level is level
    assert win.photo_layer.instances["alpha"][0] == 1  # holding on the photo
    win._match(job_started[0], "dive", "Matching the next level…")
    wait_for_level(win, qapp)
    assert win.level.number == 2 and win.dive is None


def test_a_random_mosaic_start_fills_the_view_and_dives_once_prepared(window, qapp):
    win = window
    win.settings.update(start="mosaic")
    win.set_settings(win.settings)
    win.start_button.click()
    # The mosaic shows at once, filling the view; the index builds meanwhile.
    level = win.level
    assert level is not None and level.number == 0 and win.photo_layer is None
    view = win.canvas.width() / win.canvas.height()
    assert level.ctx.width / level.ctx.height == pytest.approx(view, rel=0.01)
    assert win._job_kind == "prepare" and not win.busy and win.progress.isVisibleTo(win)
    assert "Random mosaic" in win.level_label.text()
    assert win.new_start_button.isVisibleTo(win) and win.new_start_button.isEnabled()

    # A dive while preparing plays, and its matching waits until the index is ready.
    i = len(level.scene) // 2
    sx, sy = win.canvas.camera.world_to_screen(*level.scene.center[i])
    win._on_double_click(float(sx), float(sy))
    assert win.dive is not None and win._queued is not None
    assert not win.new_start_button.isEnabled()
    run_animations(win.canvas, 1.0)
    assert win.dive.done and win.level is level  # holding on the photo
    wait_for_level(win, qapp)  # prepared: the queued dive starts matching
    assert level.variety is not None  # prepared, the start's tiles weighed too
    assert win._queued is None and win._job_kind == "dive"
    wait_for_level(win, qapp)
    assert win.level.number == 1 and win.dive is None
    run_animations(win.canvas, 1.0)

    # New Start: another random mosaic, at once; only its tiles' variety is prepared (the
    # index is kept), even when New Start is pressed again meanwhile.
    assert win.new_start_button.isEnabled()
    win.new_start_button.click()
    assert win.level.number == 0 and win._job_kind == "prepare" and win.level.variety is None
    first = win.level.scene.slot.copy()
    win.new_start_button.click()
    assert not np.array_equal(win.level.scene.slot, first)
    for _ in range(2):
        wait_for_level(win, qapp)
    assert win.level.variety is not None and win._job is None
    assert win.status.text() == "Ready."


def test_new_start_shows_another_random_photo(window, qapp):
    win = window
    win.start_button.click()
    wait_for_level(win, qapp)
    run_animations(win.canvas, 1.0)
    first = win.level
    win.new_start_button.click()
    assert win.level is None and win.photo_layer.instances["alpha"][0] == 1
    wait_for_level(win, qapp)
    assert win.level is not first and win.level.number == 2  # the session counts on


def test_diving_on_and_on_keeps_no_level_behind(window, qapp):
    """Infinite means flat: each dive lets the last level's textures and jobs go."""
    import gc

    from skitter.ui.jobs import Job
    from skitter.ui.render.tile_textures import TileTextures

    def ours():
        """This window's textures (other tests in this process may leave some behind)."""
        library = win.app_session.library
        return [o for o in gc.get_objects()
                if isinstance(o, TileTextures) and o._library is library]  # fmt: skip

    def alive() -> tuple[int, int]:
        """(textures, their jobs) still alive, once those winding down are done."""
        gc.collect()
        for textures in ours():
            textures.wait()  # a replaced level's cancelled job may still be winding down
        qapp.processEvents()  # textures let go from a zero-delay timer
        gc.collect()
        textures = ours()
        jobs = sum(1 for o in gc.get_objects() if isinstance(o, Job) and o.parent() in textures)
        return len(textures), jobs

    win = window
    win.settings.update(start="mosaic")
    win.set_settings(win.settings)
    win.start_button.click()
    counts = []
    for _ in range(6):
        wait_for_level(win, qapp)
        run_animations(win.canvas, 2.0)
        win.textures.wait()  # full-size tiles in, their job stopped
        qapp.processEvents()
        counts.append(alive())
        scene = win.level.scene
        sx, sy = win.canvas.camera.world_to_screen(*scene.center[len(scene) // 2])
        win._on_double_click(float(sx), float(sy))
        assert win.dive is not None
    assert counts[-1] == counts[1] and counts[-1][0] == 1, counts


def run_auto(win, qapp, seconds: float, until=lambda: False) -> None:
    """Let timers and animations run for about seconds of time (or until until())."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end and not until():
        qapp.processEvents()
        run_animations(win.canvas, 0.05)
        if win._job is not None and not win._job.running:
            win._job.wait()


def photo_corners(win, layer, index: int = 0) -> np.ndarray:
    """(4, 2) screen corners of a layer's sprite (its first: a photo's)."""
    inst = layer.instances[index]
    (cx, cy), (w, h), r = inst["pos"], inst["size"], float(inst["rotation"])
    c, s = np.cos(r), np.sin(r)
    corners = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        x, y = cx + c * dx - s * dy, cy + s * dx + c * dy
        corners.append(win.canvas.camera.world_to_screen(x, y))
    return np.array(corners)


@pytest.mark.parametrize("slicer", ["grid", "pile"])  # pile: tiles at angles, some turned
def test_continuous_auto_dive_zooms_on_from_mosaic_to_mosaic_seamlessly(window, qapp, slicer):
    win = window
    win.plan.slicers["pile"].update(rotation=20.0, portrait=0.4)
    win.settings.update(motion="flow", flow_seconds=2.0, slicer=slicer)
    win.set_settings(win.settings)
    assert not win.forms["auto"].is_shown("dwell_seconds")  # steps' settings: hidden
    assert win.forms["auto"].is_shown("flow_seconds")
    win.start_button.click()
    wait_for_level(win, qapp)
    run_animations(win.canvas, 1.0)

    # Each switch of level: the next mosaic's tiles, shown inside the tile, stay put on
    # screen as they move into their own world; so do this mosaic's, under them, fading.
    switches = []
    arrive = win._flow_arrive

    def watched(over):  # compared exactly where they meet: the end, the next one's start
        flow = win._flow
        flow.u = 1.0
        win._flow_frame()
        assert flow.placed and flow.photo.instances["alpha"][0] == 0  # its mosaic is in
        inner, outer = flow.layer, win.tile_layer
        k, j = len(inner.instances) // 2, (flow.index + 1) % len(outer.instances)
        before = photo_corners(win, inner, k), photo_corners(win, outer, j)
        arrive(0.0)
        assert win.tile_layer is inner and win._last_layer is outer
        after = photo_corners(win, inner, k), photo_corners(win, outer, j)
        switches.append((before, after))

    win._flow_arrive = watched
    zooms = []  # (level, zoom) frame by frame
    win.auto_button.click()
    assert win._flow is not None and win.busy and win.dive is None
    end = time.perf_counter() + 30
    while time.perf_counter() < end and win.level.number < 4:
        qapp.processEvents()
        run_animations(win.canvas, 0.05)
        zooms.append((win.level.number, win.canvas.camera.zoom))
        if win._job is not None and not win._job.running:
            win._job.wait()
    assert win.level.number >= 4 and len(switches) >= 3 and win.dive is None
    for before, after in switches:
        np.testing.assert_allclose(after[0], before[0], atol=0.5)  # pixels
        np.testing.assert_allclose(after[1], before[1], atol=0.5)
    for number in {n for n, _ in zooms}:  # within a level, the view only zooms in
        z = [zoom for n, zoom in zooms if n == number]
        assert np.all(np.diff(z) >= -1e-9)
    # On the canvas at most: the last level fading, this one's, the next one's in its tile
    # and its photo; nothing more is kept.
    assert len(win.canvas.layers) <= 4

    # Stopping leaves the view where it is, with nothing of the next level behind.
    win.auto_button.click()
    assert win._flow is None and not win.busy and win._job is None
    assert win.canvas.pixelate_above == 4.0


def test_continuous_auto_dive_makes_the_next_level_ahead_and_stages_it(window, qapp):
    """While the view heads for the tile, its mosaic's textures load and its tile layer and
    photo go up to the GPU (staged), so the switch costs no frame; stopping lets go of them."""
    win = window
    win.settings.update(motion="flow", flow_seconds=3.0)
    win.set_settings(win.settings)
    win.start_button.click()
    wait_for_level(win, qapp)
    run_animations(win.canvas, 1.0)
    win.auto_button.click()
    flow = win._flow
    run_auto(win, qapp, 10.0, until=lambda: flow.next is not None)
    assert flow.photo in win.canvas._staging and flow.textures is not None
    for job in list(win._layer_jobs):
        job.wait()
    qapp.processEvents()
    thumbs, textures = flow.layer, flow.textures  # its tiles from the thumbnails, first
    assert thumbs is not None and thumbs in win.canvas._staging
    textures.wait()  # its full-size crops (the app's pool is warm: much sooner)
    qapp.processEvents()
    for job in list(win._layer_jobs):
        job.wait()
    qapp.processEvents()
    full = flow.full
    assert full is not None and full in win.canvas._staging
    assert full.textures.shape[0] == len(textures.pages)  # from the full-size pages
    run_auto(win, qapp, 10.0, until=lambda: win.level is flow.next)
    assert win.textures is textures and win.tile_layer is thumbs
    assert win._tile_swap is full  # it swaps in once staged (offscreen, nothing draws)

    # Stopping lets go of the next level's, made meanwhile.
    flow = win._flow
    run_auto(win, qapp, 10.0, until=lambda: flow.textures is not None)
    textures = flow.textures
    win.auto_button.click()
    assert flow.textures is None and textures._closing and win.canvas._staging == []


def test_continuous_auto_dive_holds_for_a_mosaic_not_made_yet(window, qapp):
    win = window
    win.settings.update(motion="flow", flow_seconds=2.0)
    win.set_settings(win.settings)
    win.start_button.click()
    wait_for_level(win, qapp)
    run_animations(win.canvas, 1.0)
    match = win._flow_match
    win._flow_match = lambda: None  # "making it takes long"
    win.auto_button.click()
    flow = win._flow
    run_auto(win, qapp, 4.0)
    assert win._flow is flow and flow.u == pytest.approx(flow.hold) and flow.speed == 0
    assert not flow.placed and flow.photo is None  # the photo waits for its mosaic
    win._flow_match = match
    match()
    run_auto(win, qapp, 10.0, until=lambda: win.level.number >= 2)
    assert win.level.number >= 2 and win._flow is not None and win._flow is not flow
    win.auto_button.click()


def test_auto_dive_dwells_zooms_in_rests_and_dives_on_and_on(window, qapp):
    win = window
    win.settings.update(start="mosaic", dwell_seconds=0.0, approach_seconds=0.3,
                        approach_tiles=3.0, pause_seconds=0.0, dive_seconds=0.3,
                        fade_seconds=0.1)  # fmt: skip
    win.set_settings(win.settings)
    assert not win.auto_button.isVisibleTo(win)  # only with a session
    win.start_button.click()
    assert win.auto_button.isVisibleTo(win) and not win.auto
    win.textures.wait()
    level = win.level
    zoom = win.canvas.camera.zoom
    win.auto_button.click()
    assert win.auto and win.auto_button.text() == "Stop Auto-Dive"

    # The view heads for a tile and zooms in on it (outlined), then dives into it.
    qapp.processEvents()  # the dwell (0 s) ends
    assert win._highlight is not None
    target = win._highlight.instances["pos"][0].copy()
    run_animations(win.canvas, 0.3)
    cam = win.canvas.camera
    assert cam.zoom > zoom * 2
    np.testing.assert_allclose(cam.center, target, atol=1e-6)  # the tile in the middle
    tile_width = level.ctx.tile_size[0]
    assert cam.viewport[0] / (cam.zoom * tile_width) == pytest.approx(3.0)  # 3 tiles across
    qapp.processEvents()  # the pause (0 s) ends: the dive starts
    assert win.dive is not None and win._highlight is None
    np.testing.assert_allclose(level.scene.center[win.dive.index], target)

    # And on: it keeps diving, level after level, until stopped.
    run_auto(win, qapp, 20.0, until=lambda: win.level.number >= 2)
    assert win.level.number >= 2
    win.auto_button.click()
    assert not win.auto and win.auto_button.text() == "Auto-Dive"
    run_auto(win, qapp, 10.0, until=lambda: not win.busy)  # a dive under way finishes
    number = win.level.number
    run_auto(win, qapp, 0.5)
    assert win.level.number == number and win._auto_token is None


def test_auto_dive_loops_back_home_and_replays_each_pass(window, qapp):
    win = window
    win.settings.update(start="mosaic", dwell_seconds=0.0, approach_seconds=0.1,
                        pause_seconds=0.0, dive_seconds=0.3, fade_seconds=0.05, loop=True,
                        loop_dives=(2, 3), look_ahead=2)  # fmt: skip
    win.set_settings(win.settings)
    win.start_button.click()
    win.set_auto(True)
    run_auto(win, qapp, 30.0, until=lambda: win.loop is not None and win.loop.closed
             and win._loop_pos == 0 and not win.busy)  # fmt: skip
    loop = win.loop
    assert loop is not None and loop.closed, win.level_label.text()
    n = len(loop.levels)
    assert 2 <= n <= 3 and len(loop.picks) == n
    closing = loop.levels[-1]
    assert closing.home >= 0 and loop.picks[-1] == closing.home
    assert closing.scene.slot[closing.home] == loop.home.slot  # home came back
    assert loop.levels[0].source == loop.home
    assert win.level is loop.levels[0] and loop.passes == 1  # back at the first level
    assert "Loop of" in win.level_label.text()

    # Every pass replays the same levels and tiles: nothing is matched again.
    matched = win.infinite.levels
    visited = []

    def seen():
        if not visited or visited[-1] is not win.level:
            visited.append(win.level)
        return loop.passes >= 2 and win._loop_pos == 0

    run_auto(win, qapp, 30.0, until=seen)
    assert loop.passes == 2 and win.infinite.levels == matched
    assert [id(lv) for lv in visited] == [id(lv) for lv in loop.levels + loop.levels[:1]]

    # A dive by hand leaves the loop.
    win.set_auto(False)
    run_auto(win, qapp, 10.0, until=lambda: not win.busy)
    scene = win.level.scene
    sx, sy = win.canvas.camera.world_to_screen(*scene.center[0])
    win._on_double_click(float(sx), float(sy))
    assert win.loop is None and win.dive is not None


def test_input_on_the_mosaic_or_esc_stops_auto_dive(window, qapp):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent

    win = window
    win.settings.update(start="mosaic", dwell_seconds=0.0, approach_seconds=5.0)
    win.set_settings(win.settings)
    win.start_button.click()
    win.set_auto(True)
    qapp.processEvents()
    run_animations(win.canvas, 0.2)
    assert win._highlight is not None  # heading for a tile
    center = QPointF(win.canvas.width() / 2, win.canvas.height() / 2)
    wheel = QWheelEvent(center, win.canvas.mapToGlobal(center), QPoint(), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)  # fmt: skip
    QApplication.sendEvent(win.canvas, wheel)
    assert not win.auto and win._highlight is None and not win.auto_button.isChecked()
    zoom = win.canvas.camera.zoom
    run_animations(win.canvas, 1.0)  # the approach stopped where it was
    assert win.canvas.camera.zoom == pytest.approx(zoom)

    win.toggle_auto()  # A
    assert win.auto
    win._on_escape()
    assert not win.auto


def test_auto_dive_picks_tiles_in_view(window, qapp):
    win = window
    win.settings.update(start="mosaic", columns=12)
    win.set_settings(win.settings)
    win.start_button.click()
    scene, cam = win.level.scene, win.canvas.camera
    picks = {win.pick_tile() for _ in range(200)}
    assert len(picks) > 20  # anywhere in view
    win.settings.update(next_tile="middle")
    distance = np.linalg.norm(scene.center - cam.center, axis=1)
    near = set(np.argsort(distance)[: max(1, len(scene) // 10)].tolist())
    assert {win.pick_tile() for _ in range(100)} <= near
    win.canvas.set_view(scene.center[0], cam.zoom * 6)  # zoomed in on a corner
    half = cam.viewport / (2 * cam.zoom)
    for _ in range(50):
        assert np.all(np.abs(scene.center[win.pick_tile()] - cam.center) <= half)


def test_pacing_settings_apply_during_a_session_and_are_kept(window, qapp):
    from skitter.ui.infinite_window import load_settings

    win = window
    win.start_button.click()
    wait_for_level(win, qapp)
    win.forms["auto"].editor("dwell_seconds").widget.setValue(7.5)
    assert win.settings.dwell_seconds == 7.5
    assert load_settings()[0].dwell_seconds == 7.5  # kept for the next session too


def test_settings_persist_between_windows(window, qapp):
    from skitter.ui.infinite_window import InfiniteWindow

    win = window
    win.settings.update(columns=7, slicer="pile")
    win.set_settings(win.settings)
    win.start_button.click()
    wait_for_level(win, qapp)
    win.stop()
    other = InfiniteWindow(win.app_session)
    assert other.settings.columns == 7 and other.settings.slicer == "pile"
    assert other.slicer_form._target is other.plan.slicers["pile"]
    other.shutdown()
    other.deleteLater()


def test_the_panel_fits_its_width(window, qapp):
    """Side panels never scroll sideways (see AGENTS.md), here with the Windows UI font."""
    import os

    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QScrollArea

    path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "segoeui.ttf")
    font = QFontDatabase.addApplicationFont(path) if os.path.exists(path) else -1
    if font < 0:
        pytest.skip("needs the Windows UI font (Segoe UI)")
    from skitter.ui.infinite_window import InfiniteWindow

    # A window of its own: combo boxes size to their contents when first shown.
    win = InfiniteWindow(window.app_session)
    win.settings.update(columns=6, max_uses=0, fade_seconds=0.0)
    win.set_settings(win.settings)
    try:
        win.setFont(QFont("Segoe UI", 9))
        win.resize(1000, 700)
        win.show()
        for state in ("idle", "running"):
            if state == "running":
                win.start_button.click()
                wait_for_level(win, qapp)
            qapp.processEvents()
            for scroll in win.findChildren(QScrollArea):
                content = scroll.widget()
                need, have = content.minimumSizeHint().width(), scroll.viewport().width()
                assert need <= have, f"{state}: needs {need} px, has {have}"
    finally:
        win.close()
        win.deleteLater()
        QFontDatabase.removeApplicationFont(font)

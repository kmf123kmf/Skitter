"""Infinite Mosaic: sessions of levels (core/infinite.py) and its window (headless)."""

import time

import numpy as np
import pytest
from PIL import Image

from skitter.core.infinite import (
    SOURCE_PX_PER_TILE,
    InfiniteSession,
    InfiniteSettings,
    TileView,
    centered_crop,
    centered_crops,
    picture_size,
)
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
    win.shutdown()
    session.shutdown()
    library.close()
    win.deleteLater()


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
    assert not any(form.isEnabled() for form in win.forms.values())
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

    def held(picture, kind, message):  # the match "takes longer than the dive"
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
    assert win._prepared and win._queued is None and win._job_kind == "dive"
    wait_for_level(win, qapp)
    assert win.level.number == 1 and win.dive is None
    run_animations(win.canvas, 1.0)

    # New Start: another random mosaic, at once (nothing to prepare again).
    assert win.new_start_button.isEnabled()
    win.new_start_button.click()
    assert win.level.number == 0 and win._job is None and win.status.text() == "Ready."
    first = win.level.scene.slot.copy()
    win.new_start_button.click()
    assert not np.array_equal(win.level.scene.slot, first)


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

    def alive(cls) -> int:
        qapp.processEvents()  # textures let go from a zero-delay timer
        gc.collect()
        return sum(1 for o in gc.get_objects() if isinstance(o, cls))

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
        counts.append((alive(TileTextures), alive(Job)))
        scene = win.level.scene
        sx, sy = win.canvas.camera.world_to_screen(*scene.center[len(scene) // 2])
        win._on_double_click(float(sx), float(sy))
        assert win.dive is not None
    assert counts[-1] == counts[1] and counts[-1][0] == 1, counts


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
        win.shutdown()
        win.deleteLater()
        QFontDatabase.removeApplicationFont(font)

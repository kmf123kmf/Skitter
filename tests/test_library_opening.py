"""The tile library opens in the background (it takes seconds when large): nothing waits for
it on the UI thread, nothing shows half-opened, and what needs it now waits for it."""

import numpy as np
from PIL import Image


def opening_job(session):
    return session._opening[0]


def test_the_library_opens_in_the_background_and_is_taken_on_once_open(qapp):
    from skitter.ui.session import Session

    session = Session()
    changes = []
    session.library_changed.connect(lambda: changes.append(session.library_opening))
    session.open_library_in_background()
    assert session.library is None and session.library_opening and changes == [True]
    session.open_library_in_background()  # once is enough
    job = opening_job(session)
    job.wait()
    assert session.library is not None and not session.library_opening
    assert changes == [True, False]
    session.open_library_in_background()  # open already: nothing to do
    assert not session.library_opening
    session.shutdown()
    session.library.close()


def test_what_needs_the_library_now_takes_on_the_one_opening(qapp):
    from skitter.ui.session import Session

    session = Session()
    session.open_library_in_background()
    job = opening_job(session)
    library = session.open_library()  # waits for it, not a second opening
    assert session.library is library and not session.library_opening
    job.wait()  # its own result, delivered late: the same library, kept open
    assert session.library is library and not library.closed
    session.shutdown()
    library.close()


def test_closing_while_the_library_opens_closes_it(qapp):
    from skitter.ui.session import Session

    session = Session()
    session.open_library_in_background()
    job = opening_job(session)
    session.shutdown()
    job.wait()
    assert session.library is None and not session.library_opening
    assert job._thread is None or not job._thread.is_alive()


def test_the_library_panel_says_it_is_opening_and_waits(qapp, tmp_path):
    from skitter.ui.library_panel import LibraryPanel
    from skitter.ui.session import Session

    session = Session()
    panel = LibraryPanel(session, lambda: 1.0)
    panel.open_library()  # as switching to its tab does: in the background
    assert session.library_opening and "Opening the library" in panel.status.text()
    assert not panel.add_button.isEnabled() and not panel.update_button.isEnabled()
    assert not panel.progress.isHidden()
    opening_job(session).wait()
    assert session.library is not None and panel.add_button.isEnabled()
    assert "Opening" not in panel.status.text() and panel.progress.isHidden()

    # Adding a folder needs the library now: one opening is waited for.
    session.library.close()
    session.library = None
    session.open_library_in_background()
    photos = tmp_path / "photos"
    photos.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(photos / "a.png")
    assert panel.add_folder(photos)
    assert session.library is not None and len(session.library.roots) == 1
    session.shutdown()
    session.library.close()
    panel.deleteLater()


def test_infinite_start_goes_ahead_once_the_library_is_open(qapp):
    from skitter.ui.infinite_window import InfiniteWindow
    from skitter.ui.session import Session

    session = Session()
    win = InfiniteWindow(session)
    win.show()
    win.start_buttons["photo"].click()  # nothing open yet: it opens, then shows one
    assert session.library_opening and "Opening the library" in win.setup_status.text()
    assert not win.start_buttons["photo"].isEnabled() and not win.running
    opening_job(session).wait()
    # The (empty, test) library is open: the pick went ahead, and says what it needs.
    assert win._start_when_open is None and "No photos" in win.library_label.text()
    assert not win.start_buttons["photo"].isEnabled()  # nothing to start with
    win.close()
    session.shutdown()
    session.library.close()

"""The GPU canvas widget (headless: Qt runs offscreen, so nothing is painted here)."""

from PySide6.QtOpenGLWidgets import QOpenGLWidget


def test_camera_has_the_new_size_before_the_base_class_paints(qapp, monkeypatch):
    # QOpenGLWidget.resizeEvent recreates the framebuffer and paints at once; that
    # paint must already see the new viewport, or a single resize (snapping the
    # window) leaves a stretched frame on screen.
    from skitter.ui.canvas import MosaicCanvas

    seen = []
    original = QOpenGLWidget.resizeEvent

    def resize_event(self, event):
        seen.append((event.size().width(), event.size().height(), self.camera.viewport.tolist()))
        original(self, event)

    monkeypatch.setattr(QOpenGLWidget, "resizeEvent", resize_event)
    canvas = MosaicCanvas()
    canvas.resize(300, 200)
    canvas.show()
    qapp.processEvents()
    canvas.resize(500, 120)
    qapp.processEvents()
    assert seen and all([w, h] == viewport for w, h, viewport in seen)
    assert canvas.camera.viewport.tolist() == [500, 120]
    canvas.close()

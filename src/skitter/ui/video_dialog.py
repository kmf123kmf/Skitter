"""Export Animation window: render the build animation to a video file.

Modeless, like Export Image (export_window.py). Settings live in the project
(project.video_settings) and are shared with the Animate tab, which edits
the size and framing too: the window opens with what the tab shows, and
edits in either place show in both.
"""

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QLabel

from skitter.core.animation.video import FORMATS, estimate_bytes, free_bytes, plan_video
from skitter.ui.export_window import ExportWindow
from skitter.ui.render.tile_textures import detail_sizes
from skitter.ui.style import WARNING_STYLE, format_bytes, muted

VIDEO_EXTENSIONS = {f.extension for f in FORMATS.values() if f.extension}


class VideoExportDialog(ExportWindow):
    title = "Export Animation"
    what = "animation"
    job = "video"
    folder_key = "export/video_folder"
    min_width = 520

    def __init__(self, session, parent=None):
        super().__init__(session, session.video_progress, session.video_finished, parent)
        self.format_note = muted(QLabel())
        self.options_layout.insertWidget(1, self.format_note)  # under the settings
        self.size_label = QLabel()
        self.length_label = QLabel()
        self.detail_label = QLabel()
        self.detail_label.setWordWrap(True)
        self.disk_label = QLabel()
        self.disk_label.setWordWrap(True)
        self.info.addRow("Video:", self.size_label)
        self.info.addRow("Length:", self.length_label)
        self.info.addRow("Tile detail:", self.detail_label)
        self.info.addRow("Disk:", self.disk_label)
        session.animation_changed.connect(self._on_animation_changed)
        # The camera's closest zoom sets the texture detail; edits come fast (live keying).
        session.camera_changed.connect(lambda: self._refresh() if self.isVisible() else None)
        self._start()

    def _project_settings(self):
        return self.session.project.video_settings

    @property
    def fmt(self):
        return self.settings.video_format

    # Paths

    def _default_name(self) -> str:
        source = self.session.project.source_path
        stem = f"{source.stem} animation" if source else "mosaic animation"
        return stem + self.fmt.extension

    def _match_extension(self) -> None:
        text = self.path.text().strip()
        if not text:
            return
        path = Path(text)
        stem = path.stem if path.suffix.lower() in VIDEO_EXTENSIONS else path.name
        self.path.setText(str(path.with_name(stem + self.fmt.extension)))

    def browse(self) -> None:
        if self.fmt.sequence:
            folder = QFileDialog.getExistingDirectory(
                self, "Folder for the new frame folder", str(Path(self.path.text()).parent)
            )
            if folder:
                self.path.setText(str(Path(folder) / Path(self.path.text()).name))
        else:
            ext = self.fmt.extension
            path, _ = QFileDialog.getSaveFileName(
                self, self.title, self.path.text(), f"{self.fmt.name} (*{ext})"
            )
            if path:
                self.path.setText(path if path.lower().endswith(ext) else path + ext)
        self._refresh()

    def _replaces(self, target: Path) -> bool:
        return target.exists() and not self.fmt.sequence  # a sequence makes a new folder

    # Settings and readouts

    def _on_setting_changed(self, name: str) -> None:
        if name == "format":
            self._match_extension()
        self.session.animation_edited()  # the Animate tab's export frame follows

    def _on_animation_changed(self) -> None:
        """Settings edited here or on the Animate tab (they share them)."""
        self.form.refresh()
        self._refresh()

    def _problems(self, target: Path | None) -> list[str]:
        return self.session.video_problems(target)

    def _refresh(self) -> None:
        self.format_note.setText(self.fmt.note)
        super()._refresh()

    def _show_readouts(self) -> None:
        scene = self.session.scene
        project = self.session.project
        timeline = project.animation(scene)
        plan = plan_video(scene, timeline.duration, self.settings,
                          project.animation_look.background, timeline.spans)  # fmt: skip
        closest = project.camera_track.path(plan.video_clock, plan.view).max_zoom
        fps = float(plan.fps)
        self.size_label.setText(f"{plan.width:,} × {plan.height:,} px at {fps:g} fps")
        samples = 8 * self.settings.supersampling**2 * plan.samples
        self.length_label.setText(
            f"{plan.seconds:.1f} s, {plan.frames:,} frames ({samples:,} samples per pixel)"
        )
        _, scale, _ = detail_sizes(scene.size * plan.scale * closest)
        tile_px = plan.scale * scene.tile_size[0]
        text = f"{tile_px:,.0f} px base tiles"
        if closest > 1:
            text += f" ({tile_px * closest:,.0f} px at the camera's closest)"
        text += "" if scale >= 1 else f", textures reduced to {scale:.0%} (memory)"
        self.detail_label.setText(text)
        self._show_disk(estimate_bytes(plan, self.fmt))

    def _show_disk(self, estimate: int | None) -> None:
        target = self._target()
        free = free_bytes(target) if target is not None else None
        room = f"; {format_bytes(free)} free" if free is not None else ""
        short = estimate is not None and free is not None and free < estimate * 1.1
        if estimate is None:
            self.disk_label.setText(f"Size shows while exporting{room}")
        else:
            warning = " — not enough space" if short else ""
            self.disk_label.setText(f"About {format_bytes(estimate)}{room}{warning}")
        self.disk_label.setStyleSheet(WARNING_STYLE if short else "")

    # Export

    def _begin(self, target: Path) -> None:
        self.session.start_video_export(target)

    def _saved(self, path: str, report) -> str:
        w, h = report.size
        return (f"Saved {Path(path).name}: {w:,} × {h:,}, {report.seconds:.1f} s, "
                f"{format_bytes(report.bytes)}, in {report.elapsed:.0f} s.")  # fmt: skip

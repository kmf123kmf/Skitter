"""Export Image window: save the mosaic, rendered at full detail, as PNG or JPEG."""

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QFileDialog, QLabel

from skitter.core.assembly import (
    FORMATS,
    SIZE_BY,
    SIZE_PARAMS,
    check_size,
    enlargement,
    export_frame,
)
from skitter.ui.export_window import ExportWindow
from skitter.ui.style import WARNING_STYLE

FILTERS = {"png": "PNG image (*.png)", "jpeg": "JPEG image (*.jpg *.jpeg)"}
SUFFIX_FORMATS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}
ENLARGED = 1.05  # a tile shown this much larger than its photo's pixels counts as enlarged


def _pixels(width: float, height: float) -> str:
    def number(value: float) -> str:
        return f"{value:,.1f}".removesuffix(".0")

    if abs(width - height) < 0.05:
        return f"{number(width)} px"
    return f"{number(width)} × {number(height)} px"


class ExportDialog(ExportWindow):
    title = "Export Image"
    what = "image"
    job = "export"
    folder_key = "export/folder"

    def __init__(self, session, parent=None):
        super().__init__(session, session.export_progress, session.export_finished, parent)
        self.size_label = QLabel()
        self.tile_label = QLabel()
        self.detail_label = QLabel()
        self.detail_label.setWordWrap(True)
        self.detail_label.setToolTip(
            "Tiles shown larger than their photos have pixels for look soft. "
            "A smaller size, or larger photos, avoid it."
        )
        self.info.addRow("Image size:", self.size_label)
        self.info.addRow("Base tile:", self.tile_label)
        self.info.addRow("Tile photos:", self.detail_label)
        self._start()

    def _project_settings(self):
        return self.session.project.export_settings

    # Paths

    def _default_name(self) -> str:
        source = self.session.project.source_path
        stem = f"{source.stem} mosaic" if source else "mosaic"
        return stem + self.settings.extension

    def _on_path_edited(self) -> None:
        """A typed .png / .jpg extension chooses the format."""
        fmt = SUFFIX_FORMATS.get(Path(self.path.text().strip()).suffix.lower())
        if fmt is not None and fmt != self.settings.format:
            self.settings.format = fmt
            self.form.set_target(self.settings)
        self._refresh()

    def _match_extension(self) -> None:
        text = self.path.text().strip()
        if not text:
            return
        path = Path(text)
        if SUFFIX_FORMATS.get(path.suffix.lower()) != self.settings.format:
            stem = path.stem if path.suffix.lower() in SUFFIX_FORMATS else path.name
            self.path.setText(str(path.with_name(stem + self.settings.extension)))

    def browse(self) -> None:
        current = self.settings.format
        filters = [FILTERS[current]] + [f for k, f in FILTERS.items() if k != current]
        path, chosen = QFileDialog.getSaveFileName(
            self, self.title, self.path.text(), ";;".join(filters)
        )
        if not path:
            return
        fmt = SUFFIX_FORMATS.get(Path(path).suffix.lower())
        if fmt is None:  # no extension typed: use the chosen filter's
            fmt = next(k for k, f in FILTERS.items() if f == chosen)
            path += FORMATS[fmt][1]
        self.path.setText(path)
        self._on_path_edited()

    # Settings and readouts

    def _on_setting_changed(self, name: str) -> None:
        if name == "format":
            self._match_extension()
        self._refresh()
        self.session.settings_edited.emit()

    def _problems(self, target: Path | None) -> list[str]:
        session = self.session
        if not session.can_export:
            return ["Match tiles first: there is no mosaic to export."]
        problems = []
        size = check_size(export_frame(session.scene, self.settings).size, self.settings)
        if size:
            problems.append(size)
        if target is not None and not target.parent.is_dir():
            problems.append(f"The folder {target.parent} does not exist.")
        return problems

    def _show_readouts(self) -> None:
        session = self.session
        frame = export_frame(session.scene, self.settings)
        w, h = frame.size
        channels = 4 if self.settings.alpha else 3
        memory = w * h * channels / 2**20
        self.size_label.setText(f"{w:,} × {h:,} px ({w * h / 1e6:,.1f} MP, {memory:,.0f} MB)")
        tw, th = session.slice_context.tile_size
        s = frame.scale
        self.tile_label.setText(_pixels(tw * s, th * s))
        self._show_detail(frame.scale)
        self._sync_sizes((max(1, round(tw * s)), w, h))

    def _sync_sizes(self, sizes: tuple[int, int, int]) -> None:
        """Show the resulting size in the inactive size fields, so switching keeps it."""
        for (key, _), name, value in zip(SIZE_BY, SIZE_PARAMS, sizes, strict=True):
            if key == self.settings.size_by:
                continue
            param = type(self.settings).__dict__[name]
            value = min(max(value, param.min), param.max)
            setattr(self.settings, name, value)
            widget = self.form.editor(name).widget
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

    def _show_detail(self, scale: float) -> None:
        """Whether the tile photos have enough pixels for this size."""
        library = self.session.library
        ratio = enlargement(self.session.scene, library.width, scale)
        enlarged = ratio > ENLARGED
        if not enlarged.any():
            self.detail_label.setText("Full detail (no tile is enlarged)")
            self.detail_label.setStyleSheet("")
            return
        share = enlarged.mean()
        self.detail_label.setText(
            f"{np.count_nonzero(enlarged):,} of {len(ratio):,} tiles ({share:.0%}) enlarged, "
            f"up to {ratio.max():,.1f}×"
        )
        self.detail_label.setStyleSheet(WARNING_STYLE if share > 0.1 else "")

    # Export

    def _begin(self, target: Path) -> None:
        self.session.start_export(target, self.settings)

    def _saved(self, path: str, report) -> str:
        w, h = report.size
        return f"Saved {Path(path).name} ({w:,} × {h:,} px)."

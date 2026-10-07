"""Step 3: the tile library, the photos the mosaic is built from.

The user picks folders; Update reads new and changed images into the
library cache in the background (see core/tiles/library.py). The page shows
library statistics and what the library can paint (core/tiles/palette.py):
a color map of the tiles by hue and lightness, the source picture in the
library's nearest colors or as a heat map of how far off they are, or a
sample of tiles. Tile colors are analyzed in the background whenever the
library changes.
"""

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
)

from skitter.core.color import oklab_to_srgb
from skitter.core.tiles.palette import (
    CHROMAS,
    NEAR,
    ColorMap,
    Coverage,
    TileColors,
    color_map,
    coverage,
    hue_of,
    photo_stats,
    tile_colors,
)
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.jobs import Job, JobCancelled
from skitter.ui.render.atlas import build_atlas
from skitter.ui.render.heat import heat_colors
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.steps.base import StepPage, side_panel
from skitter.ui.steps.tile_picker import fixed_width
from skitter.ui.style import muted, progress_bar, show_progress

SAMPLE_TILES = 2500
SAMPLE_CELL = 64.0  # world units per sample tile
MAP_CELL = 64.0  # world units per color map cell
MAP_GAP = 24.0  # between the gray column and the hues
MAP_TILE = 0.78  # tile size in its cell; the cell's own color frames it
MISSING_ALPHA = 0.3  # cells no tile comes near: their color, faded

COLOR_MAP, PAINTED, ERROR, SAMPLE = "map", "painted", "error", "sample"
DISPLAY_MODES = (
    (COLOR_MAP, "Color map"),
    (PAINTED, "Source in library colors"),
    (ERROR, "Source color error"),
    (SAMPLE, "Sample of tiles"),
)
SOURCE_MODES = (PAINTED, ERROR)


def _format_bytes(n: float) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "bytes" else f"{n:,.1f} {unit}"
        n /= 1024
    raise AssertionError("unreachable")


def _square_crops(width, height) -> np.ndarray:
    """(N, 4) the centered square window of each photo, as fractions."""
    width = np.asarray(width, np.float64)
    height = np.maximum(np.asarray(height, np.float64), 1)
    fx = np.minimum(1, height / np.maximum(width, 1))  # window width / photo width
    fy = np.minimum(1, width / height)
    return np.stack([(1 - fx) / 2, (1 - fy) / 2, (1 + fx) / 2, (1 + fy) / 2], axis=1)


class TilesStep(StepPage):
    title = "Tiles"

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.canvas = MosaicCanvas()
        self.canvas.clamp_to_bounds = True
        self.canvas.cursor_moved.connect(self._hover_at)
        self._layers: list[SpriteLayer] = []
        self._shown: tuple | None = None  # what the canvas shows (see _show)
        self._colors: TileColors | None = None
        self._colors_version: int | None = None  # library version the colors are of
        self._colors_job: Job | None = None
        self._colors_failed: int | None = None  # library version whose analysis failed
        self._map: ColorMap | None = None
        self._coverage: Coverage | None = None
        self._empty = QLabel("Add folders of photos to use as tiles,\nthen press Update.")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet("color: palette(placeholder-text); font-size: 14pt;")
        self._view = QStackedWidget()
        self._view.addWidget(self._empty)
        self._view.addWidget(self.canvas)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._view, stretch=1)
        layout.addWidget(
            side_panel(
                self._build_folders_group(),
                self._build_update_group(),
                self._build_display_group(),
                self._build_stats_group(),
                self._build_errors_group(),
            )  # fmt: skip
        )
        session.library_changed.connect(self._refresh)
        session.library_progress.connect(self._on_progress)
        session.busy_changed.connect(self._refresh)
        session.source_committed.connect(self._on_source_committed)
        session.layout_changed.connect(self._refresh_stats)
        self._refresh()

    # Construction

    def _build_folders_group(self) -> QGroupBox:
        self.folders = QListWidget()
        self.folders.setMaximumHeight(120)
        self.add_button = QPushButton("Add Folder…")
        self.add_button.clicked.connect(self.add_folder_dialog)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_folder)
        buttons = QHBoxLayout()
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch()
        group = QGroupBox("Folders")
        layout = QVBoxLayout(group)
        layout.addWidget(self.folders)
        layout.addLayout(buttons)
        layout.addWidget(muted(QLabel("Images in these folders and their subfolders are tiles.")))
        return group

    def _build_update_group(self) -> QGroupBox:
        self.update_button = QPushButton("Update Library")
        self.update_button.setToolTip("Read new and changed images; unchanged ones are skipped.")
        self.update_button.clicked.connect(self.update_library)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.session.cancel_job)
        buttons = QHBoxLayout()
        buttons.addWidget(self.update_button)
        buttons.addWidget(self.cancel_button)
        self.progress = progress_bar()
        self.status = muted(QLabel())
        group = QGroupBox("Library")
        layout = QVBoxLayout(group)
        layout.addLayout(buttons)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        return group

    def _build_display_group(self) -> QGroupBox:
        self.display_mode = QComboBox()
        for value, label in DISPLAY_MODES:
            self.display_mode.addItem(label, value)
        self.display_mode.setToolTip(
            "Color map: the tiles by hue (across) and lightness (down), grays on the left. "
            "Each cell shows the tile nearest its color, framed by that color; faded cells "
            "are colors no tile comes near.\n"
            "Source: the picture painted with the nearest tile colors, or how far off they "
            "are. Matching also tints tiles, which closes small gaps."
        )
        self.display_mode.currentIndexChanged.connect(self._show)
        for combo in (self.display_mode,):  # never wider than the panel
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(12)
        self.colorfulness = QComboBox()
        for value, label, _ in CHROMAS:
            self.colorfulness.addItem(label, value)
        self.colorfulness.setCurrentIndex(1)
        self.colorfulness.setToolTip("How saturated the colors of the map are.")
        self.colorfulness.currentIndexChanged.connect(self._on_colorfulness)
        self._summary = fixed_width(muted(QLabel("—")), lines=2)
        self._hover = fixed_width(QLabel("—"), lines=2)
        group = QGroupBox("Display")
        self._display_form = form = QFormLayout(group)
        form.addRow("Show:", self.display_mode)
        form.addRow("Colorfulness:", self.colorfulness)
        form.addRow(self._summary)
        form.addRow("Under cursor:", self._hover)
        return group

    def _build_stats_group(self) -> QGroupBox:
        self._tiles = QLabel("—")
        self._failed = QLabel("—")
        self._missing = QLabel("—")
        self._photo_size = QLabel("—")
        self._photo_size.setToolTip("Short side of the photos: the median, and the smallest tenth.")
        self._shapes = fixed_width(QLabel("—"))
        self._shapes.setWordWrap(True)
        self._detail = QLabel("—")
        self._detail.setToolTip(
            "The widest base tile (in exported pixels) that 9 in 10 photos fill without "
            "being enlarged, for the Mosaic tile shape. Larger tiles show some photos "
            "softer than they could be."
        )
        self._disk = QLabel("—")
        self._location = fixed_width(muted(QLabel("—")))  # long paths clip, not the panel
        self._location.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        group = QGroupBox("Contents")
        form = QFormLayout(group)
        form.addRow("Tiles:", self._tiles)
        form.addRow("Unreadable:", self._failed)
        form.addRow("Missing:", self._missing)
        form.addRow("Photo size:", self._photo_size)
        form.addRow("Shapes:", self._shapes)
        form.addRow("Sharp up to:", self._detail)
        form.addRow("Cache size:", self._disk)
        form.addRow("Cache:", self._location)
        return group

    def _build_errors_group(self) -> QGroupBox:
        self.errors = QListWidget()
        self.errors.setMaximumHeight(100)
        self._errors_group = QGroupBox("Unreadable files")
        QVBoxLayout(self._errors_group).addWidget(self.errors)
        self._errors_group.hide()
        return self._errors_group

    # Actions

    def on_enter(self) -> None:
        if self.session.library is None:
            self.session.open_library()

    def add_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add Tile Folder")
        if folder:
            self.add_folder(folder)

    def add_folder(self, folder) -> None:
        library = self.session.library
        if library is None:  # not `or`: an empty library is falsy
            library = self.session.open_library()
        self.session.set_library_folders([*library.roots, folder])

    def remove_folder(self) -> None:
        row = self.folders.currentRow()
        library = self.session.library
        if library is None or row < 0:
            return
        roots = library.roots
        del roots[row]
        self.session.set_library_folders(roots)

    def update_library(self) -> None:
        if self.session.library is not None and not self.session.busy:
            self.progress.setRange(0, 0)
            self.status.setText("Starting…")
            self.session.update_library()

    def is_complete(self) -> bool:
        return self.session.library_ready

    def set_display(self, mode: str) -> None:
        self.display_mode.setCurrentIndex(self.display_mode.findData(mode))

    def wait_for_colors(self, timeout: float | None = None) -> None:
        """Block until the tile colors are analyzed (tests)."""
        if self._colors_job is not None:
            self._colors_job.wait(timeout)

    # Display

    def _on_progress(self, message: str, fraction: float) -> None:
        self.status.setText(message)
        show_progress(self.progress, fraction)

    def _refresh(self) -> None:
        session = self.session
        library = session.library
        busy = session.busy
        updating = busy == "library"
        self.folders.clear()
        if library is not None:
            self.folders.addItems([str(p) for p in library.roots])
        has_folders = library is not None and bool(library.roots)
        for widget in (self.add_button, self.remove_button):
            widget.setEnabled(not busy)
        self.update_button.setEnabled(has_folders and not busy)
        self.cancel_button.setEnabled(updating)
        self.progress.setVisible(updating)

        if library is None:
            for label in (self._tiles, self._failed, self._missing, self._disk, self._location):
                label.setText("—")
        else:
            counts = library.counts()
            self._tiles.setText(f"{counts['ok']:,}")
            self._failed.setText(f"{counts['failed']:,}")
            self._missing.setText(f"{counts['missing']:,}")
            self._disk.setText(_format_bytes(library.disk_bytes()))
            self._location.setText(str(library.folder))
        self._refresh_stats()

        report, error = session.library_report, session.library_error
        if not updating:
            if error:
                self.status.setText(f"Update failed: {error}")
            elif report is not None:
                text = (f"Added {report.added:,}, updated {report.updated:,}, "
                        f"{report.unchanged:,} unchanged, {report.missing:,} missing, "
                        f"{report.failed:,} unreadable.")  # fmt: skip
                self.status.setText(("Cancelled. " if report.cancelled else "") + text)
            elif library is not None and not has_folders:
                self.status.setText("Add a folder to begin.")
        self.errors.clear()
        if report is not None and report.errors:
            self.errors.addItems([f"{path}: {message}" for path, message in report.errors])
        self._errors_group.setVisible(self.errors.count() > 0)
        if updating:
            self._cancel_colors()
        else:
            self._analyze()
            self._show()
        self.state_changed.emit()

    def _refresh_stats(self) -> None:
        library = self.session.library
        ids = library.ids if library is not None else np.zeros(0, np.int64)
        if not len(ids):
            for label in (self._photo_size, self._shapes, self._detail):
                label.setText("—")
            return
        stats = photo_stats(library.width[ids], library.height[ids],
                            self.session.project.layout.tile_aspect)  # fmt: skip
        self._photo_size.setText(f"{stats.median_side:,} px (smallest {stats.small_side:,} px)")
        self._shapes.setText(f"{stats.portrait:.0%} portrait, {stats.landscape:.0%} landscape, "
                             f"{stats.square:.0%} square")  # fmt: skip
        self._detail.setText(f"{stats.detail_px:,} px tiles")

    def _on_source_committed(self) -> None:
        self._coverage = self._shown = None
        self._show()

    def _on_colorfulness(self) -> None:
        self._map = None
        self._show()

    # Tile colors (background)

    def _analyze(self) -> None:
        """Analyze tile colors if the library changed since they were."""
        library = self.session.library
        if library is None or not len(library.ids):
            self._colors = self._map = self._coverage = None
            self._colors_version = None
            return
        version = library.version
        if version in (self._colors_version, self._colors_failed) or self._colors_job is not None:
            return
        self._colors = self._map = self._coverage = None

        def work(progress, cancelled):
            def report(done, total):
                if cancelled():
                    raise JobCancelled
                progress(f"Analyzing tile colors: {done:,} of {total:,}", done / max(total, 1))

            return tile_colors(library, progress=report)

        job = Job(work, parent=self)
        job.progress.connect(lambda message, _: self._summary.setText(message))
        job.finished.connect(lambda colors: self._set_colors(colors, version))
        job.failed.connect(lambda message: self._colors_failed_with(message, version))
        job.stopped.connect(self._colors_job_stopped)
        self._colors_job = job
        self._summary.setText("Analyzing tile colors…")
        job.start()

    def _colors_job_stopped(self) -> None:
        self._colors_job = None
        if self.session.busy != "library":
            self._analyze()  # the library may have changed while it ran
            self._show()

    def _colors_failed_with(self, message: str, version: int) -> None:
        self._colors_failed = version
        self._summary.setText(f"Color analysis failed: {message}")

    def _cancel_colors(self) -> None:
        if self._colors_job is not None:
            self._colors_job.cancel()

    def _set_colors(self, colors: TileColors, version: int) -> None:
        library = self.session.library
        if library is None or library.version != version:
            return  # the library changed meanwhile: _refresh analyzes again
        self._colors, self._colors_version = colors, version
        self._map = self._coverage = None
        self._shown = None
        self._show()

    # Views

    def _mode(self) -> str:
        return self.display_mode.currentData()

    def _show(self) -> None:
        """Show the chosen view (or the tile sample until colors are analyzed)."""
        library = self.session.library
        ids = library.ids if library is not None else np.zeros(0, np.int64)
        has_source = self.session.project.source_final is not None
        model = self.display_mode.model()
        for mode in SOURCE_MODES:
            model.item(self.display_mode.findData(mode)).setEnabled(has_source)
        if self._mode() in SOURCE_MODES and not has_source:
            self.set_display(COLOR_MAP)
            return
        self._display_form.setRowVisible(self.colorfulness, self._mode() == COLOR_MAP)
        if not len(ids):
            self._clear()
            self._shown = None
            self._view.setCurrentWidget(self._empty)
            self._summary.setText("—")
            return
        self._view.setCurrentWidget(self.canvas)
        mode = self._mode() if self._colors is not None else SAMPLE
        key = (mode, self._colors_version, self.colorfulness.currentData(), library.version)
        if key == self._shown:
            return
        self._shown = key
        self._clear()
        self._hover.setText("—")
        if mode == COLOR_MAP:
            self._show_map()
        elif mode in SOURCE_MODES:
            self._show_coverage(mode)
        else:
            self._show_sample(ids)

    def _clear(self) -> None:
        for layer in self._layers:
            self.canvas.remove_layer(layer)
        self._layers = []

    def _add(self, layer: SpriteLayer) -> None:
        self._layers.append(self.canvas.add_layer(layer))

    def _show_sample(self, ids) -> None:
        library = self.session.library
        rng = np.random.default_rng(0)
        sample = np.sort(rng.choice(ids, min(len(ids), SAMPLE_TILES), replace=False))
        atlas = build_atlas(library.thumbs, library.thumb_size, sample)
        n = len(sample)
        columns = math.ceil(math.sqrt(n * 4 / 3))
        index = np.arange(n)
        instances = make_instances(n)
        instances["pos"] = np.stack([index % columns + 0.5, index // columns + 0.5], 1) * (
            SAMPLE_CELL
        )
        aspect = library.width[sample] / np.maximum(library.height[sample], 1)
        fit = 0.92 * SAMPLE_CELL
        instances["size"] = np.stack(
            [fit * np.minimum(1, aspect), fit * np.minimum(1, 1 / aspect)], axis=1
        )
        layer, uv = atlas.locate(sample, np.tile([0, 0, 1, 1], (n, 1)))
        instances["layer"], instances["uv"] = layer, uv
        self._add(SpriteLayer(atlas.pages, instances))
        rows = math.ceil(n / columns)
        self.canvas.fit_to(0, 0, columns * SAMPLE_CELL, rows * SAMPLE_CELL)
        if self._colors is None and self._colors_job is not None:
            return  # the summary shows the analysis
        self._summary.setText(f"A random sample of {n:,} tiles.")

    def _color_map(self) -> ColorMap:
        if self._map is None:
            chroma = dict((key, c) for key, _, c in CHROMAS)[self.colorfulness.currentData()]
            self._map = color_map(self._colors, chroma)
        return self._map

    @staticmethod
    def _cell_center(row, column) -> np.ndarray:
        x = (np.asarray(column) + 0.5) * MAP_CELL + np.where(np.asarray(column) > 0, MAP_GAP, 0)
        return np.stack([x, (np.asarray(row) + 0.5) * MAP_CELL], axis=-1)

    def _map_cell(self, x: float, y: float) -> tuple[int, int] | None:
        rows, columns = self._color_map().shape
        row = math.floor(y / MAP_CELL)
        column = 0 if x < MAP_CELL else math.floor((x - MAP_GAP) / MAP_CELL)
        if x < 0 or MAP_CELL <= x < MAP_CELL + MAP_GAP or not (0 <= row < rows):
            return None
        return (row, column) if 0 <= column < columns else None

    def _show_map(self) -> None:
        library = self.session.library
        cmap = self._color_map()
        rows, columns = np.nonzero(cmap.shown)
        reached = cmap.reached[rows, columns]

        # Every displayable cell in its own color; faded where no tile comes near.
        swatches = make_instances(len(rows))
        swatches["pos"] = self._cell_center(rows, columns)
        swatches["size"] = MAP_CELL
        swatches["tint"][:, :3] = oklab_to_srgb(cmap.target[rows, columns])
        swatches["tint"][:, 3] = 1.0
        swatches["alpha"] = np.where(reached, 1.0, MISSING_ALPHA)
        self._add(SpriteLayer(None, swatches))

        # The nearest tile in each cell some tile comes near, framed by the cell's color.
        r, c = rows[reached], columns[reached]
        slots = self._colors.slots[cmap.tile[r, c]]
        if len(slots):
            unique = np.unique(slots)
            atlas = build_atlas(library.thumbs, library.thumb_size, unique)
            tiles = make_instances(len(slots))
            tiles["pos"] = self._cell_center(r, c)
            tiles["size"] = MAP_TILE * MAP_CELL
            crops = _square_crops(library.width[slots], library.height[slots])
            tiles["layer"], tiles["uv"] = atlas.locate(slots, crops)
            self._add(SpriteLayer(atlas.pages, tiles))
        width = cmap.shape[1] * MAP_CELL + MAP_GAP
        self.canvas.fit_to(0, 0, width, cmap.shape[0] * MAP_CELL)
        self._summary.setText(
            f"Tiles come within ΔE {NEAR * 100:g} of {cmap.share_reached():.0%} of these "
            "colors. Matching's tinting closes smaller gaps."
        )

    def _show_coverage(self, mode: str) -> None:
        if self._coverage is None:
            self._coverage = coverage(self.session.project.source_final, self._colors)
        cov = self._coverage
        h, w = cov.error.shape
        if mode == PAINTED:
            texture = cov.painted
        else:
            texture = np.zeros((h, w, 4), np.uint8)
            texture[..., :3] = np.round(heat_colors(cov.error.ravel()) * 255).reshape(h, w, 3)
            texture[..., 3] = cov.painted[..., 3]
        sprite = make_instances(1)
        sprite["pos"] = (w / 2, h / 2)
        sprite["size"] = (w, h)
        self._add(SpriteLayer(texture, sprite))
        self.canvas.fit_to(0, 0, w, h)
        self._summary.setText(
            f"Mean ΔE {cov.mean_error:.1f} to the nearest tile colors; "
            f"{cov.share_far:.0%} of the picture has none within ΔE {NEAR * 100:g}."
        )

    def _hover_at(self, x: float, y: float) -> None:
        mode = self._shown[0] if self._shown else None
        text = "—"
        if mode == COLOR_MAP:
            cell = self._map_cell(x, y)
            if cell is not None:
                text = self._describe_cell(*cell)
        elif mode in SOURCE_MODES and self._coverage is not None:
            h, w = self._coverage.error.shape
            if 0 <= x < w and 0 <= y < h:
                error = self._coverage.error[int(y), int(x)]
                text = "Outside the picture" if np.isnan(error) else (
                    f"ΔE {error:.1f} to the nearest tile color"
                )  # fmt: skip
        self._hover.setText(text)

    def _describe_cell(self, row: int, column: int) -> str:
        cmap = self._color_map()
        lightness = round(cmap.target[row, column, 0] * 100)
        hue = hue_of(column)
        name = f"Gray, lightness {lightness}" if hue is None else (
            f"Hue {hue:.0f}°, lightness {lightness}"
        )  # fmt: skip
        if not cmap.shown[row, column]:
            return f"{name}\nNot a displayable color"
        nearest = cmap.distance[row, column] * 100
        if cmap.reached[row, column]:
            count = cmap.near[row, column]
            tiles = "1 tile" if count == 1 else f"{count:,} tiles"
            return f"{name}\n{tiles} near (closest ΔE {nearest:.1f})"
        return f"{name}\nNo tile near (closest ΔE {nearest:.1f})"

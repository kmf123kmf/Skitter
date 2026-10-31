"""Step 3: the tile library, the photos the mosaic is built from.

The side panel manages the library (ui/library_panel.py's LibraryPanel,
shared with the Infinite Mosaic: folders, updating, contents). The page
shows what the library can paint (core/tiles/palette.py): a color map of
the tiles by hue and lightness, the source picture in the library's nearest
colors or as a heat map of how far off they are, or a sample of tiles. Tile
colors are analyzed in the background whenever the library changes.
"""

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
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
    tile_colors,
)
from skitter.ui.canvas import MosaicCanvas
from skitter.ui.jobs import Job, JobCancelled
from skitter.ui.library_panel import LibraryPanel
from skitter.ui.render.atlas import build_atlas
from skitter.ui.render.heat import heat_colors
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.steps.base import StepPage, side_panel, view_and_panel
from skitter.ui.style import MUTED_STYLE, fixed_width, muted

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
EMPTY = "Add folders of photos to use as tiles,\nthen press Update."  # the view, no tiles yet
OPENING = "Opening the library…"  # the view while it opens (in the background)


def _square_crops(width, height) -> np.ndarray:
    """(N, 4) the centered square window of each photo, as fractions."""
    width = np.asarray(width, np.float64)
    height = np.maximum(np.asarray(height, np.float64), 1)
    fx = np.minimum(1, height / np.maximum(width, 1))  # window width / photo width
    fy = np.minimum(1, width / height)
    return np.stack([(1 - fx) / 2, (1 - fy) / 2, (1 + fx) / 2, (1 + fy) / 2], axis=1)


class TilesStep(StepPage):
    id = "tiles"
    title = "Tiles"

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.canvas = MosaicCanvas()
        self.canvas.clamp_to_bounds = True
        self.canvas.cursor_moved.connect(self._hover_at)
        self._layers: list[SpriteLayer] = []
        self._shown: tuple | None = None  # what the canvas shows (see _show)
        self._colors: TileColors | None = None
        self._colors_version: tuple | None = None  # library state the colors are of
        self._colors_job: Job | None = None
        self._colors_failed: tuple | None = None  # library state whose analysis failed
        self._map: ColorMap | None = None
        self._coverage: Coverage | None = None
        self._empty = QLabel(EMPTY)
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet(f"{MUTED_STYLE} font-size: 14pt;")
        self._view = QStackedWidget()
        self._view.addWidget(self._empty)
        self._view.addWidget(self.canvas)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.library = LibraryPanel(session, lambda: session.project.layout.tile_aspect)
        panel = side_panel(
            self.library,
            self._build_display_group(),
            resizable=True,  # deep folder trees need the room
        )
        self.splitter = view_and_panel(self._view, panel, "tiles")
        layout.addWidget(self.splitter)
        session.library_changed.connect(self._refresh)
        session.busy_changed.connect(self._refresh)
        session.source_committed.connect(self._on_source_committed)
        session.project_replaced.connect(self._on_source_committed)
        session.layout_changed.connect(self.library.refresh_stats)
        self._refresh()

    # Construction

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

    # Actions

    def on_enter(self) -> None:
        self.library.open_library()

    def is_complete(self) -> bool:
        """Matching may follow without a library: some methods need none (Matching says
        when the chosen one does). Not while the library updates."""
        return self.session.busy != "library"

    def set_display(self, mode: str) -> None:
        self.display_mode.setCurrentIndex(self.display_mode.findData(mode))

    def wait_for_colors(self, timeout: float | None = None) -> None:
        """Block until the tile colors are analyzed (tests)."""
        if self._colors_job is not None:
            self._colors_job.wait(timeout)

    # Display

    def _refresh(self) -> None:
        """The library changed (or a job started or stopped): its colors and the view."""
        if self.session.busy == "library":
            self._cancel_colors()
        else:
            self._analyze()
            self._show()
        self.state_changed.emit()

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
        if library is not None and library.closed:
            return  # shut down (a job stopping afterwards lands here)
        if library is None or not len(library.ids):
            self._colors = self._map = self._coverage = None
            self._colors_version = None
            return
        version = library.state
        if version in (self._colors_version, self._colors_failed) or self._colors_job is not None:
            return
        # The old colors stay shown until the new ones are in (_show), not the sample.

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

    def shutdown(self) -> None:
        if self._colors_job is not None:
            self._colors_job.cancel()
            self._colors_job.wait()

    def _colors_job_stopped(self) -> None:
        self._colors_job = None
        library = self.session.library
        if library is not None and library.closed:
            return  # shut down: its stop arrived after the library closed
        if self.session.busy != "library":
            self._analyze()  # the library may have changed while it ran
            self._show()

    def _colors_failed_with(self, message: str, version: tuple) -> None:
        self._colors_failed = version
        self._summary.setText(f"Color analysis failed: {message}")

    def _cancel_colors(self) -> None:
        if self._colors_job is not None:
            self._colors_job.cancel()

    def _set_colors(self, colors: TileColors, version: tuple) -> None:
        library = self.session.library
        if library is None or library.closed or library.state != version:
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
        self._empty.setText(OPENING if self.session.library_opening else EMPTY)
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
        key = (mode, self._colors_version, self.colorfulness.currentData(), library.state)
        if key == self._shown:
            return
        stale = self._colors is not None and self._colors_version != library.state
        if stale and self._shown is not None and self._shown[0] == mode:
            return  # the library changed: keep this view until its colors are analyzed
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

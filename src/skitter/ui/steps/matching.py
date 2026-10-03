"""Step 4: choose a tile for every region and preview the mosaic.

Matching runs in the background (see core/matching). The preview draws the
session's scene of the result (core/scene.py) with its shared textures:
thumbnails first, full-size crops once read (ui/render/tile_textures.py).
A heat map shows where the mosaic differs most from the image as seen from
a distance.
"""

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from skitter.core.matching.matcher import MatchResult
from skitter.core.scene import MosaicScene
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.render.tile_textures import TileTextures
from skitter.ui.steps.base import StepPage, side_panel
from skitter.ui.widgets.image_viewer import ImageViewer
from skitter.ui.widgets.param_form import ParamForm

WARNING_STYLE = "color: #c42b1c;"
TILES, HEAT, SOURCE = "tiles", "heat", "source"
DISPLAY_MODES = ((TILES, "Tiles"), (HEAT, "Error heat map"), (SOURCE, "Source image"))
HEAT_MAX_DE = 25.0  # ΔE shown fully red
HEAT_ALPHA = 0.75


def heat_colors(error: np.ndarray) -> np.ndarray:
    """(N, 3) colors from green (no error) through yellow to red (HEAT_MAX_DE and above)."""
    t = np.clip(np.nan_to_num(error, nan=0.0) / HEAT_MAX_DE, 0, 1)[:, None]
    green, yellow, red = np.array([[0.2, 0.75, 0.3], [1.0, 0.85, 0.2], [0.9, 0.2, 0.15]])
    low = green + (yellow - green) * np.minimum(t * 2, 1)
    return np.where(t < 0.5, low, yellow + (red - yellow) * (t * 2 - 1))


def _muted(label: QLabel) -> QLabel:
    label.setWordWrap(True)
    label.setStyleSheet("color: palette(placeholder-text);")
    return label


class MatchingStep(StepPage):
    title = "Matching"

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.viewer = ImageViewer()
        self._tile_layer: SpriteLayer | None = None
        self._heat_layer: SpriteLayer | None = None
        self._shown: MosaicScene | None = None
        self._textures: TileTextures | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.viewer, stretch=1)
        layout.addWidget(
            side_panel(
                self._build_run_group(),
                self._build_result_group(),
                self._build_display_group(),
                self._build_settings_group(),
            )  # fmt: skip
        )
        session.source_committed.connect(self._sync_image)
        session.layout_changed.connect(self._sync_image)
        for signal in (session.slicing_changed, session.library_changed,
                       session.matching_changed, session.busy_changed):  # fmt: skip
            signal.connect(self._refresh)
        session.matching_progress.connect(self._on_progress)
        self.viewer.canvas.cursor_moved.connect(self._on_hover)
        self.viewer.canvas.cursor_left.connect(lambda: self._hover.setText("—"))
        self._refresh()

    # Construction

    def _build_run_group(self) -> QGroupBox:
        self.run_button = QPushButton("Match Tiles")
        self.run_button.setToolTip("Choose a tile for every region (runs in the background).")
        self.run_button.clicked.connect(self.run_matching)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.session.cancel_job)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.status = _muted(QLabel())
        self._error = QLabel()
        self._error.setWordWrap(True)
        self._error.setStyleSheet(WARNING_STYLE)
        group = QGroupBox("Matching")
        layout = QVBoxLayout(group)
        layout.addLayout(buttons)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self._error)
        return group

    def _build_settings_group(self) -> QGroupBox:
        self.form = ParamForm()
        self.form.set_target(self.session.project.match_settings)
        self.form.changed.connect(lambda _: self.session.match_settings_edited())
        group = QGroupBox("Settings")
        QVBoxLayout(group).addWidget(self.form)
        return group

    def _build_result_group(self) -> QGroupBox:
        self._labels: dict[str, QLabel] = {}
        group = QGroupBox("Result")
        form = QFormLayout(group)
        rows = (
            ("score", "Score:", "Mean color difference (ΔE) seen from a distance; lower is "
                                "better. About 2 is barely visible."),
            ("scales", "By distance:", "ΔE when blurred over ½, 1 and 2 tiles."),
            ("ssim", "Structure:", "Similarity of light and dark structure (1 = identical)."),
            ("search", "Search:", "Search accuracy: extra cost of approximate search compared "
                                  "with trying every tile, on a sample of regions."),
            ("tiles", "Tiles used:", "Distinct tile images used, of regions placed."),
            ("uses", "Most uses:", "Most times one tile appears."),
            ("rules", "Rule breaks:", "Regions given a tile against the reuse rules because "
                                      "no allowed tile was left."),
            ("gamut", "Missing colors:", "Share of the mosaic whose color no tile comes close "
                                         "to; more tinting or more tiles help."),
            ("time", "Time:", ""),
        )  # fmt: skip
        for key, label, tip in rows:
            value = QLabel("—")
            value.setToolTip(tip)
            self._labels[key] = value
            form.addRow(label, value)
        self._hover = QLabel("—")
        self._hover.setWordWrap(True)
        form.addRow("Under cursor:", self._hover)
        return group

    def _build_display_group(self) -> QGroupBox:
        self.display_mode = QComboBox()
        for value, label in DISPLAY_MODES:
            self.display_mode.addItem(label, value)
        self.display_mode.currentIndexChanged.connect(self._apply_display)
        self._detail = _muted(QLabel("—"))
        self._detail.setToolTip(
            "Tiles first show as small thumbnails, then at full size once read from their files."
        )
        group = QGroupBox("Display")
        form = QFormLayout(group)
        form.addRow("Show:", self.display_mode)
        form.addRow("Tile detail:", self._detail)
        return group

    # Actions

    def run_matching(self) -> None:
        if self.session.can_match and not self.session.busy:
            self.progress.setRange(0, 0)
            self.status.setText("Starting…")
            self.session.start_matching()

    def is_complete(self) -> bool:
        """A valid mosaic exists: it may be out of date (settings changed since it was
        made) but still fits the regions and tiles, so it can be animated and exported."""
        return self.session.mosaic_is_valid

    def on_enter(self) -> None:
        self._sync_image()
        self._refresh()

    # Display

    def _sync_image(self) -> None:
        final = self.session.project.source_final
        size = self.session.mosaic_size()
        if final is None or size is None:
            return
        if self.viewer.image is not final:
            self.viewer.show_image(final, size)
        elif self.viewer.world_size != size:
            self.viewer.set_world_size(*size)

    def _on_progress(self, message: str, fraction: float) -> None:
        self.status.setText(message)
        if fraction < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(fraction * 1000))

    def _refresh(self) -> None:
        session = self.session
        busy = session.busy
        matching = busy == "matching"
        self.run_button.setEnabled(session.can_match and not busy)
        self.cancel_button.setEnabled(matching)
        self.progress.setVisible(matching)
        self.form.setEnabled(not matching)
        self._error.setText(session.match_error or "")
        self._error.setVisible(bool(session.match_error))
        if not matching:
            if not session.can_match:
                self.status.setText(
                    "Slice the image and build a tile library first."
                    if not session.library_ready
                    else "No regions to match."
                )
            elif session.project.matches is None:
                self.status.setText("Press Match Tiles to build the mosaic.")
            elif not session.matching_is_current:
                self.status.setText(
                    "Out of date: settings or the library changed. Press Match Tiles to "
                    "update; until then later steps use the mosaic shown."
                )
            else:
                self.status.setText("Up to date.")
        if session.scene is not self._shown and not matching:
            self._show_scene(session.scene)
        self.state_changed.emit()

    def _show_scene(self, scene: MosaicScene | None) -> None:
        canvas = self.viewer.canvas
        for layer in (self._tile_layer, self._heat_layer):
            if layer is not None:
                canvas.remove_layer(layer)
        self._tile_layer = self._heat_layer = None
        if self._textures is not None:
            self._textures.changed.disconnect(self._on_textures_changed)
            self._textures.status_changed.disconnect(self._detail.setText)
        self._shown, self._textures = scene, self.session.textures
        self._show_stats(None if scene is None else scene.result)
        self._detail.setText("—")
        if scene is None or self._textures is None:
            return
        self._textures.changed.connect(self._on_textures_changed)
        self._textures.status_changed.connect(self._detail.setText)
        self._detail.setText(self._textures.status)
        if self.viewer.image is None:
            self._sync_image()
        if not len(scene):
            return
        tiles = self._textures.instances()
        self._tile_layer = canvas.add_layer(SpriteLayer(self._textures.pages, tiles))

        heat = make_instances(len(scene))
        for field in ("pos", "size", "rotation"):
            heat[field] = tiles[field]
        heat["tint"][:, :3] = heat_colors(scene.error)
        heat["tint"][:, 3] = 1.0
        heat["alpha"] = HEAT_ALPHA
        self._heat_layer = canvas.add_layer(SpriteLayer(None, heat))
        x0, y0, x1, y1 = scene.bounds
        self.viewer.set_content_bounds((x0, y0, x1 - x0, y1 - y0))
        self._apply_display()

    def _on_textures_changed(self) -> None:
        """Full-size tiles arrived: swap them in below the heat map."""
        if self._tile_layer is None:
            return
        canvas = self.viewer.canvas
        index = canvas.layers.index(self._tile_layer)
        canvas.remove_layer(self._tile_layer)
        layer = SpriteLayer(self._textures.pages, self._textures.instances())
        self._tile_layer = canvas.add_layer(layer, index)
        self._apply_display()

    def _apply_display(self) -> None:
        mode = self.display_mode.currentData()
        if self._tile_layer is not None:
            self._tile_layer.visible = mode in (TILES, HEAT)
        if self._heat_layer is not None:
            self._heat_layer.visible = mode == HEAT
        self.viewer.canvas.update()

    def _show_stats(self, result: MatchResult | None) -> None:
        labels = self._labels
        if result is None:
            for label in labels.values():
                label.setText("—")
            return
        q, stats = result.quality, result.stats
        labels["score"].setText(f"ΔE {q.score:.1f}")
        labels["scales"].setText(" / ".join(f"{v:.1f}" for v in q.by_scale.values()))
        labels["ssim"].setText(f"{q.ssim:.2f}")
        if all(r.exact for r in result.regret):
            labels["search"].setText("exact (small library)")
        else:
            worst = max(r.relative for r in result.regret)
            effort = max(r.effort for r in result.regret)
            labels["search"].setText(f"{worst * 100:.1f}% regret at effort {effort}")
        labels["tiles"].setText(f"{stats['unique_tiles']:,} of {stats['placed']:,}")
        labels["uses"].setText(f"{stats['most_uses']:,}")
        breaks = stats["rule_violations"]
        labels["rules"].setText(f"{breaks:,}")
        labels["rules"].setStyleSheet(WARNING_STYLE if breaks else "")
        gap = stats["gamut_gap"]
        labels["gamut"].setText(f"{gap * 100:.1f}%")
        labels["gamut"].setStyleSheet(WARNING_STYLE if gap > 0.05 else "")
        labels["time"].setText(f"{sum(stats['timings'].values()):.1f} s")

    def _on_hover(self, x: float, y: float) -> None:
        library = self.session.library
        if self._shown is None or library is None:
            return
        result = self._shown.result
        index = result.regions.hit_test(x, y)
        if index < 0 or result.tile[index] < 0:
            self._hover.setText("—")
            return
        path = library.paths([result.tile[index]])[0]
        error = result.quality.region_error[index]
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        mirrored = ", mirrored" if result.mirrored[index] else ""
        self._hover.setText(f"{name}{mirrored}; ΔE {error:.1f}")

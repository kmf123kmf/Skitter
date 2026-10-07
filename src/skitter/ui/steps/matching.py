"""Step 4: choose a tile for every region and preview the mosaic.

Matching runs in the background (see core/matching). The preview draws the
session's scene of the result (core/scene.py) with its shared textures:
thumbnails first, full-size crops once read (ui/render/tile_textures.py).
While matching runs, the view follows the run instead: target colors, then
the tiles chosen so far as they improve (ui/render/match_preview.py).
A heat map shows where the mosaic differs most from the image as seen from
a distance.

Edit mode picks tiles by hand (core/matching/edit.py): click a tile, then
choose from its remembered candidates in the side panel. Hovering a
candidate previews it in place; picks can be undone, reverted, and kept
when matching again.
"""

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from skitter.core.matching.matcher import MatchResult
from skitter.core.scene import MosaicScene
from skitter.ui.render.heat import heat_colors
from skitter.ui.render.match_preview import PreviewBuilder, PreviewFrame
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.render.tile_textures import TileTextures
from skitter.ui.steps.base import StepPage, side_panel
from skitter.ui.steps.edit_mode import EditMode
from skitter.ui.steps.tile_picker import (
    EditOverlays,
    elide,
    file_name,
    fixed_width,
)
from skitter.ui.style import WARNING_STYLE, muted, progress_bar, show_progress
from skitter.ui.widgets.image_viewer import ImageViewer
from skitter.ui.widgets.param_form import ParamForm

PICK_STYLE = "color: #3b9cff;"
TILES, HEAT, SOURCE = "tiles", "heat", "source"
DISPLAY_MODES = ((TILES, "Tiles"), (HEAT, "Error heat map"), (SOURCE, "Source image"))
HEAT_ALPHA = 0.75
KEEP, DISCARD = True, False


class MatchingStep(StepPage):
    id = "matching"
    title = "Matching"
    export_requested = Signal()  # the user pressed Export Image…

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.viewer = ImageViewer()
        self._tile_layer: SpriteLayer | None = None
        self._heat_layer: SpriteLayer | None = None
        self._shown: MosaicScene | None = None
        self._textures: TileTextures | None = None
        self.overlays = EditOverlays(self.viewer.canvas)
        self.tile_of: np.ndarray = np.zeros(0, np.int64)  # region -> scene tile (-1: none)
        # The run in progress (shown in place of the mosaic while matching)
        self._run_sketch: SpriteLayer | None = None  # target colors of regions without a tile
        self._run_tiles: SpriteLayer | None = None
        self._run_latest = None  # the newest preview, built when this tab shows
        self._builder = PreviewBuilder(self)
        self._builder.ready.connect(self._show_run)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.viewer, stretch=1)
        self.settings_group = self._build_settings_group()
        self.edit = EditMode(self)
        self.edit.editing_changed.connect(lambda on: self.settings_group.setVisible(not on))
        layout.addWidget(
            side_panel(
                self._build_run_group(),
                self._build_result_group(),
                self.edit.picker,
                self._build_display_group(),
                self.settings_group,
                self._build_export_group(),
            )  # fmt: skip
        )
        session.project_replaced.connect(self._on_project_replaced)
        session.source_committed.connect(self._sync_image)
        session.layout_changed.connect(self._sync_image)
        for signal in (session.slicing_started, session.slicing_changed, session.library_changed,
                       session.matching_changed, session.busy_changed):  # fmt: skip
            signal.connect(self._refresh)
        session.matching_progress.connect(self._on_progress)
        session.matching_detail.connect(self._on_detail)
        session.matching_preview.connect(self._on_run_preview)
        session.mosaic_edited.connect(self._on_mosaic_edited)
        for signal in session.export_signals():
            signal.connect(self._refresh_export)
        canvas = self.viewer.canvas
        canvas.cursor_moved.connect(self._on_hover)
        canvas.cursor_left.connect(self._on_cursor_left)
        canvas.clicked.connect(self.edit.on_click)
        self.viewer.double_click_zooms = False  # double-click edits the tile instead
        canvas.double_clicked.connect(self.edit.on_double_click)
        self._refresh()
        self._refresh_export()

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
        self.progress = progress_bar()
        self.status = muted(QLabel())
        # Within a long step (indexing, searching, widening): a thinner bar, while it reports.
        self.sub_progress = progress_bar()
        self.sub_progress.setRange(0, 1000)
        self.sub_progress.setFixedHeight(max(4, self.progress.sizeHint().height() // 2))
        self.sub_status = fixed_width(muted(QLabel()))
        self.sub_status.setWordWrap(False)
        self._error = QLabel()
        self._error.setWordWrap(True)
        self._error.setStyleSheet(WARNING_STYLE)
        group = QGroupBox("Matching")
        layout = QVBoxLayout(group)
        layout.addLayout(buttons)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.sub_progress)
        layout.addWidget(self.sub_status)
        layout.addWidget(self._error)
        self._show_detail(False)
        return group

    def _on_project_replaced(self) -> None:
        self.edit.set_editing(False)
        self.form.set_target(self.session.project.match_settings)

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
            ("search", "Search:", "Search accuracy: how much worse (ΔE) the fast search's "
                                  "best tiles are than trying every tile, on a sample of "
                                  "regions, and the search effort used."),
            ("tiles", "Tiles used:", "Distinct tile images used, of regions placed."),
            ("uses", "Most uses:", "Most times one tile appears."),
            ("rules", "Rule breaks:", "Regions given a tile against the reuse rules because "
                                      "no allowed tile was left, or picked so by hand."),
            ("gamut", "Missing colors:", "Share of the mosaic whose color no tile comes close "
                                         "to; more tinting or more tiles help."),
            ("picks", "Picked by hand:", "Tiles chosen by hand in edit mode."),
            ("time", "Time:", ""),
        )  # fmt: skip
        for key, label, tip in rows:
            value = QLabel("—")
            value.setToolTip(tip)
            self._labels[key] = value
            form.addRow(label, value)
        self._hover = fixed_width(QLabel("—"), lines=2)  # name, then details: never moves
        form.addRow("Under cursor:", self._hover)
        return group

    def _build_display_group(self) -> QGroupBox:
        self.display_mode = QComboBox()
        for value, label in DISPLAY_MODES:
            self.display_mode.addItem(label, value)
        self.display_mode.currentIndexChanged.connect(self._apply_display)
        self._detail = fixed_width(muted(QLabel("—")), lines=2)  # the run's stage: no jumps
        self._detail.setToolTip(
            "Tiles first show as small thumbnails, then at full size once read from their files."
        )
        group = QGroupBox("Display")
        form = QFormLayout(group)
        form.addRow("Show:", self.display_mode)
        form.addRow("Tile detail:", self._detail)
        return group

    def _build_export_group(self) -> QGroupBox:
        self.export_button = QPushButton("Export Image…")
        self.export_button.setToolTip("Size, format, tile detail and more, then export.")
        self.export_button.clicked.connect(self.export_requested)
        group = QGroupBox("Image")
        QVBoxLayout(group).addWidget(self.export_button)
        return group

    def _refresh_export(self) -> None:
        self.export_button.setEnabled(self.session.can_export)

    # Actions

    def run_matching(self) -> None:
        if not self.session.can_match or self.session.busy:
            return
        keep = False
        count = self.session.manual_picks
        if count:
            answer = self.ask_keep_picks(count)
            if answer is None:
                return
            keep = answer
        self.edit.set_editing(False)
        self.progress.setRange(0, 0)
        self.status.setText("Starting…")
        self.session.start_matching(keep_picks=keep)

    def ask_keep_picks(self, count: int) -> bool | None:
        """Keep (True) or discard (False) manual picks when matching again; None: cancel."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Match Tiles")
        noun = "tile" if count == 1 else "tiles"
        box.setText(f"You picked {count:,} {noun} by hand. Keep them?")
        box.setInformativeText(
            "Kept picks stay where they are and count toward the reuse rules; every other "
            "region is matched again around them."
        )
        keep = box.addButton("Keep Picks", QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton("Discard Picks", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(keep)
        box.exec()
        clicked = box.clickedButton()
        return KEEP if clicked is keep else DISCARD if clicked is discard else None

    def is_complete(self) -> bool:
        """A valid mosaic exists: it may be out of date (settings changed since it was
        made) but still fits the regions and tiles, so it can be animated and exported."""
        return self.session.mosaic_is_valid

    def on_enter(self) -> None:
        self._sync_image()
        self._refresh()
        if self._run_latest is not None:  # skipped while another tab showed
            self._builder.submit(self._run_latest, self.session.library,
                                 self.session.slice_context)  # fmt: skip

    def on_leave(self) -> None:
        self.edit.preview(-1)

    # What edit mode sees of the display

    @property
    def scene_shown(self) -> MosaicScene | None:
        """The mosaic drawn (None while a run shows, or with no mosaic)."""
        return self._shown

    @property
    def tile_layer(self) -> SpriteLayer | None:
        return self._tile_layer

    @property
    def tiles_view(self) -> bool:
        return self.display_mode.currentData() == TILES

    @property
    def source_view(self) -> bool:
        return self.display_mode.currentData() == SOURCE

    def leave_source_view(self) -> None:
        """Show the tiles (picking needs them) if only the source image shows."""
        if self.source_view:
            self.display_mode.setCurrentIndex(0)

    def _on_cursor_left(self) -> None:
        self._hover.setText("—")
        self.overlays.show_hover()

    def _on_mosaic_edited(self, regions) -> None:
        """Manual picks changed some tiles: update the layers in place."""
        session = self.session
        scene = session.scene
        self._shown = scene
        if self._tile_layer is not None:
            self._tile_layer.instances = self._textures.instances()
            self._tile_layer.mark_dirty()
        if self._heat_layer is not None:
            self._heat_layer.instances["tint"][:, :3] = heat_colors(scene.error)
            self._heat_layer.mark_dirty()
        self._show_stats(scene.result)
        self.edit.mosaic_edited()
        self.viewer.canvas.update()
        self.state_changed.emit()

    def _on_patched(self, rects) -> None:
        if self._tile_layer is not None:
            self._tile_layer.patch_texture(rects, self._textures.pages)
            self.viewer.canvas.update()

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
        self._show_detail(False)  # a new step: its own detail (if any) follows
        self.status.setText(message)
        show_progress(self.progress, fraction)

    def _on_detail(self, message: str, fraction: float) -> None:
        if self.session.busy != "matching":
            return
        self.sub_status.setText(message)
        self.sub_progress.setValue(round(min(max(fraction, 0.0), 1.0) * 1000))
        self._show_detail(True)

    def _show_detail(self, shown: bool) -> None:
        self.sub_progress.setVisible(shown)
        self.sub_status.setVisible(shown)

    def _refresh(self) -> None:
        session = self.session
        busy = session.busy
        matching = busy == "matching"
        if not matching:
            self._show_detail(False)
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
                text = "Up to date."
                if session.dropped_picks:
                    text += (f" {session.dropped_picks:,} hand-picked tiles could not be kept: "
                             "their photos are no longer in the library.")  # fmt: skip
                self.status.setText(text)
        ended = not matching and self._end_run_view()
        if not matching and (ended or session.scene is not self._shown):
            self.edit.select(-1)
            self._show_scene(session.scene)
        self.edit.refresh()
        self.state_changed.emit()

    def _show_scene(self, scene: MosaicScene | None) -> None:
        canvas = self.viewer.canvas
        for layer in (self._tile_layer, self._heat_layer):
            if layer is not None:
                canvas.remove_layer(layer)
        self._tile_layer = self._heat_layer = None
        self.overlays.clear()
        if self._textures is not None:
            self._textures.changed.disconnect(self._on_textures_changed)
            self._textures.patched.disconnect(self._on_patched)
            self._textures.status_changed.disconnect(self._detail.setText)
        self._shown, self._textures = scene, self.session.textures
        self._show_stats(None if scene is None else scene.result)
        self._detail.setText("—")
        if scene is None:
            self.viewer.set_content_bounds(None)  # not the last mosaic's (it may be larger)
        if scene is None or self._textures is None:
            return
        self._textures.changed.connect(self._on_textures_changed)
        self._textures.patched.connect(self._on_patched)
        self._textures.status_changed.connect(self._detail.setText)
        self._detail.setText(self._textures.status)
        if self.viewer.image is None:
            self._sync_image()
        self.tile_of = np.full(len(scene.result.tile), -1, np.int64)
        self.tile_of[scene.region] = np.arange(len(scene))
        if not len(scene):
            return
        tiles = self._textures.instances()
        self._tile_layer = canvas.add_layer(
            SpriteLayer(self._textures.pages, tiles), self.overlays.first
        )

        heat = make_instances(len(scene))
        for field in ("pos", "size", "rotation"):
            heat[field] = tiles[field]
        heat["tint"][:, :3] = heat_colors(scene.error)
        heat["tint"][:, 3] = 1.0
        heat["alpha"] = HEAT_ALPHA
        self._heat_layer = canvas.add_layer(SpriteLayer(None, heat), self.overlays.first)
        x0, y0, x1, y1 = scene.bounds
        self.viewer.set_content_bounds((x0, y0, x1 - x0, y1 - y0))
        self._apply_display()

    def _on_textures_changed(self) -> None:
        """Full-size tiles arrived (or pages were repacked): swap them in below the heat map."""
        if self._tile_layer is None:
            return
        canvas = self.viewer.canvas
        index = canvas.layers.index(self._tile_layer)
        canvas.remove_layer(self._tile_layer)
        layer = SpriteLayer(self._textures.pages, self._textures.instances())
        self._tile_layer = canvas.add_layer(layer, index)
        if self.overlays.cover.texture_from is not None:
            self.overlays.cover.texture_from = layer
        self._apply_display()

    def _apply_display(self) -> None:
        mode = self.display_mode.currentData()
        for layer in (self._run_sketch, self._run_tiles):
            if layer is not None:
                layer.visible = mode in (TILES, HEAT)
        if self._tile_layer is not None:
            self._tile_layer.visible = mode in (TILES, HEAT)
        if self._heat_layer is not None:
            self._heat_layer.visible = mode == HEAT
        self.edit.display_changed()
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
            worst = max(r.extra_de for r in result.regret)
            effort = max(r.effort for r in result.regret)
            labels["search"].setText(f"+{worst:.2f} ΔE vs exact, effort {effort:,}")
        labels["tiles"].setText(f"{stats['unique_tiles']:,} of {stats['placed']:,}")
        labels["uses"].setText(f"{stats['most_uses']:,}")
        breaks = stats["rule_violations"]
        labels["rules"].setText(f"{breaks:,}")
        labels["rules"].setStyleSheet(WARNING_STYLE if breaks else "")
        gap = stats["gamut_gap"]
        labels["gamut"].setText(f"{gap * 100:.1f}%")
        labels["gamut"].setStyleSheet(WARNING_STYLE if gap > 0.05 else "")
        labels["picks"].setText(f"{result.manual_count:,}")
        labels["time"].setText(f"{sum(stats['timings'].values()):.1f} s")

    def _on_hover(self, x: float, y: float) -> None:
        library = self.session.library
        if self._shown is None or library is None:
            return
        result = self._shown.result
        index = result.regions.hit_test(x, y)
        self.edit.hover(index)
        if index < 0 or result.tile[index] < 0:
            self._hover.setText("—")
            return
        path = library.paths([result.tile[index]])[0]
        error = result.quality.region_error[index]
        full = file_name(path)
        mirrored = ", mirrored" if result.mirrored[index] else ""
        picked = ", picked by hand" if result.manual is not None and result.manual[index] else ""
        details = f"ΔE {error:.1f}{mirrored}{picked}"
        self._hover.setText(elide(self._hover, full) + "\n" + details)
        self._hover.setToolTip(full)

    # The run in progress

    def _on_run_preview(self, preview) -> None:
        session = self.session
        if session.busy != "matching":  # arrived after the run ended
            return
        self._run_latest = preview
        if self.isVisible():
            self._builder.submit(preview, session.library, session.slice_context)

    def _show_run(self, frame: PreviewFrame) -> None:
        """Show a frame of the run in place of the mosaic (or the last frame)."""
        if self.session.busy != "matching":
            return
        canvas = self.viewer.canvas
        if self._run_sketch is None and self._run_tiles is None:  # the run's first frame
            regions = None if self._run_latest is None else self._run_latest.regions
            if regions is not None and len(regions):  # fit what the run covers (may overhang)
                self.viewer.set_content_bounds(regions.extent())
            for layer in (self._tile_layer, self._heat_layer):
                if layer is not None:
                    canvas.remove_layer(layer)
            self._tile_layer = self._heat_layer = None
            self._shown = None
            self.overlays.clear()
            self._show_stats(None)
            self._hover.setText("—")
        if frame.pages is not None:
            if self._run_tiles is not None:
                canvas.remove_layer(self._run_tiles)
            self._run_tiles = canvas.add_layer(frame.pages, self.overlays.first)
        elif frame.tiles is not None and self._run_tiles is not None:
            self._run_tiles.patch_texture(frame.patches)  # the new thumbnails' cells
            self._run_tiles.instances = frame.tiles
            self._run_tiles.mark_dirty()
        if self._run_sketch is not None:
            canvas.remove_layer(self._run_sketch)
        below = canvas.layers.index(self._run_tiles) if self._run_tiles else self.overlays.first
        self._run_sketch = canvas.add_layer(frame.sketch, below)
        text = frame.stage.value
        if frame.tiles is not None:
            text += f"\n{frame.placed:,} of {frame.needed:,} with a tile"
        self._detail.setText(text)
        self._apply_display()

    def _end_run_view(self) -> bool:
        """Stop showing the run (it ended); True if it was shown."""
        self._builder.reset()
        self._run_latest = None
        shown = self._run_sketch is not None or self._run_tiles is not None
        for layer in (self._run_sketch, self._run_tiles):
            if layer is not None:
                self.viewer.canvas.remove_layer(layer)
        self._run_sketch = self._run_tiles = None
        return shown

    def shutdown(self) -> None:
        self._builder.close()
        self.edit.shutdown()

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

import html

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from skitter.core.assembly import TileFiles
from skitter.core.matching import edit as picks
from skitter.core.matching.matcher import MatchResult
from skitter.core.scene import MosaicScene
from skitter.ui.render.match_preview import PreviewBuilder, PreviewFrame
from skitter.ui.render.sprites import SpriteLayer, make_instances
from skitter.ui.render.tile_textures import TileTextures
from skitter.ui.steps.base import StepPage, side_panel
from skitter.ui.steps.tile_picker import (
    CropReader,
    EditOverlays,
    TilePicker,
    crop_size,
    elide,
    file_name,
    fixed_width,
    marker_instances,
    region_patch,
    thumb_candidate,
    tinted,
    to_qimage,
    upper_closure,
)
from skitter.ui.widgets.candidate_grid import CandidateItem
from skitter.ui.widgets.image_viewer import ImageViewer
from skitter.ui.widgets.param_form import ParamForm

WARNING_STYLE = "color: #c42b1c;"
PICK_STYLE = "color: #3b9cff;"
TILES, HEAT, SOURCE = "tiles", "heat", "source"
DISPLAY_MODES = ((TILES, "Tiles"), (HEAT, "Error heat map"), (SOURCE, "Source image"))
HEAT_MAX_DE = 25.0  # ΔE shown fully red
HEAT_ALPHA = 0.75
KEEP, DISCARD = True, False


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
    export_requested = Signal()  # the user pressed Export Image…

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.viewer = ImageViewer()
        self._tile_layer: SpriteLayer | None = None
        self._heat_layer: SpriteLayer | None = None
        self._shown: MosaicScene | None = None
        self._textures: TileTextures | None = None
        self.overlays = EditOverlays(self.viewer.canvas)
        # Edit mode
        self.editing = False
        self.selected = -1  # region being edited (-1: none)
        self._tile_of: np.ndarray = np.zeros(0, np.int64)  # region -> scene tile (-1: none)
        self._refs = np.zeros(0, np.int64)  # the picker's candidates
        self._crops: list[np.ndarray] = []  # each one's crop, untinted and unmirrored
        self._crop_of: dict[int, np.ndarray] = {}  # the same by candidate
        self._image_of: dict[int, QImage] = {}  # each one's crop as shown (tinted, mirrored)
        self._read: set[int] = set()  # candidates whose crops came from their files
        self._shifts = np.zeros((0, 3), np.float32)
        self._mirrored = np.zeros(0, bool)
        self._previewing = -1
        self._feedback = ""  # result of the last pick, shown until the selection changes
        self._reader = CropReader(self)
        self._reader.ready.connect(self._on_crop)
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
        layout.addWidget(
            side_panel(
                self._build_run_group(),
                self._build_result_group(),
                self._build_picker(),
                self._build_display_group(),
                self.settings_group,
                self._build_export_group(),
            )  # fmt: skip
        )
        session.source_committed.connect(self._sync_image)
        session.layout_changed.connect(self._sync_image)
        for signal in (session.slicing_changed, session.library_changed,
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
        canvas.clicked.connect(self._on_click)
        self.viewer.double_click_zooms = False  # double-click edits the tile instead
        canvas.double_clicked.connect(self._on_double_click)
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
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.status = _muted(QLabel())
        # Within a long step (indexing tiles): a thinner bar, shown while it reports.
        self.sub_progress = QProgressBar()
        self.sub_progress.setTextVisible(False)
        self.sub_progress.setRange(0, 1000)
        self.sub_progress.setFixedHeight(max(4, self.progress.sizeHint().height() // 2))
        self.sub_status = fixed_width(_muted(QLabel()))
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
        self._detail = fixed_width(_muted(QLabel("—")), lines=2)  # the run's stage: no jumps
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

    def _build_picker(self) -> TilePicker:
        picker = self.picker = TilePicker()
        session = self.session

        def action(text, slot, *keys) -> QAction:
            act = QAction(text, self)
            act.setShortcuts([QKeySequence(k) for k in keys])
            act.triggered.connect(slot)
            self.addAction(act)  # active while this tab shows
            return act

        self.edit_action = action("Edit Tiles", lambda: self.set_editing(not self.editing), "E")
        self.undo_action = action("Undo Pick", session.undo_pick, QKeySequence.StandardKey.Undo)
        self.redo_action = action("Redo Pick", session.redo_pick, QKeySequence.StandardKey.Redo,
                                  "Ctrl+Shift+Z")  # fmt: skip
        self.escape_action = action("Back", self.escape, "Escape")
        picker.edit_button.clicked.connect(self.set_editing)
        picker.undo_button.clicked.connect(session.undo_pick)
        picker.redo_button.clicked.connect(session.redo_pick)
        picker.revert_all_button.clicked.connect(lambda: session.revert_picks())
        picker.more_button.clicked.connect(self.find_more)
        picker.revert_button.clicked.connect(self.revert_tile)
        picker.grid.hovered.connect(self._preview)
        picker.grid.activated.connect(self.pick)
        picker.grid.finished.connect(self.pick_and_finish)
        picker.grid.escaped.connect(self.escape)
        picker.show_selection(False)
        return picker

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
        self.set_editing(False)
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
        self._preview(-1)

    # Edit mode

    def can_edit(self) -> bool:
        session = self.session
        result = session.project.matches
        return (
            session.mosaic_is_valid
            and session.busy not in ("matching", "library")
            and result is not None
            and result.candidates is not None
            and self._shown is session.scene
            and self._tile_layer is not None
        )

    def set_editing(self, on: bool) -> None:
        on = bool(on) and self.can_edit()
        if on and self.display_mode.currentData() == SOURCE:
            self.display_mode.setCurrentIndex(0)
        changed = on != self.editing
        self.editing = on
        if not on:
            self.select(-1)
            self.overlays.clear()
        self.settings_group.setVisible(not on)
        if changed:
            self.viewer.canvas.unsetCursor()
            self.status_message.emit("Edit mode: click a tile to pick another photo for it."
                                     if on else "")  # fmt: skip
        self._refresh_edit()

    def escape(self) -> None:
        """Esc: stop previewing, then deselect, then leave edit mode."""
        if self._previewing >= 0:
            self.picker.grid.clear_preview()
            self._preview(-1)
        elif self.selected >= 0:
            self.select(-1)
        elif self.editing:
            self.set_editing(False)

    def select(self, region: int) -> None:
        """Edit a region's tile (-1: none)."""
        if region >= 0 and not (self.editing and self.session.can_pick(region)):
            region = -1
        if region != self.selected:
            self._feedback = ""
            self._crop_of, self._image_of, self._read = {}, {}, set()
        self.selected = region
        self._previewing = -1
        self._reader.cancel()
        self.overlays.show_preview(None, None)
        if region < 0:
            self.picker.show_selection(False)
            self.overlays.show_selection(None, None, 0.0, None)
            self.overlays.show_cover(np.zeros(0, np.int64), None)
            self._refresh_edit()
            return
        self._load_candidates()
        result = self.session.project.matches
        ctx = self.session.slice_context
        regions = result.regions
        patch = region_patch(ctx.image, ctx.scale, regions.center[region], regions.size[region],
                             regions.rotation[region])  # fmt: skip
        self.picker.set_target(patch)
        self.picker.show_selection(True)
        self._show_selection()
        self._refresh_edit()
        self.picker.grid.setFocus()

    def pick(self, index: int) -> None:
        """Show the picker's candidate `index` in the selected region."""
        if self.selected < 0 or not 0 <= index < len(self._refs):
            return
        r, ref = self.selected, int(self._refs[index])
        result = self.session.project.matches
        if ref == result.choice[r]:
            return
        before = float(result.quality.region_error[r])
        if ref == result.candidates.auto[r]:
            self.session.revert_picks([r])  # the matcher's own choice: no longer a manual pick
        else:
            self.session.pick_tile(r, ref)
        after = float(self.session.project.matches.quality.region_error[r])
        self._feedback = f"ΔE {before:.1f} → {after:.1f}"
        self._show_info()

    def pick_and_finish(self, index: int) -> None:
        """Double-clicked candidate: pick it and leave edit mode."""
        self.pick(index)
        self.set_editing(False)

    def revert_tile(self) -> None:
        if self.selected >= 0:
            self.session.revert_picks([self.selected])

    def find_more(self) -> None:
        if self.selected < 0:
            return
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            added = self.session.find_more_candidates(self.selected)
        finally:
            self.unsetCursor()
        if added:
            self._load_candidates()
            self.status_message.emit(f"Found {added} more candidates.")
        else:
            self.picker.more_button.setEnabled(False)
            self.status_message.emit("No other tiles fit this region.")

    def _load_candidates(self) -> None:
        """Fill the picker with the selected region's candidates (crops from thumbnails
        until read from their files)."""
        session, r = self.session, self.selected
        result, library = session.project.matches, session.library
        cands = result.candidates
        refs, costs = cands.ranked(r)
        current = int(result.choice[r])
        labels = [str(i + 1) for i in range(len(refs))]
        # The tile shown now and the matcher's choice are always offered, ranked or not.
        for ref, cost in ((int(cands.auto[r]), cands.auto_cost[r]), (current, result.cost[r])):
            if ref >= 0 and ref not in refs:
                refs = np.concatenate([[ref], refs])
                costs = np.concatenate([[cost], costs])
                labels = ["•", *labels]
        slots, rects, mirrored, means = cands.crops(r, refs)
        shifts = (result.tint * (result.tint_target[r] - means)).astype(np.float32)
        aspect = float(result.regions.size[r, 0] / result.regions.size[r, 1])
        crops = [self._crop_of.get(int(ref)) for ref in refs]
        crops = [c if c is not None else thumb_candidate(library, slots[i], rects[i], aspect)
                 for i, c in enumerate(crops)]  # fmt: skip
        for i, ref in enumerate(refs):
            if int(ref) not in self._image_of:
                self._image_of[int(ref)] = to_qimage(tinted(crops[i], shifts[i], mirrored[i]))
        status = picks.reuse(result, session.slice_context, r, slots)
        names = [file_name(p) for p in library.paths(slots)]
        color_de = 100 * np.linalg.norm(means + shifts - result.tint_target[r], axis=1)
        best = float(np.min(costs[np.isfinite(costs)], initial=np.inf))
        items = []
        for i, ref in enumerate(refs):
            warning = []
            if status.over[i]:
                warning.append(f"Already used {status.uses[i]}× elsewhere "
                               f"(limit {status.max_uses}).")  # fmt: skip
            if status.close[i]:
                warning.append(f"Another use {status.nearest[i]:.1f} tiles away "
                               f"(spacing {status.spacing:g}).")  # fmt: skip
            extra = (costs[i] - best) / max(abs(best), 1e-9) if np.isfinite(costs[i]) else np.nan
            lines = [
                f"<b>{labels[i]}. {names[i]}</b>" + (" (mirrored)" if mirrored[i] else ""),
                f"Color ΔE {color_de[i]:.1f} after tinting",
                "Best match" if extra == 0 else f"Match cost +{extra:.0%} over the best",
            ]
            retained = float(np.prod(rects[i, 2:] - rects[i, :2]))
            if retained < 0.999:
                lines.append(f"Keeps {retained:.0%} of the photo")
            if ref == cands.auto[r]:
                lines.append("★ The matcher's choice")
            if ref == current:
                lines.append("Shown now")
            lines += [f"<span style='{WARNING_STYLE}'>! {w}</span>" for w in warning]
            items.append(CandidateItem(
                image=self._image_of[int(ref)],
                rank=labels[i], current=bool(ref == current), auto=bool(ref == cands.auto[r]),
                warning=" ".join(warning), tooltip="<br>".join(lines),
            ))  # fmt: skip
        self._refs, self._crops, self._shifts, self._mirrored = refs, crops, shifts, mirrored
        self._crop_of = {int(ref): crop for ref, crop in zip(refs, crops, strict=True)}
        self.picker.grid.set_items(items, aspect)
        unread = [i for i, ref in enumerate(refs) if int(ref) not in self._read]
        if unread:
            files = TileFiles.read(library, slots[unread])
            index = files.index(slots[unread])
            size = crop_size(aspect)
            self._reader.read((r, refs.tobytes()), [
                (i, files.paths[j], rects[i], size, files.thumbs[j], files.thumb_size[j],
                 shifts[i], bool(mirrored[i]))
                for i, j in zip(unread, index, strict=True)
            ])  # fmt: skip
        self.picker.more_button.setEnabled(True)
        self._show_info()

    def _on_crop(self, token, index: int, rgb: np.ndarray, shown: np.ndarray) -> None:
        if self.selected < 0 or token != (self.selected, self._refs.tobytes()):
            return
        if not 0 <= index < len(self._crops):
            return
        ref = int(self._refs[index])
        self._crops[index] = self._crop_of[ref] = rgb
        self._read.add(ref)
        image = self._image_of[ref] = to_qimage(shown)
        self.picker.grid.set_image(index, image)
        if index == self._previewing:
            self._preview(index, force=True)

    def _preview(self, index: int, force: bool = False) -> None:
        """Show candidate `index` in place in the mosaic (-1: the tile shown now)."""
        if self.selected < 0 or index >= len(self._refs):
            index = -1
        result = self.session.project.matches
        if index >= 0 and result is not None and self._refs[index] == result.choice[self.selected]:
            index = -1
        if index == self._previewing and not force:
            return
        self._previewing = index
        if index < 0:
            self.overlays.show_preview(None, None)
        else:
            tile = int(self._tile_of[self.selected])
            inst = self._tile_layer.instances[[tile]].copy()
            inst["layer"] = 0.0
            inst["uv"] = (1.0, 0.0, 0.0, 1.0) if self._mirrored[index] else (0.0, 0.0, 1.0, 1.0)
            inst["offset"] = self._shifts[index]
            self.overlays.show_preview(self._crops[index], inst)
        self._show_selection()
        self._show_info()

    def _show_selection(self) -> None:
        """Outline the selected region; in Tiles view dim the rest (keeping the tile and
        what lies on it bright)."""
        r = self.selected
        if r < 0 or self._shown is None:
            return
        regions = self.session.project.matches.regions
        tiles_view = self.display_mode.currentData() == TILES
        bounds = self._shown.bounds if tiles_view else None
        self.overlays.show_selection(regions.center[r], regions.size[r], regions.rotation[r],
                                     bounds)  # fmt: skip
        tile = int(self._tile_of[r])
        if tiles_view:
            cover = upper_closure(self._shown, tile)
            if self._previewing >= 0:
                cover = cover[cover != tile]  # the preview takes its place
            self.overlays.show_cover(cover, self._tile_layer, bright=tile)
        else:
            self.overlays.show_cover(np.zeros(0, np.int64), None)

    def _show_info(self) -> None:
        r = self.selected
        if r < 0:
            return
        result = self.session.project.matches
        library = self.session.library
        shown = self._previewing if self._previewing >= 0 else None
        if shown is None:
            slot, mirrored = int(result.tile[r]), bool(result.mirrored[r])
        else:
            slot = int(result.candidates.crops(r, [self._refs[shown]])[0][0])
            mirrored = bool(self._mirrored[shown])
        info = self.picker.info
        full = file_name(library.paths([slot])[0]) + (" (mirrored)" if mirrored else "")
        info.setToolTip(full)
        width = info.contentsRect().width()
        if width < 60:  # not laid out yet (first selection): beside the target patch
            width = self.picker.contentsRect().width() - self.picker.target.width() - 30
        if shown is not None:
            label = self.picker.grid.items[shown].rank
            prefix = f"Preview: {label}. "
            name = html.escape(elide(info, full, width - info.fontMetrics().horizontalAdvance(
                prefix) - 8))  # fmt: skip
            lines = [f"<b>Preview:</b> {label}. {name}", "Click to pick it; Esc to cancel."]
        else:
            how = "picked by hand" if result.manual[r] else "the matcher's choice"
            error = result.quality.region_error[r]
            name = html.escape(elide(info, full, width, bold=True))
            lines = [f"<b>{name}</b>", f"ΔE {error:.1f} · {how}"]
            if self._feedback:
                lines.append(f"<span style='{PICK_STYLE}'>Picked: {self._feedback}</span>")
        info.setText("<br>".join(lines))
        self.picker.revert_button.setEnabled(bool(result.manual[r]))

    def _refresh_edit(self) -> None:
        session = self.session
        picker = self.picker
        can_edit = self.can_edit()
        if self.editing and not can_edit:
            self.set_editing(False)
            return
        for widget in (picker.edit_button, self.edit_action):
            widget.blockSignals(True)
            widget.setCheckable(True)
            widget.setChecked(self.editing)
            widget.blockSignals(False)
        picker.edit_button.setEnabled(can_edit)
        self.edit_action.setEnabled(can_edit or self.editing)
        self.escape_action.setEnabled(self.editing)
        for button, act, can, text in (
            (picker.undo_button, self.undo_action, session.can_undo_pick, "Undo pick"),
            (picker.redo_button, self.redo_action, session.can_redo_pick, "Redo pick"),
        ):
            button.setEnabled(can)
            act.setEnabled(can)
            keys = act.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
            button.setToolTip(f"{text} ({keys})")
        count = session.manual_picks
        picker.revert_all_button.setEnabled(bool(count) and can_edit)
        noun = "tile" if count == 1 else "tiles"
        if not can_edit and session.project.matches is None:
            hint = "Match tiles first; then pick any tile by hand here."
        elif not can_edit:
            hint = "Picking is unavailable while matching or updating the library."
        elif not self.editing:
            hint = ("Double-click a tile to choose another photo for it (or press "
                    "Edit Tiles, E).")  # fmt: skip
            if count:
                hint += f" {count:,} {noun} picked by hand so far."
        elif self.selected < 0:
            hint = "Click a tile to choose another photo for it. Drag to pan; Esc to finish."
        else:
            hint = ("Hover a candidate to preview it in place, click to pick it, "
                    "double-click to pick it and finish. "
                    "★ the matcher's choice; ! breaks a reuse rule.")  # fmt: skip
        picker.hint.setText(hint)
        self._show_markers()

    def _show_markers(self) -> None:
        result = self.session.project.matches
        if not self.editing or result is None or result.manual is None or self._shown is None:
            self.overlays.show_markers(make_instances(0))
            return
        tiles = self._tile_of[np.flatnonzero(result.manual)]
        self.overlays.show_markers(marker_instances(self._shown, tiles[tiles >= 0]))

    def _region_at(self, x: float, y: float) -> int:
        result = self.session.project.matches
        if result is None or self._shown is None:
            return -1
        return int(result.regions.hit_test(x, y))

    def _on_click(self, x: float, y: float) -> None:
        if not self.editing:
            return
        r = self._region_at(x, y)
        self.select(r if self.session.can_pick(r) else -1)

    def _on_double_click(self, sx: float, sy: float) -> None:
        """Double-clicking a tile edits it (entering edit mode if needed)."""
        x, y = self.viewer.canvas.camera.screen_to_world(sx, sy)
        r = self._region_at(float(x), float(y))
        if not self.session.can_pick(r):
            return
        if not self.editing:
            self.set_editing(True)
        if self.editing:
            self.select(r)

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
        if self.selected >= 0:
            self._previewing = -1
            self.overlays.show_preview(None, None)
            self._load_candidates()
            self._show_selection()
        self._refresh_edit()
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
        if fraction < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(fraction * 1000))

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
            self.select(-1)
            self._show_scene(session.scene)
        self._refresh_edit()
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
        self._tile_of = np.full(len(scene.result.tile), -1, np.int64)
        self._tile_of[scene.region] = np.arange(len(scene))
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
        if self.editing and mode == SOURCE:
            self.set_editing(False)
        self._show_selection()
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
        labels["picks"].setText(f"{result.manual_count:,}")
        labels["time"].setText(f"{sum(stats['timings'].values()):.1f} s")

    def _on_hover(self, x: float, y: float) -> None:
        library = self.session.library
        if self._shown is None or library is None:
            return
        result = self._shown.result
        index = result.regions.hit_test(x, y)
        if self.editing:
            canvas = self.viewer.canvas
            if index >= 0 and self.session.can_pick(index):
                regions = result.regions
                self.overlays.show_hover(regions.center[index], regions.size[index],
                                         regions.rotation[index])  # fmt: skip
                canvas.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.overlays.show_hover()
                canvas.unsetCursor()
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
                box = regions.bounds()
                (x0, y0), (x1, y1) = box[:, :2].min(axis=0), box[:, 2:].max(axis=0)
                self.viewer.set_content_bounds((x0, y0, x1 - x0, y1 - y0))
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
        self._reader.cancel()

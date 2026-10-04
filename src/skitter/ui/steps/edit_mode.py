"""Edit mode of the Matching tab: picking tiles by hand.

Click a tile (or double-click it, from outside edit mode), then choose
another photo for it from its remembered candidates in the side panel.
Hovering a candidate previews it in place; picks can be undone, reverted,
and kept when matching again (core/matching/edit.py does the picking).

`EditMode` owns the picker panel (tile_picker.py), its shortcuts and the
selection. It shows the mosaic the tab draws (its `scene_shown`,
`tile_layer` and `tile_of`) through the tab's overlays.
"""

import html

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence

from skitter.core.assembly import TileFiles
from skitter.core.matching import edit as picks
from skitter.ui.render.sprites import make_instances
from skitter.ui.steps.tile_picker import (
    CropReader,
    TilePicker,
    crop_size,
    elide,
    file_name,
    marker_instances,
    region_patch,
    thumb_candidate,
    tinted,
    to_qimage,
    upper_closure,
)
from skitter.ui.style import WARNING_STYLE
from skitter.ui.widgets.candidate_grid import CandidateItem

PICK_STYLE = "color: #3b9cff;"


class EditMode(QObject):
    """Picking tiles by hand on a MatchingStep (`step`)."""

    editing_changed = Signal(bool)

    def __init__(self, step):
        super().__init__(step)
        self.step = step
        self.session = step.session
        self.viewer = step.viewer
        self.overlays = step.overlays
        self.editing = False
        self.selected = -1  # region being edited (-1: none)
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
        self.picker = self._build_picker()

    # Calls from the tab

    def hover(self, region: int) -> None:
        """The cursor is over a region (-1: none): outline it if it can be picked."""
        if not self.editing:
            return
        canvas = self.viewer.canvas
        if region >= 0 and self.session.can_pick(region):
            regions = self.step.scene_shown.result.regions
            self.overlays.show_hover(regions.center[region], regions.size[region],
                                     regions.rotation[region])  # fmt: skip
            canvas.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.overlays.show_hover()
            canvas.unsetCursor()

    def display_changed(self) -> None:
        """The tab's display mode changed."""
        if self.editing and self.step.source_view:
            self.set_editing(False)
        self._show_selection()

    def mosaic_edited(self) -> None:
        """Picks changed some tiles (the tab has updated its layers)."""
        if self.selected >= 0:
            self._previewing = -1
            self.overlays.show_preview(None, None)
            self._load_candidates()
            self._show_selection()
        self.refresh()

    def shutdown(self) -> None:
        self._reader.cancel()

    # Construction

    def _build_picker(self) -> TilePicker:
        picker = TilePicker()
        session = self.session

        def action(text, slot, *keys) -> QAction:
            act = QAction(text, self.step)
            act.setShortcuts([QKeySequence(k) for k in keys])
            act.triggered.connect(slot)
            self.step.addAction(act)  # active while the tab shows
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
        picker.grid.hovered.connect(self.preview)
        picker.grid.activated.connect(self.pick)
        picker.grid.finished.connect(self.pick_and_finish)
        picker.grid.escaped.connect(self.escape)
        picker.show_selection(False)
        return picker

    def can_edit(self) -> bool:
        session = self.session
        result = session.project.matches
        return (
            session.mosaic_is_valid
            and session.busy not in ("matching", "library")
            and result is not None
            and result.candidates is not None
            and self.step.scene_shown is session.scene
            and self.step.tile_layer is not None
        )

    def set_editing(self, on: bool) -> None:
        on = bool(on) and self.can_edit()
        if on:
            self.step.leave_source_view()
        changed = on != self.editing
        self.editing = on
        if not on:
            self.select(-1)
            self.overlays.clear()
        self.editing_changed.emit(on)
        if changed:
            self.viewer.canvas.unsetCursor()
            self.step.status_message.emit("Edit mode: click a tile to pick another photo for it."
                                     if on else "")  # fmt: skip
        self.refresh()

    def escape(self) -> None:
        """Esc: stop previewing, then deselect, then leave edit mode."""
        if self._previewing >= 0:
            self.picker.grid.clear_preview()
            self.preview(-1)
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
            self.refresh()
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
        self.refresh()
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
        self.step.setCursor(Qt.CursorShape.WaitCursor)
        try:
            added = self.session.find_more_candidates(self.selected)
        finally:
            self.step.unsetCursor()
        if added:
            self._load_candidates()
            self.step.status_message.emit(f"Found {added} more candidates.")
        else:
            self.picker.more_button.setEnabled(False)
            self.step.status_message.emit("No other tiles fit this region.")

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
            self.preview(index, force=True)

    def preview(self, index: int, force: bool = False) -> None:
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
            tile = int(self.step.tile_of[self.selected])
            inst = self.step.tile_layer.instances[[tile]].copy()
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
        if r < 0 or self.step.scene_shown is None:
            return
        regions = self.session.project.matches.regions
        tiles_view = self.step.tiles_view
        bounds = self.step.scene_shown.bounds if tiles_view else None
        self.overlays.show_selection(regions.center[r], regions.size[r], regions.rotation[r],
                                     bounds)  # fmt: skip
        tile = int(self.step.tile_of[r])
        if tiles_view:
            cover = upper_closure(self.step.scene_shown, tile)
            if self._previewing >= 0:
                cover = cover[cover != tile]  # the preview takes its place
            self.overlays.show_cover(cover, self.step.tile_layer, bright=tile)
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

    def refresh(self) -> None:
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
        if (
            not self.editing
            or result is None
            or result.manual is None
            or self.step.scene_shown is None
        ):
            self.overlays.show_markers(make_instances(0))
            return
        tiles = self.step.tile_of[np.flatnonzero(result.manual)]
        self.overlays.show_markers(marker_instances(self.step.scene_shown, tiles[tiles >= 0]))

    def _region_at(self, x: float, y: float) -> int:
        result = self.session.project.matches
        if result is None or self.step.scene_shown is None:
            return -1
        return int(result.regions.hit_test(x, y))

    def on_click(self, x: float, y: float) -> None:
        if not self.editing:
            return
        r = self._region_at(x, y)
        self.select(r if self.session.can_pick(r) else -1)

    def on_double_click(self, sx: float, sy: float) -> None:
        """Double-clicking a tile edits it (entering edit mode if needed)."""
        x, y = self.viewer.canvas.camera.screen_to_world(sx, sy)
        r = self._region_at(float(x), float(y))
        if not self.session.can_pick(r):
            return
        if not self.editing:
            self.set_editing(True)
        if self.editing:
            self.select(r)

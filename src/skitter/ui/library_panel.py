"""The tile library's panel, shared by every window that manages the library (the Tiles
tab, the Infinite Mosaic's Library tab).

The user picks folders; Update reads new and changed images into the
library cache in the background (see core/tiles/library.py). The Folders
tree shows the library's folders with their photo counts: unticking one
leaves its photos out of matching at once (nothing is read again); a root
can be forgotten (dropped from the library) or, when it moved or its drive
is unplugged (offline), located at its new place. Contents shows the
library's statistics, and Unreadable files what the last update could not
read.

Every instance works on the app `Session`'s library and follows its signals,
so all of them show the same library and an edit in one shows in the others.
Library features belong here, so each window gets them. A window whose work
depends on the library as it is (a running Infinite Mosaic session) locks the
panel (`set_locked`): the library can be looked at but not changed.
"""

import os
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMenu,
    QMessageBox,
    QPushButton,
    QStyle,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from skitter.core.tiles.palette import photo_stats
from skitter.ui.style import fixed_width, format_bytes, muted, progress_bar, show_progress
from skitter.ui.widgets.height_grip import HeightGrip

PATH_ROLE = Qt.ItemDataRole.UserRole  # a folder item's path (library folder_key form)
ROOT_ROLE = Qt.ItemDataRole.UserRole + 1  # a root item's root, as the library lists it
CHECKS = {"on": Qt.CheckState.Checked, "off": Qt.CheckState.Unchecked,
          "partial": Qt.CheckState.PartiallyChecked}  # fmt: skip


def _disk_subfolders(path: str, first: bool = False) -> list[str]:
    """The subfolders of a folder on disk, in the library's path form, sorted (first:
    stop at the first found, to tell whether there are any). Empty if it isn't there."""
    found = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        found.append(os.path.normcase(os.path.abspath(entry.path)))
                        if first:
                            break
                except OSError:
                    continue
    except OSError:
        return []
    return sorted(found)


def _natural(path: str) -> list:
    """Sort key: numbers in names compare as numbers (folder 2 before folder 10)."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]


def _display_name(path: str) -> str:
    """A folder's name as the disk spells it (the library keeps paths case-folded)."""
    try:
        real = os.path.realpath(path)
    except OSError:
        real = path
    return os.path.basename(real.rstrip(os.sep)) or real


class LibraryPanel(QWidget):
    """The library's folders, updating, contents and unreadable files, for a side panel.

    tile_aspect: the host's tile shape (width / height), for Contents' Sharp up
    to; call `refresh_stats` when it changes."""

    def __init__(self, session, tile_aspect: Callable[[], float], parent=None):
        super().__init__(parent)
        self.session = session
        self.tile_aspect = tile_aspect
        self._folders_key: tuple | None = None  # what the folder tree shows
        self._locked: str | None = None  # why the library can't be changed, if it can't
        self.lock_note = muted(QLabel())
        self.lock_note.setWordWrap(True)
        self.lock_note.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.lock_note)
        for group in (self._build_folders_group(), self._build_update_group(),
                      self._build_stats_group(), self._build_errors_group()):  # fmt: skip
            layout.addWidget(group)
        session.library_changed.connect(self.refresh)
        session.library_progress.connect(self._on_progress)
        session.busy_changed.connect(self.refresh)
        self.refresh()

    # Construction

    def _build_folders_group(self) -> QGroupBox:
        self.folders = QTreeWidget()
        self.folders.setColumnCount(2)
        self.folders.setHeaderLabels(["Folder", "Photos"])
        self.folders_grip = HeightGrip(self.folders, "tiles/folders_height", default=200)
        header = self.folders.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.folders.itemExpanded.connect(self._fill_folder)
        self.folders.itemChanged.connect(self._on_folder_checked)
        self.folders.currentItemChanged.connect(lambda *_: self._update_folder_buttons())
        self.folders.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.folders.customContextMenuRequested.connect(self._folder_menu)
        self.add_button = QPushButton("Add Folder…")
        self.add_button.clicked.connect(self.add_folder_dialog)
        self.locate_button = QPushButton("Locate…")
        self.locate_button.setToolTip(
            "The folder moved, or its drive has a new letter: point "
            "the library at its new place (nothing is read again)."
        )
        self.locate_button.clicked.connect(lambda: self.locate_folder())
        self.forget_button = QPushButton("Forget…")
        self.forget_button.setToolTip(
            "Drop the folder and its photos from the library (the files stay on disk)."
        )
        self.forget_button.clicked.connect(lambda: self.forget_folder())
        buttons = QHBoxLayout()
        buttons.addWidget(self.add_button)
        buttons.addStretch()
        buttons.addWidget(self.locate_button)
        buttons.addWidget(self.forget_button)
        group = QGroupBox("Folders")
        layout = QVBoxLayout(group)
        layout.addWidget(self.folders)
        layout.addWidget(self.folders_grip)
        layout.addLayout(buttons)
        layout.addWidget(muted(QLabel("Photos in ticked folders are used as tiles. Unticking "
                                      "is instant: nothing is read again.")))  # fmt: skip
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
        self.status.setWordWrap(True)
        group = QGroupBox("Library")
        layout = QVBoxLayout(group)
        layout.addLayout(buttons)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
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
            "being enlarged, for the tile shape. Larger tiles show some photos softer than "
            "they could be."
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

    def open_library(self) -> None:
        """Open the library if the session has none open yet."""
        if self.session.library is None:  # not `not`: an empty library is falsy
            self.session.open_library()

    def set_locked(self, reason: str | None) -> None:
        """Keep the library from being changed here, saying why (None: allow it again)."""
        if reason == self._locked:
            return
        self._locked = reason
        self.lock_note.setText(reason or "")
        self.lock_note.setVisible(reason is not None)
        self._folders_key = None  # the tree's tick boxes follow
        self.refresh()

    @property
    def locked(self) -> bool:
        return self._locked is not None

    def add_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add Tile Folder")
        if folder:
            self.add_folder(folder)

    def add_folder(self, folder) -> bool:
        if self.locked:
            return False
        self.open_library()
        self.session.add_library_folder(folder)
        return True

    def selected_root(self):
        """The library root of the current item's tree (None: nothing selected)."""
        item = self.folders.currentItem()
        while item is not None and item.parent() is not None:
            item = item.parent()
        return None if item is None else item.data(0, ROOT_ROLE)

    def forget_folder(self, confirm: bool = True) -> bool:
        """Drop the selected root and its photos from the library (asking first)."""
        root, session = self.selected_root(), self.session
        if root is None or session.busy or self.locked:
            return False
        if confirm:
            known = session.library.folder_info(root).known
            question = (
                f"Forget {root}?\n\nThe library's {known:,} photos from it are dropped (the "
                "files stay on disk); adding the folder again reads them all again."
                if known
                else f"Forget {root}? The library has no photos from it yet."
            )
            used = session.mosaic_uses_folder(root)
            if used:
                question += (
                    f"\n\n{used:,} tiles of the current mosaic show its photos: "
                    "it will need matching again."
                )
            answer = QMessageBox.question(self, "Forget Folder", question)
            if answer != QMessageBox.StandardButton.Yes:
                return False
        session.forget_library_folder(root)
        return True

    def locate_folder(self, new_place=None) -> bool:
        """Point the selected root at its new place (asking for it, unless given)."""
        root, session = self.selected_root(), self.session
        if root is None or session.busy or self.locked:
            return False
        if new_place is None:
            new_place = QFileDialog.getExistingDirectory(self, f"Locate {root}")
            if not new_place:
                return False
        try:
            moved = session.relink_library_folder(root, new_place)
        except ValueError as exc:
            QMessageBox.warning(self, "Locate Folder", f"Not moved: {exc}.")
            return False
        self.status.setText(f"Found {moved:,} photos at {new_place}.")
        return True

    def update_library(self) -> bool:
        if self.session.library is None or self.session.busy or self.locked:
            return False
        self.progress.setRange(0, 0)
        self.status.setText("Starting…")
        self.session.update_library()
        return True

    # Display

    def _on_progress(self, message: str, fraction: float) -> None:
        self.status.setText(message)
        show_progress(self.progress, fraction)

    def refresh(self) -> None:
        session = self.session
        library = session.library
        busy = session.busy
        updating = busy == "library"
        editable = not busy and not self.locked
        self._fill_folders()
        self.folders.setEnabled(library is not None and not updating)
        has_folders = library is not None and bool(library.roots)
        self.add_button.setEnabled(editable)
        self._update_folder_buttons()
        self.update_button.setEnabled(has_folders and editable)
        self.cancel_button.setEnabled(updating)
        self.progress.setVisible(updating)

        if library is None:
            for label in (self._tiles, self._failed, self._missing, self._disk, self._location):
                label.setText("—")
        else:
            counts = library.counts()
            used = len(library)
            self._tiles.setText(f"{counts['ok']:,}" if used == counts["ok"]
                                else f"{used:,} in use of {counts['ok']:,}")  # fmt: skip
            self._failed.setText(f"{counts['failed']:,}")
            self._missing.setText(f"{counts['missing']:,}")
            self._disk.setText(format_bytes(library.disk_bytes()))
            self._location.setText(str(library.folder))
        self.refresh_stats()

        report, error = session.library_report, session.library_error
        if not updating:
            if error:
                self.status.setText(f"Update failed: {error}")
            elif report is not None:
                text = (f"Added {report.added:,}, updated {report.updated:,}, "
                        f"{report.unchanged:,} unchanged, {report.missing:,} missing, "
                        f"{report.failed:,} unreadable.")  # fmt: skip
                if report.skipped:
                    text += f" {report.skipped:,} new in unticked folders, not read."
                if report.offline:
                    count = len(report.offline)
                    text += (
                        f" {count} folder{'s' if count > 1 else ''} offline, skipped: "
                        "plug the drive in, or Locate… a moved folder."
                    )
                self.status.setText(("Cancelled. " if report.cancelled else "") + text)
            elif library is not None and not has_folders:
                self.status.setText("Add a folder to begin.")
            if library is not None and any(
                library.folder_info(r).known == 0 for r in library.roots if library.is_online(r)
            ):
                self.status.setText("Press Update Library to read the photos of new folders.")
        self.errors.clear()
        if report is not None and report.errors:
            self.errors.addItems([f"{path}: {message}" for path, message in report.errors])
        self._errors_group.setVisible(self.errors.count() > 0)

    def refresh_stats(self) -> None:
        library = self.session.library
        ids = library.ids if library is not None else np.zeros(0, np.int64)
        if not len(ids):
            for label in (self._photo_size, self._shapes, self._detail):
                label.setText("—")
            return
        stats = photo_stats(library.width[ids], library.height[ids], float(self.tile_aspect()))
        self._photo_size.setText(f"{stats.median_side:,} px (smallest {stats.small_side:,} px)")
        self._shapes.setText(f"{stats.portrait:.0%} portrait, {stats.landscape:.0%} landscape, "
                             f"{stats.square:.0%} square")  # fmt: skip
        self._detail.setText(f"{stats.detail_px:,} px tiles")

    # Folder tree

    def _fill_folders(self) -> None:
        """Show the library's roots, keeping what was expanded and selected (rebuilt only
        when the folders, their contents or what's in use changed)."""
        tree, library = self.folders, self.session.library
        key = None if library is None else (id(library), library.state, tuple(
            (root, library.is_online(root)) for root in library.roots), self.locked)  # fmt: skip
        if key == self._folders_key:
            return
        self._folders_key = key
        expanded, current = set(), None
        stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item.isExpanded():
                expanded.add(item.data(0, PATH_ROLE))
            stack.extend(item.child(i) for i in range(item.childCount()))
        if tree.currentItem() is not None:
            current = tree.currentItem().data(0, PATH_ROLE)
        tree.blockSignals(True)
        tree.clear()
        if library is not None:
            warning = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            for root in library.roots:
                item = self._folder_item(library.folder_info(root), str(root))
                item.setData(0, ROOT_ROLE, root)
                if not library.is_online(root):
                    item.setIcon(0, warning)
                    item.setText(1, "offline")
                    item.setToolTip(0, f"{root} isn't there: plug its drive in, or Locate… "
                                       "it if it moved. Its photos are kept.")  # fmt: skip
                tree.addTopLevelItem(item)
        tree.blockSignals(False)
        self._expand(expanded, current)

    def _expand(self, paths: set, current) -> None:
        stack = [self.folders.topLevelItem(i) for i in range(self.folders.topLevelItemCount())]
        while stack:
            item = stack.pop()
            path = item.data(0, PATH_ROLE)
            if path == current:
                self.folders.setCurrentItem(item)
            if path in paths:
                item.setExpanded(True)  # fills it (itemExpanded)
                stack.extend(item.child(i) for i in range(item.childCount()))

    def _folder_item(self, info, label: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([label, self._count_text(info)])
        item.setData(0, PATH_ROLE, info.path)
        if not self.locked:  # locked: the tick shows, but can't be changed
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        else:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, CHECKS[info.state])
        item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        item.setToolTip(0, info.path)
        item.setToolTip(1, self._count_tip(info))
        if info.children or _disk_subfolders(info.path, first=True):
            item.addChild(QTreeWidgetItem())  # a placeholder until expanded
        return item

    @staticmethod
    def _count_text(info) -> str:
        if info.photos == 0:
            if info.missing:
                return f"{info.missing:,} missing"
            return "not read yet" if info.included else "not read"
        if info.used == info.photos:
            return f"{info.photos:,}"
        return f"{info.used:,} of {info.photos:,}"

    @staticmethod
    def _count_tip(info) -> str:
        if info.known == 0:
            if not info.included:
                return "Unticked before it was read: Update Library skips it until ticked."
            return "Nothing read from this folder yet: press Update Library."
        lines = [f"{info.photos:,} photos read, {info.used:,} of them in use"]
        if info.missing:
            lines.append(f"{info.missing:,} not found by the last update: they come back "
                         "without reading if the files return unchanged")  # fmt: skip
        if info.known > info.photos + info.missing:
            lines.append(f"{info.known - info.photos - info.missing:,} unreadable")
        return "\n".join(lines)

    def _fill_folder(self, item: QTreeWidgetItem) -> None:
        """Make an expanded folder's subfolder items (once)."""
        if item.childCount() != 1 or item.child(0).data(0, PATH_ROLE) is not None:
            return
        library = self.session.library
        self.folders.blockSignals(True)
        item.takeChildren()
        info = library.folder_info(item.data(0, PATH_ROLE))
        # Its subfolders on disk too: ones not read yet (new, or left out before their
        # first update) can be browsed and ticked like the rest.
        children = set(info.children) | set(_disk_subfolders(info.path))
        for path in sorted(children, key=_natural):  # 2 before 10
            item.addChild(self._folder_item(library.folder_info(path), _display_name(path)))
        self.folders.blockSignals(False)

    def _on_folder_checked(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0 or self.session.library is None or self.locked:
            return
        on = item.checkState(0) != Qt.CheckState.Unchecked
        path = item.data(0, PATH_ROLE)
        # Next turn: the change rebuilds the tree, which mustn't happen inside its signal.
        QTimer.singleShot(0, lambda: self.session.set_folder_included(path, on))

    def _update_folder_buttons(self) -> None:
        enabled = self.selected_root() is not None and not self.session.busy and not self.locked
        self.locate_button.setEnabled(enabled)
        self.forget_button.setEnabled(enabled)

    def _folder_menu(self, position) -> None:
        item = self.folders.itemAt(position)
        if item is None:
            return
        self.folders.setCurrentItem(item)
        menu = QMenu(self)
        path = item.data(0, PATH_ROLE)
        if Path(path).is_dir():
            menu.addAction("Show in Folder", lambda: os.startfile(path)
                           if hasattr(os, "startfile") else None)  # fmt: skip
        if item.parent() is None:
            editable = not self.session.busy and not self.locked
            menu.addSeparator()
            menu.addAction("Locate…", lambda: self.locate_folder()).setEnabled(editable)
            menu.addAction("Forget…", lambda: self.forget_folder()).setEnabled(editable)
        menu.popup(self.folders.viewport().mapToGlobal(position))

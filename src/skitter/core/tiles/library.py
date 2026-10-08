"""The tile library: an on-disk cache of every candidate tile image.

A library lives in a folder of its own:

- `library.sqlite`: the root folders scanned, and one row per image file
  (path, size and modification time, upright pixel size, status).
- `thumbs.u8`: a memory-mapped (capacity, 32, 32, 3) uint8 array of analysis
  thumbnails, one slot per image row (the thumbnail fills its top-left
  corner). At 500,000 tiles this is about 1.5 GB on disk, paged in as needed.

Updating is incremental: a rescan reads only new or changed files, keeps
failures recorded (and doesn't retry them until the file changes), and marks
vanished files missing without reusing their slots. `version` changes with
every change to the contents, so derived data (descriptors, indexes) can be
cached by it.

Folders: the roots are scanned with their subfolders. A root that isn't
there when updating (an unplugged drive, a renamed folder) is offline: its
photos are left as they were, not marked missing; `relink` points it at its
new place, keeping every photo read. `forget` drops a root and its photos.
Any folder can be left out of matching (`set_included`): its photos stay
read, but `ids` and `len()` count only the photos in use, and `selection`
changes (not `version`). `readable_ids` are all the photos read.

A library is used from one thread at a time.
"""

import os
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from skitter.core.imaging import IMAGE_EXTENSIONS
from skitter.core.tiles.ingest import THUMB, Thumbnail, load_thumbnails

OK, FAILED, MISSING = 1, 2, 3
_COMMIT_EVERY = 2048

_SCHEMA = """
CREATE TABLE IF NOT EXISTS roots (path TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS folder_rules (path TEXT PRIMARY KEY, included INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS tiles (
    slot INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    tw INTEGER NOT NULL DEFAULT 0,
    th INTEGER NOT NULL DEFAULT 0,
    status INTEGER NOT NULL,
    error TEXT
);
"""

Progress = Callable[[str, int, int], None]  # (phase, done, total)
RELINK_SAMPLE = 20  # photos checked at a folder's new place before relinking


def folder_key(path) -> str:
    """A folder or file path as the library stores it (absolute, normalized case)."""
    return os.path.normcase(str(Path(path).resolve()))


def is_under(path: str, folder: str) -> bool:
    """Whether path (folder_key form) is folder or inside it."""
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


@dataclass(frozen=True)
class FolderInfo:
    """A folder of the library as the folder tree shows it."""

    path: str  # folder_key form
    photos: int  # readable photos in it and its subfolders
    used: int  # of those, in use (not left out)
    included: bool  # whether its own photos are in use (by its rule or its parent's)
    children: tuple[str, ...]  # subfolders the library knows photos in, sorted
    missing: int = 0  # photos read once but not found by the last update
    known: int = 0  # every photo the library has a row for (readable, missing, unreadable)
    mixed: bool = False  # a subfolder is set the other way (in use, or left out)

    @property
    def state(self) -> str:
        """ "on", "off" or "partial" (some of its photos, or subfolders, left out)."""
        if self.mixed:
            return "partial"
        if self.photos == 0:
            return "on" if self.included else "off"
        return "on" if self.used == self.photos else "off" if self.used == 0 else "partial"


@dataclass
class UpdateReport:
    added: int = 0
    updated: int = 0
    failed: int = 0
    missing: int = 0
    unchanged: int = 0
    offline: list[str] = field(default_factory=list)  # roots that weren't there (skipped)
    # New photos not read, being in folders left out (folders left out that were never
    # read aren't even scanned: their photos aren't counted).
    skipped: int = 0
    cancelled: bool = False
    errors: list[tuple[str, str]] = field(default_factory=list)  # first few (path, error)


def default_library_folder() -> Path:
    """Per-user cache location for the tile library."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "Skitter" / "library"


def walk_images(
    root: Path, skip: Callable[[str], bool] = lambda folder: False
) -> Iterator[tuple[str, int, float]]:
    """(path, size, mtime) of every image file under root (os.scandir: one stat each).
    skip(folder) (library path form) leaves a subfolder out, with all inside it."""
    stack = [str(root)]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if not skip(os.path.normcase(os.path.abspath(entry.path))):
                        stack.append(entry.path)
                elif os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS:
                    st = entry.stat()
                    yield os.path.normcase(os.path.abspath(entry.path)), st.st_size, st.st_mtime
            except OSError:
                continue


class TileLibrary:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.folder / "library.sqlite", check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._thumbs_path = self.folder / "thumbs.u8"
        self._thumbs: np.memmap | None = None
        self.closed = False  # close() was called: nothing can be read any more
        self._reload()

    def close(self) -> None:
        self._thumbs = None
        self._db.close()
        self.closed = True

    # Roots

    @property
    def roots(self) -> list[Path]:
        return [Path(p) for (p,) in self._db.execute("SELECT path FROM roots ORDER BY path")]

    def set_roots(self, roots) -> None:
        with self._db:
            self._db.execute("DELETE FROM roots")
            self._db.executemany(
                "INSERT INTO roots VALUES (?)", [(str(Path(r).resolve()),) for r in roots]
            )
        self._select()  # the folder tree hangs off the roots

    def add_root(self, folder) -> None:
        """Add a folder to scan. One inside a root is put back in use instead (it's
        scanned already); one holding roots replaces them (their photos stay)."""
        key = folder_key(folder)
        roots = self.roots
        if any(is_under(key, folder_key(root)) for root in roots):
            self.set_included(folder, True)
            return
        kept = [root for root in roots if not is_under(folder_key(root), key)]
        self.set_roots([*kept, folder])

    def is_online(self, root) -> bool:
        """Whether a root folder is there to scan."""
        return Path(root).is_dir()

    def forget(self, root) -> int:
        """Stop scanning a root and drop its photos from the library (mosaics that use
        them need matching again). Returns how many photos were dropped."""
        key = folder_key(root)
        slots = self._slots_under(key)
        with self._db:
            self._db.executemany("DELETE FROM tiles WHERE slot = ?", [(int(s),) for s in slots])
            self._db.execute("DELETE FROM roots WHERE path = ?", (str(Path(root).resolve()),))
            self._delete_rules_under(key)
            self._bump_version()
        self._reload()
        return len(slots)

    def relink(self, root, new_root) -> int:
        """Point a root that moved at its new place, keeping every photo read (their
        paths change). ValueError if the photos aren't found there. Returns how many
        photos moved."""
        old, new = folder_key(root), folder_key(new_root)
        if not Path(new_root).is_dir():
            raise ValueError(f"{new_root} is not a folder")
        slots = self._slots_under(old)
        ok = slots[self.status[slots] == OK]
        sample = ok[np.linspace(0, len(ok) - 1, min(len(ok), RELINK_SAMPLE)).astype(int)]
        if len(sample):
            found = sum(os.path.exists(new + self._paths[s][len(old) :]) for s in sample)
            if found < max(1, len(sample) // 2):
                raise ValueError(f"the photos of {root} are not in {new_root} "
                                 f"({found} of {len(sample)} checked found)")  # fmt: skip
        rows = [(new + self._paths[s][len(old) :], int(s)) for s in slots]
        rules = self._rules()
        with self._db:
            self._db.executemany("UPDATE tiles SET path = ? WHERE slot = ?", rows)
            moved = (str(Path(new_root).resolve()), str(Path(root).resolve()))
            self._db.execute("UPDATE roots SET path = ? WHERE path = ?", moved)
            self._delete_rules_under(old)
            self._db.executemany(
                "INSERT OR REPLACE INTO folder_rules VALUES (?, ?)",
                [(new + path[len(old):], int(on)) for path, on in rules.items()
                 if is_under(path, old)],
            )  # fmt: skip
            self._bump_version()
        self._reload()
        return len(slots)

    # Folders in use

    def set_included(self, folder, included: bool) -> None:
        """Use (or leave out of matching) the photos in folder and all its subfolders."""
        key = folder_key(folder)
        with self._db:
            self._delete_rules_under(key)
            if self._rule_for(key, self._rules()) != included:
                self._db.execute("INSERT INTO folder_rules VALUES (?, ?)", (key, int(included)))
        self._select()

    def folder_info(self, path) -> FolderInfo:
        """A folder of the tree: a root, or a folder inside one."""
        key = folder_key(path)
        photos, used, missing, known, children = self._folders.get(key, (0, 0, 0, 0, ()))
        rules = self._rules()
        included = self._rule_for(key, rules)
        mixed = any(on != included for path, on in rules.items()
                    if path != key and is_under(path, key))  # fmt: skip
        return FolderInfo(key, photos, used, included, tuple(sorted(children)), missing, known,
                          mixed)  # fmt: skip

    @property
    def selection(self) -> tuple:
        """What's in use, comparable (cache derived data by (version, selection))."""
        return tuple(sorted(self._rules().items()))

    def _rules(self) -> dict[str, bool]:
        return {path: bool(on) for path, on in self._db.execute("SELECT * FROM folder_rules")}

    @staticmethod
    def _rule_for(folder: str, rules: dict[str, bool]) -> bool:
        """Whether a folder is in use: its own rule, else its nearest parent's (default on)."""
        path = folder
        while True:
            if path in rules:
                return rules[path]
            parent = os.path.dirname(path)
            if parent == path:
                return True
            path = parent

    def _delete_rules_under(self, folder: str) -> None:
        self._db.executemany(
            "DELETE FROM folder_rules WHERE path = ?",
            [(path,) for path in self._rules() if is_under(path, folder)],
        )

    def _slots_under(self, folder: str) -> np.ndarray:
        return np.array([s for s, p in enumerate(self._paths) if p and is_under(p, folder)],
                        np.int64)  # fmt: skip

    def _bump_version(self) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta VALUES ('version', ?)",
                         (str(self.version + 1),))  # fmt: skip

    def _index_dirs(self) -> None:
        """Each slot's folder (paths change only on reload)."""
        dirs: dict[str, int] = {}
        self._dir_of = np.fromiter(
            (dirs.setdefault(p.rpartition(os.sep)[0], len(dirs)) for p in self._paths),
            np.int64, len(self._paths),
        )  # fmt: skip
        self._dirs = dirs

    def _select(self) -> None:
        """Work out which photos are in use, and the folder tree's counts."""
        rules = self._rules()
        dirs, dir_of = self._dirs, self._dir_of
        known: dict[str, bool] = {}  # folder -> in use, filled walking up

        def in_use(folder: str) -> bool:
            chain, path = [], folder
            while path not in known and path not in rules:
                chain.append(path)
                parent = os.path.dirname(path)
                if parent == path:
                    break
                path = parent
            on = known.get(path, rules.get(path, True))
            for p in chain:
                known[p] = on
            return on

        on = np.array([in_use(d) for d in dirs], bool)
        self.in_use = on[dir_of] if len(dir_of) else np.zeros(0, bool)
        readable = self.status == OK
        used = readable & self.in_use
        counts = np.stack([np.bincount(dir_of[mask], minlength=len(dirs)) for mask in
                           (readable, used, self._row & (self.status == MISSING), self._row)],
                          axis=1)  # fmt: skip
        roots = [folder_key(root) for root in self.roots]
        folders: dict[str, list] = {root: [np.zeros(4, np.int64), set()] for root in roots}
        for d, i in dirs.items():
            if not counts[i, 3]:  # no photo known here
                continue
            root = next((r for r in roots if is_under(d, r)), None)
            if root is None:
                continue
            path = d
            while True:
                entry = folders.setdefault(path, [np.zeros(4, np.int64), set()])
                entry[0] += counts[i]
                if path == root:
                    break
                parent = os.path.dirname(path)
                folders.setdefault(parent, [np.zeros(4, np.int64), set()])[1].add(path)
                path = parent
        self._folders = {path: (*(int(v) for v in c), tuple(children))
                         for path, (c, children) in folders.items()}  # fmt: skip

    # Contents (arrays indexed by slot)

    @property
    def version(self) -> int:
        row = self._db.execute("SELECT value FROM meta WHERE key = 'version'").fetchone()
        return int(row[0]) if row else 0

    def __len__(self) -> int:
        """Number of tiles in use (readable, in a folder not left out)."""
        return int(np.count_nonzero((self.status == OK) & self.in_use))

    @property
    def ids(self) -> np.ndarray:
        """Slots of tiles in use (readable, in a folder not left out), ascending."""
        return np.flatnonzero((self.status == OK) & self.in_use)

    @property
    def readable_ids(self) -> np.ndarray:
        """Slots of every readable tile, in use or not, ascending."""
        return np.flatnonzero(self.status == OK)

    @property
    def state(self) -> tuple:
        """(version, selection): changes whenever `ids` or their pictures may have."""
        return self.version, self.selection

    @property
    def thumbs(self) -> np.ndarray:
        """(slots, 32, 32, 3) uint8 thumbnails (memory-mapped, read-only use)."""
        if self._thumbs is None:
            return np.zeros((0, THUMB, THUMB, 3), np.uint8)
        return self._thumbs[: len(self.status)]

    def paths(self, slots) -> list[str]:
        return [self._paths[i] for i in np.asarray(slots, dtype=np.int64)]

    def counts(self) -> dict[str, int]:
        return {
            "ok": int(np.count_nonzero(self.status == OK)),
            "failed": int(np.count_nonzero(self.status == FAILED)),
            "missing": int(np.count_nonzero(self.status == MISSING)),
        }

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.folder.iterdir() if p.is_file())

    def _reload(self) -> None:
        rows = self._db.execute(
            "SELECT slot, path, width, height, tw, th, status FROM tiles ORDER BY slot"
        ).fetchall()
        n = max(rows[-1][0] + 1 if rows else 0, self._next_slot())
        self._paths = [""] * n
        self.width = np.zeros(n, np.int64)  # upright full-size image width
        self.height = np.zeros(n, np.int64)
        self.thumb_size = np.zeros((n, 2), np.int64)  # (w, h) of each thumbnail
        self.status = np.full(n, MISSING, np.int8)
        self._row = np.zeros(n, bool)  # slots with a row (not free or forgotten)
        for slot, path, w, h, tw, th, status in rows:
            self._row[slot] = True
            self._paths[slot] = path
            self.width[slot], self.height[slot] = w, h
            self.thumb_size[slot] = (tw, th)
            self.status[slot] = status
        self._open_thumbs(n)
        self._index_dirs()
        self._select()

    def _next_slot(self) -> int:
        """Slots below this have been used (forgotten photos' slots aren't reused)."""
        row = self._db.execute("SELECT value FROM meta WHERE key = 'next_slot'").fetchone()
        return int(row[0]) if row else 0

    def _open_thumbs(self, slots: int) -> None:
        item = THUMB * THUMB * 3
        have = self._thumbs_path.stat().st_size // item if self._thumbs_path.exists() else 0
        capacity = max(have, 1024)
        while capacity < slots:
            capacity *= 2
        if capacity != have or self._thumbs is None:
            self._thumbs = None
            with open(self._thumbs_path, "ab") as f:
                f.truncate(capacity * item)
            self._thumbs = np.memmap(
                self._thumbs_path, np.uint8, "r+", shape=(capacity, THUMB, THUMB, 3)
            )

    # Updating

    def update(
        self,
        workers: int | None = None,
        progress: Progress = lambda phase, done, total: None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> UpdateReport:
        """Bring the library up to date with its roots."""
        report = UpdateReport()
        known = {
            path: (slot, size, mtime, status)
            for slot, path, size, mtime, status in self._db.execute(
                "SELECT slot, path, size, mtime, status FROM tiles"
            )
        }
        # Missing photos read before keep their thumbnails: back unchanged, they're not read.
        was_read = {slot for (slot,) in self._db.execute(
            "SELECT slot FROM tiles WHERE status = ? AND width > 0", (MISSING,))}  # fmt: skip
        revived: list[int] = []
        found: dict[str, tuple[int, float]] = {}
        offline = []
        rules = self._rules()
        put_back = [path for path, on in rules.items() if on]
        used: dict[str, bool] = {}  # folder -> in use, as met

        def in_use(folder: str) -> bool:
            if folder not in used:
                used[folder] = self._rule_for(folder, rules)
            return used[folder]

        def unread_and_left_out(folder: str) -> bool:
            """Not worth scanning: left out, nothing read from it, nothing inside put back."""
            return (not in_use(folder) and self._folders.get(folder, (0,) * 4)[3] == 0
                    and not any(is_under(p, folder) for p in put_back))  # fmt: skip

        for root in self.roots:
            if not self.is_online(root):
                offline.append(folder_key(root))
                report.offline.append(str(root))
                continue
            if unread_and_left_out(folder_key(root)):
                continue  # left out before its first read
            for path, size, mtime in walk_images(root, unread_and_left_out):
                if path not in known and not in_use(path.rpartition(os.sep)[0]):
                    report.skipped += 1  # new, in a folder left out: not worth reading
                    continue
                found[path] = (size, mtime)
                if len(found) % 5000 == 0:
                    progress("Scanning", len(found), 0)
                    if cancelled():
                        report.cancelled = True
                        return report

        todo: list[tuple[str, int]] = []  # (path, slot)
        next_slot = max((max(s for s, *_ in known.values()) + 1) if known else 0,
                        self._next_slot())  # fmt: skip
        for path, (size, mtime) in found.items():
            old = known.get(path)
            if old is None:
                todo.append((path, next_slot))
                next_slot += 1
                report.added += 1
            elif old[3] == MISSING and old[1:3] == (size, mtime) and old[0] in was_read:
                revived.append(old[0])
                report.unchanged += 1
            elif old[1:3] != (size, mtime) or old[3] == MISSING:
                todo.append((path, old[0]))
                report.updated += 1
            else:
                report.unchanged += 1
        gone = [slot for path, (slot, *_, status) in known.items()
                if path not in found and status != MISSING
                and not any(is_under(path, root) for root in offline)]  # fmt: skip
        report.missing = len(gone)

        self._open_thumbs(next_slot)
        thumbs = self._thumbs
        rows = []
        total = len(todo)
        progress("Reading", 0, total)
        try:
            results = load_thumbnails([p for p, _ in todo], workers=workers, cancelled=cancelled)
            for done, (i, result) in enumerate(results, 1):
                path, slot = todo[i]
                size, mtime = found[path]
                if isinstance(result, Thumbnail):
                    th, tw = result.pixels.shape[:2]
                    thumbs[slot] = 0
                    thumbs[slot, :th, :tw] = result.pixels
                    rows.append((slot, path, size, mtime, result.width, result.height,
                                 tw, th, OK, None))  # fmt: skip
                else:
                    report.failed += 1
                    if len(report.errors) < 20:
                        report.errors.append((path, result))
                    rows.append((slot, path, size, mtime, 0, 0, 0, 0, FAILED, result))
                if len(rows) >= _COMMIT_EVERY:
                    self._write(rows)
                    rows = []
                    progress("Reading", done, total)
            report.cancelled = cancelled()
        finally:
            self._write(rows)
            if gone and not report.cancelled:
                with self._db:
                    self._db.executemany(
                        "UPDATE tiles SET status = ? WHERE slot = ?", [(MISSING, s) for s in gone]
                    )
            with self._db:
                self._db.executemany("UPDATE tiles SET status = ? WHERE slot = ?",
                                     [(OK, s) for s in revived])  # fmt: skip
                self._db.execute("INSERT OR REPLACE INTO meta VALUES ('next_slot', ?)",
                                 (str(max(next_slot, self._next_slot())),))  # fmt: skip
                if todo or gone or revived:
                    self._bump_version()
            self._reload()
        progress("Reading", total, total)
        return report

    def _write(self, rows) -> None:
        if not rows:
            return
        self._thumbs.flush()
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO tiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )

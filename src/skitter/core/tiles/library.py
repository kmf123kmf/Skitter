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
every update, so derived data (descriptors, indexes) can be cached by it.

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


@dataclass
class UpdateReport:
    added: int = 0
    updated: int = 0
    failed: int = 0
    missing: int = 0
    unchanged: int = 0
    cancelled: bool = False
    errors: list[tuple[str, str]] = field(default_factory=list)  # first few (path, error)


def default_library_folder() -> Path:
    """Per-user cache location for the tile library."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "Skitter" / "library"


def walk_images(root: Path) -> Iterator[tuple[str, int, float]]:
    """(path, size, mtime) of every image file under root (os.scandir: one stat each)."""
    stack = [str(root)]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
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
        self._reload()

    def close(self) -> None:
        self._thumbs = None
        self._db.close()

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

    # Contents (arrays indexed by slot)

    @property
    def version(self) -> int:
        row = self._db.execute("SELECT value FROM meta WHERE key = 'version'").fetchone()
        return int(row[0]) if row else 0

    def __len__(self) -> int:
        """Number of usable tiles."""
        return int(np.count_nonzero(self.status == OK))

    @property
    def ids(self) -> np.ndarray:
        """Slots of usable tiles, ascending."""
        return np.flatnonzero(self.status == OK)

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
        n = rows[-1][0] + 1 if rows else 0
        self._paths = [""] * n
        self.width = np.zeros(n, np.int64)  # upright full-size image width
        self.height = np.zeros(n, np.int64)
        self.thumb_size = np.zeros((n, 2), np.int64)  # (w, h) of each thumbnail
        self.status = np.full(n, MISSING, np.int8)
        for slot, path, w, h, tw, th, status in rows:
            self._paths[slot] = path
            self.width[slot], self.height[slot] = w, h
            self.thumb_size[slot] = (tw, th)
            self.status[slot] = status
        self._open_thumbs(n)

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
        found: dict[str, tuple[int, float]] = {}
        for root in self.roots:
            for path, size, mtime in walk_images(root):
                found[path] = (size, mtime)
                if len(found) % 5000 == 0:
                    progress("Scanning", len(found), 0)
                    if cancelled():
                        report.cancelled = True
                        return report

        todo: list[tuple[str, int]] = []  # (path, slot)
        next_slot = (max(s for s, *_ in known.values()) + 1) if known else 0
        for path, (size, mtime) in found.items():
            old = known.get(path)
            if old is None:
                todo.append((path, next_slot))
                next_slot += 1
                report.added += 1
            elif old[1:3] != (size, mtime) or old[3] == MISSING:
                todo.append((path, old[0]))
                report.updated += 1
            else:
                report.unchanged += 1
        gone = [slot for path, (slot, *_, status) in known.items()
                if path not in found and status != MISSING]  # fmt: skip
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
            if todo or gone:
                with self._db:
                    self._db.execute(
                        "INSERT OR REPLACE INTO meta VALUES ('version', ?)",
                        (str(self.version + 1),),
                    )
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

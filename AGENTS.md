# Agent notes

## Environment
- Python 3.11–3.13 only (moderngl, numba, faiss-cpu lack 3.14 wheels). The venv is `.venv` (Python 3.11).
- PowerShell: run tools from the venv directly, e.g. `.venv\Scripts\python.exe -m pytest -q`.

## Verify
- `.venv\Scripts\python.exe -m pytest -q` (headless; Qt runs offscreen; about 15 s)
- `.venv\Scripts\ruff.exe check .` and `.venv\Scripts\ruff.exe format --check .`
- Matching at scale: `.venv\Scripts\python.exe scripts\bench_matching.py`

## Conventions
- `skitter.core` must not import Qt. Long core operations take `progress`/`cancelled` callbacks; the UI runs them via `Session` + `ui/jobs.Job`.
- Settings are declared with `Param`s on a `Configurable` (`core/slicing/params.py`); `ParamForm` builds their UI.
- numba kernels use `cache=True, nogil=True` (so background jobs don't block the UI thread).
- Tests must never touch the real tile library: `tests/conftest.py` points `LOCALAPPDATA` at a temp dir; open libraries in `tmp_path`.
- UI preferences go through `ui/preferences.settings()` (INI); `tests/conftest.py` redirects its path.
- `TileLibrary` defines `__len__`, so an empty library is falsy: compare with `is None`.

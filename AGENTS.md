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
- Slicing runs in the background: operations take `apply(regions, ctx, progress)` / `subdivide(region, ctx, progress)` and should call `progress(fraction)` at natural points in slow work (it also cancels). Tests slice inline by default (`tests/conftest.py`); use the `background_slicing` fixture to test the worker thread.
- New slicers: lattice patterns go through `slicing/frame.PinnedFrame` (anchor + rotation for free); shared settings (anchor, rotation, orientation, seed) come from `slicing/common.py`; size tiles with `ctx.tile_dims(scale)`.
- numba kernels use `cache=True, nogil=True` (so background jobs don't block the UI thread).
- Tests must never touch the real tile library: `tests/conftest.py` points `LOCALAPPDATA` at a temp dir; open libraries in `tmp_path`.
- UI preferences go through `ui/preferences.settings()` (INI); `tests/conftest.py` redirects its path.
- Side panels (`steps/base.side_panel`) never scroll sideways, so their content must fit the panel's width (320 px of content; `resizable` panels start there and can be dragged wider, see `view_and_panel`). Let long or changing text wrap (`setWordWrap(True)`), especially beside a button on one row; clip long paths (`fixed_width`). `test_every_side_panel_fits_its_width` checks every tab with the Windows UI font (offscreen Qt measures text far wider): when a panel gains a state that shows new widgets, add that state to the test.
- Project state is saved in `.skitter` files (`core/project_file.py`): new project settings must be added to `document()` and `_project()` there (Configurables round-trip through `values()` / `from_values()`). Views that hold a project settings object rebind on `Session.project_replaced`.
- Choreographies (`core/animation/`) declare their `phase` (build, show or clear; ids unique across phases) and keep its boundaries (`phases.py`): a build ends exactly at `TileFrame.final`, a show starts and ends at it, a clear starts at it (and should end with nothing left, for loops). `test_every_choreography_keeps_its_phase_boundaries` checks every registered one; new show effects go in `show.py`.
- `TileLibrary` defines `__len__`, so an empty library is falsy: compare with `is None`.

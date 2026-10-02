# Skitter

A photo mosaic generator with an animated graphical interface.

## Stack

- **numpy** and **Pillow** handle image processing and mosaic algorithms (`skitter.core`)
- **PySide6 (Qt 6)** provides the application shell: windows, menus, panels, dialogs (`skitter.ui`)
- **moderngl** drives the canvas. It needs OpenGL 3.3 or newer. All tiles in a layer draw in a single instanced call (`skitter.ui.render`)
- **faiss-cpu** (nearest-neighbor search), **numba** (inner loops) and **scipy** (blurs) power tile matching (`skitter.core.tiles`, `skitter.core.matching`)

`skitter.core` must not import Qt. That keeps the algorithms testable without a display and lets them run from scripts.

## Rendering model

- A `SpriteLayer` holds:
  - a stack of same-sized textures (a GL texture array)
  - a structured numpy array with one row per sprite: `pos`, `size`, `rotation`, `alpha`, `layer`, `tint`
- Animations are functions `f(seconds) -> keep_running`, registered with `canvas.add_animation`.
  - Each frame they write directly into the instance arrays and call `layer.mark_dirty()`.
  - The canvas uploads changed arrays and redraws once per vsync while any animation is running, and stays idle otherwise.
  - Animations advance after each frame is presented (`frameSwapped`), never inside `paintGL()`. Their side effects (signals that show, hide or resize widgets) would corrupt Qt's paint pass and crash. Keep `paintGL()` render-only.
- Navigation: set `canvas.bounds` to the content's world rect.
  - `zoom_to_fit()` enters fit mode, which refits on every resize. Any other zoom or pan leaves fit mode.
  - `clamp_to_bounds` stops the view from scrolling past the content's edges.
- Use the vectorized curves in `skitter.core.easing` to animate thousands of tiles in one numpy expression.
- World coordinates follow image conventions (y points down). Sprite `pos` is the sprite's center.

## Source step

- Edits are non-destructive: `Project` keeps `source_original` plus a list of edit values (`core.edits`: flip, rotate, crop). `source_image` is recomputed from them.
  - `Session` provides undo, redo and revert (revert keeps the edits available to redo).
- `ImageViewer` (`ui/widgets/image_viewer.py`) is the reusable viewer. It has:
  - fit by default, scrollbars only when the image overflows
  - a bottom bar with the cursor's pixel and RGB value, zoom out, an editable zoom %, zoom in, Fit (Ctrl+0) and 1:1 (Ctrl+1)
  - wheel zoom about the cursor, drag to pan, double-click to toggle fit/100%
  - sharp pixels at 400% and above
  - animated transitions for flip, rotate and crop
- `CropOverlay` is a transparent child widget of the canvas that draws the crop box. Left drags edit the box; wheel and middle-drag fall through to the canvas.
  - The box math lives in `core.geometry` (Qt-free, tested).

## Workflow tabs

- Each step of mosaic generation is a tab: a `StepPage` subclass listed in order in `skitter/ui/steps/__init__.py`. Currently: Source, Slicing, Tiles, Matching.
- All steps share one `Session`, which wraps the `core.project.Project` data and emits signals when it changes.
  - Steps change the project only through `Session` methods, so other steps are notified.
- The footer has **Back** and **Next: \<step\>** (Ctrl+Enter). Next calls the current step's `advance()`, which commits its work, then opens the next tab.
- A tab is enabled only while every earlier step reports `is_complete()`. A step's `can_advance()` enables Next.
  - Steps emit `state_changed` whenever either may have changed.
  - Steps get `on_enter()` / `on_leave()` calls when their tab is switched to or away from.
- **Source → Slicing hand-off:** Next on Source freezes a read-only copy of the edited image as `project.source_final`, and `Session` emits `source_committed`. Later steps use only `source_final`.
  - Editing the source afterwards locks Slicing until Next is pressed again. Undoing back to the committed state unlocks it.
  - Re-committing an unchanged image emits nothing, so downstream work survives.

To add a step: subclass `StepPage`, set `title`, implement `is_complete()` (and `advance()` if it has work to commit), and append the class to `STEPS`.

## Slicing

Slicing divides the final image into regions for tile matching. The result is always a `RegionSet`: rectangles with a center, size, rotation (radians, clockwise on screen) and stacking order, stored as parallel numpy arrays (`skitter.core.slicing`, Qt-free).

### Mosaic layout

- The user picks the **base tile** (width in mosaic px and an aspect ratio) and the number of **columns** (`MosaicLayout`). Everything else is derived:
  - **Mosaic canvas** = the source image scaled uniformly to columns x tile width. Height follows the source's aspect ratio, so the canvas usually holds a fractional number of tile rows.
  - **Slicing works in mosaic pixels.** The source is only used to sample colors (`SliceContext.patch` handles the scaling).
- Slicers try to **cover 100% of the canvas**, and the resulting tiles are what they are. Edge tiles may overhang, and the grid centers its overhang by default. Cropping or squaring off the assembled mosaic is a later post-processing step, not slicing's job.
- **Size settings** use `TileSizeParam`, measured in base tiles (1.0 = one tile), so plans follow the user's tile size. Resizing the mosaic means changing columns: more tiles of the same size.
- The Slicing tab shows, live:
  - mosaic size, rows (exact and whole), source px per tile (warns below 2), memory
  - region count, **density** (regions relative to a plain grid of base tiles; a grid is about 1.0x)
  - **coverage** (estimated by stratified sampling; warns below 100%)
  - smallest / median / largest region
- In the preview, parts of regions past the image edge show as neutral gray.

### Plans and operations

- A **slicing operation** transforms regions: `apply(regions, ctx) -> regions`.
- A **plan** (`project.slicing_plan`) is an ordered list of stages, each an operation plus an enabled flag. It starts from one region covering the whole canvas.
  - Splitting, adjusting, filtering and hand-drawn regions all fit this one interface, and operations nest: a grid inside jittered regions gives rotated cells.
- `Session` re-evaluates the plan when it, the layout, or the final image changes, reusing cached results for unchanged leading stages.
- The Slicing tab lists the stages (add from a menu grouped by category, reorder, enable/disable, remove), generates a settings form from the selected operation's parameters, and draws the regions on the GPU.
- Built-in operations:
  - **Grid**: base-tile cells with centered overhang, or a fixed count that fits exactly.
  - **Brick Bond**: courses of base-tile bricks, each shifted along its length (running, third, quarter, custom or random). Courses run horizontally (rows shift) or vertically (columns shift).
  - **Brick Pattern**: repeating patterns that mix brick directions: herringbone (any tile shape, any rotation; 45° gives diagonal herringbone) and basketweave. Bricks lying the other way are base tiles turned 90°, so regions keep the tile shape.
  - **Quadtree**: splits where the image has detail, down to a minimum in tiles. Use after a Grid.
  - **Photo Pile**: overlapping rotated photos in the tile shape. Spread 1.0 guarantees coverage.
  - **Jitter**, **Gap**, **Stacking Order**.

### Overlap and stacking

- Regions may overlap. Each region has a `z` value: higher z lies on top and hides what it covers. Equal z stacks in array order (later on top), and only relative order matters.
  - Use `stacking_order()` / `stacking_rank()` to read the stack, `restacked(key)` to reorder it, and `hit_test()` to find the topmost region at a point.
- `Subdivider` keeps each parent's place in the stack: a lower photo's pieces stay below everything above it. Pieces stack among themselves by the z values `subdivide` returns.
- `replace()` keeps z, so moving and resizing operations need do nothing. Operations that create or change overlap set z explicitly.
- Preview modes (Display group):
  - **Stacked:** regions draw bottom to top, each filled with the image pixels at its position, so upper regions hide the outlines of lower ones. Optional drop shadows, and dimming of uncovered areas.
  - **Outlines:** every region's full extent is visible.
  - **Hidden.**
- Hovering highlights the topmost region under the cursor (its full extent) and shows its size, rotation and stack layer.

### Writing a new operation

```python
from skitter.core.slicing import IntParam, RegionSet, Subdivider, register_operation


@register_operation
class Stripes(Subdivider):
    id = "stripes"  # stable id used in saved plans
    name = "Stripes"
    description = "Split each region into vertical stripes."
    count = IntParam(8, "Stripes", min=1, max=500)

    def subdivide(self, region, ctx):
        # Work in the region's local frame: [0, width] x [0, height], no rotation.
        return RegionSet.grid(region.width, region.height, self.count, 1)
```

- Subclass `Subdivider` to split each region independently in its local frame (the framework handles position and rotation). Subclass `SlicingOperation` and implement `apply` for anything else.
- Parameters (`IntParam`, `FloatParam`, `BoolParam`, `ChoiceParam`) give validation, the generated settings form, and saving. `when=` greys a parameter out depending on others. A new `Param` type needs an editor factory: `@register_editor` in `ui/widgets/param_form.py`.
- Sizes: use `TileSizeParam` for lengths and convert with `ctx.tile_size` (base tile in mosaic px). `ctx.width/height` is the canvas.
- `ctx.image` / `ctx.luminance` give the read-only final image in source pixels. `ctx.patch(region)` samples the source inside a (possibly rotated) canvas region on a grid aligned with it.
- Operations must be deterministic (take a seed parameter for randomness) and must not modify inputs. Set `z` when the regions you create overlap (see above). RegionSets are immutable: build new ones with `replace`, `from_arrays`, `from_rects`, `grid`, `concat`.
- **Brick patterns** (`slicing/patterns.py`): a pattern is a repeating unit, a few bricks plus two period vectors, built by a function registered with `@register_pattern(id, name)`; `tile_pattern` fills a region with copies. Describe bricks in landscape terms with `PatternBuilder.add(cx, cy, horizontal)`; it makes each one a region in the base tile's shape (turned for portrait tiles). Pass `length=`/`thickness=` for bricks of other shapes. A pattern that needs settings lists them in `options`, and they become `PatternSlicer` parameters of the same name. Built-in patterns live in `operations/pattern.py`.
- Optionally override `summary()` for the stage list. Make sure the module is imported (built-ins are imported by `skitter/core/slicing/operations/__init__.py`).

## Tiles

- The **tile library** (`core/tiles/library.py`) is an on-disk cache in its own folder (default `%LOCALAPPDATA%\Skitter\library`):
  - `library.sqlite`: the folders to scan, plus one row per image (path, size, mtime, upright pixel size, status).
  - `thumbs.u8`: a memory-mapped array of 32 px analysis thumbnails, about 3 KB per tile (1.5 GB at 500,000 tiles), paged in as needed.
- **Update** is incremental. It reads only new or changed files, records unreadable ones (not retried until they change) and marks vanished ones missing. `version` changes with each update, so derived data is cached by it.
- Ingest (`core/tiles/ingest.py`) decodes JPEGs at reduced scale (`Image.draft`) in a process pool. Progress and cancel go through callbacks.
- Long work runs off the UI thread as a `ui/jobs.Job`, one at a time, started through `Session` (`update_library`, `start_matching`). Its signals arrive on the UI thread.

## Matching

Matching (`core/matching`, Qt-free) gives every visible region a tile crop:

1. **Targets** (`targets.py`): each region is sampled in its own rotated frame and described in OKLab (`core/color.py`):
   - the mean color;
   - 2 x 2 and 4 x 4 grids of how each cell differs from that mean;
   - the lightness spread per 2 x 2 cell (`core/tiles/descriptors.py`).

   A raster of the region stack (`raster.py`) masks the cells hidden under higher regions. Cells too small to carry detail are masked too.
2. **Candidates** (`core/tiles/crops.py`, `index.py`): regions are grouped into shape (aspect) classes. For each class, each tile gets crop windows chosen by its own aspect: one centered crop if it nearly fits, otherwise up to `crops` crops along its long axis, optionally mirrored. Tiles that are far too long are skipped.
3. **Search**:
   - faiss uses an exact index up to 50,000 candidates and IVF-SQ8 beyond that.
   - The top `candidates` are reranked exactly. The cost is `(1 - t)^2 |mean difference|^2 + |structure difference|^2`, with per-dimension weights, where `t` is the tint strength. A crop penalty favors tiles that keep more of their photo.
   - Before the full search, a sample of regions is also searched exactly. If the approximate results cost noticeably more (the regret), search effort is raised.
4. **Assignment** (`assign.py`, numba):
   - A greedy pass gives each region its cheapest allowed candidate, honoring `max_uses` per image and the minimum spacing between repeats.
   - A refinement pass then moves and swaps tiles to lower the total cost.
   - Optional error diffusion passes average-color error on to later regions.
5. **Quality** (`quality.py`):
   - A proxy render paints each region's 4 x 4 cells in stacking order and compares the result with the image after blurring over ½, 1 and 2 tiles, giving ΔE scores, lightness SSIM and a per-region error.
   - Adaptive passes search harder for the worst regions and keep the result only if the score improves.

- Tinting moves only a tile's average color toward its region's (presets None, Subtle and Custom); tiles are never blended with the source.
- The Matching tab previews the result with thumbnails packed into texture atlases (`ui/render/atlas.py`; each sprite's `uv` selects its cell and its `offset` applies the tint). It also has an error heat map and the statistics above.
- `scripts/bench_matching.py` benchmarks index build, search accuracy and speed, and assignment at library scale.

## Layout

```
src/skitter/
  app.py            entry point (--demo N runs the flying-tiles demo)
  core/             numpy-only image processing and mosaic logic
    imaging.py      load/save/resize/crop helpers
    easing.py       vectorized easing curves
    edits.py        non-destructive edits: flip, rotate, crop
    geometry.py     rectangle math for interactive tools (crop box)
    color.py        sRGB <-> OKLab (vectorized and for numba kernels)
    project.py      Project dataclass (state across all steps)
    tiles/          tile library
      library.py    on-disk cache: sqlite metadata + memory-mapped thumbnails
      ingest.py     parallel, reduced-scale thumbnail decoding
      crops.py      aspect classes and aspect-guided crop windows
      descriptors.py  OKLab grid-pyramid descriptors of tile crops
    matching/       region-to-tile matching
      targets.py    region descriptors with visibility masks
      raster.py     rasterized region stacks
      index.py      candidate sets, faiss index, exact rerank and search
      assign.py     reuse-constrained assignment and refinement
      quality.py    proxy render and distance-blurred scores
      matcher.py    the pipeline and its caches
      settings.py   MatchSettings (Params)
    slicing/        slicing framework
      layout.py     MosaicLayout: base tile, columns, canvas size
      regions.py    Region / RegionSet (rotated rectangles)
      params.py     declarative operation parameters
      base.py       SliceContext, SlicingOperation, Subdivider, registry
      plan.py       SlicingPlan, stages, cached evaluation
      analysis.py   coverage, density and size summary
      patterns.py   brick pattern framework (repeating units, tiler)
      operations/   built-ins: grid, bond, pattern, quadtree, pile, jitter, gap, stacking
  ui/
    main_window.py  tabbed window, Back/Next footer, step gating, menus
    session.py      observable Project wrapper shared by steps
    steps/
      base.py       StepPage base class
      source.py     step 1: source image selection and editing
      slicing.py    step 2: slicing plan editor and region preview
      tiles.py      step 3: tile library folders, update, statistics
      matching.py   step 4: matching settings, run, mosaic preview, heat map
    jobs.py         background jobs with progress and cancel
    widgets/
      image_viewer.py  canvas + scrollbars + zoom bar + edit transitions
      crop_overlay.py  interactive crop box over a canvas
      region_overlay.py  stacked GPU preview of a RegionSet (fill, outlines, shadows, hover)
      param_form.py    settings form generated from Params
    canvas.py       GPU canvas widget: layers, animations, navigation
    icons.py        vector toolbar icons drawn with QPainter
    demo.py         flying-tiles stress demo window
    render/
      camera.py     2D pan/zoom math
      sprites.py    SpriteLayer data and instanced renderer
      atlas.py      thumbnail texture atlases
  resources/
    shaders/        GLSL sources
tests/              pytest suite (headless; Qt widgets run offscreen)
```

## Setup

Python 3.11 to 3.13 (moderngl, numba and faiss-cpu have no wheels for 3.14 yet).

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Run

```powershell
python -m skitter      # or just: skitter
python -m skitter --demo 50000
pytest
ruff check .
python scripts/bench_matching.py --candidates 1000000 --regions 50000
```

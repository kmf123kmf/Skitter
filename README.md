# Skitter

A photo mosaic generator with an animated graphical interface.

## Stack

- **numpy** and **Pillow** handle image processing and mosaic algorithms (`skitter.core`). **pillow-heif** adds HEIC / HEIF photos (iPhone); readable types are listed in `core/imaging.py` (`IMAGE_EXTENSIONS`)
- **PySide6 (Qt 6)** provides the application shell: windows, menus, panels, dialogs (`skitter.ui`)
- **moderngl** drives the canvas. It needs OpenGL 3.3 or newer. All tiles in a layer draw in a single instanced call (`skitter.ui.render`)
- **PyAV** (FFmpeg) encodes animation videos (`core/animation/encode.py`). Note: its bundled x264/x265, like pillow-heif's x265, are GPL; distributing Skitter with them means GPLv3
- **faiss-cpu** (nearest-neighbor search), **numba** (inner loops) and **scipy** (blurs) power tile matching (`skitter.core.tiles`, `skitter.core.matching`)

`skitter.core` must not import Qt. That keeps the algorithms testable without a display and lets them run from scripts.

## Rendering model

- A `SpriteLayer` holds:
  - a stack of same-sized textures (a GL texture array)
  - a structured numpy array with one row per sprite: `pos`, `size`, `rotation`, `alpha`, `layer`, `tint`, `uv` (the texture rect shown, e.g. an atlas cell) and `offset` (an OKLab shift of every texel, matching's tint)
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
- **Transparency is a mask.** Every source loads as RGBA (`load_image`; opaque when the file has no alpha), and edits move the alpha with the picture, so a future mask drawn in the app (say an ellipse, for an oval mosaic) is just an edit that changes alpha. Alpha is a solid mask: pixels at least half opaque are visible (`imaging.visible_mask`); the rest lie outside the picture, and the rest of the workflow treats the mask's edge exactly like the image's border:
  - `SliceContext.image` gives hidden pixels the color of the nearest visible one (`fill_hidden`, scipy's distance transform: about 1.6 s for a 24 MP image, only when the image has hidden pixels), as samples past the border read the nearest edge pixel. Slicing operations, matching targets and quality scoring all read that image, so edge tiles match only what shows.
  - After the plan, regions that touch no visible pixel are dropped (`slicing/mask.py`: a summed-area table of the mask, built once per context, settles most from each region's box, exactly for upright ones; rotated ones on the edge are sampled at most 0.7 px apart, so even a 1 px line isn't missed; 8 ms for a 15,000-tile grid on a 24 MP oval, 165 ms for a 40,000-photo pile). An edge on a pixel boundary doesn't reach the pixel beyond it. Holes smaller than a tile's reach are covered, like anything else a touching tile overhangs, and a stray visible pixel gets a tile of its own; the rest stay whole, overhanging the mask's edge as edge tiles overhang the border. Coverage and density count the visible part only.
  - Region overlays draw parts over hidden pixels the neutral gray of parts past the border. A transparent PNG export leaves the hidden area empty.
  - The Source tab shows how much of the image is transparent; an entirely transparent image can't go on.
- `ImageViewer` (`ui/widgets/image_viewer.py`) is the reusable viewer. It has:
  - fit by default, scrollbars only when the image overflows
  - a bottom bar with the cursor's pixel and RGB value, zoom out, an editable zoom %, zoom in, Fit (Ctrl+0) and 1:1 (Ctrl+1)
  - wheel zoom about the cursor, drag to pan, double-click to toggle fit/100% (on the Matching tab, double-click edits a tile instead)
  - sharp pixels at 400% and above
  - animated transitions for flip, rotate and crop
- `CropOverlay` is a transparent child widget of the canvas that draws the crop box. Left drags edit the box; wheel and middle-drag fall through to the canvas.
  - The box math lives in `core.geometry` (Qt-free, tested).

## Workflow tabs

- Each step of mosaic generation is a tab: a `StepPage` subclass listed in order in `skitter/ui/steps/__init__.py`. Currently: Source, Slicing, Tiles, Matching, Animate.
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

### Project files

**File → New / Open / Open Recent / Save / Save As** (Ctrl+N, Ctrl+O, Ctrl+S, Ctrl+Shift+S; Open Source Image moved to Ctrl+Shift+O). A project is one `.skitter` file, a zip (`core/project_file.py`):

- `project.json`: a format version and every step's settings, the source's name and edits, whether it was committed, and the matched mosaic's non-array data. A separate `view` part records how the app showed the project: the tab it was saved on, by the step's stable `id` (not its title or position), so renamed, reordered or new tabs don't break old files; an unknown id falls back to the furthest tab the project reaches. View state isn't an unsaved change.
- `source.png`: a lossless copy of the source as loaded (RGB unless it has transparency), so a project doesn't depend on the original file.
- `regions.npz`: the regions, if the source was committed. They are kept, not sliced again, so the mosaic still fits exactly.
- `mosaic.npz`: the matched mosaic, reduced to what matters (`core/matching/saved.py`): per region the photo's index into a list of file paths, its crop window, mirroring and whether it was picked by hand, plus tint targets and costs. Candidate lists for picking by hand are not kept (their indices only mean something to the run that made them).

Opening rebuilds the result: photos are found by path in the current library (regions whose photo is gone are left empty, counted and reported), their crops described again from the thumbnails, quality evaluated anew. The mosaic shows, exports and animates at once; Edit Tiles asks to run matching again first, which keeps the picks (as pins) and brings the candidates back. Paths aren't a new dependency: the preview's full-size tiles and the export read the original photos anyway.

Loading is forgiving: settings or operations a version doesn't know keep their defaults or are left out (`Configurable.from_values`, `SlicingPlan.from_dict(problems=...)`), each listed in a note after opening. A newer format is refused. Saving writes a `.part` file and renames it, so a failed save never damages the existing file.

In the app, `Session.new_project` / `open_project` replace the whole `Project`. `project_replaced` is emitted first, and views holding its settings objects (settings forms, the export dialogs) rebind; then the usual signals replay what it holds (source, commit, layout, regions, mosaic), so each tab rebuilds as if the user had worked through it. `Session.modified` compares `project_file.document` and the identities of the image, regions and mosaic with their state at the last save or open. The title shows `*` while modified, and New, Open and closing offer to save first.

### Preferences

Display choices that belong to the user rather than the project (the Slicing tab's line color, line opacity and Show structure, the Animate tab's Show export frame, the folders the export dialogs last used, and recent projects) are stored with `QSettings` in an INI file, through `ui/preferences.settings()`. `tests/conftest.py` redirects that file to a temp folder, so tests never read or write the real one.

## Slicing

Slicing divides the final image into regions for tile matching. The result is always a `RegionSet`: rectangles with a center, size, rotation (radians, clockwise on screen) and stacking order, stored as parallel numpy arrays (`skitter.core.slicing`, Qt-free).

### Mosaic layout

- The user picks the **tile aspect** and the number of **columns** (`MosaicLayout`). Nothing has a size in pixels until export (see below), so the same mosaic can be exported at any resolution. Everything else is derived:
  - **Mosaic canvas** = the source image scaled uniformly to `columns` base tiles across. Height follows the source's aspect ratio, so the canvas usually holds a fractional number of tile rows.
  - **Slicing and matching work in mosaic units**: a base tile is `TILE_UNIT` (100) units wide. The unit only sets the internal scale; results don't depend on it (`SliceContext` takes another `tile_width`, which tests use). The source is only used to sample colors (`SliceContext.patch` handles the scaling).
- Slicers try to **cover 100% of the canvas**, and the resulting tiles are what they are. Edge tiles may overhang, and the grid centers its overhang by default. Trimming the overhang is an export option, not slicing's job.
- **Size settings** use `TileSizeParam`, measured in base tiles (1.0 = one tile), so plans follow the user's tile size. Resizing the mosaic means changing columns: more tiles of the same size.
- The Slicing tab shows, live:
  - rows (exact and whole), source px per tile (warns below 2)
  - region count, **density** (regions relative to a plain grid of base tiles; a grid is about 1.0x)
  - **coverage** (estimated by stratified sampling; warns below 100%)
  - smallest / median / largest region, in base tiles
- In the preview, parts of regions past the image edge show as neutral gray.

### Plans and operations

- A **slicing operation** transforms regions: `apply(regions, ctx, progress) -> regions`. `progress` (a `Progress`) is optional to use: call it with the share done at natural points if a stage can take a while; it reports to the UI and raises `SlicingCancelled` once the run is replaced. `Subdivider` calls it between regions and hands each `subdivide` its slice (`progress.part`).
- A **plan** (`project.slicing_plan`) is an ordered list of stages, each an operation plus an enabled flag. It starts from one region covering the whole canvas.
  - Splitting, adjusting, filtering and hand-drawn regions all fit this one interface, and operations nest: a grid inside jittered regions gives rotated cells.
- `Session` re-evaluates the plan when it, the layout, or the final image changes, reusing cached results for unchanged leading stages.
  - **In the background:** each run is a `Job` on a snapshot (`plan.copy()`, the context, the cached stages), so editing while it runs is safe. A new edit cancels the run under way (it stops at its next progress call) and starts another; only the latest run's result is installed (`_slicing_run` numbers them). Plan edits keep the current regions shown until the new ones are in; a new image or layout clears them at once (they don't fit). `slicing_started` / `slicing_progress` / `slicing_changed` tell the UI; while `slicing_running`, the Slicing tab shows progress and Next and matching wait. The context's heavy derived data (the transparency fill, luminance) is computed on first use, in the run. A 65,000-tile Contour Rows run (3.5 s) pauses the UI at most about 0.2 s, when the regions are installed, against 3.7 s inline.
  - Tests slice inline (`Session.slice_in_background`, off in `tests/conftest.py`); `tests/test_background_slicing.py` covers the worker thread through the `background_slicing` fixture.
- The Slicing tab lists the stages (add from a menu grouped by category, reorder, enable/disable, remove), generates a settings form from the selected operation's parameters, and draws the regions on the GPU.
- **Shared building blocks** for new slicers:
  - `slicing/frame.py`: a `PinnedFrame` for lattice slicers (Grid, Brick Bond, Brick Pattern): pinned at the region's center or top-left corner and turned about it. `bounds()` says what to fill in the pattern's own unturned frame; `place()` turns the tiles laid there into the region, faces them up and keeps those overlapping it (edge tiles whole, overhanging); `span()` lays tiles along one axis, centered on the pin or from it. A new lattice (hexagons, scales) only says where its tiles go, and gets anchors and rotation for free.
  - `slicing/common.py`: setting factories (`anchor_param`, `rotation_param`, `orientation_param`, `seed_param`) so shared settings read and range alike; `SliceContext.tile_dims(scale)` sizes a tile in base tiles.
  - `regions.overlapping` / `regions.upright` (rectangle overlap; the rotation nearest upright with the same outline), `ctx.patch(..., antialias=True)`, and the `FollowsStructure` protocol (`structure.py`) for slicers that can show what they follow.
- Built-in operations:
  - **Grid**: cells of the Mosaic tile shape (scaled by Cell size), with centered overhang, turned by **Rotation** about the anchor. Its only sizes are in base tiles, so the Mosaic box alone sets tile size and shape.
  - **Split**: divides each region into Across x Down equal pieces (counts are per region, e.g. per jittered photo). Keep tile shape picks Down so pieces match the tile aspect.
  - **Brick Bond**: courses of base-tile bricks, each shifted along its length (running, third, quarter, custom or random). Courses run horizontally (rows shift) or vertically (columns shift). **Rotation** turns the whole pattern about its anchor (as Brick Pattern does): courses are laid, unturned, over the region as the turned pattern sees it, numbered from the anchor so the bond carries on, then turned, keeping the bricks that overlap the region (whole, overhanging). At 0° the result is as before. Brick Pattern, Brick Bond and Contour Rows share `regions.overlapping` and `regions.upright` (rotations into the half or quarter turn nearest upright).
  - **Brick Pattern**: repeating patterns that mix brick directions: herringbone (any tile shape, any rotation; 45° gives diagonal herringbone) and basketweave. Bricks lying the other way are base tiles turned 90°, so regions keep the tile shape.
  - **Quadtree**: splits where the image has detail, down to ½, ¼, ⅛… of each starting cell. Use after a Grid.
  - **Photo Pile**: overlapping rotated photos in the tile shape. Spread 1.0 guarantees coverage.
  - **Contour Rows**: tiles in rows that follow the image's edges and outlines and ripple outward from them, like a Roman mosaic (opus vermiculatum). Every tile keeps the base tile shape (matching indexes one aspect class), its long side along its row, turned to whichever way with the same footprint is nearest upright (`upright`: a row has no direction, so tiles traced "backwards" would otherwise show their photos upside down; rectangles end within ±90°, squares within ±45°). Three layers, Qt-free:
    - `slicing/structure.py` analyzes the region's patch at 8 samples per row height: thin edges (blurred by **Smoothness**, non-maximum suppressed, at least **Edge strength** of the 99th-percentile gradient, fragments shorter than **Shortest edge** dropped) plus the mask's edge are the guides; `rows` is each sample's distance to the nearest guide in row heights (no guides: straight rows from the top); **Background** (**Outline rows**, default 3; 0: rows ripple everywhere): past that many rows from any guide, and everywhere when there is none, rows run straight at **Background angle** (the `straight` row field), a plain background around halos of outline rows, as in Roman mosaics (opus tessellatum around opus vermiculatum). **Texture** (**Follow texture**): the image's grain is measured at fine scale (structure tensor, about half a sample of blur, averaged over a row, leaving out gradients within half a row of an outline so an outline's own strength doesn't make a band of texture beside it); where it is strong (at least Edge strength of the image's strong gradients) and clear, at least a row from any guide, patches of 4 tiles or more are texture, and tiles there follow the grain instead of rippling across it (stripes, bands, a shoreline). The patch is antialiased (`ctx.patch(..., antialias=True)`: blurred by 0.7 sample spacings before sampling, cached per blur), and strength is judged against the image before that blur, so detail finer than the samples neither aliases into a false grain (moire) nor counts. `flow` is a smooth direction everywhere: the grain in texture; elsewhere the rows' where they're clear and the grain where they aren't, averaged as doubled angles.
    - `slicing/rows.py` (numba) traces each row's centerline (rows = k + 1/2), nearest the guides first, stepping a tile length along the level set and back onto it, laying a tile on each chord; a row stops at **Overlap** with tiles already laid, a sharp turn, where rows meet, or on hidden samples. Outline rows keep out of texture and the background; the background gets straight rows the same way (`_level_rows` lays both; the field extends linearly past the patch, so rows overhanging its border stay true). Then flow rows, traced along the flow the same way, fill texture first and then whatever room outline rows left, each seeded from the uncovered samples nearest what is laid, its first tile nudged up to half a height across the flow to fit. Fillers turned along the flow cover what is left. Stacking: outline rows (nearer guides on top), background rows, flow rows, then fillers. Slivers under 5% of a tile are left as grout: covering them all would nearly double the count.
    - **Show structure** (Slicing tab, Display): draws what tiles follow, over the regions: edges found in the image yellow, the mask's edge cyan, texture magenta, and white strokes about a row apart for the flow (`ui/widgets/structure_overlay.py`, one RGBA sprite over the canvas). It shows only the selected stage, while it is enabled and follows structure (it helps tune the settings being edited); operations opt in by implementing `structure(region, ctx)` (the `FollowsStructure` protocol in `structure.py`), which `ContourSlicer.subdivide` uses too, so the picture is exactly what slicing sees (computed over the whole canvas: exact for a first stage). Edges count only on visible samples: the filled-in colors of hidden ones have seams that are no edges.
    - `operations/contour.py` is a `Subdivider`, so it fills each region it's given (rotated ones too, after Pile or Split). About 0.8 s for 17,000 tiles, 3 s for 65,000 (in the background, like all slicing); numba compiles once (about 3 s, then cached).
  - **Jitter**, **Stacking Order**.

### Overlap and stacking

- Regions may overlap. Each region has a `z` value: higher z lies on top and hides what it covers. Equal z stacks in array order (later on top), and only relative order matters.
  - Use `stacking_order()` / `stacking_rank()` to read the stack, `restacked(key)` to reorder it, and `hit_test()` to find the topmost region at a point.
- `Subdivider` keeps each parent's place in the stack: a lower photo's pieces stay below everything above it. Pieces stack among themselves by the z values `subdivide` returns.
- `replace()` keeps z, so moving and resizing operations need do nothing. Operations that create or change overlap set z explicitly.
- Preview modes (Display group):
  - **Stacked:** regions draw bottom to top, each filled with the image pixels at its position, so upper regions hide the outlines of lower ones. Optional dimming of uncovered areas.
  - **Outlines:** every region's full extent is visible.
  - **Hidden.**
- **Line color** (a preset list) and **Line opacity** (10 to 100%) restyle the outlines live (`RegionOverlay.set_line_color` / `set_line_alpha`, through the sprite shader's `u_line_alpha`). They apply to the outline and its dark inner edge only: the image fill stays opaque and the hover highlight keeps its color. Both are remembered between runs (see Preferences below).
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
- Parameters (`IntParam`, `FloatParam`, `TileSizeParam`, `RangeParam`, `BoolParam`, `ChoiceParam`, `ColorParam`) give validation, the generated settings form, and saving. `when=` greys a parameter out depending on others. A new `Param` type needs an editor factory: `@register_editor` in `ui/widgets/param_form.py`.
- Sizes: use `TileSizeParam` for lengths and convert with `ctx.tile_size` (base tile in mosaic units). `ctx.width/height` is the canvas. Never use absolute lengths: they would change meaning with the export size.
- `ctx.image` / `ctx.luminance` give the read-only final image in source pixels. `ctx.patch(region)` samples the source inside a (possibly rotated) canvas region on a grid aligned with it.
- Operations must be deterministic (take a seed parameter for randomness) and must not modify inputs. Set `z` when the regions you create overlap (see above). RegionSets are immutable: build new ones with `replace`, `from_arrays`, `from_rects`, `grid`, `concat`.
- **Brick patterns** (`slicing/patterns.py`): a pattern is a repeating unit, a few bricks plus two period vectors, built by a function registered with `@register_pattern(id, name)`; `tile_pattern` fills a region with copies. Describe bricks in landscape terms with `PatternBuilder.add(cx, cy, horizontal)`; it makes each one a region in the base tile's shape (turned for portrait tiles). Pass `length=`/`thickness=` for bricks of other shapes. A pattern that needs settings lists them in `options`, and they become `PatternSlicer` parameters of the same name. Built-in patterns live in `operations/pattern.py`.
- Optionally override `summary()` for the stage list. Make sure the module is imported (built-ins are imported by `skitter/core/slicing/operations/__init__.py`).

## Tiles

- The **tile library** (`core/tiles/library.py`) is an on-disk cache in its own folder (default `%LOCALAPPDATA%\Skitter\library`):
  - `library.sqlite`: the folders to scan, plus one row per image (path, size, mtime, upright pixel size, status).
  - `thumbs.u8`: a memory-mapped array of 32 px analysis thumbnails, about 3 KB per tile (1.5 GB at 500,000 tiles), paged in as needed.
- **Update** is incremental. It reads only new or changed files, records unreadable ones (not retried until they change) and marks vanished ones missing. `version` changes with each update, so derived data is cached by it.
- Ingest (`core/tiles/ingest.py`) decodes JPEGs at reduced scale and HEIC photos from their embedded preview (`Image.draft`; about 7 ms per 12 MP iPhone photo, or about 350 ms if a HEIC has no preview) in a process pool. Progress and cancel go through callbacks.
- Long work runs off the UI thread as a `ui/jobs.Job`, one at a time, started through `Session` (`update_library`, `start_matching`, `start_export`, `start_video_export`). Its signals arrive on the UI thread.
- **What the library can paint** (`core/tiles/palette.py`, shown on the Tiles tab). Everything comes from the thumbnails (about 2 s for 300,000 tiles, in the background, cached by library `version`):
  - **Color map**: a hue x lightness grid (grays in their own column) at a chosen colorfulness (Muted, Medium, Vivid). Each cell shows the tile whose average color is nearest, framed by the cell's color; cells no tile comes within ΔE 5 of are faded. Hover for how many tiles are near.
  - **Source in library colors** / **Source color error**: the source repainted with the nearest tile averages, or as a heat map of the ΔE to them. A quick estimate of where a mosaic will struggle; matching also weighs structure, and tinting closes small gaps.
  - **Statistics**: photo short sides (median, smallest tenth), shapes, and **Sharp up to**: the widest base tile (export pixels) 9 in 10 photos fill without enlarging, for the Mosaic tile shape.
  - **Sample of tiles**: a random sample, as before.

## Matching

Matching (`core/matching`, Qt-free) gives every visible region a tile crop:

1. **Targets** (`targets.py`): each region is sampled in its own rotated frame and described in OKLab (`core/color.py`):
   - the mean color;
   - 2 x 2 and 4 x 4 grids of how each cell differs from that mean;
   - the lightness spread per 2 x 2 cell (`core/tiles/descriptors.py`).

   A raster of the region stack (`raster.py`) masks the cells hidden under higher regions. Cells too small to carry detail are masked too. Each region's **detail level** (`detail_levels`) records which structure blocks (2 x 2 grid, 4 x 4 grid, texture) it uses at all.
2. **Candidates** (`core/tiles/crops.py`, `index.py`): regions are grouped into shape (aspect) classes. For each class, each tile gets crop windows chosen by its own aspect: one centered crop if it nearly fits, otherwise up to `crops` crops along its long axis, optionally mirrored. Tiles that are far too long are skipped.
3. **Search**:
   - faiss uses an exact index up to 50,000 candidates and IVF-SQ8 beyond that.
   - Building candidates and an index takes a while for large libraries (about 30 s for 300,000 photos, mostly training the IVF clusters). It reports progress as it goes, shown under the Matching status as a second, thinner bar, and Cancel stops it. The clusters are trained one k-means round at a time (`train_ivf`), which gives exactly faiss's own result; tests check that. Indexes are kept between runs, so the bar shows only when the library, crop settings or weights change, or the tint changes by more than 0.15.
   - The top `candidates` are reranked exactly. The cost is `(1 - t)^2 |mean difference|^2 + |structure difference|^2`, with per-dimension weights, where `t` is the tint strength. A crop penalty favors tiles that keep more of their photo.
   - Regions are searched per shape class **and detail level**, each level with an index weighted like its cost (only the dimensions it uses). One index for all would rank small regions on fine detail their cost ignores: on a 300,000-photo library with low-detail regions that missed the best tile for 96% of them at any effort, and fixing it improved the mosaic's score from ΔE 1.07 to 0.68 and made the search 40 times faster.
   - Before the full search, a sample of regions is also searched exactly. If the approximate results cost noticeably more (the regret), search effort is raised (x4), as long as that clearly lowers the regret, up to every cell of the index (no fixed cap). The Matching tab's Search line shows the worst extra ΔE over the exact best and the effort used.
   - Searches (the main one, widening, adaptive passes) run in batches sized to take about 3 s each, never under 2,048 regions, in a shuffled order. Each region's search is independent, so batching never changes results (tested). The second progress bar counts regions, previews update after each batch, and Cancel stops between batches. Smaller batches lose faiss throughput, more inside the app than in a script. At the highest search effort (about 3.5 ms per region), measured inside the app: 512 regions +47%, 1,024 +16%, 2,048 +9%, 4,096 none. So at that effort a batch takes about 7 s; at low effort batches grow up to 65,536 regions and cost nothing.
4. **Assignment** (`assign.py`, numba):
   - A greedy pass gives each region its cheapest allowed candidate, honoring `max_uses` per image and the minimum spacing between repeats.
   - **Widening:** regions left without an allowed candidate are searched again. In a large plain area every region wants the same few photos, so waiting regions are grouped by their favorite photo. Each group is searched once, from its average target, deep enough for its size (about twice the photos it needs, counting crops, up to 4,096 candidates), and photos already at their use limit are skipped. Each member takes an interleaved slice of the result by photo, so no two slices share a photo, reranked for its own target.
   - Regions still without an allowed candidate are forced onto their least-used candidate, preferring the one farthest from other uses of its photo. Rule breaks then spread over many photos instead of piling onto one.
   - Tested on blank-heavy images against the 300,000-photo library (white page, black/white halves, flat gray; 600 and 1,700 regions; 3 uses spaced 4 tiles apart, and 1 use): no rule breaks, where the old widening let one photo fill up to 1,343 regions.
   - A refinement pass then moves and swaps tiles to lower the total cost.
   - Optional error diffusion passes average-color error on to later regions.
5. **Quality** (`quality.py`):
   - A proxy render paints each region's 4 x 4 cells in stacking order and compares the result with the image after blurring over ½, 1 and 2 tiles, giving ΔE scores, lightness SSIM and a per-region error.
   - Adaptive passes search harder for the worst regions, the same way as widening. They keep the result only if the score improves and no more regions break the reuse rules; the score doesn't see repetition.

- Tinting moves only a tile's average color toward its region's (presets None, Subtle and Custom); tiles are never blended with the source.
- The finished mosaic is a **scene** (`core/scene.py`, `session.scene`): every placed tile in its final state, bottom to top (position, size, rotation, crop, mirroring, tint), plus attributes to order or group tiles by (distance from center, reading order, lightness, colors, match error, repeated photos). The Matching preview, export and animations all read it.
- The Matching tab previews the scene with the session's shared tile textures (`ui/render/tile_textures.py`, `session.textures`): thumbnails packed into texture atlases at once (`ui/render/atlas.py`; each sprite's `uv` selects its cell and its `offset` applies the tint). It also has an error heat map and the statistics above.
- While matching runs, the tab shows the run instead of the old mosaic. `Matcher.run(preview=...)` hands out a `MatchPreview` at each point with something to show: target colors, then each region's best tile before the reuse rules, filling in batch by batch as the search goes. Then the first assignment; regions the reuse rules left waiting keep showing their best tile. During widening, each waiting region switches to a tentative tile as its batch finishes: its cheapest candidate the reuse rules allow given the tiles already placed (`Assignment.first_allowed`), which is close to what the assignment then picks. Then the complete assignment, and every refining and adaptive pass. Previews are copies and only read the run's state, so results are identical with or without them. `ui/render/match_preview.py` builds their sprite layers in one background thread, newest first, at most every 0.2 s. Thumbnails go into a `GrowingAtlas` (`ui/render/atlas.py`), so each frame uploads only its new thumbnails; Display › Tile detail names the stage.
  - A background job then reads each used tile from its original file and cuts its crop at one texel per mosaic unit (`core/tiles/render.py`, process pool, JPEGs decoded at reduced scale). The crops are shelf-packed into atlas pages (`pack_images`) and replace the thumbnails. Regions showing the same crop at the same size share one image. Tiles smaller than the base tile (Quadtree cells, say) get more texels, as many as a base tile up to `MAX_BOOST` (4) times their size, so their photos stay sharp when the view zooms in. If the total would exceed `DETAIL_TEXELS` (about 400 MB of GPU memory), that boost gives way first, then every crop is scaled down by the same factor. Video export reads its own crops at the video's resolution, without the boost. Unreadable files fall back to their thumbnails.
- `scripts/bench_matching.py` benchmarks index build, search accuracy and speed, and assignment at library scale.

### Picking tiles by hand

The result remembers each region's 32 best candidates (`candidates.py`; indices into the candidate sets matching already holds, about 8 bytes each: 15 MB for 60,000 regions). On the Matching tab, double-clicking a tile (or **Edit Tiles**, E) enters edit mode (`ui/steps/edit_mode.py`, widgets in `tile_picker.py`):

- Hovering a tile outlines it; clicking selects it. The rest of the mosaic dims, and so do tiles lying on it, so only its visible part stands out.
- The side panel shows the part of the image the region covers and a grid of its candidates (`ui/widgets/candidate_grid.py`), best first, tinted as they would show: the tile shown now is framed, the matcher's choice has a star, and candidates that would break a reuse rule have a warning (the tooltip says which). Crops show from thumbnails at once and from their files moments later.
- Hovering a candidate, or moving to it with the arrow keys, previews it in place; a click or Enter picks it, and a double-click picks it and leaves edit mode. **Find More** searches every tile exactly for that region and lists the next 32 (`index.exact_top`). **Revert Tile** and **Revert All** restore the matcher's choice; picks can be undone and redone (Ctrl+Z, Ctrl+Y). Esc stops previewing, deselects, then leaves edit mode. Hand-picked tiles are marked in a corner.
- Picks may break the reuse rules (they count as rule breaks). Each pick makes a new result (`core/matching/edit.py`; the old one is untouched, so exports in progress are unaffected) and rescores only around the region (`quality.rescore`: exact, a few tens of milliseconds), so the score and heat map stay current. The scene keeps its cached geometry, and the textures put the new crop in free space of the existing pages, patched on the GPU in place (`TileTextures.set_scene`, `SpriteLayer.patch_texture`); the Animate tab and exports use the edited mosaic.
- **Match Tiles** with picks present asks whether to keep them. Kept picks are **pins**: the matcher places them first, counts them toward the reuse rules, never moves them, and fits everything else around them (even with other crop settings: missing crops are added to the candidate set). Picks whose photos left the library are dropped, and the status says how many.

## Export

**Mosaic → Export Image** (enabled only while the mosaic is valid: matched for the current regions, committed source and unchanged tiles) renders the mosaic at full detail in the background (`core/assembly.py`, `ui/export_dialog.py`).

- **Size** is chosen here, and only here: by base tile width, image width or image height in pixels. The window shows the resulting image and tile size and how many tiles would be shown larger than their photos have pixels for.
- **Framing**: the image frame (trims overhanging tiles) or whole tiles. **Background**: white, black, gray, or transparent (PNG). **Format**: PNG or JPEG (quality setting, no chroma subsampling).
- Each tile is its matched crop read from the original photo at output resolution (Lanczos), mirrored and tinted exactly as matching modeled it (an OKLab shift per pixel), then placed with bicubic sampling. Edges use 4 x 4 coverage samples, drawn front to back, so seams between tiles never show the background. Large images render in strips to bound memory.

## Animated construction

Groundwork for animating the tiles into the finished mosaic (`core/animation/`):

- A **choreography** is a configurable recipe (settings as `Param`s, registered with `@register_choreography`) that plans a **timeline** for a scene. `timeline.frame(t)` gives every tile's center, size, rotation, alpha, tint and draw order at time t: a pure function of time, so it can be played, paused, scrubbed or rendered frame by frame. `frame(duration)` must be exactly the finished mosaic (`TileFrame.final`), which a test checks for every registered choreography.
- `FlightTimeline` covers tiles travelling from a start state to their final state with per-tile delays and easing; tiles are hidden until they set off, opaque in flight, and draw above landed ones. `TossTimeline` throws tiles onto the table under gravity: steady sideways speed, an exact parabolic arc in height (gravity = 8 x arc / flight time²), constant spin that stops on impact, bounces that each keep a share of the speed, and a damped settling wobble, all closed form. A tile counts as at rest (and joins the bottom-first landing order and even pacing) only once settled. The built-in **Assemble** offers Toss (default), Drop or Glide, ordered by distance, reading order, lightness or random. **Drop** uses the same timeline started at the camera's height at rest (gravity = 2 x camera height / fall time²): tiles linger near the lens, looming large, then speed up and shrink onto their spots, bouncing and settling like tossed ones. **Slant** starts each tile beside its spot, outward from the middle, so it slides in from the picture's edge already large (0: a straight drop, except that tiles near the middle start just far enough out to come in from the edge; they loom over the view). **Spin**, **Flips**, **Settle wobble** and **Bounce** are ranges (`RangeParam`, edited as one compact "low – high" row): each tile draws its own amount, in a random direction (equal ends: all the same). **Flips** (`Flips`) turn tiles over in flight a whole number of times about one of their own edges, so each lands face up and unmirrored; seen from above a flipping tile foreshortens, shows the Photo backs color while back up, and darkens edge-on.
- **Camera and light** (`core/animation/look.py`, Animate tab): a camera looks straight down at the table from Camera height, so tiles in the air look larger (H / (H - h)) and further from the middle; a distant light casts each airborne tile's soft shadow on the table, offset away from the light, blurred and faded with height. Tiles draw in three groups: at rest, shadows, in the air (nearest the camera on top). Nothing at the lens is seen: tiles above 97% of the camera height (33x their size) are hidden, and below it fully opaque, never see-through. Choreographies keep tiles that high off the picture (`clear_of_axis`): a drop starts each tile at least far enough from the camera's axis that it comes into view beside the mosaic and slides in, and throws peak at most at 80% of the camera height. Preview and video use the same camera, and timelines get the look (`Choreography.timeline(scene, look)`) so a drop starts at whatever height the camera is.
- The built-in **Deal** (`core/animation/deal.py`, `DealTimeline`) deals tiles like cards from 1 to 8 decks on the table; each tile goes to its nearest deck. **Deck position** is an edge, a corner or the center, **Offset** how far out from the edge (0: on it; less: over the mosaic); several decks sit **Around the mosaic** (evenly by angle, starting at the deck position, where each way out meets the offset rectangle) or **Side by side** along that edge (not for corners). A tile slides off the top of its deck at constant deceleration, like a card on felt; pushed off equally hard, slide time grows with the square root of the distance and the farthest takes **Slide time**. Landings follow the bottom-first order and the same pacing as Assemble (Wind-down, Last gap); each tile leaves its deck its slide time before it lands, and the first landing is as early as lets every tile leave at 0 or later. Decks are real stacks: each card lies a little higher than the one under it (at most a tile size for the whole deck), the first to leave on top, so draw order by height keeps them right, and a leaving card keeps its height until a tile diagonal away from its deck so it never slips under the cards it lay on. **Orbit** (`DeckPath`) sends the decks circling the mosaic's center as they deal: **Radius** (in mosaic sizes, from the center), **Starting angle** (0° above, clockwise), **Orbit speed** (turns over the whole animation, so changing Duration keeps the look) and **Orbit direction**; several decks are spaced evenly round the circle, and each turns with its travel (bottom toward the middle). Tiles ride their deck until they leave, from wherever it is then, sliding straight. Each tile is dealt by the deck nearest it when it leaves, and when it leaves depends on how far it slides from there, so planning refines both over a few passes (128 ms for 15,000 tiles). With an orbit, Nearest first deals each tile as a deck passes it, tiles nearest the orbit on the first passes and farther ones on later ones (inward round after round; Farthest first: outward). **Neatness** squares the stack; **Face down** shows the photo backs in the deck and turns each card over as it leaves the deck, as a dealer does (lifting, in 0.3 s, at most 40% of its slide), so it slides face up. Orders: Nearest first (default), Farthest first, and Assemble's.
- **Overlapping tiles land bottom first** (`core/animation/landing.py`). `scene.overlaps` lists every overlapping pair (rotated rectangles, exact), and `landing_order` turns each tile's preferred place in the sequence into a landing order in which a tile never lands before the tiles it lies on (a prioritized topological sort), so nothing pops under its neighbors on landing. Random order instead draws a **uniformly random bottom-first order** (`random_landing_order`: independent uniform keys conditioned on lower < upper, sampled by Gibbs sweeps, about 8 ms for 3,400 tiles), so tiles land close together in space and time exactly as often as chance has it; random preferences through `landing_order` would land a held-back tile right beside the one it lies on (92% of landings had a neighbor within 0.15 s on a pile, against 50% by chance). Orders without a direction are listed in `assemble.RANDOM_ORDERS`, and a test holds every order to its kind: random ones to chance, directed ones to their direction. For directed orders, a tile freed by the landing of its last support then waits a few landings (`landing_gap`: n / 150, 3 to 50) while others can go; otherwise the sequence climbs one stack after another (on a pile, about 70% of landings fell on one of the last three), and with it landings spread over the mosaic as they would without the rule, in nearly the preferred order. Assemble then spaces landings evenly, so piles and grids build at the same steady pace, and slows down over the last **Wind-down** share of the duration (`landing_times`) until the last two land **Last gap** apart: at an even pace a fixed number of tiles land in every frame, so the last few, with little else moving, would visibly land at once. A test checks every registered choreography against a real Photo Pile: an overlapping lower tile is never drawn above the tile that covers it, unless it is higher in the air. Tiles that have touched down draw in stacking order even while bouncing; only tiles still falling draw by height. **Stacked settling:** a tile that another lands on (`cover_after_impact`) makes only the hops that end back on the table before the other lands (else, seen from the camera, it would rise and grow out from under a tile lying flat on it), and its rocking goes on past that and dies out within `COVER_DAMP` (0.15 s), as if pressed down. Impacts never move, so landing orders, pacing and the random-order statistics are unaffected. Assemble fits each tile's own settling into Duration; the window comes from tiles nothing lands on (whether a tile is covered depends only on the order), so the last tile comes to rest exactly at the end.
- **Camera moves** (`core/animation/camera.py`, the Animate tab's Camera group): independent of the choreography, but driven by the same animation time. A `CameraMove` (registered with `@register_camera_move`, settings as `Param`s) plans a `CameraPath`, and `path.shot(t)` gives the `Shot` at time t: the table point at the middle of the frame, a zoom relative to the video's framing (zoom 1 is exactly `view_rect`) and a turn (clockwise; the picture looks turned back). A move animates the framing, like a long lens panning, zooming and turning over the picture; the table camera of look.py stays above the middle, so perspective, shadows and the choreographies' rules about its axis still hold. Timing is a share of the animation (**Moves during**), so a move stays in step when Duration changes. Built-ins: **Static** (default); **Pull back** (from a close-up on a **Focus** point, **Close-up** times closer, out to the whole view) and **Push in** (the reverse); **Pan** (close up from one side or corner to another, then, by default, pulling back over the last quarter of the move); **Rotate** (starts turned, and optionally closer, and turns upright onto the whole view); **Follow** (keeps the landing spots of the tiles in flight in view: the middle 80% of them, with **Room** around, no closer than **Closest**, averaged over **Smoothness**, and easing back to the whole view over the last share; it samples the timeline 96 times, about 40 ms for 2,400 tiles). Moves other than Follow are `Segment`s, each a spiral similarity: the frame scales and turns about one fixed table point, so nothing slides across the frame on its own (a pure zoom when the turn doesn't change, a straight pan when neither does), with the zoom changing geometrically; close-ups near an edge shift just enough to stay on the mosaic. The video renders each moment blended into a frame through its own shot, so camera motion blurs too, and requests tile textures at the path's closest zoom (`max_zoom`), so close-ups stay sharp. `Camera2D` turns for this (`rotation`, also in the sprite shader). On the Animate tab, Show export frame draws the current shot, and **Follow camera** keeps the view on it, turning with it; panning or zooming by hand stops following.
- **Camera keyframes** (`core/animation/keyframes.py`; on the Animate tab, `ui/steps/camera_keys.py`). A `CameraTrack` holds `CameraKey`s: a `Shot` at a `KeyTime`, whether the camera **stops** there (eases to rest) or **passes through**, and how it moves on to the next key (Smooth, Steady, or Hold then cut). Keys live on the **video clock** (`VideoClock` in video.py: start hold, animation, end hold), so the camera can move over the empty table before the build and over the finished mosaic after it; tiles still follow animation time, still during the holds. A `KeyTime` anchors a key to its part of the video (seconds into the start hold; a share of the animation, or seconds from its start when the track is pinned with `stretch=False`; seconds after the end), so keys stay put relative to their part when durations change. Motion: a shot is a similarity map of the table onto the frame (p -> alpha p + beta, complex; alpha = zoom e^(-i turn), its log kept unwrapped so keys can spin several turns); between keys the camera follows a cubic Bezier of such maps (de Casteljau with geodesics, the spiral segments of camera.py), with Catmull-Rom tangents through keys that pass and none at keys that stop. Keys about one table point therefore keep that point fixed in the frame all the way, with no sideways drift (tested). Until the track has keys, the chosen camera move runs on animation time (`video_camera_path`, the one place that decides the camera for preview, export and the export dialog). The Animate preview plays on the video clock too (`ClockedTimeline`), holds included; `VideoPlan.video_moments` gives the camera's times for each frame's motion-blur samples and `moments` the same instants in animation time.
- **Keying the camera** (Animate tab). **Look through camera** (C, or the viewfinder button left of the transport) turns the view into the camera: the export frame stays put and panning, zooming (wheel) and turning (Shift+wheel, `MosaicCanvas.rotatable`) the mosaic under it frames a shot, held as a pending framing until **Add key** (K) keeps it at the current time (replacing a key at that moment); moving in time drops a framing that wasn't keyed. Without the viewfinder, K keys what the camera shows now. **[** / **]** jump to the previous / next key, **Delete** removes the selected key. The **Keyframes** group (top of the side panel) edits the selected key exactly (time, where the frame's middle is across and down the mosaic, zoom, turn, Comes to rest, Then) and whether keys keep their share of the animation when its length changes. Every edit replaces the project's `CameraTrack` through `Session.set_camera_track` (preview, export and unsaved changes follow). While the track has no keys, the Camera group's move runs.
- `ui/render/player.py` plays a timeline on a canvas (`TimelinePlayer`, `seek`/`play`/`pause`, `loop`, `speed`). Under the Animate tab's view, `ui/widgets/transport.py` (`TransportBar`) holds the controls: a full-width timeline strip (`ui/widgets/timeline_strip.py`: a ruler, the start and end holds shaded, the playhead, and camera keys as marks, a diamond where the camera stops and a circle where it passes through; click or drag empty ground to scrub, drag a key to retime it, right-click it for Comes to rest, Then and Delete), the time, to start / previous frame / play / next frame / to end centered, Loop and playback speed (0.25x to 2x). Keys while the view has focus: Space, Home, End, Left / Right (one frame of the export's frame rate, so stepping lands on exported frames), Shift+Left / Right (one second). The **Animate** tab (unlocked while a valid mosaic exists, even if matching settings changed since it was made) picks a choreography, generates its settings form, and plays or scrubs it; it opens on the finished mosaic, Play runs from the start, and leaving the tab pauses. Updating every tile each frame with numpy costs about 0.25 µs per tile (15,000 tiles run at over 100 fps).

## Animation export

**Mosaic → Export Animation** (Ctrl+Shift+E, or the Animate tab's button) renders the current choreography to a file in the background.

- **Formats** (`core/animation/video.py`): MP4 (H.264, H.265, AV1), WebM (VP9), animated WebP, GIF, MOV (ProRes 4444 with alpha, else 422 HQ) and PNG sequences. Transparency (Animate tab background: Transparent) is offered only by WebM, WebP, ProRes and PNG; other formats refuse it with an explanation.
- **Settings**: resolution presets (720p to 4K, square, 4:5, 9:16, mosaic shape, custom up to 4K), frame rate (24 to 60 or custom; 29.97 means 30000/1001), Fit or Fill framing with margin, holds at start and end, loop (GIF/WebP), quality and encoding speed, supersampling and motion blur (samples, shutter angle).
- **Rendering** (`ui/render/video_renderer.py`): an offscreen OpenGL context on the worker thread, the same sprite shader as the preview (so the video matches the Animate tab), 8x multisampling times supersampling² samples per pixel (no seams between tiles), linear-light premultiplied blending in float buffers, motion blur from several moments per frame, and sections of at most 2048 px so GPU memory stays bounded. Tile textures are cut from the original photos at output resolution. The last frame is exactly the finished mosaic and matches the still export.
- **Encoding** (`core/animation/encode.py`): frames stream into PyAV, BT.709 conversion and tags, MP4 fast start, a two-pass optimized palette for GIF. Output goes to a `.part` file (or folder) renamed only on success; cancel or failure deletes it.
- **Animate tab**: Background (color or transparent, previewed as a checkerboard); a **Video** group with the settings that decide what the video shows (resolution, width, height, framing, margin) and **Show export frame**, which outlines that area and dims the rest. The tab and the Export Animation window edit the same `project.video_settings`, refresh each other (`ParamForm.refresh`), and the window opens with whatever the tab shows.

## Layout

```
src/skitter/
  app.py            entry point
  core/             numpy-only image processing and mosaic logic
    imaging.py      load/save helpers, the transparency mask (visible_mask, fill_hidden)
    easing.py       vectorized easing curves
    edits.py        non-destructive edits: flip, rotate, crop
    geometry.py     rectangle math for interactive tools (crop box)
    color.py        sRGB <-> OKLab (vectorized and for numba kernels)
    project.py      Project dataclass (state across all steps)
    project_file.py  .skitter project files: save, forgiving load
    scene.py        MosaicScene: the finished mosaic's tiles in their final state
    assembly.py     full-detail mosaic rendering and export settings
    animation/      choreographies (Assemble, Deal), camera moves, timelines, tile frames, landing orders; video settings and encoding
    tiles/          tile library
      library.py    on-disk cache: sqlite metadata + memory-mapped thumbnails
      ingest.py     parallel, reduced-scale thumbnail decoding
      crops.py      aspect classes and aspect-guided crop windows
      render.py     full-detail crops read from the original tile files
      descriptors.py  OKLab grid-pyramid descriptors of tile crops
      palette.py    what the library can paint: color map, source coverage, photo stats
    matching/       region-to-tile matching
      targets.py    region descriptors with visibility masks
      raster.py     rasterized region stacks
      index.py      candidate sets, faiss index, exact rerank and search
      assign.py     reuse-constrained assignment and refinement
      quality.py    proxy render and distance-blurred scores; local rescoring
      candidates.py remembered candidates per region; pins
      edit.py       manual picks: apply, revert, find more, reuse status
      matcher.py    the pipeline's entry point and its caches (Matcher)
      run.py        one run, phase by phase (MatchRun), and its progress (Reporter)
      result.py     MatchResult, MatchPreview and preview stages
      saved.py      a matched mosaic as a project file keeps it, and restored
      settings.py   MatchSettings (Params)
    slicing/        slicing framework
      layout.py     MosaicLayout: tile aspect and columns; mosaic units
      regions.py    Region / RegionSet (rotated rectangles)
      params.py     declarative operation parameters
      base.py       SliceContext, SlicingOperation, Subdivider, registry
      plan.py       SlicingPlan, stages, cached evaluation
      analysis.py   coverage, density and size summary
      patterns.py   brick pattern framework (repeating units, tiler)
      mask.py       drops regions that touch no visible pixel (transparency mask)
      structure.py  edge, texture and flow analysis that contour rows follow
      rows.py       numba row tracing for contour rows
      operations/   built-ins: grid, split, bond, pattern, quadtree, pile, contour, jitter, stacking
  ui/
    main_window.py  tabbed window, Back/Next footer, step gating, menus
    session.py      observable Project wrapper shared by steps
    export_dialog.py  Export Image window
    video_dialog.py   Export Animation window
    steps/
      base.py       StepPage base class
      source.py     step 1: source image selection and editing
      slicing.py    step 2: slicing plan editor and region preview
      tiles.py      step 3: tile library folders, update, statistics, color map and coverage
      matching.py   step 4: matching settings, run, mosaic preview, heat map
      edit_mode.py  step 4's edit mode: selecting, previewing and picking tiles by hand
      tile_picker.py  picking tiles by hand: side panel, canvas overlays, crop reader
      animate.py    step 5: choreography settings, playback and scrubbing of the build animation
    jobs.py         background jobs with progress and cancel
    preferences.py  per-user display preferences (QSettings INI)
    style.py        shared looks: muted and warning text, progress bars
    widgets/
      image_viewer.py  canvas + scrollbars + zoom bar + edit transitions
      crop_overlay.py  interactive crop box over a canvas
      region_overlay.py  stacked GPU preview of a RegionSet (fill, outlines, hover; adjustable line color and opacity)
      structure_overlay.py  what contour rows follow (edges, mask edge, texture, flow), drawn over the canvas
      wheel_guard.py   keeps the mouse wheel from changing a setting by accident
      param_form.py    settings form generated from Params
      candidate_grid.py  grid of tile candidates to pick from
    canvas.py       GPU canvas widget: layers, animations, navigation
    icons.py        vector toolbar icons drawn with QPainter
    render/
      camera.py     2D pan/zoom math
      sprites.py    SpriteLayer data and instanced renderer
      atlas.py      texture atlases: thumbnail cells (fixed or growing), packed full-detail crops
      tile_textures.py  a scene's tile textures, shared by views (full detail in background)
      match_preview.py  a matching run in progress as sprite layers (built in background)
      player.py     plays an animation timeline on a sprite layer
      video_renderer.py  offscreen GPU frames for video export (supersampling, motion blur)
      video_export.py    the video export job (textures, frames, encoding)
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
pytest
ruff check .
python scripts/bench_matching.py --candidates 1000000 --regions 50000
```

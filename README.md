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

- A **slicing operation** transforms regions: `apply(regions, ctx) -> regions`.
- A **plan** (`project.slicing_plan`) is an ordered list of stages, each an operation plus an enabled flag. It starts from one region covering the whole canvas.
  - Splitting, adjusting, filtering and hand-drawn regions all fit this one interface, and operations nest: a grid inside jittered regions gives rotated cells.
- `Session` re-evaluates the plan when it, the layout, or the final image changes, reusing cached results for unchanged leading stages.
- The Slicing tab lists the stages (add from a menu grouped by category, reorder, enable/disable, remove), generates a settings form from the selected operation's parameters, and draws the regions on the GPU.
- Built-in operations:
  - **Grid**: cells of the Mosaic tile shape (scaled by Cell size), with centered overhang. Its only sizes are in base tiles, so the Mosaic box alone sets tile size and shape.
  - **Split**: divides each region into Across x Down equal pieces (counts are per region, e.g. per jittered photo). Keep tile shape picks Down so pieces match the tile aspect.
  - **Brick Bond**: courses of base-tile bricks, each shifted along its length (running, third, quarter, custom or random). Courses run horizontally (rows shift) or vertically (columns shift).
  - **Brick Pattern**: repeating patterns that mix brick directions: herringbone (any tile shape, any rotation; 45° gives diagonal herringbone) and basketweave. Bricks lying the other way are base tiles turned 90°, so regions keep the tile shape.
  - **Quadtree**: splits where the image has detail, down to a minimum in tiles. Use after a Grid.
  - **Photo Pile**: overlapping rotated photos in the tile shape. Spread 1.0 guarantees coverage.
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
   - Building candidates and an index takes a while for large libraries (about 30 s for 300,000 photos, mostly training the IVF clusters). It reports progress as it goes, shown under the Matching status as a second, thinner bar, and Cancel stops it. The clusters are trained one k-means round at a time (`train_ivf`), which gives exactly faiss's own result; tests check that. Indexes are kept between runs, so the bar shows only when the library, crop settings or weights change, or the tint changes by more than 0.15.
   - The top `candidates` are reranked exactly. The cost is `(1 - t)^2 |mean difference|^2 + |structure difference|^2`, with per-dimension weights, where `t` is the tint strength. A crop penalty favors tiles that keep more of their photo.
   - Before the full search, a sample of regions is also searched exactly. If the approximate results cost noticeably more (the regret), search effort is raised.
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
- While matching runs, the tab shows the run instead of the old mosaic. `Matcher.run(preview=...)` hands out a `MatchPreview` at each point with something to show: target colors, then each region's best tile before the reuse rules, filling in batch by batch as the search goes. Then the first assignment, where regions still waiting show their target color. During widening, each waiting region shows a tentative tile as its batch finishes: its cheapest candidate the reuse rules allow given the tiles already placed (`Assignment.first_allowed`), which is close to what the assignment then picks. Then the complete assignment, and every refining and adaptive pass. Previews are copies and only read the run's state, so results are identical with or without them. `ui/render/match_preview.py` builds their sprite layers (thumbnail atlases) in the background, newest first, at most every 0.2 s; Display › Tile detail names the stage.
  - A background job then reads each used tile from its original file and cuts its crop at one texel per mosaic unit (`core/tiles/render.py`, process pool, JPEGs decoded at reduced scale). The crops are shelf-packed into atlas pages (`pack_images`) and replace the thumbnails. Regions showing the same crop at the same size share one image. If the total would exceed `DETAIL_TEXELS` (about 400 MB of GPU memory), every crop is scaled down by the same factor. Unreadable files fall back to their thumbnails.
- `scripts/bench_matching.py` benchmarks index build, search accuracy and speed, and assignment at library scale.

### Picking tiles by hand

The result remembers each region's 32 best candidates (`candidates.py`; indices into the candidate sets matching already holds, about 8 bytes each: 15 MB for 60,000 regions). On the Matching tab, double-clicking a tile (or **Edit Tiles**, E) enters edit mode (`ui/steps/tile_picker.py`):

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
- **Overlapping tiles land bottom first.** `scene.overlaps` lists every overlapping pair (rotated rectangles, exact), and `landing_order` turns each tile's preferred place in the sequence into a landing order in which a tile never lands before the tiles it lies on (a prioritized topological sort), so nothing pops under its neighbors on landing. Random order instead draws a **uniformly random bottom-first order** (`random_landing_order`: independent uniform keys conditioned on lower < upper, sampled by Gibbs sweeps, about 8 ms for 3,400 tiles), so tiles land close together in space and time exactly as often as chance has it; random preferences through `landing_order` would land a held-back tile right beside the one it lies on (92% of landings had a neighbor within 0.15 s on a pile, against 50% by chance). Orders without a direction are listed in `assemble.RANDOM_ORDERS`, and a test holds every order to its kind: random ones to chance, directed ones to their direction. For directed orders, a tile freed by the landing of its last support then waits a few landings (`landing_gap`: n / 150, 3 to 50) while others can go; otherwise the sequence climbs one stack after another (on a pile, about 70% of landings fell on one of the last three), and with it landings spread over the mosaic as they would without the rule, in nearly the preferred order. Assemble then spaces landings evenly, so piles and grids build at the same steady pace (adjustable pacing is a possible later setting). A test checks every registered choreography against a real Photo Pile: an overlapping lower tile is never drawn above the tile that covers it, unless it is higher in the air. Tiles that have touched down draw in stacking order even while bouncing; only tiles still falling draw by height. **Stacked settling:** a tile that another lands on (`cover_after_impact`) makes only the hops that end back on the table before the other lands (else, seen from the camera, it would rise and grow out from under a tile lying flat on it), and its rocking goes on past that and dies out within `COVER_DAMP` (0.15 s), as if pressed down. Impacts never move, so landing orders, pacing and the random-order statistics are unaffected. Assemble fits each tile's own settling into Duration; the window comes from tiles nothing lands on (whether a tile is covered depends only on the order), so the last tile comes to rest exactly at the end.
- `ui/render/player.py` plays a timeline on a canvas (`TimelinePlayer`, `seek`/`play`/`pause`). The **Animate** tab (unlocked while a valid mosaic exists, even if matching settings changed since it was made) picks a choreography, generates its settings form, and plays or scrubs it; it opens on the finished mosaic, Play runs from the start, and leaving the tab pauses. Updating every tile each frame with numpy costs about 0.25 µs per tile (15,000 tiles run at over 100 fps).

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
    imaging.py      load/save/resize/crop helpers
    easing.py       vectorized easing curves
    edits.py        non-destructive edits: flip, rotate, crop
    geometry.py     rectangle math for interactive tools (crop box)
    color.py        sRGB <-> OKLab (vectorized and for numba kernels)
    project.py      Project dataclass (state across all steps)
    scene.py        MosaicScene: the finished mosaic's tiles in their final state
    assembly.py     full-detail mosaic rendering and export settings
    animation/      choreographies, timelines, tile frames; video settings and encoding
    tiles/          tile library
      library.py    on-disk cache: sqlite metadata + memory-mapped thumbnails
      ingest.py     parallel, reduced-scale thumbnail decoding
      crops.py      aspect classes and aspect-guided crop windows
      render.py     full-detail crops read from the original tile files
      descriptors.py  OKLab grid-pyramid descriptors of tile crops
    matching/       region-to-tile matching
      targets.py    region descriptors with visibility masks
      raster.py     rasterized region stacks
      index.py      candidate sets, faiss index, exact rerank and search
      assign.py     reuse-constrained assignment and refinement
      quality.py    proxy render and distance-blurred scores; local rescoring
      candidates.py remembered candidates per region; pins
      edit.py       manual picks: apply, revert, find more, reuse status
      matcher.py    the pipeline and its caches
      settings.py   MatchSettings (Params)
    slicing/        slicing framework
      layout.py     MosaicLayout: tile aspect and columns; mosaic units
      regions.py    Region / RegionSet (rotated rectangles)
      params.py     declarative operation parameters
      base.py       SliceContext, SlicingOperation, Subdivider, registry
      plan.py       SlicingPlan, stages, cached evaluation
      analysis.py   coverage, density and size summary
      patterns.py   brick pattern framework (repeating units, tiler)
      operations/   built-ins: grid, split, bond, pattern, quadtree, pile, jitter, stacking
  ui/
    main_window.py  tabbed window, Back/Next footer, step gating, menus
    session.py      observable Project wrapper shared by steps
    export_dialog.py  Export Image window
    video_dialog.py   Export Animation window
    steps/
      base.py       StepPage base class
      source.py     step 1: source image selection and editing
      slicing.py    step 2: slicing plan editor and region preview
      tiles.py      step 3: tile library folders, update, statistics
      matching.py   step 4: matching settings, run, mosaic preview, heat map, edit mode
      tile_picker.py  picking tiles by hand: side panel, canvas overlays, crop reader
      animate.py    step 5: choreography settings, playback and scrubbing of the build animation
    jobs.py         background jobs with progress and cancel
    widgets/
      image_viewer.py  canvas + scrollbars + zoom bar + edit transitions
      crop_overlay.py  interactive crop box over a canvas
      region_overlay.py  stacked GPU preview of a RegionSet (fill, outlines, hover)
      param_form.py    settings form generated from Params
      candidate_grid.py  grid of tile candidates to pick from
    canvas.py       GPU canvas widget: layers, animations, navigation
    icons.py        vector toolbar icons drawn with QPainter
    render/
      camera.py     2D pan/zoom math
      sprites.py    SpriteLayer data and instanced renderer
      atlas.py      texture atlases: thumbnail cells, packed full-detail crops
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

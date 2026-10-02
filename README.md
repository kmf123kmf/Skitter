# Skitter

A photo mosaic generator with an animated graphical interface.

## Stack

- **numpy** and **Pillow** handle image processing and mosaic algorithms (`skitter.core`)
- **PySide6 (Qt 6)** provides the application shell: windows, menus, panels, dialogs (`skitter.ui`)
- **moderngl** drives the canvas. It needs OpenGL 3.3 or newer. All tiles in a layer draw in a single instanced call (`skitter.ui.render`)

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

- Each step of mosaic generation is a tab: a `StepPage` subclass listed in order in `skitter/ui/steps/__init__.py`. Currently: Source, then Slicing.
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

Slicing divides `project.source_final` into regions for tile matching. The result is always a `RegionSet`: rectangles with a center, size and rotation (radians, clockwise on screen), stored as parallel numpy arrays (`skitter.core.slicing`, Qt-free).

- A **slicing operation** transforms regions: `apply(regions, ctx) -> regions`.
- A **plan** (`project.slicing_plan`) is an ordered list of stages, each an operation plus an enabled flag. It starts from one region covering the whole image.
  - Splitting, adjusting, filtering and hand-drawn regions all fit this one interface, and operations nest: a grid inside jittered regions gives rotated cells.
- `Session` re-evaluates the plan when it or the final image changes, reusing cached results for unchanged leading stages.
- The Slicing tab lists the stages (add from a menu grouped by category, reorder, enable/disable, remove), generates a settings form from the selected operation's parameters, and draws the regions on the GPU.
- Built-in operations: **Grid**, **Quadtree** (splits where the image has detail), **Photo Pile** (overlapping rotated photos; coverage guaranteed), **Jitter**, **Gap**, **Stacking Order**.

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
- `ctx.image` / `ctx.luminance` give the read-only final image. `ctx.patch(region)` samples the pixels inside a (possibly rotated) region on a grid aligned with it.
- Operations must be deterministic (take a seed parameter for randomness) and must not modify inputs. Set `z` when the regions you create overlap (see above). RegionSets are immutable: build new ones with `replace`, `from_arrays`, `from_rects`, `grid`, `concat`.
- Optionally override `summary()` for the stage list. Make sure the module is imported (built-ins are imported by `skitter/core/slicing/operations/__init__.py`).

## Layout

```
src/skitter/
  app.py            entry point (--demo N runs the flying-tiles demo)
  core/             numpy-only image processing and mosaic logic
    imaging.py      load/save/resize/crop helpers
    easing.py       vectorized easing curves
    edits.py        non-destructive edits: flip, rotate, crop
    geometry.py     rectangle math for interactive tools (crop box)
    project.py      Project dataclass (state across all steps)
    slicing/        slicing framework
      regions.py    Region / RegionSet (rotated rectangles)
      params.py     declarative operation parameters
      base.py       SliceContext, SlicingOperation, Subdivider, registry
      plan.py       SlicingPlan, stages, cached evaluation
      operations/   built-ins: grid, quadtree, pile, jitter, gap, stacking
  ui/
    main_window.py  tabbed window, Back/Next footer, step gating, menus
    session.py      observable Project wrapper shared by steps
    steps/
      base.py       StepPage base class
      source.py     step 1: source image selection and editing
      slicing.py    step 2: slicing plan editor and region preview
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
  resources/
    shaders/        GLSL sources
tests/              pytest suite (headless; Qt widgets run offscreen)
```

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Run

```powershell
python -m skitter      # or just: skitter
python -m skitter --demo 50000
pytest
ruff check .
```

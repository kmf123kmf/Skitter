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
- Use the vectorized curves in `skitter.core.easing` to animate thousands of tiles in one numpy expression.
- World coordinates follow image conventions (y points down). Sprite `pos` is the sprite's center.

## Workflow tabs

- Each step of mosaic generation is a tab: a `StepPage` subclass listed in order in `skitter/ui/steps/__init__.py`.
- All steps share one `Session`, which wraps the `core.project.Project` data and emits signals when it changes.
  - Steps change the project only through `Session` methods, so other steps are notified.
- A tab is enabled only when every earlier step reports `is_complete()`.
  - Steps emit `completion_changed` whenever that may have changed.
  - Steps get `on_enter()` / `on_leave()` calls when their tab is switched to or away from.

To add a step: subclass `StepPage`, set `title`, implement `is_complete()`, and append the class to `STEPS`.

## Layout

```
src/skitter/
  app.py            entry point (--demo N runs the flying-tiles demo)
  core/             numpy-only image processing and mosaic logic
    imaging.py      load/save/resize/crop helpers
    easing.py       vectorized easing curves
    project.py      Project dataclass (state across all steps)
  ui/
    main_window.py  tabbed window, step gating, menus
    session.py      observable Project wrapper shared by steps
    steps/
      base.py       StepPage base class
      source.py     step 1: source image selection
    canvas.py       GPU canvas widget: layers, animations, pan/zoom input
    demo.py         flying-tiles stress demo window
    render/
      camera.py     2D pan/zoom math
      sprites.py    SpriteLayer data and instanced renderer
  resources/
    shaders/        GLSL sources
tests/              pytest suite (Qt-free parts)
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

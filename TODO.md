# To do

## Survive running out of GPU memory (before shipping to other machines)

It works on an RTX 3080 (10 GB). An Infinite Mosaic session holds about 1–1.3 GB of GPU
memory at 40 columns (more at 64), including the stacks reserved at start:

- 3 detail-page stacks, 432 MB
- 2 two-page 4096² stacks and 1 thumbnail stack, about 360 MB

On a smaller GPU, nothing handles running out:

- moderngl doesn't raise when the driver can't back a texture. The texture exists with no
  storage, and the tiles drawn from it come out black or garbage.
- With no detail stack, ZoomDetail still points sharpened tiles at detail pages. The shader
  then reads them from the tile texture at a page past its end, so they show the wrong
  photo.
- An exception in `paintGL` repeats every frame. The error hook in `app.py` would then show
  a dialog for each one.

Plan, steps 1–3 first:

1. Detect it. Read `ctx.error` right after each large allocation (tile-layer stacks, detail
   stacks, mipmap builds, `reserve_details`, `reserve_textures`). On `GL_OUT_OF_MEMORY`,
   release the texture and treat it as failed. This costs nothing per frame.
2. Fall back for each kind of texture:
   - reservations: skip them
   - a detail stack: no detail for the session, and ZoomDetail stops, so tiles keep their
     usual textures
   - an Infinite Mosaic full-size layer: keep the thumbnail layer
   - a main app tile layer: retry at a smaller `DETAIL_TEXELS`, then fall back to
     thumbnails
   - the photo: a scaled-down copy

   After one failure, lower the budgets for the rest of the session (fewer detail pages,
   smaller `DETAIL_TEXELS`, no reservations).
3. Make the error hook safe: one dialog at a time, and each distinct error shown only once
   (later ones only logged).
4. Optional: read free video memory at start (`GL_NVX_gpu_memory_info`, `GL_ATI_meminfo`;
   Intel doesn't report it) and start with the smaller budgets when it's tight. This also
   helps when Windows swaps textures out to system memory instead of failing, which runs
   but stutters badly.

Tests can simulate `GL_OUT_OF_MEMORY` with a patched allocation. Using up the GPU for real
isn't safe to test. A driver crash ("display driver stopped responding") is out of reach.

## Frame pacing (low priority: optimization)

Each animated frame steps the motion by the wall-clock time between `frameSwapped`
callbacks and then asks for the next frame at once. So the app runs as fast as the display
allows. Measured 2026-11-01 (Infinite Mosaic, 144 Hz, windowed): steady motion is even
within about ±1.5 ms (1–2% of frames step more than 15% off the usual size).

The right fix depends on the display mode. On the dev machine, G-Sync engages only in full
screen:

- **Fixed refresh (windowed):** round each motion step to whole display refreshes, so steady
  motion is exactly even. Don't cap frames below the refresh here: that repeats a frame
  about 3 times a second.
- **Variable refresh (full screen with G-Sync or FreeSync):** wall-clock steps are already
  right; don't round them. At the 144 fps ceiling G-Sync acts like vsync and frames queue,
  so cap at about 3 fps under `QScreen.refreshRate()` to keep it in range. Test first with
  the driver's Max Frame Rate (141) for the venv's python.exe, full screen, against no cap.
  Add the in-app cap only if the zoom is clearly smoother.

If built: in-app pacing in `MosaicCanvas` that follows the mode (full screen: cap; windowed:
rounded steps), about 30–40 lines. Guessing the mode from full screen fits the dev machine
but not every PC, so a setting would be needed: "Frame pacing: Automatic / Variable refresh
/ Fixed refresh".

`frameSwapped` times are when the driver accepts a frame, not when it is shown. To judge
pacing properly, measure display times (`DwmGetCompositionTimingInfo`).

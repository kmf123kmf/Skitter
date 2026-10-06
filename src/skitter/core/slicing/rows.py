"""Tiles laid in rows along a row field (structure.py), then gaps filled.

Coordinates are in samples of the field: sample (j, i) covers [i, i + 1) x
[j, j + 1). Tiles are `length` along their row and `height` across it.

Rows: row k's centerline is the level set rows = k + 1/2. Rows nearest the
guides go first. From each seed point on a centerline not yet covered, the
row is traced both ways: step a tile length along the level set (a midpoint
step, then Newton steps back onto it) and lay a tile on the chord. A trace
stops where the next tile would overlap tiles already laid by more than
`max_overlap` of its area, at a sharp turn or where the field has no clear
direction (where rows from two guides meet), on hidden samples, or past the
patch. An occupancy grid records what is covered.

Outline rows keep out of `texture` and `background` (structure.py): their
seeds and tiles stay off them. The background gets straight rows, laid the
same way along the field `straight`.

Flow rows: in texture, and wherever outline rows left room (where rows
from two guides meet), tiles follow the flow instead: rows traced along it
the same way, without a level set to keep to. Seeds in texture go first,
each set from the uncovered samples nearest what is already laid outward,
so flow rows pack against the rows beside them; the first tile of each is
nudged across the flow, up to half a tile height either way, to fit
against its neighbors.

Gaps: the samples still uncovered (visible ones) get filler tiles turned
along the flow, each covering enough new ground, first on a coarse lattice,
then on every sample. Slivers too thin to cover 5% of a tile are left: a
whole tile for each would nearly double the count, and they read as grout
between rows.

Stacking: outline rows nearer guides on top, so edges stay clean; then
background rows; then flow rows; fillers under everything, so they only
show through the gaps.
"""

import math

import numpy as np
from numba import njit
from scipy import ndimage

from skitter.core.slicing.structure import StructureField

MIN_SLOPE = 0.5  # of a clean slope (1 row per row height): less is where rows meet
MAX_TURN = math.radians(50.0)  # sharpest bend a row takes from one tile to the next
# (lattice step in tile heights, new share of a tile it must cover, at least)
FILL_PASSES = ((0.5, 0.3), (0.0, 0.05))
NUDGES = (0.0, 0.25, -0.25, 0.5, -0.5)  # a flow row's first tile, across the flow (heights)


def place_rows(field: StructureField, visible, length: float, height: float,
               max_overlap: float, fill: bool = True,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """(centers (N, 2), angles (N,), z (N,)) of the tiles, in samples and radians
    (clockwise, along the row); z stacks outline rows (nearer guides on top), then
    background rows, then flow rows, then fillers (see the module docstring)."""
    visible = np.ascontiguousarray(visible, np.bool_)
    cap = int(4 * visible.size / max(length * height, 1e-9)) + 64
    out = np.zeros((cap, 4), np.float64)  # cx, cy, angle, z
    occupied = np.zeros(visible.shape, np.bool_)
    rows, (gx, gy) = field.rows, field.grad
    outline = visible & ~field.texture & ~field.background
    n, top = _level_rows(rows, gx, gy, outline, occupied, length, height, max_overlap, out, 0, 0.0)
    if field.background.any():
        nx, ny = field.straight_grad
        straight_x = np.full(visible.shape, nx, np.float32)
        straight_y = np.full(visible.shape, ny, np.float32)
        n, top = _level_rows(field.straight, straight_x, straight_y, visible & field.background,
                             occupied, length, height, max_overlap, out, n, top)  # fmt: skip
    fx = np.ascontiguousarray(field.flow[0], np.float32)
    fy = np.ascontiguousarray(field.flow[1], np.float32)
    free = visible & ~occupied
    if free.any():
        near = ndimage.distance_transform_edt(free)  # to what is laid (or hidden)
        fj, fi = np.nonzero(free)
        order = np.lexsort((near[fj, fi], ~field.texture[fj, fi]))  # texture first
        n = _lay_flow(fx, fy, visible, occupied, fj[order], fi[order], length, height,
                      max_overlap, top - 0.5, out, n)  # fmt: skip
    if fill:
        for step, share in FILL_PASSES:
            n = _fill(fx, fy, visible, occupied, max(step * height, 1.0), share,
                      length, height, -top + 1.0, out, n)  # fmt: skip
    return out[:n, :2].copy(), out[:n, 2].copy(), out[:n, 3].copy()


def _level_rows(rows, gx, gy, allowed, occupied, length, height, max_overlap, out, n,
                top: float) -> tuple[int, float]:  # fmt: skip
    """Lay rows along the level sets of a row field on `allowed` samples, row k at z
    top - k; (tiles laid so far, the z below them all)."""
    rows = np.ascontiguousarray(rows, np.float32)
    gx = np.ascontiguousarray(gx, np.float32)
    gy = np.ascontiguousarray(gy, np.float32)
    allowed = np.ascontiguousarray(allowed, np.bool_)
    slope = np.hypot(gx, gy)
    offset = np.abs(rows - np.floor(rows) - 0.5)
    # Samples within half a sample of a centerline (a centerline midway between two
    # samples seeds both).
    seed = allowed & (offset <= 0.5 / height + 1e-6) & (slope > MIN_SLOPE / height)
    sj, si = np.nonzero(seed)
    level = np.floor(rows[sj, si]).astype(np.int64)
    order = np.lexsort((si, sj, level))
    n = _lay_rows(rows, gx, gy, allowed, occupied, sj[order], si[order], level[order],
                  length, height, max_overlap, top, out, n)  # fmt: skip
    return n, top - (float(level.max()) + 1.0 if len(level) else 0.0)


@njit(cache=True, nogil=True, inline="always")
def _at(a, x, y):
    """Bilinear sample of a at continuous point (x, y); past the grid, extended along
    its edge cells' slope (rows run on past the patch: tiles overhang its border)."""
    h, w = a.shape
    if w < 2 or h < 2:
        return a[min(max(int(y), 0), h - 1), min(max(int(x), 0), w - 1)]
    fx, fy = x - 0.5, y - 0.5
    i0 = min(max(int(math.floor(fx)), 0), w - 2)
    j0 = min(max(int(math.floor(fy)), 0), h - 2)
    tx, ty = fx - i0, fy - j0  # outside 0..1 past the grid: extrapolates
    top = a[j0, i0] * (1 - tx) + a[j0, i0 + 1] * tx
    bottom = a[j0 + 1, i0] * (1 - tx) + a[j0 + 1, i0 + 1] * tx
    return top * (1 - ty) + bottom * ty


@njit(cache=True, nogil=True)
def _cover(occupied, cx, cy, angle, half_l, half_h, stamp):
    """(samples inside the tile, of them already covered); stamp marks them covered."""
    h, w = occupied.shape
    c, s = math.cos(angle), math.sin(angle)
    ex = abs(c) * half_l + abs(s) * half_h
    ey = abs(s) * half_l + abs(c) * half_h
    total = taken = 0
    for j in range(max(0, int(cy - ey)), min(h, int(cy + ey) + 1)):
        for i in range(max(0, int(cx - ex)), min(w, int(cx + ex) + 1)):
            px, py = i + 0.5 - cx, j + 0.5 - cy
            if abs(c * px + s * py) <= half_l and abs(-s * px + c * py) <= half_h:
                total += 1
                if occupied[j, i]:
                    taken += 1
                elif stamp:
                    occupied[j, i] = True
    return total, taken


@njit(cache=True, nogil=True)
def _onto(rows, gx, gy, x, y, target):
    """Newton steps from (x, y) onto the level set rows = target."""
    for _ in range(3):
        dx, dy = _at(gx, x, y), _at(gy, x, y)
        g2 = dx * dx + dy * dy
        if g2 < 1e-12:
            break
        k = (target - _at(rows, x, y)) / g2
        x, y = x + k * dx, y + k * dy
    return x, y


@njit(cache=True, nogil=True)
def _along(gx, gy, x, y, wx, wy):
    """Unit direction along the rows at (x, y), the way (wx, wy) points; (0, 0) if none."""
    dx, dy = _at(gx, x, y), _at(gy, x, y)
    norm = math.hypot(dx, dy)
    if norm < 1e-12:
        return 0.0, 0.0, 0.0
    tx, ty = -dy / norm, dx / norm
    if tx * wx + ty * wy < 0:
        tx, ty = -tx, -ty
    return tx, ty, norm


@njit(cache=True, nogil=True)
def _trace(rows, gx, gy, visible, occupied, x, y, wx, wy, target, length, height,
           max_overlap, z, out, n):  # fmt: skip
    h, w = rows.shape
    least = MIN_SLOPE / height
    px, py = wx, wy
    while n < out.shape[0]:
        tx, ty, slope = _along(gx, gy, x, y, px, py)
        if slope < least:
            break
        mx, my = x + tx * length / 2, y + ty * length / 2
        tx, ty, slope = _along(gx, gy, mx, my, tx, ty)
        if slope < least:
            break
        nx, ny = _onto(rows, gx, gy, x + tx * length, y + ty * length, target)
        cx, cy = (x + nx) / 2, (y + ny) / 2
        if not (-length < cx < w + length and -length < cy < h + length):
            break
        chord = math.hypot(nx - x, ny - y)
        if chord < 0.5 * length:
            break
        ux, uy = (nx - x) / chord, (ny - y) / chord
        if ux * px + uy * py < math.cos(MAX_TURN):
            break
        ci, cj = int(cx), int(cy)
        if 0 <= ci < w and 0 <= cj < h and not visible[cj, ci]:
            break
        angle = math.atan2(uy, ux)
        total, taken = _cover(occupied, cx, cy, angle, length / 2, height / 2, False)
        if total == 0 or taken > max_overlap * total:
            break
        _cover(occupied, cx, cy, angle, length / 2, height / 2, True)
        out[n, 0], out[n, 1], out[n, 2], out[n, 3] = cx, cy, angle, z
        n += 1
        x, y, px, py = nx, ny, ux, uy
    return n


@njit(cache=True, nogil=True)
def _lay_rows(rows, gx, gy, visible, occupied, sj, si, level, length, height, max_overlap,
              top, out, n):  # fmt: skip
    for k in range(len(sj)):
        j, i = sj[k], si[k]
        if occupied[j, i] or n >= out.shape[0]:
            continue
        target = level[k] + 0.5
        x, y = _onto(rows, gx, gy, i + 0.5, j + 0.5, target)
        tx, ty, slope = _along(gx, gy, x, y, 1.0, 0.0)
        if slope < MIN_SLOPE / height:
            continue
        z = top - float(level[k])
        for way in (1.0, -1.0):  # both ways from the seed
            n = _trace(rows, gx, gy, visible, occupied, x, y, way * tx, way * ty, target,
                       length, height, max_overlap, z, out, n)  # fmt: skip
    return n


@njit(cache=True, nogil=True)
def _fill(fx, fy, visible, occupied, step, share, length, height, bottom, out, n):
    h, w = occupied.shape
    y = 0.5
    while y < h and n < out.shape[0]:
        x = 0.5
        while x < w and n < out.shape[0]:
            i, j = int(x), int(y)
            if visible[j, i] and not occupied[j, i]:
                angle = math.atan2(fy[j, i], fx[j, i])
                total, taken = _cover(occupied, x, y, angle, length / 2, height / 2, False)
                if total > 0 and total - taken >= share * total:
                    _cover(occupied, x, y, angle, length / 2, height / 2, True)
                    out[n, 0], out[n, 1], out[n, 2], out[n, 3] = x, y, angle, -bottom
                    n += 1
            x += step
        y += step
    return n


@njit(cache=True, nogil=True, inline="always")
def _flow_at(fx, fy, x, y, wx, wy):
    """The flow at (x, y), nearest sample, turned the way (wx, wy) points (a flow has no
    sign: averaging across a sign flip would cancel it out)."""
    h, w = fx.shape
    i, j = min(max(int(x), 0), w - 1), min(max(int(y), 0), h - 1)
    tx, ty = fx[j, i], fy[j, i]
    if tx * wx + ty * wy < 0:
        tx, ty = -tx, -ty
    return tx, ty


@njit(cache=True, nogil=True)
def _trace_flow(fx, fy, visible, occupied, x, y, wx, wy, length, height, max_overlap, z,
                out, n):  # fmt: skip
    """A flow row from (x, y), heading (wx, wy): as _trace, along the flow."""
    h, w = occupied.shape
    px, py = wx, wy
    while n < out.shape[0]:
        tx, ty = _flow_at(fx, fy, x, y, px, py)
        tx, ty = _flow_at(fx, fy, x + tx * length / 2, y + ty * length / 2, tx, ty)
        nx, ny = x + tx * length, y + ty * length
        cx, cy = (x + nx) / 2, (y + ny) / 2
        if not (-length < cx < w + length and -length < cy < h + length):
            break
        if tx * px + ty * py < math.cos(MAX_TURN):
            break
        ci, cj = int(cx), int(cy)
        if 0 <= ci < w and 0 <= cj < h and not visible[cj, ci]:
            break
        angle = math.atan2(ty, tx)
        total, taken = _cover(occupied, cx, cy, angle, length / 2, height / 2, False)
        if total == 0 or taken > max_overlap * total:
            break
        _cover(occupied, cx, cy, angle, length / 2, height / 2, True)
        out[n, 0], out[n, 1], out[n, 2], out[n, 3] = cx, cy, angle, z
        n += 1
        x, y, px, py = nx, ny, tx, ty
    return n


@njit(cache=True, nogil=True)
def _lay_flow(fx, fy, visible, occupied, sj, si, length, height, max_overlap, z, out, n):
    for k in range(len(sj)):
        j, i = sj[k], si[k]
        if occupied[j, i] or n >= out.shape[0]:
            continue
        x, y = i + 0.5, j + 0.5
        tx, ty = fx[j, i], fy[j, i]
        angle = math.atan2(ty, tx)
        placed = False
        cx, cy = x, y
        for nudge in NUDGES:  # fit the first tile against its neighbors
            cx, cy = x - ty * nudge * height, y + tx * nudge * height
            total, taken = _cover(occupied, cx, cy, angle, length / 2, height / 2, False)
            if total > 0 and taken <= max_overlap * total:
                placed = True
                break
        if not placed:
            continue
        _cover(occupied, cx, cy, angle, length / 2, height / 2, True)
        out[n, 0], out[n, 1], out[n, 2], out[n, 3] = cx, cy, angle, z
        n += 1
        for way in (1.0, -1.0):  # then the row both ways from its ends
            n = _trace_flow(fx, fy, visible, occupied, cx + way * tx * length / 2,
                            cy + way * ty * length / 2, way * tx, way * ty, length, height,
                            max_overlap, z, out, n)  # fmt: skip
    return n

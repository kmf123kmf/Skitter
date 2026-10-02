"""Rectangle math for interactive tools such as the crop box.

Rects are (left, top, right, bottom) tuples in image pixel coordinates.
Handles name the corner or edge being dragged: "tl", "t", "tr", "r", "br",
"b", "bl", "l".
"""

Rect = tuple[float, float, float, float]

CORNERS = ("tl", "tr", "br", "bl")
EDGES = ("t", "r", "b", "l")


def move_rect(rect: Rect, dx: float, dy: float, bounds: Rect) -> Rect:
    """Translate rect by (dx, dy) without leaving bounds or changing size."""
    left, top, right, bottom = rect
    dx = min(max(dx, bounds[0] - left), bounds[2] - right)
    dy = min(max(dy, bounds[1] - top), bounds[3] - bottom)
    return (left + dx, top + dy, right + dx, bottom + dy)


def resize_rect(
    rect: Rect,
    handle: str,
    point: tuple[float, float],
    bounds: Rect,
    aspect: float | None = None,
    min_size: float = 1.0,
) -> Rect:
    """Drag one handle of rect to point, staying inside bounds.

    The opposite corner or edge stays fixed. With an aspect ratio
    (width / height), dragging an edge resizes the other dimension
    symmetrically about the rect's center line.
    """
    left, top, right, bottom = rect
    b_left, b_top, b_right, b_bottom = bounds
    px, py = point
    horizontal = "l" in handle or "r" in handle
    vertical = "t" in handle or "b" in handle

    if horizontal:
        sx = -1 if "l" in handle else 1
        ax = right if sx < 0 else left
        room_x = ax - b_left if sx < 0 else b_right - ax
        w = min(max((px - ax) * sx, min_size), room_x)
    else:
        cx = (left + right) / 2
        room_x = 2 * min(cx - b_left, b_right - cx)
        w = right - left

    if vertical:
        sy = -1 if "t" in handle else 1
        ay = bottom if sy < 0 else top
        room_y = ay - b_top if sy < 0 else b_bottom - ay
        h = min(max((py - ay) * sy, min_size), room_y)
    else:
        cy = (top + bottom) / 2
        room_y = 2 * min(cy - b_top, b_bottom - cy)
        h = bottom - top

    if aspect:
        if horizontal and vertical:
            # Follow whichever dimension the pointer pulls further.
            if w / h > aspect:
                h = w / aspect
            else:
                w = h * aspect
        elif horizontal:
            h = w / aspect
        else:
            w = h * aspect
        scale = min(1.0, room_x / w, room_y / h)
        w, h = w * scale, h * scale

    if horizontal:
        left, right = (ax - w, ax) if sx < 0 else (ax, ax + w)
    else:
        left, right = cx - w / 2, cx + w / 2
    if vertical:
        top, bottom = (ay - h, ay) if sy < 0 else (ay, ay + h)
    else:
        top, bottom = cy - h / 2, cy + h / 2
    return (left, top, right, bottom)


def fit_aspect(rect: Rect, aspect: float) -> Rect:
    """Largest rect with the given aspect ratio centered inside rect."""
    left, top, right, bottom = rect
    w, h = right - left, bottom - top
    if w / h > aspect:
        w = h * aspect
    else:
        h = w / aspect
    cx, cy = (left + right) / 2, (top + bottom) / 2
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def round_rect(rect: Rect, bounds: Rect) -> tuple[int, int, int, int]:
    """Snap to whole pixels inside integer bounds, at least 1 px each way."""
    b_left, b_top, b_right, b_bottom = (int(v) for v in bounds)
    left = min(max(round(rect[0]), b_left), b_right - 1)
    top = min(max(round(rect[1]), b_top), b_bottom - 1)
    right = max(min(round(rect[2]), b_right), left + 1)
    bottom = max(min(round(rect[3]), b_bottom), top + 1)
    return (left, top, right, bottom)

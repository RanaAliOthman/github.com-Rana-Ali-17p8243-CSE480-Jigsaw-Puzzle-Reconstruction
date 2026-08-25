"""Synthetic jigsaw generation.

Cutting a known image into pieces gives puzzles whose solution is known exactly,
which is useful for two things the photographs cannot provide: unit tests with
deterministic expectations, and puzzles of sizes other than 5x7 for studying how
reconstruction behaves as a puzzle grows.
"""
import numpy as np

__all__ = ["side_curve", "piece_mask", "cut_image"]

TAB, BLANK, FLAT = 1, -1, 0


def _bump(t, centre, halfwidth):
    """Smooth bump of unit height, exactly zero outside the given half width."""
    u = (np.asarray(t, dtype=np.float64) - centre) / halfwidth
    out = np.zeros_like(u)
    inside = np.abs(u) < 1.0
    out[inside] = np.exp(1.0 - 1.0 / (1.0 - u[inside] ** 2))
    return out


def side_curve(kind, samples=200, depth=0.22, width=0.24, seed=None):
    """Signed outward offset along one side, as a fraction of the side length.

    The tab is a smooth bump on a narrower neck, which is what a die-cut jigsaw
    edge looks like.  Every component is compactly supported, so the offset is
    exactly zero over the stretch of the side nearest each corner -- a real
    piece is straight there, and a profile that did not vanish at the corners
    would round them off and displace the corner the descriptor depends on.
    """
    t = np.linspace(0.0, 1.0, samples)
    if kind == FLAT:
        return np.zeros_like(t)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.02, 0.02) if seed is not None else 0.0
    scale = 1.0 + (rng.uniform(-0.08, 0.08) if seed is not None else 0.0)
    centre = 0.5 + jitter
    half = width * scale
    profile = depth * (_bump(t, centre, half)
                       - 0.30 * _bump(t, centre - half * 1.05, half * 0.42)
                       - 0.30 * _bump(t, centre + half * 1.05, half * 0.42))
    return profile * (1 if kind == TAB else -1)


def piece_mask(kinds, size=128, margin=48, samples=200, seed=None):
    """Binary mask of one piece whose four sides have the given types.

    ``kinds`` is ordered top, right, bottom, left using the library's clockwise
    convention, and the body square of side ``size`` is centred in a canvas of
    side ``size + 2 * margin``.
    """
    canvas = size + 2 * margin
    corners = np.array([[margin, margin], [margin + size, margin],
                        [margin + size, margin + size], [margin, margin + size]],
                       dtype=np.float64)
    polygon = []
    for index, kind in enumerate(kinds):
        start, end = corners[index], corners[(index + 1) % 4]
        chord = end - start
        length = np.linalg.norm(chord)
        direction = chord / length
        normal = np.array([chord[1], -chord[0]]) / length   # outward, clockwise
        offsets = side_curve(kind, samples,
                             seed=None if seed is None else seed * 4 + index)
        t = np.linspace(0.0, 1.0, samples)
        points = (start[None, :] + t[:, None] * chord[None, :]
                  + (offsets * length)[:, None] * normal[None, :])
        polygon.append(points[:-1])
    polygon = np.vstack(polygon)

    # Even-odd fill by ray casting; avoids depending on a drawing library.
    ys, xs = np.mgrid[0:canvas, 0:canvas]
    inside = np.zeros((canvas, canvas), bool)
    x0, y0 = polygon[:, 0], polygon[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for a, b, c, d in zip(x0, y0, x1, y1):
        crosses = ((b > ys) != (d > ys))
        with np.errstate(divide="ignore", invalid="ignore"):
            boundary = (c - a) * (ys - b) / np.where(d - b == 0, np.nan, d - b) + a
        inside ^= crosses & (xs < boundary)
    return inside.astype(np.uint8) * 255


def cut_image(image, rows, cols, margin=48, seed=480):
    """Cut ``image`` into an interlocking ``rows`` x ``cols`` puzzle.

    Returns
    -------
    (pieces, layout)
        ``pieces`` is a list of ``(image, mask)`` on the canonical canvas in
        row-major order; ``layout`` records the tab/blank type of every side so
        tests can assert against the true adjacency.
    """
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    size = min(height // rows, width // cols)

    # Choose each internal seam once, then give the two pieces opposite types.
    horizontal = rng.choice([TAB, BLANK], size=(rows, cols - 1))
    vertical = rng.choice([TAB, BLANK], size=(rows - 1, cols))

    pieces, layout = [], []
    for row in range(rows):
        for col in range(cols):
            top = FLAT if row == 0 else -vertical[row - 1, col]
            bottom = FLAT if row == rows - 1 else vertical[row, col]
            left = FLAT if col == 0 else -horizontal[row, col - 1]
            right = FLAT if col == cols - 1 else horizontal[row, col]
            kinds = (top, right, bottom, left)
            mask = piece_mask(kinds, size=size, margin=margin, seed=int(rng.integers(1 << 30)))

            canvas = size + 2 * margin
            patch = np.zeros((canvas, canvas, 3), np.uint8)
            y0, x0 = row * size - margin, col * size - margin
            for dy in range(canvas):
                sy = y0 + dy
                if 0 <= sy < height:
                    lo = max(0, -x0)
                    hi = min(canvas, width - x0)
                    if hi > lo:
                        patch[dy, lo:hi] = image[sy, x0 + lo:x0 + hi]
            patch[mask == 0] = 0
            pieces.append((patch, mask))
            layout.append({"row": row, "column": col, "kinds": kinds})
    return pieces, layout

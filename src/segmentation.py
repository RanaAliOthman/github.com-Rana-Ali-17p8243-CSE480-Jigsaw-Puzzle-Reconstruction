"""Foreground segmentation, binary morphology and connected-component labelling.

The connected-component labeller and the morphological operators are written
from scratch; OpenCV is used only for colour-space conversion and for the
affine warp in :func:`normalize_orientation`.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .enhancement import convolution
from .thresholding import otsu_threshold

__all__ = [
    "Component", "connected_components", "binary_erode", "binary_dilate",
    "binary_open", "binary_close", "foreground_mask", "extract_pieces",
    "normalize_orientation", "propagate_labels", "separate_touching",
    "components_from_labels", "fill_holes",
]


@dataclass
class Component:
    """One labelled connected region."""
    label: int
    area: int
    bbox: tuple      # (x, y, width, height)
    centroid: tuple  # (x, y)
    mask: np.ndarray  # boolean crop of the component within its bounding box


# --------------------------------------------------------------------------
# Binary morphology, built on the library convolution routine
# --------------------------------------------------------------------------
def _structuring_count(mask, size):
    """Number of set neighbours inside a ``size x size`` window, per pixel."""
    return convolution((np.asarray(mask) > 0).astype(np.float64), np.ones((size, size)))


def binary_erode(mask, size=3):
    """Erosion: a pixel survives only if the whole window is foreground."""
    return (_structuring_count(mask, size) >= size * size - 1e-6).astype(np.uint8) * 255


def binary_dilate(mask, size=3):
    """Dilation: a pixel is set if any pixel in the window is foreground."""
    return (_structuring_count(mask, size) > 1e-6).astype(np.uint8) * 255


def binary_open(mask, size=3, iterations=1):
    """Opening removes speckle smaller than the structuring element."""
    out = mask
    for _ in range(iterations):
        out = binary_dilate(binary_erode(out, size), size)
    return out


def binary_close(mask, size=3, iterations=1):
    """Closing fills pinholes and hairline cracks in the silhouette."""
    out = mask
    for _ in range(iterations):
        out = binary_erode(binary_dilate(out, size), size)
    return out


# --------------------------------------------------------------------------
# Connected-component labelling
# --------------------------------------------------------------------------
def _row_runs(row, width):
    """Start/stop indices of every maximal run of True in a boolean row."""
    changes = np.diff(row.astype(np.int8))
    starts = (np.flatnonzero(changes == 1) + 1).tolist()
    stops = (np.flatnonzero(changes == -1) + 1).tolist()
    if row[0]:
        starts.insert(0, 0)
    if row[-1]:
        stops.append(width)
    return list(zip(starts, stops))


def connected_components(mask, connectivity=8):
    """Label connected foreground regions, implemented from scratch.

    The classic textbook formulation floods every pixel individually.  This
    implementation instead performs the equivalent two-pass run-length
    algorithm: each row is decomposed into maximal runs of foreground pixels,
    runs that touch a run in the previous row are merged with a union-find
    structure, and a second pass renumbers the roots.  The result is identical
    to per-pixel labelling but the Python-level loop runs once per *run*
    instead of once per pixel, which is what allows a 1920x1080 puzzle
    photograph to be labelled in a fraction of a second.

    Parameters
    ----------
    mask : array_like
        Non-zero pixels are treated as foreground.
    connectivity : int
        ``4`` for edge neighbours only, ``8`` to also join diagonal contacts.

    Returns
    -------
    (labels, components)
        ``labels`` is an int32 image with ``0`` for background and ``1..N`` for
        the regions; ``components`` is a list of :class:`Component`, ordered by
        first appearance in raster order.
    """
    foreground = np.asarray(mask) > 0
    height, width = foreground.shape
    labels = np.zeros((height, width), np.int32)

    parent = [0]

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    slack = 1 if connectivity == 8 else 0
    records = []          # (row, start, stop, provisional label)
    previous_runs = []    # runs of the preceding row
    for y in range(height):
        row = foreground[y]
        if not row.any():
            previous_runs = []
            continue
        current_runs = []
        for start, stop in _row_runs(row, width):
            label = 0
            for prev_start, prev_stop, prev_label in previous_runs:
                if start <= prev_stop + slack - 1 and prev_start <= stop + slack - 1:
                    if label == 0:
                        label = find(prev_label)
                    else:
                        union(label, prev_label)
            if label == 0:
                label = len(parent)
                parent.append(label)
            records.append((y, start, stop, label))
            current_runs.append((start, stop, label))
        previous_runs = current_runs

    # Second pass: renumber roots in order of first appearance.
    renumber = {}
    for y, start, stop, provisional in records:
        root = find(provisional)
        if root not in renumber:
            renumber[root] = len(renumber) + 1

    count = len(renumber)
    areas = np.zeros(count + 1, np.int64)
    min_x = np.full(count + 1, width, np.int64)
    max_x = np.full(count + 1, -1, np.int64)
    min_y = np.full(count + 1, height, np.int64)
    max_y = np.full(count + 1, -1, np.int64)
    sum_x = np.zeros(count + 1, np.float64)
    sum_y = np.zeros(count + 1, np.float64)

    for y, start, stop, provisional in records:
        label = renumber[find(provisional)]
        labels[y, start:stop] = label
        length = stop - start
        areas[label] += length
        min_x[label] = min(min_x[label], start)
        max_x[label] = max(max_x[label], stop - 1)
        min_y[label] = min(min_y[label], y)
        max_y[label] = max(max_y[label], y)
        sum_x[label] += (start + stop - 1) * length / 2.0
        sum_y[label] += y * length

    components = []
    for label in range(1, count + 1):
        x0, y0 = int(min_x[label]), int(min_y[label])
        x1, y1 = int(max_x[label]) + 1, int(max_y[label]) + 1
        components.append(Component(
            label=label,
            area=int(areas[label]),
            bbox=(x0, y0, x1 - x0, y1 - y0),
            centroid=(float(sum_x[label] / areas[label]), float(sum_y[label] / areas[label])),
            mask=labels[y0:y1, x0:x1] == label,
        ))
    return labels, components


# --------------------------------------------------------------------------
# Foreground extraction
# --------------------------------------------------------------------------
def foreground_mask(image, min_area_ratio=0.0005, morph_size=5):
    """Separate pieces from the background with the library's Otsu threshold.

    Both polarities of the Otsu result are considered and the one whose image
    border is *least* occupied is kept, so the routine works both for pieces
    photographed on a dark mat and for pieces on a light floor.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    binary, _ = otsu_threshold(gray)
    candidates = [binary, 255 - binary]
    height, width = gray.shape
    binary = min(candidates, key=lambda z: np.mean(
        np.r_[z[0], z[-1], z[:, 0], z[:, -1]] > 0))

    binary = binary_open(binary, morph_size)
    binary = binary_close(binary, morph_size, iterations=2)

    labels, components = connected_components(binary)
    out = np.zeros_like(binary)
    for component in components:
        _, _, bw, bh = component.bbox
        aspect = bw / max(bh, 1)
        extent = component.area / max(bw * bh, 1)
        if (component.area >= height * width * min_area_ratio
                and 0.25 < aspect < 4 and extent > 0.15):
            out[labels == component.label] = 255
    return out


def extract_pieces(image, mask=None, min_area_ratio=0.0005):
    """Crop every labelled region to its padded bounding box."""
    binary = foreground_mask(image, min_area_ratio) if mask is None else mask
    labels, components = connected_components(binary)
    result = []
    for component in components:
        x, y, w, h = component.bbox
        pad = max(3, int(0.05 * max(w, h)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1 = min(image.shape[1], x + w + pad)
        y1 = min(image.shape[0], y + h + pad)
        full = (labels == component.label).astype(np.uint8) * 255
        result.append({
            "id": component.label,
            "image": image[y0:y1, x0:x1].copy(),
            "mask": full[y0:y1, x0:x1],
            "bbox": (x0, y0, x1 - x0, y1 - y0),
            "area": component.area,
        })
    return result, binary, labels


def normalize_orientation(image, mask):
    """Rotate a piece so its minimum-area bounding rectangle is axis aligned."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return image, mask, 0.0
    points = np.column_stack([xs, ys]).astype(np.float32)
    angle = cv2.minAreaRect(points)[-1]
    if angle > 45:
        angle -= 90
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(np.ceil(h * sin + w * cos))
    new_h = int(np.ceil(h * cos + w * sin))
    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]
    size = (new_w, new_h)
    return (cv2.warpAffine(image, matrix, size),
            cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST),
            float(angle))


def fill_holes(binary):
    """Fill background regions fully enclosed by foreground.

    A puzzle piece is simply connected: its blanks are indentations of the outer
    boundary, never enclosed holes.  Any enclosed background region is therefore
    an artefact -- dark printed artwork that fell below the threshold, or a
    specular dropout -- and belongs to the piece.  Enclosed regions are found by
    labelling the background and discarding every region that reaches the image
    border.
    """
    binary = np.asarray(binary)
    background = (binary == 0).astype(np.uint8) * 255
    labels, components = connected_components(background, connectivity=4)
    if not components:
        return (binary > 0).astype(np.uint8) * 255

    height, width = binary.shape
    border = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    border = set(int(v) for v in border if v)
    out = (binary > 0).astype(np.uint8) * 255
    for component in components:
        if component.label not in border:
            out[labels == component.label] = 255
    return out


# --------------------------------------------------------------------------
# Separation of touching pieces
# --------------------------------------------------------------------------
def _shift(array, dy, dx):
    """Translate an array by (dy, dx), filling the exposed border with zeros."""
    out = np.zeros_like(array)
    ys = slice(max(dy, 0), array.shape[0] + min(dy, 0))
    xs = slice(max(dx, 0), array.shape[1] + min(dx, 0))
    yd = slice(max(-dy, 0), array.shape[0] + min(-dy, 0))
    xd = slice(max(-dx, 0), array.shape[1] + min(-dx, 0))
    out[ys, xs] = array[yd, xd]
    return out


def propagate_labels(seeds, domain):
    """Grow labelled seeds over ``domain`` until they meet (a discrete watershed).

    Every iteration lets each labelled pixel offer its label to its four
    neighbours.  A pixel is claimed only when all the offers it receives agree,
    so pixels equidistant from two different seeds are never assigned and are
    left as a one-pixel separating ridge.  This is the skeleton-by-influence-zone
    of the seeds, and it is what physically separates two jigsaw pieces that
    touch at a single tab.
    """
    labels = np.asarray(seeds, np.int32).copy()
    domain = np.asarray(domain) > 0
    sentinel = np.iinfo(np.int32).max
    while True:
        unassigned = domain & (labels == 0)
        if not unassigned.any():
            break
        offers = np.stack([_shift(labels, dy, dx)
                           for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))])
        present = offers > 0
        highest = offers.max(axis=0)
        lowest = np.where(present, offers, sentinel).min(axis=0)
        claimable = unassigned & (highest > 0) & (highest == lowest)
        if not claimable.any():
            break
        labels[claimable] = highest[claimable]
    return labels


def _seed_by_erosion(region, parts, max_iterations=80, min_fraction=0.15):
    """Erode a region until it breaks into ``parts`` pieces; return those seeds.

    A split is only accepted once *every* one of the ``parts`` largest surviving
    components is a substantial fraction of an expected single piece.  Without
    that condition a few stray pixels shaved off a tab would be mistaken for the
    second piece and the blob would not actually be divided.
    """
    region = np.asarray(region) > 0
    minimum = max(20.0, min_fraction * region.sum() / parts)
    current = region.astype(np.uint8) * 255
    for _ in range(max_iterations):
        current = binary_erode(current, 3)
        if not current.any():
            break
        labels, components = connected_components(current)
        ranked = sorted(components, key=lambda c: -c.area)[:parts]
        if len(ranked) >= parts and ranked[-1].area >= minimum:
            seeds = np.zeros(region.shape, np.int32)
            for index, component in enumerate(ranked, start=1):
                seeds[labels == component.label] = index
            return seeds
    return None


def components_from_labels(labels):
    """Rebuild :class:`Component` records from an existing label image."""
    labels = np.asarray(labels, np.int32)
    components = []
    for label in range(1, int(labels.max()) + 1):
        region = labels == label
        area = int(region.sum())
        if not area:
            continue
        ys, xs = np.nonzero(region)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        components.append(Component(
            label=label,
            area=area,
            bbox=(x0, y0, x1 - x0, y1 - y0),
            centroid=(float(xs.mean()), float(ys.mean())),
            mask=region[y0:y1, x0:x1],
        ))
    return components


def separate_touching(binary, expected_area=None, tolerance=1.45):
    """Split blobs whose area indicates that several pieces are touching.

    The number of pieces inside a blob is estimated from its area relative to
    the median single-piece area.  That estimate seeds an erosion-based
    separation followed by :func:`propagate_labels`.

    The result is returned as a *label image* rather than a binary mask: the
    watershed ridge between two touching pieces is only one pixel wide, and
    eight-connected labelling would immediately bridge it again diagonally.
    Carrying the labels through avoids relying on the ridge at all.

    Returns
    -------
    (labels, components)
        Same contract as :func:`connected_components`.
    """
    labels, components = connected_components(binary)
    if not components:
        return labels, components
    if expected_area is None:
        areas = sorted(c.area for c in components if c.area > 0)
        significant = [a for a in areas if a >= 0.25 * areas[-1]]
        expected_area = float(np.median(significant)) if significant else float(areas[-1])

    out = np.zeros(labels.shape, np.int32)
    next_label = 0
    for component in components:
        region = labels == component.label
        parts = int(round(component.area / expected_area))
        if parts < 2 or component.area < tolerance * expected_area:
            next_label += 1
            out[region] = next_label
            continue
        x, y, w, h = component.bbox
        crop = region[y:y + h, x:x + w]
        seeds = _seed_by_erosion(crop, parts)
        if seeds is None:
            next_label += 1
            out[region] = next_label
            continue
        grown = propagate_labels(seeds, crop)
        window = out[y:y + h, x:x + w]
        for part in range(1, parts + 1):
            claimed = grown == part
            if claimed.any():
                next_label += 1
                window[claimed] = next_label
        # Pixels on the ridge stay unassigned; hand each to its nearest label so
        # no piece silhouette loses a boundary pixel.
        leftover = crop & (window == 0)
        if leftover.any():
            filled = propagate_labels(window, crop)
            window[leftover] = filled[leftover]
    return out, components_from_labels(out)

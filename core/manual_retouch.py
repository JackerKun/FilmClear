"""Render a manual stroke as a sparse patch in original image coordinates."""

import cv2
import numpy as np

from .dust_remover import DustRemover
from .grain_repair import restore_grain
from .linear_defects import detect_strokes


def _stroke_distance(shape, points):
    centerline = np.ones(shape, np.uint8)
    cv2.polylines(centerline, [points.astype(np.int32)], False, 0, 1)
    centerline[points[0, 1], points[0, 0]] = 0
    return cv2.distanceTransform(centerline, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def _guided_fiber(image, points, radius):
    """Match a dragged brush to a nearby narrow ridge before restoring it."""
    height, width = image.shape[:2]
    if radius > 32 or np.ptp(points, axis=0).max() < 16:
        return None
    margin = max(14, 2 * radius)
    left, top = map(int, np.maximum(points.min(axis=0) - margin, 0))
    right, bottom = map(int, np.minimum(points.max(axis=0) + margin + 1, (width, height)))
    shape = (bottom - top, right - left)
    if shape[0] * shape[1] > 160_000:
        return None
    candidates = detect_strokes(image, region=(left, top, right, bottom))
    if not candidates:
        return None
    distance = _stroke_distance(shape, points - (left, top))
    brush = distance <= radius + 0.5
    nearby = distance <= max(radius + 3, radius * 1.5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    matched = None
    for item in candidates:
        x, y, w, h = item.bbox
        if item.confidence < 0.64 or min(x - left, y - top, right - x - w, bottom - y - h) < 6:
            continue
        contour = np.zeros(shape, np.uint8)
        cv2.drawContours(contour, [item.contour, *item.holes], -1, 1, cv2.FILLED, offset=(-left, -top))
        inside = contour != 0
        if np.mean(brush[inside]) < 0.55 or np.mean(nearby[inside]) < 0.80:
            continue
        if np.mean(cv2.dilate(contour, kernel)[distance == 0]) < 0.65:
            continue
        if matched is not None:
            return None  # An ambiguous stroke stays under direct brush control.
        matched = contour
    if matched is None:
        return None

    from skimage.restoration import inpaint_biharmonic

    # Replace the fiber and its soft halo fully. Blend only into the clean rim.
    distance = cv2.distanceTransform(1 - matched, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    mask = np.uint8(distance <= 5) * 255
    # Give grain sampling room beyond the small detection window.
    outer_x, outer_y = max(0, left - 96), max(0, top - 96)
    pixels = image[outer_y:bottom + 96, outer_x:right + 96]
    region = np.s_[top - outer_y:bottom - outer_y, left - outer_x:right - outer_x]
    repaired = pixels.copy()
    maximum = np.iinfo(image.dtype).max
    repaired[region] = np.rint(inpaint_biharmonic(pixels[region], mask != 0, channel_axis=-1) * maximum)
    grain_mask = np.zeros(pixels.shape[:2], np.uint8)
    grain_mask[region] = mask
    restore_grain(pixels, repaired, grain_mask)
    y, x = np.nonzero(mask)
    strength = np.clip((5.5 - distance[y, x]) / 2.5, 0, 1)
    return _sparse_patch(image, y + top, x + left, repaired[region][y, x], strength)


def retouch_patch(image, points, radius, hardness, *, mode, opacity=1.0,
                  source=None, repair_params=None):
    height, width = image.shape[:2]
    points = np.clip(np.rint(points).astype(np.int32), (0, 0), (width - 1, height - 1))
    radius = max(1, int(radius))
    if mode == "spot" and repair_params.get("method") in ("auto", "grain"):
        guided = _guided_fiber(image, points, radius)
        if guided is not None:
            return guided
    margin = radius + 1
    x1, y1 = np.maximum(points.min(axis=0) - margin, 0)
    x2, y2 = np.minimum(points.max(axis=0) + margin + 1, (width, height))
    distance = _stroke_distance((y2 - y1, x2 - x1), points - (x1, y1))
    rows, cols = np.nonzero(distance < radius + 0.5)
    feather = max(1.0, radius * (1.0 - hardness))
    alpha = np.minimum(1.0, (radius + 0.5 - distance[rows, cols]) / feather)
    y, x = rows + y1, cols + x1

    if mode == "clone":
        dx, dy = np.rint(source).astype(np.int32) - points[0]
        sy, sx = y + dy, x + dx
        valid = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
        y, x, alpha = y[valid], x[valid], alpha[valid]
        target = image[sy[valid], sx[valid]]
    elif mode == "spot":
        remover = DustRemover(**repair_params)
        halo = max(96, 4 * remover.inpaint_radius)
        left, top = max(0, x1 - halo), max(0, y1 - halo)
        pixels = image[top:min(height, y2 + halo), left:min(width, x2 + halo)]
        mask = np.zeros(pixels.shape[:2], np.uint8)
        mask[y - top, x - left] = 255
        if len(x) == mask.size:
            return None, np.empty(0, dtype=np.intp)
        result = remover.apply_mask(pixels, mask)
        target = result[y - top, x - left]
    else:
        raise ValueError(f"Unsupported retouch mode: {mode}")

    return _sparse_patch(image, y, x, target, alpha * opacity)


def _sparse_patch(image, y, x, target, strength):
    maximum = np.iinfo(image.dtype).max
    before = image[y, x]
    after = np.rint(before + (np.asarray(target, dtype=np.float32) - before) * strength[:, None])
    after = np.clip(after, 0, maximum).astype(image.dtype)
    changed = np.any(before != after, axis=1)
    positions = y * image.shape[1] + x
    if not np.any(changed):
        return None, np.empty(0, dtype=np.intp)
    patch = (positions[changed], before[changed], after[changed])
    return patch, positions[strength >= 0.95]


def covered_candidates(items, positions, changed, width):
    """Recognize candidates covered by the nearly opaque part of a stroke."""
    if not len(positions):
        return set()
    resolved = set()
    for item in items:
        center = item.center_y * width + item.center_x
        index = np.searchsorted(positions, center)
        if index == len(positions) or positions[index] != center:
            continue
        if item.area > len(positions):
            continue
        x, y, w, h = item.bbox
        if item.contour is None:
            rows, cols = np.indices((h, w))
            rows, cols = rows.ravel(), cols.ravel()
        else:
            shape = np.zeros((h, w), np.uint8)
            cv2.drawContours(shape, [item.contour, *item.holes], -1, 1, cv2.FILLED, offset=(-x, -y))
            rows, cols = np.nonzero(shape)
        footprint = (rows + y) * width + cols + x
        indices = np.minimum(np.searchsorted(positions, footprint), len(positions) - 1)
        if not footprint.size or not np.array_equal(positions[indices], footprint):
            continue
        touched = np.minimum(np.searchsorted(changed, footprint), len(changed) - 1)
        if np.any(changed[touched] == footprint):
            resolved.add(item.id)
    return resolved

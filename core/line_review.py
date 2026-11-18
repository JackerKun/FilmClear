"""Original-pixel context for deciding whether a traced line needs review."""

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.morphology import skeletonize


def review_scene_lines(image, items, *, filter_structures=False, keep_filtered=False, peers=()):
    lines = [item for item in items
             if item.dust_type in ("scratch", "fiber") and item.contour is not None]
    paired = _paired_scene_lines([*lines, *peers], image.shape) if filter_structures else set()
    kept = []
    for item in items:
        if item.dust_type not in ("scratch", "fiber") or item.contour is None:
            kept.append(item)
            continue
        # Whole-photo RGB scans leave line-like features around the eyes and
        # mouth to deliberate local review. Keep the spot review unchanged.
        if filter_structures and (item.face_protected or id(item) in paired):
            if keep_filtered:
                item.scene_line = True
                item.selected = False
                kept.append(item)
            continue
        evidence = _scene_line_evidence(image, item)
        if filter_structures and not keep_filtered and evidence in ("junction", "parallel", "edge"):
            continue
        if evidence:
            item.scene_line = True
            item.selected = False
        kept.append(item)
    return kept


def _paired_scene_lines(items, shape):
    """Find repeated straight structures with matching endpoints and widths."""
    size = np.asarray(shape[:2][::-1], dtype=float)
    lines = {0: [], 1: []}
    for item in items:
        points = item.contour.reshape(-1, 2).astype(np.float32)
        if len(points) < 3:
            continue
        dx, dy, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        axis = int(abs(dy) > abs(dx))
        if min(abs(dx), abs(dy)) > 0.17:
            continue
        start, end = points[:, axis].min(), points[:, axis].max()
        length = end - start
        width = item.area / max(1, length)
        if (length < max(36, 7 * width) or length > 0.32 * size[axis]
                or start < 0.03 * size[axis] or end > 0.97 * size[axis]):
            continue
        axis_width = np.ptp(np.percentile(points @ np.array((-dy, dx)), (5, 95)))
        if axis_width > max(5, 3.4 * width):
            continue
        lines[axis].append((start, end, item.center[1 - axis], width, id(item)))

    repeated = set()
    minimum_error = max(5, 0.0035 * min(size))
    for axis, parallel in lines.items():
        parallel.sort()
        for index, (start, end, position, width, item_id) in enumerate(parallel):
            tolerance = max(minimum_error, 0.07 * (end - start))
            for other_start, other_end, other_position, other_width, other_id in parallel[index + 1:]:
                if other_start - start > tolerance:
                    break
                separation = abs(other_position - position)
                if (abs(other_end - end) <= tolerance
                        and 4 * max(width, other_width) < separation < 0.35 * size[1 - axis]
                        and 0.58 <= width / other_width <= 1.72):
                    repeated.update((item_id, other_id))
    return repeated


def _scene_line_evidence(image, item):
    x, y, w, h = item.bbox
    # Bound the analysis of very long, curved traces. All image profiles
    # below are still sampled from the source scan.
    scale = max(1.0, np.sqrt(w * h / 800_000))
    mask = np.zeros((int(np.ceil(h / scale)), int(np.ceil(w / scale))), np.uint8)
    contours = [np.rint((part - (x, y)) / scale).astype(np.int32)
                for part in (item.contour, *item.holes)]
    cv2.drawContours(mask, contours, -1, 1, cv2.FILLED)
    padded = np.pad(mask, 1)
    skeleton = skeletonize(padded != 0)
    sy, sx = np.nonzero(skeleton)
    if len(sx) < 12:
        return None

    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
    radius = max(1.7, float(np.median(distance[sy, sx])) * scale)
    neighbors = np.stack([skeleton[sy + dy, sx + dx] for dx, dy in (
        (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)
    )])
    branches = (neighbors & ~np.roll(neighbors, 1, axis=0)).sum(axis=0) >= 3
    points = np.column_stack((sx - 1, sy - 1)).astype(np.float32)
    if _has_junction(mask, points[branches], max(8, 2.8 * radius / scale)):
        return "junction"

    endpoints = neighbors.sum(axis=0) == 1
    halo = int(np.ceil(max(68, 12 * radius)))
    left, top = max(0, x - halo), max(0, y - halo)
    patch = image[top:y + h + halo, left:x + w + halo]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    if gray.dtype == np.uint16:
        gray = cv2.convertScaleAbs(gray, alpha=255 / 65535)
    gray = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0.9)
    points = (points * scale + (x - left, y - top)).astype(np.float32)
    if _follows_scene_edge(gray, points, radius):
        return "edge"
    if not endpoints.any():
        return None
    dx, dy, cx, cy = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    axis = np.array((dx, dy))
    center = np.array((cx, cy))
    along = (points - center) @ axis
    across = (points - center) @ (-dy, dx)
    # A small spur can send the tangent at a nearly straight line's end
    # off in the wrong direction. Fit its whole path in this case.
    span = np.ptp(along)
    if span >= 9 * radius and np.percentile(np.abs(across), 90) <= max(2.0 * radius, 0.022 * span):
        start, end = float(along.min()), float(along.max())
        if any(_continues_past_end(gray, center + edge * axis, direction, radius)
               for edge, direction in ((start, -axis), (end, axis))):
            return "continuation"
        if _has_parallel_lines(gray, center, axis, start, end, radius):
            return "parallel"
    for tip in points[endpoints]:
        nearby = points[np.linalg.norm(points - tip, axis=1) <= max(11, 4 * radius)]
        if len(nearby) < 5:
            continue
        direction = cv2.fitLine(nearby, cv2.DIST_L2, 0, 0.01, 0.01).ravel()[:2]
        if np.dot(tip - nearby.mean(axis=0), direction) < 0:
            direction = -direction
        if _continues_past_end(gray, tip, direction, radius):
            return "continuation"
    return None


def _follows_scene_edge(gray, points, radius):
    """Look for a persistent step in the scene on both scales of a trace."""
    profiles = []
    offsets = radius * np.array((-7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7))
    sample_indices = np.linspace(0, len(points) - 1, min(32, len(points)), dtype=int)
    for point in points[sample_indices]:
        nearby = points[np.linalg.norm(points - point, axis=1) <= max(9, 4 * radius)]
        if len(nearby) < 4:
            continue
        dx, dy, _, _ = cv2.fitLine(nearby, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        coordinates = point + offsets[:, None] * (-dy, dx)
        if np.all((coordinates >= 0) & (coordinates < np.asarray(gray.shape[::-1]) - 1)):
            profiles.append(coordinates)
    if len(profiles) < 12:
        return False
    values = map_coordinates(gray, np.asarray(profiles)[..., ::-1].transpose(2, 0, 1),
                             order=1, mode="nearest")
    near_a, near_b = values[:, 2:4].mean(axis=1), values[:, 7:9].mean(axis=1)
    far_a, far_b = values[:, :2].mean(axis=1), values[:, -2:].mean(axis=1)
    peak = values[:, 4:7]
    ridge = np.maximum(peak.max(axis=1) - np.maximum(near_a, near_b),
                       np.minimum(near_a, near_b) - peak.min(axis=1))
    step = np.abs(near_a - near_b)
    along_edge = ((step >= np.maximum(10, ridge)) & (np.abs(far_a - far_b) >= 9)
                  & ((near_a - near_b) * (far_a - far_b) > 0))
    return along_edge.mean() >= 0.75


def _has_junction(mask, junctions, radius):
    """Require three sustained arms beyond the short spurs on a true fiber."""
    if not len(junctions):
        return False
    angles = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    rings = radius * np.array((1, 1.75, 2.5))[:, None, None] * directions
    for point in junctions:
        x, y = np.moveaxis(np.rint(point + rings).astype(np.intp), -1, 0)
        valid = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        hits = np.zeros(valid.shape, bool)
        hits[valid] = mask[y[valid], x[valid]]
        # Require angular support on neighboring samples, too.
        hits &= np.roll(hits, 1, axis=1) | np.roll(hits, -1, axis=1)
        if np.all(np.count_nonzero(hits & ~np.roll(hits, 1, axis=1), axis=1) >= 3):
            return True
    return False


def _ridge_profiles(gray, positions, direction, radius):
    normal = np.array((-direction[1], direction[0]))
    offsets = radius * np.array((-2.9, -1.9, -0.45, 0, 0.45, 1.9, 2.9))
    coordinates = positions[:, None] + offsets[None, :, None] * normal
    samples = map_coordinates(gray, coordinates[..., ::-1].transpose(2, 0, 1),
                              order=1, mode="nearest")
    first = samples[:, :2].mean(axis=1)
    last = samples[:, -2:].mean(axis=1)
    core = samples[:, 2:5]
    return (np.column_stack((core.max(axis=1) - np.maximum(first, last),
                             np.minimum(first, last) - core.min(axis=1))),
            (first + last) / 2)


def _continues_past_end(gray, tip, direction, radius):
    inside = tip - np.linspace(2, max(9, 3.4 * radius), 8)[:, None] * direction
    inside_profiles, inside_background = _ridge_profiles(gray, inside, direction, radius)
    strength = np.median(inside_profiles, axis=0)
    polarity = int(np.argmax(strength))
    if strength[polarity] < 5:
        return False
    outside = tip + np.linspace(max(5, 1.6 * radius), max(42, 9 * radius), 20)[:, None] * direction
    if (np.min(outside) < 3 * radius
            or np.any(outside.max(axis=0) >= np.array(gray.shape[::-1]) - 3 * radius)):
        return False
    outside_profiles, outside_background = _ridge_profiles(gray, outside, direction, radius)
    support = outside_profiles >= max(4.5, 0.42 * strength[polarity])
    # Check the near and far stretches separately, so an unrelated ridge
    # farther away does not count as a continuation.
    near, far = support[:10].mean(axis=0), support[10:].mean(axis=0)
    if near[polarity] >= 0.7 and far[polarity] >= 0.6:
        return True
    # A wire can switch from dark to bright as the scene behind it changes.
    shift = abs(np.median(inside_background) - np.median(outside_background))
    return shift >= max(14, strength[polarity]) and near[1 - polarity] >= 0.7 and far[1 - polarity] >= 0.8


def _has_parallel_lines(gray, center, direction, start, end, radius):
    span = end - start
    if span < max(36, 10 * radius):
        return False
    along = np.linspace(start + 0.12 * span, end - 0.12 * span, 24)
    positions = center + along[:, None] * direction
    profiles, _ = _ridge_profiles(gray, positions, direction, radius)
    strength = np.maximum(profiles[:, 0], profiles[:, 1])
    minimum = max(5.0, 0.65 * float(np.median(strength)))
    normal = np.array((-direction[1], direction[0]))
    offsets = np.arange(max(10, 4 * radius), max(52, 8 * radius), max(2, 0.5 * radius))
    offsets = np.concatenate((-offsets[::-1], offsets))
    neighbors = positions[None] + offsets[:, None, None] * normal
    valid = ((neighbors >= 3 * radius) & (neighbors < np.array(gray.shape[::-1]) - 3 * radius)).all(axis=2)
    profiles, _ = _ridge_profiles(gray, neighbors.reshape(-1, 2), direction, radius)
    responses = profiles.reshape(len(offsets), len(positions), 2)
    coverage = ((responses >= minimum) & valid[..., None]).mean(axis=1)
    matched = offsets[np.any(coverage >= 0.72, axis=1)]
    # Count distinct neighbors, avoiding multiple samples of one wide line.
    return len(matched) >= 2 and np.ptp(matched) >= max(12, 4 * radius)

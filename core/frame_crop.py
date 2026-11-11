"""Detect a scanned film window and render reversible, original-pixel crops."""

from math import cos, floor, radians, sin

import cv2
import numpy as np

from .scan_io import display_pixels


def _edge_line(plane):
    height, width = plane.shape
    inset = width // 12
    strip = cv2.GaussianBlur(plane[:max(12, round(height * 0.18))].astype(np.float32), (3, 3), 0)
    gap = 4
    hits = ((strip[:-gap] < 28) & (strip[gap:] > 32)
            & (strip[gap:] - strip[:-gap] > 16)).astype(np.uint8)
    hits[:, :inset] = hits[:, width - inset:] = 0
    lines = cv2.HoughLinesP(hits * 255, 1, np.pi / 720, max(18, width // 7),
                            minLineLength=max(20, width // 5), maxLineGap=max(6, width // 18))
    if lines is None:
        return None
    columns = np.arange(inset, width - inset)
    best = None
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(x2 - x1) < width * 0.2 or abs(y2 - y1) > abs(x2 - x1) * 0.12:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        rows = np.rint(columns * slope + intercept).astype(int)
        valid = (rows >= 6) & (rows < len(hits) - 5)
        xs, ys = columns[valid], rows[valid]
        if not len(xs):
            continue
        support = np.any(np.array([hits[ys + offset, xs] for offset in range(-3, 4)]), axis=0)
        coverage = support.sum() / len(columns)
        dark = np.median(np.array([strip[np.rint(ys * fraction).astype(int), xs]
                                   for fraction in (0.25, 0.55, 0.8)]), axis=0)
        if coverage < 0.30 or np.mean(dark < 25) < 0.68:
            continue
        xs, ys = xs[support], ys[support]
        offsets = np.arange(-3, 4)[:, None]
        contrast = strip[ys + offsets + gap, xs] - strip[ys + offsets, xs]
        peaks = np.argmax(contrast, axis=0)
        targets = ys + offsets[:, 0][peaks] + gap / 2
        keep = np.ones(len(xs), bool)
        for _ in range(3):
            fitted = np.polyfit(xs[keep], targets[keep], 1)
            residual = np.abs(targets - np.polyval(fitted, xs))
            keep = residual <= max(1.5, 3 * np.median(residual))
        score = coverage - float(np.median(residual)) * 0.03
        if best is None or score > best[0]:
            best = score, fitted, coverage
    return None if best is None else (best[1], best[2])


def _aligned_edge(plane, slope):
    """Look for a weaker edge at the angle supported by the other sides."""
    height, width = plane.shape
    strip = cv2.GaussianBlur(plane[:round(height * 0.15)].astype(np.float32), (3, 3), 0)
    columns = np.arange(width // 12, width - width // 12, 2)
    depths = np.arange(6, len(strip) - 4)
    rows = np.rint(depths[:, None] + slope * (columns - width / 2)).astype(int)
    valid = (rows >= 5) & (rows < len(strip) - 3)
    rows = np.clip(rows, 5, len(strip) - 4)
    outside, inside = strip[rows - 3, columns], strip[rows + 3, columns]
    dark = strip[np.rint(rows * 0.55).astype(int), columns]
    support = valid & (outside < 24) & (inside > 22) & (inside - outside > 11)
    coverage = support.mean(axis=1)
    score = np.where((dark < 24).mean(axis=1) >= 0.78, coverage - depths * 0.0008, -1)
    index = int(np.argmax(score))
    if score[index] < 0 or coverage[index] < 0.35:
        return None
    return slope, depths[index] - slope * width / 2, float(coverage[index])


def _thin_edge(plane):
    """Find a narrow matte without downsampling across the border itself."""
    depth, length = plane.shape[:2]
    limit = round(depth * 0.018)
    if limit < 3:
        return None
    gap = min(6, max(1, round(depth * 0.0015)))
    strip = display_pixels(np.ascontiguousarray(plane[:limit + gap + 1]))
    if strip.ndim == 3:
        strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    width = min(900, length)
    # Average along the edge to suppress grain, keeping every row across it.
    if width < length:
        strip = cv2.resize(strip, (width, len(strip)), interpolation=cv2.INTER_AREA)
    strip = cv2.blur(strip.astype(np.float32), (5, 1))
    sums = np.vstack((np.zeros((1, width), np.float32), np.cumsum(strip, axis=0)))
    rows = np.arange(2, limit + 1)
    start = np.maximum(0, rows - gap)
    before = (sums[rows] - sums[start]) / (rows - start)[:, None]
    after = (sums[rows + gap] - sums[rows]) / gap
    matte = sums[rows] / rows[:, None]
    contrast = after - before
    dark = (matte <= 24) & (before <= 34)
    strong = dark & (after >= 24) & (contrast >= 12)
    if np.count_nonzero(strong.any(axis=0)) < width * 0.14:
        return None
    weak = dark & (after >= 11) & (contrast >= 5)
    lines = cv2.HoughLinesP(np.uint8(strong) * 255, 1, np.pi / 720, max(10, width // 28),
                            minLineLength=max(9, width // 20), maxLineGap=max(4, width // 24))
    if lines is None:
        return None

    columns = np.arange(width)
    window = np.arange(-max(1, gap // 2), max(1, gap // 2) + 1)[:, None]

    def support(slope, intercept):
        prediction = np.rint(slope * columns + intercept).astype(int)
        depths = prediction + window
        valid = (depths >= rows[0]) & (depths <= rows[-1])
        indices = np.clip(depths - rows[0], 0, len(rows) - 1)
        reliable = valid & strong[indices, columns]
        gradient = np.where(reliable, contrast[indices, columns], -1)
        peaks = gradient.argmax(axis=0)
        matched = gradient[peaks, columns] >= 0
        return matched, depths[peaks, columns], (valid & weak[indices, columns]).any(axis=0)

    best = None
    for x1, y1, x2, y2 in lines[:, 0]:
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 + rows[0] - slope * x1
        ends = intercept + slope * np.array((0, width - 1))
        if min(ends) < 1 or max(ends) > limit + 1:
            continue
        matched, targets, softer = support(slope, intercept)
        coverage = matched.mean()
        if coverage < 0.14 or softer.mean() < 0.28:
            continue
        xs = columns[matched]
        span = (xs[-1] - xs[0]) / width
        # A local bright object at the edge must not establish a crop line.
        if span < 0.48 and coverage < 0.52:
            continue
        fitted = np.polyfit(xs, targets[matched], 1)
        sampled = np.clip(np.rint(np.polyval(fitted, columns)).astype(int) - rows[0], 0, len(rows) - 1)
        if np.mean(matte[sampled, columns] <= 24) < 0.88:
            continue
        score = coverage + 0.2 * softer.mean() + 0.1 * span
        if best is None or score > best[0]:
            best = score, fitted, float(coverage)
    if best is None:
        return None
    _, (slope, intercept), coverage = best
    ends = intercept + slope * np.array((0, width - 1))
    if min(ends) < -1 or max(ends) > limit + 1:
        return None
    # Recover the coordinate of the first original column after area averaging.
    step = length / width
    return slope / step, intercept - slope * (step - 1) / (2 * step), coverage


def _sprocket_boundary(plane):
    """Fit the inside tips of a row of black holes, including cut-off holes."""
    height, width = plane.shape
    strip = plane[:round(height * 0.17)]
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(strip <= 1))
    holes = [(x + w / 2, y + h, w) for x, y, w, h, area in stats[1:]
             if 0.018 * width <= w <= 0.085 * width and h >= 6
             and y + h < len(strip) - 2 and area >= 0.78 * w * h]
    for _, depth, hole_width in holes:
        row = sorted(hole for hole in holes if abs(hole[1] - depth) <= hole_width * 0.28
                     and abs(hole[2] - hole_width) <= hole_width * 0.18)
        if len(row) < 4 or row[-1][0] - row[0][0] < 0.26 * width:
            continue
        x, y, _ = np.asarray(row).T
        gaps = np.diff(x)
        for pitch in gaps[(gaps > 1.65 * hole_width) & (gaps < 3.6 * hole_width)]:
            if np.count_nonzero(abs(gaps - pitch) <= 0.08 * pitch) < 3:
                continue
            slope, intercept = np.polyfit(x, y, 1)
            if abs(slope) < 0.06 and np.median(np.abs(y - (x * slope + intercept))) < 2.5:
                return slope, intercept
    return None


def _after_sprockets(line, holes, width):
    if holes is None:
        return True
    offsets = (line[0] - holes[0]) * np.array((0.15, 0.85)) * width + line[1] - holes[1]
    return offsets.min() >= -2 and offsets.max() <= 0.10 * width


_SIDES = ((1, False), (0, True), (1, True), (0, False))


def _edge_coefficients(shape, axis, reverse, slope, intercept):
    if reverse:
        slope, intercept = -slope, shape[1 - axis] - 1 - intercept
    return np.array((-slope, 1, -intercept) if axis == 1 else (1, -slope, -intercept))


def _frame_lines(gray, defaults):
    """Estimate the frame from multiple broad edges or opposing sprocket rows."""
    edges, scores, sprockets = [], [], []
    for axis, reverse in _SIDES:
        plane = gray if axis == 1 else gray.T
        plane = plane[::-1] if reverse else plane
        sprockets.append(_sprocket_boundary(plane))
        result = _edge_line(plane)
        slope, intercept = result[0] if result is not None else (0.0, 0.0)
        if result is not None and not _after_sprockets(result[0], sprockets[-1], plane.shape[1]):
            result = None
        edges.append(_edge_coefficients(gray.shape, axis, reverse, slope, intercept))
        scores.append(result[1] if result is not None else 0.0)
    confirmed_film = any(sprockets[i] is not None and sprockets[i + 2] is not None for i in (0, 1))
    if not any(scores) and not confirmed_film:
        return None
    # Scanner edges share one orientation. Scene roofs and shadows often do not.
    angles = np.arctan((-edges[0][0], edges[1][1], -edges[2][0], edges[3][1]))
    votes = np.asarray(scores)
    agreement = np.abs(angles[:, None] - angles[None, :]) < 0.018
    direction = int(np.argmax(agreement @ votes))
    for index, (axis, reverse) in enumerate(_SIDES):
        if scores[index] and (agreement[direction, index]
                              or (sprockets[index] is not None and scores[index] >= 0.65)):
            continue
        plane = gray if axis == 1 else gray.T
        slope = np.tan(angles[direction]) * (1 if axis == 1 else -1) * (-1 if reverse else 1)
        weak = _aligned_edge(plane[::-1] if reverse else plane, slope)
        holes = sprockets[index]
        if weak is None or not _after_sprockets(weak, holes, plane.shape[1]):
            scores[index] = 0.0
            edges[index] = (_edge_coefficients(gray.shape, axis, reverse, holes[0], holes[1] + 2)
                            if confirmed_film and holes is not None else np.array(defaults[index], float))
        else:
            edges[index] = _edge_coefficients(gray.shape, axis, reverse, *weak[:2])
            scores[index] = weak[2]
    count = sum(score > 0 for score in scores)
    if not confirmed_film and (count < 2 or sum(scores) < (1.3 if count == 2 else 1.6)):
        return None
    return edges, scores, confirmed_film


def detect_frame(image):
    """Return clockwise exposure corners, or None when evidence is insufficient."""
    height, width = image.shape[:2]
    if min(height, width) < 120:
        return None
    scale = min(1.0, 1400 / max(height, width))
    size = round(width * scale), round(height * scale)
    if min(size) < 120:
        return None
    thumb = display_pixels(cv2.resize(image, size, interpolation=cv2.INTER_AREA) if scale < 1 else image)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY) if thumb.ndim == 3 else thumb
    h, w = gray.shape
    defaults = np.array(((0, 1, 0), (1, 0, -(w - 1)), (0, 1, -(h - 1)), (1, 0, 0)), dtype=float)
    frame = _frame_lines(gray, defaults)
    edges, scores, confirmed_film = frame if frame is not None else (list(defaults), [0.0] * 4, False)
    native_scale = (np.array((height, width)) - 1) / (np.array((h, w)) - 1)
    for index, (axis, reverse) in enumerate(_SIDES):
        # Perforated scans already use a different physical reference. A strip
        # at their outside edge may be the film base, not the exposure window.
        if confirmed_film or scores[index]:
            continue
        plane = image if axis == 1 else image.swapaxes(0, 1)
        narrow = _thin_edge(plane[::-1] if reverse else plane)
        if narrow is None:
            continue
        slope, intercept, coverage = narrow
        depth_scale = native_scale[1 - axis]
        edges[index] = _edge_coefficients(gray.shape, axis, reverse,
                                          slope * native_scale[axis] / depth_scale, intercept / depth_scale)
        scores[index] = coverage
    count = sum(score > 0 for score in scores)
    if frame is None and not count:
        return None
    corners = []
    for first, second in ((3, 0), (0, 1), (1, 2), (2, 3)):
        a, b, c = edges[first]
        d, e, f = edges[second]
        determinant = a * e - b * d
        if abs(determinant) < 1e-6:
            return None
        corners.append(((b * f - c * e) / determinant, (c * d - a * f) / determinant))
    corners = np.asarray(corners, np.float32)
    if (np.any(corners < -2) or np.any(corners > np.array((w - 1, h - 1)) + 2)
            or cv2.contourArea(corners) < 0.50 * w * h):
        return None
    corners = np.clip(corners, (0, 0), (w - 1, h - 1))
    corners *= native_scale[::-1]
    return {"corners": corners.tolist(),
            "confidence": float(np.mean(scores)), "detected_edges": int(count)}


# Format presets constrain the selection; the source pixels retain their proportions.
FILM_FORMATS = (
    ("free", "自由比例", 0, 0),
    ("135", "135", 3, 2),
    ("xpan", "XPan", 65, 24),
    ("645", "6×4.5", 4, 3),
    ("66", "6×6", 1, 1),
    ("67", "6×7", 69, 56),
    ("68", "6×8", 76, 56),
    ("69", "6×9", 3, 2),
    ("612", "6×12", 2, 1),
    ("617", "6×17", 3, 1),
)
_ASPECTS = {key: w / h for key, _, w, h in FILM_FORMATS if h}


def crop_aspect(aspect, width, height):
    ratio = _ASPECTS.get(aspect)
    return 1 / ratio if ratio is not None and height > width else ratio


def rotation_matrix(rotation):
    angle = radians(rotation)
    c, s = cos(angle), sin(angle)
    return np.array(((c, -s), (s, c)))


def frame_points(rect, rotation):
    left, top, right, bottom = rect
    return np.array(((left, top), (right, top), (right, bottom), (left, bottom))) @ rotation_matrix(rotation)


def fit_rect(shape, rect, rotation, aspect="free"):
    """Keep a rectangle inside the scan while maintaining the requested aspect."""
    height, width = shape[:2]
    left, top, right, bottom = rect
    center = np.array(((left + right) / 2, (top + bottom) / 2))
    extent = np.array((right - left, bottom - top), dtype=float)
    if not np.all(np.isfinite((*center, *extent, rotation))) or np.any(extent <= 0):
        raise ValueError("裁切边界无效")
    ratio = crop_aspect(aspect, *extent)
    if ratio is not None:
        extent = np.minimum(extent, (extent[1] * ratio, extent[0] / ratio))
    rotation_axes = rotation_matrix(rotation)
    half_bounds = np.abs(rotation_axes.T) @ (extent / 2)
    bounds = np.array((width, height))
    factor = min(1.0, *(bounds / (2 * half_bounds)))
    center_in_source = np.clip(center @ rotation_axes, half_bounds * factor, bounds - half_bounds * factor)
    center = center_in_source @ rotation_axes.T
    half = extent * factor / 2
    return [*(center - half), *(center + half)]


def crop_from_edges(shape, corners=None, *, rotation=0.0, aspect="free", border_percent=0.0, bounding=False):
    """Use detected edges to propose an inside rectangle at the chosen rotation."""
    height, width = shape[:2]
    if corners is None:
        corners = ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1))
    points = np.asarray(corners, dtype=float)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError("裁切边界无效")
    points = points @ rotation_matrix(rotation).T
    if bounding:
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
    else:
        left = max(points[0, 0], points[3, 0])
        top = max(points[0, 1], points[1, 1])
        right = min(points[1, 0], points[2, 0])
        bottom = min(points[2, 1], points[3, 1])
    rect = fit_rect(shape, (left, top, right + 1, bottom + 1), rotation, aspect)
    return {"rect": rect, "rotation": rotation, "aspect": aspect, "border_percent": border_percent}


def rectangular_settings(shape, settings):
    if "rect" in settings:
        return settings
    # Older workspaces stored four corners. Use their rectangular framing
    # without applying the former perspective transform.
    return crop_from_edges(shape, settings["corners"], bounding=not settings.get("straighten", True),
                           border_percent=settings.get("border_percent", 0))


def crop_plan(shape, settings):
    """Return the source rectangle, a rigid transform and native output dimensions."""
    height, width = shape[:2]
    settings = rectangular_settings(shape, settings)
    rotation = float(settings["rotation"])
    left, top, right, bottom = settings["rect"]
    rect = np.asarray((left, top, right, bottom), dtype=float)
    if not np.all(np.isfinite(rect)) or right - left < 1 or bottom - top < 1:
        raise ValueError("裁切边界无效")
    points = frame_points(rect, rotation)
    bounds = np.array((width, height))
    if np.any(points < -1e-5) or np.any(points > bounds + 1e-5):
        raise ValueError("裁切边界超出原片")
    center = (rect[:2] + rect[2:]) / 2
    source_center = center @ rotation_matrix(rotation)
    vectors = points - source_center
    limits = np.divide(np.where(vectors > 0, bounds - source_center, -source_center),
                       vectors, out=np.full_like(vectors, np.inf), where=np.abs(vectors) > 1e-8)
    factor = max(1.0, min(1 + max(0, float(settings.get("border_percent", 0))) / 50, float(limits.min())))
    extent = (rect[2:] - rect[:2]) * factor
    output = tuple(max(1, floor(value + 1e-6)) for value in extent)
    origin = center - np.asarray(output) / 2
    if rotation == 0:
        origin = np.clip(np.rint(origin), (0, 0), bounds - output)
    output_rect = [*origin, *(origin + output)]
    matrix = np.eye(3)
    matrix[:2, :2] = rotation_matrix(rotation)
    matrix[:2, 2] = -origin
    return frame_points(output_rect, rotation), matrix, output, (factor - 1) * 50


def apply_crop(image, settings):
    if settings is None:
        return image
    _, matrix, (width, height), _ = crop_plan(image.shape, settings)
    if matrix[0, 0] == 1 and matrix[1, 0] == 0:
        x, y = np.rint(-matrix[:2, 2]).astype(int)
        return image[y:y + height, x:x + width].copy()
    # Qt geometry uses pixel edges, while OpenCV samples at pixel centers.
    affine = matrix[:2].copy()
    affine[:, 2] += affine[:, :2] @ np.full(2, 0.5) - 0.5
    return cv2.warpAffine(image, affine, (width, height), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

"""A narrow, two-sided ridge pass for short scratches and curved fibers."""

import cv2
import numpy as np

from .native_detector import NativeDustDetector

LINE_PREVIEW_EDGE = 4800
_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (1, -2))
_UNITS = 16


def _stroke_responses(gray):
    pixels = gray.astype(np.float32)
    center = cv2.GaussianBlur(pixels, (7, 7), 0.9)
    noise = cv2.boxFilter(np.abs(pixels - center), -1, (13, 13))
    norm = 1.65 + 0.72 * noise
    pad = 12
    padded = cv2.copyMakeBorder(center, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    h, w = pixels.shape

    def neighbors(dx, dy, distance):
        step = max(1, round(distance / np.hypot(dx, dy)))
        x, y = dx * step, dy * step
        return (padded[pad + y:pad + y + h, pad + x:pad + x + w],
                padded[pad - y:pad - y + h, pad - x:pad - x + w])

    responses = [np.zeros_like(center), np.zeros_like(center)]
    for dx, dy in _DIRECTIONS:
        near_a, near_b = neighbors(dx, dy, 3)
        wide_a, wide_b = neighbors(dx, dy, 6)
        far_a, far_b = neighbors(dx, dy, 10)
        for response, sign in zip(responses, (1, -1)):
            near = np.minimum(sign * (center - near_a), sign * (center - near_b))
            wide = np.minimum(sign * (center - wide_a), sign * (center - wide_b))
            far_low = np.minimum(sign * (center - far_a), sign * (center - far_b))
            far_high = np.maximum(sign * (center - far_a), sign * (center - far_b))
            # Both flanks must support the same contrast polarity.
            gate = np.clip((far_low + 2) / np.maximum(3, 0.22 * far_high), 0, 1)
            score = np.maximum(near, 0.70 * wide) * gate / norm
            np.maximum(response, score, out=response)
    return responses


def _without_flank_echoes(candidates):
    """Suppress weaker, opposite-polarity fragments hugging a stronger ridge."""
    kept = []
    for item, bright in sorted(candidates, key=lambda pair: pair[0].confidence, reverse=True):
        for other, other_bright in kept:
            if (bright == other_bright or other.area < 2 * item.area
                    or other.confidence - item.confidence < 0.025):
                continue
            if cv2.pointPolygonTest(other.contour, item.center, True) < -9:
                continue
            points = item.contour.reshape(-1, 2).astype(float)
            near = sum(cv2.pointPolygonTest(other.contour, tuple(p), True) >= -9 for p in points)
            if near >= 0.80 * len(points):
                break
        else:
            kept.append((item, bright))
    return kept


def detect_strokes(image, *, region=None, confidence=0.34, progress=None):
    height, width = image.shape[:2]
    if min(height, width) < 3:
        return []
    x1, y1, x2, y2 = region or (0, 0, width, height)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return []
    halo, tile = 36, 896
    left, top = max(0, x1 - halo), max(0, y1 - halo)
    patch = image[top:min(height, y2 + halo), left:min(width, x2 + halo)]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    if gray.dtype == np.uint16:
        gray = cv2.convertScaleAbs(gray, alpha=255 / 65535)
    left_x, top_y = x1 - left, y1 - top
    right_x, bottom_y = x2 - left, y2 - top
    maps = [np.zeros((y2 - y1, x2 - x1), np.uint8) for _ in range(2)]
    total = ((x2 - x1 + tile - 1) // tile) * ((y2 - y1 + tile - 1) // tile)
    done = 0
    for y in range(top_y, bottom_y, tile):
        bottom = min(y + tile, bottom_y)
        ty = max(0, y - halo)
        for x in range(left_x, right_x, tile):
            right = min(x + tile, right_x)
            tx = max(0, x - halo)
            fields = _stroke_responses(gray[ty:bottom + halo, tx:right + halo])
            inner = np.s_[y - ty:bottom - ty, x - tx:right - tx]
            dest = np.s_[y - top_y:bottom - top_y, x - left_x:right - left_x]
            for target, field in zip(maps, fields):
                target[dest] = np.clip(field[inner] * _UNITS, 0, 255)
            done += 1
            if progress is not None:
                progress(done, total)

    # Keep high-contrast, coherent fragments in the default pass. The existing
    # multiscale detector remains responsible for long or broader defects.
    local_search = region is not None
    minimum = 13 if local_search else 23
    extent = max(150, 0.06 * min(height, width))
    strictness = float(np.clip((confidence - 0.25) / 0.09, 0, 1))
    seed = (4.5 if local_search else 5.5 + 1.5 * strictness) * _UNITS
    core = (3.2 if local_search else 4.0 + strictness) * _UNITS
    core_fraction = 0.25 if local_search else 0.38
    support = round((1.9 + 0.3 * strictness) * _UNITS)
    items = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for field, bright in zip(maps, (True, False)):
        mask = cv2.compare(field, support, cv2.CMP_GE)
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, dst=mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        links = hierarchy[0]
        for index, contour in enumerate(contours):
            if links[index, 3] != -1:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if not minimum <= max(w, h) <= extent or min(w, h) < 3:
                continue
            parts = [contour]
            child = links[index, 2]
            while child != -1:
                parts.append(contours[child])
                child = links[child, 0]
            area = cv2.contourArea(contour) - sum(cv2.contourArea(hole) for hole in parts[1:])
            perimeter = sum(cv2.arcLength(part, True) for part in parts)
            if area < 16 or perimeter * perimeter < 24 * area or 2 * area > 9 * perimeter:
                continue
            local = np.zeros((h, w), np.uint8)
            cv2.drawContours(local, parts, -1, 1, cv2.FILLED, offset=(-x, -y))
            response = field[y:y + h, x:x + w]
            values = response[local != 0]
            if not values.size or values.max() < seed or np.mean(values >= core) < core_fraction:
                continue
            cutoff = max(4.5 * _UNITS, float(values.max()) * 0.45)
            if not local_search and not NativeDustDetector._has_two_sided_ridge(
                gray, local, response, x + left_x, y + top_y, cutoff, bright
            ):
                continue
            cy, cx = np.nonzero(local & (response >= cutoff))
            nearest = np.argmin((cx - cx.mean()) ** 2 + (cy - cy.mean()) ** 2)
            a, b = cv2.minAreaRect(contour)[1]
            kind = "scratch" if (max(a, b) + 1) / (min(a, b) + 1) >= 3.4 else "fiber"
            for part in parts:
                part += (x1, y1)
            score = min(0.94, 0.54 + float(np.median(values)) / (_UNITS * 45))
            item = NativeDustDetector._item(
                contour, x + int(cx[nearest]) + x1, y + int(cy[nearest]) + y1,
                kind, score, cv2.countNonZero(local),
            )
            item.holes = tuple(parts[1:])
            items.append((item, bright))
    if local_search:
        items = _without_flank_echoes(items)
    return [item for item, _ in items]



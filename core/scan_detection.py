"""Choose an infrared or RGB detection path and return original-pixel contours."""

from dataclasses import dataclass

import cv2
import numpy as np

from .dust_item import DustItem
from .linear_defects import LINE_PREVIEW_EDGE, detect_strokes
from .native_detector import NativeDustDetector
from .scene_focus import SceneFocus

PREVIEW_EDGE = 2800


@dataclass
class ScanResult:
    items: list[DustItem]
    source: str


def _infrared_calibration(red, infrared):
    x = np.log1p(red.astype(np.float32)).ravel()
    y = infrared.ravel().astype(np.float32)
    if x.std() < 0.08 or y.std() < 1 or np.corrcoef(x, y)[0, 1] < 0.45:
        return None
    center = float(np.median(x))
    centered = x - center
    design = np.column_stack((np.ones_like(x), centered, centered ** 2))
    keep = np.ones(len(x), dtype=bool)
    for _ in range(3):
        if np.count_nonzero(keep) < 20:
            return None
        coef = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        error = y - design @ coef
        deviation = np.abs(error - np.median(error))
        keep = deviation < 3.5 * (np.median(deviation) + 1)
    if np.median(deviation) > 0.85 * np.median(np.abs(y - np.median(y))):
        return None
    lower, upper = np.percentile(centered, (1, 99))
    return center, lower, upper, coef[1], coef[2]


def _infrared_defects(plane, region, progress, image=None):
    if plane.dtype not in (np.uint8, np.uint16):
        return None
    h, w = plane.shape
    rows, cols = max(1, h // 256), max(1, w // 256)
    sample = plane[::rows, ::cols].astype(np.float32)
    # Scan grain produces low-amplitude local differences. A sensor whose
    # infrared exposure is almost entirely black offers no useful reference.
    baseline = float(np.median(sample))
    if baseline <= 0.005 * np.iinfo(plane.dtype).max:
        return None
    left_sample = plane[::rows, :-1:cols].astype(np.float32)
    right_sample = plane[::rows, 1::cols].astype(np.float32)
    noise = float(np.median(np.abs(right_sample - left_sample))) if left_sample.size else 0
    if noise > 0.12 * baseline:
        return None
    calibration = None
    if image is not None and image.ndim == 3:
        calibration = _infrared_calibration(image[::rows, ::cols, 2], sample)
    x1, y1, x2, y2 = region or (0, 0, w, h)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return []
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    seeds = np.zeros_like(mask)
    tile_size, halo = 1024, 60
    total = ((x2 - x1 + tile_size - 1) // tile_size
             * ((y2 - y1 + tile_size - 1) // tile_size))
    done = 0
    for y in range(y1, y2, tile_size):
        end_y = min(y + tile_size, y2)
        top = max(0, y - halo)
        for x in range(x1, x2, tile_size):
            end_x = min(x + tile_size, x2)
            left = max(0, x - halo)
            window = np.s_[top:end_y + halo, left:end_x + halo]
            patch = plane[window].astype(np.float32)
            if calibration is not None:
                middle, lower, upper, slope, curve = calibration
                # Use a smoothed visible plane to avoid explaining away a
                # dark obstruction present in both the visible and IR scans.
                red = cv2.GaussianBlur(image[window][..., 2].astype(np.float32), (0, 0), 2.4)
                density = np.clip(np.log1p(red) - middle, lower, upper)
                patch -= density * (slope + curve * density)
            patch = cv2.GaussianBlur(patch, (5, 5), 0.65)
            area = np.s_[y - top:end_y - top, x - left:end_x - left]
            target = np.s_[y - y1:end_y - y1, x - x1:end_x - x1]
            # Dark obstructions are more evident against an upper local
            # envelope than against an average containing the defect itself.
            near = cv2.morphologyEx(
                patch, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            )[area]
            broad = cv2.boxFilter(patch, -1, (65, 65))[area]
            background = np.maximum(near, broad)
            residual = background - patch[area]
            cutoff = np.maximum(background * 0.025, max(noise * 6, baseline * 0.007, 1.5))
            seeds[target] = (residual >= cutoff).astype(np.uint8)
            mask[target] = (residual >= cutoff * 0.58).astype(np.uint8)
            done += 1
            progress(done, total)
    if np.count_nonzero(seeds) > seeds.size * 0.23:
        return None

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    items = []
    if hierarchy is None:
        return items
    links = hierarchy[0]
    for index, contour in enumerate(contours):
        if links[index, 3] != -1:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width * height < 4 or cv2.contourArea(contour) > 0.035 * w * h:
            continue
        parts = [contour]
        child = links[index, 2]
        while child != -1:
            parts.append(contours[child])
            child = links[child, 0]
        local = np.zeros((height, width), np.uint8)
        cv2.drawContours(local, parts, -1, 1, cv2.FILLED, offset=(-x, -y))
        dark = local & seeds[y:y + height, x:x + width]
        if cv2.countNonZero(dark) < 3:
            continue
        py, px = np.nonzero(dark)
        nearest = np.argmin((px - px.mean()) ** 2 + (py - py.mean()) ** 2)
        center_x, center_y = x + int(px[nearest]) + x1, y + int(py[nearest]) + y1
        for part in parts:
            part += (x1, y1)
        items.append(DustItem(
            0, x + x1, y + y1, width, height, cv2.countNonZero(local),
            center_x, center_y, dust_type="infrared", confidence=0.9,
            contour=contour, holes=tuple(parts[1:]), aspect_ratio=max(width, height) / min(width, height),
        ))
    return items


def _map_from_preview(items, source_shape, preview_shape):
    h, w = source_shape[:2]
    ph, pw = preview_shape[:2]
    factors = np.array((w / pw, h / ph))
    limits = np.array((w - 1, h - 1))

    def mapped(points):
        return np.clip(np.rint((np.asarray(points) + 0.5) * factors - 0.5), 0, limits).astype(np.int32)

    for item in items:
        item.preview = True
        item.contour = mapped(item.contour)
        item.holes = tuple(mapped(hole) for hole in item.holes)
        item.x, item.y, item.width, item.height = cv2.boundingRect(item.contour)
        item.center_x, item.center_y = map(int, mapped(item.center))
        item.area = max(1, round(item.area * factors.prod()))
        item.aspect_ratio = max(item.width, item.height) / min(item.width, item.height)
    return items


def _resize_for_scan(image, maximum):
    h, w = image.shape[:2]
    if max(h, w) <= maximum:
        return image
    ratio = maximum / max(h, w)
    return cv2.resize(image, (max(1, round(w * ratio)), max(1, round(h * ratio))),
                      interpolation=cv2.INTER_AREA)


def _overlaps_stroke(item, stroke):
    if not (stroke.x - 3 <= item.center_x < stroke.x + stroke.width + 3
            and stroke.y - 3 <= item.center_y < stroke.y + stroke.height + 3):
        return False
    return cv2.pointPolygonTest(stroke.contour, item.center, True) >= -3


def _classify_surface_patches(items):
    for item in items:
        if item.dust_type not in ("bright", "dark") or item.area < 130 or item.confidence < 0.39:
            continue
        if min(item.width, item.height) < 10 or item.aspect_ratio > 3:
            continue
        contour = item.contour
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        perimeter = cv2.arcLength(contour, True)
        broad = item.area >= 500 and min(item.width, item.height) >= 18 and item.confidence >= 0.48
        ragged = (hull_area and perimeter and
                  (cv2.contourArea(contour) / hull_area < 0.9
                   or 4 * np.pi * item.area / perimeter ** 2 < 0.65))
        if broad or ragged:
            item.dust_type = "damage"


def detect_scan(image, params, *, infrared=None, quick=True, region=None, subject_first=True,
                stroke_floor=0.0, progress=None):
    report = progress if progress is not None else lambda *args: None
    infrared_items = None
    if infrared is not None:
        infrared_items = _infrared_defects(
            infrared, region, lambda done, total: report("红外", done, total), image=image
        )
        if infrared_items is not None and region is None:
            for index, item in enumerate(infrared_items, 1):
                item.id = index
            return ScanResult(infrared_items, "红外")
    detector = NativeDustDetector(**params)
    source = "RGB 原图"
    scan = image
    if quick and region is None:
        scan = _resize_for_scan(image, PREVIEW_EDGE)
        if scan is not image:
            source = "RGB 快速预览"

    items = detector.detect(scan, region=region, aggressive=region is not None,
                            progress=lambda done, total: report(source, done, total))
    if scan is not image:
        _map_from_preview(items, image.shape, scan.shape)
    line_image = _resize_for_scan(image, LINE_PREVIEW_EDGE) if quick and region is None else image
    strokes = detect_strokes(
        line_image, region=region, confidence=params.get("min_confidence", 0.25),
        progress=lambda done, total: report("划痕与毛絮", done, total),
    )
    if line_image is not image:
        _map_from_preview(strokes, image.shape, line_image.shape)
    strokes = [stroke for stroke in strokes if stroke.confidence >= stroke_floor]
    minimum_confidence = params.get("min_confidence", 0.25)
    focus = SceneFocus(image) if subject_first and region is None else None
    if focus is not None:
        for forehead in focus.forehead_regions():
            for candidate in detect_strokes(image, region=forehead):
                if candidate.confidence >= 0.645 and not any(
                    _overlaps_stroke(candidate, stroke) or _overlaps_stroke(stroke, candidate)
                    for stroke in strokes
                ):
                    strokes.append(candidate)
        strokes = focus.filter(strokes, minimum_confidence)
    items = [item for item in items if not any(_overlaps_stroke(item, stroke) for stroke in strokes)]
    _classify_surface_patches(items)
    if focus is not None:
        items = focus.filter(items, minimum_confidence)
    if infrared_items is not None:
        infrared_items = [item for item in infrared_items
                          if not any(_overlaps_stroke(item, stroke) for stroke in strokes)]
        items = [item for item in items if not any(
            _overlaps_stroke(item, defect) for defect in infrared_items
        )]
        source = "红外 + RGB 原图"
    items = sorted([*items, *strokes, *(infrared_items or [])],
                   key=lambda item: item.confidence, reverse=True)
    for i, item in enumerate(items, 1):
        item.id = i
    if infrared is not None and infrared_items is None:
        source += "（红外信号不可用）"
    return ScanResult(items, source)


def detect_brush(image, params, points, radius, *, infrared=None, progress=None):
    """Use a brush or near-closed loop to scope detection in original pixels."""
    height, width = image.shape[:2]
    points = np.clip(np.rint(points).astype(np.int32), (0, 0), (width - 1, height - 1))
    x1, y1 = map(int, np.maximum(points.min(axis=0) - radius, 0))
    x2, y2 = map(int, np.minimum(points.max(axis=0) + radius + 1, (width, height)))
    local = points - (x1, y1)
    search = np.zeros((y2 - y1, x2 - x1), np.uint8)
    closed = len(local) >= 3 and np.linalg.norm(local[0] - local[-1]) <= radius
    if closed:
        cv2.fillPoly(search, [local], 255)
    cv2.polylines(search, [local], False, 255, 2 * radius)
    cv2.circle(search, tuple(local[0]), radius, 255, cv2.FILLED)

    # Trace candidates with room beyond the gesture, then use the gesture only
    # to decide which complete contours to return.
    halo = 48
    region = (max(0, x1 - halo), max(0, y1 - halo),
              min(width, x2 + halo), min(height, y2 + halo))
    # A rough brush can cover much more texture than the defect itself.
    # Reject weak ridge fragments before they can hide a nearby dust spot.
    result = detect_scan(image, params, region=region, infrared=infrared,
                         stroke_floor=0.64, progress=progress)
    kept = []
    for item in result.items:
        if (item.dust_type in ("bright", "dark", "damage")
                and item.confidence < params.get("min_confidence", 0.25)):
            continue
        x, y = item.center
        if x1 <= x < x2 and y1 <= y < y2 and search[y - y1, x - x1]:
            kept.append(item)
        elif item.dust_type in ("fiber", "scratch"):
            left, top = max(x1, item.x), max(y1, item.y)
            right, bottom = min(x2, item.x + item.width), min(y2, item.y + item.height)
            if left >= right or top >= bottom:
                continue
            overlap = np.zeros((bottom - top, right - left), np.uint8)
            cv2.drawContours(overlap, [item.contour, *item.holes], -1, 255,
                             cv2.FILLED, offset=(-left, -top))
            overlap &= search[top - y1:bottom - y1, left - x1:right - x1]
            if cv2.countNonZero(overlap) >= 0.35 * item.area:
                kept.append(item)
    result.items = kept
    return result


def refine_candidate(image, item, params, *, progress=None):
    """Retrace a proxy contour on original pixels using its neighborhood."""
    height, width = image.shape[:2]
    margin = max(20, min(96, max(item.width, item.height) // 2))
    region = (max(0, item.x - margin), max(0, item.y - margin),
              min(width, item.x + item.width + margin), min(height, item.y + item.height + margin))
    candidates = detect_scan(image, params, region=region, progress=progress).items
    related = ("fiber", "scratch")
    scale = max(4, min(item.width, item.height))
    matched = []
    for candidate in candidates:
        if (candidate.dust_type != item.dust_type
                and not (candidate.dust_type in related and item.dust_type in related)):
            continue
        distance = np.hypot(candidate.center_x - item.center_x, candidate.center_y - item.center_y)
        overlap = (min(item.x + item.width, candidate.x + candidate.width) > max(item.x, candidate.x)
                   and min(item.y + item.height, candidate.y + candidate.height) > max(item.y, candidate.y))
        if overlap and distance <= max(scale, 0.4 * max(item.width, item.height)):
            matched.append((distance / scale + abs(np.log(candidate.area / max(1, item.area))) * 0.18,
                            candidate))
    return min(matched, key=lambda pair: pair[0])[1] if matched else None

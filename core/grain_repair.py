"""Restore local film texture in the interior of a repaired contour."""

import cv2
import numpy as np


def restore_grain(image, repaired, mask, *, automatic=False):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    height, width = mask.shape
    maximum = np.iinfo(image.dtype).max
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if automatic and (area < 65 or max(w, h) < 12):
            continue
        left, top = max(0, x - 6), max(0, y - 6)
        right, bottom = min(width, x + w + 6), min(height, y + h + 6)
        region = np.s_[top:bottom, left:right]
        active = np.uint8(labels[region] == label)
        ring = (cv2.dilate(active, kernel) != 0) & (mask[region] == 0)
        if np.count_nonzero(ring) < 12:
            continue
        local = repaired[region].astype(np.float32)
        base = cv2.GaussianBlur(local, (0, 0), 1.2)
        best = None
        gap_x, gap_y = w + 8, h + 8
        offsets = ((gap_x, 0), (-gap_x, 0), (0, gap_y), (0, -gap_y),
                   (gap_x, 6), (-gap_x, -6), (6, gap_y), (-6, -gap_y))
        for dx, dy in offsets:
            if not (0 <= left + dx < right + dx <= width and 0 <= top + dy < bottom + dy <= height):
                continue
            donor_slice = np.s_[top + dy:bottom + dy, left + dx:right + dx]
            if np.any(mask[donor_slice]):
                continue
            donor = image[donor_slice].astype(np.float32)
            smooth = cv2.GaussianBlur(donor, (0, 0), 1.2)
            difference = np.abs(base[ring] - smooth[ring])
            score = float(np.mean(difference))
            if best is None or score < best[0]:
                best = score, donor - smooth
        if best is None:
            continue
        inside = active != 0
        # Retain the interpolated edge and add only fine-scale residuals.
        weight = np.minimum(1, cv2.distanceTransform(active, cv2.DIST_L2, 3) / 3)[inside]
        residual = best[1][inside]
        if residual.ndim == 2:
            weight = weight[:, None]
        target = repaired[region]
        target[inside] = np.clip(np.rint(target[inside] + residual * weight), 0, maximum).astype(image.dtype)

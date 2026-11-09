"""Native-resolution detection of small spots and thin film defects."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from .dust_item import DustItem


class NativeDustDetector:
    """Score both polarities against local texture before extracting contours."""

    TILE_SIZE = 1024
    SCALES = (1.5, 2.5, 4.0, 6.0, 9.0, 13.0)
    RIDGE_UNITS = 16

    def __init__(self, threshold: int = 20, min_area: int = 3,
                 min_confidence: float = 0.25):
        self.threshold = threshold
        self.min_area = min_area
        self.min_confidence = min_confidence

    def detect(self, image: np.ndarray, region=None, aggressive=False,
               progress: Callable[[int, int], None] | None = None) -> list[DustItem]:
        height, width = image.shape[:2]
        if min(height, width) < 3:
            return []

        # Keep small native-pixel scales at every resolution. Extend the range
        # in both directions for resized exports and high-resolution scans.
        scales = self.SCALES
        short_side = min(height, width)
        if short_side < 2000:
            scales = (0.85,) + scales
        elif short_side >= 4000:
            scales += (19.0, 26.0)
        peak_radius = int(np.clip(round(short_side * 0.003), 3, 9))
        peak_kernel = np.ones((2 * peak_radius + 1,) * 2, np.uint8)
        halo = int(np.ceil(13 * scales[-1])) + peak_radius + 4

        x1, y1, x2, y2 = (0, 0, width, height)
        if region is not None:
            x1, y1, x2, y2 = region
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
        if x1 >= x2 or y1 >= y2:
            return []

        # Read outside a user selection for context, and retain coordinates in
        # the full scan. Tile halos also include the normalization windows.
        ctx_x, ctx_y = max(0, x1 - halo), max(0, y1 - halo)
        source = image[ctx_y:min(height, y2 + halo), ctx_x:min(width, x2 + halo)]
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source
        if gray.dtype == np.uint16:
            gray = cv2.convertScaleAbs(gray, alpha=255 / 65535)
        left, top, right, bottom = x1 - ctx_x, y1 - ctx_y, x2 - ctx_x, y2 - ctx_y

        confidence = float(np.clip(self.min_confidence, 0.01, 0.95))
        sensitivity = self.threshold / 20 * (0.65 if aggressive else 1.0)
        spot_cutoff = 5 * confidence / (1 - confidence) * sensitivity
        ridge_cutoff = 8 * confidence / (1 - confidence) * sensitivity
        min_area = min(self.min_area, 3) if aggressive else self.min_area

        # Merge narrow responses at original coordinates before tracing long
        # fibers, so a fiber can cross arbitrarily many tiles.
        ridge_maps = [np.zeros((y2 - y1, x2 - x1), np.uint8) for _ in range(2)]
        spots = []
        tile_size = self.TILE_SIZE
        tile_total = ((right - left + tile_size - 1) // tile_size
                      * ((bottom - top + tile_size - 1) // tile_size))
        done = 0
        for y in range(top, bottom, tile_size):
            end_y = min(y + tile_size, bottom)
            for x in range(left, right, tile_size):
                end_x = min(x + tile_size, right)
                tx, ty = max(0, x - peak_radius), max(0, y - peak_radius)
                tx2 = min(gray.shape[1], end_x + peak_radius)
                ty2 = min(gray.shape[0], end_y + peak_radius)
                inner = np.s_[y - ty:end_y - ty, x - tx:end_x - tx]
                dest = np.s_[y - top:end_y - top, x - left:end_x - left]
                score, scale_index, polarity, ridges = self._responses(
                    gray, (tx, ty, tx2, ty2), scales
                )
                peaks = cv2.dilate(score, peak_kernel)[inner]
                tile_score = score[inner]
                py, px = np.where((tile_score >= spot_cutoff) & (tile_score == peaks))
                for dx, dy in zip(px.tolist(), py.tolist()):
                    sy, sx = y - ty + dy, x - tx + dx
                    spots.append((float(score[sy, sx]), x + dx, y + dy,
                                  scales[scale_index[sy, sx]], bool(polarity[sy, sx])))
                for field, values in zip(ridge_maps, ridges):
                    field[dest] = np.clip(values[inner] * self.RIDGE_UNITS, 0, 255)
                done += 1
                if progress is not None:
                    progress(done, tile_total)

        items = []
        covered = np.zeros_like(ridge_maps[0])
        for bright, field in zip((True, False), ridge_maps):
            for item in self._fibers(field, ridge_cutoff, min_area, peak_radius,
                                     gray, left, top, bright):
                cv2.drawContours(covered, [item.contour, *item.holes],
                                 -1, 255, cv2.FILLED)
                for contour in (item.contour, *item.holes):
                    contour += (left, top)
                item.x += left
                item.y += top
                item.center_x += left
                item.center_y += top
                items.append(item)

        for score, x, y, sigma, bright in sorted(spots, reverse=True):
            if covered[y - top, x - left]:
                continue
            contour = self._spot_contour(gray, x, y, sigma, bright, aggressive)
            if contour is None:
                continue
            item = self._item(contour, x, y, "bright" if bright else "dark",
                              score / (score + 5))
            if item.area >= min_area:
                cv2.drawContours(covered, [contour], -1, 255, cv2.FILLED,
                                 offset=(-left, -top))
                items.append(item)

        items.sort(key=lambda item: item.confidence, reverse=True)
        for i, item in enumerate(items, 1):
            item.id = i
            item.selected = aggressive
            item.x += ctx_x
            item.y += ctx_y
            item.center_x += ctx_x
            item.center_y += ctx_y
            for contour in (item.contour, *item.holes):
                contour += (ctx_x, ctx_y)
        return items

    @staticmethod
    def _responses(gray, bounds, scales):
        x1, y1, x2, y2 = bounds
        shape = (y2 - y1, x2 - x1)
        score = np.zeros(shape, np.float32)
        ridge_bright = np.zeros_like(score)
        ridge_dark = np.zeros_like(score)
        scale_index = np.zeros(shape, np.uint8)
        polarity = np.zeros(shape, bool)

        for i, sigma in enumerate(scales):
            # Include both the Gaussian and local normalization support.
            halo = int(np.ceil(13 * sigma)) + 3
            left, top = max(0, x1 - halo), max(0, y1 - halo)
            patch = gray[top:y2 + halo, left:x2 + halo].astype(np.float32)
            inner = np.s_[y1 - top:y2 - top, x1 - left:x2 - left]
            smooth = cv2.GaussianBlur(patch, (0, 0), sigma)
            scale = sigma * sigma / 4
            xx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, scale=scale)
            yy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, scale=scale)
            xy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, scale=scale)
            trace = (xx + yy) * 0.5
            gap = cv2.magnitude((xx - yy) * 0.5, xy)
            # The Hessian eigenvalues describe curvature in two directions.
            # Spots curve in both; a thin fiber curves mainly across its width.
            window = 2 * int(9 * sigma) + 1
            norm = 1 + cv2.boxFilter(abs(trace) + gap, -1, (window, window))[inner]
            lo, hi = trace[inner] - gap[inner], trace[inner] + gap[inner]
            bright = -hi > lo
            candidate = np.maximum(-hi, lo) / norm
            improve = candidate > score
            score[improve] = candidate[improve]
            scale_index[improve] = i
            polarity[improve] = bright[improve]
            np.maximum(ridge_bright, (-lo - 2 * abs(hi)) / norm, out=ridge_bright)
            np.maximum(ridge_dark, (hi - 2 * abs(lo)) / norm, out=ridge_dark)

        return score, scale_index, polarity, (ridge_bright, ridge_dark)

    def _fibers(self, field, cutoff, min_area, peak_radius, gray, left, top, bright):
        seed = int(np.ceil(cutoff * self.RIDGE_UNITS))
        if seed > 255:
            return
        support = max(1, int(seed * 0.64))
        mask = cv2.compare(field, support, cv2.CMP_GE)
        # Bridge small gaps before tracing the contours. A symmetric kernel
        # keeps the mask centered on the original defect.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, dst=mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return
        links = hierarchy[0]
        for index, contour in enumerate(contours):
            if links[index, 3] != -1:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if max(w, h) < max(11, 2 * peak_radius):
                continue
            holes = []
            child = links[index, 2]
            while child != -1:
                holes.append(contours[child])
                child = links[child, 0]
            area = cv2.contourArea(contour) - sum(cv2.contourArea(hole) for hole in holes)
            _, (a, b), _ = cv2.minAreaRect(contour)
            aspect = (max(a, b) + 1) / (min(a, b) + 1)
            if aspect < 2.6 and (area > 0.14 * (w * w + h * h) or max(w, h) < 28):
                continue

            local = np.zeros((h, w), np.uint8)
            cv2.drawContours(local, [contour, *holes], -1, 255,
                             cv2.FILLED, offset=(-x, -y))
            pixel_area = cv2.countNonZero(local)
            if pixel_area < min_area:
                continue
            response = field[y:y + h, x:x + w]
            _, peak, _, (dx, dy) = cv2.minMaxLoc(response, mask=local)
            if peak < seed:
                continue
            if not self._has_two_sided_ridge(
                gray, local, response, x + left, y + top, max(seed, peak * 0.63), bright
            ):
                continue
            strength = peak / self.RIDGE_UNITS
            item = self._item(contour, x + dx, y + dy, "fiber",
                              strength / (strength + 8), pixel_area)
            item.holes = tuple(holes)
            yield item

    @staticmethod
    def _has_two_sided_ridge(gray, local, response, x, y, minimum, bright):
        """A real fiber rises or falls against both sides of its cross-section."""
        py, px = np.where((local != 0) & (response >= minimum))
        samples = np.linspace(0, len(px) - 1, min(len(px), 24), dtype=int)
        centers, normals = [], []
        for k in samples:
            cx, cy = int(px[k]), int(py[k])
            x1, y1 = max(0, cx - 12), max(0, cy - 12)
            iy, ix = np.nonzero(local[y1:cy + 13, x1:cx + 13])
            if len(ix) < 4:
                continue
            vx, vy, _, _ = cv2.fitLine(
                np.column_stack((ix, iy)).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
            ).ravel()
            centers.append((x + cx, y + cy))
            normals.append((-vy, vx))
        if not centers:
            return False

        centers = np.asarray(centers, np.float32)
        normals = np.asarray(normals, np.float32)
        offsets = np.array(
            [0, 3, 5, 8, 12, -3, -5, -8, -12, 18, 28, 40, -18, -28, -40],
            np.float32,
        )
        coordinates = centers[:, None, :] + normals[:, None, :] * offsets[None, :, None]
        # Sample directly so very large panorama dimensions do not hit
        # remap's internal limit on image sizes.
        sx = np.clip(coordinates[:, :, 0], 0, gray.shape[1] - 1)
        sy = np.clip(coordinates[:, :, 1], 0, gray.shape[0] - 1)
        x0, y0 = sx.astype(np.intp), sy.astype(np.intp)
        x1, y1 = np.minimum(x0 + 1, gray.shape[1] - 1), np.minimum(y0 + 1, gray.shape[0] - 1)
        ax, ay = sx - x0, sy - y0
        upper = gray[y0, x0] * (1 - ax) + gray[y0, x1] * ax
        lower = gray[y1, x0] * (1 - ax) + gray[y1, x1] * ax
        values = upper * (1 - ay) + lower * ay
        direction = 1 if bright else -1
        side_a = direction * (values[:, :1] - values[:, 1:5])
        side_b = direction * (values[:, :1] - values[:, 5:9])
        weaker = np.minimum(side_a, side_b)
        stronger = np.maximum(side_a, side_b)
        bilateral = np.any((weaker >= 3.0) & (weaker >= 0.32 * stronger), axis=1)

        # A bright seam on a dark object can pass a narrow profile. At a
        # larger distance its two backgrounds usually differ substantially.
        far_a = direction * (values[:, 0] - np.median(values[:, 9:12], axis=1))
        far_b = direction * (values[:, 0] - np.median(values[:, 12:15], axis=1))
        same_background = (np.minimum(far_a, far_b) >= 2) & (
            np.abs(far_a - far_b) <= np.maximum(8, 0.66 * np.maximum(far_a, far_b))
        )
        if max(local.shape) >= 100:
            # A long folded or bevelled surface builds contrast gradually.
            # A thin fiber already has contrast a few pixels off its axis.
            near = np.max(weaker[:, :2], axis=1)
            bilateral &= near >= np.maximum(2, 0.20 * np.maximum(far_a, far_b))
        return np.count_nonzero(bilateral & same_background) >= max(
            1, np.ceil(len(centers) * 0.55)
        )

    @staticmethod
    def _spot_contour(gray, x, y, sigma, bright, aggressive):
        radius = max(3, int(np.ceil(3 * sigma)))
        context = max(radius + 2, int(np.ceil(9 * sigma)))
        x1, y1 = max(0, x - context), max(0, y - context)
        x2, y2 = min(gray.shape[1], x + context + 1), min(gray.shape[0], y + context + 1)
        patch = gray[y1:y2, x1:x2].astype(np.float32)
        smooth = cv2.GaussianBlur(patch, (0, 0), max(0.5, sigma * 0.3))
        background = cv2.GaussianBlur(patch, (0, 0), 2.5 * sigma)
        residual = smooth - background if bright else background - smooth
        px, py = x - x1, y - y1
        center_response = float(residual[py, px])
        if center_response < 1:
            return None

        # Broad step edges produce a compact response at corners. Verify
        # that the spot is locally distinct in several directions.
        distance = max(3, int(np.ceil(2.3 * sigma)))
        angles = np.arange(8) * (np.pi / 4)
        sx = np.clip(px + np.rint(distance * np.cos(angles)).astype(int), 0, patch.shape[1] - 1)
        sy = np.clip(py + np.rint(distance * np.sin(angles)).astype(int), 0, patch.shape[0] - 1)
        deltas = (1 if bright else -1) * (smooth[py, px] - smooth[sy, sx])
        if np.count_nonzero(deltas > 0.7) < (4 if aggressive else 5):
            return None

        search = np.zeros(patch.shape, np.uint8)
        cv2.circle(search, (px, py), radius, 255, cv2.FILLED)
        mask = cv2.compare(residual, max(0.8, center_response * 0.23), cv2.CMP_GE)
        cv2.bitwise_and(mask, search, dst=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
                contour += (x1, y1)
                return contour
        return None

    @staticmethod
    def _item(contour, center_x, center_y, kind, confidence, pixel_area=None):
        x, y, w, h = cv2.boundingRect(contour)
        if pixel_area is None:
            local = np.zeros((h, w), np.uint8)
            cv2.drawContours(local, [contour], -1, 255, cv2.FILLED, offset=(-x, -y))
            pixel_area = cv2.countNonZero(local)
        return DustItem(0, x, y, w, h, pixel_area, center_x, center_y,
                        dust_type=kind, confidence=confidence, contour=contour,
                        aspect_ratio=max(w, h) / min(w, h))

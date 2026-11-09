"""Local face and texture context for prioritizing visible film defects."""

from pathlib import Path

import cv2
import numpy as np

from .scan_io import display_pixels

_MODELS = Path(__file__).resolve().parents[1] / "models"


def _detect_faces(pixels):
    h, w = pixels.shape[:2]
    if min(h, w) < 40:
        return []
    bgr = pixels if pixels.ndim == 3 else cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    version = tuple(map(int, cv2.__version__.split(".")[:2]))
    name = "face_detection_yunet_2022mar.onnx" if version < (4, 8) else "face_detection_yunet_2023mar.onnx"
    model = _MODELS / name
    if model.is_file() and hasattr(cv2, "FaceDetectorYN_create"):
        try:
            detector = cv2.FaceDetectorYN_create(str(model), "", (w, h), 0.86)
            for scale in (1, 0.5):
                view = bgr if scale == 1 else cv2.resize(bgr, None, fx=scale, fy=scale,
                                                        interpolation=cv2.INTER_AREA)
                vh, vw = view.shape[:2]
                view = cv2.copyMakeBorder(view, 0, -vh % 32, 0, -vw % 32, cv2.BORDER_REPLICATE)
                detector.setInputSize(view.shape[1::-1])
                _, faces = detector.detect(view)
                if faces is not None:
                    return [face[:4] / scale for face in faces
                            if 0 <= face[4] < vw and 0 <= face[5] < vh]
            return []
        except cv2.error:
            # Older OpenCV builds can still use the bundled frontal-face cascade.
            pass
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        return []
    return cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))


def _perforated_margins(gray):
    """Recognize rows of similarly sized, periodically spaced sprocket holes."""
    margins = []
    for axis, plane in enumerate((gray, gray.T)):
        h, w = plane.shape
        if min(h, w) < 100:
            continue
        _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(plane < 10))
        holes = [(x, y, width, height) for x, y, width, height, area in stats[1:]
                 if 0.012 * w <= width <= 0.075 * w
                 and 0.02 * h <= height <= 0.14 * h
                 and 0.50 <= width / height <= 1.5
                 and area >= 0.81 * width * height]
        for bottom in (False, True):
            edge_holes = [hole for hole in holes if (
                hole[1] >= 0.80 * h if bottom else hole[1] + hole[3] <= 0.20 * h
            )]
            for _, y, width, height in edge_holes:
                aligned = sorted((hole for hole in edge_holes
                                  if abs(hole[1] - y) <= height * 0.24
                                  and 0.80 * width <= hole[2] <= 1.20 * width
                                  and 0.80 * height <= hole[3] <= 1.20 * height))
                if len(aligned) < 4 or aligned[-1][0] - aligned[0][0] < 0.22 * w:
                    continue
                gaps = np.diff([hole[0] for hole in aligned])
                plausible = gaps[(gaps >= 1.5 * width) & (gaps <= 5 * width)]
                if plausible.size < 3:
                    continue
                typical = np.median(plausible)
                if np.count_nonzero(np.abs(plausible - typical) <= typical * 0.11) < 3:
                    continue
                boundary = (min(row[1] for row in aligned) - height * 0.12 if bottom
                            else max(row[1] + row[3] for row in aligned) + height * 0.12)
                margins.append((1 - axis, bottom, boundary))
                break
    return margins


class SceneFocus:
    """Rate film candidates in the scene without changing their contours."""

    def __init__(self, image):
        height, width = image.shape[:2]
        scale = min(1, 1000 / max(height, width))
        thumb = image if scale == 1 else cv2.resize(
            image, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        pixels = display_pixels(thumb)
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY) if pixels.ndim == 3 else pixels
        self.faces = _detect_faces(pixels)
        self.margins = _perforated_margins(gray)
        h, w = gray.shape
        self.factors = (w / width, h / height)
        soft = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
        dx = cv2.Sobel(soft, -1, 1, 0, ksize=3, scale=1 / 8)
        dy = cv2.Sobel(soft, -1, 0, 1, ksize=3, scale=1 / 8)
        gradient = cv2.blur(cv2.magnitude(dx, dy), (19, 19))
        fine_detail = cv2.blur(np.abs(gray - soft), (19, 19))
        self.texture = np.clip(np.maximum((fine_detail - 4) / 4, (gradient - 4) / 7), 0, 1)

    def forehead_regions(self):
        """Look for coherent narrow fibers above the eyes at original resolution."""
        sx, sy = self.factors
        for x, y, width, height in self.faces:
            yield (round((x + 0.04 * width) / sx), round((y - 0.15 * height) / sy),
                   round((x + 0.96 * width) / sx), round((y + 0.35 * height) / sy))

    def filter(self, items, min_confidence):
        kept = []
        h, w = self.texture.shape
        sx, sy = self.factors
        for item in items:
            point = (min(w - 1, round(item.center_x * sx)), min(h - 1, round(item.center_y * sy)))
            if any((point[axis] >= boundary) == bottom for axis, bottom, boundary in self.margins):
                continue
            x, y = point
            face = any(left - 0.08 * width <= x <= left + 1.08 * width
                       and top - 0.20 * height <= y <= top + height
                       for left, top, width, height in self.faces)
            texture = float(self.texture[y, x])
            if item.dust_type in ("fiber", "scratch"):
                cutoff = 0.59 + (0.04 if face else 0.13) * texture
            else:
                cutoff = min_confidence + (0.06 if face else 0.20) * texture
            if item.confidence >= cutoff:
                kept.append(item)
        return kept

"""Local face and texture context for prioritizing visible film defects."""

import logging
from pathlib import Path

import cv2
import numpy as np

from .frame_crop import _frame_defaults, _frame_lines
from .scan_io import display_pixels

_MODELS = Path(__file__).resolve().parents[1] / "models"
LOG = logging.getLogger(__name__)


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
            if version >= (4, 14):
                # Loading bytes also handles Windows paths containing non-ASCII characters.
                detector = cv2.FaceDetectorYN_create("onnx", model.read_bytes(), b"", (w, h), 0.86)
            else:
                detector = cv2.FaceDetectorYN_create(str(model), "", (w, h), 0.86)
            for scale in (1, 0.5):
                view = bgr if scale == 1 else cv2.resize(bgr, None, fx=scale, fy=scale,
                                                        interpolation=cv2.INTER_AREA)
                vh, vw = view.shape[:2]
                view = cv2.copyMakeBorder(view, 0, -vh % 32, 0, -vw % 32, cv2.BORDER_REPLICATE)
                detector.setInputSize(view.shape[1::-1])
                _, faces = detector.detect(view)
                if faces is not None:
                    return [face[:14] / scale for face in faces
                            if 0 <= face[4] < vw and 0 <= face[5] < vh]
            return []
        except (OSError, cv2.error) as error:
            LOG.warning("Local YuNet face detection unavailable: %s", error)

    # OpenCV 5 moved Haar cascades out of the standard package. The scene
    # filter can still prioritize smooth areas without face locations.
    classifier = getattr(cv2, "CascadeClassifier", None)
    cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if classifier is None or not cascade_dir:
        return []
    try:
        cascade = classifier(str(Path(cascade_dir) / "haarcascade_frontalface_default.xml"))
        if not cascade.empty():
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            return cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))
    except cv2.error as error:
        LOG.warning("Local Haar face detection unavailable: %s", error)
        return []
    return []


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


def _dark_frame_margins(gray):
    """Find the exposure window when cropped perforations merge with the border."""
    if min(gray.shape) < 120:
        return []
    margins = []
    strong = set()
    for axis, plane in ((1, gray), (0, gray.T)):
        depth, length = plane.shape
        inset = length // 10
        for end, oriented in ((False, plane), (True, plane[::-1])):
            strip = oriented[:round(0.13 * depth), inset:-inset]
            strip = cv2.blur(strip.astype(np.float32), (5, 3))
            outside, inside = strip[:-4], strip[4:]
            hits = (outside <= 18) & (inside >= 27) & (inside - outside >= 18)
            votes = np.convolve(hits.mean(axis=1), np.ones(5), mode="same")
            # Leave room to verify a dark band just outside the transition.
            row = 7 + int(np.argmax(votes[7:]))
            band = hits[row - 3:row + 4]
            coverage = band.any(axis=0).mean()
            dark_band = np.median(strip[max(0, row - 12):row - 3], axis=0) < 20
            if coverage < 0.48 or dark_band.mean() < 0.78:
                continue
            # Keep the outer side of a slightly tilted transition in the search.
            offset = row - 1 + np.percentile(np.nonzero(band)[0], 15)
            margins.append((axis, end, depth - offset if end else offset))
            if coverage >= 0.72:
                strong.add((axis, end))
    # Opposing, well-supported transitions distinguish a film frame from a
    # single dark wall or a shadow that happens to touch an image edge.
    return margins if any((axis, False) in strong and (axis, True) in strong for axis in (0, 1)) else []


def _perforated_exposure(gray):
    if min(gray.shape) < 120:
        return ()
    result = _frame_lines(gray, _frame_defaults(gray.shape))
    if result is None or not result[2]:
        return ()
    edges, scores, _ = result
    planes = []
    for index, (edge, score, inward) in enumerate(zip(edges, scores, (1, -1, -1, 1))):
        normal_x, normal_y, offset = edge * inward
        # With opposing sprocket rows confirmed, also screen the narrow
        # scanner fringe when a dark side of the exposure cannot be traced.
        depth = gray.shape[index % 2]
        margin = 1.5 if score > 0 else 0.008 * depth
        planes.append((normal_x, normal_y, offset - margin))
    return tuple(planes)


class SceneFocus:
    """Rate film candidates in the scene without changing their contours."""

    def __init__(self, image, *, prioritize_subject=True, filter_scene=True):
        height, width = image.shape[:2]
        scale = min(1, 1000 / max(height, width))
        thumb = image if scale == 1 else cv2.resize(
            image, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        pixels = display_pixels(thumb)
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY) if pixels.ndim == 3 else pixels
        h, w = gray.shape
        self.shape = gray.shape
        self.factors = (w / width, h / height)
        self.margins = _perforated_margins(gray) + _dark_frame_margins(gray) if filter_scene else []
        self.exposure = _perforated_exposure(gray) if filter_scene else ()
        faces = _detect_faces(pixels)
        self.faces = [face[:4] for face in faces]
        self.features = [ellipse for face in faces for ellipse in self._facial_features(face)]
        self.texture = None
        if not prioritize_subject or not filter_scene:
            return
        soft = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
        dx = cv2.Sobel(soft, -1, 1, 0, ksize=3, scale=1 / 8)
        dy = cv2.Sobel(soft, -1, 0, 1, ksize=3, scale=1 / 8)
        gradient = cv2.blur(cv2.magnitude(dx, dy), (19, 19))
        fine_detail = cv2.blur(np.abs(gray - soft), (19, 19))
        self.texture = np.clip(np.maximum((fine_detail - 4) / 4, (gradient - 4) / 7), 0, 1)

    @staticmethod
    def _facial_features(face):
        x, y, w, h = face[:4]
        if len(face) >= 14:
            eye_a, eye_b, nose, lip_a, lip_b = face[4:14].reshape(5, 2)
            mouth = (lip_a + lip_b) / 2
            mouth_width = max(abs(lip_a[0] - lip_b[0]), 0.22 * w)
        else:
            eye_a, eye_b = (x + 0.30 * w, y + 0.38 * h), (x + 0.70 * w, y + 0.38 * h)
            nose, mouth = (x + 0.50 * w, y + 0.55 * h), (x + 0.50 * w, y + 0.76 * h)
            mouth_width = 0.42 * w
        return (
            ((eye_a[0], eye_a[1] - 0.045 * h), (0.22 * w, 0.15 * h)),
            ((eye_b[0], eye_b[1] - 0.045 * h), (0.22 * w, 0.15 * h)),
            (nose, (0.16 * w, 0.13 * h)),
            (mouth, (0.5 * mouth_width + 0.09 * w, 0.13 * h)),
        )

    def protect_features(self, items):
        """Keep candidates touching sensitive facial features available for review."""
        sx, sy = self.factors
        for item in items:
            left, top = item.x * sx, item.y * sy
            right, bottom = (item.x + item.width) * sx, (item.y + item.height) * sy
            for (cx, cy), (rx, ry) in self.features:
                # Include the narrow padding applied around a repair contour.
                rx, ry = rx + 3 * sx, ry + 3 * sy
                px, py = min(max(cx, left), right), min(max(cy, top), bottom)
                if ((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2 > 1:
                    continue
                if item.contour is not None:
                    # Check the actual filled shape in ellipse coordinates, so a
                    # curved fiber's empty bounding box cannot protect an eye.
                    factors = np.array((sx, sy)) / (rx, ry)
                    center = np.array((cx, cy)) / (rx, ry)
                    contour = (item.contour * factors - center).astype(np.float32)
                    if cv2.pointPolygonTest(contour, (0, 0), True) < -1:
                        continue
                    if any(cv2.pointPolygonTest((hole * factors - center).astype(np.float32),
                                                (0, 0), True) > 1 for hole in item.holes):
                        continue
                item.face_protected = True
                item.selected = False
                break

    def forehead_regions(self):
        """Look for coherent narrow fibers above the eyes at original resolution."""
        sx, sy = self.factors
        for x, y, width, height in self.faces:
            yield (round((x + 0.04 * width) / sx), round((y - 0.15 * height) / sy),
                   round((x + 0.96 * width) / sx), round((y + 0.35 * height) / sy))

    def within_exposure(self, item):
        sx, sy = self.factors
        if self.exposure:
            low = (item.x * sx, item.y * sy)
            high = ((item.x + item.width - 1) * sx, (item.y + item.height - 1) * sy)
            for a, b, offset in self.exposure:
                worst = a * (low[0] if a >= 0 else high[0]) + b * (low[1] if b >= 0 else high[1])
                if worst + offset >= 0:
                    continue
                if item.contour is None or np.any(item.contour.reshape(-1, 2) @ (a * sx, b * sy) + offset < 0):
                    return False
            return True
        point = (round(item.center_x * sx), round(item.center_y * sy))
        if item.dust_type not in ("fiber", "scratch", "infrared"):
            return not any((point[axis] >= boundary) == bottom for axis, bottom, boundary in self.margins)
        low = (item.x * sx, item.y * sy)
        high = ((item.x + item.width - 1) * sx, (item.y + item.height - 1) * sy)
        return not any((high[axis] >= boundary if bottom else low[axis] < boundary)
                       for axis, bottom, boundary in self.margins)

    def filter(self, items, min_confidence):
        kept = []
        h, w = self.shape
        sx, sy = self.factors
        for item in items:
            point = (min(w - 1, round(item.center_x * sx)), min(h - 1, round(item.center_y * sy)))
            if not self.within_exposure(item):
                continue
            linear = item.dust_type in ("fiber", "scratch")
            if self.texture is None:
                kept.append(item)
                continue
            x, y = point
            face = any(left - 0.08 * width <= x <= left + 1.08 * width
                       and top - 0.20 * height <= y <= top + height
                       for left, top, width, height in self.faces)
            texture = float(self.texture[y, x])
            if linear:
                cutoff = 0.59 + (0.04 if face else 0.13) * texture
            else:
                cutoff = min_confidence + (0.06 if face else 0.20) * texture
            if item.confidence >= cutoff:
                kept.append(item)
        return kept

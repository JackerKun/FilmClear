"""Display-only layers for spotting dust and checking RGB/IR alignment."""

import cv2
import numpy as np

from .scan_io import display_pixels


def dust_view(image, contrast):
    pixels = display_pixels(image)
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY) if pixels.ndim == 3 else pixels
    background = cv2.GaussianBlur(gray, (0, 0), 5)
    gain = max(1.0, contrast / 9)
    return cv2.addWeighted(gray, -gain, background, gain, 127)


def infrared_view(image, infrared, *, overlay=False):
    sample = infrared[::max(1, infrared.shape[0] // 512), ::max(1, infrared.shape[1] // 512)]
    lo, hi = np.percentile(sample, (1, 99))
    if hi <= lo:
        lo, hi = float(infrared.min()), float(infrared.max())
    gain = 255 / max(1, hi - lo)
    visible = cv2.addWeighted(infrared, gain, infrared, 0, -lo * gain, dtype=cv2.CV_8U)
    if not overlay:
        return visible
    tint = cv2.applyColorMap(visible, cv2.COLORMAP_MAGMA)
    return cv2.addWeighted(display_pixels(image), 0.65, tint, 0.35, 0)

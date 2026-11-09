import unittest

import cv2
import numpy as np

from core.dust_remover import DustRemover
from core.native_detector import NativeDustDetector


class NativeDustDetectorTest(unittest.TestCase):
    def test_small_image_preserves_polarity_and_region_coordinates(self):
        image = np.full((77, 112), 122, np.uint8)
        cv2.circle(image, (21, 18), 2, 189, cv2.FILLED)
        cv2.circle(image, (96, 65), 3, 53, cv2.FILLED)
        detector = NativeDustDetector()

        items = detector.detect(image)
        self.assertTrue(any(item.dust_type == "bright" and item.center == (21, 18)
                            for item in items))
        self.assertTrue(any(item.dust_type == "dark" and item.center == (96, 65)
                            for item in items))
        self.assertFalse(any(item.selected for item in items))

        items = detector.detect(image, region=(87, 58, 104, 74), aggressive=True)
        self.assertTrue(any(item.selected and item.center == (96, 65) for item in items))

    def test_thin_fiber_is_continuous_across_tiles(self):
        image = np.full((360, 1250), 120, np.uint8)
        x = np.arange(45, 1210)
        y = 177 + np.rint(10 * np.sin(x / 58))
        line = np.column_stack((x, y)).astype(np.int32)
        cv2.polylines(image, [line], False, 42, 2, cv2.LINE_AA)

        items = NativeDustDetector().detect(image)
        spanning_fibers = [item for item in items
                           if item.dust_type == "fiber" and item.width > 1100]
        self.assertEqual(len(spanning_fibers), 1)
        repaired = DustRemover().remove_single(image, spanning_fibers[0])
        for column in (1023, 1024):
            row = int(y[column - x[0]])
            self.assertGreater(int(repaired[row, column]), int(image[row, column]))

    def test_bright_seam_at_scene_boundary_is_rejected(self):
        image = np.full((380, 420), 70, np.uint8)
        image[:, 230:] = 145
        image[:, 228:231] = 175
        image = cv2.GaussianBlur(image, (5, 5), 1)
        cv2.line(image, (72, 36), (79, 331), 180, 2, cv2.LINE_AA)

        items = NativeDustDetector().detect(image)
        fibers = [item for item in items if item.dust_type == "fiber"]
        self.assertTrue(any(item.height > 250 and item.center_x < 100 for item in fibers))
        self.assertFalse(any(item.height > 100 and 200 < item.center_x < 260
                             for item in fibers))

    def test_repair_preserves_fiber_hole_and_pixels_outside_mask(self):
        image = np.full((234, 270, 3), 120, np.uint8)
        cv2.circle(image, (125, 126), 52, (41, 41, 41), 2, cv2.LINE_AA)
        items = NativeDustDetector().detect(image)
        ring = next(item for item in items if item.dust_type == "fiber" and item.holes)

        remover = DustRemover()
        mask = remover.create_mask(image.shape, [ring], selected_only=False)
        repeated = remover.create_mask(image.shape, [ring] * 8, selected_only=False)
        np.testing.assert_array_equal(mask, repeated)
        self.assertEqual(mask[126, 125], 0)

        repaired = remover.remove_single(image, ring)
        np.testing.assert_array_equal(repaired[mask == 0], image[mask == 0])
        self.assertGreater(int(repaired[126, 177, 0]), int(image[126, 177, 0]))

import unittest

import cv2
import numpy as np

from core.frame_crop import apply_crop, crop_from_edges, crop_plan, detect_frame


class FrameCropTest(unittest.TestCase):
    def test_rotated_exposure_is_detected_and_manually_rotated_at_native_depth(self):
        image = np.full((720, 1080, 3), 7, np.uint8)
        image[60:660, 88:992] = (119, 148, 178)
        image[235:275, 320:395] = (65, 95, 145)
        matrix = cv2.getRotationMatrix2D((540, 360), 2.4, 1)
        tilted = cv2.warpAffine(image, matrix, (1080, 720))
        corners = np.array([[[88, 60], [991, 60], [991, 659], [88, 659]]], np.float32)
        expected = cv2.transform(corners, matrix)[0]
        detected = detect_frame(tilted)
        self.assertIsNotNone(detected)
        self.assertEqual(detected["detected_edges"], 4)
        self.assertLess(np.max(np.linalg.norm(np.asarray(detected["corners"]) - expected, axis=1)), 4)
        self.assertEqual(crop_from_edges(tilted.shape, detected["corners"])["rotation"], 0)
        settings = crop_from_edges(tilted.shape, detected["corners"], rotation=2.4)
        _, transform, _, _ = crop_plan(tilted.shape, settings)
        np.testing.assert_allclose(transform[:2, :2].T @ transform[:2, :2], np.eye(2), atol=1e-10)
        source = tilted.astype(np.uint16) * 257
        tight = apply_crop(source, settings)
        retained = dict(settings, border_percent=2)
        with_border = apply_crop(source, retained)
        self.assertEqual(tight.dtype, np.uint16)
        self.assertEqual(with_border.dtype, np.uint16)
        self.assertAlmostEqual(with_border.shape[1] / tight.shape[1], 1.04, delta=0.003)
        self.assertAlmostEqual(crop_plan(source.shape, retained)[3], 2.0)
        self.assertGreater(np.mean(tight[4:-4, 4:-4, 0] > 20000), 0.99)
        self.assertLess(np.mean(with_border[:, 3, 0]), 5000)

    def test_single_dark_scene_edge_does_not_offer_a_crop(self):
        image = np.full((320, 480, 3), 105, np.uint8)
        image[:38] = 5
        self.assertIsNone(detect_frame(image))

    def test_cut_off_sprocket_rows_allow_a_safe_estimate_for_a_dark_frame(self):
        image = np.full((500, 740, 3), 8, np.uint8)
        for x in range(30, 730, 88):
            image[:36, x:x + 35] = 0
            image[-38:, x:x + 35] = 0
        image[100:420, 90:550] = 11
        settings = detect_frame(image)
        self.assertIsNotNone(settings)
        self.assertEqual(settings["detected_edges"], 0)
        corners = np.asarray(settings["corners"])
        self.assertTrue(np.all((36 <= corners[:2, 1]) & (corners[:2, 1] < 42)))
        self.assertTrue(np.all((455 < corners[2:, 1]) & (corners[2:, 1] < 462)))

    def test_exact_rectangular_crop_preserves_pixels(self):
        image = np.arange(140 * 180 * 3, dtype=np.uint16).reshape(140, 180, 3)
        settings = {"corners": [[20, 30], [155, 30], [155, 119], [20, 119]], "border_percent": 0}
        np.testing.assert_array_equal(apply_crop(image, settings), image[30:120, 20:156])

    def test_xpan_preset_uses_portrait_ratio_when_the_scan_is_vertical(self):
        settings = crop_from_edges((930, 400), aspect="xpan")
        _, _, (width, height), _ = crop_plan((930, 400), settings)
        self.assertAlmostEqual(width / height, 24 / 65, delta=0.002)

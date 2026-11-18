import unittest

import cv2
import numpy as np

from core.line_review import review_scene_lines
from core.native_detector import NativeDustDetector


class LineReviewTest(unittest.TestCase):
    @staticmethod
    def candidate(mask, center):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return NativeDustDetector._item(contours[0], *center, "fiber", 0.73)

    def test_continuation_across_background_is_deferred(self):
        image = np.full((160, 320), 155, np.uint8)
        image[:, :107] = 65
        image[78:83, 35:220] = 108
        mask = np.zeros_like(image)
        mask[78:83, 110:220] = 255
        wire = self.candidate(mask, (165, 80))
        isolated = np.full_like(image, 155)
        isolated[mask != 0] = 108
        scratch = self.candidate(mask, (165, 80))
        review_scene_lines(image, [wire])
        review_scene_lines(isolated, [scratch])
        self.assertTrue(wire.scene_line)
        self.assertFalse(wire.selected)
        self.assertFalse(scratch.scene_line)
        self.assertTrue(scratch.selected)

    def test_junctions_and_repeated_parallel_lines_are_deferred(self):
        image = np.full((200, 320), 155, np.uint8)
        branch = np.zeros_like(image)
        cv2.line(branch, (65, 60), (230, 60), 255, 4)
        cv2.line(branch, (122, 60), (158, 83), 255, 4)
        repeat = np.zeros_like(image)
        cv2.line(repeat, (65, 142), (230, 142), 255, 4)
        for offset in (-20, 20):
            cv2.line(image, (60, 142 + offset), (235, 142 + offset), 95, 4)
        image[(branch | repeat) > 0] = 95
        items = [self.candidate(branch, (125, 60)), self.candidate(repeat, (145, 142))]
        review_scene_lines(image, items)
        self.assertTrue(all(item.scene_line and not item.selected for item in items))

    def test_whole_photo_filters_matched_lines_and_facial_outlines(self):
        image = np.full((410, 600), 155, np.uint8)
        items = []
        for start, end in (((160, 190), (162, 270)), ((273, 189), (269, 272)),
                           ((345, 282), (392, 275)), ((365, 138), (412, 144))):
            mask = np.zeros_like(image)
            cv2.line(mask, start, end, 255, 4)
            image[mask != 0] = 95
            items.append(self.candidate(mask, tuple((a + b) // 2 for a, b in zip(start, end))))
        items[-1].face_protected = True
        kept = review_scene_lines(image, items, filter_structures=True)
        self.assertEqual(kept, [items[2]])
        self.assertTrue(kept[0].selected)
        self.assertIn(items[-1], review_scene_lines(image, items))

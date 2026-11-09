import tempfile
import unittest
import json
from html import unescape
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms
import tifffile

from core.batch_export import ExportEntry, SOURCE_FORMAT, export_batch
from core.dust_item import DustItem
from core.dust_remover import DustRemover
from core.inspection import dust_view, infrared_view
from core.repeated_defects import repeated_defects
from core.review_report import create_review_report
from core.scan_detection import detect_scan, refine_candidate
from core.scan_io import signature


class ProfessionalWorkflowTest(unittest.TestCase):
    def test_review_report_escapes_names_and_does_not_overwrite_previous_report(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "图 & 原片.png"
            target = Path(directory) / "图 & 导出.png"
            Image.new("RGB", (20, 16), "#bb9280").save(source)
            Image.new("RGB", (20, 16), "#bd9381").save(target)
            entry = ExportEntry(source, signature(source), (), ())
            arguments = ([entry], [(source, 0, target)], directory, "edited",
                         lambda *_: None, lambda: False)
            first = create_review_report(*arguments)
            second = create_review_report(*arguments)
            self.assertNotEqual(first, second)
            report = first.read_text(encoding="utf-8")
            self.assertIn("图 &amp; 原片.png", report)
            self.assertIn("%26", report)
            self.assertEqual(report.count("data:image/jpeg;base64,"), 2)

    def test_review_report_separates_patched_selected_and_pending_defects(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "original.png"
            target = Path(directory) / "result.png"
            Image.new("RGB", (100, 80), "#a28973").save(source)
            Image.new("RGB", (100, 80), "#a38973").save(target)
            selected = DustItem(1, 10, 12, 5, 5, 25, 12, 14, "bright")
            pending = DustItem(2, 66, 53, 4, 4, 16, 68, 55, "dark", selected=False)
            positions = np.array([y * 100 + x for y in range(23, 27) for x in range(40, 45)])
            entry = ExportEntry(source, signature(source), ((1, positions, ()),),
                                (selected,), 1, (selected, pending))
            report = create_review_report([entry], [(source, 1, target)], directory, "repair",
                                          lambda *_: None, lambda: False)
            text = report.read_text(encoding="utf-8")
            data = json.loads(unescape(text.split("<template>", 1)[1].split("</template>", 1)[0]))
            self.assertEqual([mark["status"] for mark in data], ["applied", "applied", "pending"])
            self.assertEqual(data[0]["box"], [40, 23, 5, 4])
            self.assertEqual([len(mark["images"]) for mark in data], [2, 2, 2])
            self.assertIn("script-src 'sha256-", text)

    def test_repaired_export_keeps_rgb_icc_and_exif(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
            source = Path(directory) / "scan.jpg"
            rgb = Image.new("RGB", (24, 16), "#c59b71")
            tags = rgb.getexif()
            tags[271] = "Film Scanner"
            tags[274] = 6
            rgb.save(source, icc_profile=profile, exif=tags)
            entry = ExportEntry(source, None, (), ())
            for suffix in (SOURCE_FORMAT, ".png", ".tiff"):
                with self.subTest(suffix=suffix):
                    saved, errors = export_batch([entry], directory, suffix, "repair", {},
                                                 lambda *_: None, lambda: False)
                    self.assertFalse(errors)
                    with Image.open(saved[0][2]) as output:
                        self.assertEqual(output.info["icc_profile"], profile)
                        self.assertEqual(output.getexif()[271], "Film Scanner")
                        self.assertIn(output.getexif().get(274), (None, 1))
                        self.assertEqual(output.size, (16, 24))

    def test_sixteen_bit_exports_keep_profile_and_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
            source = Path(directory) / "scan.tiff"
            rgb = np.arange(21 * 16 * 3, dtype=np.uint16).reshape(21, 16, 3) * 59
            tifffile.imwrite(source, rgb, photometric="rgb",
                             extratags=[(34675, "B", len(profile), profile, True)])
            for suffix in (".tiff", ".png"):
                with self.subTest(suffix=suffix):
                    saved, errors = export_batch([ExportEntry(source, None, (), ())], directory,
                                                 suffix, "repair", {}, lambda *_: None, lambda: False)
                    self.assertFalse(errors)
                    with Image.open(saved[0][2]) as output:
                        self.assertEqual(output.info["icc_profile"], profile)
                    pixels = cv2.imread(str(saved[0][2]), cv2.IMREAD_UNCHANGED)
                    np.testing.assert_array_equal(pixels[..., ::-1], rgb)

    def test_grain_preserves_unmasked_pixels_and_increases_local_texture(self):
        random = np.random.default_rng(42)
        clean = np.rint(127 + random.normal(0, 8, (100, 140, 3))).astype(np.uint8)
        mask = np.zeros(clean.shape[:2], np.uint8)
        cv2.ellipse(mask, (70, 50), (10, 22), 15, 0, 360, 255, cv2.FILLED)
        damaged = clean.copy()
        damaged[mask != 0] = 245
        inner = cv2.erode(mask, np.ones((7, 7), np.uint8)) != 0
        energies = []
        for method in ("ns", "grain"):
            repaired = DustRemover(method).apply_mask(damaged, mask)
            np.testing.assert_array_equal(repaired[mask == 0], clean[mask == 0])
            values = repaired.astype(np.float32)
            energies.append((values - cv2.GaussianBlur(values, (0, 0), 1.2))[inner].std())
        self.assertGreater(energies[1], energies[0] * 2)

    def test_display_layers_preserve_rgb_and_ir_arrays(self):
        image = np.full((48, 64, 3), 44000, np.uint16)
        image[15, 19] = 64000
        plane = np.full(image.shape[:2], 12000, np.uint16)
        plane[12:20, 14:20] = 4000
        rgb_before, ir_before = image.copy(), plane.copy()
        self.assertEqual(dust_view(image, 40).shape, plane.shape)
        self.assertEqual(infrared_view(image, plane).dtype, np.uint8)
        self.assertEqual(infrared_view(image, plane, overlay=True).shape, image.shape)
        np.testing.assert_array_equal(image, rgb_before)
        np.testing.assert_array_equal(plane, ir_before)

    def test_repeated_candidates_require_three_comparable_photos(self):
        def item(x, y, kind="bright"):
            return DustItem(1, x - 3, y - 3, 7, 7, 35, x, y, kind)

        photos = [(str(index), (2000, 1400), [item(300 + index, 550), item(75 * index, 40)])
                  for index in range(3)]
        photos.append(("unrelated", (2000, 900), [item(300, 354)]))
        groups = repeated_defects(photos)
        self.assertEqual(len(groups), 1)
        self.assertEqual([path for path, _ in groups[0]], ["0", "1", "2"])

    def test_quick_candidate_can_be_retraced_on_original_pixels(self):
        image = np.full((750, 3200, 3), 110, np.uint8)
        cv2.circle(image, (1700, 300), 5, (240, 240, 240), cv2.FILLED)
        settings = dict(threshold=20, min_area=3, min_confidence=0.34)
        item = min(detect_scan(image, settings).items, key=lambda item: abs(item.center_x - 1700))
        self.assertTrue(item.preview)
        result = refine_candidate(image, item, settings)
        self.assertIsNotNone(result)
        self.assertFalse(result.preview)
        self.assertGreaterEqual(cv2.pointPolygonTest(result.contour, (1700, 300), False), 0)

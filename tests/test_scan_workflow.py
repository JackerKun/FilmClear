from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import tifffile

from core.batch_export import SOURCE_FORMAT, ExportEntry, export_batch
from core.dust_remover import DustRemover
from core.scene_focus import SceneFocus
from core.scan_detection import detect_brush, detect_scan
from core.scan_io import read_ir_plane, read_scan, signature


PARAMS = dict(threshold=20, min_area=3, min_confidence=0.34)
REPAIR = dict(method="telea", inpaint_radius=3)
DEMO = Path(__file__).resolve().parents[1] / "demo"


class ScanWorkflowTest(unittest.TestCase):
    def demo_path(self, name):
        path = DEMO / name
        if not path.is_file():
            self.skipTest(f"Local scan fixture is unavailable: {name}")
        return path

    def test_vuescan_rgbi_and_ordinary_alpha(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "raw.tiff"
            original = np.full((128, 176, 4), (27000, 24000, 18000, 50000), np.uint16)
            original[50:57, 74:82, 3] = 24000
            original[50:57, 74:82, :3] = 6000
            tifffile.imwrite(source, original, photometric="rgb", extrasamples="UNASSALPHA",
                             software="VueScan")
            scan = read_scan(source)
            self.assertEqual(scan.ir_source, "TIFF RGBI")
            self.assertEqual(scan.image.dtype, np.uint16)
            np.testing.assert_array_equal(scan.infrared, original[..., 3])
            np.testing.assert_array_equal(scan.image, original[..., 2::-1])
            defects = detect_scan(scan.image, PARAMS, infrared=scan.infrared)
            self.assertEqual(defects.source, "红外")
            self.assertEqual(len(defects.items), 1)
            x, y, w, h = defects.items[0].bbox
            self.assertLessEqual(x, 74)
            self.assertLessEqual(y, 50)
            self.assertGreaterEqual(x + w, 82)
            self.assertGreaterEqual(y + h, 57)
            self.assertLessEqual(w * h, 100)

            ordinary = Path(directory) / "graphic.tiff"
            tifffile.imwrite(ordinary, original, photometric="rgb", extrasamples="UNASSALPHA",
                             software="Image editor")
            self.assertIsNone(read_scan(ordinary).infrared)
            np.testing.assert_array_equal(
                read_ir_plane(ordinary, original.shape[:2], embedded=True), original[..., 3]
            )

            compressed = Path(directory) / "compressed.tiff"
            bgri = np.ascontiguousarray(original[..., (2, 1, 0, 3)])
            cv2.imwrite(str(compressed), bgri, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
            np.testing.assert_array_equal(read_scan(compressed).infrared, original[..., 3])

            folder = Path(directory) / "output"
            entry = ExportEntry(source, signature(source), (), tuple(defects.items))
            records = []
            saved, errors = export_batch([entry], folder, ".tiff", "repair", REPAIR,
                                         lambda _, done, total: records.append((done, total)),
                                         lambda: False)
            self.assertFalse(errors)
            self.assertEqual(records, [(0, 1), (1, 1)])
            repaired = read_scan(saved[0][2]).image
            self.assertEqual(repaired.dtype, np.uint16)
            self.assertEqual(repaired.shape, scan.image.shape)
            np.testing.assert_array_equal(repaired[:45], scan.image[:45])
            self.assertGreater(int(repaired[54, 77, 0]), int(scan.image[54, 77, 0]))

            saved, errors = export_batch([entry, entry], folder, ".png", "original",
                                         REPAIR, lambda *_: None, lambda: False)
            self.assertFalse(errors)
            self.assertNotEqual(saved[0][2], saved[1][2])
            for _, _, path in saved:
                self.assertEqual(path.read_bytes(), source.read_bytes())

    def test_sidecar_and_invalid_infrared_fallback(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "frame.png"
            image = np.full((99, 152, 3), 120, np.uint8)
            cv2.circle(image, (62, 30), 3, (215, 215, 215), cv2.FILLED)
            cv2.imwrite(str(source), image)
            sidecar = source.with_name("FRAME_IR.TIF")
            cv2.imwrite(str(sidecar), np.zeros(image.shape[:2], np.uint16))
            scan = read_scan(source)
            self.assertEqual(scan.ir_source, sidecar.name)
            result = detect_scan(scan.image, PARAMS, infrared=scan.infrared)
            self.assertIn("红外信号不可用", result.source)
            self.assertTrue(any(item.center == (62, 30) for item in result.items))
            cv2.imwrite(str(sidecar), np.full((40, 60), 45000, np.uint16))
            wrong = read_scan(source)
            self.assertIsNone(wrong.infrared)
            self.assertIn("尺寸", wrong.ir_notice)

    def test_quick_contours_are_mapped_to_the_original(self):
        image = np.full((900, 3100, 3), 100, np.uint8)
        cv2.circle(image, (2285, 574), 6, (205, 205, 205), cv2.FILLED)
        found = detect_scan(image, PARAMS)
        self.assertEqual(found.source, "RGB 快速预览")
        item = min(found.items, key=lambda item: abs(item.center_x - 2285))
        self.assertLessEqual(abs(item.center_x - 2285), 2)
        self.assertLessEqual(abs(item.center_y - 574), 2)
        self.assertGreaterEqual(cv2.pointPolygonTest(item.contour, (2285, 574), False), 0)

    def test_short_scratches_and_curled_fibers_from_demo(self):
        cases = (
            ("2026-10-29-0008.jpg", (1400, 175, 1560, 280), (1482, 223), "scratch"),
            ("2026-10-29-0008.jpg", (1975, 410, 2110, 600), (2042, 505), "scratch"),
            ("2026-10-30-0006.jpg", (2410, 1520, 2545, 1680), (2491, 1590), "fiber"),
            ("2026-10-30-0006.jpg", (2535, 1410, 2640, 1520), (2601, 1457), "fiber"),
        )
        images = {name: cv2.imread(str(self.demo_path(name))) for name in {case[0] for case in cases}}
        for name, (x, y, x2, y2), center, kind in cases:
            with self.subTest(photo=name, center=center):
                candidates = detect_scan(images[name][y:y2, x:x2], PARAMS).items
                self.assertTrue(any(item.dust_type == kind and
                                    np.hypot(item.center_x + x - center[0],
                                             item.center_y + y - center[1]) < 12
                                    for item in candidates))

    def test_broad_irregular_surface_defect(self):
        image = np.full((220, 240, 3), 115, np.uint8)
        points = np.array([(90, 85), (110, 84), (117, 98), (108, 101),
                           (111, 120), (92, 113), (85, 105)], np.int32)
        cv2.fillPoly(image, [points], (200, 200, 200))
        items = detect_scan(image, PARAMS).items
        self.assertTrue(any(item.dust_type == "damage"
                            and cv2.pointPolygonTest(item.contour, (99, 101), False) >= 0
                            for item in items))

    def test_forehead_fiber_is_traced_in_original_coordinates(self):
        image = cv2.imread(str(self.demo_path("2026-10-29-0007.jpg")))
        region = (2481, 375, 2560, 470)
        target = [(2502, 461), (2513, 455), (2520, 430), (2527, 400), (2539, 385), (2550, 384)]
        for options in ({"region": region}, {}):
            with self.subTest(options=options):
                items = detect_scan(image, PARAMS, **options).items
                if options:
                    self.assertEqual(len(items), 1)
                fiber = next(item for item in items if item.dust_type == "fiber" and all(
                    cv2.pointPolygonTest(item.contour, point, True) >= -1 for point in target
                ))
                mask = DustRemover().create_mask(image.shape, [fiber])
                self.assertTrue(all(mask[y, x] for x, y in target))
                self.assertFalse(mask[409, 2490])  # Neighboring hairline.
                self.assertLess(cv2.countNonZero(mask), 0.30 * 79 * 95)
        rough = [(2492, 461), (2509, 455), (2516, 435), (2520, 401), (2539, 380), (2555, 379)]
        items = detect_brush(image, PARAMS, rough, 14).items
        self.assertEqual(len(items), 1)
        mask = DustRemover().create_mask(image.shape, items)
        self.assertTrue(all(mask[y, x] for x, y in target))
        self.assertFalse(mask[409, 2490])

    def test_brush_finds_bright_spot_inside_a_rough_loop(self):
        image = cv2.imread(str(self.demo_path("2026-10-28-0001.jpg")))
        gesture = [(1848, 6612), (1874, 6606), (1900, 6625), (1903, 6653),
                   (1853, 6651), (1848, 6612)]
        items = detect_brush(image, PARAMS, gesture, 15).items
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.dust_type, "bright")
        self.assertGreaterEqual(cv2.pointPolygonTest(item.contour, (1879, 6635), False), 0)
        self.assertLess(cv2.pointPolygonTest(item.contour, gesture[0], False), 0)
        rough_pass = [(1854, 6618), (1884, 6618)]
        near = detect_brush(image, PARAMS, rough_pass, 28).items
        self.assertEqual([item.center for item in near], [item.center])

    def test_open_brush_scopes_candidates_to_its_path(self):
        image = np.full((160, 180, 3), 110, np.uint8)
        for center in ((27, 60), (100, 60)):
            cv2.circle(image, center, 4, (235, 235, 235), cv2.FILLED)
        items = detect_brush(image, PARAMS, [(25, 35), (25, 130), (135, 130)], 10).items
        self.assertEqual([item.center for item in items], [(27, 60)])
        self.assertEqual(detect_brush(image, PARAMS, [(80, 110)], 9).items, [])

    def test_subject_first_protects_perforations_and_reduces_landscape_candidates(self):
        image = cv2.imread(str(self.demo_path("2026-10-29-0016.jpg")))
        focus = SceneFocus(image)
        self.assertEqual({(axis, bottom) for axis, bottom, _ in focus.margins},
                         {(1, False), (1, True)})
        result = detect_scan(image, PARAMS)
        self.assertGreater(len(result.items), 0)
        self.assertLess(len(result.items), 75)
        self.assertTrue(all(650 < item.center_y < 3600 for item in result.items))

    def test_real_vuescan_fixture(self):
        scan = read_scan(self.demo_path("VueScan-RGBI-raw0048-crop.tif"))
        self.assertEqual(scan.image.shape, (1000, 1300, 3))
        self.assertEqual(scan.infrared.dtype, np.uint16)
        result = detect_scan(scan.image, PARAMS, infrared=scan.infrared)
        self.assertEqual(result.source, "红外")
        self.assertEqual([item.id for item in result.items], list(range(1, len(result.items) + 1)))
        # Three plainly visible defects in the sky, plus a bound on
        # false responses to the rock texture visible in the IR plane.
        for x, y in ((928, 118), (940, 544), (994, 781)):
            self.assertTrue(any(np.hypot(i.center_x - x, i.center_y - y) < 10
                                for i in result.items))
        self.assertLess(len(result.items), 150)

    def test_repaired_batch_can_keep_each_original_extension(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            image = np.full((50, 60, 3), 109, np.uint8)
            inputs = [folder / name for name in ("one.jpeg", "two.png")]
            for path in inputs:
                cv2.imwrite(str(path), image)
            entries = [ExportEntry(path, signature(path), (), ()) for path in inputs]
            saved, errors = export_batch(entries, folder, SOURCE_FORMAT, "edited",
                                         REPAIR, lambda *_: None, lambda: False)
            self.assertFalse(errors)
            self.assertEqual([item[2].name for item in saved], ["one_clean.jpeg", "two_clean.png"])
            for _, _, output in saved:
                self.assertEqual(cv2.imread(str(output)).shape, image.shape)

    def test_stale_source_reports_error_and_continues_batch(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            first, second = folder / "first.png", folder / "second.png"
            image = np.full((50, 60, 3), 100, np.uint8)
            cv2.imwrite(str(first), image)
            cv2.imwrite(str(second), image)
            entries = [ExportEntry(first, (1, 1), (), ()), ExportEntry(second, signature(second), (), ())]
            saved, errors = export_batch(entries, folder, ".png", "edited", REPAIR,
                                         lambda *_: None, lambda: False)
            self.assertEqual([item[0] for item in saved], [second])
            self.assertEqual([item[0] for item in errors], [first])
            self.assertTrue(np.array_equal(cv2.imread(str(saved[0][2])), image))

            done_count = []
            saved, errors = export_batch(
                entries[::-1], folder, ".png", "edited", REPAIR,
                lambda _, done, total: done_count.append(done), lambda: 1 in done_count
            )
            self.assertEqual([item[0] for item in saved], [second])
            self.assertEqual(done_count, [0, 1])
            self.assertFalse(errors)

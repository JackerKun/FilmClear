import json
import sqlite3
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from core.dust_item import DustItem
from core.fiber_learning import (
    FiberExample, FiberLearningStore, FiberModel, candidate_key, combine_fiber_models, fiber_features,
    read_fiber_model, train_fiber_model, write_fiber_model,
)
from core.scan_detection import detect_brush, detect_scan
from core.scan_io import read_scan


def line(item_id, x, y):
    contour = np.array([[[x, y]], [[x + 55, y + 6]], [[x + 56, y + 10]], [[x, y + 4]]], np.int32)
    return DustItem(item_id, x, y, 57, 11, 225, x + 26, y + 5, dust_type="scratch", contour=contour)


def sample_examples():
    return [
        FiberExample(f"private-photo-{group}-{label}-{sample}", f"private-roll-{group}", label, True,
                     (2 * label - 1 + (sample - 3) / 50,) * 14)
        for group in range(3) for label in (0, 1) for sample in range(6)
    ]


class FiberLearningTest(unittest.TestCase):
    def test_roll_validation_and_shareable_model_roundtrip(self):
        model = train_fiber_model(sample_examples())
        self.assertEqual(model.metrics["validation"], "roll")
        self.assertEqual(model.metrics["evaluated"], 36)
        self.assertEqual(model.metrics["scene_deferred"], 18)
        self.assertEqual(model.metrics["defects_deferred"], 0)
        photos = [replace(row, roll="one roll") for row in sample_examples()]
        photo_validation = train_fiber_model(photos)
        self.assertEqual(photo_validation.metrics["validation"], "photo")
        one_photo = train_fiber_model([replace(row, photo="p", roll="r") for row in photos])
        self.assertEqual(one_photo.metrics["validation"], "none")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fiber-review.json"
            write_fiber_model(path, model)
            self.assertNotIn("private-", path.read_text())
            self.assertEqual(read_fiber_model(path), model)
            bad = json.loads(path.read_text())
            bad["weights"][0] = float("nan")
            with self.assertRaises(ValueError):
                FiberModel.decode(json.dumps(bad))

    @patch("core.fiber_learning._BUNDLED_MODEL", Path())
    def test_examples_and_models_are_saved_with_independent_activation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            store = FiberLearningStore(path)
            store.label("photo", "roll A", "line", 1, True, (0,) * 14)
            store.label("photo", "roll A", "line", 0, True, (1,) * 14)
            store.set_roll("photo", "roll B")
            self.assertEqual(store.photo_labels("photo"), ("roll B", {"line": 0}))
            self.assertEqual(store.counts(), (0, 1, 1))
            reopened = FiberLearningStore(path)
            self.assertEqual(reopened.examples(), [FiberExample("photo", "roll B", 0, True, (1,) * 14)])
            model = train_fiber_model(sample_examples())
            first = reopened.save_trained(model)
            self.assertIsNone(reopened.active_model())
            reopened.set_enabled([first], True)
            newer = replace(model, weights=tuple(weight * 2 for weight in model.weights))
            second, = reopened.add_models([("模型 B", "/tmp/model-b.json", newer)])
            reopened.set_enabled([second], True)
            self.assertEqual(reopened.active_model(), combine_fiber_models((model, newer)))
            reopened.set_enabled([first], False)
            self.assertEqual(FiberLearningStore(path).active_model(), newer)
            self.assertEqual(reopened.library()[0].source, "/tmp/model-b.json")
            reopened.remove_label("photo", "line")
            self.assertEqual(reopened.counts(), (0, 0, 0))

    @patch("core.fiber_learning._BUNDLED_MODEL", Path())
    def test_legacy_models_migrate_once_with_the_prior_activation_setting(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), TemporaryDirectory() as directory:
                store = FiberLearningStore(Path(directory) / "config.json")
                model = train_fiber_model(sample_examples())
                with sqlite3.connect(store.path) as db:
                    db.execute("CREATE TABLE models (slot TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                    db.executemany("INSERT INTO models VALUES (?, ?)",
                                   [(slot, model.encode()) for slot in ("active", "trained")])
                self.assertEqual(store.active_model(legacy_enabled=enabled), model if enabled else None)
                entry, = store.library()
                self.assertEqual(entry.enabled, enabled)
                store.set_enabled([entry.id], not enabled)
                self.assertEqual(store.active_model(legacy_enabled=enabled), None if enabled else model)

    def test_removing_bundled_model_persists_after_reopening(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            bundled = folder / "bundled.json"
            model = train_fiber_model(sample_examples())
            write_fiber_model(bundled, model)
            with patch("core.fiber_learning._BUNDLED_MODEL", bundled):
                store = FiberLearningStore(folder / "config.json")
                entry, = store.library()
                self.assertTrue(entry.enabled)
                store.remove_models([entry.id])
                reopened = FiberLearningStore(folder / "config.json")
                self.assertEqual(reopened.library(), [])
                self.assertIsNone(reopened.active_model())
                self.assertEqual(read_fiber_model(bundled), model)
                reopened.add_models([("重新导入", str(bundled), model)])
                self.assertEqual(len(reopened.library()), 1)
                self.assertIsNone(reopened.active_model())

    def test_merged_models_keep_distinct_templates_and_export_their_outlines(self):
        with TemporaryDirectory() as directory:
            store = FiberLearningStore(Path(directory) / "config.json")
            trace = line(1, 25, 30)
            store.label_many("photo", "roll", 1, [("line", False, (1.,) * 14, trace)])
            first = train_fiber_model(store.examples())
            self.assertGreaterEqual(len(first.outlines[0]), 3)
            second = train_fiber_model([FiberExample("p", "r", 1, False, (3.,) * 14, traced=True)])
            merged = combine_fiber_models((first, second))
            self.assertTrue(merged.recognizes((1.,) * 14))
            self.assertTrue(merged.recognizes((3.,) * 14))
            self.assertFalse(merged.recognizes((2.,) * 14))
            self.assertFalse(merged.defers((2.,) * 14))
            self.assertEqual(combine_fiber_models((merged, second)), merged)
            destination = Path(directory) / "combined.json"
            write_fiber_model(destination, merged)
            self.assertEqual(read_fiber_model(destination), merged)

    def test_model_only_defers_existing_candidates_on_full_rgb_scan(self):
        image = np.full((240, 320, 3), 120, np.uint8)
        scratch = line(1, 110, 60)
        cv2.drawContours(image, [scratch.contour], -1, (90, 90, 90), cv2.FILLED)
        spot = DustItem(2, 240, 130, 5, 5, 20, 242, 132, dust_type="bright")
        model = replace(train_fiber_model(sample_examples()), weights=(0.,) * 14 + (-12.,))
        params = dict(threshold=20, min_area=3, min_confidence=.34)

        def scan(**options):
            with patch("core.scan_detection.NativeDustDetector.detect", return_value=[replace(scratch), replace(spot)]), \
                    patch("core.scan_detection.detect_strokes", return_value=[]):
                return detect_scan(image, params, fiber_model=model, **options).items

        self.assertEqual([(row.dust_type, row.selected) for row in scan()], [("scratch", False), ("bright", True)])
        self.assertTrue(all(row.selected for row in scan(learning_candidates=True)))
        self.assertTrue(all(row.selected for row in scan(region=(90, 40, 200, 110))))

    def test_candidate_features_and_identity_work_on_sixteen_bit_scans(self):
        item = line(1, 20, 32)
        image = np.full((100, 125, 3), 19000, np.uint16)
        cv2.drawContours(image, [item.contour], -1, (45000,) * 3, cv2.FILLED)
        self.assertTrue(np.isfinite(fiber_features(image, item)).all())
        self.assertEqual(candidate_key(item), candidate_key(replace(item, id=40)))
        self.assertNotEqual(candidate_key(item), candidate_key(line(1, 35, 32)))

    def test_brush_sample_recovers_a_faint_outline_missing_from_full_scan(self):
        demo = Path(__file__).resolve().parents[1] / "demo" / "HDR-window-16bit.tif"
        image = read_scan(demo, read_ir=False).image
        params = dict(threshold=20, min_area=3, min_confidence=.34)
        path = [(486, 441), (479, 451), (469, 465), (459, 480)]
        sample, = [item for item in detect_brush(image, params, path, 3, guided=True).items
                   if item.dust_type == "scratch" and item.height >= 39]
        model = FiberModel.decode(train_fiber_model([
            FiberExample("reference", "roll", 1, False, fiber_features(image, sample), traced=True),
        ]).encode())

        # The saved sample comes from the original exposure. Scan different
        # pixel values with the same weak line and adjacent window edge.
        altered = np.rint(image.astype(np.float32) * .88 + 65535 * .07).astype(np.uint16)

        def crossing_candidates(fiber_model):
            return [item for item in detect_scan(altered, params, fiber_model=fiber_model).items
                    if item.dust_type in ("scratch", "fiber") and 445 < item.x < 495
                    and 425 < item.y < 482]

        self.assertEqual(crossing_candidates(None), [])
        found, = crossing_candidates(model)
        self.assertLessEqual(found.y, 445)
        self.assertGreaterEqual(found.y + found.height, 479)
        self.assertTrue(found.learned_fiber)
        self.assertFalse(found.selected)

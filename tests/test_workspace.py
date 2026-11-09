import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from config import Config
from core.native_detector import NativeDustDetector
from ui.main_window import MainWindow


class WorkspaceTest(unittest.TestCase):
    def test_review_and_pixel_history(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            with patch("ui.main_window.Config", lambda: Config(directory)):
                window = MainWindow()
            self.addCleanup(window._stop_workers)
            self.assertTrue(window.btn_marks.isChecked())
            self.assertTrue(window.image_viewer._show_all)
            first, second, third = (Path(directory) / name for name in ("one.png", "two.png", "three.png"))
            image = np.full((180, 200, 3), 116, np.uint8)
            cv2.circle(image, (85, 88), 3, (224, 224, 224), cv2.FILLED)
            cv2.imwrite(str(first), image)
            cv2.imwrite(str(second), image)
            cv2.imwrite(str(third), image)

            def finish_job():
                deadline = time.monotonic() + 5
                while window._job is not None and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.01)
                self.assertIsNone(window._job)

            window.file_panel.load_files([first, second, third])
            finish_job()
            detector = NativeDustDetector(min_confidence=0.34)
            candidates = detector.detect(image)
            dust = min(candidates, key=lambda item: abs(item.center_x - 85))
            window._on_detection_finished(candidates, None)
            self.assertTrue(all(item.selected for item in window._dust_items))
            row = window.dust_panel.dust_list.item(0)
            self.assertEqual(row.checkState().name, "Checked")

            dust.selected = False
            window._dust_items[0].selected = False
            window.file_panel.select_file(str(second))
            finish_job()
            window._on_detection_finished(detector.detect(image), None)
            self.assertTrue(all(item.selected for item in window._dust_items))
            window.file_panel.select_file(str(first))
            finish_job()
            self.assertFalse(window._dust_items[0].selected)
            self.assertEqual(window.dust_panel.dust_list.item(0).checkState().name, "Unchecked")

            initial_transform = window.image_viewer.transform()
            window._on_dust_hovered(dust.id)
            self.assertEqual(window.dust_panel.preview._target, dust.center)
            self.assertEqual(window.image_viewer.transform(), initial_transform)
            window._delete_dust_item(dust.id)
            window._on_detection_finished(detector.detect(image), None)
            self.assertFalse(any(item.center == dust.center for item in window._dust_items))
            window._undo()
            window._undo()
            self.assertTrue(any(item.id == dust.id for item in window._dust_items))

            window._repair_single_dust(dust.id)
            finish_job()
            self.assertFalse(np.array_equal(window._current_image, image))
            healed = window._current_image.copy()
            window.btn_compare.click()
            self.assertFalse(window.image_viewer._before.isNull())
            window._undo()
            np.testing.assert_array_equal(window._current_image, image)
            self.assertTrue(window.image_viewer._before.isNull())
            window._redo()
            np.testing.assert_array_equal(window._current_image, healed)

            window.file_panel.select_file(str(second))
            finish_job()
            self.assertTrue(all(item.selected for item in window._dust_items))
            window.file_panel.select_file(str(first))
            finish_job()
            np.testing.assert_array_equal(window._current_image, healed)
            window._undo()
            np.testing.assert_array_equal(window._current_image, image)
            window._redo()
            np.testing.assert_array_equal(window._current_image, healed)

            output = Path(directory) / "exports"
            window._start_export([str(first), str(second)], output, ".png", "repair")
            finish_job()
            self.assertEqual(window.overlay_bar.value(), 2)
            np.testing.assert_array_equal(cv2.imread(str(output / "one_clean.png")), healed)
            np.testing.assert_array_equal(cv2.imread(str(output / "two_clean.png")), healed)

            window.resize(window.minimumSize())
            window.show()
            window.centralWidget().setSizes([300, 350, 390])
            app.processEvents()
            self.assertTrue(window.image_viewer._show_all)
            window.btn_marks.click()
            self.assertFalse(window.image_viewer._show_all)
            window.btn_marks.click()
            self.assertTrue(window.image_viewer._show_all)
            for button in (*window.edit_buttons.values(), window.btn_marks,
                           window.btn_fit, window.btn_zoom_in, window.btn_zoom_out):
                self.assertFalse(button.icon().isNull())
                self.assertAlmostEqual(button.geometry().center().y(),
                                       window.btn_marks.geometry().center().y(), delta=1)
            for mode, button in window.edit_buttons.items():
                self.assertTrue(button.isVisible())
                self.assertGreaterEqual(button.width(), button.sizeHint().width())
                button.click()
                app.processEvents()
                self.assertIs(window.edit_group.checkedButton(), button)
                self.assertEqual(window.image_viewer._selection_mode, mode)
                self.assertEqual(window.brush_radius.isVisible(), mode == "draw")

            marks = np.full_like(image, 116)
            for center in ((45, 55), (141, 53)):
                cv2.circle(marks, center, 3, (224, 224, 224), cv2.FILLED)
            window._on_detection_finished(detector.detect(marks), None)
            self.assertEqual(len(window._dust_items), 2)
            window.edit_buttons["check"].click()
            window._on_region_selected(39, 47, 52, 62)
            self.assertEqual(sum(item.selected for item in window._dust_items), 1)
            window._on_region_selected(135, 48, 148, 59, additive=True)
            self.assertEqual(sum(item.selected for item in window._dust_items), 2)
            window.dust_panel.btn_ignore_selected.click()
            self.assertEqual(len(window._ignored_items), 2)
            window._undo()
            self.assertEqual(sum(item.selected for item in window._dust_items), 2)
            self.assertEqual(len(window._ignored_items), 0)

            window.edit_buttons["ignore"].click()
            window._on_region_selected(18, 48, 31, 60)
            self.assertEqual(len(window._ignored_items), 0)
            window._on_region_selected(12, 14, 165, 65)
            self.assertEqual(len(window._ignored_items), 2)
            window._undo()
            window.edit_buttons["repair"].click()
            window._on_region_selected(12, 14, 165, 65)
            finish_job()
            self.assertFalse(window._dust_items)
            window._undo()
            self.assertEqual(len(window._dust_items), 2)

            window.btn_compare.click()
            self.assertTrue(window.edit_buttons[None].isChecked())
            app.processEvents()
            self.assertTrue(window.wipe_controls.isVisible())
            for button in window.compare_buttons.values():
                self.assertTrue(button.isVisible())
                self.assertFalse(button.icon().isNull())
                self.assertGreaterEqual(button.width(), button.sizeHint().width())
            window.compare_slider.setValue(34)
            for mode in ("side", "stack"):
                button = window.compare_buttons[mode]
                button.click()
                app.processEvents()
                self.assertIs(window.compare_group.checkedButton(), button)
                self.assertTrue(window.wipe_controls.isHidden())
                original, edited = window.original_viewer, window.image_viewer
                self.assertTrue(original.isVisible())
                self.assertEqual(original.comparison_label, "原片")
                self.assertEqual(edited.comparison_label, "修复后")
                if mode == "side":
                    self.assertLess(original.x(), edited.x())
                else:
                    self.assertLess(original.y(), edited.y())
                original._zoom_to(5.5)
                original.centerOn(QPointF(92, 83))
                app.processEvents()
                self.assertAlmostEqual(original.transform().m11(), edited.transform().m11())
                a = original.mapToScene(original.viewport().rect().center())
                b = edited.mapToScene(edited.viewport().rect().center())
                self.assertLess((a - b).manhattanLength(), 2)
            window.compare_buttons["wipe"].click()
            self.assertTrue(window.wipe_controls.isVisible())
            self.assertTrue(window.original_viewer.isHidden())
            self.assertEqual(window.compare_slider.value(), 34)
            self.assertFalse(window.image_viewer._before.isNull())
            window.btn_compare.click()
            self.assertTrue(window.compare_row.isHidden())
            window.file_panel.select_file(str(third))
            finish_job()
            draw = window.edit_buttons["draw"]
            draw.click()
            viewer = window.image_viewer
            viewer._zoom_to(5.5)
            viewer.centerOn(85, 88)
            app.processEvents()
            center = viewer.viewport().rect().center()
            scroll_before = viewer.horizontalScrollBar().value()
            QTest.keyPress(draw, Qt.Key_Space)
            self.assertEqual(viewer.viewport().cursor().shape(), Qt.OpenHandCursor)
            QTest.mousePress(viewer.viewport(), Qt.LeftButton, pos=center)
            QTest.mouseMove(viewer.viewport(), center + QPoint(45, 0))
            self.assertLess(viewer.horizontalScrollBar().value(), scroll_before)
            QTest.keyRelease(viewer, Qt.Key_Space)
            self.assertEqual(viewer.viewport().cursor().shape(), Qt.CrossCursor)
            QTest.mouseRelease(viewer.viewport(), Qt.LeftButton, pos=center + QPoint(45, 0))
            self.assertIsNone(window._job)
            self.assertTrue(draw.isChecked())
            QTest.mouseClick(viewer.viewport(), Qt.LeftButton,
                             pos=viewer.mapFromScene(QPointF(85, 88)))
            finish_job()
            self.assertEqual(window._dust_items[0].center, (85, 88))
            window.file_panel.select_file(str(first))
            finish_job()
            window.file_panel.select_file(str(third))
            finish_job()
            self.assertEqual(window.dust_panel.dust_list.count(), 1)
            self.assertEqual(window._dust_items[0].dust_type, "bright")
            window.close()

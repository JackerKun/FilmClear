import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QItemSelectionModel, QLocale, QPoint, QPointF, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QDialog, QMenu, QPushButton, QTreeWidget

from config import Config
from core.fiber_learning import (FiberExample, FiberLearningStore, FiberModel, read_fiber_model,
                                 train_fiber_model, write_fiber_model)
from core.frame_crop import apply_crop, crop_plan
from core.dust_item import DustItem
from core.native_detector import NativeDustDetector
from core.scan_detection import detect_scan
from core.scan_io import read_scan
from ui.batch_crop_dialog import BatchCropDialog
from ui.export_dialog import ExportDialog
from ui.fiber_learning_dialog import FiberLabelDialog, FiberModelDialog, FiberTrainingDialog
from ui.image_viewer import RETOUCH_MODES
from ui.crop_dialog import CropDialog
from ui.main_window import MainWindow
from ui.notice_dialog import NoticeDialog


@patch("core.fiber_learning._BUNDLED_MODEL", Path())
class WorkspaceTest(unittest.TestCase):
    def test_numeric_parameters_use_standard_digits_in_chinese_hong_kong_and_macau(self):
        app = QApplication.instance() or QApplication([])
        self.addCleanup(QLocale.setDefault, QLocale())
        for name in ("zh-Hans-HK", "zh-Hans-MO"):
            with self.subTest(locale=name), tempfile.TemporaryDirectory() as directory, \
                    patch("ui.main_window.Config", lambda: Config(directory)):
                QLocale.setDefault(QLocale(name))
                window = MainWindow()
                crop = CropDialog(np.zeros((100, 180, 3), np.uint8), None, window)
                try:
                    fields = window.findChildren(QAbstractSpinBox)
                    self.assertIn(crop.rotation, fields)
                    self.assertIn(crop.border, fields)
                    for field in fields:
                        self.assertEqual(field.locale().name(), "C")
                    radius = window.dust_panel.spin_inpaint_radius
                    self.assertEqual(radius.cleanText(), "3")
                    radius.stepUp()
                    self.assertEqual(radius.cleanText(), "4")
                    crop.border.lineEdit().setText(f"{crop.border.prefix()}2.5{crop.border.suffix()}")
                    crop.border.interpretText()
                    self.assertEqual(crop.border.value(), 2.5)
                    self.assertEqual(QLocale().name(), QLocale(name).name())
                finally:
                    crop.close()
                    window.close()

    def test_batch_crop_can_apply_only_confirmed_frames_and_reset_selected_crops(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory, patch("ui.main_window.Config", lambda: Config(directory)):
            paths = [str(Path(directory).resolve() / name)
                     for name in ("a-full.png", "b-review.png", "c-review.png", "d-skip.png")]
            dark = np.full((300, 460, 3), 8, np.uint8)
            framed = dark.copy()
            framed[30:270, 35:425] = (105, 132, 170)
            for path, image in zip(paths, (framed, dark, dark, dark)):
                cv2.imwrite(path, image)

            checks = {}

            def confirm(crop):
                checks["review_options"] = (crop.border.value(), crop.aspect.currentData())
                crop.rotation.setValue(0.6)
                crop.next.click()
                checks["next"] = crop.title.text()
                crop.previous.click()
                checks["restored_rotation"] = crop.rotation.value()
                crop.apply_button.click()
                checks["after_confirmation"] = crop.title.text()
                crop.apply_button.click()
                checks["returned_at"] = crop.title.text()
                crop.rotation.setValue(1.2)
                return QDialog.Rejected

            def run_dialog(dialog):
                if dialog._results is None:
                    checks["initial"] = dialog.selection()
                    dialog.border.setValue(2)
                    dialog.aspect.setCurrentIndex(dialog.aspect.findData("135"))
                else:
                    checks["before_apply"] = window._crop
                    dialog.select_all.click()
                    checks["before_review"] = dialog.selection()
                    dialog.photos.setCurrentItem(dialog._rows[paths[1]])
                    dialog.review_button.click()
                    checks["after_review"] = dialog.selection()
                    checks["summary"] = dialog.summary.text()
                    dialog.select_confirmed.click()
                    checks["confirmed_only"] = dialog.selection()
                return QDialog.Accepted

            def reset_dialog(dialog):
                dialog._rows[paths[2]].setCheckState(0, Qt.Checked)
                checks["reset_selection"] = dialog.reset_selection()
                dialog.reset_button.click()
                return dialog.result()

            def wait():
                deadline = time.monotonic() + 5
                while window._job is not None and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.01)
                app.processEvents()
                self.assertIsNone(window._job)

            window = MainWindow()
            window.show()
            window.file_panel.load_files(paths)
            wait()
            with patch.object(BatchCropDialog, "exec", run_dialog), patch.object(CropDialog, "exec", confirm):
                window.btn_batch_crop.click()
                wait()
            self.assertEqual(checks["initial"], paths)
            self.assertIsNone(checks["before_apply"])
            self.assertEqual(checks["before_review"], paths[:1])
            self.assertCountEqual(checks["after_review"], paths[:3])
            self.assertIn("手动确认 2 · 自动识别 1", checks["summary"])
            self.assertEqual(checks["confirmed_only"], paths[1:3])
            self.assertEqual(checks["review_options"], (2.0, "135"))
            self.assertEqual(checks["next"], "c-review.png")
            self.assertEqual(checks["restored_rotation"], 0.6)
            self.assertEqual(checks["after_confirmation"], "c-review.png")
            self.assertEqual(checks["returned_at"], "d-skip.png")
            self.assertIsNone(window._crop)
            self.assertNotIn(paths[3], window._photos)
            expected = window._photos[paths[1]].crop
            self.assertEqual(expected["border_percent"], 2)
            self.assertEqual(expected["rotation"], 0.6)
            second = window._photos[paths[2]].crop
            window.close()

            window = MainWindow()
            window.show()
            app.processEvents()
            wait()
            window._on_file_selected(paths[1])
            wait()
            self.assertEqual(window._crop, expected)
            window._undo()
            self.assertIsNone(window._crop)
            window._redo()
            self.assertEqual(window._crop, expected)
            with patch.object(BatchCropDialog, "exec", reset_dialog):
                window.btn_batch_crop.click()
            self.assertEqual(checks["reset_selection"], [paths[2]])
            self.assertEqual(window._crop, expected)
            self.assertIsNone(window._photos[paths[2]].crop)
            window._on_file_selected(paths[2])
            wait()
            self.assertEqual(window.image_viewer.sceneRect().width(), framed.shape[1])
            window._undo()
            self.assertEqual(window._crop, second)
            window.close()

    def test_crop_survives_reopen_and_exports_after_source_pixel_repairs(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory, patch("ui.main_window.Config", lambda: Config(directory)):
            path = Path(directory) / "film.png"
            image = np.full((300, 460, 3), 1000, np.uint16)
            image[30:270, 35:425] = (21000, 27000, 34000)
            image[100:120, 110:130] = (31000, 38000, 43000)
            cv2.imwrite(str(path), image)

            def wait(window):
                deadline = time.monotonic() + 5
                while window._job is not None and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.01)
                self.assertIsNone(window._job)

            initial_border = []

            def accept_crop():
                dialog = app.activeModalWidget()
                if isinstance(dialog, CropDialog):
                    initial_border.append(dialog.border.value())
                    dialog.border.setValue(2)
                    dialog.apply_button.click()

            window = MainWindow()
            window.show()
            window.file_panel.load_files([path])
            wait(window)
            QTimer.singleShot(0, accept_crop)
            window.btn_crop.click()
            self.assertEqual(initial_border, [0])
            settings = window._crop
            self.assertEqual(settings["border_percent"], 2)
            viewer = window.image_viewer
            _, _, (width, height), _ = crop_plan(image.shape, settings)
            self.assertEqual((viewer.sceneRect().width(), viewer.sceneRect().height()), (width, height))
            viewer.zoom_100()
            window.edit_buttons["clone"].click()
            QTest.mouseClick(viewer.viewport(), Qt.LeftButton, Qt.AltModifier, viewer.image_position((120, 110)))
            QTest.mouseClick(viewer.viewport(), Qt.LeftButton, pos=viewer.image_position((240, 155)))
            wait(window)
            np.testing.assert_array_equal(window._current_image[155, 240], image[110, 120])
            edited = window._current_image.copy()
            output = path.parent / "exports"
            window._start_export([str(path)], output, ".png", "edited", include_review=True)
            wait(window)
            np.testing.assert_array_equal(cv2.imread(str(output / "film_clean.png"), cv2.IMREAD_UNCHANGED),
                                          apply_crop(edited, settings))
            report = window._last_review_report.read_text(encoding="utf-8")
            self.assertIn(f'data-width="{width}" data-height="{height}"', report)
            self.assertIn("已修复区域", report)
            np.testing.assert_array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), image)
            window.close()

            window = MainWindow()
            window.show()
            app.processEvents()
            wait(window)
            self.assertEqual(window._crop, settings)
            np.testing.assert_array_equal(window._current_image, edited)
            window.cancel_crop_action.trigger()
            self.assertIsNone(window._crop)
            self.assertEqual(window.image_viewer.sceneRect().width(), image.shape[1])
            np.testing.assert_array_equal(window._current_image, edited)
            window._undo()
            self.assertEqual(window._crop, settings)
            window._undo()
            np.testing.assert_array_equal(window._current_image, image)
            self.assertEqual(window._crop, settings)
            window._undo()
            self.assertIsNone(window._crop)
            self.assertEqual(window.image_viewer.sceneRect().width(), image.shape[1])
            window._redo()
            self.assertEqual(window._crop, settings)
            window.close()

    def test_sidebar_folding_preserves_widths_and_restores_layout(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory, patch("ui.main_window.Config", lambda: Config(directory)):
            window = MainWindow()
            window.resize(1400, 900)
            window.show()
            app.processEvents()
            splitter = window.centralWidget()
            splitter.setSizes([240, 822, 330])
            window._remember_sidebar_widths()
            original = splitter.sizes()
            left = window._sidebar_buttons["left"]
            header_left = window._sidebar_header_buttons["left"]
            for side, button in window._sidebar_buttons.items():
                handle = window._sidebar_handles[side]
                center = handle.mapTo(window, handle.rect().center())
                self.assertAlmostEqual(button.geometry().center().x(), center.x(), delta=1)
                self.assertIs(window.childAt(button.geometry().center()), button)
            window.file_panel.search.setText("scan")
            self.assertFalse(header_left.icon().isNull())
            header_left.click()
            app.processEvents()
            self.assertTrue(window.file_panel.isHidden())
            self.assertFalse(window.dust_panel.isHidden())
            self.assertEqual(splitter.sizes()[2], original[2])
            self.assertIn("展开", left.accessibleName())
            self.assertFalse(left.isChecked())
            self.assertEqual(header_left.toolTip(), left.toolTip())
            self.assertFalse(left.icon().isNull())
            self.assertEqual(left.x(), splitter.mapTo(window, QPoint()).x())

            viewer = window.image_viewer
            viewer.setFocus()
            QTest.keyClick(viewer, Qt.Key_F, Qt.ControlModifier | Qt.ShiftModifier)
            app.processEvents()
            self.assertEqual(splitter.sizes(), [0, splitter.width(), 0])
            right = window._sidebar_buttons["right"]
            self.assertEqual(right.geometry().right(), splitter.mapTo(window, splitter.rect().topRight()).x())
            QTest.keyClick(viewer, Qt.Key_F, Qt.ControlModifier)
            app.processEvents()
            self.assertTrue(window.file_panel.search.hasFocus())
            self.assertTrue(header_left.isChecked())
            self.assertEqual(window.file_panel.search.text(), "scan")
            self.assertEqual(splitter.sizes()[0], original[0])
            self.assertTrue(window.dust_panel.isHidden())
            window.close()

            window = MainWindow()
            window.show()
            app.processEvents()
            self.assertFalse(window.file_panel.isHidden())
            self.assertTrue(window.dust_panel.isHidden())
            window._sidebar_buttons["right"].click()
            app.processEvents()
            self.assertTrue(window._sidebar_header_buttons["right"].isChecked())
            self.assertEqual(window.centralWidget().sizes(), original)
            window.close()

    def test_model_management_imports_previews_and_deletes_without_closing(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory, patch("ui.main_window.Config", lambda: Config(directory)):
            window = MainWindow(restore_workspace=False)
            folder = Path(directory)
            paths = [folder / "fibers.json", folder / "scratches.json"]
            outline = ((0, 70), (15, 5), (100, 0), (20, 80))
            examples = [FiberExample("p", "r", 1, False, (float(value),) * 14, traced=True,
                                      outline=outline if value == 0 else ())
                        for value in range(8)]
            write_fiber_model(paths[0], train_fiber_model(examples))
            write_fiber_model(paths[1], train_fiber_model([
                FiberExample("p", "r", 1, False, (20.,) * 14, traced=True),
            ]))
            dialog = FiberModelDialog(window, window._fiber_store, directory)
            dialog.active_changed.connect(window._set_fiber_model)
            dialog.show()
            app.processEvents()
            try:
                self.assertEqual({button.text() for button in dialog.findChildren(QPushButton)},
                                 {"导入 JSON…", "启用所选", "停用所选", "删除所选…", "关闭", "上一页", "下一页"})
                self.assertFalse(dialog.delete.isEnabled())
                with patch("ui.fiber_learning_dialog.QFileDialog.getOpenFileNames",
                           return_value=([str(path) for path in paths], "")):
                    dialog.import_button.click()
                self.assertTrue(dialog.isVisible())
                self.assertEqual(dialog.models.count(), 2)
                self.assertEqual(dialog.source.text(), str(paths[-1]))
                self.assertEqual(dialog.gallery.page_text.text(), "1 / 1")
                self.assertIn("示意", dialog.gallery.tiles[0][2].text())
                dialog.models.setCurrentRow(1)
                self.assertEqual(dialog.source.text(), str(paths[0]))
                self.assertEqual(dialog.gallery.page_text.text(), "1 / 2")
                self.assertNotIn("示意", dialog.gallery.tiles[0][2].text())
                self.assertFalse(dialog.gallery.tiles[0][1].pixmap().isNull())
                dialog.gallery.forward.click()
                self.assertEqual(dialog.gallery.page_text.text(), "2 / 2")
                self.assertFalse(dialog.gallery.forward.isEnabled())
                dialog.models.selectAll()
                dialog.enable.click()
                self.assertEqual(len(window._scan_settings()["fiber_model"].members), 2)
                self.assertEqual(dialog.gallery.page_text.text(), "2 / 2")
                dialog.models.setCurrentRow(0, QItemSelectionModel.ClearAndSelect)
                self.assertEqual(dialog.gallery.page_text.text(), "1 / 1")
                original_json = paths[1].read_bytes()
                with patch("ui.fiber_learning_dialog.NoticeDialog.exec", return_value=QDialog.Rejected):
                    dialog.delete.click()
                self.assertEqual(dialog.models.count(), 2)
                with patch("ui.fiber_learning_dialog.NoticeDialog.exec", return_value=QDialog.Accepted):
                    dialog.delete.click()
                self.assertEqual(paths[1].read_bytes(), original_json)
                self.assertEqual(dialog.models.count(), 1)
                self.assertEqual(dialog.source.text(), str(paths[0]))
                self.assertEqual(window._scan_settings()["fiber_model"], read_fiber_model(paths[0]))
                self.assertEqual(dialog.gallery.page_text.text(), "1 / 2")
                dialog.gallery.forward.click()
                self.assertEqual(dialog.gallery.page_text.text(), "2 / 2")
                dialog.disable.click()
                self.assertNotIn("fiber_model", window._scan_settings())
                self.assertTrue(dialog.isVisible())
            finally:
                dialog.close_button.click()
                self.assertFalse(dialog.isVisible())
                window.close()
                window._stop_workers()

    def test_annotation_training_shows_all_saved_contours_and_uses_the_displayed_samples(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            store = FiberLearningStore(folder / "config.json")
            imported = train_fiber_model([FiberExample("p", "r", 1, False, (3.,) * 14, traced=True)])
            imported_id, = store.add_models([("已导入", "/tmp/existing.json", imported)])
            store.set_enabled([imported_id], True)
            contour = np.array([[[15, 20]], [[50, 24]], [[45, 29]], [[15, 24]]], np.int32)
            trace = DustItem(1, 15, 20, 36, 10, 144, 31, 24, dust_type="scratch", contour=contour)
            for i in range(8):
                store.label_many(f"old-photo-{i}", "old roll", 1, [(f"trace-{i}", True, (.9,) * 14, trace)])
            store.label("legacy-photo", "legacy roll", "scene", 0, True, (.2,) * 14)
            dialog = FiberLabelDialog(None, folder / "current.jpg", np.zeros((80, 100, 3), np.uint8),
                                      [(trace, "current-scene", (.1,) * 14)], store, "current",
                                      params={}, can_previous=False, can_next=False, model=None)
            dialog.show()
            app.processEvents()
            dialog.candidates.setCurrentRow(0)
            dialog.label_buttons[1].click()
            dialog.tabs.setCurrentIndex(dialog.training_index)
            app.processEvents()
            panel = dialog.training
            self.assertEqual(panel.summary.text(), "已保存 10 处标注")
            self.assertEqual(panel.gallery.page_text.text(), "1 / 2")
            self.assertIn("真实缺陷", panel.gallery.tiles[0][2].text())
            self.assertIn("示意", panel.gallery.tiles[8][2].text())
            panel.gallery.forward.click()
            self.assertIn("画面线条", panel.gallery.tiles[0][2].text())
            self.assertNotIn("示意", panel.gallery.tiles[0][2].text())
            self.assertTrue(panel._examples[-1].outline)
            self.assertFalse(panel._examples[-1].traced)
            QTest.keyClick(panel.train, Qt.Key_Space)
            deadline = time.monotonic() + 10
            while panel.busy and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            if panel.busy:
                panel._job.requestInterruption()
                panel._job.wait()
            self.assertFalse(panel.busy)
            self.assertTrue(dialog.isVisible())
            entry, existing = store.library()
            self.assertEqual(entry.model.metrics["samples"], 10)
            self.assertFalse(entry.enabled)
            self.assertEqual(existing.model, imported)
            exported = folder / "trained.json"
            with patch("ui.fiber_learning_dialog.QFileDialog.getSaveFileName", return_value=(str(exported), "")):
                panel.export.click()
            self.assertEqual(read_fiber_model(exported), entry.model)
            dialog.close_button.click()
            reopened = FiberTrainingDialog(None, FiberLearningStore(folder / "config.json"))
            self.assertEqual(reopened.panel.summary.text(), "已保存 10 处标注")
            self.assertTrue(reopened.panel.export.isEnabled())
            self.assertEqual(reopened.panel._model, entry.model)
            reopened.close()

    def test_space_drag_keeps_brush_active_and_guides_a_faint_scratch(self):
        app = QApplication.instance() or QApplication([])
        source = Path(__file__).resolve().parents[1] / "demo" / "HDR-window-16bit.tif"
        image = read_scan(source, read_ir=False).image
        params = dict(threshold=20, min_area=3, min_confidence=.34)
        with tempfile.TemporaryDirectory() as directory:
            store = FiberLearningStore(Path(directory) / "config.json")
            dialog = FiberLabelDialog(None, source, image, [], store, "test-photo", params=params,
                                      can_previous=False, can_next=False, model=None,
                                      start_view=(False, 2.88, 470, 461))
            dialog.show()
            app.processEvents()
            canvas = dialog.canvas
            viewport = canvas.viewport()
            middle = viewport.rect().center()
            scrollbar = canvas.horizontalScrollBar()
            before = scrollbar.value()
            dialog.draw_button.setFocus()
            QTest.keyPress(dialog.draw_button, Qt.Key_Space)
            QTest.mousePress(viewport, Qt.LeftButton, pos=middle)
            QTest.mouseMove(viewport, middle + QPoint(60, 0))
            QTest.mouseRelease(viewport, Qt.LeftButton, pos=middle + QPoint(60, 0))
            QTest.keyRelease(canvas, Qt.Key_Space)
            self.assertLess(scrollbar.value(), before)
            self.assertIsNone(dialog._job)
            self.assertFalse(canvas.space_pan)

            start, end = (canvas.image_position(point) for point in ((486, 441), (458, 481)))
            QTest.mousePress(viewport, Qt.LeftButton, pos=start)
            QTest.mouseMove(viewport, (start + end) / 2)
            QTest.mouseRelease(viewport, Qt.LeftButton, pos=end)
            deadline = time.monotonic() + 15
            while dialog._job is not None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            if dialog._job is not None:
                dialog._job.requestInterruption()
                dialog._job.wait()
            self.assertIsNone(dialog._job)
            item, key, _ = next(row for row in dialog.rows if row[0].height >= 39)
            self.assertIn(key, dialog._checked)
            self.assertEqual(item.dust_type, "scratch")
            self.assertLessEqual(item.y, 443)
            self.assertGreaterEqual(item.y + item.height, 480)
            dialog.save_defects.click()
            self.assertEqual(store.counts(), (1, 0, 1))
            dialog.close()

    def test_brush_and_box_label_original_contours_for_future_scans(self):
        app = QApplication.instance() or QApplication([])
        demo = Path(__file__).resolve().parents[1] / "demo"
        source = demo / "HDR-window-16bit.tif"
        image = read_scan(source, read_ir=False).image
        params = dict(threshold=20, min_area=3, min_confidence=.34)
        with tempfile.TemporaryDirectory() as directory:
            store = FiberLearningStore(Path(directory) / "config.json")

            def dialog_for_photo():
                return FiberLabelDialog(None, source, image, [], store, "test-photo", params=params,
                                        can_previous=False, can_next=False, model=None)

            dialog = dialog_for_photo()
            dialog.show()
            app.processEvents()
            canvas = dialog.canvas

            def drag(points):
                points = [canvas.image_position(point) for point in points]
                QTest.mousePress(canvas.viewport(), Qt.LeftButton, pos=points[0])
                for point in points[1:-1]:
                    QTest.mouseMove(canvas.viewport(), point)
                QTest.mouseRelease(canvas.viewport(), Qt.LeftButton, pos=points[-1])
                deadline = time.monotonic() + 15
                while dialog._job is not None and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.01)
                if dialog._job is not None:
                    dialog._job.requestInterruption()
                    dialog._job.wait()
                self.assertIsNone(dialog._job)

            drag([(1005, 588), (1023, 585), (1036, 600), (1032, 612), (1021, 621)])
            hook = (1003, 582, 36, 41)
            self.assertTrue(any(item.bbox == hook for item, _, _ in dialog.rows))
            box = next(button for button in dialog.modes.buttons() if button.property("selectionMode") == "detect")
            box.click()
            drag([(975, 572), (1052, 639)])
            self.assertEqual(sum(item.bbox == hook for item, _, _ in dialog.rows), 1)
            for index, (item, _, _) in enumerate(dialog.rows):
                if item.bbox != hook:
                    dialog.candidates.item(index).setCheckState(Qt.Unchecked)
            dialog.save_defects.click()
            self.assertEqual(store.counts(), (1, 0, 1))
            dialog.close()
            dialog = dialog_for_photo()
            self.assertEqual([item.bbox for item, _, _ in dialog.rows], [hook])
            model = FiberModel.decode(train_fiber_model(store.examples()).encode())
            self.assertTrue(model.templates)
            other = read_scan(demo / "HDR-window-reference" / "HDR-window-UltraHDR.jpg", read_ir=False).image
            self.assertFalse(any(item.bbox == hook for item in detect_scan(other, params).items))
            match, = [item for item in detect_scan(other, params, fiber_model=model).items if item.bbox == hook]
            self.assertFalse(match.selected)

    def test_marker_modes_use_the_displayed_shape_and_survive_reopen(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory, patch("ui.main_window.Config", lambda: Config(directory)):
            window = MainWindow()
            self.addCleanup(window._stop_workers)
            window.show()
            image = np.full((240, 280, 3), 85, np.uint8)
            window._original_image = window._current_image = image
            viewer = window.image_viewer
            viewer.load_image(image)
            window._refresh_controls()
            viewer.zoom_100()
            contour = np.array([(60, 50), (155, 50), (155, 58), (68, 58), (68, 150), (60, 150)], np.int32)
            fiber = DustItem(1, 60, 50, 96, 101, 1500, 100, 54, dust_type="fiber", contour=contour)
            viewer.update_annotations([fiber])
            app.processEvents()

            def marker_visible(canvas, point):
                pixels = canvas.viewport().grab().toImage()
                position = canvas.image_position(point) * pixels.devicePixelRatio()
                color = pixels.pixelColor(position)
                return color.green() - color.red() > 25

            clicked = []
            viewer.annotation_clicked.connect(clicked.append)
            end = viewer.image_position((64, 141))
            center = viewer.image_position(fiber.center)
            self.assertTrue(window._marker_style_actions["contour"].isChecked())
            self.assertTrue(marker_visible(viewer, (60, 140)))
            QTest.mouseClick(viewer.viewport(), Qt.LeftButton, pos=end)
            QTest.mouseClick(viewer.viewport(), Qt.LeftButton, pos=viewer.image_position((110, 100)))
            self.assertEqual(clicked, [1])

            window._marker_style_actions["circle"].trigger()
            self.assertFalse(window._marker_style_actions["contour"].isChecked())
            self.assertFalse(marker_visible(viewer, (60, 140)))
            QTest.mouseClick(viewer.viewport(), Qt.LeftButton, pos=end)
            QTest.mouseClick(viewer.viewport(), Qt.LeftButton, pos=center)
            self.assertEqual(clicked, [1, 1])
            viewer.reset_highlight()
            self.assertTrue(marker_visible(viewer, (103, 54)))

            viewer.set_comparison(image)
            viewer.set_original_marks([fiber])
            self.assertFalse(marker_visible(viewer, (60, 140)))
            window._marker_style_actions["contour"].trigger()
            self.assertTrue(marker_visible(viewer, (60, 140)))
            window._marker_style_actions["circle"].trigger()
            window.close()

            reopened = MainWindow()
            self.addCleanup(reopened._stop_workers)
            self.assertTrue(reopened._marker_style_actions["circle"].isChecked())
            self.assertEqual(reopened.original_viewer._marker_style, "circle")
            reopened.close()

    def test_bulk_selection_skips_candidates_awaiting_review(self):
        app = QApplication.instance() or QApplication([])
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("ui.main_window.Config", lambda: Config(directory)),
        ):
            window = MainWindow()
            self.addCleanup(window._stop_workers)
            window._original_image = window._current_image = np.full((105, 120, 3), 110, np.uint8)
            ordinary = DustItem(1, 20, 20, 5, 5, 25, 22, 22)
            protected = DustItem(2, 80, 30, 5, 5, 25, 82, 32, face_protected=True)
            line = DustItem(3, 40, 60, 32, 4, 128, 54, 62, dust_type="scratch", scene_line=True)
            old_line = DustItem(3, 40, 60, 32, 4, 128, 54, 62, dust_type="scratch")
            learned = DustItem(4, 36, 79, 26, 8, 70, 48, 83, dust_type="fiber", learned_fiber=True)
            window._on_detection_finished([ordinary, protected, old_line], None)
            self.assertTrue(window._dust_items[2].selected)
            window._on_detection_finished([ordinary, protected, line, learned], None)
            app.processEvents()
            panel = window.dust_panel
            self.assertEqual([item.selected for item in window._dust_items], [True, False, False, False])
            self.assertIn("场景线待核对", panel.dust_list.item(2).text())
            self.assertIn("模型补检", panel.dust_list.item(3).text())
            self.assertFalse(window._merge_scan_review([learned], [], [])[0].selected)
            panel.btn_check_visible.click()
            self.assertFalse(window._dust_items[1].selected)
            self.assertFalse(window._dust_items[2].selected)
            self.assertFalse(window._dust_items[3].selected)
            panel.dust_list.item(1).setCheckState(Qt.Checked)
            panel.dust_list.item(2).setCheckState(Qt.Checked)
            panel.dust_list.item(3).setCheckState(Qt.Checked)
            window._on_detection_finished([ordinary, protected, line, learned], None)
            self.assertTrue(window._dust_items[1].selected)
            self.assertTrue(window._dust_items[2].selected)
            self.assertTrue(window._dust_items[3].selected)
            panel.btn_uncheck_visible.click()
            window._on_detection_finished([protected], (74, 23, 95, 49))
            window._on_detection_finished([line], (39, 59, 74, 65))
            self.assertFalse(window._dust_items[1].selected)
            self.assertFalse(window._dust_items[2].selected)
            window.close()

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

            window.file_panel.load_files([first, second])
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
            panel = window.file_panel
            header_actions = (panel.btn_add_photos, panel.btn_open, panel.btn_clear)
            for button in header_actions:
                self.assertFalse(button.icon().isNull())
                self.assertTrue(button.accessibleName())
                self.assertEqual(button.y(), header_actions[0].y())
            panel.search.setText("TWO")
            self.assertTrue(panel.file_list.item(0).isHidden())
            self.assertFalse(panel.file_list.item(1).isHidden())
            panel.search.clear()
            panel.filter_buttons["unscanned"].click()
            self.assertTrue(panel.file_list.item(0).isHidden())
            panel.select_next()
            finish_job()
            self.assertEqual(window._current_filepath, str(second.resolve()))
            window._on_detection_finished(detector.detect(image), None)
            self.assertTrue(all(item.selected for item in window._dust_items))
            panel.filter_buttons["all"].click()
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
            window._start_export([str(first), str(second)], output, ".png", "repair", include_review=True)
            finish_job()
            self.assertEqual(window.overlay_bar.value(), 2)
            np.testing.assert_array_equal(cv2.imread(str(output / "one_clean.png")), healed)
            np.testing.assert_array_equal(cv2.imread(str(output / "two_clean.png")), healed)
            report = window._last_review_report.read_text(encoding="utf-8")
            self.assertIn("one_clean.png", report)
            self.assertIn("two_clean.png", report)
            self.assertEqual(report.count("data:image/jpeg;base64,"), 8)

            window.resize(window.minimumSize())
            window.show()
            window.centralWidget().setSizes([300, 350, 390])
            app.processEvents()

            viewer = window.image_viewer
            viewer.setFocus()
            self.assertEqual(window.edit_buttons[None].text(), "指针")
            viewer.zoom_100()
            QTest.keyClick(viewer, Qt.Key_Equal)
            self.assertAlmostEqual(viewer.transform().m11(), 1.2)
            QTest.keyClick(viewer, Qt.Key_Minus)
            self.assertAlmostEqual(viewer.transform().m11(), 1.0)
            QTest.keyClick(viewer, Qt.Key_S)
            self.assertEqual(viewer._selection_mode, "clone")
            radius = window.retouch_radius.value()
            QTest.keyClick(viewer, Qt.Key_BracketRight)
            self.assertGreater(window.retouch_radius.value(), radius)
            self.assertEqual(viewer.retouch_radius, window.retouch_radius.value())
            QTest.keyClick(viewer, Qt.Key_BracketLeft)
            self.assertEqual(window.retouch_radius.value(), radius)
            window.retouch_radius.setFocus()
            QTest.keyClick(window.retouch_radius.lineEdit(), Qt.Key_BracketRight)
            self.assertGreater(window.retouch_radius.value(), radius)
            QTest.keyClick(window.retouch_radius.lineEdit(), Qt.Key_BracketLeft)
            self.assertEqual(window.retouch_radius.value(), radius)
            viewer.setFocus()
            QTest.keyClick(viewer, Qt.Key_P)
            search_radius = window.brush_radius.value()
            QTest.keyClick(viewer, Qt.Key_BracketRight)
            self.assertGreater(window.brush_radius.value(), search_radius)
            self.assertEqual(window.retouch_radius.value(), radius)
            QTest.keyClick(viewer, Qt.Key_BracketLeft)
            QTest.keyClick(viewer, Qt.Key_S)
            panel.search.setFocus()
            QTest.keyClicks(panel.search, "v[h]")
            self.assertEqual(panel.search.text(), "v[h]")
            self.assertEqual(viewer._selection_mode, "clone")
            self.assertEqual(window.retouch_radius.value(), radius)
            self.assertAlmostEqual(viewer.transform().m11(), 1.0)
            panel.search.clear()
            viewer.setFocus()
            QTest.keyClick(viewer, Qt.Key_V)
            self.assertIsNone(viewer._selection_mode)
            self.assertEqual(viewer.viewport().cursor().shape(), Qt.ArrowCursor)
            for key, mode in ((Qt.Key_2, "side"), (Qt.Key_1, "wipe")):
                QTest.keyClick(viewer, key, Qt.ControlModifier)
                self.assertTrue(window.btn_compare.isChecked())
                self.assertTrue(window.compare_buttons[mode].isChecked())
            QTest.keyClick(viewer, Qt.Key_B)
            self.assertFalse(window.btn_compare.isChecked())

            shortcuts = {}

            def read_shortcuts():
                dialog = app.activeModalWidget()
                if dialog is not None:
                    tree = dialog.findChild(QTreeWidget, "shortcutList")
                    if tree is not None:
                        for index in range(tree.topLevelItemCount()):
                            section = tree.topLevelItem(index)
                            shortcuts.update((section.child(i).text(0), section.child(i).text(1))
                                             for i in range(section.childCount()))
                    dialog.reject()

            QTimer.singleShot(0, read_shortcuts)
            QTest.keyClick(viewer, Qt.Key_F1)
            self.assertEqual(shortcuts.get("缩小工具半径"), "[")
            self.assertEqual(shortcuts.get("放大工具半径"), "]")
            self.assertEqual(shortcuts.get("缩小画布"), "-")
            self.assertEqual(shortcuts.get("放大画布"), "=")
            self.assertEqual(shortcuts.get("指针"), "V")
            self.assertIn("临时抓手", shortcuts)
            self.assertTrue(window.image_viewer._show_all)
            window.btn_marks.click()
            self.assertFalse(window.image_viewer._show_all)
            window.btn_marks.click()
            self.assertTrue(window.image_viewer._show_all)
            for button in (*window.edit_buttons.values(), window.btn_marks,
                           window.btn_fit, window.btn_zoom_in, window.btn_zoom_out):
                self.assertFalse(button.icon().isNull())
                if button.parent() is window.edit_toolbar:
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
                self.assertEqual(window.retouch_radius.isVisible(), mode in RETOUCH_MODES)

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
                self.assertIn(dust.center, [mark.center for mark in original._before_marks])
                before = original.viewport().grab().toImage()
                after = edited.viewport().grab().toImage()
                window.btn_marks.click()
                self.assertNotEqual(original.viewport().grab().toImage(), before)
                self.assertEqual(edited.viewport().grab().toImage(), after)
                window.btn_marks.click()
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
            for position in (0, 100):
                window.compare_slider.setValue(position)
                before = window.image_viewer.viewport().grab().toImage()
                window.btn_marks.click()
                after = window.image_viewer.viewport().grab().toImage()
                self.assertEqual(before == after, position == 0)
                window.btn_marks.click()
            window.btn_compare.click()
            self.assertTrue(window.compare_row.isHidden())
            first_history = window._history
            first_row = window.file_panel.file_list.item(0)
            with patch("ui.main_window.QFileDialog.getOpenFileNames", return_value=([str(third)], "")):
                window.file_panel.btn_add_photos.click()
                finish_job()
                self.assertEqual(window._current_filepath, str(third.resolve()))
                window.file_panel.btn_add_photos.click()
            self.assertEqual(window.file_panel.photo_paths(), [str(p.resolve()) for p in (first, second, third)])
            self.assertIs(window.file_panel.file_list.item(0), first_row)
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
            self.assertEqual(viewer.viewport().cursor().shape(), Qt.BlankCursor)
            QTest.mouseRelease(viewer.viewport(), Qt.LeftButton, pos=center + QPoint(45, 0))
            self.assertIsNone(window._job)
            self.assertTrue(draw.isChecked())
            QTest.mouseClick(viewer.viewport(), Qt.LeftButton,
                             pos=viewer.mapFromScene(QPointF(85, 88)))
            finish_job()
            self.assertEqual(window._dust_items[0].center, (85, 88))
            window.file_panel.select_file(str(first))
            finish_job()
            self.assertIs(window._history, first_history)
            window.file_panel.select_file(str(third))
            finish_job()
            self.assertEqual(window.dust_panel.dust_list.count(), 1)
            self.assertEqual(window._dust_items[0].dust_type, "bright")
            window._start_batch_scan([str(second.resolve()), str(third.resolve())])
            finish_job()
            self.assertTrue(window._photos[str(second.resolve())].scan_source)
            window._step_candidate(1)
            window.dust_panel.btn_trial.click()
            self.assertEqual(len(window.dust_panel.preview._comparison), 3)
            group = [(str(path.resolve()), window._photos[str(path.resolve())].candidates[0])
                     for path in (second, third)]
            window._repair_repeat_group(group)
            finish_job()
            self.assertGreater(window._photos[str(second.resolve())].revision, 0)
            self.assertFalse(window._photos[str(second.resolve())].candidates)
            final_pixels = window._current_image.copy()
            final_view = window.image_viewer.view_position()
            window.close()

            with patch("ui.main_window.Config", lambda: Config(directory)):
                window = MainWindow()
            self.addCleanup(window._stop_workers)
            window.show()
            app.processEvents()
            finish_job()
            self.assertEqual(window._current_filepath, str(third.resolve()))
            self.assertEqual(window.file_panel.photo_paths(), [str(p.resolve()) for p in (first, second, third)])
            np.testing.assert_array_equal(window._current_image, final_pixels)
            self.assertAlmostEqual(window.image_viewer.view_position()[1], final_view[1])
            window.btn_compare.click()
            self.assertIn((85, 88), [mark.center for mark in window.image_viewer._before_marks])
            window.btn_compare.click()
            window._undo()
            np.testing.assert_array_equal(window._current_image, image)
            self.assertEqual(window._dust_items[0].center, (85, 88))
            window._redo()
            np.testing.assert_array_equal(window._current_image, final_pixels)
            replacement = np.full((140, 160, 3), 75, np.uint8)
            cv2.imwrite(str(first), replacement)
            window.file_panel.select_file(str(first))
            finish_job()
            np.testing.assert_array_equal(window._current_image, replacement)
            self.assertFalse(window._dust_items)
            self.assertEqual(window._revision, 0)
            self.assertIn("原图已变化", window.lbl_status_file.text())
            window.close()

    def test_removing_imports_keeps_edits_and_empty_workspace(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, third = (root / name for name in ("one.png", "two.png", "three.png"))
            image = np.full((120, 140, 3), 116, np.uint8)
            cv2.circle(image, (65, 59), 3, (224, 224, 224), cv2.FILLED)
            for path in (first, second, third):
                cv2.imwrite(str(path), image)

            def create_window():
                with patch("ui.main_window.Config", lambda: Config(directory)):
                    window = MainWindow()
                self.addCleanup(window._stop_workers)
                return window

            def finish_job(window):
                deadline = time.monotonic() + 5
                app.processEvents()
                while window._job is not None and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.01)
                self.assertIsNone(window._job)

            def export_paths(window, batch):
                paths = []

                def inspect_dialog(dialog):
                    photos = dialog.photos
                    paths.extend(photos.item(row).data(Qt.UserRole) for row in range(photos.count()))
                    dialog.reject()
                    return QDialog.Rejected

                with patch.object(ExportDialog, "exec", inspect_dialog):
                    window._show_export_dialog(batch=batch)
                return paths

            window = create_window()
            window.file_panel.load_files([first, second, third])
            finish_job(window)
            first_path, second_path, third_path = (str(path.resolve()) for path in (first, second, third))
            self.assertEqual(window._current_filepath, first_path)
            window._on_detection_finished(NativeDustDetector(min_confidence=0.34).detect(image), None)
            self.assertTrue(window._dust_items)
            with patch("ui.main_window.sys.platform", "darwin"), patch(
                "ui.main_window.QProcess.startDetached", return_value=(True, 42)
            ) as reveal:
                window.reveal_photo_action.trigger()
                reveal.assert_called_once_with("/usr/bin/open", ["-R", first_path])

            window._remove_photo(third_path)
            self.assertEqual(window._current_filepath, first_path)
            window.remove_photo_action.trigger()
            finish_job(window)
            self.assertEqual(window._current_filepath, second_path)
            self.assertEqual(window.file_panel.photo_paths(), [second_path])
            self.assertEqual(export_paths(window, batch=False), [second_path])
            window.config.set("last_directory", directory)
            prompts = []

            def answer_clear(confirm):
                box = app.activeModalWidget()
                if isinstance(box, NoticeDialog):
                    prompts.append((box.message_label.text(), box.cancel_button.isDefault()))
                    if confirm:
                        box.confirm_button.click()
                    else:
                        QTest.keyClick(box, Qt.Key_Return)

            QTimer.singleShot(0, lambda: answer_clear(False))
            window.file_panel.btn_clear.click()
            self.assertEqual(window.file_panel.photo_paths(), [second_path])
            self.assertEqual(window._current_filepath, second_path)
            self.assertIn("1 张照片", prompts[0][0])
            self.assertTrue(prompts[0][1])
            QTimer.singleShot(0, lambda: answer_clear(True))
            window.file_panel.btn_clear.click()
            self.assertIsNone(window._current_filepath)
            self.assertEqual(window.image_viewer.get_image_size(), (0, 0))
            self.assertEqual(window.file_panel.photo_paths(), [])
            self.assertEqual(export_paths(window, batch=True), [])
            self.assertTrue(all(path.is_file() for path in (first, second, third)))
            window.close()

            window = create_window()
            finish_job(window)
            self.assertEqual(window.file_panel.photo_paths(), [])
            window._add_photos([first])
            finish_job(window)
            self.assertTrue(any(item.center == (65, 59) for item in window._dust_items))
            self.assertEqual(export_paths(window, batch=True), [first_path])
            window.close()

    def test_reset_cached_repairs_with_and_without_original_available(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "one.png", root / "two.png"
            hidden = root / "offline.png"
            image = np.full((120, 140, 3), 116, np.uint8)
            cv2.circle(image, (65, 59), 3, (224, 224, 224), cv2.FILLED)
            for path in (first, second):
                cv2.imwrite(str(path), image)
            paths = [str(path.resolve()) for path in (first, second)]
            detector = NativeDustDetector(min_confidence=0.34)

            def open_window():
                with patch("ui.main_window.Config", lambda: Config(directory)):
                    window = MainWindow()
                self.addCleanup(window._stop_workers)
                window.show()
                app.processEvents()
                return window

            def finish_job(window):
                deadline = time.monotonic() + 5
                while window._job is not None and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.01)
                self.assertIsNone(window._job)

            def repair(window):
                window._on_detection_finished(detector.detect(image), None)
                window._on_remove_dust()
                finish_job(window)
                self.assertGreater(window._revision, 0)

            window = open_window()
            window.file_panel.load_files([first, second])
            finish_job(window)
            repair(window)
            window.file_panel.select_file(second)
            finish_job(window)
            repair(window)
            window.close()
            second.rename(hidden)

            window = open_window()
            finish_job(window)
            self.assertEqual(window.file_panel.photo_paths(), paths)
            self.assertEqual(window._current_filepath, paths[0])
            self.assertGreater(window._revision, 0)
            self.assertTrue(window.reset_photo_action.isEnabled())
            with patch("ui.main_window.NoticeDialog.exec", return_value=QDialog.Rejected):
                window.reset_photo_action.trigger()
            self.assertGreater(window._revision, 0)
            with patch("ui.main_window.NoticeDialog.exec", return_value=QDialog.Accepted):
                window.reset_photo_action.trigger()
            np.testing.assert_array_equal(window._current_image, image)
            self.assertEqual(window._history.count(), 0)
            self.assertFalse(window._repaired_marks)
            self.assertFalse(window._scan_source)
            self.assertFalse(window.btn_compare.isEnabled())
            repair(window)

            photo_list = window.file_panel.file_list
            position = photo_list.visualItemRect(photo_list.item(1)).center()

            class ResetMenu(QMenu):
                def exec(self, _):
                    return next(action for action in self.actions()
                                if action.text() == "重置照片…" and action.isEnabled())

            with patch("ui.main_window.QMenu", ResetMenu), patch(
                "ui.main_window.NoticeDialog.exec", return_value=QDialog.Accepted
            ):
                window._photo_context_menu(position)
            self.assertGreater(window._revision, 0)
            self.assertEqual(window._photos[paths[1]].revision, 0)
            self.assertFalse(window._photos[paths[1]].patches)
            self.assertFalse(second.exists())
            window.close()

            hidden.rename(second)
            window = open_window()
            finish_job(window)
            self.assertGreater(window._revision, 0)
            window.file_panel.select_file(second)
            finish_job(window)
            np.testing.assert_array_equal(window._current_image, image)
            np.testing.assert_array_equal(cv2.imread(str(second)), image)
            self.assertEqual(window._history.count(), 0)
            self.assertFalse(window._scan_source)
            self.assertFalse(window.reset_photo_action.isEnabled())
            repair(window)
            window.close()

    def test_manual_retouch_gestures_save_export_and_undo_native_pixels(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "scan.png"
            image = np.full((120, 160, 3), (13000, 17000, 21000), np.uint16)
            for x in range(15, 50):
                image[22:37, x] = (30000 + 203 * x, 25001 + 71 * x, 39999)
            image[74:77, 99:116] = 61000
            self.assertTrue(cv2.imwrite(str(path), image))

            def open_window():
                with patch("ui.main_window.Config", lambda: Config(directory)):
                    window = MainWindow()
                self.addCleanup(window._stop_workers)
                window.show()
                app.processEvents()
                return window

            def finish_job(window):
                deadline = time.monotonic() + 5
                while window._job is not None and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.01)
                self.assertIsNone(window._job)

            window = open_window()
            window.file_panel.load_files([path])
            finish_job(window)
            viewer = window.image_viewer
            viewer.zoom_100()
            window.retouch_radius.setValue(6)
            window.retouch_hardness.setValue(100)
            viewport = viewer.viewport()

            def at(x, y):
                return viewer.mapFromScene(QPointF(x, y))

            self.assertNotIn("paint", window.edit_buttons)
            window.edit_buttons["clone"].click()
            QTest.mouseMove(viewport, at(60, 45))
            self.assertEqual(viewport.cursor().shape(), Qt.BlankCursor)
            QTest.keyPress(viewport, Qt.Key_Alt)
            self.assertEqual(viewport.cursor().shape(), Qt.CrossCursor)
            QTest.keyRelease(viewport, Qt.Key_Alt)
            self.assertEqual(viewport.cursor().shape(), Qt.BlankCursor)
            small = viewport.grab().toImage()
            window.retouch_radius.setValue(18)
            large = viewport.grab().toImage()
            self.assertNotEqual(small.pixelColor(at(78, 45)), large.pixelColor(at(78, 45)))
            window.retouch_radius.setValue(6)
            QTest.mouseClick(viewport, Qt.LeftButton, pos=at(60, 45))
            self.assertEqual(window._revision, 0)
            QTest.mouseClick(viewport, Qt.LeftButton, Qt.AltModifier, at(23, 29))
            self.assertEqual(window._revision, 0)
            window.retouch_opacity.setValue(50)
            QTest.mouseClick(viewport, Qt.LeftButton, pos=at(35, 85))
            finish_job(window)
            expected = np.rint((image[29, 23].astype(float) + image[85, 35]) / 2)
            np.testing.assert_array_equal(window._current_image[85, 35], expected)
            np.testing.assert_array_equal(window._current_image[85, 42], image[85, 42])
            window._undo()
            np.testing.assert_array_equal(window._current_image, image)
            window._redo()
            np.testing.assert_array_equal(window._current_image[85, 35], expected)

            window.retouch_opacity.setValue(100)
            QTest.mousePress(viewport, Qt.LeftButton, pos=at(60, 45))
            QTest.mouseMove(viewport, at(76, 45))
            QTest.mouseRelease(viewport, Qt.LeftButton, pos=at(76, 45))
            finish_job(window)
            np.testing.assert_array_equal(window._current_image[45, 60:77], image[29, 23:40])
            np.testing.assert_array_equal(window._current_image[29, 23:40], image[29, 23:40])

            candidate = DustItem(1, 99, 74, 17, 3, 51, 107, 75, dust_type="scratch")
            window._on_detection_finished([candidate], None)
            window.edit_buttons["spot"].click()
            QTest.keyClick(viewer, Qt.Key_BracketRight)
            self.assertEqual(viewer.retouch_radius, 7)
            QTest.keyClick(viewer, Qt.Key_BracketLeft)
            self.assertEqual(viewer.retouch_radius, 6)
            QTest.mouseMove(viewport, at(98, 75))
            self.assertEqual(viewport.cursor().shape(), Qt.BlankCursor)
            QTest.mousePress(viewport, Qt.LeftButton, pos=at(98, 75))
            QTest.mouseRelease(viewport, Qt.LeftButton, pos=at(116, 75))
            finish_job(window)
            self.assertFalse(window._dust_items)
            self.assertLess(int(window._current_image[75, 107].max()), 23000)
            np.testing.assert_array_equal(window._current_image[69, 90:121], image[69, 90:121])
            window.btn_compare.click()
            self.assertIn(candidate.center, [mark.center for mark in viewer._before_marks])
            window.btn_compare.click()
            edited = window._current_image.copy()
            output = root / "export"
            window._start_export([str(path)], output, ".png", "edited")
            finish_job(window)
            np.testing.assert_array_equal(cv2.imread(str(output / "scan_clean.png"), cv2.IMREAD_UNCHANGED), edited)
            np.testing.assert_array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), image)
            window.close()

            window = open_window()
            finish_job(window)
            np.testing.assert_array_equal(window._current_image, edited)
            window._undo()
            self.assertEqual(window._dust_items[0].center, candidate.center)
            self.assertEqual(int(window._current_image[75, 107, 0]), 61000)
            window._redo()
            np.testing.assert_array_equal(window._current_image, edited)
            window.close()

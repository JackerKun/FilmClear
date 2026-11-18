"""Label a photo's line candidates and manage the locally trained model."""

from dataclasses import replace
from math import ceil
from pathlib import Path
import sqlite3

import cv2
from PySide6.QtCore import QEvent, QItemSelectionModel, QPointF, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QApplication, QButtonGroup, QDialog, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from core.fiber_learning import (
    LINE_TYPES, MODEL_NAME, FiberModelGroup, candidate_key, combine_fiber_models, fiber_features,
    read_fiber_model, train_fiber_model, write_fiber_model,
)
from core.scan_detection import detect_brush, detect_scan

from .image_viewer import ImageViewer
from .notice_dialog import NoticeDialog
from .numeric_fields import use_standard_digits
from .theme_manager import ThemeManager
from .tool_icons import tool_icon
from .workspace_job import WorkspaceJob


def _button(title, callback=None, style="secondaryBtn"):
    button = QPushButton(title)
    button.setObjectName(style)
    button.setAutoDefault(False)
    if callback is not None:
        button.clicked.connect(lambda: callback())
    return button


class FiberLabelDialog(QDialog):
    def __init__(self, parent, source, image, rows, store, photo, *, can_previous, can_next, model, params,
                 start_view=None):
        super().__init__(parent)
        self.setWindowTitle("毛絮学习 · 标注与训练")
        self.setObjectName("fiberLabelDialog")
        available = self.screen().availableGeometry()
        self.resize(min(1200, available.width() - 40), min(800, available.height() - 72))
        self.store, self.photo = store, photo
        self.image, self.params = image, params
        self._start_view = start_view
        self.saved_roll, self.labels = store.photo_labels(photo)
        self.navigation = 0
        self._can_previous, self._can_next = can_previous, can_next
        self._job = None
        self._checked = set()
        traces = {row[1]: row for row in store.photo_traces(photo)}
        self._traced = set(traces)
        by_key = {row[1]: row for row in rows}
        by_key.update(traces)
        self.rows = sorted(by_key.values(), key=lambda row: (
            row[1] in self.labels, abs(model.score(row[2]) - 0.5) if model else -row[0].confidence
        ))
        self.rows = [(replace(item, id=i), key, features)
                     for i, (item, key, features) in enumerate(self.rows, 1)]

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("fiberWorkflow")
        outer.addWidget(self.tabs, 1)
        annotation = QWidget()
        layout = QVBoxLayout(annotation)
        layout.setContentsMargins(12, 12, 12, 8)
        self.tabs.addTab(annotation, "照片标注")
        self.training = FiberTrainingPanel(store)
        self.training_index = self.tabs.addTab(self.training, "全部标注与训练")
        self.tabs.currentChanged.connect(self._change_page)
        self.training.busy_changed.connect(self._refresh)
        layout.setSpacing(10)
        header = QHBoxLayout()
        heading = QLabel(source.name)
        heading.setTextFormat(Qt.PlainText)
        heading.setObjectName("titleLabel")
        header.addWidget(heading, 1)
        self.previous = _button("上一张", lambda: self._leave(-1))
        self.previous.setEnabled(can_previous)
        header.addWidget(self.previous)
        self.next = _button("下一张", lambda: self._leave(1))
        self.next.setEnabled(can_next)
        header.addWidget(self.next)
        layout.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(10)
        view = QVBoxLayout()
        self.canvas = ImageViewer()
        self.canvas.setAccessibleName("原片与识别轮廓")
        self.canvas.load_image(image)
        self.canvas.annotation_clicked.connect(self._select_outline)
        self.canvas.stroke_drawn.connect(self._trace_stroke)
        self.canvas.region_selected.connect(self._trace_box)
        view.addWidget(self._canvas_tools())
        view.addWidget(self.canvas, 1)
        self.hint = QLabel()
        self.hint.setObjectName("subtitleLabel")
        self.hint.setWordWrap(True)
        self.hint.setMinimumHeight(18)
        view.addWidget(self.hint)
        body.addLayout(view, 1)

        card = QFrame()
        card.setObjectName("card")
        card.setFixedWidth(302)
        sidebar = QVBoxLayout(card)
        sidebar.setContentsMargins(12, 12, 12, 12)
        sidebar.setSpacing(8)
        self.roll = QLineEdit(self.saved_roll or str(source.parent))
        self.roll.setCursorPosition(len(self.roll.text()))
        self.roll.setToolTip("同卷照片使用同一个名称，训练时可按整卷留出验证。")
        self.roll.editingFinished.connect(self._save_roll)
        form = QFormLayout()
        form.addRow("胶卷分组", self.roll)
        sidebar.addLayout(form)
        title = QLabel("轮廓与样本")
        title.setObjectName("sectionLabel")
        sidebar.addWidget(title)
        self.candidates = QListWidget()
        self.candidates.setAccessibleName("识别轮廓与训练样本")
        self.candidates.setUniformItemSizes(True)
        sidebar.addWidget(self.candidates, 1)
        self.summary = QLabel()
        self.summary.setObjectName("subtitleLabel")
        sidebar.addWidget(self.summary)
        checked_actions = QHBoxLayout()
        self.save_defects = _button("保存为真实缺陷", lambda: self._save_checked(1), "")
        self.save_scene = _button("保存为画面线条", lambda: self._save_checked(0))
        checked_actions.addWidget(self.save_defects, 1)
        checked_actions.addWidget(self.save_scene, 1)
        sidebar.addLayout(checked_actions)
        review = QHBoxLayout()
        current = QLabel("当前一处")
        current.setObjectName("subtitleLabel")
        review.addWidget(current)
        self.label_buttons = []
        for label, title, key in ((1, "真实缺陷", "1"), (0, "画面线条", "2"), (None, "跳过", "3")):
            button = _button(title, lambda value=label: self._advance(value))
            button.setToolTip(f"标注当前一处 · {key}")
            shortcut = QShortcut(QKeySequence(key), self.candidates)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda value=label: self._advance(value))
            review.addWidget(button, 1)
            self.label_buttons.append(button)
        sidebar.addLayout(review)
        row = QHBoxLayout()
        self.clear_checked = _button("取消勾选", self._clear_checks, "textBtn")
        self.clear_label = _button("清除本条标注", self._clear_label, "textBtn")
        row.addWidget(self.clear_checked)
        row.addWidget(self.clear_label)
        row.addStretch()
        sidebar.addLayout(row)
        body.addWidget(card)
        layout.addLayout(body, 1)

        footer = QHBoxLayout()
        self.feedback = QLabel()
        self.feedback.setObjectName("subtitleLabel")
        self.feedback.setWordWrap(True)
        footer.addWidget(self.feedback, 1)
        self.close_button = _button("关闭", self._close_or_cancel)
        layout.addLayout(footer)
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_row.addWidget(self.close_button)
        outer.addLayout(close_row)
        self.candidates.addItems([""] * len(self.rows))
        for index in range(len(self.rows)):
            self._update_row(index)
        self.candidates.currentRowChanged.connect(self._focus_candidate)
        self.candidates.itemChanged.connect(self._change_check)
        self.modes.buttonToggled.connect(self._change_mode)
        self.draw_button.setChecked(True)
        self.feedback.setText("描画或框选得到的轮廓会自动勾选。核对后可一次保存。")
        self.canvas.setFocus()

    def _change_page(self, index):
        if index == self.training_index:
            self.canvas.set_space_pan(False)
            self.training.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        QApplication.instance().installEventFilter(self)
        if self._start_view is not None:
            self.canvas.restore_view(self._start_view)
            self._start_view = None

    def hideEvent(self, event):
        QApplication.instance().removeEventFilter(self)
        self.canvas.set_space_pan(False)
        super().hideEvent(event)

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.WindowDeactivate and watched is self:
            self.canvas.set_space_pan(False)
        if (kind in (QEvent.ShortcutOverride, QEvent.KeyPress, QEvent.KeyRelease)
                and self.tabs.currentIndex() == 0
                and isinstance(watched, QWidget) and watched.window() is self
                and event.key() == Qt.Key_Space and self._job is None
                and (self.canvas.space_pan or (event.modifiers() == Qt.NoModifier
                     and not isinstance(watched, (QLineEdit, QAbstractSpinBox))))):
            if kind != QEvent.ShortcutOverride and not event.isAutoRepeat():
                self.canvas.set_space_pan(kind == QEvent.KeyPress)
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _canvas_tools(self):
        tools = QFrame()
        tools.setObjectName("canvasTools")
        row = QHBoxLayout(tools)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(4)
        self.modes = QButtonGroup(tools)
        color = "#e5edf4" if ThemeManager().is_dark else "#22323e"
        for title, mode, icon in (("沿毛絮描画", "draw", "draw"), ("框选查找", "detect", "detect"),
                                  ("查看与移动", "", "pointer")):
            button = _button(title, style="toolBtn")
            button.setIcon(tool_icon(icon, color))
            button.setCheckable(True)
            button.setProperty("selectionMode", mode)
            self.modes.addButton(button)
            row.addWidget(button)
            if mode == "draw":
                self.draw_button = button
        self.radius = QSpinBox()
        use_standard_digits(self.radius)
        self.radius.setRange(2, 120)
        self.radius.setValue(8)
        self.radius.setPrefix("半径 ")
        self.radius.setSuffix(" px")
        self.radius.setToolTip("画笔在屏幕上的覆盖半径")
        self.radius.valueChanged.connect(self._set_brush_radius)
        self._set_brush_radius(self.radius.value())
        row.addWidget(self.radius)
        row.addStretch()
        for icon, title, action in (("fit", "适合窗口", self.canvas.fit_to_window),
                                    ("zoom-out", "缩小", self.canvas.zoom_out),
                                    ("zoom-in", "放大", self.canvas.zoom_in)):
            button = _button("", action, "toolBtn")
            button.setIcon(tool_icon(icon, color))
            button.setToolTip(title)
            button.setAccessibleName(title)
            row.addWidget(button)
        zoom = QLabel(f"{self.canvas.get_zoom_percent()}%")
        zoom.setObjectName("zoomLabel")
        zoom.setMinimumWidth(35)
        self.canvas.zoom_changed.connect(lambda percent: zoom.setText(f"{percent}%"))
        row.addWidget(zoom)
        return tools

    def _set_brush_radius(self, radius):
        self.canvas.brush_radius = radius
        self.canvas.viewport().update()

    def _change_mode(self, button, checked):
        if not checked:
            return
        mode = button.property("selectionMode")
        self.canvas.set_selection_mode(mode)
        self.radius.setEnabled(mode == "draw")
        self.hint.setText({
            "draw": "沿毛絮拖动画笔，松开后识别轮廓。按住空格可移动照片。",
            "detect": "拖出一个框，查找框内的毛絮和划痕。按住空格可移动照片。",
            "": "拖动画布移动，点击轮廓定位到列表。",
        }[mode])
        self._refresh()

    def _trace_stroke(self, points):
        radius = ceil(self.canvas.brush_radius / self.canvas.transform().m11())
        self._find_outlines(brush=(points, radius))

    def _trace_box(self, x1, y1, x2, y2, _additive):
        self._find_outlines(box=(x1, y1, x2, y2))

    def _find_outlines(self, *, brush=None, box=None):
        if self._job is not None:
            return
        image, params = self.image, self.params

        def work(progress):
            def report(stage, done, total):
                progress(f"正在识别轮廓 · {stage}", done, total)

            if brush is not None:
                result = detect_brush(image, params, *brush, progress=report, guided=True)
            else:
                x1, y1, x2, y2 = box
                h, w = image.shape[:2]
                region = (max(0, x1 - 32), max(0, y1 - 32), min(w, x2 + 32), min(h, y2 + 32))
                result = detect_scan(image, params, region=region, progress=report)
                result.items = [item for item in result.items
                                if x1 <= item.center_x < x2 and y1 <= item.center_y < y2]
            return [(item, candidate_key(item), fiber_features(image, item)) for item in result.items
                    if item.dust_type in LINE_TYPES and item.contour is not None]

        job = self._job = WorkspaceJob(work, self)
        job.progress.connect(self._report_search)
        job.completed.connect(self._add_outlines)
        job.failed.connect(self._search_failed)
        job.finished.connect(self._search_finished)
        self.canvas.setEnabled(False)
        self.canvas.set_space_pan(False)
        self.close_button.setText("停止查找")
        self.feedback.setText("正在原片中查找轮廓…")
        self._refresh()
        job.start()

    def _report_search(self, message):
        if self._job is not None and not self._job.isInterruptionRequested():
            self.feedback.setText(message)

    def _search_failed(self, error):
        self.feedback.setText(f"轮廓识别失败 · {error}")

    def _search_finished(self):
        self._job.deleteLater()
        self._job = None
        self.canvas.setEnabled(True)
        self.canvas.setFocus()
        self.close_button.setText("关闭")
        self._refresh()

    @staticmethod
    def _same_outline(a, b):
        width = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
        height = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
        if width <= 0 or height <= 0:
            return False
        overlap = width * height
        return (overlap >= 0.72 * (a.width * a.height + b.width * b.height - overlap)
                and cv2.pointPolygonTest(a.contour, b.center, True) >= -2
                and cv2.pointPolygonTest(b.contour, a.center, True) >= -2)

    def _add_outlines(self, found):
        if not found:
            self.feedback.setText("这一笔没有找到连续轮廓。可以放大后贴近毛絮重描，或用框选查找。")
            return
        first = None
        added = 0
        with QSignalBlocker(self.candidates):
            for item, key, features in found:
                index = next((i for i, (_, known, _) in enumerate(self.rows) if known == key), None)
                if index is None:
                    index = next((i for i, (known, _, _) in enumerate(self.rows)
                                  if self._same_outline(item, known)), None)
                if index is None:
                    index = len(self.rows)
                    self.rows.append((replace(item, id=index + 1, selected=False), key, features))
                    self.candidates.addItem("")
                else:
                    known, key, _ = self.rows[index]
                    if key not in self._traced:
                        self.rows[index] = (replace(item, id=known.id, selected=known.selected,
                                                    face_protected=known.face_protected,
                                                    scene_line=known.scene_line), key, features)
                self._traced.add(key)
                added += key not in self._checked
                self._checked.add(key)
                self._update_row(index)
                if first is None:
                    first = index
            self.candidates.setCurrentRow(first)
        self._refresh()
        self.feedback.setText(f"本次查找 {len(found)} 处轮廓，新增勾选 {added} 处")

    def _update_row(self, index):
        item, key, _ = self.rows[index]
        label = {0: "画面线条", 1: "真实缺陷"}.get(self.labels.get(key), "待标注")
        row = self.candidates.item(index)
        with QSignalBlocker(self.candidates):
            row.setText(f"{item.type_name}  {item.width} × {item.height} px  ·  {label}")
            row.setToolTip(f"#{item.id:03d} · " + ("手动查找" if key in self._traced else "全图扫描"))
            row.setCheckState(Qt.Checked if key in self._checked else Qt.Unchecked)

    def _change_check(self, index_item):
        index = self.candidates.row(index_item)
        _, key, _ = self.rows[index]
        if index_item.checkState() == Qt.Checked:
            self._checked.add(key)
        else:
            self._checked.discard(key)
        self._refresh()

    def _refresh(self):
        index = self.candidates.currentRow()
        ready = self._job is None
        self.tabs.setTabEnabled(0, not self.training.busy)
        self.tabs.setTabEnabled(self.training_index, ready)
        self.close_button.setEnabled(not self.training.busy)
        self.previous.setEnabled(ready and not self.training.busy and self._can_previous)
        self.next.setEnabled(ready and not self.training.busy and self._can_next)
        for button in self.label_buttons:
            button.setEnabled(index >= 0 and ready)
        self.clear_label.setEnabled(index >= 0 and ready and self.rows[index][1] in self.labels)
        for button in (self.save_defects, self.save_scene, self.clear_checked):
            button.setEnabled(bool(self._checked) and ready)
        self.summary.setText(f"已勾选 {len(self._checked)} 处   ·   已保存 {len(self.labels)} 处")
        focused = self.rows[index][0].id if index >= 0 else None
        all_marks = self.modes.checkedButton().property("selectionMode") == ""
        self.canvas.update_annotations([
            replace(item, selected=key in self._checked or self.labels.get(key) == 1)
            for item, key, _ in self.rows
            if all_marks or item.id == focused or key in self._checked
        ])
        if focused is None:
            self.canvas.reset_highlight()
        else:
            self.canvas.highlight_dust(focused)

    def _select_outline(self, dust_id):
        self.candidates.setCurrentRow(dust_id - 1)

    def _focus_candidate(self, index):
        self._refresh()
        if index >= 0:
            item = self.rows[index][0]
            view = self.canvas.viewport()
            scale = min(view.width() / (item.width + 90), view.height() / (item.height + 90), 2.5)
            self.canvas.highlight_dust(item.id, locate=True, min_zoom=scale)

    def _save_roll(self):
        roll = self.roll.text().strip()
        if roll and self.labels and roll != self.saved_roll:
            try:
                self.store.set_roll(self.photo, roll)
            except (OSError, sqlite3.Error) as error:
                NoticeDialog(self, "分组保存失败", str(error), warning=True).exec()
                return
            self.saved_roll = roll

    def _save_samples(self, indexes, label):
        roll = self.roll.text().strip()
        if not roll:
            self.feedback.setText("请填写胶卷分组。")
            self.roll.setFocus()
            return False
        samples = []
        for index in indexes:
            item, key, features = self.rows[index]
            trace = item if key in self._traced else None
            samples.append((key, item.selected and not item.requires_review, features, trace))
        try:
            previews = {self.rows[index][1]: self.rows[index][0].contour for index in indexes}
            self.store.label_many(self.photo, roll, label, samples, previews=previews)
        except (OSError, ValueError, sqlite3.Error) as error:
            NoticeDialog(self, "标注保存失败", str(error), warning=True).exec()
            return False
        self.saved_roll = roll
        for index, (key, _, _, _) in zip(indexes, samples):
            self.labels[key] = label
            self._checked.discard(key)
            self._update_row(index)
        self._refresh()
        self.feedback.setText(f"已保存 {len(indexes)} 处" + ("真实缺陷" if label == 1 else "画面线条"))
        return True

    def _save_checked(self, label):
        indexes = [index for index, (_, key, _) in enumerate(self.rows) if key in self._checked]
        if indexes and self._job is None:
            self._save_samples(indexes, label)

    def _clear_checks(self):
        self._checked.clear()
        for index in range(len(self.rows)):
            self._update_row(index)
        self._refresh()

    def _advance(self, label):
        index = self.candidates.currentRow()
        if index < 0 or self._job is not None:
            return
        if label is not None and not self._save_samples([index], label):
            return
        if index + 1 < len(self.rows):
            self.candidates.setCurrentRow(index + 1)
        self.candidates.setFocus()

    def _clear_label(self):
        index = self.candidates.currentRow()
        if index >= 0:
            _, key, _ = self.rows[index]
            try:
                self.store.remove_label(self.photo, key)
            except (OSError, sqlite3.Error) as error:
                NoticeDialog(self, "清除标注失败", str(error), warning=True).exec()
                return
            self.labels.pop(key, None)
            self._update_row(index)
            self._refresh()

    def _can_leave(self):
        if self._job is not None or self.training.busy:
            return False
        if self._checked:
            dialog = NoticeDialog(self, "有勾选轮廓尚未保存", "返回标注可先保存已勾选的样本。",
                                  confirm="丢弃勾选并离开", cancel="返回标注")
            if dialog.exec() != QDialog.Accepted:
                return False
        self._save_roll()
        return True

    def reject(self):
        if self._can_leave():
            super().reject()

    def _close_or_cancel(self):
        if self._job is None:
            self.reject()
        else:
            self._job.requestInterruption()
            self.feedback.setText("已停止本次查找")

    def _leave(self, direction):
        if self._can_leave():
            self.navigation = direction
            self.accept()


def validation_text(model):
    if model is None:
        return "暂无训练结果"
    stats = model.metrics
    samples = f"{stats['samples']} 处训练样本 · 真缺陷 {stats['defects']} · 场景线 {stats['scene']}"
    if model.templates:
        samples += f"\n已学习 {len(model.templates)} 个轮廓模板 · 补检结果需核对"
    unit = {"roll": "按整卷留出", "photo": "按整张照片留出", "none": "尚无可用的留出评估"}[stats["validation"]]
    if not stats["evaluated"]:
        return f"{samples}\n{unit}"
    return (f"{samples}\n候选复判{unit} · 评估 {stats['evaluated']} 处\n"
            f"默认勾选的场景线 {stats['baseline_scene']} → {stats['baseline_scene'] - stats['scene_deferred']}\n"
            f"默认勾选的真缺陷 {stats['baseline_defects']} → {stats['baseline_defects'] - stats['defects_deferred']}")


def _outline_icon(outline, features, label):
    pixmap = QPixmap(92, 56)
    dark = ThemeManager().is_dark
    pixmap.fill(QColor("#24333d" if dark else "#eaf2f3"))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    ink = QColor("#86e1cc" if dark else "#197565")
    if label == 0:
        ink = QColor("#fac078" if dark else "#9b591e")
    thickness = min(4.5, max(1.4, features[1])) if features is not None and not outline else 2.0
    painter.setPen(QPen(ink, thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    if outline:
        w = max(x for x, _ in outline)
        h = max(y for _, y in outline)
        scale = min(68 / max(1, w), 38 / max(1, h))
        left, top = (92 - w * scale) / 2, (56 - h * scale) / 2
        points = [QPointF(left + x * scale, top + y * scale) for x, y in outline]
        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)
        path.closeSubpath()
        fill = QColor(ink)
        fill.setAlpha(85)
        painter.fillPath(path, fill)
    elif features is not None and features[8] >= .5:
        span = min(36, max(17, 8 * features[0] - 12))
        rise = 5 + 15 * min(1, max(0, features[6]))
        path = QPainterPath(QPointF(46 - span, 28 + rise))
        path.lineTo(46 + span, 28 - rise)
    else:
        path = QPainterPath(QPointF(21, 37))
        path.cubicTo(33, 11, 47, 12, 60, 16)
        path.cubicTo(80, 21, 65, 42, 51, 39)
    painter.drawPath(path)
    painter.end()
    return pixmap


class FiberOutlineGallery(QWidget):
    def __init__(self, rows, empty_text):
        super().__init__()
        self._items = []
        self._page = 0
        self._page_size = rows * 3
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        gallery = QFrame()
        gallery.setObjectName("card")
        grid = QGridLayout(gallery)
        self.tiles = []
        for index in range(self._page_size):
            tile = QWidget()
            content = QVBoxLayout(tile)
            content.setAlignment(Qt.AlignCenter)
            content.setSpacing(4)
            icon = QLabel(alignment=Qt.AlignCenter)
            caption = QLabel(alignment=Qt.AlignCenter)
            caption.setObjectName("sectionLabel")
            content.addWidget(icon)
            content.addWidget(caption)
            self.tiles.append((tile, icon, caption))
            grid.addWidget(tile, index // 3, index % 3)
        for row in range(rows):
            grid.setRowStretch(row, 1)
        self.empty = QLabel(empty_text, alignment=Qt.AlignCenter)
        self.empty.setWordWrap(True)
        self.empty.setObjectName("subtitleLabel")
        grid.addWidget(self.empty, 0, 0, rows, 3)
        layout.addWidget(gallery, 1)

        paging = QHBoxLayout()
        paging.addStretch()
        self.back = _button("上一页", lambda: self._turn_page(-1))
        self.forward = _button("下一页", lambda: self._turn_page(1))
        self.page_text = QLabel()
        self.page_text.setObjectName("subtitleLabel")
        for widget in (self.back, self.page_text, self.forward):
            paging.addWidget(widget)
        layout.addLayout(paging)
        self._paint_page()

    def set_items(self, items, *, reset_page=False):
        self._items = items
        if reset_page:
            self._page = 0
        self._paint_page()

    def _turn_page(self, step):
        self._page += step
        self._paint_page()

    def _paint_page(self):
        count = len(self._items)
        pages = ceil(count / self._page_size)
        self._page = max(0, min(self._page, pages - 1))
        self.page_text.setText(f"{self._page + 1} / {pages}" if count else "0 / 0")
        self.back.setEnabled(self._page > 0)
        self.forward.setEnabled(self._page + 1 < pages)
        self.empty.setVisible(not count)
        start = self._page * self._page_size
        for offset, (tile, icon, caption) in enumerate(self.tiles):
            index = start + offset
            tile.setVisible(index < count)
            if index < count:
                name, outline, features, label, tooltip = self._items[index]
                caption.setText(f"{index + 1}. {name}" + ("" if outline else " · 示意"))
                tile.setToolTip(tooltip)
                icon.setPixmap(_outline_icon(outline, features, label))


class FiberTrainingPanel(QWidget):
    busy_changed = Signal()

    def __init__(self, store):
        super().__init__()
        self.store = store
        self._examples = []
        self._job = None
        self._model = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        self.summary = QLabel()
        self.summary.setObjectName("titleLabel")
        layout.addWidget(self.summary)
        self.details = QLabel()
        self.details.setObjectName("subtitleLabel")
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.gallery = FiberOutlineGallery(3, "在照片标注中保存轮廓后，可在这里核对并训练。")
        layout.addWidget(self.gallery, 1)
        footer = QHBoxLayout()
        self.feedback = QLabel("核对标注后，可用这一批样本训练。")
        self.feedback.setWordWrap(True)
        self.feedback.setObjectName("subtitleLabel")
        footer.addWidget(self.feedback, 1)
        self.export = _button("导出训练结果…", self._export_result)
        self.train = _button("训练模型", self._train, "")
        footer.addWidget(self.export)
        footer.addWidget(self.train)
        layout.addLayout(footer)
        self.export.setEnabled(False)
        self.train.setEnabled(False)

    @property
    def busy(self):
        return self._job is not None

    def refresh(self):
        if self.busy:
            return
        try:
            self._examples = self.store.examples()
        except (OSError, ValueError, sqlite3.Error) as error:
            self._examples = []
            self.feedback.setText(f"读取标注失败 · {error}")
        if self._model is None:
            try:
                self._model = self.store.latest_trained()
            except (OSError, ValueError, sqlite3.Error) as error:
                self.feedback.setText(f"读取上次训练结果失败 · {error}")
        defects = sum(example.label == 1 for example in self._examples)
        scenes = len(self._examples) - defects
        self.summary.setText(f"已保存 {len(self._examples)} 处标注")
        details = f"真实缺陷 {defects} · 画面线条 {scenes} · 胶卷 {len({e.roll for e in self._examples})}"
        approximate = sum(not example.outline for example in self._examples)
        if approximate:
            details += f" · {approximate} 处显示示意图"
        self.details.setText(details)
        self.gallery.set_items([
            ("真实缺陷" if example.label == 1 else "画面线条",
             example.outline, example.features, example.label, example.roll)
            for example in self._examples
        ])
        self._update_actions()

    def _update_actions(self):
        labels = {example.label for example in self._examples}
        can_train = labels == {0, 1} or any(example.label == 1 and example.traced
                                            for example in self._examples)
        self.train.setEnabled(can_train and not self.busy)
        self.train.setToolTip("" if can_train else "保存一处手动查找的真实缺陷，或分别标注真实缺陷和画面线条。")
        self.export.setEnabled(self._model is not None and not self.busy)

    def _train(self):
        examples = tuple(self._examples)

        def work(progress):
            model = train_fiber_model(examples, lambda done, total: progress(f"正在验证模型 · {done} / {total}"))
            self.store.save_trained(model)
            return model

        job = self._job = WorkspaceJob(work, self)
        job.progress.connect(self.feedback.setText)
        job.completed.connect(self._training_done)
        job.failed.connect(lambda message: self.feedback.setText(f"训练失败 · {message}"))
        job.finished.connect(self._training_finished)
        self.feedback.setText("正在训练毛絮模型…")
        self._update_actions()
        self.busy_changed.emit()
        job.start()

    def _training_done(self, model):
        self._model = model
        self.feedback.setText(f"已保存到模型管理。\n{validation_text(model)}")

    def _training_finished(self):
        self._job.deleteLater()
        self._job = None
        self._update_actions()
        self.busy_changed.emit()

    def _export_result(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出训练结果", str(Path.home() / MODEL_NAME), "FilmClear 模型 (*.json)"
        )
        if not path:
            return
        try:
            write_fiber_model(path, self._model)
        except (OSError, ValueError) as error:
            self.feedback.setText(f"导出失败 · {error}")
        else:
            self.feedback.setText("训练结果已导出。")


class FiberTrainingDialog(QDialog):
    """Open saved annotations when the workspace has no current photo."""

    def __init__(self, parent, store):
        super().__init__(parent)
        self.setWindowTitle("毛絮学习 · 标注与训练")
        self.setObjectName("fiberLabelDialog")
        available = self.screen().availableGeometry()
        self.resize(min(710, available.width() - 40), min(580, available.height() - 65))
        layout = QVBoxLayout(self)
        self.panel = FiberTrainingPanel(store)
        layout.addWidget(self.panel)
        close = _button("关闭", self.reject)
        layout.addWidget(close, alignment=Qt.AlignRight)
        self.panel.busy_changed.connect(lambda: close.setEnabled(not self.panel.busy))
        self.panel.refresh()

    def reject(self):
        if not self.panel.busy:
            super().reject()


class FiberModelDialog(QDialog):
    active_changed = Signal(object)

    def __init__(self, parent, store, directory):
        super().__init__(parent)
        self.setWindowTitle("毛絮学习 · 模型管理")
        self.setObjectName("fiberModelDialog")
        available = self.screen().availableGeometry()
        self.resize(min(880, available.width() - 40), min(550, available.height() - 65))
        self.store = store
        self.directory = directory
        self._entries = {}
        self._preview_id = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)
        header = QHBoxLayout()
        heading = QLabel("模型管理")
        heading.setObjectName("titleLabel")
        header.addWidget(heading, 1)
        self.status = QLabel()
        header.addWidget(self.status)
        layout.addLayout(header)

        body = QHBoxLayout()
        library = QVBoxLayout()
        library.addWidget(QLabel("模型列表"))
        self.models = QListWidget()
        self.models.setAccessibleName("毛絮模型列表")
        self.models.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.models.setMinimumWidth(210)
        library.addWidget(self.models, 1)
        body.addLayout(library, 2)

        preview = QVBoxLayout()
        self.preview_title = QLabel("缺陷预览")
        self.preview_title.setTextFormat(Qt.PlainText)
        self.preview_title.setWordWrap(True)
        self.preview_title.setObjectName("sectionLabel")
        preview.addWidget(self.preview_title)
        self.source = QLineEdit()
        self.source.setReadOnly(True)
        self.source.setAccessibleName("模型来源")
        self.source.setPlaceholderText("选中模型后显示来源")
        preview.addWidget(self.source)
        self.gallery = FiberOutlineGallery(2, "选中模型即可预览缺陷轮廓。")
        preview.addWidget(self.gallery, 1)
        body.addLayout(preview, 3)
        layout.addLayout(body, 1)

        self.feedback = QLabel("启用模型后，重新进行全图 RGB 扫描。按住 Ctrl 或 ⌘ 可多选。")
        self.feedback.setObjectName("subtitleLabel")
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)

        actions = QHBoxLayout()
        self.import_button = _button("导入 JSON…", self._import_models)
        self.delete = _button("删除所选…", self._delete_models)
        self.enable = _button("启用所选", lambda: self._set_enabled(True), "")
        self.disable = _button("停用所选", lambda: self._set_enabled(False))
        self.close_button = _button("关闭", self.reject)
        actions.addWidget(self.import_button)
        actions.addWidget(self.delete)
        actions.addStretch()
        for button in (self.enable, self.disable, self.close_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.models.currentItemChanged.connect(self._show_model)
        self.models.itemSelectionChanged.connect(self._update_actions)
        self._reload()

    def _selection(self):
        return [self._entries[item.data(Qt.UserRole)] for item in self.models.selectedItems()]

    def _reload(self, selected=None):
        current = self.models.currentItem()
        current_id = current.data(Qt.UserRole) if current is not None else None
        if selected is None:
            selected = [entry.id for entry in self._selection()]
        self._entries = {entry.id: entry for entry in self.store.library()}
        with QSignalBlocker(self.models):
            self.models.clear()
            focus = None
            for entry in self._entries.values():
                state = "已启用" if entry.enabled else "已停用"
                item = QListWidgetItem(f"{entry.name}\n{state}")
                item.setData(Qt.UserRole, entry.id)
                item.setToolTip(entry.source or "保存在本机模型库")
                self.models.addItem(item)
                if entry.id in selected:
                    item.setSelected(True)
                    if focus is None or entry.id == current_id:
                        focus = item
            if focus is None and self.models.count():
                focus = self.models.item(0)
                focus.setSelected(True)
            if focus is not None:
                self.models.setCurrentItem(focus, QItemSelectionModel.NoUpdate)
        self._show_model(focus)
        self._update_actions()
        enabled = [entry.model for entry in self._entries.values() if entry.enabled]
        self.status.setText(f"{len(enabled)} 个模型已启用" if enabled else "使用扫描规则")
        self.status.setObjectName("fiberStatusActive" if enabled else "fiberStatusIdle")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.active_changed.emit(combine_fiber_models(enabled))

    def _show_model(self, item, _previous=None):
        entry = self._entries[item.data(Qt.UserRole)] if item is not None else None
        source = (entry.source or "本机模型库") if entry is not None else ""
        self.source.setText(source)
        self.source.setCursorPosition(len(source))
        self.source.setToolTip(source)
        self.preview_title.setText(f"缺陷预览 · {entry.name}" if entry is not None else "缺陷预览")
        current_id = entry.id if entry is not None else None
        previews = []
        if entry is not None:
            model = entry.model
            members = model.members if isinstance(model, FiberModelGroup) else (model,)
            outlines = {}
            for member in members:
                for index, features in enumerate(member.templates):
                    outline = member.outlines[index] if member.outlines else ()
                    if outline or features not in outlines:
                        outlines[features] = outline
            for features, outline in outlines.items():
                kind = "划痕" if features[8] >= .5 else "毛絮"
                description = "模型保存的缺陷轮廓" if outline else "此模板仅保存特征参数，图形为示意。"
                previews.append((kind, outline, features, 1, description))
            if not previews:
                previews.append(("线状缺陷", (), None, 1, "此模型仅用于候选复判。"))
        self.gallery.set_items(previews, reset_page=current_id != self._preview_id)
        self._preview_id = current_id

    def _update_actions(self):
        entries = self._selection()
        self.delete.setEnabled(bool(entries))
        self.enable.setEnabled(any(not entry.enabled for entry in entries))
        self.disable.setEnabled(any(entry.enabled for entry in entries))

    def _run_change(self, action, message):
        try:
            self._reload(action())
        except (OSError, ValueError, sqlite3.Error) as error:
            NoticeDialog(self, "模型操作失败", str(error), warning=True).exec()
        else:
            self.feedback.setText(message)

    def _set_enabled(self, enabled):
        ids = [entry.id for entry in self._selection()]
        message = "模型已启用。重新扫描后生效。" if enabled else "模型已停用。重新扫描后生效。"
        self._run_change(lambda: self.store.set_enabled(ids, enabled), message)

    def _delete_models(self):
        entries = self._selection()
        if not entries:
            return
        prompt = NoticeDialog(
            self, "删除所选模型",
            f"从本机模型库删除 {len(entries)} 个模型。原始 JSON 文件和本机标注会保留。",
            confirm="删除模型", details="\n".join(entry.name for entry in entries),
        )
        if prompt.exec() == QDialog.Accepted:
            self._run_change(lambda: self.store.remove_models([entry.id for entry in entries]),
                             "已从模型库删除所选模型。重新扫描后生效。")

    def _import_models(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "导入毛絮模型", self.directory, "FilmClear 模型 (*.json)")
        if paths:
            self.directory = str(Path(paths[0]).parent)
            self._run_change(
                lambda: self.store.add_models([(Path(path).stem, path, read_fiber_model(path)) for path in paths]),
                "导入完成。选中模型后可直接启用。",
            )

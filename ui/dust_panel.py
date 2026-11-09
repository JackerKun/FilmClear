"""Candidate review, local detail preview, and repair controls."""

from dataclasses import replace

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from core.dust_remover import DustRemover
from core.scan_io import display_pixels

from .image_viewer import photo_pixmap


class InspectionComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self.setObjectName("inspectionCombo")

    def paintEvent(self, event):
        super().paintEvent(event)
        x, y = self.width() - 14, self.height() / 2
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(self.palette().buttonText().color(), 1.5, Qt.SolidLine, Qt.RoundCap))
        tip = QPointF(x, y + 2)
        painter.drawLine(QPointF(x - 3.5, y - 1.5), tip)
        painter.drawLine(tip, QPointF(x + 3.5, y - 1.5))


class CandidatePreview(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(84)
        self.setMaximumHeight(132)
        self.setAccessibleName("缺陷局部放大预览")
        self._pixmap = None
        self._comparison = ()
        self._target = None
        self._rect = QRectF()
        self._bounds = QRectF()

    def sizeHint(self):
        return QSize(230, 106)

    def show_candidate(self, image=None, item=None, *, original=None, repair_params=None):
        self._pixmap = None
        self._comparison = ()
        self._target = None
        if image is not None and item is not None:
            h, w = image.shape[:2]
            x, y, width, height = item.bbox
            margin_x, margin_y = max(36, width // 2), max(24, height // 2)
            aspect = max(1.25, (self.width() - 16) / (self.height() - 16))
            crop_w = max(width + 2 * margin_x, round((height + 2 * margin_y) * aspect))
            crop_h = max(height + 2 * margin_y, round(crop_w / aspect))
            crop_w, crop_h = min(w, crop_w), min(h, crop_h)
            x1 = min(w - crop_w, max(0, x + (width - crop_w) // 2))
            y1 = min(h - crop_h, max(0, y + (height - crop_h) // 2))
            x2, y2 = x1 + crop_w, y1 + crop_h
            self._pixmap = photo_pixmap(image[y1:y2, x1:x2])
            self._target = item.center
            self._bounds = QRectF(*item.bbox)
            self._rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            if original is not None and repair_params is not None:
                current = image[y1:y2, x1:x2]
                before = original[y1:y2, x1:x2]
                factor = min(1, 640 / max(crop_w, crop_h))

                def mapped(contour):
                    return np.rint((contour - (x1, y1)) * factor).astype(np.int32)

                if factor < 1:
                    size = (max(1, round(crop_w * factor)), max(1, round(crop_h * factor)))
                    current = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
                    before = cv2.resize(before, size, interpolation=cv2.INTER_AREA)
                local = replace(
                    item, x=round((x - x1) * factor), y=round((y - y1) * factor),
                    width=max(1, round(width * factor)), height=max(1, round(height * factor)),
                    contour=mapped(item.contour) if item.contour is not None else None,
                    holes=tuple(mapped(hole) for hole in item.holes),
                )
                remover = DustRemover(**repair_params)
                mask = remover.create_mask(current.shape, [local], selected_only=False,
                                           padding=max(1, round(2 * factor)))
                marked = display_pixels(current).copy()
                area = mask != 0
                marked[area] = (marked[area] * 0.32 + np.array([176, 242, 99]) * 0.68).astype(np.uint8)
                fixed = remover.apply_mask(current, mask)
                self._comparison = tuple(photo_pixmap(pixels) for pixels in (before, marked, fixed))
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#171e26"))
        if self._comparison:
            self._paint_comparison(painter)
            return
        if self._pixmap is None:
            painter.setPen(QColor("#a4b1bf"))
            painter.drawText(self.rect(), Qt.AlignCenter, "悬停候选，查看局部细节")
            return

        scale = min((self.width() - 16) / max(self._rect.width(), 1.42 * self._bounds.width()),
                    (self.height() - 16) / max(self._rect.height(), 1.42 * self._bounds.height()))
        x = (self.width() - self._rect.width() * scale) / 2
        y = (self.height() - self._rect.height() * scale) / 2
        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.translate(x, y)
        painter.scale(scale, scale)
        painter.translate(-self._rect.x(), -self._rect.y())
        painter.drawPixmap(self._rect, self._pixmap, QRectF(self._pixmap.rect()))
        painter.restore()
        center = self._bounds.center()
        px = x + (center.x() - self._rect.x()) * scale
        py = y + (center.y() - self._rect.y()) * scale
        rx = max(15.0, self._bounds.width() * scale * 0.71 + 3)
        ry = max(15.0, self._bounds.height() * scale * 0.71 + 3)
        ring = QRectF(px - rx, py - ry, 2 * rx, 2 * ry)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.NoBrush)
        for color, width in (("#14232b", 5), ("#70f4d9", 2.5)):
            painter.setPen(QPen(QColor(color), width))
            painter.drawEllipse(ring)


    def _paint_comparison(self, painter):
        gap = 5
        width = (self.width() - gap * 4) / 3
        height = self.height() - 27
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setPen(QColor("#dbe6ee"))
        for index, (label, pixmap) in enumerate(zip(("原片", "蒙版", "试修"), self._comparison)):
            x = gap + index * (width + gap)
            scale = min(width / pixmap.width(), height / pixmap.height())
            w, h = pixmap.width() * scale, pixmap.height() * scale
            painter.drawText(QRectF(x, 2, width, 19), Qt.AlignCenter, label)
            painter.drawPixmap(QRectF(x + (width - w) / 2, 23 + (height - h) / 2, w, h),
                               pixmap, QRectF(pixmap.rect()))


class CandidateList(QListWidget):
    hover_changed = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setUniformItemSizes(True)
        self._hover_id = -1
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(lambda: self._set_hover(-1))
        self.itemEntered.connect(self._enter_item)
        self.viewportEntered.connect(lambda: self._timer.start(80))

    def _set_hover(self, value):
        if value != self._hover_id:
            self._hover_id = value
            self.hover_changed.emit(value)

    def _enter_item(self, item):
        self._timer.stop()
        self._set_hover(item.data(Qt.UserRole))

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._timer.start(80)

    def reset_hover(self):
        self._timer.stop()
        self._hover_id = -1


class DustPanel(QWidget):
    dust_selected = Signal(int, bool)
    dust_hovered = Signal(int)
    dust_clicked = Signal(int)
    dust_reviewed = Signal(int)
    dismiss_clicked = Signal(int)
    select_all_clicked = Signal(bool)
    remove_clicked = Signal()
    export_clicked = Signal()
    inspection_changed = Signal()
    preview_mode_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self._items = {}
        self._detected = False
        self._setup_ui()

    def _scan_controls(self):
        grid = QGridLayout()
        grid.setSpacing(6)
        for column in range(4):
            grid.setColumnStretch(column, 1)

        self.combo_sensitivity = InspectionComboBox()
        self.combo_sensitivity.addItem("精选候选", (20, 0.34))
        self.combo_sensitivity.addItem("均衡扫描", (20, 0.29))
        self.combo_sensitivity.addItem("细查微尘", (20, 0.25))
        self.combo_sensitivity.setToolTip("精选侧重减少误报，细查会列出更多弱响应。")
        self.combo_quality = InspectionComboBox()
        self.combo_quality.addItem("快速预览", True)
        self.combo_quality.addItem("原图精查", False)
        self.combo_quality.setToolTip("快速预览缩小 RGB 后扫描。框选和红外扫描始终使用原图。")
        self.combo_source = InspectionComboBox()
        self.combo_source.addItem("优先红外", True)
        self.combo_source.addItem("仅 RGB", False)
        self.combo_source.setToolTip("检测到可用红外层时优先使用，否则自动扫描 RGB。")
        self.combo_focus = InspectionComboBox()
        self.combo_focus.addItem("主体优先", True)
        self.combo_focus.addItem("全画面", False)
        self.combo_focus.setToolTip("主体优先侧重人脸和平坦背景，减少复杂纹理中的候选。全画面使用同等灵敏度。RGB 整图扫描会避开识别出的片框。")
        self.btn_detect = QPushButton("扫描缺陷")
        self.btn_batch_scan = QPushButton("批量扫描")
        self.btn_batch_scan.setObjectName("secondaryBtn")
        self.btn_batch_scan.setToolTip("选择照片并批量扫描")
        for row, pair in enumerate(((self.combo_sensitivity, self.combo_quality),
                                    (self.combo_source, self.combo_focus),
                                    (self.btn_detect, self.btn_batch_scan))):
            for column, control in zip((0, 2), pair):
                control.setFixedHeight(34)
                grid.addWidget(control, row, column, 1, 2)

        self.layer_group = QButtonGroup(self)
        self.layer_buttons = {}
        for mode, label, hint in (("rgb", "RGB", "显示当前照片"),
                                  ("ir", "红外", "查看归一化的扫描红外层"),
                                  ("overlay", "叠加", "将红外着色并叠到照片上，检查对齐")):
            button = QPushButton(label)
            button.setObjectName("toolBtn")
            button.setToolTip(hint)
            button.setCheckable(True)
            button.setProperty("layer", mode)
            self.layer_group.addButton(button)
            self.layer_buttons[mode] = button
        self.layer_buttons["rgb"].setChecked(True)
        self.set_infrared_available(False)
        self.layer_group.buttonToggled.connect(lambda _, checked: self.inspection_changed.emit() if checked else None)
        self.btn_dust_view = QPushButton("显尘")
        self.btn_dust_view.setObjectName("toolBtn")
        self.btn_dust_view.setCheckable(True)
        self.btn_dust_view.setToolTip("以反相的局部对比突出细微灰点，只改变显示")
        self.btn_dust_view.toggled.connect(self.inspection_changed)
        for column, button in enumerate((*self.layer_buttons.values(), self.btn_dust_view)):
            button.setFixedHeight(34)
            grid.addWidget(button, 3, column)
        return grid

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(6)
        title = QLabel("缺陷检查")
        title.setObjectName("titleLabel")
        layout.addWidget(title)
        layout.addLayout(self._scan_controls())
        self.dust_contrast = QSlider(Qt.Horizontal)
        self.dust_contrast.setRange(10, 90)
        self.dust_contrast.setValue(40)
        self.dust_contrast.setToolTip("显尘对比度")
        self.dust_contrast.setAccessibleName("显尘对比度")
        self.dust_contrast.valueChanged.connect(self.inspection_changed)
        self.dust_contrast.hide()
        layout.addWidget(self.dust_contrast)

        preview_frame = QFrame()
        preview_frame.setObjectName("card")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 7, 8, 8)
        self.preview_title = QLabel("局部预览")
        self.preview_title.setObjectName("sectionLabel")
        self.preview_title.setMinimumWidth(0)
        preview_header = QHBoxLayout()
        preview_header.addWidget(self.preview_title, 1)
        self.btn_trial = QPushButton("三联预览")
        self.btn_trial.setObjectName("toolBtn")
        self.btn_trial.setCheckable(True)
        self.btn_trial.setToolTip("并排查看同一块原片、修复蒙版和试修结果")
        self.btn_trial.toggled.connect(self.preview_mode_changed)
        preview_header.addWidget(self.btn_trial)
        preview_layout.addLayout(preview_header)
        self.preview = CandidatePreview()
        preview_layout.addWidget(self.preview)
        layout.addWidget(preview_frame)

        filters = QHBoxLayout()
        self.combo_filter = InspectionComboBox()
        for label, value in (("全部候选", "all"), ("灰点", "spot"),
                             ("划痕 / 毛絮", "fiber"), ("片面破损", "damage"),
                             ("红外缺陷", "infrared"),
                             ("已勾选", "checked")):
            self.combo_filter.addItem(label, value)
        self.combo_filter.currentIndexChanged.connect(self._render_list)
        filters.addWidget(self.combo_filter, 1)
        self.btn_dismiss = QPushButton("忽略")
        self.btn_dismiss.setObjectName("secondaryBtn")
        self.btn_dismiss.setToolTip("忽略当前行，可用撤销恢复")
        self.btn_dismiss.clicked.connect(self._dismiss_current)
        self.btn_dismiss.setEnabled(False)
        self.btn_refine = QPushButton("原图核对")
        self.btn_refine.setObjectName("secondaryBtn")
        self.btn_refine.setToolTip("在原始像素上重新描出快速预览的候选轮廓 · R")
        self.btn_refine.setEnabled(False)
        filters.addWidget(self.btn_refine)
        filters.addWidget(self.btn_dismiss)
        layout.addLayout(filters)

        self.dust_list = CandidateList()
        self.dust_list.itemChanged.connect(self._on_item_changed)
        self.dust_list.currentItemChanged.connect(self._on_current_changed)
        self.dust_list.itemClicked.connect(self._on_item_clicked)
        self.dust_list.hover_changed.connect(self.dust_hovered)
        layout.addWidget(self.dust_list, 1)

        summary = QHBoxLayout()
        self.lbl_stats = QLabel("等待扫描")
        self.lbl_stats.setObjectName("subtitleLabel")
        summary.addWidget(self.lbl_stats, 1)
        self.btn_check_visible = QPushButton("勾选可见")
        self.btn_uncheck_visible = QPushButton("清空勾选")
        for button, value in ((self.btn_check_visible, True), (self.btn_uncheck_visible, False)):
            button.setObjectName("textBtn")
            button.clicked.connect(lambda _, enabled=value: self._select_visible(enabled))
            summary.addWidget(button)
        layout.addLayout(summary)

        repair = QHBoxLayout()
        self.combo_method = InspectionComboBox()
        self.combo_method.addItem("自动 · 保留胶片颗粒", "auto")
        self.combo_method.addItem("保颗粒修复", "grain")
        self.combo_method.addItem("Navier–Stokes", "ns")
        self.combo_method.addItem("Telea", "telea")
        self.combo_method.setToolTip("修复算法")
        self.combo_method.currentIndexChanged.connect(self.preview_mode_changed)
        repair.addWidget(self.combo_method, 1)
        self.spin_inpaint_radius = QSpinBox()
        self.spin_inpaint_radius.setRange(1, 20)
        self.spin_inpaint_radius.setValue(3)
        self.spin_inpaint_radius.setPrefix("半径  ")
        self.spin_inpaint_radius.setSuffix(" px")
        self.spin_inpaint_radius.valueChanged.connect(self.preview_mode_changed)
        repair.addWidget(self.spin_inpaint_radius)
        layout.addLayout(repair)

        actions = QHBoxLayout()
        self.btn_remove = QPushButton("修复所选")
        self.btn_remove.clicked.connect(self.remove_clicked)
        self.btn_remove.setEnabled(False)
        actions.addWidget(self.btn_remove, 1)
        self.btn_ignore_selected = QPushButton("忽略所选")
        self.btn_ignore_selected.setObjectName("secondaryBtn")
        self.btn_ignore_selected.setEnabled(False)
        actions.addWidget(self.btn_ignore_selected)
        self.btn_export = QPushButton("导出…")
        self.btn_export.setObjectName("secondaryBtn")
        self.btn_export.clicked.connect(self.export_clicked)
        self.btn_export.setEnabled(False)
        actions.addWidget(self.btn_export)
        layout.addLayout(actions)

    def set_infrared_available(self, available):
        if not available:
            self.layer_buttons["rgb"].setChecked(True)
        for layer in ("ir", "overlay"):
            self.layer_buttons[layer].setEnabled(available)

    def inspection_layer(self):
        return self.layer_group.checkedButton().property("layer")

    def step_candidate(self, direction):
        total = self.dust_list.count()
        if not total:
            return None
        index = self.dust_list.currentRow()
        index = (index + direction) % total if index >= 0 else (0 if direction > 0 else total - 1)
        self.dust_list.setCurrentRow(index)
        row = self.dust_list.currentItem()
        self.dust_list.scrollToItem(row)
        return row.data(Qt.UserRole)

    def get_detection_params(self):
        threshold, confidence = self.combo_sensitivity.currentData()
        return {"threshold": threshold, "min_area": 3, "min_confidence": confidence}

    def get_inpaint_params(self):
        return {"method": self.combo_method.currentData(),
                "inpaint_radius": self.spin_inpaint_radius.value()}

    def clear_list(self):
        self._detected = False
        self._items = {}
        self._render_list()

    def update_dust_list(self, items):
        self._detected = True
        self._items = {item.id: item for item in items}
        self._render_list()

    def show_candidate(self, image=None, item=None, *, original=None):
        trial = self.btn_trial.isChecked()
        self.preview.show_candidate(image, item, original=original if trial else None,
                                    repair_params=self.get_inpaint_params() if trial else None)
        self.btn_refine.setEnabled(bool(item and item.preview))
        self.preview_title.setText(
            f"#{item.id:03d}  {item.type_name} · {item.width} × {item.height}"
            if item is not None else "局部预览"
        )

    def _render_list(self):
        mode = self.combo_filter.currentData()
        selected = self.dust_list.currentItem()
        current_id = selected.data(Qt.UserRole) if selected is not None else None
        scroll = self.dust_list.verticalScrollBar().value()
        self.dust_list.blockSignals(True)
        self.dust_list.clear()
        self.dust_list.reset_hover()
        for item in self._items.values():
            is_fiber = item.dust_type in ("fiber", "scratch")
            if ((mode == "fiber" and not is_fiber)
                    or (mode == "spot" and item.dust_type not in ("bright", "dark", "dust", "unknown"))
                    or (mode == "damage" and item.dust_type != "damage")
                    or (mode == "infrared" and item.dust_type != "infrared")
                    or (mode == "checked" and not item.selected)):
                continue
            row = QListWidgetItem(f"#{item.id:03d}    {item.type_name}     {item.width} × {item.height} px")
            row.setData(Qt.UserRole, item.id)
            row.setCheckState(Qt.Checked if item.selected else Qt.Unchecked)
            source = "红外" if item.dust_type == "infrared" else ("RGB 快速预览" if item.preview else "RGB 原图")
            row.setToolTip(f"{source} · 候选分数 {item.confidence:.2f} · 原图位置 {item.center_x}, {item.center_y}")
            self.dust_list.addItem(row)
            if item.id == current_id:
                self.dust_list.setCurrentItem(row)
        self.dust_list.verticalScrollBar().setValue(scroll)
        self.dust_list.blockSignals(False)
        self.btn_dismiss.setEnabled(self.dust_list.currentItem() is not None)
        self._update_stats()
        self.show_candidate()
        self.dust_hovered.emit(-1)

    def select_candidate(self, dust_id):
        for index in range(self.dust_list.count()):
            row = self.dust_list.item(index)
            if row.data(Qt.UserRole) == dust_id:
                self.dust_list.setCurrentItem(row)
                self.dust_list.scrollToItem(row)
                break

    def _update_stats(self):
        checked = sum(item.selected for item in self._items.values())
        self.btn_remove.setEnabled(checked > 0)
        self.btn_ignore_selected.setEnabled(checked > 0)
        if not self._detected:
            self.lbl_stats.setText("等待扫描")
        else:
            self.lbl_stats.setText(f"{len(self._items)} 个候选 · 已勾选 {checked}")

    def _on_item_changed(self, row):
        checked = row.checkState() == Qt.Checked
        dust_id = row.data(Qt.UserRole)
        self._items[dust_id].selected = checked
        self.dust_selected.emit(dust_id, checked)
        if self.combo_filter.currentData() == "checked":
            self._render_list()
        else:
            self._update_stats()
        self.dust_reviewed.emit(dust_id)

    def _select_visible(self, checked):
        self.dust_list.blockSignals(True)
        for index in range(self.dust_list.count()):
            row = self.dust_list.item(index)
            row.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self._items[row.data(Qt.UserRole)].selected = checked
        self.dust_list.blockSignals(False)
        self.select_all_clicked.emit(checked)
        if self.combo_filter.currentData() == "checked":
            self._render_list()
        else:
            self._update_stats()

    def _on_current_changed(self, current, _previous):
        self.btn_dismiss.setEnabled(current is not None)
        if current is not None:
            self.dust_reviewed.emit(current.data(Qt.UserRole))

    def _on_item_clicked(self, row):
        self.dust_clicked.emit(row.data(Qt.UserRole))

    def _dismiss_current(self):
        current = self.dust_list.currentItem()
        if current is not None:
            self.dismiss_clicked.emit(current.data(Qt.UserRole))

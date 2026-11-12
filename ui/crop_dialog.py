"""Edit a rectangular film crop and preview its rigid rotation."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPainterPath, QPen, QPolygonF, QShortcut, QTransform
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QVBoxLayout,
)

from core.frame_crop import (
    FILM_FORMATS, crop_aspect, crop_from_edges, crop_plan, detect_frame, fit_rect,
    frame_points, rectangular_settings, rotation_matrix,
)
from .image_viewer import ImageViewer, photo_pixmap
from .numeric_fields import use_standard_digits


class FrameEditor(ImageViewer):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.rect = None
        self.rotation = 0.0
        self.aspect = "free"
        self.retained = None
        self._drag = None
        self.setAccessibleName("拖动边线、四角或框内移动裁切范围")

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if self.rect is None:
            return
        polygon = QPolygonF([QPointF(*point) for point in frame_points(self.rect, 0)])
        outside = QPainterPath()
        outside.addPolygon(self._photo.mapToScene(self._photo.boundingRect()))
        outside.addPolygon(polygon)
        outside.setFillRule(Qt.OddEvenFill)
        painter.fillPath(outside, QColor(5, 10, 15, 145))
        painter.setRenderHint(QPainter.Antialiasing)
        grid_pen = QPen(QColor(220, 246, 242, 105), 1)
        grid_pen.setCosmetic(True)
        painter.setPen(grid_pen)
        a, b, c, d = list(polygon)
        for fraction in (1 / 3, 2 / 3):
            painter.drawLine(a + (b - a) * fraction, d + (c - d) * fraction)
            painter.drawLine(a + (d - a) * fraction, b + (c - b) * fraction)

        pen = QPen(QColor("#6ce5ce"), 1.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawPolygon(polygon)
        if self.retained is not None:
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawPolygon(QPolygonF([self._photo.mapToScene(QPointF(*point)) for point in self.retained]))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#8af0da"))
        scale = self.transform().m11()
        for start, end in zip(polygon, list(polygon)[1:] + [polygon[0]]):
            painter.drawEllipse(start, 4.5 / scale, 4.5 / scale)
            painter.drawEllipse((start + end) * 0.5, 3 / scale, 3 / scale)

    def _local(self, pos):
        point = self.mapToScene(pos)
        return np.array((point.x(), point.y()))

    def set_rotation(self, angle):
        center = self.image_point(self.viewport().rect().center())
        self.rotation = angle
        self._photo.setTransform(QTransform().rotate(angle))
        self.setSceneRect(self._photo.sceneBoundingRect())
        if self._fit:
            self.fit_to_window()
        else:
            self.centerOn(self._photo.mapToScene(center))

    def _hit(self, pos):
        if self.rect is None:
            return None
        x, y = self._local(pos)
        left, top, right, bottom = self.rect
        tolerance = 8 / self.transform().m11()
        if not (left - tolerance <= x <= right + tolerance and top - tolerance <= y <= bottom + tolerance):
            return None
        edges = tuple(index for index, distance in (
            (0 if abs(x - left) <= abs(x - right) else 2, min(abs(x - left), abs(x - right))),
            (1 if abs(y - top) <= abs(y - bottom) else 3, min(abs(y - top), abs(y - bottom))),
        ) if distance <= tolerance)
        return edges

    @staticmethod
    def _cursor(edges):
        if not edges:
            return Qt.SizeAllCursor
        if len(edges) == 2:
            return Qt.SizeFDiagCursor if edges in ((0, 1), (2, 3)) else Qt.SizeBDiagCursor
        return Qt.SizeHorCursor if edges[0] % 2 == 0 else Qt.SizeVerCursor

    def _resize(self, edges, start, delta):
        if not edges:
            return start + np.tile(delta, 2)
        rect = start.copy()
        old_width, old_height = start[2:] - start[:2]
        ratio = crop_aspect(self.aspect, old_width, old_height)
        minimums = (4, 4) if ratio is None else (max(4, 4 * ratio), max(4, 4 / ratio))
        for edge in edges:
            opposite = (edge + 2) % 4
            axis = edge % 2
            minimum = min(minimums[axis], start[axis + 2] - start[axis])
            value = start[edge] + delta[axis]
            rect[edge] = min(value, start[opposite] - minimum) if edge < 2 else max(value, start[opposite] + minimum)

        if ratio is None:
            return rect
        width, height = rect[2:] - rect[:2]
        drive_width = all(edge % 2 == 0 for edge in edges)
        if len(edges) == 2:
            drive_width = abs(width / old_width - 1) >= abs(height / old_height - 1)
        axis = 1 if drive_width else 0
        extent = width / ratio if drive_width else height * ratio
        if axis in edges:
            rect[axis] = rect[axis + 2] - extent
        elif axis + 2 in edges:
            rect[axis + 2] = rect[axis] + extent
        else:
            center = (start[axis] + start[axis + 2]) / 2
            rect[axis], rect[axis + 2] = center - extent / 2, center + extent / 2
        return rect

    def _drag_rect(self, pos):
        edges, origin, start = self._drag
        target = self._resize(edges, start, self._local(pos) - origin)
        width, height = self.get_image_size()

        def inside(rect):
            points = frame_points(rect, self.rotation)
            return bool(np.all(points >= -1e-6) and np.all(points <= (width + 1e-6, height + 1e-6)))

        if not inside(target):
            # Clamp the movement along the same path so the opposite edge stays anchored.
            low, high = 0.0, 1.0
            for _ in range(20):
                mid = (low + high) / 2
                if inside(start + mid * (target - start)):
                    low = mid
                else:
                    high = mid
            target = start + low * (target - start)
        self.rect = target.tolist()
        self.changed.emit()
        self.viewport().update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.space_pan:
            pos = event.position().toPoint()
            edges = self._hit(pos)
            if edges is not None:
                self._drag = (edges, self._local(pos), np.array(self.rect))
                self.viewport().setCursor(self._cursor(edges))
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._drag is not None:
            self._drag_rect(pos)
            return
        super().mouseMoveEvent(event)
        edges = self._hit(pos)
        if edges is not None and not self.space_pan:
            self.viewport().setCursor(self._cursor(edges))

    def mouseReleaseEvent(self, event):
        if self._drag is not None and event.button() == Qt.LeftButton:
            self._drag_rect(event.position().toPoint())
            self._drag = None
            self._update_tool_cursor()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.set_space_pan(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.set_space_pan(False)
            event.accept()
            return
        super().keyReleaseEvent(event)


class CropDialog(QDialog):
    def __init__(self, image, current, parent=None, *, background=None, batch=False):
        super().__init__(parent)
        self.setObjectName("exportDialog")
        self.setWindowTitle("裁切与旋转")
        self.resize(1040, 730)
        self.setMinimumSize(800, 520)
        self._batch = batch
        self._clear = False
        self._auto_edges = None
        self._auto_found = False
        self._loading = False
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.title = QLabel("裁切与旋转")
        self.title.setObjectName("titleLabel")
        header.addWidget(self.title)
        header.addStretch()
        self.previous = QPushButton("‹ 上一张")
        self.position = QLabel()
        self.position.setObjectName("subtitleLabel")
        self.next = QPushButton("下一张 ›")
        for control in (self.previous, self.position, self.next):
            header.addWidget(control)
            control.setVisible(batch)
        for button, key in ((self.previous, "PgUp"), (self.next, "PgDown")):
            button.setObjectName("secondaryBtn")
            if batch:
                button.setToolTip(key)
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(button.click)
        layout.addLayout(header)

        hint = QLabel("拖动边线或四角调整裁切框，拖动框内移动。用网格和旋转角度校平画面。")
        hint.setObjectName("subtitleLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        controls = QHBoxLayout()
        auto = QPushButton("自动识别")
        auto.setObjectName("secondaryBtn")
        auto.clicked.connect(self._auto)
        controls.addWidget(auto)
        self.aspect = aspect_control()
        controls.addWidget(QLabel("画幅"))
        controls.addWidget(self.aspect)
        self.rotation = QDoubleSpinBox()
        use_standard_digits(self.rotation)
        self.rotation.setRange(-15, 15)
        self.rotation.setDecimals(2)
        self.rotation.setSingleStep(0.1)
        self.rotation.setPrefix("旋转  ")
        self.rotation.setSuffix("°")
        self.rotation.setAccessibleName("顺时针旋转角度")
        self.rotation.setToolTip("正值顺时针，负值逆时针")
        controls.addWidget(self.rotation)
        reset = QPushButton("归零")
        reset.setObjectName("secondaryBtn")
        reset.clicked.connect(lambda: self.rotation.setValue(0))
        controls.addWidget(reset)
        self.border = border_control()
        self.border.setEnabled(not batch)
        controls.addWidget(self.border)
        controls.addStretch()
        self.modes = QButtonGroup(self)
        for index, label in enumerate(("编辑边界", "结果预览")):
            button = QPushButton(label)
            button.setObjectName("toolBtn")
            button.setCheckable(True)
            self.modes.addButton(button, index)
            controls.addWidget(button)
        self.modes.button(0).setChecked(True)
        layout.addLayout(controls)

        self.stack = QStackedWidget()
        self.editor = FrameEditor()
        self.preview = ImageViewer()
        for viewer in (self.editor, self.preview):
            viewer.set_canvas_background(background)
            self.stack.addWidget(viewer)
        layout.addWidget(self.stack, 1)
        self.modes.idClicked.connect(self.stack.setCurrentIndex)
        self.status = QLabel()
        self.status.setObjectName("subtitleLabel")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        footer = QHBoxLayout()
        self.restore = QPushButton("取消裁切")
        self.restore.setObjectName("secondaryBtn")
        self.restore.setToolTip("恢复完整画面和原始角度")
        self.restore.setVisible(not batch)
        self.restore.setEnabled(current is not None)
        self.restore.clicked.connect(self._restore)
        footer.addWidget(self.restore)
        if batch:
            note = QLabel("确认后加入列表，应用所选裁切后生效。")
            note.setObjectName("subtitleLabel")
            footer.addWidget(note)
        footer.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.apply_button = buttons.button(QDialogButtonBox.Ok)
        self.apply_button.setText("确认裁切" if batch else "应用裁切")
        cancel = buttons.button(QDialogButtonBox.Cancel)
        cancel.setText("返回列表" if batch else "取消")
        if batch:
            cancel.setToolTip("保留已确认的照片和当前草稿")
        cancel.setObjectName("secondaryBtn")
        if not batch:
            buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        footer.addWidget(buttons)
        layout.addLayout(footer)
        self.border.valueChanged.connect(self._update)
        self.rotation.valueChanged.connect(self._rotate)
        self.aspect.currentIndexChanged.connect(self._change_aspect)
        self.editor.changed.connect(self._update)
        if image is not None:
            self.load_photo(image, current, detected=detect_frame(image) if current is None else None)

    def load_photo(self, image, settings, *, detected, name=None, index=0, total=1,
                   aspect="free", border_percent=0):
        self._loading = True
        self._image = image
        self._auto_found = detected is not None
        self._auto_edges = detected["detected_edges"] if detected else (0 if name or settings is None else None)
        pixmap = photo_pixmap(image)
        self.editor.load_pixmap(pixmap)
        self.preview.load_pixmap(pixmap)
        if settings is None:
            corners = detected["corners"] if detected else None
            settings = crop_from_edges(image.shape, corners, aspect=aspect, border_percent=border_percent)
        else:
            settings = rectangular_settings(image.shape, settings)
        self.editor.rect = list(settings["rect"])
        self.editor.set_rotation(settings["rotation"])
        self.editor.aspect = settings["aspect"]
        self.rotation.setValue(self.editor.rotation)
        self.aspect.setCurrentIndex(max(0, self.aspect.findData(self.editor.aspect)))
        self.border.setValue(settings["border_percent"])
        if name is not None:
            self.setWindowTitle("核对裁切 · " + Path(name).name)
            self.title.setText(Path(name).name)
            self.position.setText(f"{index + 1} / {total}")
            self.previous.setEnabled(index > 0)
            self.next.setEnabled(index + 1 < total)
            self.apply_button.setText("确认并下一张" if index + 1 < total else "确认裁切")
        self.modes.button(0).setChecked(True)
        self.stack.setCurrentIndex(0)
        self._loading = False
        self._update()

    def _auto(self):
        detected = detect_frame(self._image)
        self._auto_found = detected is not None
        self._auto_edges = detected["detected_edges"] if detected else 0
        corners = detected["corners"] if detected else None
        settings = crop_from_edges(self._image.shape, corners, rotation=self.rotation.value(),
                                   aspect=self.aspect.currentData())
        self.editor.rect = settings["rect"]
        self._update()

    def _rotate(self, value):
        if self._loading:
            return
        editor = self.editor
        rect = np.asarray(editor.rect)
        center = (rect[:2] + rect[2:]) / 2
        next_center = center @ rotation_matrix(editor.rotation) @ rotation_matrix(value).T
        editor.rect = fit_rect(self._image.shape, rect + np.tile(next_center - center, 2), value, editor.aspect)
        editor.set_rotation(value)
        self._update()

    def _change_aspect(self):
        if self._loading:
            return
        editor = self.editor
        editor.aspect = self.aspect.currentData()
        editor.rect = fit_rect(self._image.shape, editor.rect, editor.rotation, editor.aspect)
        self._update()

    def settings(self):
        if self._clear:
            return None
        return {"rect": list(self.editor.rect), "rotation": self.editor.rotation,
                "aspect": self.editor.aspect, "border_percent": self.border.value()}

    def _update(self):
        if self._loading:
            return
        self.editor.retained = None
        try:
            points, _, (width, height), actual = crop_plan(self._image.shape, self.settings())
        except ValueError as error:
            self.apply_button.setEnabled(False)
            self.status.setText(str(error))
            self.editor.viewport().update()
            return
        self.apply_button.setEnabled(True)
        if actual >= 0.05:
            self.editor.retained = points
        self.editor.viewport().update()
        self.preview.set_geometry(self.settings())
        text = f"输出 {width:,} × {height:,} px · 留边 {actual:.1f}% / 边"
        if actual + 0.05 < self.border.value():
            text += " · 已限制到原片可用边框"
        if self._auto_edges == 0:
            text = ("已按齿孔估算" if self._auto_found else "未找到可靠片框") + " · " + text
        elif self._auto_edges is not None and self._auto_edges < 4:
            text = f"已识别 {self._auto_edges}/4 边 · " + text
        self.status.setText(text)

    def _restore(self):
        self._clear = True
        self.accept()


def aspect_control():
    combo = QComboBox()
    combo.setAccessibleName("胶卷画幅")
    combo.setToolTip("按画幅锁定比例，自动适配横竖照片")
    for key, label, width, height in FILM_FORMATS:
        combo.addItem(f"{label} · {width}:{height}" if height else label, key)
    return combo


def border_control():
    border = QDoubleSpinBox()
    use_standard_digits(border)
    border.setRange(0, 10)
    border.setSingleStep(0.1)
    border.setDecimals(1)
    border.setPrefix("保留边框  ")
    border.setSuffix(" %")
    border.setAccessibleName("每边保留边框百分比")
    border.setToolTip("0% 去掉扫描边框，2% 表示每边保留相应画面尺寸的 2%。以原片实际可用边框为限。")
    return border

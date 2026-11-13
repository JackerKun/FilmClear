"""Choose the photos and output settings for an export job."""

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout,
)
from core.batch_export import SOURCE_FORMAT
from core.export_metadata import describe_export, hdr_input_error, read_metadata
from core.hdr_export import HDR_SUFFIXES, PQ_HEIC, ULTRA_HDR_JPEG
from .file_panel import Thumbnails


class ExportPhotoList(QListWidget):
    checked_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._anchor = None
        self.setUniformItemSizes(True)
        self.setIconSize(QSize(84, 58))
        self.setAccessibleName("待导出照片")
        self.itemChanged.connect(self.checked_changed)

    def set_checks(self, rows, state):
        with QSignalBlocker(self):
            for row in rows:
                self.item(row).setCheckState(state)
        self.checked_changed.emit()

    def check_all(self, state):
        self._anchor = None
        self.set_checks(range(self.count()), state)

    def _toggle(self, row, modifiers):
        if self._anchor is not None and modifiers & Qt.ShiftModifier:
            first, last = sorted((self._anchor, row))
            self.set_checks(range(first, last + 1), Qt.Checked)
        else:
            self._anchor = row
            item = self.item(row)
            item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)

    def mousePressEvent(self, event):
        index = self.indexAt(event.position().toPoint())
        if event.button() == Qt.LeftButton and index.isValid():
            self.setCurrentRow(index.row())
            self._toggle(index.row(), event.modifiers())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        # A checkbox click is handled by the same row toggle on press.
        if event.button() == Qt.LeftButton:
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.SelectAll):
            self.check_all(Qt.Checked)
        elif event.key() == Qt.Key_Space and self.currentRow() >= 0:
            self._toggle(self.currentRow(), event.modifiers())
        else:
            super().keyPressEvent(event)
            return
        event.accept()


class ExportDialog(QDialog):
    def __init__(self, rows, current, directory, batch, parent=None, *, icons=None):
        super().__init__(parent)
        self.setObjectName("exportDialog")
        self._metadata = {}
        self.setWindowTitle("批量导出照片" if batch else "导出照片")
        available = self.screen().availableGeometry()
        self.resize(min(900, available.width() - 40), min(720, available.height() - 72))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        intro = QLabel("导出使用原始像素精度")
        intro.setObjectName("titleLabel")
        layout.addWidget(intro)
        intro = QLabel("按原图修复后应用已确认的裁切和旋转。复制原文件会输出完整原片。")
        intro.setObjectName("subtitleLabel")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.photos = ExportPhotoList()
        pending = []
        self._thumbnail_rows = []
        icons = icons or {}
        for path, checked, status in rows:
            row = QListWidgetItem(f"{Path(path).name}    ·    {status}")
            row.setData(Qt.UserRole, path)
            row.setToolTip(path)
            row.setSizeHint(QSize(0, 68))
            row.setCheckState(Qt.Checked if (checked if batch else path == current) else Qt.Unchecked)
            icon = icons.get(path)
            if icon is not None and not icon.isNull():
                row.setIcon(icon)
            else:
                self._thumbnail_rows.append(self.photos.count())
                pending.append(path)
            self.photos.addItem(row)
            if path == current:
                self.photos.setCurrentItem(row)
        layout.addWidget(self.photos, 1)
        choices = QHBoxLayout()
        for label, state in (("全选", Qt.Checked), ("清空", Qt.Unchecked)):
            button = QPushButton(label)
            button.setObjectName("secondaryBtn")
            button.clicked.connect(lambda _, state=state: self.photos.check_all(state))
            choices.addWidget(button)
        hint = QLabel("点击勾选 · Shift + 点击连选")
        hint.setObjectName("subtitleLabel")
        choices.addWidget(hint)
        choices.addStretch()
        self.photo_count = QLabel()
        self.photo_count.setObjectName("subtitleLabel")
        choices.addWidget(self.photo_count)
        layout.addLayout(choices)
        settings = QFormLayout()
        settings.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        settings.setHorizontalSpacing(12)
        settings.setVerticalSpacing(8)
        self.mode = QComboBox()
        self.mode.addItem("应用勾选的缺陷并导出", "repair")
        self.mode.addItem("仅导出已应用的编辑", "edited")
        self.mode.addItem("复制原文件，保留原格式和附带信息", "original")
        settings.addRow("输出内容", self.mode)
        self.format = QComboBox()
        self.format.addItem("跟随原文件格式", SOURCE_FORMAT)
        self.format.addItem("TIFF（保留 8 / 16 位）", ".tiff")
        self.format.addItem("PNG（无损，保留 8 / 16 位）", ".png")
        self.format.addItem("JPEG（8 位）", ".jpg")
        self.format.addItem("JPEG · Ultra HDR（含 SDR 底图）", ULTRA_HDR_JPEG)
        self.format.addItem("HEIC · HDR（10 位 PQ）", PQ_HEIC)
        self.mode.currentIndexChanged.connect(
            lambda _: self.format.setEnabled(self.mode.currentData() != "original"))
        settings.addRow("文件格式", self.format)
        self.output_info = QLabel()
        self.output_info.setWordWrap(True)
        self.output_info.setObjectName("subtitleLabel")
        settings.addRow("输出信息", self.output_info)
        self.hdr_hint = QLabel("支持 16 位 sRGB TIFF / PNG、Ultra HDR JPEG 和 PQ HEIC。对照页使用 SDR 预览。")
        self.hdr_hint.setObjectName("subtitleLabel")
        self.hdr_hint.setWordWrap(True)
        settings.addRow(self.hdr_hint)
        self.include_review = QCheckBox("同时生成前后对照页（HTML）")
        self.include_review.setToolTip("包含原片、实际导出结果和一处修复局部，可离线浏览")
        settings.addRow(self.include_review)
        self.mode.currentIndexChanged.connect(self._refresh_button)
        self.format.currentIndexChanged.connect(self._refresh_button)
        folder = QHBoxLayout()
        self.directory = QLineEdit(str(directory))
        self.directory.setCursorPosition(0)
        folder.addWidget(self.directory, 1)
        browse = QPushButton("选择…")
        browse.setObjectName("secondaryBtn")
        browse.clicked.connect(self._browse)
        folder.addWidget(browse)
        settings.addRow("保存到", folder)
        layout.addLayout(settings)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.export_button = buttons.button(QDialogButtonBox.Ok)
        self.export_button.setText("导出")
        cancel = buttons.button(QDialogButtonBox.Cancel)
        cancel.setText("取消")
        cancel.setObjectName("secondaryBtn")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.photos.checked_changed.connect(self._refresh_button)
        self.directory.textChanged.connect(self._refresh_button)
        layout.addWidget(buttons)
        self._refresh_button()
        self._thumbnails = None
        if pending:
            self._thumbnails = Thumbnails(pending, self)
            self._thumbnails.ready.connect(self._show_thumbnail)
            self._thumbnails.start()

    def _show_thumbnail(self, index, image):
        with QSignalBlocker(self.photos):
            self.photos.item(self._thumbnail_rows[index]).setIcon(QIcon(QPixmap.fromImage(image)))

    def done(self, result):
        if self._thumbnails is not None:
            self._thumbnails.requestInterruption()
            self._thumbnails.wait()
        super().done(result)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "选择导出文件夹", self.directory.text())
        if path:
            self.directory.setText(path)
            self.directory.setCursorPosition(0)

    def selected_paths(self):
        return [self.photos.item(i).data(Qt.UserRole) for i in range(self.photos.count())
                if self.photos.item(i).checkState() == Qt.Checked]

    def _refresh_button(self, *_):
        paths = self.selected_paths()
        count = len(paths)
        mode, format_id = self.mode.currentData(), self.format.currentData()
        hdr = mode != "original" and format_id in HDR_SUFFIXES
        self.include_review.setEnabled(mode != "original")
        self.hdr_hint.setVisible(hdr)
        description = "选择要导出的照片"
        eligible = True
        if paths:
            path = paths[0]
            if path not in self._metadata:
                self._metadata[path] = read_metadata(path)
            metadata = self._metadata[path]
            description = describe_export(path, format_id, mode, metadata)
            if count == 1 and hdr and metadata.hdr_format is None:
                eligible = not hdr_input_error(metadata)
        if count > 1:
            description = f"首张 · {description}（其余按各原图）"
        self.output_info.setText(description)
        self.photo_count.setText(f"已选 {count} / {self.photos.count()} 张")
        self.export_button.setEnabled(bool(self.directory.text().strip() and count and eligible))

    def selection(self):
        return (self.selected_paths(), Path(self.directory.text().strip()).expanduser(),
                self.format.currentData(), self.mode.currentData(),
                self.include_review.isEnabled() and self.include_review.isChecked())

"""Choose the photos and output settings for an export job."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout,
)
from core.batch_export import SOURCE_FORMAT


class ExportDialog(QDialog):
    def __init__(self, rows, current, directory, batch, parent=None):
        super().__init__(parent)
        self.setObjectName("exportDialog")
        self.setWindowTitle("批量导出照片" if batch else "导出照片")
        self.resize(620, 445)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        intro = QLabel("导出使用原始像素尺寸")
        intro.setObjectName("titleLabel")
        layout.addWidget(intro)
        intro = QLabel("快速预览的缺陷会回到原图坐标后再修复。")
        intro.setObjectName("subtitleLabel")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.photos = QListWidget()
        rows = sorted(rows, key=lambda row: not (row[1] if batch else row[0] == current))
        for path, checked, status in rows:
            row = QListWidgetItem(f"{Path(path).name}    ·    {status}")
            row.setData(Qt.UserRole, path)
            row.setToolTip(path)
            row.setCheckState(Qt.Checked if (checked if batch else path == current) else Qt.Unchecked)
            self.photos.addItem(row)
        layout.addWidget(self.photos, 1)
        choices = QHBoxLayout()
        for label, state in (("全选", Qt.Checked), ("清空", Qt.Unchecked)):
            button = QPushButton(label)
            button.setObjectName("secondaryBtn")
            button.clicked.connect(lambda _, state=state: self._select(state))
            choices.addWidget(button)
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
        self.mode.addItem("仅导出当前已修复结果", "edited")
        self.mode.addItem("复制原文件，保留原格式和附带信息", "original")
        settings.addRow("输出内容", self.mode)
        self.format = QComboBox()
        self.format.addItem("跟随原文件格式", SOURCE_FORMAT)
        self.format.addItem("TIFF（保留 8 / 16 位）", ".tiff")
        self.format.addItem("PNG（无损，保留 8 / 16 位）", ".png")
        self.format.addItem("JPEG（8 位）", ".jpg")
        self.mode.currentIndexChanged.connect(
            lambda _: self.format.setEnabled(self.mode.currentData() != "original"))
        settings.addRow("文件格式", self.format)
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
        self.photos.itemChanged.connect(self._refresh_button)
        self.directory.textChanged.connect(self._refresh_button)
        layout.addWidget(buttons)
        self._refresh_button()

    def _select(self, state):
        for row in range(self.photos.count()):
            self.photos.item(row).setCheckState(state)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "选择导出文件夹", self.directory.text())
        if path:
            self.directory.setText(path)
            self.directory.setCursorPosition(0)

    def selected_paths(self):
        return [self.photos.item(i).data(Qt.UserRole) for i in range(self.photos.count())
                if self.photos.item(i).checkState() == Qt.Checked]

    def _refresh_button(self, *_):
        count = len(self.selected_paths())
        self.photo_count.setText(f"已选 {count} / {self.photos.count()} 张")
        self.export_button.setEnabled(bool(self.directory.text().strip() and count))

    def selection(self):
        return (self.selected_paths(), Path(self.directory.text().strip()).expanduser(),
                self.format.currentData(), self.mode.currentData())

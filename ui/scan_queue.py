"""Select which photos to scan together."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout,
)


class ScanQueueDialog(QDialog):
    def __init__(self, paths, already_scanned, parent=None):
        super().__init__(parent)
        self.setObjectName("exportDialog")
        self.setWindowTitle("批量扫描")
        self.resize(480, 380)
        layout = QVBoxLayout(self)
        label = QLabel("按当前检测设置逐张扫描。已修复的照片会从当前修复结果继续。")
        label.setWordWrap(True)
        layout.addWidget(label)
        self.photos = QListWidget()
        for path in paths:
            scanned = path in already_scanned
            row = QListWidgetItem(f"{Path(path).name}  ·  {'已扫描' if scanned else '待扫描'}")
            row.setToolTip(path)
            row.setData(Qt.UserRole, path)
            row.setCheckState(Qt.Unchecked if scanned else Qt.Checked)
            self.photos.addItem(row)
        layout.addWidget(self.photos, 1)
        options = QHBoxLayout()
        for text, state in (("全选", Qt.Checked), ("清空", Qt.Unchecked)):
            button = QPushButton(text)
            button.setObjectName("secondaryBtn")
            button.clicked.connect(lambda _, value=state: self._select(value))
            options.addWidget(button)
        options.addStretch()
        layout.addLayout(options)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.start = buttons.button(QDialogButtonBox.Ok)
        self.start.setText("开始扫描")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.photos.itemChanged.connect(lambda _: self.start.setEnabled(bool(self.selection())))
        self.start.setEnabled(bool(self.selection()))
        layout.addWidget(buttons)

    def _select(self, state):
        for row in range(self.photos.count()):
            self.photos.item(row).setCheckState(state)

    def selection(self):
        return [self.photos.item(row).data(Qt.UserRole) for row in range(self.photos.count())
                if self.photos.item(row).checkState() == Qt.Checked]

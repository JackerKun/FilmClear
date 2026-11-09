"""Compare recurring candidates before a batch decision."""

from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)


from core.scan_io import display_pixels, read_scan


class RepeatedPreview(QThread):
    ready = Signal(int, object)

    def __init__(self, group, parent):
        super().__init__(parent)
        self.group = group

    def run(self):
        for index, (path, item) in enumerate(self.group):
            if self.isInterruptionRequested():
                return
            try:
                pixels = read_scan(path, read_ir=False).image
                x, y = item.center
                radius = max(24, item.width, item.height)
                x1, y1 = max(0, x - radius), max(0, y - radius)
                crop = display_pixels(pixels[y1:y + radius, x1:x + radius]).copy()
                if not crop.size:
                    continue
                scale = min(144 / crop.shape[1], 116 / crop.shape[0])
                size = tuple(max(1, round(dimension * scale)) for dimension in crop.shape[1::-1])
                crop = cv2.resize(crop, size)
                point = (round((x - x1) * scale), round((y - y1) * scale))
                cv2.circle(crop, point, max(8, round(max(item.width, item.height) * scale * 0.66)),
                           (198, 235, 99), 1, cv2.LINE_AA)
                image = QImage(crop.data, crop.shape[1], crop.shape[0],
                               crop.strides[0], QImage.Format_BGR888).copy()
            except (OSError, ValueError, cv2.error):
                image = QImage()
            self.ready.emit(index, image)


class RepeatDialog(QDialog):
    def __init__(self, groups, parent=None):
        super().__init__(parent)
        self.setObjectName("exportDialog")
        self.setWindowTitle("同卷重复候选")
        self.resize(690, 395)
        self.groups = groups
        self.operation = None
        self._worker = None
        self._workers = set()
        layout = QVBoxLayout(self)
        description = QLabel("同一位置出现在至少三张照片。请先并排核对，确认后再处理整组。")
        description.setWordWrap(True)
        layout.addWidget(description)
        self.list = QListWidget()
        for group in groups:
            item = group[0][1]
            self.list.addItem(f"{len(group)} 张  ·  {item.type_name}  ·  参考位置 {item.center_x}, {item.center_y}")
        layout.addWidget(self.list, 1)
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroller.setFixedHeight(174)
        cards = QWidget()
        self.strip = QHBoxLayout(cards)
        self.strip.setContentsMargins(4, 3, 4, 3)
        scroller.setWidget(cards)
        layout.addWidget(scroller)
        buttons = QHBoxLayout()
        close = QPushButton("关闭")
        close.setObjectName("secondaryBtn")
        close.clicked.connect(self.reject)
        buttons.addWidget(close)
        buttons.addStretch()
        for title, operation in (("忽略此组", "ignore"), ("修复此组", "repair")):
            button = QPushButton(title)
            button.clicked.connect(lambda _, action=operation: self._accept(action))
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.list.currentRowChanged.connect(self._show_group)
        self.list.setCurrentRow(0)

    def _accept(self, operation):
        self.operation = operation
        self.accept()

    def selection(self):
        return self.operation, self.groups[self.list.currentRow()]

    def _show_group(self, row):
        if self._worker is not None:
            self._worker.requestInterruption()
        while self.strip.count():
            self.strip.takeAt(0).widget().deleteLater()
        if row < 0:
            return
        self._labels = []
        group = self.groups[row]
        for path, _ in group:
            card = QWidget()
            card.setMinimumWidth(150)
            layout = QVBoxLayout(card)
            preview = QLabel("读取中…")
            preview.setAlignment(Qt.AlignCenter)
            preview.setFixedHeight(116)
            layout.addWidget(preview)
            name = QLabel(Path(path).name)
            name.setObjectName("subtitleLabel")
            name.setToolTip(path)
            name.setAlignment(Qt.AlignCenter)
            name.setMinimumWidth(0)
            layout.addWidget(name)
            self.strip.addWidget(card, 1)
            self._labels.append(preview)
        worker = RepeatedPreview(group, self)
        self._worker = worker
        self._workers.add(worker)
        worker.ready.connect(self._set_preview)
        worker.finished.connect(self._finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _set_preview(self, index, image):
        if self.sender() is self._worker:
            pixmap = QPixmap.fromImage(image).scaled(144, 116, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if pixmap.isNull():
                self._labels[index].setText("无法显示局部")
            else:
                self._labels[index].setPixmap(pixmap)

    def _finished(self):
        worker = self.sender()
        self._workers.discard(worker)
        if worker is self._worker:
            self._worker = None

    def done(self, result):
        for worker in tuple(self._workers):
            worker.requestInterruption()
            worker.wait()
        super().done(result)

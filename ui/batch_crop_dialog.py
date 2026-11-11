"""Choose a roll, then review and apply frame suggestions."""

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
)

from core.frame_crop import crop_from_edges
from .crop_dialog import CropDialog, aspect_control, border_control


@dataclass
class CropSuggestion:
    path: str
    stamp: tuple | None = None
    size: tuple | None = None
    detected: dict | None = None
    error: str = ""

    @property
    def edges(self):
        return self.detected["detected_edges"] if self.detected else 0


class BatchCropDialog(QDialog):
    ResetCrops = 2

    def __init__(self, paths, already_cropped, read_image, parent=None, *, background=None):
        super().__init__(parent)
        self.setObjectName("exportDialog")
        self.setWindowTitle("批量裁切与旋转")
        self.resize(640, 510)
        self._read_image = read_image
        self._background = background
        self._already_cropped = already_cropped
        self._results = None
        self._confirmed = {}
        self._drafts = {}
        self._rows = {}

        layout = QVBoxLayout(self)
        title = QLabel("批量裁切与旋转")
        title.setObjectName("titleLabel")
        layout.addWidget(title)
        self.hint = QLabel("勾选这卷要处理的照片。已裁切的照片可重新勾选。")
        self.hint.setObjectName("subtitleLabel")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        options = QHBoxLayout()
        self.border = border_control()
        self.aspect = aspect_control()
        options.addWidget(self.border)
        options.addWidget(QLabel("画幅"))
        options.addWidget(self.aspect)
        options.addStretch()
        layout.addLayout(options)

        self.photos = QTreeWidget()
        self.photos.setObjectName("batchCropList")
        self.photos.setHeaderLabels(("照片", "状态"))
        self.photos.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.photos.header().setSectionResizeMode(1, QHeaderView.Interactive)
        self.photos.setColumnWidth(1, 210)
        for path in paths:
            existing = path in already_cropped
            row = self._add_photo(self.photos, path, "已有裁切" if existing else "待识别")
            row.setCheckState(0, Qt.Unchecked if existing else Qt.Checked)
        layout.addWidget(self.photos, 1)

        actions = QHBoxLayout()
        self.select_all = QPushButton("全选")
        self.clear = QPushButton("清空")
        self.review_button = QPushButton("核对裁切…")
        self.select_confirmed = QPushButton("仅勾选手动确认")
        for button in (self.select_all, self.clear, self.review_button, self.select_confirmed):
            button.setObjectName("secondaryBtn")
            actions.addWidget(button)
        self.review_button.hide()
        self.select_confirmed.hide()
        self.select_all.clicked.connect(lambda: self._select(Qt.Checked))
        self.clear.clicked.connect(lambda: self._select(Qt.Unchecked))
        self.review_button.clicked.connect(lambda: self._review_row(self.photos.currentItem()))
        self.select_confirmed.clicked.connect(self._select_confirmed)
        actions.addStretch()
        layout.addLayout(actions)

        self.summary = QLabel()
        self.summary.setObjectName("subtitleLabel")
        layout.addWidget(self.summary)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.reset_button = buttons.addButton("取消所选裁切", QDialogButtonBox.ActionRole)
        self.reset_button.setObjectName("secondaryBtn")
        self.reset_button.setToolTip("恢复勾选照片的完整画面和原始角度，可逐张撤销")
        self.reset_button.setVisible(bool(already_cropped))
        self.reset_button.clicked.connect(lambda: self.done(self.ResetCrops))
        self.apply_button = buttons.button(QDialogButtonBox.Ok)
        self.apply_button.setText("识别片框")
        buttons.button(QDialogButtonBox.Cancel).setText("关闭")
        buttons.button(QDialogButtonBox.Cancel).setObjectName("secondaryBtn")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.photos.itemChanged.connect(self._refresh)
        self.photos.currentItemChanged.connect(self._refresh)
        self.photos.itemDoubleClicked.connect(self._review_row)
        self._refresh()

    def _add_photo(self, parent, path, status):
        row = QTreeWidgetItem(parent, (Path(path).name, status))
        row.setData(0, Qt.UserRole, path)
        row.setToolTip(0, path)
        self._rows[path] = row
        return row

    def set_results(self, results):
        self._results = {result.path: result for result in results}
        self._rows.clear()
        self.photos.clear()
        self.hint.setText("四边齐全的自动结果默认勾选。返回列表会保留确认与草稿。点击「应用所选裁切」后生效。")
        self.aspect.setEnabled(False)
        self.apply_button.setText("应用所选裁切")
        self.select_all.setText("全选可应用")
        self.review_button.show()
        self.select_confirmed.show()
        self.reset_button.hide()
        groups = (("需要核对", [item for item in results if not item.error and item.edges < 4]),
                  ("已识别四边", [item for item in results if not item.error and item.edges == 4]),
                  ("无法处理", [item for item in results if item.error]))
        first_review = None
        for label, items in groups:
            if not items:
                continue
            group = QTreeWidgetItem(self.photos, (f"{label} · {len(items)} 张",))
            group.setFlags(Qt.ItemIsEnabled)
            for result in items:
                status = result.error or (f"已识别 {result.edges}/4 边" if result.edges else "需手动确定边界")
                row = self._add_photo(group, result.path, status)
                if not result.error and result.edges == 4:
                    row.setCheckState(0, Qt.Checked)
                else:
                    row.setFlags(row.flags() & ~Qt.ItemIsUserCheckable)
                    if result.error:
                        row.setToolTip(1, result.error)
                    elif first_review is None:
                        first_review = row
            group.setExpanded(True)
        if first_review is not None:
            self.photos.setCurrentItem(first_review)
        self._refresh()

    def _review_row(self, row, _column=0):
        if self._results is None or row is None:
            return
        path = row.data(0, Qt.UserRole)
        result = self._results.get(path)
        if result is None or result.error:
            return
        queue = [item for item in self._results.values() if not item.error and (item.edges < 4) == (result.edges < 4)]
        index = queue.index(result)
        editor = CropDialog(None, None, self, background=self._background, batch=True)

        def load(target):
            item = queue[target]
            image = self._read_image(item)
            if image is None:
                return False
            settings = self._drafts.get(item.path, self._suggested_settings(item))
            editor.load_photo(image, dict(settings, border_percent=self.border.value()), detected=item.detected,
                              name=item.path, index=target, total=len(queue))
            return True

        def save_draft():
            item = queue[index]
            if not editor.apply_button.isEnabled():
                return
            settings = editor.settings()
            if settings == dict(self._drafts.get(item.path, self._suggested_settings(item)),
                                border_percent=self.border.value()):
                return
            self._drafts[item.path] = settings
            self._confirmed.pop(item.path, None)
            row = self._rows[item.path]
            row.setText(1, "已调整 · 待确认")
            row.setCheckState(0, Qt.Unchecked)
            row.setFlags(row.flags() & ~Qt.ItemIsUserCheckable)

        def move(step):
            nonlocal index
            target = index + step
            if 0 <= target < len(queue):
                save_draft()
                if load(target):
                    index = target
                else:
                    editor.reject()

        def confirm():
            save_draft()
            item = queue[index]
            self._confirmed[item.path] = editor.settings()
            row = self._rows[item.path]
            row.setText(1, "已确认裁切")
            row.setFlags(row.flags() | Qt.ItemIsUserCheckable)
            row.setCheckState(0, Qt.Checked)
            if index + 1 < len(queue):
                move(1)
            else:
                editor.accept()

        editor.previous.clicked.connect(lambda: move(-1))
        editor.next.clicked.connect(lambda: move(1))
        editor.apply_button.clicked.connect(confirm)
        if load(index):
            editor.exec()
            save_draft()
            self.photos.setCurrentItem(self._rows[queue[index].path])
            self._refresh()
        editor.deleteLater()

    def _suggested_settings(self, result):
        width, height = result.size
        corners = result.detected["corners"] if result.detected else None
        return crop_from_edges((height, width), corners, aspect=self.aspect.currentData())

    def selection(self):
        return [path for path, row in self._rows.items()
                if row.flags() & Qt.ItemIsUserCheckable and row.checkState(0) == Qt.Checked]

    def reset_selection(self):
        return [path for path in self.selection() if path in self._already_cropped]

    def approved(self):
        for path in self.selection():
            result = self._results[path]
            if path in self._confirmed:
                yield result, dict(self._confirmed[path], border_percent=self.border.value())
            elif not result.error and result.edges == 4 and path not in self._drafts:
                yield result, dict(self._suggested_settings(result), border_percent=self.border.value())

    def _select(self, state):
        for row in self._rows.values():
            if row.flags() & Qt.ItemIsUserCheckable:
                row.setCheckState(0, state)

    def _select_confirmed(self):
        for path, row in self._rows.items():
            if row.flags() & Qt.ItemIsUserCheckable:
                row.setCheckState(0, Qt.Checked if path in self._confirmed else Qt.Unchecked)

    def _refresh(self, *_):
        selected = self.selection()
        self.apply_button.setEnabled(bool(selected))
        if self._results is None:
            reset_count = len(self.reset_selection())
            self.reset_button.setEnabled(reset_count > 0)
            self.reset_button.setText(f"取消所选裁切（{reset_count}）" if reset_count else "取消所选裁切")
            message = f"已勾选 {len(selected)} 张"
        else:
            confirmed = sum(path in self._confirmed for path in selected)
            pending = sum(not result.error and (result.edges < 4 or result.path in self._drafts)
                          and result.path not in self._confirmed
                          for result in self._results.values())
            message = f"待应用 {len(selected)} 张 · 手动确认 {confirmed} · 自动识别 {len(selected) - confirmed} · 待核对 {pending}"
            row = self.photos.currentItem()
            result = self._results.get(row.data(0, Qt.UserRole)) if row else None
            self.review_button.setEnabled(result is not None and not result.error)
            self.select_confirmed.setEnabled(bool(self._confirmed))
        self.summary.setText(message)

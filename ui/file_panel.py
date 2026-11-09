"""Import browser with background thumbnail decoding."""

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, QSize, QThread, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget,
)
from core.scan_io import is_ir_sidecar

from .tool_icons import tool_icon

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class Thumbnails(QThread):
    ready = Signal(int, QImage)

    def __init__(self, paths, parent, *, generation):
        super().__init__(parent)
        self.paths = paths
        self.generation = generation

    def run(self):
        for index, path in enumerate(self.paths):
            if self.isInterruptionRequested():
                return
            reader = QImageReader(path)
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid():
                reader.setScaledSize(size.scaled(QSize(104, 88), Qt.KeepAspectRatio))
            image = reader.read()
            if not image.isNull() and not self.isInterruptionRequested():
                self.ready.emit(index, image)


class FilePanel(QWidget):
    file_selected = Signal(str)
    directory_loaded = Signal(list)
    open_photos = Signal()
    open_directory = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self._files = []
        self._workers = set()
        self._thumbnail_generation = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 10, 10)
        layout.setSpacing(6)
        title = QLabel("照片")
        title.setObjectName("titleLabel")
        header = QHBoxLayout()
        header.setSpacing(4)
        header.addWidget(title, 1)
        self.btn_add_photos = QPushButton()
        self.btn_open = QPushButton()
        self.btn_clear = QPushButton()
        for button, name, tip in (
            (self.btn_add_photos, "添加照片", "添加照片… · 选择一张或多张照片"),
            (self.btn_open, "打开文件夹", "打开文件夹…"),
            (self.btn_clear, "清空列表", "清空列表 · 保留原片和修复记录"),
        ):
            button.setObjectName("photoAddBtn" if button is self.btn_add_photos else "photoActionBtn")
            button.setFixedSize(30, 30)
            button.setIconSize(QSize(16, 16))
            button.setAccessibleName(name)
            button.setToolTip(tip)
            header.addWidget(button)
        self.btn_add_photos.clicked.connect(self.open_photos)
        self.btn_open.clicked.connect(self.open_directory)
        layout.addLayout(header)

        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索照片名称")
        self.search.setAccessibleName("搜索照片名称")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._refresh_filter)
        layout.addWidget(self.search)

        filters = QHBoxLayout()
        filters.setSpacing(2)
        self.filters = QButtonGroup(self)
        self.filter_buttons = {}
        for key, text, hint in (
            ("all", "全部", "显示全部照片"),
            ("unscanned", "待扫", "只显示未扫描的照片"),
            ("candidates", "候选", "只显示还有缺陷候选的照片"),
            ("repaired", "修复", "只显示已有修复的照片"),
        ):
            button = QPushButton(text)
            button.setObjectName("photoFilterBtn")
            button.setToolTip(hint)
            button.setCheckable(True)
            button.setProperty("filter", key)
            self.filters.addButton(button)
            self.filter_buttons[key] = button
            filters.addWidget(button, 1)
        self.filter_buttons["all"].setChecked(True)
        self.filters.buttonToggled.connect(lambda _, checked: self._refresh_filter() if checked else None)
        layout.addLayout(filters)

        self.lbl_info = QLabel("将照片或文件夹拖入窗口")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setObjectName("subtitleLabel")
        layout.addWidget(self.lbl_info)

        self.file_list = QListWidget()
        self.file_list.setObjectName("fileList")
        self.file_list.setUniformItemSizes(True)
        self.file_list.setTextElideMode(Qt.ElideMiddle)
        self.file_list.setIconSize(QSize(50, 44))
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        layout.addWidget(self.file_list, 1)

    def update_icons(self, color, primary):
        self.btn_add_photos.setIcon(tool_icon("add-photo", primary))
        self.btn_open.setIcon(tool_icon("open-folder", color))
        self.btn_clear.setIcon(tool_icon("clear-list", color))

    def load_directory(self, directory):
        paths = sorted(Path(directory).iterdir(), key=lambda path: path.name.casefold())
        self.load_files(paths)

    def load_files(self, paths, *, append=False, keep_missing=False):
        photos = [path for path in dict.fromkeys(Path(path).resolve() for path in paths)
                  if path.suffix.lower() in IMAGE_EXTENSIONS and (keep_missing or path.is_file())]
        stems = {}
        for path in photos:
            stems.setdefault(path.parent, set()).add(path.stem.casefold())
        existing = set(self._files) if append else set()
        for file in existing:
            path = Path(file)
            stems.setdefault(path.parent, set()).add(path.stem.casefold())
        files = [str(path) for path in photos
                 if str(path) not in existing and not is_ir_sidecar(path, stems[path.parent])]
        if not append:
            self._thumbnail_generation += 1
            for worker in self._workers:
                worker.requestInterruption()
            self._files.clear()
            self.file_list.clear()
        self._files.extend(files)
        for path in files:
            row = QListWidgetItem(Path(path).name)
            row.setSizeHint(QSize(0, 58))
            row.setToolTip(path)
            row.setData(Qt.UserRole, (False, False, False))
            self.file_list.addItem(row)
        self._refresh_filter()
        if not append:
            self.directory_loaded.emit(files)
        if files:
            worker = Thumbnails(files, self, generation=self._thumbnail_generation)
            self._workers.add(worker)
            worker.ready.connect(self._thumbnail_ready)
            worker.finished.connect(self._thumbnail_finished)
            worker.finished.connect(worker.deleteLater)
            worker.start()
        return files

    def _thumbnail_ready(self, index, image):
        worker = self.sender()
        if worker in self._workers and worker.generation == self._thumbnail_generation:
            path = worker.paths[index]
            if path in self._files:
                self.file_list.item(self._files.index(path)).setIcon(QIcon(QPixmap.fromImage(image)))

    def _thumbnail_finished(self):
        self._workers.discard(self.sender())

    def stop_thumbnails(self):
        for worker in tuple(self._workers):
            worker.requestInterruption()
            worker.wait()

    def select_file(self, filepath, *, quiet=False):
        filepath = str(Path(filepath).resolve()) if filepath is not None else None
        if filepath is None or filepath in self._files:
            self.file_list.blockSignals(quiet)
            self.file_list.setCurrentRow(-1 if filepath is None else self._files.index(filepath))
            self.file_list.blockSignals(False)

    def photo_paths(self):
        return list(self._files)

    def photo_at(self, row):
        return self._files[row] if 0 <= row < len(self._files) else None

    def remove_file(self, path):
        row = self._files.index(path)
        with QSignalBlocker(self.file_list):
            self._files.pop(row)
            self.file_list.takeItem(row)
        self._update_summary()
        for index in (*range(row, len(self._files)), *range(row - 1, -1, -1)):
            if not self.file_list.item(index).isHidden():
                return self._files[index]
        return None

    def set_photo_status(self, path, status, count, *, scanned=False, repaired=False):
        if path in self._files:
            index = self._files.index(path)
            row = self.file_list.item(index)
            name = Path(path).name
            row.setText(name if count is None else f"{name}  ·  {count}")
            row.setToolTip(f"{path}\n{status}")
            row.setData(Qt.UserRole, (scanned, bool(count), repaired))
            self._filter_row(index, row)
            self._update_summary()

    def reveal_file(self, path):
        if path in self._files and self.file_list.item(self._files.index(path)).isHidden():
            self.search.clear()
            self.filter_buttons["all"].setChecked(True)

    def _filter_row(self, index, row):
        mode = self.filters.checkedButton().property("filter")
        scanned, candidates, repaired = row.data(Qt.UserRole)
        show = {"all": True, "unscanned": not scanned, "candidates": candidates, "repaired": repaired}[mode]
        name = Path(self._files[index]).name.casefold()
        row.setHidden(not show or self.search.text().strip().casefold() not in name)

    def _refresh_filter(self, *_):
        for index in range(self.file_list.count()):
            self._filter_row(index, self.file_list.item(index))
        self._update_summary()

    def _update_summary(self):
        rows = [self.file_list.item(index) for index in range(self.file_list.count())]
        if not rows:
            self.lbl_info.setText("将照片或文件夹拖入窗口")
            return
        visible = sum(not row.isHidden() for row in rows)
        scanned = sum(row.data(Qt.UserRole)[0] for row in rows)
        self.lbl_info.setText(f"显示 {visible}/{len(rows)} 张 · 已扫 {scanned} 张")

    def select_next(self):
        self._select_visible(1)

    def select_prev(self):
        self._select_visible(-1)

    def _select_visible(self, step):
        current = self.file_list.currentRow()
        if current < 0 and step < 0:
            current = len(self._files)
        stop = len(self._files) if step > 0 else -1
        for index in range(current + step, stop, step):
            if not self.file_list.item(index).isHidden():
                self.file_list.setCurrentRow(index)
                return

    def _on_file_selected(self, row):
        path = self.photo_at(row)
        if path is not None:
            self.file_selected.emit(path)

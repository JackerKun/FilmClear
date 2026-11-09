"""Import browser with background thumbnail decoding."""

from pathlib import Path

from PySide6.QtCore import QSize, QThread, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QImageReader, QPixmap
from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget
from core.scan_io import is_ir_sidecar

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class Thumbnails(QThread):
    ready = Signal(int, QImage)

    def __init__(self, paths, parent):
        super().__init__(parent)
        self.paths = paths

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
    open_directory = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self._files = []
        self._workers = set()
        self._thumbnail_job = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 12, 12)
        layout.setSpacing(10)
        title = QLabel("照片")
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        self.btn_open = QPushButton("打开文件夹…")
        self.btn_open.setObjectName("secondaryBtn")
        self.btn_open.clicked.connect(self.open_directory)
        layout.addWidget(self.btn_open)

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

    def load_directory(self, directory):
        paths = sorted(Path(directory).iterdir(), key=lambda path: path.name.casefold())
        self.load_files(paths)

    def load_files(self, paths):
        photos = [path for path in dict.fromkeys(Path(path).resolve() for path in paths)
                  if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()]
        stems = {}
        for path in photos:
            stems.setdefault(path.parent, set()).add(path.stem.casefold())
        files = [str(path) for path in photos if not is_ir_sidecar(path, stems[path.parent])]
        if self._thumbnail_job is not None:
            self._thumbnail_job.requestInterruption()
        self._thumbnail_job = None
        self._files = files
        self.file_list.clear()
        for path in files:
            row = QListWidgetItem(Path(path).name)
            row.setSizeHint(QSize(0, 58))
            row.setToolTip(path)
            self.file_list.addItem(row)
        self.lbl_info.setText(f"{len(files)} 张照片" if files else "未找到支持的照片")
        self.directory_loaded.emit(files)
        if not files:
            return
        worker = Thumbnails(files, self)
        self._thumbnail_job = worker
        self._workers.add(worker)
        worker.ready.connect(self._thumbnail_ready)
        worker.finished.connect(self._thumbnail_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _thumbnail_ready(self, index, image):
        if self.sender() is self._thumbnail_job:
            self.file_list.item(index).setIcon(QIcon(QPixmap.fromImage(image)))

    def _thumbnail_finished(self):
        worker = self.sender()
        self._workers.discard(worker)
        if self._thumbnail_job is worker:
            self._thumbnail_job = None

    def stop_thumbnails(self):
        for worker in tuple(self._workers):
            worker.requestInterruption()
            worker.wait()

    def select_file(self, filepath, *, quiet=False):
        if filepath is None:
            return
        filepath = str(Path(filepath).resolve())
        if filepath in self._files:
            self.file_list.blockSignals(quiet)
            self.file_list.setCurrentRow(self._files.index(filepath))
            self.file_list.blockSignals(False)

    def photo_paths(self):
        return list(self._files)

    def set_photo_status(self, path, status, count):
        if path in self._files:
            row = self.file_list.item(self._files.index(path))
            name = Path(path).name
            row.setText(name if count is None else f"{name}  ·  {count}")
            row.setToolTip(f"{path}\n{status}")

    def select_next(self):
        self.file_list.setCurrentRow(min(len(self._files) - 1, self.file_list.currentRow() + 1))

    def select_prev(self):
        if self._files:
            self.file_list.setCurrentRow(max(0, self.file_list.currentRow() - 1))

    def _on_file_selected(self, row):
        if 0 <= row < len(self._files):
            self.file_selected.emit(self._files[row])

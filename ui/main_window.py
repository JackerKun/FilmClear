"""FilmClear workspace and non-destructive editing session."""

from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path
from threading import Event

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QRectF, QSize, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QColor, QKeySequence, QPainter, QPen, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QBoxLayout, QButtonGroup, QComboBox, QFileDialog,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu,
    QMessageBox, QProgressBar, QPushButton, QSizePolicy, QSlider, QSplitter, QStatusBar,
    QSpinBox, QVBoxLayout, QWidget,
)

from config import Config
from core.batch_export import ExportEntry, export_batch, replay_patches
from core.dust_remover import DustRemover
from core.scan_detection import detect_brush, detect_scan
from core.scan_io import TIFF_EXTENSIONS, read_ir_plane, read_scan, signature

from .dust_panel import DustPanel
from .export_dialog import ExportDialog
from .file_panel import FilePanel
from .image_viewer import ImageViewer, defect_path
from .theme_manager import ThemeManager
from .tool_icons import tool_icon


class WorkspaceJob(QThread):
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    counts = Signal(int, int)

    def __init__(self, work, parent):
        super().__init__(parent)
        self.work = work

    def run(self):
        try:
            result = self.work(self._report_progress)
            if not self.isInterruptionRequested():
                self.completed.emit(result)
        except InterruptedError:
            return
        except Exception as error:
            self.failed.emit(str(error))

    def _report_progress(self, message, done=None, total=None):
        if self.isInterruptionRequested():
            raise InterruptedError
        self.progress.emit(message)
        if total is not None:
            self.counts.emit(done, total)


class BusySpinner(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(32, 32)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._advance)

    def _advance(self):
        self._angle = (self._angle - 9) % 360
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(4, 4, 24, 24)
        painter.setPen(QPen(QColor("#384957"), 3))
        painter.drawEllipse(rect)
        painter.setPen(QPen(QColor("#72e5ce"), 3, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, self._angle * 16, 105 * 16)


class EditCommand(QUndoCommand):
    """Keep candidate snapshots and only the pixels touched by a repair."""

    def __init__(self, window, title, before, after, patch=None, revision=None, ignored=()):
        super().__init__(title)
        self.window = window
        self.before = [replace(item) for item in before]
        self.after = [replace(item) for item in after]
        self.patch = patch
        self.old_revision = window._revision
        self.new_revision = self.old_revision if revision is None else revision
        self.old_ignored = list(window._ignored_items)
        self.new_ignored = [*self.old_ignored, *(replace(item) for item in ignored)]

    def undo(self):
        self.window._apply_edit(self.before, self.old_ignored, self.old_revision,
                                self.patch, use_after=False)

    def redo(self):
        self.window._apply_edit(self.after, self.new_ignored, self.new_revision,
                                self.patch, use_after=True)


@dataclass
class PhotoState:
    stamp: tuple | None
    history: QUndoStack
    candidates: list
    ignored: list
    patches: list
    revision: int
    exported_revision: int
    scan_source: str
    ir_override: tuple[str, bool] | None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.theme_manager = ThemeManager()
        self._original_image = None
        self._current_image = None
        self._current_filepath = None
        self._photo_stamp = None
        self._photos = {}
        self._applied_patches = []
        self._infrared = None
        self._ir_source = ""
        self._ir_override = None
        self._scan_source = ""
        self._dust_items = []
        self._ignored_items = []
        self._job = None
        self._export_cancel = None
        self._revision = 0
        self._exported_revision = 0
        self._next_revision = 1
        self._pinned_candidate = None
        self._syncing_comparison = False
        self._compare_leader = None
        self._history = self._new_history()

        self.setWindowTitle("FilmClear — 胶片工作台")
        self.setMinimumSize(1040, 680)
        self.setAcceptDrops(True)
        self._setup_ui()
        self._setup_menu()
        self._setup_loading()
        self._connect_signals()
        self._load_settings()
        self._refresh_controls()

    @staticmethod
    def _tool(label, hint, handler=None, *, checkable=False):
        button = QPushButton(label)
        button.setObjectName("toolBtn")
        button.setToolTip(hint)
        button.setCheckable(checkable)
        if handler is not None:
            button.clicked.connect(handler)
        return button

    def _setup_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)
        self.file_panel = FilePanel()
        self.file_panel.setMinimumWidth(164)
        self.file_panel.setMaximumWidth(300)
        splitter.addWidget(self.file_panel)

        workspace = QWidget()
        workspace.setObjectName("workspace")
        canvas = QVBoxLayout(workspace)
        canvas.setContentsMargins(12, 12, 12, 10)
        canvas.setSpacing(9)
        header = QHBoxLayout()
        self.document_title = QLabel("胶片工作台")
        self.document_title.setObjectName("titleLabel")
        self.document_title.setMinimumWidth(0)
        self.document_title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        header.addWidget(self.document_title, 1)
        self.btn_prev = self._tool("‹", "上一张照片", self.file_panel.select_prev)
        self.btn_next = self._tool("›", "下一张照片", self.file_panel.select_next)
        self.btn_undo = self._tool("撤销", "撤销当前照片的上一步操作", self._undo)
        self.btn_redo = self._tool("重做", "重做上一步操作", self._redo)
        self.btn_compare = self._tool("前后对比", "比较原片和修复结果 · B",
                                      self._toggle_comparison, checkable=True)
        for control in (self.btn_prev, self.btn_next, self.btn_undo, self.btn_redo, self.btn_compare):
            header.addWidget(control)
        canvas.addLayout(header)

        self.image_viewer = ImageViewer()
        self.original_viewer = ImageViewer()
        self.original_viewer.comparison_label = "原片"
        self.original_viewer.hide()
        viewports = QWidget()
        self.view_layout = QBoxLayout(QBoxLayout.LeftToRight, viewports)
        self.view_layout.setContentsMargins(0, 0, 0, 0)
        self.view_layout.setSpacing(6)
        self.view_layout.addWidget(self.original_viewer, 1)
        self.view_layout.addWidget(self.image_viewer, 1)
        canvas.addWidget(viewports, 1)

        self.compare_row = QFrame()
        self.compare_row.setObjectName("canvasTools")
        self.compare_row.setAccessibleName("对比布局")
        compare_layout = QHBoxLayout(self.compare_row)
        compare_layout.setContentsMargins(4, 5, 4, 5)
        compare_layout.setSpacing(4)
        self.compare_group = QButtonGroup(self.compare_row)
        self.compare_buttons = {}
        for mode, name, hint in (
            ("wipe", "滑动分割", "拖动分界线查看原片和修复结果"),
            ("side", "左右并排", "左侧显示原片，右侧显示修复结果"),
            ("stack", "上下排列", "上方显示原片，下方显示修复结果"),
        ):
            button = self._tool(name, hint, checkable=True)
            button.setObjectName("canvasToolBtn")
            button.setProperty("mode", mode)
            self.compare_group.addButton(button)
            self.compare_buttons[mode] = button
            compare_layout.addWidget(button)
        self.compare_buttons["wipe"].setChecked(True)
        self.wipe_controls = QWidget()
        wipe_layout = QHBoxLayout(self.wipe_controls)
        wipe_layout.setContentsMargins(4, 0, 0, 0)
        wipe_layout.addWidget(QLabel("原片"))
        self.compare_slider = QSlider(Qt.Horizontal)
        self.compare_slider.setRange(0, 100)
        self.compare_slider.setValue(50)
        self.compare_slider.setAccessibleName("前后对比分界")
        wipe_layout.addWidget(self.compare_slider, 1)
        wipe_layout.addWidget(QLabel("修复后"))
        compare_layout.addWidget(self.wipe_controls, 1)
        self.compare_hint = QLabel("两张图同步缩放与平移")
        self.compare_hint.setObjectName("subtitleLabel")
        self.compare_hint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.compare_hint.hide()
        compare_layout.addWidget(self.compare_hint, 1)
        self.compare_row.hide()
        canvas.addWidget(self.compare_row)

        self.edit_toolbar = QFrame()
        self.edit_toolbar.setObjectName("canvasTools")
        self.edit_toolbar.setAccessibleName("画布工具")
        edit_layout = QHBoxLayout(self.edit_toolbar)
        edit_layout.setContentsMargins(4, 5, 4, 5)
        edit_layout.setSpacing(2)
        self.edit_group = QButtonGroup(self.edit_toolbar)
        self.edit_buttons = {}
        for mode, label, name, hint in (
            (None, "平移", "浏览 / 平移", "拖动画布平移，滚动缩放"),
            ("detect", "补查", "框选补查", "拖动框住遗漏的缺陷，在原图像素上补查轮廓"),
            ("draw", "描画", "描画补查", "在缺陷附近涂抹或圈起，查找真实缺陷轮廓"),
            ("check", "勾选", "框选勾选", "框内候选会被勾选，按住 Shift 可叠加多个框"),
            ("ignore", "忽略", "框选忽略", "拖框批量忽略框内候选，可撤销"),
            ("repair", "修复", "框选修复", "拖框批量修复框内候选，可撤销"),
        ):
            if mode == "check":
                divider = QFrame()
                divider.setObjectName("canvasToolDivider")
                divider.setFixedSize(1, 22)
                edit_layout.addWidget(divider)
            button = self._tool(label, hint, checkable=True)
            button.setObjectName("canvasToolBtn")
            button.setAccessibleName(name)
            button.setProperty("mode", mode)
            self.edit_group.addButton(button)
            self.edit_buttons[mode] = button
            edit_layout.addWidget(button)
        self.edit_buttons[None].setChecked(True)

        self.brush_radius = QSpinBox()
        self.brush_radius.setRange(4, 48)
        self.brush_radius.setValue(self.image_viewer.brush_radius)
        self.brush_radius.setFixedWidth(48)
        self.brush_radius.setAccessibleName("搜索半径")
        self.brush_radius.setToolTip("搜索半径（屏幕像素）")
        self.brush_radius.hide()
        self.btn_marks = self._tool("标记", "显示或隐藏全部候选标记 · M", checkable=True)
        self.btn_marks.setObjectName("canvasToolBtn")
        self.btn_marks.setAccessibleName("全部标记")
        self.btn_marks.toggled.connect(self._toggle_marks)
        self.btn_marks.setChecked(True)
        edit_layout.addWidget(self.btn_marks)
        edit_layout.addWidget(self.brush_radius)
        edit_layout.addStretch()
        self.btn_fit = self._tool("", "适应窗口 · F", self.image_viewer.fit_to_window)
        self.btn_zoom_out = self._tool("", "缩小", self.image_viewer.zoom_out)
        self.btn_zoom_in = self._tool("", "放大", self.image_viewer.zoom_in)
        self.btn_100 = self._tool("1:1", "原始像素 · 1", self.image_viewer.zoom_100)
        for control in (self.btn_fit, self.btn_zoom_out, self.btn_zoom_in, self.btn_100):
            control.setObjectName("canvasToolBtn")
            control.setAccessibleName(control.toolTip())
            edit_layout.addWidget(control)
        self.lbl_zoom = QLabel("—")
        self.lbl_zoom.setObjectName("zoomLabel")
        self.lbl_zoom.setFixedWidth(35)
        self.lbl_zoom.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        edit_layout.addWidget(self.lbl_zoom)
        canvas.addWidget(self.edit_toolbar)
        self._canvas_icons = {
            **self.compare_buttons,
            **{mode or "pan": button for mode, button in self.edit_buttons.items()},
            "marks": self.btn_marks, "fit": self.btn_fit,
            "zoom-out": self.btn_zoom_out, "zoom-in": self.btn_zoom_in,
        }
        splitter.addWidget(workspace)

        self.dust_panel = DustPanel()
        self.dust_panel.setMinimumWidth(272)
        self.dust_panel.setMaximumWidth(390)
        splitter.addWidget(self.dust_panel)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([205, 890, 310])

        status = QStatusBar()
        self.setStatusBar(status)
        self.lbl_status_file = QLabel("打开胶片扫描照片开始")
        self.lbl_status_info = QLabel("")
        self.lbl_status_dust = QLabel("")
        status.addWidget(self.lbl_status_file, 1)
        status.addPermanentWidget(self.lbl_status_info)
        status.addPermanentWidget(self.lbl_status_dust)

    def _action(self, menu, label, callback, shortcut=None):
        action = QAction(label, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        menu.addAction(action)
        return action

    def _setup_menu(self):
        file_menu = self.menuBar().addMenu("文件")
        self._action(file_menu, "打开照片…", self._on_open_files, QKeySequence.StandardKey.Open)
        self._action(file_menu, "打开文件夹…", self._on_open_directory, "Ctrl+Shift+O")
        self.export_action = self._action(file_menu, "导出修复结果…", self._on_export,
                                          QKeySequence.StandardKey.Save)
        self.batch_export_action = self._action(
            file_menu, "批量导出…", self._on_export_all, "Ctrl+Shift+S"
        )
        self.ir_action = self._action(file_menu, "为当前照片选择红外图…", self._attach_ir)
        self.ir_fourth_action = self._action(
            file_menu, "将当前 TIFF 的第四通道作为红外", self._attach_fourth_channel
        )
        self._action(file_menu, "退出", self.close, QKeySequence.StandardKey.Quit)

        edit_menu = self.menuBar().addMenu("编辑")
        self.undo_action = self._action(edit_menu, "撤销", self._undo, QKeySequence.StandardKey.Undo)
        self.redo_action = self._action(edit_menu, "重做", self._redo, QKeySequence.StandardKey.Redo)

        view_menu = self.menuBar().addMenu("视图")
        self._action(view_menu, "前后对比", self.btn_compare.click, "B")
        self._action(view_menu, "切换候选标记", self.btn_marks.click, "M")
        self._action(view_menu, "适应窗口", self.image_viewer.fit_to_window, "F")
        self._action(view_menu, "原始像素", self.image_viewer.zoom_100, "1")
        self.theme_action = self._action(view_menu, "切换明暗主题", self._on_toggle_theme, "Ctrl+T")

    def _setup_loading(self):
        self.overlay = QFrame(self)
        self.overlay.setObjectName("loadingOverlay")
        layout = QVBoxLayout(self.overlay)
        layout.setAlignment(Qt.AlignCenter)
        card = QFrame()
        card.setObjectName("loadingCard")
        card.setMinimumWidth(270)
        contents = QVBoxLayout(card)
        contents.setContentsMargins(26, 20, 26, 20)
        contents.setSpacing(12)
        contents.addWidget(BusySpinner(), alignment=Qt.AlignHCenter)
        self.overlay_label = QLabel()
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setObjectName("loadingLabel")
        contents.addWidget(self.overlay_label)
        self.overlay_bar = QProgressBar()
        self.overlay_bar.hide()
        contents.addWidget(self.overlay_bar)
        self.overlay_cancel = QPushButton("当前文件完成后停止")
        self.overlay_cancel.clicked.connect(self._cancel_export)
        self.overlay_cancel.hide()
        contents.addWidget(self.overlay_cancel)
        layout.addWidget(card)
        self.overlay.hide()

    def _connect_signals(self):
        self.theme_manager.theme_changed.connect(self._update_canvas_icons)
        self.file_panel.open_directory.connect(self._on_open_directory)
        self.file_panel.file_selected.connect(self._on_file_selected)
        self.file_panel.directory_loaded.connect(self._on_directory_loaded)
        self.dust_panel.btn_detect.clicked.connect(self._on_detect_dust)
        self.dust_panel.dust_selected.connect(self._refresh_marks)
        self.dust_panel.select_all_clicked.connect(self._refresh_marks)
        self.dust_panel.remove_clicked.connect(self._on_remove_dust)
        self.dust_panel.btn_ignore_selected.clicked.connect(
            lambda: self._ignore([item for item in self._dust_items if item.selected])
        )
        self.dust_panel.export_clicked.connect(self._on_export)
        self.dust_panel.dust_clicked.connect(self._on_dust_clicked)
        self.dust_panel.dust_reviewed.connect(self._pin_candidate)
        self.dust_panel.dust_hovered.connect(self._on_dust_hovered)
        self.dust_panel.dismiss_clicked.connect(self._delete_dust_item)
        self.image_viewer.annotation_clicked.connect(self._on_annotation_clicked)
        self.image_viewer.annotation_right_clicked.connect(self._on_annotation_right_clicked)
        self.image_viewer.region_selected.connect(self._on_region_selected)
        self.image_viewer.stroke_drawn.connect(self._on_stroke_drawn)
        self.edit_group.buttonToggled.connect(self._on_edit_mode_changed)
        self.brush_radius.valueChanged.connect(self._on_brush_radius_changed)
        self.image_viewer.zoom_changed.connect(lambda zoom: self.lbl_zoom.setText(f"{zoom}%"))
        self.image_viewer.split_changed.connect(self.compare_slider.setValue)
        self.compare_slider.valueChanged.connect(self.image_viewer.set_split)
        self.compare_group.buttonToggled.connect(
            lambda _, checked: self._toggle_comparison(True) if checked and self.btn_compare.isChecked() else None
        )
        self._compare_timer = QTimer(self)
        self._compare_timer.setSingleShot(True)
        self._compare_timer.timeout.connect(self._sync_comparison)
        for viewer in (self.image_viewer, self.original_viewer):
            viewer.view_changed.connect(lambda v=viewer: self._queue_comparison_sync(v))
        QApplication.instance().aboutToQuit.connect(self._stop_workers)
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.WindowDeactivate and watched is self:
            self._set_space_pan(False)
        if (kind not in (QEvent.ShortcutOverride, QEvent.KeyPress, QEvent.KeyRelease)
                or event.key() != Qt.Key_Space or self._current_image is None):
            return super().eventFilter(watched, event)
        if not isinstance(watched, QWidget) or watched.window() is not self:
            return super().eventFilter(watched, event)
        if not self.image_viewer.space_pan and (
            self._job is not None or event.modifiers() != Qt.NoModifier
            or isinstance(watched, (QAbstractSpinBox, QComboBox, QLineEdit))
        ):
            return super().eventFilter(watched, event)
        if kind != QEvent.ShortcutOverride and not event.isAutoRepeat():
            self._set_space_pan(kind == QEvent.KeyPress)
        event.accept()
        return True

    def _set_space_pan(self, active):
        self.image_viewer.set_space_pan(active)
        self.original_viewer.set_space_pan(active)

    def _new_history(self):
        history = QUndoStack(self)
        history.setUndoLimit(24)
        history.canUndoChanged.connect(self._refresh_controls)
        history.canRedoChanged.connect(self._refresh_controls)
        return history

    def _load_settings(self):
        self.theme_manager.set_theme(self.config.get("theme", "dark"))
        self.resize(self.config.get("window.width", 1400), self.config.get("window.height", 900))
        last = self.config.get("last_directory", "")
        if last and Path(last).is_dir():
            QTimer.singleShot(0, lambda: self._load_directory(last))

    def _save_settings(self):
        self.config.set("theme", self.theme_manager.current_theme)
        self.config.set("window.width", self.width())
        self.config.set("window.height", self.height())
        if self._current_filepath:
            self.config.set("last_directory", str(Path(self._current_filepath).parent))
        self.config.save()

    def _refresh_controls(self, *_args):
        has_image = self._current_image is not None
        is_tiff = self._current_filepath and Path(self._current_filepath).suffix.lower() in TIFF_EXTENSIONS
        busy = self._job is not None
        for control in (self.btn_fit, self.btn_zoom_in, self.btn_zoom_out, self.btn_100,
                        self.edit_toolbar, self.brush_radius, self.btn_marks, self.dust_panel.btn_detect):
            control.setEnabled(has_image and not busy)
        for controls, enabled in (
            ((self.btn_compare,), self._revision != 0),
            ((self.dust_panel.btn_export, self.export_action, self.ir_action), has_image),
            ((self.ir_fourth_action,), bool(is_tiff)),
            ((self.batch_export_action,), bool(self.file_panel.photo_paths()) or has_image),
            ((self.btn_undo, self.undo_action), self._history.canUndo()),
            ((self.btn_redo, self.redo_action), self._history.canRedo()),
        ):
            for control in controls:
                control.setEnabled(enabled and not busy)

    def _start_job(self, label, work, completed, *, on_error=None):
        if self._job is not None:
            return
        job = WorkspaceJob(work, self)
        self._job = job
        job.completed.connect(completed)
        job.failed.connect(on_error if on_error is not None else self._show_error)
        job.progress.connect(self.overlay_label.setText)
        job.counts.connect(self._update_progress)
        job.finished.connect(self._finish_job)
        job.finished.connect(job.deleteLater)
        self.centralWidget().setEnabled(False)
        self._refresh_controls()
        self.overlay_label.setText(label)
        self.overlay_bar.hide()
        self.overlay_cancel.setVisible(self._export_cancel is not None)
        self.overlay_cancel.setEnabled(True)
        self.overlay.setGeometry(self.rect())
        self.overlay.raise_()
        self.overlay.show()
        job.start()

    def _finish_job(self):
        self.overlay.hide()
        self._job = None
        self._export_cancel = None
        self.centralWidget().setEnabled(True)
        self._refresh_controls()

    def _update_progress(self, done, total):
        self.overlay_bar.setRange(0, total)
        self.overlay_bar.setValue(done)
        self.overlay_bar.show()

    def _cancel_export(self):
        if self._export_cancel is not None:
            self._export_cancel.set()
            self.overlay_cancel.setEnabled(False)

    def _show_error(self, message):
        self.overlay.hide()
        self.lbl_status_file.setText("操作未完成")
        QMessageBox.warning(self, "FilmClear", message)

    def _on_open_files(self):
        if self._job is not None:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "打开照片", self._last_directory(),
            "照片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)"
        )
        if paths:
            self.file_panel.load_files(paths)

    def _on_open_directory(self):
        if self._job is not None:
            return
        directory = QFileDialog.getExistingDirectory(self, "打开照片文件夹", self._last_directory())
        if directory:
            self._load_directory(directory)

    def _last_directory(self):
        if self._current_filepath:
            return str(Path(self._current_filepath).parent)
        return str(Path.home())

    def _load_directory(self, directory):
        if self._job is None:
            try:
                self.file_panel.load_directory(directory)
            except OSError as error:
                self._show_error(str(error))

    def _confirm_quit(self):
        self._stash_current()
        pending = sum(state.revision != 0 and state.revision != state.exported_revision
                      for state in self._photos.values())
        if not pending:
            return True
        box = QMessageBox(QMessageBox.Warning, "修复结果尚未导出",
                          f"有 {pending} 张照片的修复尚未导出。退出会结束本次工作记录。", parent=self)
        discard = box.addButton("放弃本次修改", QMessageBox.DestructiveRole)
        cancel = box.addButton("继续编辑", QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        return box.clickedButton() is discard

    def _on_directory_loaded(self, files):
        for path in files:
            if path in self._photos:
                self._update_file_status(path, self._photos[path])
        if files:
            self.file_panel.select_file(files[0], quiet=True)
            self._on_file_selected(files[0])
        else:
            self.lbl_status_file.setText("文件夹中没有支持的照片")

    def _on_file_selected(self, path):
        path = str(Path(path).resolve())
        if path == self._current_filepath:
            return
        if self._job is not None:
            self.file_panel.select_file(self._current_filepath, quiet=True)
            return
        self._load_photo(path)

    def _load_photo(self, path):
        state = self._photos.get(path)

        def read(_progress):
            stamp = signature(path)
            scan = read_scan(path)
            if signature(path) != stamp:
                raise ValueError("读取期间照片发生变化，请重新打开。")
            restored = state if state is not None and state.stamp == stamp else None
            if restored and restored.ir_override:
                ir_path, embedded = restored.ir_override
                try:
                    scan.infrared = read_ir_plane(ir_path, scan.image.shape[:2], embedded=embedded)
                    scan.ir_source = "TIFF 第四通道" if embedded else Path(ir_path).name
                except (OSError, ValueError, cv2.error) as error:
                    scan.ir_notice = str(error)
            image = replay_patches(scan.image, restored.patches) if restored else scan.image
            return scan, image, stamp, restored

        def failed(message):
            self.file_panel.select_file(self._current_filepath, quiet=True)
            self._show_error(f"无法导入 {Path(path).name}\n{message}")

        def loaded(result):
            scan, image, stamp, restored = result
            self._show_photo(path, image, scan=scan, stamp=stamp, restored=restored)

        self._start_job("正在导入照片…", read, loaded, on_error=failed)

    def _stash_current(self):
        if self._current_filepath is None:
            return
        state = PhotoState(
            self._photo_stamp, self._history, self._dust_items, self._ignored_items,
            self._applied_patches, self._revision, self._exported_revision,
            self._scan_source, self._ir_override,
        )
        self._photos[self._current_filepath] = state
        self._update_file_status(self._current_filepath, state)

    def _update_file_status(self, path, state):
        if state.scan_source or state.candidates or state.revision:
            count = len(state.candidates)
            checked = sum(item.selected for item in state.candidates)
            source = state.scan_source or "手动编辑"
            self.file_panel.set_photo_status(
                path, f"{source} · {count} 个候选 · 已勾选 {checked}", count
            )
        else:
            self.file_panel.set_photo_status(path, "未扫描", None)

    def _show_photo(self, path, image, *, scan=None, stamp=None, restored=None):
        self._stash_current()
        stale = self._photos.get(path)
        self._current_filepath = path
        self._photo_stamp = stamp
        self._current_image = image
        self._original_image = scan.image if scan is not None else image
        self._infrared = scan.infrared if scan is not None else None
        self._ir_source = scan.ir_source if scan is not None else ""
        self._revision = restored.revision if restored else 0
        self._exported_revision = restored.exported_revision if restored else 0
        self._history = restored.history if restored else self._new_history()
        if stale and not restored:
            stale.history.deleteLater()
            self._photos.pop(path, None)
        self._pinned_candidate = None
        self._dust_items = restored.candidates if restored else []
        self._ignored_items = restored.ignored if restored else []
        self._applied_patches = restored.patches if restored else []
        self._scan_source = restored.scan_source if restored else ""
        self._ir_override = restored.ir_override if restored else None
        self.btn_compare.setChecked(False)
        self._toggle_comparison(False)
        self.edit_buttons[None].setChecked(True)
        self.image_viewer.load_image(image)
        if self._scan_source or self._dust_items or self._revision:
            self._refresh_candidates()
        else:
            self.dust_panel.clear_list()
            self.lbl_status_dust.clear()
        h, w = image.shape[:2]
        self.document_title.setText(Path(path).name)
        self.document_title.setToolTip(path)
        status = f"已恢复 {self._scan_source} 的候选" if self._scan_source else "照片已导入"
        if not self._scan_source and (self._dust_items or self._revision):
            status = "已恢复手动编辑"
        if stale and not restored:
            status = "原图已变化，旧扫描记录已重置"
        if scan and scan.ir_notice:
            status += f" · {scan.ir_notice}"
        self.lbl_status_file.setText(status)
        self._update_image_info(w, h)

    def _update_image_info(self, w, h):
        source = " · IR" if self._infrared is not None else ""
        self.lbl_status_info.setText(f"{w:,} × {h:,} px{source}")
        self.lbl_status_info.setToolTip(self._ir_source)

    def _attach_ir(self):
        if self._current_image is None or self._job is not None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择已对齐的红外图", self._last_directory(), "红外图 (*.tif *.tiff *.png)"
        )
        if path:
            self._load_ir_override(path)

    def _attach_fourth_channel(self):
        if self._current_filepath and self._job is None:
            self._load_ir_override(self._current_filepath, embedded=True)

    def _load_ir_override(self, path, *, embedded=False):
        h, w = self._current_image.shape[:2]

        def loaded(plane):
            self._infrared = plane
            self._ir_override = (path, embedded)
            self._ir_source = "TIFF 第四通道" if embedded else Path(path).name
            self.dust_panel.combo_source.setCurrentIndex(0)
            self._update_image_info(w, h)
            self.lbl_status_file.setText("已关联红外图，可重新扫描")

        self._start_job("正在读取红外图…",
                        lambda _: read_ir_plane(path, (h, w), embedded=embedded), loaded)

    def dragEnterEvent(self, event):
        if self._job is None and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if self._job is not None:
            return
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if len(paths) == 1 and paths[0].is_dir():
            self._load_directory(paths[0])
        else:
            self.file_panel.load_files(paths)
        event.acceptProposedAction()

    def _toggle_comparison(self, enabled):
        mode = self.compare_group.checkedButton().property("mode")
        paired = enabled and mode != "wipe"
        self._compare_timer.stop()
        self._syncing_comparison = True
        if enabled:
            self.edit_buttons[None].setChecked(True)
        self.image_viewer.clear_comparison()
        self.image_viewer.comparison_label = "修复后" if paired else ""
        self.wipe_controls.setVisible(mode == "wipe")
        self.compare_hint.setVisible(paired)
        self.compare_row.setVisible(enabled)
        self.original_viewer.setVisible(paired)
        if paired:
            direction = QBoxLayout.LeftToRight if mode == "side" else QBoxLayout.TopToBottom
            self.view_layout.setDirection(direction)
            self.view_layout.activate()
            if not self.original_viewer.get_image_size()[0]:
                self.original_viewer.load_image(self._original_image)
            self.original_viewer.match_view(self.image_viewer)
        else:
            self.original_viewer.clear_image()
            if enabled:
                self.image_viewer.set_comparison(self._original_image)
        self.image_viewer.viewport().update()
        self._syncing_comparison = False

    def _queue_comparison_sync(self, viewer):
        if self._syncing_comparison or self.original_viewer.isHidden():
            return
        self._compare_leader = viewer
        self._compare_timer.start(0)

    def _sync_comparison(self):
        if self.original_viewer.isHidden():
            return
        leader = self._compare_leader
        follower = self.original_viewer if leader is self.image_viewer else self.image_viewer
        self._syncing_comparison = True
        follower.match_view(leader)
        self._syncing_comparison = False
        self.lbl_zoom.setText(f"{self.image_viewer.get_zoom_percent()}%")

    def _toggle_marks(self, enabled):
        self.image_viewer.set_show_all(enabled)

    def _update_canvas_icons(self, theme):
        color = "#e5edf4" if theme == ThemeManager.DARK else "#202a35"
        for name, button in self._canvas_icons.items():
            button.setIconSize(QSize(13, 13))
            button.setIcon(tool_icon(name, color))
            button.setMinimumWidth(button.sizeHint().width())

    def _on_edit_mode_changed(self, button, checked):
        if not checked:
            return
        mode = button.property("mode")
        if mode and self.btn_compare.isChecked():
            self.btn_compare.setChecked(False)
            self._toggle_comparison(False)
        self.image_viewer.set_selection_mode(mode)
        if mode in ("check", "ignore", "repair"):
            self.btn_marks.setChecked(True)
        self.brush_radius.setVisible(mode == "draw")
        self.lbl_status_file.setText(button.toolTip() if mode else "就绪")

    def _on_brush_radius_changed(self, radius):
        self.image_viewer.brush_radius = radius

    def _on_stroke_drawn(self, points):
        radius = ceil(self.brush_radius.value() / self.image_viewer.transform().m11())
        self._start_detection(brush=(points, radius))

    def _on_detect_dust(self):
        self._start_detection()

    def _on_region_selected(self, x1, y1, x2, y2, additive=False):
        if self._job is not None:
            return
        mode = self.edit_group.checkedButton().property("mode")
        if mode in (None, "detect"):
            self._start_detection((x1, y1, x2, y2))
            return
        box = QRectF(x1, y1, x2 - x1, y2 - y1)
        items = [item for item in self._dust_items if defect_path(item).intersects(box)]
        if not items:
            self.lbl_status_file.setText("框内没有候选 · 可切换至框选补查")
        elif mode == "ignore":
            self._ignore(items)
        elif mode == "repair":
            self._repair(items)
        elif mode == "check":
            ids = {item.id for item in items}
            updated = [replace(item, selected=item.id in ids or (additive and item.selected))
                       for item in self._dust_items]
            self._history.push(EditCommand(self, "框选勾选", self._dust_items, updated))
            self.lbl_status_file.setText(f"框内有 {len(items)} 处候选 · 可在列表核对")

    def _start_detection(self, region=None, *, brush=None):
        if self._current_image is None or self._job is not None:
            return
        image = self._current_image
        params = self.dust_panel.get_detection_params()
        quick = self.dust_panel.combo_quality.currentData()
        subject_first = self.dust_panel.combo_focus.currentData()
        infrared = self._infrared if self.dust_panel.combo_source.currentData() else None

        def scan(progress):
            def report(source, done, total):
                progress(f"正在扫描 {source} · {done} / {total}", done, total)

            if brush is not None:
                return detect_brush(image, params, *brush, infrared=infrared, progress=report)
            return detect_scan(
                image, params, infrared=infrared, quick=quick, region=region,
                subject_first=subject_first, progress=report,
            )

        label = "正在检测描画范围…" if brush is not None else "正在扫描灰尘和毛絮…"
        self._start_job(label, scan, lambda result: self._on_detection_finished(
            result.items, region, result.source, brushed=brush is not None
        ))

    def _on_detection_finished(self, items, region, source="RGB 原图", *, brushed=False):
        items = [item for item in items if not any(
            self._same_defect(item, ignored) for ignored in self._ignored_items
        )]
        for item in items:
            item.selected = True
        focus_id = None
        local = region is not None or brushed
        if not local:
            merged = items
            self._scan_source = source
        else:
            merged = [replace(item) for item in self._dust_items]
            self._scan_source = self._scan_source or source
            next_id = max((item.id for item in merged), default=0) + 1
            for item in items:
                existing = next((dust for dust in merged if self._same_defect(dust, item)), None)
                if existing is not None:
                    existing.selected = True
                    item.id = existing.id
                    linear = ("scratch", "fiber")
                    if item.dust_type in linear and (existing.dust_type not in linear or item.area > existing.area):
                        merged[merged.index(existing)] = replace(item, id=existing.id)
                else:
                    item.id = next_id
                    next_id += 1
                    merged.append(item)
                if focus_id is None:
                    focus_id = item.id
        if not local or items:
            title = "描画补查" if brushed else "框选补查" if local else "扫描候选"
            self._history.push(EditCommand(self, title, self._dust_items, merged))
        if focus_id is not None:
            self.dust_panel.select_candidate(focus_id)
            self._pin_candidate(focus_id)
        self._stash_current()
        if brushed:
            status = f"描画范围找到 {len(items)} 处候选" if items else "描画范围内未找到缺陷"
        elif local:
            status = f"{source} · 框选补查完成" if items else "框内未找到候选 · 可试试描画补查"
        else:
            status = f"{source} · 全图扫描完成"
        self.lbl_status_file.setText(status)

    @staticmethod
    def _same_defect(first, second):
        for outer, inner in ((first, second), (second, first)):
            x, y = inner.center
            if not (outer.x <= x < outer.x + outer.width and outer.y <= y < outer.y + outer.height):
                continue
            if outer.contour is None or (
                cv2.pointPolygonTest(outer.contour, (x, y), False) >= 0
                and all(cv2.pointPolygonTest(hole, (x, y), False) <= 0 for hole in outer.holes)
            ):
                return True
        return False

    def _refresh_candidates(self):
        self._pinned_candidate = None
        self.image_viewer.update_annotations(self._dust_items)
        self.dust_panel.update_dust_list(self._dust_items)
        self.lbl_status_dust.setText(f"{len(self._dust_items)} 个候选")

    def _refresh_marks(self, *_args):
        self.image_viewer.viewport().update()
        self._stash_current()

    def _on_dust_hovered(self, dust_id):
        if dust_id == -1:
            dust_id = self._pinned_candidate
        item = next((item for item in self._dust_items if item.id == dust_id), None)
        self.image_viewer.highlight_dust(dust_id)
        self.dust_panel.show_candidate(self._current_image, item)

    def _on_dust_clicked(self, dust_id):
        self._pin_candidate(dust_id)
        self.image_viewer.highlight_dust(dust_id, locate=True)

    def _pin_candidate(self, dust_id):
        self._pinned_candidate = dust_id
        self._on_dust_hovered(dust_id)

    def _on_annotation_clicked(self, dust_id):
        self.dust_panel.select_candidate(dust_id)
        self._pinned_candidate = dust_id
        self._on_dust_hovered(dust_id)

    def _on_annotation_right_clicked(self, dust_id, position):
        menu = QMenu(self)
        repair = menu.addAction("修复此项")
        dismiss = menu.addAction("忽略此项")
        action = menu.exec(position)
        if action is repair:
            self._repair_single_dust(dust_id)
        elif action is dismiss:
            self._delete_dust_item(dust_id)

    def _delete_dust_item(self, dust_id):
        self._ignore([item for item in self._dust_items if item.id == dust_id])

    def _ignore(self, items):
        if self._job is not None or not items:
            return
        ids = {item.id for item in items}
        remaining = [item for item in self._dust_items if item.id not in ids]
        title = f"忽略 {len(items)} 处缺陷"
        self._history.push(EditCommand(self, title, self._dust_items, remaining, ignored=items))
        self.lbl_status_file.setText(f"已{title} · 可撤销")

    def _repair_single_dust(self, dust_id):
        self._repair([item for item in self._dust_items if item.id == dust_id])

    def _on_remove_dust(self):
        self._repair([item for item in self._dust_items if item.selected])

    def _repair(self, items):
        if not items or self._current_image is None or self._job is not None:
            return
        image = self._current_image
        before = list(self._dust_items)
        ids = {item.id for item in items}
        after = [item for item in before if item.id not in ids]
        remover = DustRemover(**self.dust_panel.get_inpaint_params())

        def repair(_progress):
            mask = remover.create_mask(image.shape, items, selected_only=False)
            repaired = remover.apply_mask(image, mask)
            positions = np.flatnonzero(mask.ravel())
            return positions, image.reshape(-1, 3)[positions], repaired.reshape(-1, 3)[positions]

        def commit(patch):
            revision = self._next_revision
            self._next_revision += 1
            self._history.push(EditCommand(self, f"修复 {len(items)} 处缺陷",
                                           before, after, patch, revision))
            self.lbl_status_file.setText(f"已修复 {len(items)} 处，可前后对比")

        self._start_job("正在修复选中的缺陷…", repair, commit)

    def _apply_edit(self, items, ignored, revision, patch, *, use_after):
        if patch is not None:
            positions, before, after = patch
            if use_after:
                self._applied_patches.append((revision, positions, after))
            else:
                self._applied_patches.pop()
            if revision == 0:
                self._current_image = self._original_image
            else:
                if self._current_image is self._original_image:
                    self._current_image = self._original_image.copy()
                self._current_image.reshape(-1, 3)[positions] = after if use_after else before
            self.image_viewer.replace_image(self._current_image)
        self._revision = revision
        if revision == 0 and self.btn_compare.isChecked():
            self.btn_compare.setChecked(False)
            self._toggle_comparison(False)
        self._dust_items = [replace(item) for item in items]
        self._ignored_items = list(ignored)
        self._refresh_candidates()
        self._stash_current()
        self._refresh_controls()

    def _undo(self):
        if self._job is None and self._history.canUndo():
            self.lbl_status_file.setText(f"已撤销 · {self._history.undoText()}")
            self._history.undo()

    def _redo(self):
        if self._job is None and self._history.canRedo():
            self.lbl_status_file.setText(f"已重做 · {self._history.redoText()}")
            self._history.redo()

    def _on_export(self):
        self._show_export_dialog(batch=False)

    def _on_export_all(self):
        self._show_export_dialog(batch=True)

    def _show_export_dialog(self, *, batch):
        if self._job is not None:
            return
        self._stash_current()
        paths = list(dict.fromkeys([
            *self.file_panel.photo_paths(), *self._photos, self._current_filepath,
        ]))
        rows = []
        for path in filter(None, paths):
            state = self._photos.get(path)
            checked = sum(item.selected for item in state.candidates) if state else 0
            edited = bool(state and state.revision)
            has_work = bool(checked or edited)
            status = (f"{checked} 项已勾选" if state and (state.scan_source or state.candidates)
                      else "未扫描")
            if edited:
                status += " · 已有修复"
            rows.append((path, has_work, status))
        if not rows:
            return
        if batch and not any(row[1] for row in rows):
            rows = [(path, path == self._current_filepath, status) for path, _, status in rows]
        default = self.config.get("output_dir", "") or self._last_directory()
        dialog = ExportDialog(rows, self._current_filepath, Path(default).resolve(), batch, self)
        if dialog.exec() == ExportDialog.Accepted:
            self._start_export(*dialog.selection())

    def _start_export(self, paths, directory, extension, mode):
        if self._job is not None or not paths:
            return
        self._stash_current()
        entries = []
        for path in paths:
            path = str(Path(path).resolve())
            state = self._photos.get(path)
            entries.append(ExportEntry(
                Path(path), state.stamp if state else None,
                tuple(state.patches) if state else (),
                tuple(item for item in state.candidates if item.selected) if state else (),
                state.revision if state else 0,
            ))
        cancel = self._export_cancel = Event()
        repair_params = self.dust_panel.get_inpaint_params()
        self.config.set("output_dir", str(directory))

        def save(progress):
            return export_batch(entries, directory, extension, mode, repair_params, progress, cancel.is_set)

        def finished(result):
            saved, errors = result
            if mode != "original":
                for path, revision, _target in saved:
                    state = self._photos.get(str(path))
                    if state:
                        state.exported_revision = revision
                    if str(path) == self._current_filepath:
                        self._exported_revision = revision
            status = f"已导出 {len(saved)} / {len(entries)} 张"
            if errors:
                status += f" · 失败 {len(errors)} 张"
            if cancel.is_set() and len(saved) + len(errors) < len(entries):
                status += " · 已停止"
            self.lbl_status_file.setText(status)
            self.overlay.hide()
            if errors:
                box = QMessageBox(QMessageBox.Warning, "导出结果", status, parent=self)
                box.setDetailedText("\n".join(f"{path.name} · {reason}" for path, reason in errors))
                box.exec()
            elif saved:
                self.lbl_status_file.setToolTip(str(directory))

        self._start_job("正在准备原图导出…", save, finished)

    def _on_toggle_theme(self):
        self.theme_manager.toggle_theme()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "overlay"):
            self.overlay.setGeometry(self.rect())

    def _stop_workers(self):
        if self._job is not None:
            self._job.requestInterruption()
            self._job.wait()
        self.file_panel.stop_thumbnails()

    def closeEvent(self, event):
        if self._job is not None or not self._confirm_quit():
            event.ignore()
            return
        self._stop_workers()
        self._save_settings()
        event.accept()

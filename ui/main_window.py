"""FilmClear workspace and non-destructive editing session."""

from dataclasses import dataclass, field, replace
import logging
from math import ceil
from pathlib import Path
import sqlite3
import sys
from threading import Event
from zipfile import BadZipFile

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPoint, QProcess, QRectF, QSize, QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence, QPainter, QPen, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QBoxLayout, QButtonGroup, QColorDialog, QComboBox, QDialog,
    QDialogButtonBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu,
    QProgressBar, QPushButton, QSizePolicy, QSlider, QSplitter, QStatusBar,
    QSpinBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from config import Config
from core.batch_export import ExportEntry, export_batch, replay_patches
from core.dust_remover import DustRemover
from core.frame_crop import crop_from_edges, crop_plan, detect_frame
from core.inspection import dust_view, infrared_view
from core.manual_retouch import covered_candidates, retouch_patch
from core.repeated_defects import repeated_defects
from core.review_report import create_review_report
from core.scan_detection import detect_brush, detect_scan, refine_candidate
from core.scan_io import TIFF_EXTENSIONS, Scan, read_ir_plane, read_scan, signature
from core.workspace_store import WorkspaceStore
from release_check import GITHUB_PROJECT

from .batch_crop_dialog import BatchCropDialog, CropSuggestion
from .crop_dialog import CropDialog
from .dust_panel import DustPanel
from .export_dialog import ExportDialog
from .file_panel import IMAGE_EXTENSIONS, FilePanel
from .image_viewer import DEFAULT_CANVAS_BACKGROUND, RETOUCH_MODES, ImageViewer, defect_path
from .notice_dialog import NoticeDialog
from .numeric_fields import use_standard_digits
from .scan_queue import ScanQueueDialog
from .repeat_dialog import RepeatDialog
from .theme_manager import ThemeManager
from .tool_icons import tool_icon
from .update_checker import UpdateChecker


class WorkspaceJob(QThread):
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    counts = Signal(int, int)
    item_ready = Signal(object)

    def __init__(self, work, parent, *, stream=False):
        super().__init__(parent)
        self.work = work
        self.stream = stream

    def run(self):
        try:
            args = (self._report_progress, self.item_ready.emit) if self.stream else (self._report_progress,)
            result = self.work(*args)
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
        self.setObjectName("loadingSpinner")
        self.setFixedSize(22, 22)
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
        rect = QRectF(3, 3, 16, 16)
        accent = self.palette().windowText().color()
        track = QColor(accent)
        track.setAlpha(55)
        painter.setPen(QPen(track, 2))
        painter.drawEllipse(rect)
        painter.setPen(QPen(accent, 2, Qt.SolidLine, Qt.RoundCap))
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

    def repaired_items(self):
        remaining = {item.id for item in self.after}
        return [replace(item, selected=True) for item in self.before if item.id not in remaining]

    def undo(self):
        if self.window._restoring_workspace:
            return
        self.window._apply_edit(self.before, self.old_ignored, self.old_revision,
                                self.patch, use_after=False)

    def redo(self):
        if self.window._restoring_workspace:
            return
        self.window._apply_edit(self.after, self.new_ignored, self.new_revision,
                                self.patch, use_after=True, repaired=self.repaired_items() if self.patch else ())


class CropCommand(QUndoCommand):
    patch = None

    def __init__(self, window, settings, revision, *, state=None):
        super().__init__("取消裁切" if settings is None else "裁切与旋转")
        self.window = window
        self.before = window._crop if state is None else state.crop
        self.after = settings
        self.old_revision = window._revision if state is None else state.revision
        self.new_revision = revision

    def undo(self):
        if not self.window._restoring_workspace:
            self.window._apply_crop(self.before, self.old_revision)

    def redo(self):
        if not self.window._restoring_workspace:
            self.window._apply_crop(self.after, self.new_revision)


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
    view: tuple | None = None
    size: tuple | None = None
    repaired_marks: dict = field(default_factory=dict)
    crop: dict | None = None


class MainWindow(QMainWindow):
    def __init__(self, *, restore_workspace=True):
        super().__init__()
        self.config = Config()
        self._workspace = WorkspaceStore(self.config.config_path)
        self._workspace_dirty = set()
        self._workspace_manifest_dirty = False
        self._restoring_workspace = False
        self._resume_path = None
        self._workspace_timer = QTimer(self)
        self._workspace_timer.setSingleShot(True)
        self._workspace_timer.setInterval(450)
        self._workspace_timer.timeout.connect(self._save_workspace)
        self.theme_manager = ThemeManager()
        self._original_image = None
        self._current_image = None
        self._current_filepath = None
        self._photo_stamp = None
        self._photos = {}
        self._applied_patches = []
        self._repaired_marks = {}
        self._crop = None
        self._infrared = None
        self._ir_source = ""
        self._ir_override = None
        self._scan_source = ""
        self._dust_items = []
        self._ignored_items = []
        self._job = None
        self._export_cancel = None
        self._last_export_folder = None
        self._last_review_report = None
        self._failed_scan_paths = []
        self._revision = 0
        self._exported_revision = 0
        self._next_revision = 1
        self._pinned_candidate = None
        self._syncing_comparison = False
        self._compare_leader = None
        self._history = self._new_history()
        self._manual_update_check = False
        self._update_checker = UpdateChecker(QApplication.applicationVersion(), self)
        self._update_checker.check_finished.connect(self._finish_update_check)

        self.setWindowTitle("FilmClear — 胶片工作台")
        self.setMinimumSize(1040, 680)
        self.setAcceptDrops(True)
        self._setup_ui()
        self._setup_menu()
        self._setup_loading()
        self._connect_signals()
        self._load_settings(restore_workspace=restore_workspace)
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
        self._sidebar_buttons = {}
        self._sidebar_header_buttons = {}
        for side in ("left", "right"):
            edge_button = QPushButton(self)
            header_button = QPushButton()
            for button, width, height, icon_size in ((edge_button, 20, 44, 14), (header_button, 28, 28, 16)):
                button.setObjectName("sidebarToggle")
                button.setCheckable(True)
                button.setFixedSize(width, height)
                button.setIconSize(QSize(icon_size, icon_size))
                button.setChecked(True)
            self._sidebar_buttons[side] = edge_button
            self._sidebar_header_buttons[side] = header_button
        header.addWidget(self._sidebar_header_buttons["left"])
        self.document_title = QLabel("胶片工作台")
        self.document_title.setObjectName("titleLabel")
        self.document_title.setMinimumWidth(0)
        self.document_title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        header.addWidget(self.document_title, 1)
        self.btn_prev = self._tool("‹", "上一张照片", self.file_panel.select_prev)
        self.btn_next = self._tool("›", "下一张照片", self.file_panel.select_next)
        self.btn_undo = self._tool("撤销", "撤销当前照片的上一步操作", self._undo)
        self.btn_redo = self._tool("重做", "重做上一步操作", self._redo)
        self.btn_compare = self._tool("前后对比", "比较原片和修复结果",
                                      self._toggle_comparison, checkable=True)
        for control in (self.btn_prev, self.btn_next, self.btn_undo, self.btn_redo, self.btn_compare):
            header.addWidget(control)
        header.addWidget(self._sidebar_header_buttons["right"])
        canvas.addLayout(header)

        self._update_banner = QFrame()
        self._update_banner.setObjectName("canvasTools")
        update_layout = QHBoxLayout(self._update_banner)
        update_layout.setContentsMargins(10, 4, 4, 4)
        self._update_label = QLabel()
        self._update_label.setTextFormat(Qt.PlainText)
        update_layout.addWidget(self._update_label, 1)
        self._update_release_page = QUrl()
        download = QPushButton("去下载")
        download.setToolTip("在浏览器打开 GitHub 发布页")
        download.clicked.connect(lambda: QDesktopServices.openUrl(self._update_release_page))
        update_layout.addWidget(download)
        dismiss = QPushButton("稍后")
        dismiss.setObjectName("secondaryBtn")
        dismiss.clicked.connect(self._update_banner.hide)
        update_layout.addWidget(dismiss)
        self._update_banner.hide()
        canvas.addWidget(self._update_banner)

        self.image_viewer = ImageViewer()
        self.original_viewer = ImageViewer()
        self.original_viewer.comparison_label = "原片"
        self.original_viewer.hide()
        viewports = self._canvas_viewports = QWidget()
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
            (None, "指针", "指针", "点击选择候选，拖动可平移画布"),
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
        use_standard_digits(self.brush_radius)
        self.brush_radius.setRange(4, 48)
        self.brush_radius.setValue(self.image_viewer.brush_radius)
        self.brush_radius.setFixedWidth(48)
        self.brush_radius.setAccessibleName("搜索半径")
        self.brush_radius.setToolTip("搜索半径（屏幕像素） · [ 缩小 / ] 放大")
        self.brush_radius.hide()
        self.btn_marks = self._tool("标记", "显示或隐藏缺陷标记，包含对比中的原片", checkable=True)
        self.btn_marks.setObjectName("canvasToolBtn")
        self.btn_marks.setAccessibleName("全部标记")
        self.btn_marks.toggled.connect(self._toggle_marks)
        self.btn_marks.setChecked(True)
        edit_layout.addWidget(self.btn_marks)
        edit_layout.addWidget(self.brush_radius)
        edit_layout.addStretch()
        self.btn_fit = self._tool("", "适应窗口", self.image_viewer.fit_to_window)
        self.btn_zoom_out = self._tool("", "缩小", self.image_viewer.zoom_out)
        self.btn_zoom_in = self._tool("", "放大", self.image_viewer.zoom_in)
        self.btn_100 = self._tool("1:1", "原始像素", self.image_viewer.zoom_100)
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
        canvas.addWidget(self._setup_retouch_toolbar())
        self._canvas_icons = {
            **self.compare_buttons,
            **{mode or "pointer": button for mode, button in self.edit_buttons.items()},
            "marks": self.btn_marks, "fit": self.btn_fit,
            "zoom-out": self.btn_zoom_out, "zoom-in": self.btn_zoom_in,
            "crop": self.btn_crop,
            "batch-crop": self.btn_batch_crop,
        }
        splitter.addWidget(workspace)

        self.dust_panel = DustPanel()
        self.dust_panel.setMinimumWidth(272)
        self.dust_panel.setMaximumWidth(390)
        splitter.addWidget(self.dust_panel)
        self._sidebar_panels = {"left": self.file_panel, "right": self.dust_panel}
        self._sidebar_widths = {"left": 205, "right": 310}
        self._sidebar_handles = {"left": splitter.handle(1), "right": splitter.handle(2)}
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([205, 890, 310])

        status = QStatusBar()
        self.setStatusBar(status)
        self.lbl_status_file = QLabel("打开胶片扫描照片开始")
        self.lbl_status_info = QLabel("")
        self.lbl_status_dust = QLabel("")
        status.addWidget(self.lbl_status_file, 1)
        self.btn_export_folder = QPushButton("导出目录")
        self.btn_review_report = QPushButton("查看对照")
        for button, handler in ((self.btn_export_folder, self._open_export_folder),
                                (self.btn_review_report, self._open_review_report)):
            button.setObjectName("textBtn")
            button.clicked.connect(handler)
            status.addPermanentWidget(button)
            button.hide()
        status.addPermanentWidget(self.lbl_status_info)
        status.addPermanentWidget(self.lbl_status_dust)

    def check_for_updates(self, *, manual=False):
        if manual:
            self._manual_update_check = True
            self._check_update_action.setEnabled(False)
            self.statusBar().showMessage("正在检查更新…")
        self._update_checker.check()

    def _finish_update_check(self, tag, release_page):
        if not release_page.isEmpty():
            self._update_label.setText(f"FilmClear {tag} 已发布")
            self._update_release_page = release_page
            self._update_banner.show()
        elif tag:
            self._update_banner.hide()
        if self._manual_update_check:
            self._manual_update_check = False
            self._check_update_action.setEnabled(True)
            if not release_page.isEmpty():
                message = f"发现新版本 {tag}"
            elif tag:
                message = f"当前版本 {QApplication.applicationVersion()}，暂无可用更新"
            else:
                message = "暂时无法检查更新，请稍后重试"
            self.statusBar().showMessage(message, 7000)

    def _setup_retouch_toolbar(self):
        toolbar = QFrame()
        toolbar.setObjectName("canvasTools")
        toolbar.setAccessibleName("手动修图工具")
        self.retouch_toolbar = toolbar
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)
        label = QLabel("手动修图")
        label.setObjectName("subtitleLabel")
        layout.addWidget(label)
        for mode, title, hint in (
            ("clone", "仿制图章", "Option/Alt + 点击设置取样点，然后在目标处点击或拖画"),
            ("spot", "污点修复", "点击污点或沿着划痕拖画，用周围像素修补笔迹覆盖区域"),
        ):
            button = self._tool(title, hint, checkable=True)
            button.setObjectName("canvasToolBtn")
            button.setAccessibleName(title)
            button.setProperty("mode", mode)
            self.edit_group.addButton(button)
            self.edit_buttons[mode] = button
            layout.addWidget(button)
        self.btn_crop = self._tool("裁切", "裁切扫描边框与微调旋转", self._on_crop)
        self.btn_crop.setObjectName("canvasToolBtn")
        layout.addWidget(self.btn_crop)
        self.btn_batch_crop = self._tool("批量裁切", "统一留边并识别整卷片框", self._on_batch_crop)
        self.btn_batch_crop.setObjectName("canvasToolBtn")
        layout.addWidget(self.btn_batch_crop)
        layout.addStretch()

        self.retouch_radius = QSpinBox()
        self.retouch_hardness = QSpinBox()
        self.retouch_opacity = QSpinBox()
        for spin, title, tip, limits, value, suffix in (
            (self.retouch_radius, "半径", "工具半径（原图像素） · [ 缩小 / ] 放大", (1, 512), 14, " px"),
            (self.retouch_hardness, "硬度", "降低硬度可让笔刷边缘渐变", (0, 100), 70, "%"),
            (self.retouch_opacity, "不透明度", "降低不透明度可保留部分原图", (1, 100), 100, "%"),
        ):
            use_standard_digits(spin)
            spin.setRange(*limits)
            spin.setValue(value)
            spin.setPrefix(title + " ")
            spin.setSuffix(suffix)
            spin.setAccessibleName(title)
            spin.setToolTip(tip)
            spin.hide()
            layout.addWidget(spin)
        self.retouch_radius.valueChanged.connect(self._on_retouch_radius_changed)
        return toolbar

    def _action(self, menu, label, callback, shortcut=None):
        action = QAction(label, self)
        if isinstance(shortcut, QKeySequence.StandardKey):
            action.setShortcuts(shortcut)
        elif shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        menu.addAction(action)
        return action

    def _button_shortcut(self, menu, label, button, shortcut):
        action = self._action(menu, label, button.click, shortcut)
        self._shortcut_buttons[button] = action
        tip = button.toolTip() or label
        button.setToolTip(f"{tip} · {action.shortcut().toString(QKeySequence.NativeText)}")

    def _setup_menu(self):
        self._shortcut_buttons = {}
        file_menu = self.menuBar().addMenu("文件")
        self._action(file_menu, "添加照片…", self._on_open_files, QKeySequence.StandardKey.Open)
        self._action(file_menu, "打开文件夹…", self._on_open_directory, "Ctrl+Shift+O")
        self.reveal_photo_action = self._action(file_menu, "打开照片所在位置",
                                                lambda: self._reveal_photo(self._current_filepath))
        self.reset_photo_action = self._action(file_menu, "重置当前照片…",
                                               lambda: self._reset_photo(self._current_filepath))
        self.remove_photo_action = self._action(file_menu, "从列表移除当前照片",
                                                lambda: self._remove_photo(self._current_filepath))
        self.clear_photos_action = self._action(file_menu, "清空照片列表", self._clear_photos)
        file_menu.addSeparator()
        self.export_action = self._action(file_menu, "导出修复结果…", self._on_export,
                                          QKeySequence.StandardKey.Save)
        self.batch_scan_action = self._action(file_menu, "批量扫描…", self._show_scan_queue, "Ctrl+Shift+Return")
        self.retry_scan_action = self._action(file_menu, "重试失败的扫描", self._retry_batch_scan)
        self.batch_export_action = self._action(
            file_menu, "批量导出…", self._on_export_all, "Ctrl+Shift+S"
        )
        self.ir_action = self._action(file_menu, "为当前照片选择红外图…", self._attach_ir)
        self.ir_fourth_action = self._action(
            file_menu, "将当前 TIFF 的第四通道作为红外", self._attach_fourth_channel
        )
        self._button_shortcut(file_menu, "上一张照片", self.btn_prev, "PgUp")
        self._button_shortcut(file_menu, "下一张照片", self.btn_next, "PgDown")
        self._action(file_menu, "退出", self.close, QKeySequence.StandardKey.Quit)

        edit_menu = self.menuBar().addMenu("编辑")
        self.undo_action = self._action(edit_menu, "撤销", self._undo, QKeySequence.StandardKey.Undo)
        self.redo_action = self._action(edit_menu, "重做", self._redo, QKeySequence.StandardKey.Redo)
        self.repeat_action = self._action(edit_menu, "同卷重复候选…", self._show_repeated)
        self._button_shortcut(edit_menu, "扫描缺陷", self.dust_panel.btn_detect, "Ctrl+Return")
        self._button_shortcut(edit_menu, "修复所选", self.dust_panel.btn_remove, "Ctrl+R")
        self._button_shortcut(edit_menu, "忽略所选", self.dust_panel.btn_ignore_selected, "Ctrl+Shift+I")
        self._button_shortcut(edit_menu, "勾选可见候选", self.dust_panel.btn_check_visible, "Ctrl+Shift+A")
        self._button_shortcut(edit_menu, "清空可见候选的勾选", self.dust_panel.btn_uncheck_visible, "Ctrl+Shift+D")

        tool_menu = self.menuBar().addMenu("工具")
        for mode, shortcut in ((None, "V"), ("detect", "Q"), ("draw", "P"), ("check", "G"),
                               ("ignore", "X"), ("repair", "E"), ("clone", "S"), ("spot", "H")):
            button = self.edit_buttons[mode]
            self._button_shortcut(tool_menu, button.accessibleName(), button, shortcut)
        self._button_shortcut(tool_menu, "裁切与旋转…", self.btn_crop, "C")
        self._button_shortcut(tool_menu, "批量裁切与旋转…", self.btn_batch_crop, "Ctrl+Shift+C")
        self.cancel_crop_action = self._action(tool_menu, "取消当前照片裁切",
                                              lambda: self._cancel_crops((self._current_filepath,)))
        tool_menu.addSeparator()
        self._radius_actions = (
            self._action(tool_menu, "缩小工具半径", lambda: self._step_tool_radius(-1), "["),
            self._action(tool_menu, "放大工具半径", lambda: self._step_tool_radius(1), "]"),
        )

        view_menu = self.menuBar().addMenu("视图")
        self._button_shortcut(view_menu, "前后对比", self.btn_compare, "B")
        self._comparison_actions = []
        for mode, shortcut in (("wipe", "Ctrl+1"), ("side", "Ctrl+2"), ("stack", "Ctrl+3")):
            action = self._action(view_menu, self.compare_buttons[mode].text(),
                                  lambda _, value=mode: self._show_comparison_layout(value), shortcut)
            self._comparison_actions.append(action)
        self._button_shortcut(view_menu, "切换候选标记", self.btn_marks, "M")
        self._action(view_menu, "上一处候选", lambda: self._step_candidate(-1), "K")
        self._action(view_menu, "下一处候选", lambda: self._step_candidate(1), "J")
        self._action(view_menu, "原图核对候选", self._refine_current_candidate, "R")
        self._action(view_menu, "切换显尘", self.dust_panel.btn_dust_view.click, "D")
        self._button_shortcut(view_menu, "切换三联预览", self.dust_panel.btn_trial, "T")
        self._button_shortcut(view_menu, "缩小画布", self.btn_zoom_out, "-")
        self._button_shortcut(view_menu, "放大画布", self.btn_zoom_in, "=")
        self._button_shortcut(view_menu, "适应窗口", self.btn_fit, "F")
        self._button_shortcut(view_menu, "原始像素", self.btn_100, "1")
        self._action(view_menu, "搜索照片", self._focus_photo_search, QKeySequence.StandardKey.Find)
        self.focus_view_action = self._action(view_menu, "切换专注视图", self._toggle_focus_view, "Ctrl+Shift+F")
        self.theme_action = self._action(view_menu, "切换明暗主题", self._on_toggle_theme, "Ctrl+T")

        help_menu = self.menuBar().addMenu("说明")
        self._action(help_menu, "键盘快捷键…", self._show_shortcuts, "F1")
        help_menu.addSeparator()
        self._check_update_action = self._action(help_menu, "检查更新",
                                                 lambda: self.check_for_updates(manual=True))
        self._action(help_menu, "打开 GitHub", lambda: QDesktopServices.openUrl(QUrl(GITHUB_PROJECT)))
        self._shortcut_menus = (file_menu, edit_menu, tool_menu, view_menu, help_menu)

    def _show_shortcuts(self):
        dialog = QDialog(self)
        dialog.setObjectName("exportDialog")
        dialog.setWindowTitle("键盘快捷键")
        dialog.resize(500, 560)
        layout = QVBoxLayout(dialog)
        tree = QTreeWidget()
        tree.setObjectName("shortcutList")
        tree.setHeaderLabels(["操作", "快捷键"])
        tree.setColumnWidth(0, 250)
        layout.addWidget(tree)
        for menu in self._shortcut_menus:
            section = QTreeWidgetItem(tree, [menu.title()])
            for action in menu.actions():
                keys = [key.toString(QKeySequence.NativeText) for key in action.shortcuts() if not key.isEmpty()]
                if keys:
                    QTreeWidgetItem(section, [action.text().rstrip("…"), " / ".join(keys)])
        gestures = QTreeWidgetItem(tree, ["画布操作"])
        alt = "Option" if sys.platform == "darwin" else "Alt"
        for label, keys in (("临时抓手", "按住空格并拖动"), ("仿制图章取样", f"{alt} + 点击"),
                            ("追加框选勾选", "Shift + 拖框"), ("缩放", "滚轮或触控板捏合")):
            QTreeWidgetItem(gestures, [label, keys])
        tree.expandAll()
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def _setup_loading(self):
        self.overlay = QFrame(self)
        self.overlay.setObjectName("loadingOverlay")
        layout = QVBoxLayout(self.overlay)
        card = QFrame()
        card.setObjectName("loadingCard")
        card.setFixedWidth(360)
        contents = QVBoxLayout(card)
        contents.setContentsMargins(18, 15, 18, 15)
        contents.setSpacing(9)

        heading = QHBoxLayout()
        heading.setSpacing(9)
        heading.addWidget(BusySpinner(), alignment=Qt.AlignTop)
        self.overlay_label = QLabel()
        self.overlay_label.setObjectName("loadingLabel")
        self.overlay_label.setWordWrap(True)
        heading.addWidget(self.overlay_label, 1)
        contents.addLayout(heading)

        self.overlay_meter = QWidget()
        meter = QHBoxLayout(self.overlay_meter)
        meter.setContentsMargins(0, 0, 0, 0)
        meter.setSpacing(10)
        self.overlay_bar = QProgressBar()
        self.overlay_bar.setObjectName("loadingProgress")
        self.overlay_bar.setFixedHeight(6)
        self.overlay_bar.setTextVisible(False)
        meter.addWidget(self.overlay_bar, 1)
        self.overlay_percent = QLabel()
        self.overlay_percent.setObjectName("loadingPercent")
        self.overlay_percent.setFixedWidth(36)
        self.overlay_percent.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        meter.addWidget(self.overlay_percent)
        self.overlay_meter.hide()
        contents.addWidget(self.overlay_meter)

        self.overlay_cancel = QPushButton("当前文件完成后停止")
        self.overlay_cancel.setObjectName("secondaryBtn")
        self.overlay_cancel.clicked.connect(self._cancel_export)
        self.overlay_cancel.hide()
        contents.addWidget(self.overlay_cancel, alignment=Qt.AlignRight)
        layout.addWidget(card, alignment=Qt.AlignCenter)
        self.overlay.hide()

    def _connect_signals(self):
        self.theme_manager.theme_changed.connect(self._update_canvas_icons)
        for side, button in self._sidebar_buttons.items():
            button.toggled.connect(lambda visible, value=side: self._toggle_sidebar(value, visible))
            self._sidebar_header_buttons[side].clicked[bool].connect(button.setChecked)
        self.centralWidget().splitterMoved.connect(self._remember_sidebar_widths)
        self.file_panel.open_photos.connect(self._on_open_files)
        self.file_panel.open_directory.connect(self._on_open_directory)
        self.file_panel.file_selected.connect(self._on_file_selected)
        self.file_panel.directory_loaded.connect(self._on_directory_loaded)
        self.file_panel.btn_clear.clicked.connect(self._clear_photos)
        self.file_panel.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_panel.file_list.customContextMenuRequested.connect(self._photo_context_menu)
        self.dust_panel.btn_detect.clicked.connect(self._on_detect_dust)
        self.dust_panel.btn_batch_scan.clicked.connect(self._show_scan_queue)
        self.dust_panel.btn_refine.clicked.connect(self._refine_current_candidate)
        self._inspection_timer = QTimer(self)
        self._inspection_timer.setSingleShot(True)
        self._inspection_timer.setInterval(65)
        self._inspection_timer.timeout.connect(self._update_inspection_view)
        self.dust_panel.inspection_changed.connect(self._inspection_changed)
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
        self.dust_panel.preview_mode_changed.connect(self._refresh_local_preview)
        self.dust_panel.dismiss_clicked.connect(self._delete_dust_item)
        self.image_viewer.annotation_clicked.connect(self._on_annotation_clicked)
        for viewer in (self.image_viewer, self.original_viewer):
            viewer.canvas_menu_requested.connect(self._show_canvas_menu)
        self.image_viewer.region_selected.connect(self._on_region_selected)
        self.image_viewer.stroke_drawn.connect(self._on_stroke_drawn)
        self.image_viewer.sample_requested.connect(self._sample_retouch)
        self.edit_group.buttonToggled.connect(self._on_edit_mode_changed)
        self.brush_radius.valueChanged.connect(self._on_brush_radius_changed)
        self.image_viewer.view_changed.connect(self._queue_view)
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
        if kind in (QEvent.Move, QEvent.Resize) and watched in (
            self.centralWidget(), self._canvas_viewports, *self._sidebar_handles.values()
        ):
            self._position_sidebar_buttons()
        if kind == QEvent.EnabledChange:
            action = self._shortcut_buttons.get(watched)
            if action is not None:
                action.setEnabled(watched.isEnabled())
        if kind == QEvent.WindowDeactivate and watched is self:
            self._set_space_pan(False)
            self.image_viewer.set_sampling_cursor(False)
        if (kind in (QEvent.ShortcutOverride, QEvent.KeyPress)
                and isinstance(watched, (QAbstractSpinBox, QLineEdit)) and watched.window() is self
                and event.modifiers() == Qt.NoModifier and event.key() in (Qt.Key_BracketLeft, Qt.Key_BracketRight)):
            control = self._tool_radius_control()
            if control is not None and (watched is control or control.isAncestorOf(watched)):
                action = self._radius_actions[0 if event.key() == Qt.Key_BracketLeft else 1]
                if kind == QEvent.KeyPress:
                    action.trigger()
                event.accept()
                return True
        if (kind == QEvent.ShortcutOverride and isinstance(watched, (QLineEdit, QAbstractSpinBox))
                and watched.window() is self and event.text().isprintable()
                and not event.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier)):
            event.accept()
            return True
        if (kind in (QEvent.KeyPress, QEvent.KeyRelease) and event.key() == Qt.Key_Alt
                and isinstance(watched, QWidget) and watched.window() is self):
            self.image_viewer.set_sampling_cursor(kind == QEvent.KeyPress)
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

    def _load_settings(self, *, restore_workspace):
        self.theme_manager.set_theme(self.config.get("theme", "dark"))
        background = self.config.get("canvas_background")
        self._set_canvas_background(background if isinstance(background, str) and QColor(background).isValid() else None)
        self.resize(self.config.get("window.width", 1400), self.config.get("window.height", 900))
        for side, panel in self._sidebar_panels.items():
            width = self.config.get(f"sidebars.{side}.width", self._sidebar_widths[side])
            if isinstance(width, int):
                self._sidebar_widths[side] = min(panel.maximumWidth(), max(panel.minimumWidth(), width))
            self._sidebar_buttons[side].setChecked(not self.config.get(f"sidebars.{side}.collapsed", False))
        self._layout_sidebars()
        logging.info("Reading workspace state")
        try:
            manifest, snapshots = self._workspace.load()
            self._restore_workspace_states(snapshots)
            for path, view in manifest.get("views", {}).items():
                if path in self._photos:
                    self._photos[path].view = view
        except (OSError, ValueError, KeyError, TypeError, BadZipFile, sqlite3.Error) as error:
            self.lbl_status_file.setText(f"工作区恢复失败 · {error}")
            manifest = {}
        if not restore_workspace:
            self.lbl_status_file.setText("安全启动 · 已跳过自动打开照片和检查更新")
            return
        if "paths" in manifest:
            if manifest["paths"]:
                self._resume_path = manifest.get("active")
                QTimer.singleShot(0, lambda: self._reopen_workspace_photos(manifest["paths"]))
            return
        last = self.config.get("last_directory", "")
        if last and Path(last).is_dir():
            QTimer.singleShot(0, lambda: self._load_directory(last))

    def _reopen_workspace_photos(self, paths):
        logging.info("Reopening the workspace photo list (%s entries)", len(paths))
        self.file_panel.load_files(paths, keep_missing=True)

    def _restore_workspace_states(self, snapshots):
        self._restoring_workspace = True
        try:
            for path, state in snapshots.items():
                history = self._new_history()
                history.blockSignals(True)
                for record in state["commands"]:
                    if record.get("kind") == "crop":
                        command = CropCommand(self, record["after"], record["new_revision"])
                        command.before = record["before"]
                        keys = ("old_revision", "new_revision")
                    else:
                        command = EditCommand(self, record["title"], record["before"], record["after"], record["patch"])
                        keys = ("old_revision", "new_revision", "old_ignored", "new_ignored")
                    for key in keys:
                        setattr(command, key, record[key])
                    history.push(command)
                    self._next_revision = max(self._next_revision, command.new_revision + 1)
                history.setIndex(state["index"])
                history.blockSignals(False)
                repaired = state.get("repaired_marks")
                if repaired is None:
                    repaired = [(command.new_revision, command.repaired_items())
                                for index in range(history.index())
                                if (command := history.command(index)).patch is not None]
                self._next_revision = max(self._next_revision, state["revision"] + 1)
                self._photos[path] = PhotoState(
                    tuple(state["stamp"]) if state["stamp"] else None, history,
                    state["candidates"], state["ignored"], state["patches"],
                    state["revision"], state["exported_revision"], state["scan_source"],
                    tuple(state["ir_override"]) if state["ir_override"] else None,
                    state.get("view"), state.get("size"), dict(repaired), state.get("crop"),
                )
        finally:
            self._restoring_workspace = False

    @staticmethod
    def _workspace_snapshot(state):
        commands = []
        for index in range(state.history.count()):
            command = state.history.command(index)
            if isinstance(command, CropCommand):
                commands.append(dict(kind="crop", title=command.text(), before=command.before, after=command.after,
                                     old_revision=command.old_revision, new_revision=command.new_revision))
                continue
            record = {name: getattr(command, name) for name in (
                "before", "after", "patch", "old_revision", "new_revision", "old_ignored", "new_ignored"
            )}
            commands.append(dict(record, title=command.text()))
        return {
            "stamp": state.stamp, "candidates": state.candidates, "ignored": state.ignored,
            "patches": state.patches, "revision": state.revision,
            "exported_revision": state.exported_revision, "scan_source": state.scan_source,
            "ir_override": state.ir_override, "view": state.view, "size": state.size,
            "commands": commands, "index": state.history.index(),
            "repaired_marks": list(state.repaired_marks.items()),
            "crop": state.crop,
        }

    def _queue_workspace(self, path=None):
        if self._restoring_workspace:
            return
        self._workspace_manifest_dirty = True
        if path or self._current_filepath:
            self._workspace_dirty.add(path or self._current_filepath)
        self._workspace_timer.start()

    def _queue_view(self):
        if self._current_filepath and not self._restoring_workspace:
            self._workspace_manifest_dirty = True
            self._workspace_timer.start()

    def _save_workspace(self):
        self._workspace_timer.stop()
        if not self._workspace_manifest_dirty:
            return True
        self._stash_current(schedule=False)
        try:
            self._workspace.save(
                self.file_panel.photo_paths(), self._current_filepath,
                {path: self._workspace_snapshot(self._photos[path]) for path in self._workspace_dirty
                 if path in self._photos},
                {path: state.view for path, state in self._photos.items() if state.view},
            )
        except (OSError, ValueError, sqlite3.Error) as error:
            self.lbl_status_file.setText(f"工作区保存失败 · {error}")
            return False
        self._workspace_dirty.clear()
        self._workspace_manifest_dirty = False
        return True

    def _save_settings(self):
        self.config.set("theme", self.theme_manager.current_theme)
        self.config.set("window.width", self.width())
        self.config.set("window.height", self.height())
        self.config.set("sidebars", {side: {"width": self._sidebar_widths[side], "collapsed": panel.isHidden()}
                                     for side, panel in self._sidebar_panels.items()})
        if self._current_filepath:
            self.config.set("last_directory", str(Path(self._current_filepath).parent))
        self.config.save()

    def _refresh_controls(self, *_args):
        has_image = self._current_image is not None
        has_photos = bool(self.file_panel.photo_paths())
        is_tiff = self._current_filepath and Path(self._current_filepath).suffix.lower() in TIFF_EXTENSIONS
        busy = self._job is not None
        self.focus_view_action.setEnabled(not busy)
        for button in self._sidebar_buttons.values():
            button.setEnabled(not busy)
        for control in (self.btn_fit, self.btn_zoom_in, self.btn_zoom_out, self.btn_100,
                        self.edit_toolbar, self.brush_radius, self.btn_marks, self.btn_crop,
                        self.retouch_radius, self.retouch_hardness, self.retouch_opacity,
                        self.edit_buttons["clone"], self.edit_buttons["spot"],
                        self.dust_panel.btn_detect):
            control.setEnabled(has_image and not busy)
        for controls, enabled in (
            ((self.btn_compare, *self._comparison_actions), bool(self._applied_patches)),
            ((self.btn_prev, self.btn_next, self.btn_batch_crop), has_photos),
            ((self.dust_panel.btn_check_visible, self.dust_panel.btn_uncheck_visible), has_image),
            ((self.dust_panel.btn_export, self.export_action, self.ir_action), has_image),
            ((self.remove_photo_action, self.reveal_photo_action), has_image and has_photos),
            ((self.reset_photo_action,), self._can_reset_photo(self._current_filepath)),
            ((self.cancel_crop_action,), self._has_crop(self._current_filepath)),
            ((self.clear_photos_action, self.file_panel.btn_clear), has_photos),
            ((self.ir_fourth_action,), bool(is_tiff)),
            ((self.batch_export_action, self.batch_scan_action, self.dust_panel.btn_batch_scan),
             has_photos or has_image),
            ((self.retry_scan_action,), bool(self._failed_scan_paths)),
            ((self.repeat_action,), len(self.file_panel.photo_paths()) >= 3),
            ((self.btn_undo, self.undo_action), self._history.canUndo()),
            ((self.btn_redo, self.redo_action), self._history.canRedo()),
        ):
            for control in controls:
                control.setEnabled(enabled and not busy)
        for button, action in self._shortcut_buttons.items():
            action.setEnabled(button.isEnabled())
        self._refresh_radius_actions()

    def _start_job(self, label, work, completed, *, on_error=None, on_item=None,
                   cancel_label="当前文件完成后停止", show_overlay=True):
        if self._job is not None:
            return
        job = WorkspaceJob(work, self, stream=on_item is not None)
        if on_item is not None:
            job.item_ready.connect(on_item)
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
        self.overlay_meter.hide()
        self.overlay_cancel.setText(cancel_label)
        self.overlay_cancel.setVisible(self._export_cancel is not None)
        self.overlay_cancel.setEnabled(True)
        self.overlay.setGeometry(self.rect())
        if show_overlay:
            self.overlay.raise_()
            self.overlay.show()
        job.start()
        return job

    def _finish_job(self):
        self.overlay.hide()
        self._job = None
        self._export_cancel = None
        self.centralWidget().setEnabled(True)
        self._refresh_controls()

    def _update_progress(self, done, total):
        self.overlay_bar.setRange(0, total)
        self.overlay_bar.setValue(done)
        self.overlay_percent.setText(self.overlay_bar.text())
        self.overlay_meter.show()

    def _cancel_export(self):
        if self._export_cancel is not None:
            self._export_cancel.set()
            self.overlay_cancel.setEnabled(False)

    def _show_error(self, message):
        self.overlay.hide()
        self.lbl_status_file.setText("操作未完成")
        NoticeDialog(self, "操作未完成", message, warning=True).exec()

    def _on_open_files(self):
        if self._job is not None:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "添加照片", self._last_directory(),
            "照片 (" + " ".join("*" + suffix for suffix in sorted(IMAGE_EXTENSIONS)) + ")"
        )
        if paths:
            self._add_photos(paths)

    def _add_photos(self, paths):
        files = self.file_panel.load_files(paths, append=True)
        if files:
            self._on_directory_loaded(files)
        else:
            self.lbl_status_file.setText("没有新增照片")

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

    def _photo_context_menu(self, position):
        if self._job is not None:
            return
        photos = self.file_panel.file_list
        path = self.file_panel.photo_at(photos.indexAt(position).row())
        if path is None:
            return
        menu = QMenu(self)
        reveal = menu.addAction("打开所在位置")
        cancel_crop = menu.addAction("取消裁切")
        cancel_crop.setEnabled(self._has_crop(path))
        reset = menu.addAction("重置照片…")
        reset.setEnabled(self._can_reset_photo(path))
        remove = menu.addAction("从列表移除此照片")
        chosen = menu.exec(photos.viewport().mapToGlobal(position))
        if chosen is reveal:
            self._reveal_photo(path)
        elif chosen is cancel_crop:
            self._cancel_crops((path,))
        elif chosen is reset:
            self._reset_photo(path)
        elif chosen is remove:
            self._remove_photo(path)

    def _reveal_photo(self, path):
        if path is None:
            return
        path = Path(path)
        if path.is_file() and sys.platform == "darwin":
            opened, _ = QProcess.startDetached("/usr/bin/open", ["-R", str(path)])
        elif path.is_file() and sys.platform == "win32":
            opened, _ = QProcess.startDetached("explorer.exe", ["/select,", str(path)])
        else:
            opened = path.parent.is_dir() and QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
        if not opened:
            self.lbl_status_file.setText("无法打开照片所在位置")

    def _can_reset_photo(self, path):
        if path is None:
            return False
        if path == self._current_filepath:
            return bool(self._history.count() or self._scan_source or self._revision
                        or self._dust_items or self._ignored_items)
        state = self._photos.get(path)
        return bool(state and (state.history.count() or state.scan_source or state.revision
                               or state.candidates or state.ignored))

    def _reset_photo(self, path):
        if self._job is not None or not self._can_reset_photo(path):
            return
        name = Path(path).name
        answer = NoticeDialog(
            self, "重置照片",
            f"将重置 {name} 的扫描、修复和裁切。\n撤销历史会清空，原片及已导出的文件会保留。",
            confirm="重置照片",
        ).exec()
        if not answer:
            return
        active = path == self._current_filepath
        if active:
            self._stash_current(schedule=False)
        state = self._photos[path]
        state.history.clear()
        reset = replace(state, candidates=[], ignored=[], patches=[], repaired_marks={},
                        revision=0, exported_revision=0, scan_source="", crop=None)
        if active:
            scan = Scan(self._original_image, self._infrared, self._ir_source)
            self._show_photo(path, scan.image, scan=scan, stamp=state.stamp, restored=reset)
            self._stash_current(schedule=False)
        else:
            self._photos[path] = reset
            self._update_file_status(path, reset)
        self._failed_scan_paths = [failed for failed in self._failed_scan_paths if failed != path]
        self._queue_workspace(path)
        if self._save_workspace():
            next_step = "可重新扫描" if active or Path(path).is_file() else "等待原片重新连接"
            self.lbl_status_file.setText(f"已重置 {name} · {next_step}")
        self._refresh_controls()

    def _remove_photo(self, path):
        if self._job is not None or path not in self.file_panel.photo_paths():
            return
        replacement = self.file_panel.remove_file(path)
        self._failed_scan_paths = [failed for failed in self._failed_scan_paths if failed != path]
        if path == self._current_filepath:
            self._clear_current_photo()
            self.file_panel.select_file(replacement, quiet=True)
            if replacement is not None:
                self._on_file_selected(replacement)
        self._queue_workspace()
        self._refresh_controls()
        self.lbl_status_file.setText(f"已从列表移除 {Path(path).name}")

    def _clear_photos(self):
        count = len(self.file_panel.photo_paths())
        if self._job is not None or not count:
            return
        if not NoticeDialog(self, "清空照片列表",
                            f"将从列表移除全部 {count} 张照片。\n原片和修复记录会保留。",
                            confirm="清空列表").exec():
            return
        self._failed_scan_paths.clear()
        self.file_panel.load_files([])
        self.file_panel.search.clear()
        self.file_panel.filter_buttons["all"].setChecked(True)
        self.lbl_status_file.setText("已清空照片列表")

    def _clear_current_photo(self):
        self._stash_current()
        self._current_filepath = self._photo_stamp = None
        self._current_image = self._original_image = self._infrared = None
        self._ir_override = self._pinned_candidate = None
        self._scan_source = self._ir_source = ""
        self._revision = self._exported_revision = 0
        self._history = self._new_history()
        self._applied_patches = []
        self._repaired_marks = {}
        self._crop = None
        self._dust_items = []
        self._ignored_items = []
        self.btn_compare.setChecked(False)
        self._toggle_comparison(False)
        self.edit_buttons[None].setChecked(True)
        self.image_viewer.clear_image()
        self.dust_panel.set_infrared_available(False)
        self.dust_panel.clear_list()
        self.document_title.setText("胶片工作台")
        self.document_title.setToolTip("")
        self.lbl_status_info.clear()
        self.lbl_status_info.setToolTip("")
        self.lbl_status_dust.clear()
        self.lbl_zoom.clear()

    def _load_directory(self, directory):
        if self._job is None:
            logging.info("Opening a photo directory")
            try:
                self.file_panel.load_directory(directory)
            except OSError as error:
                self._show_error(str(error))

    def _confirm_quit(self):
        self._stash_current()
        if self._save_workspace():
            return True
        return bool(NoticeDialog(self, "工作区尚未保存", "本地工作区保存失败。退出后可能丢失本次编辑。",
                                 confirm="放弃本次修改", cancel="继续编辑").exec())

    def _on_directory_loaded(self, files):
        logging.info("Photo list loaded (%s entries)", len(files))
        if self._current_filepath and self._current_filepath not in self.file_panel.photo_paths():
            self._clear_current_photo()
        self._queue_workspace()
        for path in files:
            if path in self._photos:
                self._update_file_status(path, self._photos[path])
        path = self._resume_path
        if path not in files or not Path(path).is_file():
            path = next((candidate for candidate in files if Path(candidate).is_file()), None)
        self._resume_path = None
        if path:
            self.file_panel.reveal_file(path)
            self.file_panel.select_file(path, quiet=True)
            self._on_file_selected(path)
        elif files:
            self.lbl_status_file.setText("原片暂不可用 · 可右键重置已缓存的编辑")
        else:
            self.lbl_status_file.setText("文件夹中没有支持的照片")
        self._refresh_controls()

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
        logging.info("Starting photo decode (%s)", Path(path).suffix.lower())

        def failed(message):
            self.file_panel.select_file(self._current_filepath, quiet=True)
            self._show_error(f"无法导入 {Path(path).name}\n{message}")

        def loaded(result):
            logging.info("Photo decode complete")
            scan, image, stamp, restored = result
            self._show_photo(path, image, scan=scan, stamp=stamp, restored=restored)

        self._start_job("正在导入照片…", lambda _: self._read_photo(path, state), loaded, on_error=failed)

    @staticmethod
    def _read_photo(path, state):
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

    def _stash_current(self, *, schedule=True):
        if self._current_filepath is None:
            return
        state = PhotoState(
            self._photo_stamp, self._history, self._dust_items, self._ignored_items,
            self._applied_patches, self._revision, self._exported_revision,
            self._scan_source, self._ir_override,
            self.image_viewer.view_position(), self.image_viewer.get_image_size(),
            self._repaired_marks,
            self._crop,
        )
        self._photos[self._current_filepath] = state
        if schedule:
            self._queue_workspace(self._current_filepath)
        self._update_file_status(self._current_filepath, state)

    def _update_file_status(self, path, state):
        if state.scan_source or state.candidates or state.revision:
            count = len(state.candidates)
            checked = sum(item.selected for item in state.candidates)
            source = state.scan_source or ("手动编辑" if state.patches else "未扫描")
            crop = " · 已裁切" if state.crop is not None else ""
            self.file_panel.set_photo_status(
                path, f"{source} · {count} 个候选 · 已勾选 {checked}{crop}", count,
                scanned=bool(state.scan_source), repaired=bool(state.patches),
            )
        else:
            self.file_panel.set_photo_status(path, "未扫描", None)

    def _show_photo(self, path, image, *, scan=None, stamp=None, restored=None):
        if self._current_filepath is None:
            self._history.deleteLater()
        else:
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
        self._repaired_marks = restored.repaired_marks if restored else {}
        self._crop = restored.crop if restored else None
        self._scan_source = restored.scan_source if restored else ""
        self._ir_override = restored.ir_override if restored else None
        self.btn_compare.setChecked(False)
        self._toggle_comparison(False)
        self.edit_buttons[None].setChecked(True)
        self.image_viewer.load_image(image)
        if self._crop is not None:
            self.image_viewer.set_geometry(self._crop)
        self.dust_panel.set_infrared_available(self._infrared is not None)
        if self.dust_panel.inspection_layer() != "rgb" or self.dust_panel.btn_dust_view.isChecked():
            self._update_inspection_view()
        if restored and restored.view:
            self.image_viewer.restore_view(restored.view)
        self._queue_workspace(path)
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
        if self._crop is not None:
            _, _, (w, h), _ = crop_plan((h, w), self._crop)
            source += " · 已裁切"
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
            self._queue_workspace()
            self.dust_panel.set_infrared_available(True)
            self._update_inspection_view()
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
            self._add_photos(paths)
        event.acceptProposedAction()

    def _show_comparison_layout(self, mode):
        self.compare_buttons[mode].setChecked(True)
        if not self.btn_compare.isChecked():
            self.btn_compare.click()

    def _remember_sidebar_widths(self, *_args):
        left, _, right = self.centralWidget().sizes()
        for side, width in (("left", left), ("right", right)):
            if width:
                self._sidebar_widths[side] = width

    def _layout_sidebars(self):
        splitter = self.centralWidget()
        left, right = (0 if panel.isHidden() else self._sidebar_widths[side]
                       for side, panel in self._sidebar_panels.items())
        handles = int(left > 0) + int(right > 0)
        middle = max(1, splitter.width() - left - right - handles * splitter.handleWidth())
        splitter.setSizes([left, middle, right])
        self._position_sidebar_buttons()

    def _position_sidebar_buttons(self):
        splitter = self.centralWidget()
        bounds = splitter.rect().translated(splitter.mapTo(self, QPoint()))
        y = self._canvas_viewports.mapTo(self, self._canvas_viewports.rect().center()).y()
        for side, button in self._sidebar_buttons.items():
            handle = self._sidebar_handles[side]
            if self._sidebar_panels[side].isHidden():
                x = bounds.left() if side == "left" else bounds.right() - button.width() + 1
            else:
                x = handle.mapTo(self, handle.rect().center()).x() - button.width() // 2
            button.move(x, y - button.height() // 2)

    def _toggle_sidebar(self, side, visible):
        panel = self._sidebar_panels[side]
        panel.setVisible(visible)
        self._update_sidebar_button(side)
        self._layout_sidebars()

    def _update_sidebar_button(self, side):
        button = self._sidebar_buttons[side]
        header_button = self._sidebar_header_buttons[side]
        visible = button.isChecked()
        name = "照片栏" if side == "left" else "缺陷检查栏"
        label = ("收起" if visible else "展开") + name
        header_button.setChecked(visible)
        for control in (button, header_button):
            control.setToolTip(label)
            control.setAccessibleName(label)
        direction = "left" if (side == "left") == visible else "right"
        color = "#e5edf4" if self.theme_manager.is_dark else "#202a35"
        button.setIcon(tool_icon(f"chevron-{direction}", color))
        header_button.setIcon(tool_icon(f"sidebar-{side}", color))

    def _toggle_focus_view(self):
        visible = not any(button.isChecked() for button in self._sidebar_buttons.values())
        for button in self._sidebar_buttons.values():
            button.setChecked(visible)

    def _focus_photo_search(self):
        self._sidebar_buttons["left"].setChecked(True)
        self.file_panel.search.setFocus()

    def _toggle_comparison(self, enabled):
        mode = self.compare_group.checkedButton().property("mode")
        paired = enabled and mode != "wipe"
        self._compare_timer.stop()
        self._syncing_comparison = True
        if enabled:
            self.edit_buttons[None].setChecked(True)
            inspection_active = self.dust_panel.inspection_layer() != "rgb" or self.dust_panel.btn_dust_view.isChecked()
            if inspection_active:
                self.dust_panel.layer_buttons["rgb"].setChecked(True)
                self.dust_panel.btn_dust_view.setChecked(False)
                self._inspection_timer.stop()
                self._update_inspection_view()
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
            self.original_viewer.set_geometry(self._crop, fit=False)
            self.original_viewer.match_view(self.image_viewer)
        else:
            self.original_viewer.clear_image()
            if enabled:
                self.image_viewer.set_comparison(self._original_image)
        self._refresh_comparison_marks()
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

    def _inspection_changed(self):
        self.dust_panel.dust_contrast.setVisible(self.dust_panel.btn_dust_view.isChecked())
        if self._syncing_comparison:
            return
        if self.btn_compare.isChecked() and (
            self.dust_panel.inspection_layer() != "rgb" or self.dust_panel.btn_dust_view.isChecked()
        ):
            self.btn_compare.setChecked(False)
            self._toggle_comparison(False)
        self._inspection_timer.start()

    def _update_inspection_view(self):
        if self._current_image is None:
            return
        image = self._current_image
        layer = self.dust_panel.inspection_layer()
        if layer != "rgb" and self._infrared is not None:
            image = infrared_view(image, self._infrared, overlay=layer == "overlay")
        if self.dust_panel.btn_dust_view.isChecked():
            image = dust_view(image, self.dust_panel.dust_contrast.value())
        self.image_viewer.replace_image(image)

    def _toggle_marks(self, enabled):
        self.image_viewer.set_show_all(enabled)
        self.original_viewer.set_show_all(enabled)

    def _refresh_comparison_marks(self):
        if not self.btn_compare.isChecked():
            return
        marks = {item.center: item for batch in self._repaired_marks.values() for item in batch}
        marks.update((item.center, item) for item in self._dust_items)
        viewer = self.original_viewer if not self.original_viewer.isHidden() else self.image_viewer
        viewer.set_original_marks(list(marks.values()))

    def _update_canvas_icons(self, theme):
        color = "#e5edf4" if theme == ThemeManager.DARK else "#202a35"
        self.file_panel.update_icons(color, "#10251f" if theme == ThemeManager.DARK else "#ffffff")
        for side in self._sidebar_buttons:
            self._update_sidebar_button(side)
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
        if mode in RETOUCH_MODES:
            self.dust_panel.layer_buttons["rgb"].setChecked(True)
            self.dust_panel.btn_dust_view.setChecked(False)
            if self._inspection_timer.isActive():
                self._inspection_timer.stop()
                self._update_inspection_view()
        self.brush_radius.setVisible(mode == "draw")
        self.retouch_radius.setVisible(mode in RETOUCH_MODES)
        self.retouch_hardness.setVisible(mode in RETOUCH_MODES)
        self.retouch_opacity.setVisible(mode == "clone")
        self._refresh_radius_actions()
        self.lbl_status_file.setText(button.toolTip() if mode else "就绪")

    def _tool_radius_control(self):
        mode = self.edit_group.checkedButton().property("mode")
        if mode == "draw":
            return self.brush_radius
        if mode in RETOUCH_MODES:
            return self.retouch_radius
        return None

    def _refresh_radius_actions(self):
        enabled = (self._current_image is not None and self._job is None
                   and self._tool_radius_control() is not None)
        for action in self._radius_actions:
            action.setEnabled(enabled)

    def _step_tool_radius(self, direction):
        control = self._tool_radius_control()
        if control is None:
            return
        value = control.value()
        target = round(value * 1.2 ** direction)
        control.setValue(target if target != value else value + direction)

    def _on_brush_radius_changed(self, radius):
        self.image_viewer.brush_radius = radius
        self.image_viewer.viewport().update()

    def _on_stroke_drawn(self, points):
        mode = self.edit_group.checkedButton().property("mode")
        if mode == "draw":
            radius = ceil(self.brush_radius.value() / self.image_viewer.transform().m11())
            self._start_detection(brush=(points, radius))
        elif mode in RETOUCH_MODES:
            self._retouch(mode, points)

    def _on_retouch_radius_changed(self, radius):
        self.image_viewer.retouch_radius = radius
        self.image_viewer.viewport().update()

    def _sample_retouch(self, point):
        if self._current_image is None or self._job is not None:
            return
        self.image_viewer.clone_source = point
        self.image_viewer.viewport().update()
        self.lbl_status_file.setText("已设置仿制取样点 · 点击或拖动画笔复制像素")

    def _retouch(self, mode, points):
        image = self._current_image
        if image is None or self._job is not None:
            return
        source = self.image_viewer.clone_source
        if mode == "clone" and source is None:
            self.lbl_status_file.setText("请先用 Option/Alt + 点击设置仿制取样点")
            return
        source = (source.x(), source.y()) if source is not None else None
        radius = self.retouch_radius.value()
        hardness = self.retouch_hardness.value() / 100
        opacity = 1.0 if mode == "spot" else self.retouch_opacity.value() / 100
        params = self.dust_panel.get_inpaint_params()
        title = self.edit_buttons[mode].text()
        before = list(self._dust_items)

        def work(_progress):
            patch, footprint = retouch_patch(image, points, radius, hardness, mode=mode,
                                             opacity=opacity, source=source, repair_params=params)
            resolved = covered_candidates(before, footprint, patch[0], image.shape[1]) if patch is not None else set()
            return patch, [item for item in before if item.id not in resolved]

        def commit(result):
            patch, after = result
            if patch is None:
                self.lbl_status_file.setText(f"{title} · 这笔没有改变像素")
                return
            revision = self._next_revision
            self._next_revision += 1
            self._history.push(EditCommand(self, title, before, after, patch, revision))
            self.lbl_status_file.setText(f"{title} · 已应用，可撤销")

        self._start_job(f"正在应用{title}…", work, commit, show_overlay=False)

    def _on_detect_dust(self):
        self._start_detection()

    def _on_crop(self):
        if self._current_image is None or self._job is not None:
            return
        dialog = CropDialog(self._current_image, self._crop, self, background=self.config.get("canvas_background"))
        accepted = dialog.exec() == CropDialog.Accepted
        settings = dialog.settings()
        dialog.deleteLater()
        if not accepted:
            return
        if settings is None:
            self._cancel_crops((self._current_filepath,))
        elif self._record_crop(self._current_filepath, settings):
            self.lbl_status_file.setText("已更新裁切 · 可撤销")

    def _has_crop(self, path):
        if path == self._current_filepath:
            return self._crop is not None
        state = self._photos.get(path)
        return state is not None and state.crop is not None

    def _cancel_crops(self, paths):
        if self._job is not None:
            return
        count = sum(self._record_crop(path, None) for path in paths if self._has_crop(path))
        if count:
            self.lbl_status_file.setText(f"已取消 {count} 张裁切 · 可逐张撤销")

    def _record_crop(self, path, settings):
        state = None if path == self._current_filepath else self._photos[path]
        if settings == (self._crop if state is None else state.crop):
            return False
        revision = self._next_revision
        self._next_revision += 1
        command = CropCommand(self, settings, revision, state=state)
        if state is None:
            self._history.push(command)
        else:
            self._restoring_workspace = True
            try:
                state.history.push(command)
            finally:
                self._restoring_workspace = False
            state.crop, state.revision, state.view = settings, revision, None
            self._update_file_status(path, state)
            self._queue_workspace(path)
        return True

    def _on_batch_crop(self):
        paths = self.file_panel.photo_paths()
        if self._job is not None or not paths:
            return
        self._stash_current()
        dialog = BatchCropDialog(paths, {path for path in paths if self._has_crop(path)},
                                 self._read_batch_crop, self, background=self.config.get("canvas_background"))
        action = dialog.exec()
        if action != BatchCropDialog.Accepted:
            if action == BatchCropDialog.ResetCrops:
                self._cancel_crops(dialog.reset_selection())
            dialog.deleteLater()
            return
        selected = dialog.selection()
        aspect = dialog.aspect.currentData()
        known_stamps = {path: self._photos[path].stamp for path in selected if path in self._photos}
        cancel = self._export_cancel = Event()
        results = []

        def work(progress):
            suggestions = []
            for index, path in enumerate(selected):
                if cancel.is_set():
                    break
                progress(f"正在识别片框 · {Path(path).name}", index, len(selected))
                try:
                    stamp = signature(path)
                    if known_stamps.get(path) not in (None, stamp):
                        raise ValueError("原图已变化，请重新打开照片")
                    image = read_scan(path, read_ir=False).image
                    if signature(path) != stamp:
                        raise ValueError("读取期间原图发生变化")
                    detected = detect_frame(image)
                    corners = detected["corners"] if detected else None
                    crop_plan(image.shape, crop_from_edges(image.shape, corners, aspect=aspect))
                    suggestions.append(CropSuggestion(path, stamp, image.shape[1::-1], detected))
                except (OSError, ValueError, cv2.error) as error:
                    suggestions.append(CropSuggestion(path, error=str(error)))
                progress(f"已检查 {index + 1}/{len(selected)} 张片框", index + 1, len(selected))
            return suggestions

        def review_results():
            try:
                if cancel.is_set():
                    self.lbl_status_file.setText(f"已停止片框识别 · 已检查 {len(results)} 张")
                if results:
                    dialog.set_results(results)
                    if dialog.exec() == BatchCropDialog.Accepted:
                        self._apply_batch_crops(dialog.approved())
            finally:
                dialog.deleteLater()

        job = self._start_job("正在准备批量识别片框…", work, results.extend, cancel_label="停止识别")
        job.finished.connect(review_results)

    def _read_batch_crop(self, result):
        try:
            if signature(result.path) != result.stamp:
                raise ValueError("原图已变化，请重新打开后识别")
            image = read_scan(result.path, read_ir=False).image
            if signature(result.path) != result.stamp:
                raise ValueError("读取期间原图发生变化")
        except (OSError, ValueError, cv2.error) as error:
            NoticeDialog(QApplication.activeModalWidget() or self, "无法核对裁切",
                         str(error), warning=True).exec()
            return None
        return image

    def _apply_batch_crops(self, approved):
        applied, unchanged, errors = 0, 0, []
        for result, settings in approved:
            path = result.path
            try:
                if signature(path) != result.stamp:
                    raise ValueError("原图已变化，请重新打开后识别")
            except (OSError, ValueError) as error:
                errors.append(f"{Path(path).name} · {error}")
                continue
            if path != self._current_filepath and path not in self._photos:
                self._photos[path] = PhotoState(result.stamp, self._new_history(), [], [], [], 0, 0, "",
                                                None, size=result.size)
            if self._record_crop(path, settings):
                applied += 1
            else:
                unchanged += 1
        status = f"已应用 {applied} 张裁切 · 可逐张撤销"
        if unchanged:
            status += f" · {unchanged} 张设置相同"
        if errors:
            status += f" · {len(errors)} 张未应用"
            NoticeDialog(self, "批量裁切结果", status, warning=True, details="\n".join(errors)).exec()
        self.lbl_status_file.setText(status)

    def _apply_crop(self, settings, revision):
        self.btn_compare.setChecked(False)
        self._toggle_comparison(False)
        self.edit_buttons[None].setChecked(True)
        self._crop, self._revision = settings, revision
        self.image_viewer.set_geometry(settings)
        h, w = self._current_image.shape[:2]
        self._update_image_info(w, h)
        self._stash_current()
        self._refresh_controls()

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

    def _scan_settings(self):
        return {
            "params": self.dust_panel.get_detection_params(),
            "quick": self.dust_panel.combo_quality.currentData(),
            "subject_first": self.dust_panel.combo_focus.currentData(),
        }

    def _show_scan_queue(self):
        if self._job is not None:
            return
        self._stash_current()
        paths = self.file_panel.photo_paths()
        if not paths and self._current_filepath:
            paths = [self._current_filepath]
        if not paths:
            return
        dialog = ScanQueueDialog(paths, {path for path, state in self._photos.items()
                                        if state.scan_source}, self)
        if dialog.exec() == ScanQueueDialog.Accepted:
            self._start_batch_scan(dialog.selection())

    def _retry_batch_scan(self):
        self._start_batch_scan(self._failed_scan_paths)

    def _start_batch_scan(self, paths):
        if self._job is not None or not paths:
            return
        self._stash_current()
        settings = self._scan_settings()
        prefer_ir = self.dust_panel.combo_source.currentData()
        entries = [(path, self._photos.get(path)) for path in paths]
        cancel = self._export_cancel = Event()

        def work(progress, emit):
            count, errors = 0, []
            for index, (path, state) in enumerate(entries):
                if cancel.is_set():
                    break
                progress(f"正在扫描 {Path(path).name} · {index + 1}/{len(entries)}", index, len(entries))
                try:
                    scan, image, stamp, restored = self._read_photo(path, state)
                    if state is not None and state.stamp is not None and restored is None:
                        raise ValueError("原图已变化，请重新打开后扫描")

                    def report(*_):
                        if cancel.is_set():
                            raise InterruptedError

                    result = detect_scan(image, **settings, infrared=scan.infrared if prefer_ir else None,
                                         progress=report)
                    emit((path, stamp, image.shape[1::-1], result))
                    count += 1
                except InterruptedError:
                    break
                except (OSError, ValueError, cv2.error) as error:
                    errors.append((path, str(error)))
                progress(f"已处理 {index + 1}/{len(entries)} 张", index + 1, len(entries))
            return count, errors

        def scanned(payload):
            path, stamp, size, result = payload
            if path == self._current_filepath:
                self._on_detection_finished(result.items, None, result.source)
                return
            state = self._photos.get(path)
            if state is None:
                state = PhotoState(stamp, self._new_history(), [], [], [], 0, 0, "", None, size=size)
                self._photos[path] = state
            items = self._merge_scan_review(result.items, state.candidates, state.ignored)
            self._set_inactive_candidates(state, items, "批量扫描候选")
            state.scan_source = result.source
            self._update_file_status(path, state)
            self._queue_workspace(path)

        def done(payload):
            count, errors = payload
            self._failed_scan_paths = [path for path, _ in errors]
            status = f"已扫描 {count} / {len(entries)} 张"
            if cancel.is_set():
                status += " · 已停止"
            if errors:
                status += f" · {len(errors)} 张失败，可从文件菜单重试"
                self.overlay.hide()
                details = "\n".join(f"{Path(path).name} · {reason}" for path, reason in errors)
                NoticeDialog(self, "批量扫描结果", status, warning=True, details=details).exec()
            self.lbl_status_file.setText(status)

        self._start_job("正在准备批量扫描…", work, done, on_item=scanned, cancel_label="停止扫描")

    def _set_inactive_candidates(self, state, items, title, *, ignored=(), patch=None, revision=None):
        command = EditCommand(self, title, state.candidates, items, patch)
        command.old_revision = state.revision
        command.new_revision = state.revision if revision is None else revision
        command.old_ignored = state.ignored
        command.new_ignored = [*state.ignored, *ignored]
        self._restoring_workspace = True
        try:
            state.history.push(command)
        finally:
            self._restoring_workspace = False
        state.candidates = items
        state.ignored = command.new_ignored
        if patch is not None:
            positions, _, after = patch
            state.patches.append((revision, positions, after))
            state.repaired_marks[revision] = command.repaired_items()
            state.revision = revision

    def _merge_scan_review(self, items, existing, ignored):
        candidates = []
        unchecked = [item for item in existing if not item.selected]
        for item in items:
            if any(self._same_defect(item, old) for old in ignored):
                continue
            selected = not any(self._same_defect(item, old) for old in unchecked)
            candidates.append(replace(item, selected=selected))
        return candidates

    def _start_detection(self, region=None, *, brush=None):
        if self._current_image is None or self._job is not None:
            return
        image = self._current_image
        settings = self._scan_settings()
        infrared = self._infrared if self.dust_panel.combo_source.currentData() else None

        def scan(progress):
            def report(source, done, total):
                progress(f"正在扫描 {source} · {done} / {total}", done, total)

            if brush is not None:
                return detect_brush(image, settings["params"], *brush, infrared=infrared, progress=report)
            return detect_scan(
                image, **settings, infrared=infrared, region=region, progress=report,
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
            merged = self._merge_scan_review(items, self._dust_items, ())
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
        self._refresh_comparison_marks()
        self.dust_panel.update_dust_list(self._dust_items)
        self.lbl_status_dust.setText(f"{len(self._dust_items)} 个候选")

    def _refresh_marks(self, *_args):
        self.image_viewer.viewport().update()
        self._refresh_comparison_marks()
        self._stash_current()

    def _on_dust_hovered(self, dust_id):
        if dust_id == -1:
            dust_id = self._pinned_candidate
        item = next((item for item in self._dust_items if item.id == dust_id), None)
        self.image_viewer.highlight_dust(dust_id)
        self.dust_panel.show_candidate(self._current_image, item, original=self._original_image)

    def _refresh_local_preview(self):
        hovered = self.dust_panel.dust_list._hover_id
        self._on_dust_hovered(hovered)

    def _step_candidate(self, direction):
        if self._job is not None:
            return
        dust_id = self.dust_panel.step_candidate(direction)
        if dust_id is not None:
            self._pin_candidate(dust_id)
            self.image_viewer.highlight_dust(dust_id, locate=True, min_zoom=0)

    def _refine_current_candidate(self):
        if self._job is not None or self._current_image is None:
            return
        row = self.dust_panel.dust_list.currentItem()
        dust_id = row.data(Qt.UserRole) if row is not None else self._pinned_candidate
        item = next((dust for dust in self._dust_items if dust.id == dust_id), None)
        if item is None or not item.preview:
            return
        image = self._current_image
        params = self.dust_panel.get_detection_params()

        def work(progress):
            return refine_candidate(image, item, params, progress=lambda stage, done, total:
                                    progress(f"原图核对 · {stage}", done, total))

        def done(candidate):
            if candidate is None:
                self.lbl_status_file.setText("原图上未找到匹配轮廓 · 保留当前候选")
                return
            refined = replace(candidate, id=item.id, selected=item.selected)
            updated = [refined if dust.id == item.id else dust for dust in self._dust_items]
            self._history.push(EditCommand(self, "原图核对", self._dust_items, updated))
            self.dust_panel.select_candidate(item.id)
            self._pin_candidate(item.id)
            self.lbl_status_file.setText("已按原图像素更新候选轮廓")

        self._start_job("正在核对原图轮廓…", work, done)

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

    def _set_canvas_background(self, color):
        for viewer in (self.image_viewer, self.original_viewer):
            viewer.set_canvas_background(color)
        self.config.set("canvas_background", color)

    def _show_canvas_menu(self, dust_id, position):
        menu = QMenu(self)
        repair = dismiss = None
        if dust_id is not None:
            repair = menu.addAction("修复此项")
            dismiss = menu.addAction("忽略此项")
            menu.addSeparator()
        background = menu.addMenu("画布背景")
        current = self.config.get("canvas_background")
        presets = {}
        for label, color in (("默认", None), ("深灰", "#333333"), ("中灰", "#808080"),
                             ("黑色", "#000000"), ("白色", "#ffffff")):
            option = background.addAction(label)
            option.setCheckable(True)
            option.setChecked(current == color)
            presets[option] = color
        background.addSeparator()
        custom = background.addAction("自定义颜色…")
        action = menu.exec(position)
        if action is None:
            return
        if action in presets:
            self._set_canvas_background(presets[action])
            self.config.save()
        elif action is custom:
            color = QColorDialog.getColor(QColor(current or DEFAULT_CANVAS_BACKGROUND), self, "画布背景色")
            if color.isValid():
                self._set_canvas_background(color.name())
                self.config.save()
        elif action is repair:
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

    def _show_repeated(self):
        if self._job is not None:
            return
        self._stash_current()
        entries = []
        for path in self.file_panel.photo_paths():
            state = self._photos.get(path)
            if state is None or not state.size:
                continue
            try:
                if signature(path) != state.stamp:
                    continue
            except OSError:
                continue
            entries.append((path, state.size, state.candidates))
        groups = repeated_defects(entries)
        if not groups:
            NoticeDialog(self, "同卷重复候选", "没有发现同位置的重复候选。可先批量扫描同一卷。").exec()
            return
        dialog = RepeatDialog(groups, self)
        if dialog.exec() != RepeatDialog.Accepted:
            return
        operation, group = dialog.selection()
        if operation == "ignore":
            self._ignore_repeat_group(group)
        else:
            self._repair_repeat_group(group)

    def _ignore_repeat_group(self, group):
        for path, item in group:
            if path == self._current_filepath:
                self._ignore([item])
                continue
            state = self._photos[path]
            remaining = [dust for dust in state.candidates if dust.id != item.id]
            self._set_inactive_candidates(state, remaining, "忽略重复候选", ignored=(item,))
            self._update_file_status(path, state)
            self._queue_workspace(path)
        self.lbl_status_file.setText(f"已忽略 {len(group)} 张照片中的重复候选 · 可逐张撤销")

    @staticmethod
    def _repair_patch(image, items, params):
        remover = DustRemover(**params)
        mask = remover.create_mask(image.shape, items, selected_only=False)
        repaired = remover.apply_mask(image, mask)
        positions = np.flatnonzero(mask.ravel())
        return positions, image.reshape(-1, 3)[positions], repaired.reshape(-1, 3)[positions]

    def _repair_repeat_group(self, group):
        params = self.dust_panel.get_inpaint_params()
        entries = [(path, item, self._photos[path]) for path, item in group]
        cancel = self._export_cancel = Event()

        def work(progress):
            result, errors = [], []
            for index, (path, item, state) in enumerate(entries):
                if cancel.is_set():
                    break
                progress(f"修复 {Path(path).name}", index, len(entries))
                try:
                    _, image, _, restored = self._read_photo(path, state)
                    if restored is None:
                        raise ValueError("原图已变化，请重新核对")
                    result.append((path, item, self._repair_patch(image, [item], params)))
                except (OSError, ValueError, cv2.error) as error:
                    errors.append(f"{Path(path).name} · {error}")
                progress(f"已处理 {index + 1}/{len(entries)} 张", index + 1, len(entries))
            return result, errors

        def done(payload):
            result, errors = payload
            title = "修复重复候选"
            for path, item, patch in result:
                state = self._photos[path]
                items = [dust for dust in state.candidates if dust.id != item.id]
                revision = self._next_revision
                self._next_revision += 1
                if path == self._current_filepath:
                    self._history.push(EditCommand(self, title, self._dust_items, items, patch, revision))
                else:
                    self._set_inactive_candidates(state, items, title, patch=patch, revision=revision)
                    self._update_file_status(path, state)
                    self._queue_workspace(path)
            status = f"重复候选已修复 {len(result)} / {len(group)} 张 · 可逐张撤销"
            self.lbl_status_file.setText(status)
            if errors:
                self.overlay.hide()
                NoticeDialog(self, title, status, warning=True, details="\n".join(errors)).exec()

        self._start_job("正在修复重复候选…", work, done)

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
        params = self.dust_panel.get_inpaint_params()

        def commit(patch):
            revision = self._next_revision
            self._next_revision += 1
            self._history.push(EditCommand(self, f"修复 {len(items)} 处缺陷",
                                           before, after, patch, revision))
            self.lbl_status_file.setText(f"已修复 {len(items)} 处，可前后对比")

        self._start_job("正在修复选中的缺陷…", lambda _: self._repair_patch(image, items, params), commit)

    def _apply_edit(self, items, ignored, revision, patch, *, use_after, repaired=()):
        if patch is not None:
            positions, before, after = patch
            if use_after:
                self._applied_patches.append((revision, positions, after))
                self._repaired_marks[revision] = repaired
            else:
                self._applied_patches.pop()
                self._repaired_marks.pop(self._revision, None)
            if revision == 0:
                self._current_image = self._original_image
            else:
                if self._current_image is self._original_image:
                    self._current_image = self._original_image.copy()
                self._current_image.reshape(-1, 3)[positions] = after if use_after else before
            self._update_inspection_view()
        self._revision = revision
        if not self._applied_patches and self.btn_compare.isChecked():
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
        rows = []
        for path in self.file_panel.photo_paths():
            state = self._photos.get(path)
            checked = sum(item.selected for item in state.candidates) if state else 0
            edited = bool(state and state.revision)
            has_work = bool(checked or edited)
            status = (f"{checked} 项已勾选" if state and (state.scan_source or state.candidates)
                      else "未扫描")
            if state and state.patches:
                status += " · 已有修复"
            if state and state.crop is not None:
                status += " · 已裁切"
            rows.append((path, has_work, status))
        if not rows:
            return
        if batch and not any(row[1] for row in rows):
            rows = [(path, path == self._current_filepath, status) for path, _, status in rows]
        default = self.config.get("output_dir", "") or self._last_directory()
        dialog = ExportDialog(rows, self._current_filepath, Path(default).resolve(), batch, self,
                              icons=self.file_panel.photo_icons())
        if dialog.exec() == ExportDialog.Accepted:
            self._start_export(*dialog.selection())

    def _start_export(self, paths, directory, extension, mode, include_review=False):
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
                tuple(state.candidates) if state and include_review else (),
                state.crop if state else None,
            ))
        cancel = self._export_cancel = Event()
        repair_params = self.dust_panel.get_inpaint_params()
        self.config.set("output_dir", str(directory))

        def save(progress):
            review_enabled = include_review and mode != "original"
            if review_enabled:
                for entry in entries:
                    if entry.stamp is None:
                        try:
                            entry.stamp = signature(entry.path)
                        except OSError:
                            pass  # Export reports unreadable files independently.
            saved, errors = export_batch(entries, directory, extension, mode, repair_params,
                                         progress, cancel.is_set)
            report = review_error = None
            if review_enabled and saved and not cancel.is_set():
                try:
                    report = create_review_report(entries, saved, directory, mode, progress, cancel.is_set)
                except OSError as error:
                    review_error = str(error)
            return saved, errors, report, review_error

        def finished(result):
            saved, errors, report, review_error = result
            if mode != "original":
                for path, revision, _target in saved:
                    state = self._photos.get(str(path))
                    if state:
                        state.exported_revision = revision
                        self._queue_workspace(str(path))
                    if str(path) == self._current_filepath:
                        self._exported_revision = revision
            status = f"已导出 {len(saved)} / {len(entries)} 张"
            if errors:
                status += f" · 失败 {len(errors)} 张"
            if review_error:
                status += " · 对照页生成失败"
            elif report:
                status += " · 已生成对照页"
            if cancel.is_set():
                status += " · 已停止"
            if saved:
                self._last_export_folder = Path(directory)
                self.btn_export_folder.show()
                self.lbl_status_file.setToolTip(str(directory))
                self._last_review_report = report
                self.btn_review_report.setVisible(report is not None)
            self.lbl_status_file.setText(status)
            self.overlay.hide()
            if errors or review_error:
                details = [f"{path.name} · {reason}" for path, reason in errors]
                if review_error:
                    details.append(f"对照页 · {review_error}")
                NoticeDialog(self, "导出结果", status, warning=True, details="\n".join(details)).exec()

        self._start_job("正在准备原图导出…", save, finished)

    def _open_export_folder(self):
        self._open_export_path(self._last_export_folder)

    def _open_review_report(self):
        self._open_export_path(self._last_review_report)

    def _open_export_path(self, path):
        if path is None or not path.exists() or not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve()))):
            self.lbl_status_file.setText("无法打开导出位置")

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
        self._update_checker.stop()
        self._save_settings()
        event.accept()

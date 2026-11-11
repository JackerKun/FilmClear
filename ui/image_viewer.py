"""Photo canvas with native-coordinate annotations and split comparison."""

import math

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QTransform
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QRubberBand


from core.scan_io import display_pixels
from core.frame_crop import crop_plan


RETOUCH_MODES = ("clone", "spot")
STROKE_MODES = ("draw", *RETOUCH_MODES)
DEFAULT_CANVAS_BACKGROUND = "#222a33"


def photo_pixmap(image):
    pixels = np.ascontiguousarray(display_pixels(image))
    h, w = pixels.shape[:2]
    format_ = QImage.Format_Grayscale8 if pixels.ndim == 2 else QImage.Format_BGR888
    return QPixmap.fromImage(QImage(pixels.data, w, h, pixels.strides[0], format_))


def defect_path(dust):
    path = QPainterPath()
    if dust.contour is None or len(dust.contour) < 3:
        path.addRect(QRectF(*dust.bbox))
        return path
    for contour in (dust.contour, *dust.holes):
        points = contour.reshape(-1, 2)
        path.moveTo(float(points[0, 0]), float(points[0, 1]))
        for x, y in points[1:]:
            path.lineTo(float(x), float(y))
        path.closeSubpath()
    return path


class ClippedPhoto(QGraphicsPixmapItem):
    crop_path = None

    def shape(self):
        return self.crop_path if self.crop_path is not None else super().shape()


class ImageViewer(QGraphicsView):
    annotation_clicked = Signal(int)
    canvas_menu_requested = Signal(object, object)
    region_selected = Signal(int, int, int, int, bool)
    stroke_drawn = Signal(object)
    sample_requested = Signal(object)
    zoom_changed = Signal(int)
    view_changed = Signal()
    split_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._photo = ClippedPhoto()
        self.scene().addItem(self._photo)
        self._geometry = None
        self._photo.setTransformationMode(Qt.SmoothTransformation)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMouseTracking(True)

        self._items = {}
        self._focus_id = None
        self._focus_path = None
        self._show_all = False
        self._fit = True
        self._before = QPixmap()
        self._before_marks = []
        self._split = 50
        self._dragging_split = False
        self._selection_mode = None
        self.brush_radius = 14
        self.retouch_radius = 14
        self.clone_source = None
        self._pointer_pos = None
        self._sampling = False
        self.space_pan = False
        self._space_drag = None
        self.comparison_label = ""
        self._stroke = []
        self._press = None
        self._box = QRubberBand(QRubberBand.Rectangle, self.viewport())
        self.horizontalScrollBar().valueChanged.connect(self.view_changed)
        self.verticalScrollBar().valueChanged.connect(self.view_changed)
        self.zoom_changed.connect(self.view_changed)
        self.view_changed.connect(self._update_tool_cursor)
        self._update_tool_cursor()

    def load_image(self, image):
        self.load_pixmap(photo_pixmap(image))

    def load_pixmap(self, pixmap):
        self._geometry = None
        self.clone_source = None
        self._cancel_selection()
        self.clear_comparison()
        self.update_annotations([])
        self._replace_pixmap(pixmap)
        self.fit_to_window()

    def replace_image(self, image):
        self._replace_pixmap(photo_pixmap(image))

    def _replace_pixmap(self, pixmap):
        self._photo.setPixmap(pixmap)
        self.set_geometry(self._geometry, fit=False)
        self._update_tool_cursor()
        self.viewport().update()

    def clear_image(self):
        self._geometry = None
        self._photo.crop_path = None
        self._photo.setTransform(QTransform())
        self.clone_source = None
        self._cancel_selection()
        self.clear_comparison()
        self.update_annotations([])
        self._photo.setPixmap(QPixmap())
        self.setSceneRect(QRectF())
        self._update_tool_cursor()

    def get_image_size(self):
        bounds = self._photo.boundingRect()
        return int(bounds.width()), int(bounds.height())

    def set_canvas_background(self, color):
        self.setBackgroundBrush(QBrush(QColor(color or DEFAULT_CANVAS_BACKGROUND)))

    def set_geometry(self, settings, *, fit=True):
        self._geometry = settings
        bounds = self._photo.pixmap().rect()
        if settings is None:
            self._photo.crop_path = None
            self._photo.setFlag(QGraphicsItem.ItemClipsToShape, False)
            self._photo.setTransform(QTransform())
            self.setSceneRect(QRectF(bounds))
        else:
            points, matrix, (width, height), _ = crop_plan((bounds.height(), bounds.width()), settings)
            path = QPainterPath()
            path.addPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in points]))
            path.closeSubpath()
            self._photo.crop_path = path
            self._photo.setFlag(QGraphicsItem.ItemClipsToShape, True)
            self._photo.setTransform(QTransform(*matrix.T.ravel()))
            self.setSceneRect(0, 0, width, height)
        if fit:
            self.fit_to_window()
        self.viewport().update()

    def image_point(self, pos):
        return self._photo.mapFromScene(self.mapToScene(pos))

    def image_position(self, point):
        return self.mapFromScene(self._photo.mapToScene(QPointF(*point)))

    def update_annotations(self, dust_items):
        self._items = {item.id: item for item in dust_items}
        self.reset_highlight()

    def set_show_all(self, visible):
        self._show_all = visible
        self.viewport().update()

    def highlight_dust(self, dust_id, *, locate=False, min_zoom=0.65):
        dust = self._items.get(dust_id)
        if dust is None:
            self.reset_highlight()
            return
        if self._focus_id != dust_id:
            self._focus_id = dust_id
            self._focus_path = defect_path(dust)
        if locate:
            if self.transform().m11() < min_zoom:
                self._zoom_to(min_zoom)
            self.centerOn(self._photo.mapToScene(QPointF(*dust.center)))
        self.viewport().update()

    def reset_highlight(self):
        self._focus_id = None
        self._focus_path = None
        self.viewport().update()

    def set_comparison(self, original):
        self._before = photo_pixmap(original)
        self._update_tool_cursor()
        self.viewport().update()

    def clear_comparison(self):
        self._before = QPixmap()
        self._before_marks = []
        self._dragging_split = False
        self._update_tool_cursor()
        self.viewport().update()

    def set_split(self, position):
        self._split = position
        self._update_tool_cursor()
        self.viewport().update()

    def set_original_marks(self, items):
        self._before_marks = items
        self.viewport().update()

    def drawForeground(self, painter, rect):
        if self._photo.pixmap().isNull():
            painter.save()
            painter.resetTransform()
            light = self.backgroundBrush().color().lightness() > 150
            painter.setPen(QColor("#36434e" if light else "#b4c3d0"))
            painter.drawText(self.viewport().rect(), Qt.AlignCenter,
                             "导入胶片扫描照片\n在左侧添加照片或打开文件夹")
            painter.restore()
            return

        painter.save()
        painter.setWorldTransform(self._photo.sceneTransform(), True)
        if self._photo.crop_path is not None:
            painter.setClipPath(self._photo.crop_path, Qt.IntersectClip)
        rect = self._photo.mapFromScene(rect).boundingRect()
        if not self._before.isNull():
            self._draw_comparison(painter, rect)
            painter.restore()
            return
        if self.comparison_label:
            self._draw_marks(painter, rect, self._before_marks)
            painter.restore()
            self._draw_badge(painter)
            return

        self._draw_marks(painter, rect, self._items.values(), skip_id=self._focus_id)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        if self._focus_path is not None:
            for color, width in (("#17232d", 4), ("#64ebd2", 2)):
                pen = QPen(QColor(color))
                pen.setWidthF(width)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.drawPath(self._focus_path)
            focus = self._items[self._focus_id]
            point = QPointF(*focus.center)
            radius = 9 / self.transform().m11()
            if max(focus.width, focus.height) * self.transform().m11() < 18:
                painter.drawEllipse(point, radius, radius)
        if self._stroke:
            search = self._selection_mode == "draw"
            radius = self.brush_radius if search else self.retouch_radius
            pen = QPen(QColor(100, 235, 210, 85), radius * 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            pen.setCosmetic(search)
            painter.setPen(pen)
            path = QPainterPath(self._stroke[0])
            for point in self._stroke[1:]:
                path.lineTo(point)
            painter.drawPath(path)
            if len(self._stroke) == 1:
                painter.drawPoint(self._stroke[0])
        if self._selection_mode in STROKE_MODES and not self.space_pan:
            self._draw_brush_cursor(painter)
        painter.restore()
        painter.restore()

    def _draw_brush_cursor(self, painter):
        scale = self.transform().m11()
        radius = self.brush_radius / scale if self._selection_mode == "draw" else self.retouch_radius
        point = self._brush_position() if not self._is_sampling() else None
        source = self.clone_source if self._selection_mode == "clone" else None
        if source is not None and self._stroke:
            source = source + self._stroke[-1] - self._stroke[0]
        size = 6 / scale
        for color, width in (("#17232d", 3), ("#eef5ff", 1)):
            pen = QPen(QColor(color))
            pen.setCosmetic(True)
            pen.setWidth(width)
            painter.setPen(pen)
            if point is not None:
                painter.drawEllipse(point, radius, radius)
                if radius * scale < 4:
                    painter.drawPoint(point)
            if source is not None:
                painter.drawLine(source - QPointF(size, 0), source + QPointF(size, 0))
                painter.drawLine(source - QPointF(0, size), source + QPointF(0, size))

    def _draw_marks(self, painter, rect, items, *, skip_id=None):
        if not self._show_all:
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        for item in items:
            point = QPointF(*item.center)
            if item.id == skip_id or not rect.contains(point):
                continue
            pen = QPen(QColor("#65dfcc") if item.selected else QColor("#e6bb72"))
            pen.setCosmetic(True)
            pen.setWidthF(1.25)
            painter.setPen(pen)
            radius = (4 if item.selected else 3) / self.transform().m11()
            painter.drawEllipse(point, radius, radius)
        painter.restore()

    def _draw_badge(self, painter):
        painter.save()
        painter.resetTransform()
        painter.setRenderHint(QPainter.Antialiasing)
        width = painter.fontMetrics().horizontalAdvance(self.comparison_label) + 24
        rect = QRectF(12, 12, width, 27)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(18, 26, 36, 220))
        painter.drawRoundedRect(rect, 6, 6)
        painter.setPen(QColor("#e5eef6"))
        painter.drawText(rect, Qt.AlignCenter, self.comparison_label)
        painter.restore()

    def _draw_comparison(self, painter, rect):
        view = self.viewport().rect()
        split = round(view.width() * self._split / 100)
        painter.save()
        clip = QPainterPath()
        clip.addPolygon(self._photo.mapFromScene(self.mapToScene(QRect(0, 0, split, view.height()))))
        painter.setClipPath(clip, Qt.IntersectClip)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(0, 0, self._before)
        self._draw_marks(painter, rect, self._before_marks)
        painter.restore()

        painter.save()
        painter.resetTransform()
        painter.setClipping(False)
        painter.setPen(QPen(QColor("#dce8f3"), 1))
        painter.drawLine(split, 0, split, view.height())
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#dce8f3"))
        painter.drawRoundedRect(QRectF(split - 4, view.height() / 2 - 15, 8, 30), 4, 4)
        painter.restore()

    def view_position(self):
        point = self.mapToScene(self.viewport().rect().center())
        return self._fit, self.transform().m11(), point.x(), point.y()

    def restore_view(self, position):
        fit, scale, x, y = position
        if fit:
            self.fit_to_window()
            return
        self._zoom_to(scale)
        self._fit = False
        self.centerOn(x, y)

    def get_zoom_percent(self):
        return round(self.transform().m11() * 100)

    def match_view(self, other):
        scale = other.transform().m11()
        if abs(self.transform().m11() - scale) > 1e-6:
            self.setTransform(QTransform.fromScale(scale, scale))
        self.centerOn(other.mapToScene(other.viewport().rect().center()))
        self._fit = other._fit

    def fit_to_window(self):
        if self.sceneRect().isEmpty():
            return
        self._fit = True
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self.zoom_changed.emit(self.get_zoom_percent())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit:
            self.fit_to_window()

    def _zoom_to(self, target, anchor=None):
        if self.sceneRect().isEmpty():
            return
        old = self.transform().m11()
        target = min(12, max(0.015, target))
        if abs(target - old) < 1e-6:
            return
        self._fit = False
        anchor = QPointF(self.viewport().rect().center() if anchor is None else anchor)
        inverse, _ = self.viewportTransform().inverted()
        before = inverse.map(anchor)
        self.scale(target / old, target / old)
        delta = self.viewportTransform().map(before) - anchor
        horizontal, vertical = self.horizontalScrollBar(), self.verticalScrollBar()
        horizontal.setValue(horizontal.value() + round(delta.x()))
        vertical.setValue(vertical.value() + round(delta.y()))
        self.zoom_changed.emit(self.get_zoom_percent())

    def zoom_in(self):
        self._zoom_to(self.transform().m11() * 1.2)

    def zoom_out(self):
        self._zoom_to(self.transform().m11() / 1.2)

    def zoom_100(self):
        self._zoom_to(1.0)

    def wheelEvent(self, event):
        pixels, angle = event.pixelDelta(), event.angleDelta()
        if event.phase() != Qt.ScrollMomentum:
            if not pixels.isNull():
                if abs(pixels.y()) >= abs(pixels.x()):
                    step = max(-0.09, min(0.09, pixels.y() * 0.002))
                    self._zoom_to(self.transform().m11() * math.exp(step), event.position().toPoint())
                else:
                    scroll = self.horizontalScrollBar()
                    scroll.setValue(scroll.value() - pixels.x())
            elif angle.y():
                step = max(-2, min(2, angle.y() / 120))
                base = 1.14 if event.phase() == Qt.NoScrollPhase else 1.035
                self._zoom_to(self.transform().m11() * (base ** step), event.position().toPoint())
        event.accept()

    def viewportEvent(self, event):
        kind = event.type()
        if kind == QEvent.Enter:
            self._track_pointer(event.position().toPoint())
        elif kind == QEvent.Leave:
            self._track_pointer(None)
        elif kind == QEvent.NativeGesture and event.gestureType() == Qt.ZoomNativeGesture:
            factor = math.exp(max(-0.16, min(0.16, event.value())))
            self._zoom_to(self.transform().m11() * factor, event.position().toPoint())
            event.accept()
            return True
        return super().viewportEvent(event)

    def set_selection_mode(self, mode):
        self._selection_mode = mode
        self.setDragMode(QGraphicsView.NoDrag if mode else QGraphicsView.ScrollHandDrag)
        self._cancel_selection()
        self._update_tool_cursor()

    def _cancel_selection(self):
        self._press = None
        self._stroke.clear()
        self._box.hide()
        self.viewport().update()

    def _update_tool_cursor(self):
        if self.space_pan:
            cursor = Qt.ClosedHandCursor if self._space_drag is not None else Qt.OpenHandCursor
        elif self._photo.pixmap().isNull():
            cursor = Qt.ArrowCursor
        elif self._dragging_split or (self._press is None and self._on_split(self._pointer_pos)):
            cursor = Qt.SizeHorCursor
        elif self._selection_mode in STROKE_MODES:
            cursor = Qt.ArrowCursor
            if self._brush_position() is not None:
                cursor = Qt.CrossCursor if self._is_sampling() else Qt.BlankCursor
        elif self._selection_mode:
            cursor = Qt.CrossCursor
        else:
            cursor = Qt.ClosedHandCursor if self._press is not None else Qt.ArrowCursor
        if self.viewport().cursor().shape() != cursor:
            self.viewport().setCursor(cursor)

    def _on_split(self, pos):
        split = round(self.viewport().width() * self._split / 100)
        return pos is not None and not self._before.isNull() and abs(pos.x() - split) <= 12

    def _brush_position(self):
        if self._pointer_pos is not None and self.viewport().rect().contains(self._pointer_pos):
            point = self.image_point(self._pointer_pos)
            if self._photo.shape().contains(point):
                return point
        return None

    def _is_sampling(self):
        return self._selection_mode == "clone" and self._sampling and not self._stroke

    def set_sampling_cursor(self, active):
        if self._sampling != active:
            self._sampling = active
            self._update_tool_cursor()
            self.viewport().update()

    def _track_pointer(self, pos):
        self._pointer_pos = pos
        self._update_tool_cursor()
        if self._selection_mode in STROKE_MODES:
            self.viewport().update()

    def set_space_pan(self, active):
        if self.space_pan == active:
            return
        self.space_pan = active
        if active:
            self._dragging_split = False
            self._cancel_selection()
        self._update_tool_cursor()
        self.viewport().update()

    def _stroke_point(self, pos):
        point = self.image_point(pos)
        bounds = self._photo.boundingRect()
        return QPointF(min(max(0, point.x()), bounds.width() - 1),
                       min(max(0, point.y()), bounds.height() - 1))

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        self.set_sampling_cursor(bool(event.modifiers() & Qt.AltModifier))
        self._track_pointer(pos)
        if event.button() == Qt.LeftButton:
            self._space_drag = None
            if self.space_pan:
                self._space_drag = pos
                self._update_tool_cursor()
                event.accept()
                return
            self._press = pos
            if self._on_split(pos):
                self._dragging_split = True
                self._update_tool_cursor()
                return
            if self._selection_mode in STROKE_MODES:
                if self._photo.shape().contains(self.image_point(pos)):
                    point = self._stroke_point(pos)
                    if (self._selection_mode == "clone"
                            and event.modifiers() & Qt.AltModifier):
                        self._press = None
                        self.sample_requested.emit(point)
                        return
                    self._stroke = [point]
                    self.viewport().update()
                return
            if self._selection_mode:
                self._box.setGeometry(QRect(pos, pos))
                self._box.show()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        self.set_sampling_cursor(bool(event.modifiers() & Qt.AltModifier))
        self._track_pointer(pos)
        if self._space_drag is not None:
            if self.space_pan:
                delta = self._space_drag - pos
                self._space_drag = pos
                for scrollbar, step in ((self.horizontalScrollBar(), delta.x()),
                                        (self.verticalScrollBar(), delta.y())):
                    scrollbar.setValue(scrollbar.value() + step)
            event.accept()
            return
        if self._dragging_split:
            position = max(0, min(100, round(100 * pos.x() / self.viewport().width())))
            self.set_split(position)
            self.split_changed.emit(position)
            return
        if self._selection_mode and self._press is not None:
            if self._selection_mode in STROKE_MODES:
                if self._stroke:
                    self._stroke.append(self._stroke_point(pos))
                    self.viewport().update()
            else:
                self._box.setGeometry(QRect(self._press, pos).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        pos = event.position().toPoint()
        self.set_sampling_cursor(bool(event.modifiers() & Qt.AltModifier))
        self._track_pointer(pos)
        if event.button() == Qt.LeftButton:
            if self._space_drag is not None:
                self._space_drag = None
                self._update_tool_cursor()
                event.accept()
                return
            start, self._press = self._press, None
            if self._dragging_split:
                self._dragging_split = False
                self._update_tool_cursor()
                return
            if self._selection_mode in STROKE_MODES:
                if self._stroke:
                    self._stroke.append(self._stroke_point(pos))
                    points, self._stroke = self._stroke, []
                    self._update_tool_cursor()
                    self.stroke_drawn.emit([(p.x(), p.y()) for p in points])
                    self.viewport().update()
                return
            if self._selection_mode and start is not None:
                self._box.hide()
                box = self._photo.mapFromScene(self.mapToScene(QRect(start, pos).normalized())).boundingRect()
                box = box.intersected(self._photo.boundingRect()).toAlignedRect()
                if box.width() >= 3 and box.height() >= 3:
                    self.region_selected.emit(box.x(), box.y(),
                                              box.x() + box.width(), box.y() + box.height(),
                                              bool(event.modifiers() & Qt.ShiftModifier))
                return
            if start is not None and (pos - start).manhattanLength() <= 4:
                dust_id = self._annotation_at(pos)
                if dust_id is not None:
                    self.annotation_clicked.emit(dust_id)
        super().mouseReleaseEvent(event)
        self._update_tool_cursor()

    def contextMenuEvent(self, event):
        dust_id = None if self._selection_mode in RETOUCH_MODES else self._annotation_at(event.pos())
        self.canvas_menu_requested.emit(dust_id, event.globalPos())
        event.accept()

    def _annotation_at(self, pos):
        if not self._before.isNull() or self.comparison_label:
            return None
        best = None
        best_distance = 10
        for item in self._items.values():
            if self._show_all or item.id == self._focus_id:
                distance = (self.image_position(item.center) - pos).manhattanLength()
                if distance < best_distance:
                    best, best_distance = item.id, distance
        return best

"""Photo canvas with native-coordinate annotations and split comparison."""

import math

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QRubberBand


from core.scan_io import display_pixels


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


class ImageViewer(QGraphicsView):
    annotation_clicked = Signal(int)
    annotation_right_clicked = Signal(int, object)
    region_selected = Signal(int, int, int, int, bool)
    stroke_drawn = Signal(object)
    zoom_changed = Signal(int)
    view_changed = Signal()
    split_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._photo = self.scene().addPixmap(QPixmap())
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
        self._split = 50
        self._dragging_split = False
        self._selection_mode = None
        self.brush_radius = 14
        self.space_pan = False
        self._space_drag = None
        self.comparison_label = ""
        self._stroke = []
        self._press = None
        self._box = QRubberBand(QRubberBand.Rectangle, self.viewport())
        self.horizontalScrollBar().valueChanged.connect(self.view_changed)
        self.verticalScrollBar().valueChanged.connect(self.view_changed)
        self.zoom_changed.connect(self.view_changed)

    def load_image(self, image):
        self.clear_comparison()
        self.update_annotations([])
        self.replace_image(image)
        self.fit_to_window()

    def replace_image(self, image):
        self._photo.setPixmap(photo_pixmap(image))
        self.setSceneRect(QRectF(self._photo.pixmap().rect()))
        self.viewport().update()

    def clear_image(self):
        self._photo.setPixmap(QPixmap())
        self.setSceneRect(QRectF())

    def get_image_size(self):
        bounds = self._photo.boundingRect()
        return int(bounds.width()), int(bounds.height())

    def update_annotations(self, dust_items):
        self._items = {item.id: item for item in dust_items}
        self.reset_highlight()

    def set_show_all(self, visible):
        self._show_all = visible
        self.viewport().update()

    def highlight_dust(self, dust_id, *, locate=False):
        dust = self._items.get(dust_id)
        if dust is None:
            self.reset_highlight()
            return
        if self._focus_id != dust_id:
            self._focus_id = dust_id
            self._focus_path = defect_path(dust)
        if locate:
            if self.transform().m11() < 0.65:
                self._zoom_to(0.65)
            self.centerOn(*dust.center)
        self.viewport().update()

    def reset_highlight(self):
        self._focus_id = None
        self._focus_path = None
        self.viewport().update()

    def set_comparison(self, original):
        self._before = photo_pixmap(original)
        self.viewport().update()

    def clear_comparison(self):
        self._before = QPixmap()
        self._dragging_split = False
        self.viewport().update()

    def set_split(self, position):
        self._split = position
        self.viewport().update()

    def drawForeground(self, painter, rect):
        if self._photo.pixmap().isNull():
            painter.save()
            painter.resetTransform()
            painter.setPen(QColor("#9eacbc"))
            painter.drawText(self.viewport().rect(), Qt.AlignCenter,
                             "导入胶片扫描照片\n在左侧打开一个文件夹")
            painter.restore()
            return

        if not self._before.isNull():
            self._draw_comparison(painter)
            return
        if self.comparison_label:
            self._draw_badge(painter)
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        for item in self._items.values():
            if item.id == self._focus_id or not self._show_all:
                continue
            point = QPointF(*item.center)
            if not rect.contains(point):
                continue
            pen = QPen(QColor("#65dfcc") if item.selected else QColor("#e6bb72"))
            pen.setCosmetic(True)
            pen.setWidthF(1.25)
            painter.setPen(pen)
            radius = (4 if item.selected else 3) / self.transform().m11()
            painter.drawEllipse(point, radius, radius)
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
            pen = QPen(QColor(100, 235, 210, 85), self.brush_radius * 2,
                       Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            pen.setCosmetic(True)
            painter.setPen(pen)
            path = QPainterPath(self._stroke[0])
            for point in self._stroke[1:]:
                path.lineTo(point)
            painter.drawPath(path)
            if len(self._stroke) == 1:
                painter.drawPoint(self._stroke[0])
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

    def _draw_comparison(self, painter):
        view = self.viewport().rect()
        split = round(view.width() * self._split / 100)
        painter.save()
        painter.setClipRect(self.mapToScene(QRect(0, 0, split, view.height())).boundingRect())
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(0, 0, self._before)
        painter.restore()

        painter.save()
        painter.resetTransform()
        painter.setPen(QPen(QColor("#dce8f3"), 1))
        painter.drawLine(split, 0, split, view.height())
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#dce8f3"))
        painter.drawRoundedRect(QRectF(split - 4, view.height() / 2 - 15, 8, 30), 4, 4)
        painter.restore()

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
        if event.type() == QEvent.NativeGesture and event.gestureType() == Qt.ZoomNativeGesture:
            factor = math.exp(max(-0.16, min(0.16, event.value())))
            self._zoom_to(self.transform().m11() * factor, event.position().toPoint())
            event.accept()
            return True
        return super().viewportEvent(event)

    def set_selection_mode(self, mode):
        self._selection_mode = mode
        self.setDragMode(QGraphicsView.NoDrag if mode else QGraphicsView.ScrollHandDrag)
        self._update_tool_cursor()
        self._cancel_selection()

    def _cancel_selection(self):
        self._press = None
        self._stroke.clear()
        self._box.hide()
        self.viewport().update()

    def _update_tool_cursor(self):
        if self.space_pan:
            cursor = Qt.ClosedHandCursor if self._space_drag is not None else Qt.OpenHandCursor
        else:
            cursor = Qt.CrossCursor if self._selection_mode else Qt.ArrowCursor
        self.viewport().setCursor(cursor)

    def set_space_pan(self, active):
        if self.space_pan == active:
            return
        self.space_pan = active
        if active:
            self._dragging_split = False
            self._cancel_selection()
        self._update_tool_cursor()

    def _stroke_point(self, pos):
        point = self.mapToScene(pos)
        bounds = self.sceneRect()
        return QPointF(min(max(0, point.x()), bounds.width() - 1),
                       min(max(0, point.y()), bounds.height() - 1))

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        if event.button() == Qt.LeftButton:
            self._space_drag = None
            if self.space_pan:
                self._space_drag = pos
                self._update_tool_cursor()
                event.accept()
                return
            self._press = pos
            split_x = round(self.viewport().width() * self._split / 100)
            if not self._before.isNull() and abs(pos.x() - split_x) <= 12:
                self._dragging_split = True
                return
            if self._selection_mode == "draw":
                if self.sceneRect().contains(self.mapToScene(pos)):
                    self._stroke = [self._stroke_point(pos)]
                    self.viewport().update()
                return
            if self._selection_mode:
                self._box.setGeometry(QRect(pos, pos))
                self._box.show()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
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
            if self._selection_mode == "draw":
                if self._stroke:
                    self._stroke.append(self._stroke_point(pos))
                    self.viewport().update()
            else:
                self._box.setGeometry(QRect(self._press, pos).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        pos = event.position().toPoint()
        if event.button() == Qt.LeftButton:
            if self._space_drag is not None:
                self._space_drag = None
                self._update_tool_cursor()
                event.accept()
                return
            start, self._press = self._press, None
            if self._dragging_split:
                self._dragging_split = False
                return
            if self._selection_mode == "draw":
                if self._stroke:
                    self._stroke.append(self._stroke_point(pos))
                    points, self._stroke = self._stroke, []
                    self.stroke_drawn.emit([(p.x(), p.y()) for p in points])
                    self.viewport().update()
                return
            if self._selection_mode and start is not None:
                self._box.hide()
                box = self.mapToScene(QRect(start, pos).normalized()).boundingRect()
                box = box.intersected(self.sceneRect()).toAlignedRect()
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

    def contextMenuEvent(self, event):
        dust_id = self._annotation_at(event.pos())
        if dust_id is not None:
            self.annotation_right_clicked.emit(dust_id, event.globalPos())

    def _annotation_at(self, pos):
        if not self._before.isNull() or self.comparison_label:
            return None
        best = None
        best_distance = 10
        for item in self._items.values():
            if self._show_all or item.id == self._focus_id:
                distance = (self.mapFromScene(QPointF(*item.center)) - pos).manhattanLength()
                if distance < best_distance:
                    best, best_distance = item.id, distance
        return best

"""灰尘修复模块"""
import cv2
import numpy as np
from typing import List
from .dust_item import DustItem
from .grain_repair import restore_grain


class DustRemover:
    """灰尘修复器（使用 OpenCV inpaint）"""
    
    INPAINT_TELEA = "telea"
    INPAINT_NS = "ns"
    
    def __init__(self, method: str = "ns", inpaint_radius: int = 3):
        """
        初始化灰尘修复器
        
        Args:
            method: 修复算法 "telea" 或 "ns"
            inpaint_radius: 修复半径
        """
        self.method = method
        self.inpaint_radius = inpaint_radius
    
    def _get_cv_method(self) -> int:
        """获取 OpenCV inpaint 方法常量"""
        if self.method in (self.INPAINT_NS, "auto", "grain"):
            return cv2.INPAINT_NS
        return cv2.INPAINT_TELEA
    
    def create_mask(
        self,
        image_shape: tuple,
        dust_items: List[DustItem],
        selected_only: bool = True,
        padding: int = 2
    ) -> np.ndarray:
        """
        创建修复掩膜（基于轮廓形状）
        
        Args:
            image_shape: 图像尺寸 (h, w) 或 (h, w, c)
            dust_items: 灰尘列表
            selected_only: 是否只包含选中的灰尘
            padding: 掩膜扩展边距（像素）
            
        Returns:
            二值掩膜
        """
        h = image_shape[0]
        w = image_shape[1]
        mask = np.zeros((h, w), dtype=np.uint8)

        rectangles = []
        for item in dust_items:
            if selected_only and not item.selected:
                continue

            if item.contour is not None:
                cv2.drawContours(mask, [item.contour, *item.holes], -1, 255, cv2.FILLED)
            else:
                rectangles.append(item)

        padding = max(0, padding)
        if padding:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * padding + 1,) * 2)
            cv2.dilate(mask, kernel, dst=mask)
        for item in rectangles:
            x1, y1 = max(0, item.x - padding), max(0, item.y - padding)
            x2 = min(w, item.x + item.width + padding)
            y2 = min(h, item.y + item.height + padding)
            if x1 < x2 and y1 < y2:
                mask[y1:y2, x1:x2] = 255
        return mask
    
    def remove(
        self,
        image: np.ndarray,
        dust_items: List[DustItem],
        selected_only: bool = True,
        padding: int = 2
    ) -> np.ndarray:
        """
        修复图像中的灰尘
        
        Args:
            image: 原始 BGR 图像
            dust_items: 灰尘列表
            selected_only: 是否只修复选中的灰尘
            padding: 掩膜扩展边距
            
        Returns:
            修复后的图像
        """
        # 获取需要修复的灰尘
        items_to_fix = [d for d in dust_items if d.selected] if selected_only else dust_items
        
        if not items_to_fix:
            return image.copy()
        
        # 创建掩膜
        mask = self.create_mask(image.shape, items_to_fix, selected_only=False, padding=padding)
        return self.apply_mask(image, mask)

    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Repair the supplied mask and preserve every other pixel."""
        # Crop work to occupied tiles and use the same original image as
        # context for each tile. Only copy back pixels in the repair mask.
        result = image.copy()
        height, width = mask.shape
        tile_size = 1024
        halo = max(96, 4 * self.inpaint_radius)
        method = self._get_cv_method()
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                core = mask[y:y + tile_size, x:x + tile_size]
                dx, dy, w, h = cv2.boundingRect(core)
                if w == 0 or h == 0:
                    continue
                x1, y1 = x + dx, y + dy
                x2, y2 = x1 + w, y1 + h
                left, top = max(0, x1 - halo), max(0, y1 - halo)
                patch = np.s_[top:min(height, y2 + halo), left:min(width, x2 + halo)]
                pixels = image[patch]
                patch_mask = mask[patch]
                if pixels.dtype == np.uint16 and pixels.ndim == 3:
                    repaired = cv2.merge([
                        cv2.inpaint(channel, patch_mask, self.inpaint_radius, method)
                        for channel in cv2.split(pixels)
                    ])
                else:
                    repaired = cv2.inpaint(pixels, patch_mask, self.inpaint_radius, method)
                if self.method in ("auto", "grain"):
                    restore_grain(pixels, repaired, patch_mask, automatic=self.method == "auto")
                center = np.s_[y1 - top:y2 - top, x1 - left:x2 - left]
                target = np.s_[y1:y2, x1:x2]
                cv2.copyTo(repaired[center], mask[target], result[target])

        return result
    
    def remove_single(
        self,
        image: np.ndarray,
        item: DustItem,
        padding: int = 2
    ) -> np.ndarray:
        """修复单个灰尘"""
        return self.remove(image, [item], selected_only=False, padding=padding)
    
    def get_statistics(self, dust_items: List[DustItem]) -> dict:
        """获取灰尘统计信息"""
        total = len(dust_items)
        selected = sum(1 for d in dust_items if d.selected)
        total_area = sum(d.area for d in dust_items)
        selected_area = sum(d.area for d in dust_items if d.selected)
        
        type_counts = {}
        for d in dust_items:
            t = d.dust_type
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            "total": total,
            "selected": selected,
            "total_area": total_area,
            "selected_area": selected_area,
            "type_counts": type_counts
        }

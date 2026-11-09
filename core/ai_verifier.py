"""AI 视觉验证模块 - 使用显著性检测和纹理分析验证灰尘候选"""
import cv2
import numpy as np
from typing import List
from .dust_item import DustItem


class DustVerifier:
    """智能灰尘验证器
    
    使用多种视觉特征验证候选区域是否真的是灰尘：
    1. 显著性检测（Spectral Residual）
    2. 局部对比度分析
    3. 纹理一致性分析
    4. 形状分析（圆度）
    5. 颜色一致性分析
    """
    
    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence
        
    def verify(self, image: np.ndarray, candidates: List[DustItem]) -> List[DustItem]:
        """验证候选灰尘列表，返回真正可信的灰尘"""
        if not candidates:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 计算显著性图
        saliency_map = self._compute_saliency(gray)
        
        # 对每个候选进行验证
        verified = []
        for item in candidates:
            confidence = self._compute_confidence(image, gray, saliency_map, item)
            
            if confidence >= self.min_confidence:
                item.confidence = confidence
                verified.append(item)
        
        # 按置信度排序
        verified.sort(key=lambda x: x.confidence, reverse=True)
        
        # 重新编号
        for i, item in enumerate(verified):
            item.id = i + 1
        
        return verified
    
    def _compute_saliency(self, gray: np.ndarray) -> np.ndarray:
        """计算显著性图（Spectral Residual 方法）"""
        h, w = gray.shape[:2]
        scale = min(64.0 / h, 64.0 / w, 1.0)
        
        if scale < 1.0:
            small = cv2.resize(gray, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        else:
            small = gray
        
        # FFT
        fft = cv2.dft(np.float32(small), flags=cv2.DFT_COMPLEX_OUTPUT)
        magnitude, phase = cv2.cartToPolar(fft[:,:,0], fft[:,:,1])
        
        # 谱残差
        log_mag = np.log(magnitude + 1e-10)
        smooth = cv2.GaussianBlur(log_mag, (5, 5), 0)
        spectral_residual = log_mag - smooth
        
        # 重建
        mag_recon = np.exp(spectral_residual)
        real, imag = cv2.polarToCart(mag_recon, phase)
        fft_recon = cv2.merge([real, imag])
        
        saliency_small = cv2.idft(fft_recon)
        saliency_small = cv2.magnitude(saliency_small[:,:,0], saliency_small[:,:,1])
        saliency_small = cv2.GaussianBlur(saliency_small, (7, 7), 0)
        
        # 归一化
        saliency_small = (saliency_small - saliency_small.min()) / \
                        (saliency_small.max() - saliency_small.min() + 1e-10)
        
        if scale < 1.0:
            saliency = cv2.resize(saliency_small, (w, h),
                                interpolation=cv2.INTER_LINEAR)
        else:
            saliency = saliency_small
        
        return saliency
    
    def _compute_confidence(self, image, gray, saliency_map, item) -> float:
        """计算单个候选区域的综合置信度"""
        x, y, w, h = item.x, item.y, item.width, item.height
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(gray.shape[1], x + w), min(gray.shape[0], y + h)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # 1. 显著性得分（传统方法）
        sal_region = saliency_map[y1:y2, x1:x2]
        sal_score = float(np.mean(sal_region))
        
        # 2. 局部对比度得分
        contrast_score = self._compute_local_contrast(gray, x1, y1, x2, y2)
        
        # 3. 纹理一致性得分
        texture_score = self._compute_texture_consistency(gray, x1, y1, x2, y2)
        
        # 4. 形状得分
        shape_score = self._compute_shape_score(item)
        
        # 5. 颜色一致性得分
        color_score = self._compute_color_consistency(image, x1, y1, x2, y2)
        
        # 6. 深度学习显著性得分（如果有）
        dl_score = self._compute_dl_saliency(image, x1, y1, x2, y2)
        
        # 综合得分（加权）
        if dl_score > 0:
            # 有深度学习模型时，给它较高权重
            weights = {
                'saliency': 0.25,
                'contrast': 0.20,
                'texture': 0.10,
                'shape': 0.05,
                'color': 0.10,
                'dl': 0.30
            }
            confidence = (
                weights['saliency'] * sal_score +
                weights['contrast'] * contrast_score +
                weights['texture'] * texture_score +
                weights['shape'] * shape_score +
                weights['color'] * color_score +
                weights['dl'] * dl_score
            )
        else:
            # 没有深度学习模型时，只用传统特征
            weights = {
                'saliency': 0.35,
                'contrast': 0.30,
                'texture': 0.15,
                'shape': 0.10,
                'color': 0.10
            }
            confidence = (
                weights['saliency'] * sal_score +
                weights['contrast'] * contrast_score +
                weights['texture'] * texture_score +
                weights['shape'] * shape_score +
                weights['color'] * color_score
            )
        
        return float(confidence)
    
    def _compute_local_contrast(self, gray, x1, y1, x2, y2) -> float:
        """局部对比度得分"""
        region = gray[y1:y2, x1:x2]
        region_mean = np.mean(region)
        
        pad = 20
        sx1, sy1 = max(0, x1 - pad), max(0, y1 - pad)
        sx2, sy2 = min(gray.shape[1], x2 + pad), min(gray.shape[0], y2 + pad)
        
        surround = gray[sy1:sy2, sx1:sx2]
        mask = np.ones(surround.shape, dtype=bool)
        oy, ox = y1 - sy1, x1 - sx1
        rh, rw = y2 - y1, x2 - x1
        mask[oy:oy+rh, ox:ox+rw] = False
        
        if np.sum(mask) == 0:
            return 0.5
        
        surround_mean = np.mean(surround[mask])
        contrast = abs(region_mean - surround_mean)
        
        return float(min(contrast / 50.0, 1.0))
    
    def _compute_texture_consistency(self, gray, x1, y1, x2, y2) -> float:
        """纹理一致性得分（灰尘应纹理简单）"""
        region = gray[y1:y2, x1:x2]
        
        if region.size < 16:
            return 0.5
        
        region_std = np.std(region)
        laplacian = cv2.Laplacian(region, cv2.CV_64F)
        edge_strength = np.mean(np.abs(laplacian))
        
        std_score = 1.0 - min(region_std / 50.0, 1.0)
        edge_score = 1.0 - min(edge_strength / 100.0, 1.0)
        
        return float(0.6 * std_score + 0.4 * edge_score)
    
    def _compute_shape_score(self, item) -> float:
        """形状得分（灰尘接近圆形）"""
        if item.contour is None:
            aspect = item.width / (item.height + 1e-10)
            return float(1.0 - abs(1.0 - aspect) / 2.0)
        
        area = cv2.contourArea(item.contour)
        perimeter = cv2.arcLength(item.contour, True)
        
        if perimeter < 1e-10:
            return 0.5
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return float(circularity)
    
    def _compute_color_consistency(self, image, x1, y1, x2, y2) -> float:
        """颜色一致性得分（灰尘通常单色）"""
        if len(image.shape) != 3:
            return 0.5
        
        region = image[y1:y2, x1:x2]
        
        if region.size < 48:
            return 0.5
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        saturation = hsv[:,:,1]
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)
        
        sat_score = 1.0 - min(sat_mean / 128.0, 1.0)
        consistency = 1.0 - min(sat_std / 64.0, 1.0)
        
        return float(0.7 * sat_score + 0.3 * consistency)
    
    def _compute_dl_saliency(self, image, x1, y1, x2, y2) -> float:
        """使用深度学习模型计算显著性得分"""
        # 尝试加载深度学习检测器
        if not hasattr(self, '_dl_detector'):
            try:
                from .dl_detector import DLDustDetector
                self._dl_detector = DLDustDetector()
                if self._dl_detector.session is None:
                    self._dl_detector = None
            except Exception:
                self._dl_detector = None
        
        if self._dl_detector is None:
            return 0.0  # 模型不可用
        
        try:
            # 生成显著性图
            saliency_map = self._dl_detector.generate_saliency_map(image)
            
            # 计算区域的显著性得分
            region_saliency = saliency_map[y1:y2, x1:x2]
            if region_saliency.size == 0:
                return 0.0
            
            # 使用最大显著性值，并归一化到0-1
            score = float(np.max(region_saliency)) / 255.0
            return score
        except Exception:
            return 0.0


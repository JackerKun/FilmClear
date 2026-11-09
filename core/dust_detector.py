"""灰尘检测算法模块 - 支持灰尘、划痕、毛絮智能检测"""
import cv2
import numpy as np
from typing import List
from .dust_item import DustItem


class DustDetector:
    """智能灰尘检测器"""

    def __init__(self, threshold: int = 20, median_radius: int = 3,
                 min_area: int = 200):
        self.threshold = threshold
        self.median_radius = median_radius
        self.min_area = min_area
        self._aggressive_mode = False
        self.min_confidence = 0.25  # AI验证置信度阈值

    def detect(self, image: np.ndarray, region=None, aggressive=False) -> List[DustItem]:
        """检测图像中的灰尘、划痕、毛絮
        
        核心方法：LoG斑点检测 + 视觉验证
        - 用拉普拉斯金字塔检测"小圆点状"异物
        - 多层级验证排除图像内容
        
        Args:
            image: BGR 格式的图像
            region: 可选的检测区域 (x1, y1, x2, y2)，只在该区域内检测
            aggressive: 是否使用激进模式（用于手动框选区域，灵敏度更高）
        """
        self._aggressive_mode = aggressive
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
            if len(image.shape) == 3 else image.copy()

        h, w = gray.shape[:2]

        # 如果指定了区域，裁剪图像
        if region is not None:
            x1, y1, x2, y2 = region
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            gray = gray[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            offset_x, offset_y = 0, 0

        gray_work = gray
        scale = 1.0
        if max(gray.shape) > 5000:
            scale = 4000.0 / max(gray.shape)
            new_w = int(gray.shape[1] * scale)
            new_h = int(gray.shape[0] * scale)
            gray_work = cv2.resize(gray, (new_w, new_h),
                                   interpolation=cv2.INTER_AREA)

        # === 策略1: LoG斑点检测 - 专门检测圆点状异物（灰尘） ===
        m1 = self._detect_log_blobs(gray_work)

        # === 策略2: 自适应中值滤波差值 ===
        m2 = self._detect_adaptive_median(gray_work)

        # === 策略3: 毛絮检测 ===
        m3 = self._detect_fibers(gray_work)

        # === 策略4: 多尺度形态学 ===
        m4 = self._detect_multiscale_morphology(gray_work)

        # 合并：LoG + 中值差值（主要来源） + 毛絮 + 形态学补充
        combined = cv2.bitwise_or(m1, m2)
        combined = cv2.bitwise_or(combined, cv2.bitwise_or(m3, m4))

        # 形态学清理
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)

        # 分析轮廓
        items = self._analyze_contours(gray_work, combined)

        # 缩放回原图坐标
        if scale < 1.0:
            inv_scale = 1.0 / scale
            for item in items:
                item.x = int(item.x * inv_scale)
                item.y = int(item.y * inv_scale)
                item.width = int(item.width * inv_scale)
                item.height = int(item.height * inv_scale)
                item.center_x = int(item.center_x * inv_scale)
                item.center_y = int(item.center_y * inv_scale)
                item.area = int(item.area * inv_scale * inv_scale)
                if item.contour is not None:
                    item.contour = (item.contour * inv_scale).astype(
                        np.int32)

        # 应用区域偏移
        if offset_x != 0 or offset_y != 0:
            for item in items:
                item.x += offset_x
                item.y += offset_y
                item.center_x += offset_x
                item.center_y += offset_y
                if item.contour is not None:
                    item.contour = item.contour + np.array([offset_x, offset_y], dtype=np.int32)
        
        # AI 验证过滤（除非是激进模式，框选检测时跳过验证）
        if not self._aggressive_mode:
            from .ai_verifier import DustVerifier
            verifier = DustVerifier(min_confidence=self.min_confidence)
            items = verifier.verify(image, items)
        
        return items

    def _detect_log_blobs(self, gray: np.ndarray) -> np.ndarray:
        """LoG（拉普拉斯高斯）斑点检测
        
        专门检测"小圆点状"异物，这正是灰尘的视觉特征。
        在多个尺度上应用 LoG 滤波器，检测不同大小的灰尘。
        """
        gray_f = gray.astype(np.float32)
        
        # 多尺度 LoG 检测（sigma 从小到大对应灰尘从小到大）
        sigmas = [1.5, 2.5, 4.0, 6.0, 9.0, 13.0]
        
        # 计算 LoG 响应的最大值图
        log_response = np.zeros_like(gray_f)
        
        for sigma in sigmas:
            # LoG = Laplacian of Gaussian
            # 先高斯模糊再拉普拉斯
            blurred = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=sigma)
            lap = cv2.Laplacian(blurred, cv2.CV_32F)
            # sigma^2 归一化
            lap_normalized = np.abs(lap) * (sigma ** 2)
            log_response = np.maximum(log_response, lap_normalized)
        
        # 计算自适应阈值（基于局部统计）
        # 局部均值和标准差
        local_mean = cv2.GaussianBlur(log_response, (0, 0), sigmaX=50)
        local_sq = cv2.GaussianBlur(log_response ** 2, (0, 0), sigmaX=50)
        local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))
        
        # 自适应阈值：均值 + k * 标准差
        # k 值控制灵敏度
        k = 2.5
        threshold = local_mean + k * local_std
        threshold = np.maximum(threshold, 5.0)  # 最低阈值
        
        # 二值化
        mask = (log_response > threshold).astype(np.uint8) * 255
        
        # 形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _detect_by_local_contrast(self, gray: np.ndarray) -> np.ndarray:
        """局部对比度检测 - 找出与周围背景反差大的区域"""
        # 大半径模糊代表"背景"
        bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=20)
        
        # 计算每个像素与背景的差异
        diff = cv2.absdiff(gray.astype(np.float32),
                           bg.astype(np.float32))
        
        # 计算局部标准差（纹理区域标准差大，平滑区域标准差小）
        local_mean = cv2.GaussianBlur(gray.astype(np.float32),
                                      (0, 0), sigmaX=10)
        local_sq_mean = cv2.GaussianBlur(
            (gray.astype(np.float32))**2, (0, 0), sigmaX=10
        )
        local_var = local_sq_mean - local_mean ** 2
        local_var = np.maximum(local_var, 0)
        local_std = np.sqrt(local_var)
        
        # 在平滑区域（低标准差），较低的对比度就认为是灰尘
        # 在纹理区域（高标准差），需要更高的对比度
        adaptive_threshold = np.maximum(15, local_std * 1.5)
        
        # 归一化差异
        normalized_diff = diff / (adaptive_threshold + 1e-6)
        
        # 阈值化
        mask = (normalized_diff > 1.5).astype(np.uint8) * 255
        
        return mask

    def _detect_adaptive_median(self, gray: np.ndarray) -> np.ndarray:
        """自适应中值滤波差值 - 根据局部纹理复杂度动态调整阈值"""
        ksize = 2 * self.median_radius + 1
        blurred = cv2.medianBlur(gray, ksize)
        diff = cv2.absdiff(gray.astype(np.float32),
                           blurred.astype(np.float32))

        # 计算局部纹理复杂度（局部标准差）
        gray_f = gray.astype(np.float32)
        local_mean = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=15)
        local_sq = cv2.GaussianBlur(gray_f ** 2, (0, 0), sigmaX=15)
        local_var = np.maximum(local_sq - local_mean ** 2, 0)
        local_std = np.sqrt(local_var)

        # 自适应阈值：平滑区域用低阈值，复杂区域用高阈值
        # 公式: threshold = base + k * local_std
        base_thr = max(3, self.threshold * 0.15)
        adaptive_thr = base_thr + local_std * 0.4
        adaptive_thr = np.minimum(adaptive_thr, self.threshold)

        mask = (diff > adaptive_thr).astype(np.uint8) * 255
        return mask

    def _detect_in_smooth_areas(self, gray: np.ndarray) -> np.ndarray:
        """平滑区域专用检测 - 用低阈值捕捉细微灰尘
        
        在胶片边框、天空、墙壁等平滑区域，
        即使2-5个灰度差异的灰尘也是人眼可见的
        """
        gray_f = gray.astype(np.float32)

        # 计算局部纹理复杂度
        local_mean = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=30)
        local_sq = cv2.GaussianBlur(gray_f ** 2, (0, 0), sigmaX=30)
        local_var = np.maximum(local_sq - local_mean ** 2, 0)
        local_std = np.sqrt(local_var)

        # 只在非常平滑的区域进行检测（std < 10）
        smooth_mask = local_std < 10

        # 用大半径高斯模糊创建"干净背景"
        bg = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=25)
        diff = np.abs(gray_f - bg)

        # 阈值检测（在平滑区域 3.0 灰度差就算灰尘）
        dust_mask = diff > 3.0

        # 只保留平滑区域的检测结果
        result = np.zeros(gray.shape[:2], dtype=np.uint8)
        result[smooth_mask & dust_mask] = 255

        # 形态学闭操作连接相邻像素
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_close)

        # 去除太小的孤立噪点（面积 < 6px）
        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(result)
        for cnt in contours:
            if cv2.contourArea(cnt) >= 6:
                cv2.drawContours(filtered, [cnt], -1, 255, -1)

        return filtered

    def _detect_multiscale_morphology(self, gray: np.ndarray) -> np.ndarray:
        """多尺度形态学检测不同大小的灰尘"""
        results = []
        
        # 小灰尘 (3x3)
        k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat_s = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_small)
        blackhat_s = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_small)
        
        # 中等灰尘 (7x7)
        k_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tophat_m = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_med)
        blackhat_m = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_med)
        
        # 大灰尘 (15x15)
        k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat_l = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_large)
        blackhat_l = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_large)
        
        # 分别阈值化并合并
        thr = self.threshold
        for th in [tophat_s, blackhat_s, tophat_m, blackhat_m,
                   tophat_l, blackhat_l]:
            _, m = cv2.threshold(th, thr, 255, cv2.THRESH_BINARY)
            results.append(m)
        
        combined = results[0]
        for m in results[1:]:
            combined = cv2.bitwise_or(combined, m)
        
        return combined

    
    # ---- 策略1: 中值滤波差值 ----
    def _detect_by_median_diff(self, gray: np.ndarray) -> np.ndarray:
        ksize = 2 * self.median_radius + 1
        blurred = cv2.medianBlur(gray, ksize)
        diff = cv2.absdiff(gray, blurred)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        return mask

    # ---- 策略2: 形态学 Top-Hat / Black-Hat ----
    def _detect_by_morphology(self, gray: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, bright = cv2.threshold(tophat, self.threshold, 255,
                                  cv2.THRESH_BINARY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, dark = cv2.threshold(blackhat, self.threshold, 255,
                                cv2.THRESH_BINARY)
        return cv2.bitwise_or(bright, dark)

    # ---- 策略3: 毛絮/纤维检测 ----
    def _detect_fibers(self, gray: np.ndarray) -> np.ndarray:
        """用方向性核检测细长的毛絮/纤维"""
        results = []
        length = 15
        for angle in [0, 45, 90, 135]:
            kernel = self._line_kernel(length, angle)
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            diff = cv2.absdiff(gray, opened)
            thr = max(self.threshold - 5, 8)
            _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
            results.append(mask)

        # 自适应阈值辅助检测低对比度毛絮
        adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 5
        )
        blurred_a = cv2.medianBlur(adapt, 21)
        adapt_diff = cv2.absdiff(adapt, blurred_a)
        _, adapt_mask = cv2.threshold(adapt_diff, 50, 255,
                                      cv2.THRESH_BINARY)
        results.append(adapt_mask)

        combined = results[0]
        for m in results[1:]:
            combined = cv2.bitwise_or(combined, m)
        return combined

    @staticmethod
    def _line_kernel(length: int, angle: int) -> np.ndarray:
        """创建方向性线性核"""
        k = np.zeros((length, length), dtype=np.uint8)
        c = length // 2
        if angle == 0:
            k[c, :] = 1
        elif angle == 90:
            k[:, c] = 1
        elif angle == 45:
            for i in range(length):
                k[i, length - 1 - i] = 1
        elif angle == 135:
            for i in range(length):
                k[i, i] = 1
        return k

    
    def _analyze_contours(self, gray: np.ndarray,
                          mask: np.ndarray) -> List[DustItem]:
        """基于轮廓分析生成灰尘对象（带视觉显著性评分）"""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        h_img, w_img = gray.shape[:2]
        img_area = h_img * w_img
        
        # 自适应阈值：基于图像尺寸
        # 最小面积：至少 3x3 像素
        adaptive_min = max(9, int(img_area * 0.000002))  # 0.0002% of image
        # 最大面积：不超过图像面积的 2%
        adaptive_max = int(img_area * 0.02)
        
        candidates = []
        border_margin = 0  # 不过滤边缘 - 胶片边框上常有灰尘

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < adaptive_min or area > adaptive_max:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if (x < border_margin or y < border_margin or
                    x + w > w_img - border_margin or
                    y + h > h_img - border_margin):
                continue

            if w < 3 or h < 3:
                continue

            # 基本过滤
            if not self._is_real_dust_simple(gray, cnt, x, y, w, h, area):
                continue

            # 计算视觉显著性评分
            score = self.compute_salience_score(gray, cnt, x, y, w, h, area)
            if score < 0:
                continue

            aspect = max(w, h) / (min(w, h) + 1e-6)
            cx, cy = x + w // 2, y + h // 2

            candidates.append({
                'cnt': cnt, 'x': x, 'y': y, 'w': w, 'h': h,
                'area': area, 'cx': cx, 'cy': cy,
                'aspect': aspect, 'score': score
            })

        # 按评分降序排列
        candidates.sort(key=lambda c: c['score'], reverse=True)
        
        # 过滤低分候选（只保留高分）
        min_score = 0.5 if self._aggressive_mode else 0.8
        candidates = [c for c in candidates if c['score'] >= min_score]
        
        # 取评分最高的 N 个（避免太多结果）
        max_results = 200
        candidates = candidates[:max_results]

        dust_items = []
        for dust_id, c in enumerate(candidates, 1):
            dust_type = self._classify_contour(
                gray, c['cnt'], c['area'], c['aspect'],
                c['x'], c['y'], c['w'], c['h']
            )

            item = DustItem(
                id=dust_id, x=c['x'], y=c['y'],
                width=c['w'], height=c['h'],
                area=c['area'], center_x=c['cx'],
                center_y=c['cy'],
                dust_type=dust_type, selected=True,
                contour=c['cnt'], aspect_ratio=c['aspect'],
                confidence=c['score']
            )
            dust_items.append(item)

        return dust_items

    def _is_real_dust_simple(self, gray, cnt, x, y, w, h, area) -> bool:
        """宽松的基本过滤 - 让评分系统做最终决策"""
        mask_roi = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
        
        dust_region = gray[y:y+h, x:x+w]
        mask_r = mask_roi[y:y+h, x:x+w]
        dust_pixels = dust_region[mask_r > 0]
        
        if len(dust_pixels) < 4:
            return False
        
        dust_mean = float(np.mean(dust_pixels))
        
        # 近邻背景分析
        pad = 25
        y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
        surround = gray[y1:y2, x1:x2].astype(np.float32)
        
        surround_mean = float(np.mean(surround))
        contrast = abs(dust_mean - surround_mean)
        
        # 只要有微小对比度就通过，交给评分系统排序
        return contrast >= 3.0
    
    def compute_salience_score(self, gray, cnt, x, y, w, h, area) -> float:
        """计算灰尘的视觉显著性评分（越高越可能是真正的灰尘）
        
        严格验证机制：
        1. 必须有足够的对比度（与周围背景的亮度差异）
        2. 在平滑区域的灰尘更显眼
        3. 面积和形状要合理
        4. 视觉上要明显可见
        """
        mask_roi = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
        
        dust_region = gray[y:y+h, x:x+w]
        mask_r = mask_roi[y:y+h, x:x+w]
        dust_pixels = dust_region[mask_r > 0]
        
        if len(dust_pixels) < 4:
            return -1.0
        
        dust_mean = float(np.mean(dust_pixels))
        dust_std = float(np.std(dust_pixels))
        
        # === 严格视觉验证 ===
        
        # 1. 局部对比度验证（最关键）
        pad = 30
        y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
        surround = gray[y1:y2, x1:x2].astype(np.float32)
        
        s_mean = float(np.mean(surround))
        s_std = float(np.std(surround))
        contrast = abs(dust_mean - s_mean)
        
        # 根据模式调整阈值
        if self._aggressive_mode:
            # 激进模式：极低阈值，用于手动框选区域
            min_contrast = 2.0
        else:
            # 普通模式：适中阈值
            min_contrast = 3.0
        
        # 对比度必须足够大
        if contrast < min_contrast:
            return -1.0
        
        # 2. 纹理复杂度验证
        # 在复杂纹理区域，需要更高的对比度
        if s_std > 40:
            threshold = 10 if self._aggressive_mode else 15
            if contrast < threshold:
                return -1.0
        elif s_std > 25:
            threshold = 6 if self._aggressive_mode else 10
            if contrast < threshold:
                return -1.0
        elif s_std > 15:
            threshold = 4 if self._aggressive_mode else 6
            if contrast < threshold:
                return -1.0
        
        # 3. 灰尘内部一致性验证
        # 真正的灰尘内部应该比较均匀
        if dust_std > 25:
            return -1.0  # 内部太不均匀，可能是图像纹理
        
        # === 通过验证后计算评分 ===
        
        # 对比度分数（0-3分）
        contrast_score = min(contrast / 15.0, 3.0)
        
        # 纹理复杂度奖励（0.2-1.5倍）
        if s_std < 8:
            texture_bonus = 1.5  # 平滑区域加分
        elif s_std < 15:
            texture_bonus = 1.2
        elif s_std < 25:
            texture_bonus = 0.8
        else:
            texture_bonus = 0.5  # 复杂区域减分
        
        score = contrast_score * texture_bonus
        
        # 面积奖励（0.7-1.5倍）
        if area > 500:
            score *= 1.5
        elif area > 200:
            score *= 1.2
        elif area > 100:
            score *= 1.0
        elif area > 50:
            score *= 0.8
        else:
            score *= 0.5  # 太小可能是噪点
        
        # 紧凑度奖励（0.7-1.2倍）
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity > 0.7:
                score *= 1.2  # 紧凑的更像灰尘
            elif solidity < 0.4:
                score *= 0.7  # 不规则形状可能是纹理
        
        return score

    def _classify_contour(self, gray, cnt, area, aspect,
                          x, y, w, h) -> str:
        """智能分类: 灰尘/毛絮/划痕/亮斑/暗斑"""
        peri = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / (peri * peri + 1e-6)
                       if peri > 0 else 0)

        # 获取周围区域亮度
        pad = 8
        y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
        surround_mean = np.mean(gray[y1:y2, x1:x2])

        # 获取灰尘区域亮度
        mask_roi = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
        region = gray[y:y + h, x:x + w]
        mask_r = mask_roi[y:y + h, x:x + w]
        dust_px = region[mask_r > 0]
        dust_mean = np.mean(dust_px) if len(dust_px) > 0 else surround_mean
        bdiff = dust_mean - surround_mean

        # 毛絮: 非常细长 + 小面积
        if aspect > 5.0 and area < 500:
            return "fiber"
        # 划痕: 较细长
        if aspect > 4.0 and area < 2000:
            return "scratch"
        # 圆形灰尘
        if circularity > 0.5 and area < 200:
            if bdiff > 10:
                return "bright"
            elif bdiff < -10:
                return "dark"
            return "dust"
        # 按亮度分类
        if bdiff > 15:
            return "bright"
        elif bdiff < -15:
            return "dark"
        elif aspect > 2.5:
            return "fiber"
        return "dust"

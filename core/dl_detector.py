"""深度学习灰尘检测器 - 使用U-2-Net显著性检测模型"""
import cv2
import numpy as np
import os
import urllib.request
from typing import List, Optional
from .dust_item import DustItem


class DLDustDetector:
    """基于深度学习的灰尘检测器
    
    使用预训练的U-2-Net模型进行显著性检测，
    能够识别传统方法难以检测的低对比度灰尘。
    """
    
    MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
    MODEL_PATH = "models/u2netp.onnx"
    INPUT_SIZE = 320
    
    def __init__(self, min_area: int = 50, min_confidence: float = 0.3):
        self.min_area = min_area
        self.min_confidence = min_confidence
        self.session = None
        self._load_model()
    
    def _load_model(self):
        """加载ONNX模型"""
        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
            print(f"下载U-2-Net模型到 {self.MODEL_PATH}...")
            try:
                # 禁用SSL验证（用于测试环境）
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
                urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
                print("模型下载完成")
            except Exception as e:
                print(f"模型下载失败: {e}")
                print("将使用传统方法进行检测")
                return
        
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(self.MODEL_PATH)
            print("U-2-Net模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.session = None
    
    def detect(self, image: np.ndarray) -> List[DustItem]:
        """检测图像中的灰尘"""
        if self.session is None:
            print("深度学习模型不可用，返回空结果")
            return []
        
        # 1. 使用U-2-Net生成显著性图
        saliency_map = self.generate_saliency_map(image)
        
        # 2. 从显著性图中提取灰尘候选
        items = self._extract_dust_from_saliency(saliency_map, image)
        
        return items
    
    def generate_saliency_map(self, image: np.ndarray) -> np.ndarray:
        """使用U-2-Net生成显著性图（公开方法）"""
        return self._generate_saliency_map(image)
    
    def _generate_saliency_map(self, image: np.ndarray) -> np.ndarray:
        """使用U-2-Net生成显著性图"""
        h, w = image.shape[:2]
        
        # 预处理
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        
        # 调整大小
        input_image = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
        
        # 标准化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_image = (input_image - mean) / std
        
        # 转换为模型输入格式 (N, C, H, W)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0).astype(np.float32)
        
        # 推理
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_image})
        
        # 获取显著性图（第一个输出）
        saliency = output[0][0]
        
        # 后处理
        saliency = np.squeeze(saliency)
        saliency = cv2.resize(saliency, (w, h))
        
        # 归一化到0-255
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        saliency = (saliency * 255).astype(np.uint8)
        
        return saliency
    
    def _extract_dust_from_saliency(self, saliency_map: np.ndarray,
                                    image: np.ndarray) -> List[DustItem]:
        """从显著性图中提取灰尘"""
        # 自适应阈值化
        thresh = cv2.adaptiveThreshold(
            saliency_map, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 5
        )
        
        # 形态学操作清理噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        items = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # 面积过滤
            if area < self.min_area or area > 10000:
                continue
            
            # 计算边界框
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 计算置信度（基于显著性图）
            mask = np.zeros(saliency_map.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            confidence = np.mean(saliency_map[mask > 0]) / 255.0
            
            # 置信度过滤
            if confidence < self.min_confidence:
                continue
            
            # 计算中心点
            cx, cy = x + w // 2, y + h // 2
            
            # 分类
            dust_type = self._classify_dust(gray, cnt, x, y, w, h)
            
            # 创建DustItem
            item = DustItem(
                id=len(items) + 1,
                x=x, y=y, width=w, height=h,
                area=area, center_x=cx, center_y=cy,
                dust_type=dust_type, confidence=confidence,
                contour=cnt
            )
            items.append(item)
        
        # 按置信度排序
        items.sort(key=lambda x: x.confidence, reverse=True)
        
        # 重新编号
        for i, item in enumerate(items, 1):
            item.id = i
        
        return items
    
    def _classify_dust(self, gray: np.ndarray, cnt: np.ndarray,
                      x: int, y: int, w: int, h: int) -> str:
        """分类灰尘类型"""
        # 计算圆度
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if perimeter < 1e-10:
            return "unknown"
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 计算宽高比
        aspect_ratio = max(w, h) / (min(w, h) + 1e-10)
        
        # 分类逻辑
        if aspect_ratio > 5:
            return "scratch"
        elif aspect_ratio > 3:
            return "fiber"
        elif circularity > 0.7:
            # 计算亮度差异
            mask = np.zeros(gray.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            dust_mean = np.mean(gray[mask > 0])
            
            # 计算周围区域亮度
            pad = 20
            y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
            surround = gray[y1:y2, x1:x2]
            
            surround_mask = np.ones(surround.shape, dtype=bool)
            oy, ox = y - y1, x - x1
            surround_mask[oy:oy+h, ox:ox+w] = False
            
            if np.sum(surround_mask) > 0:
                surround_mean = np.mean(surround[surround_mask])
                brightness_diff = dust_mean - surround_mean
                
                if brightness_diff > 15:
                    return "bright"
                elif brightness_diff < -15:
                    return "dark"
            
            return "dust"
        else:
            return "unknown"
"""配置管理模块"""
import json
import os
import sys
from typing import Any, Dict


DEFAULT_CONFIG = {
    "theme": "dark",
    "detection": {
        "threshold": 25,
        "median_radius": 3,
        "min_area": 5,
        "max_area": 5000
    },
    "inpaint": {
        "method": "telea",
        "radius": 3,
        "padding": 2
    },
    "output_dir": "output",
    "last_directory": "",
    "window": {
        "width": 1400,
        "height": 900,
        "x": -1,
        "y": -1
    }
}


class Config:
    """配置管理器"""
    
    CONFIG_FILE = "filmclear_config.json"
    
    def __init__(self, config_dir: str = ""):
        if not config_dir:
            if getattr(sys, "frozen", False):
                from PySide6.QtCore import QStandardPaths

                config_dir = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
            else:
                config_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(config_dir, self.CONFIG_FILE)
        self._data = DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self):
        """从文件加载配置"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self._merge(self._data, loaded)
            except Exception:
                pass
    
    def save(self):
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(self.config_path) or '.', exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项（支持点分路径如 "detection.threshold"）"""
        keys = key.split('.')
        data = self._data
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return default
        return data
    
    def set(self, key: str, value: Any):
        """设置配置项（支持点分路径）"""
        keys = key.split('.')
        data = self._data
        for k in keys[:-1]:
            if k not in data or not isinstance(data[k], dict):
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value
    
    def _merge(self, base: dict, override: dict):
        """递归合并配置"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge(base[key], value)
            else:
                base[key] = value

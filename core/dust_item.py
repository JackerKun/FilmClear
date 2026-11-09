"""灰尘项数据模型"""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class DustItem:
    """表示一个灰尘/划痕/毛絮对象"""
    id: int
    x: int
    y: int
    width: int
    height: int
    area: int
    center_x: int
    center_y: int
    dust_type: str = "unknown"
    confidence: float = 0.8
    selected: bool = True
    contour: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    aspect_ratio: float = 1.0
    holes: Tuple[np.ndarray, ...] = field(default_factory=tuple, repr=False, compare=False)
    preview: bool = False

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.center_x, self.center_y)

    @property
    def type_name(self) -> str:
        names = {
            "bright": "亮斑", "dark": "暗斑",
            "scratch": "划痕", "fiber": "毛絮", "damage": "片面破损",
            "dust": "灰尘", "infrared": "红外缺陷", "manual": "手绘缺陷", "unknown": "杂质"
        }
        return names.get(self.dust_type, "杂质")

    @property
    def type_icon(self) -> str:
        icons = {
            "bright": "⚪", "dark": "⚫",
            "scratch": "📏", "fiber": "🧵",
            "dust": "💨", "unknown": "❓"
        }
        return icons.get(self.dust_type, "❓")

    @property
    def color(self) -> Tuple[int, int, int]:
        colors = {
            "bright": (0, 200, 255),
            "dark": (255, 100, 100),
            "scratch": (0, 255, 0),
            "fiber": (255, 0, 255),
            "dust": (0, 165, 255),
            "unknown": (200, 200, 200)
        }
        return colors.get(self.dust_type, (200, 200, 200))

    @property
    def highlight_color(self) -> Tuple[int, int, int]:
        """高亮时的颜色 (亮黄)"""
        return (0, 255, 255)

    def __str__(self) -> str:
        return (f"#{self.id} {self.type_name} "
                f"({self.center_x},{self.center_y}) {self.area}px²")

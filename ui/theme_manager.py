"""主题管理器"""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Signal, QObject
from .styles import DARK_THEME, LIGHT_THEME


class ThemeManager(QObject):
    """主题管理器（单例模式）"""
    
    theme_changed = Signal(str)
    
    _instance = None
    _initialized = False
    
    DARK = "dark"
    LIGHT = "light"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._current_theme = cls.DARK
        return cls._instance

    def __init__(self):
        if ThemeManager._initialized:
            return
        super().__init__()
        ThemeManager._initialized = True
    
    @property
    def current_theme(self) -> str:
        return self._current_theme
    
    @property
    def is_dark(self) -> bool:
        return self._current_theme == self.DARK
    
    def set_theme(self, theme: str, app: QApplication = None):
        """设置主题"""
        self._current_theme = theme
        
        if theme == self.DARK:
            stylesheet = DARK_THEME
        else:
            stylesheet = LIGHT_THEME
        
        if app is None:
            app = QApplication.instance()
        
        if app:
            app.setStyleSheet(stylesheet)
        
        self.theme_changed.emit(theme)
    
    def toggle_theme(self, app: QApplication = None):
        """切换主题"""
        new_theme = self.LIGHT if self.is_dark else self.DARK
        self.set_theme(new_theme, app)
        return new_theme
    
    def get_theme_name(self) -> str:
        """获取当前主题中文名"""
        return "深色主题" if self.is_dark else "浅色主题"
    
    def get_toggle_icon(self) -> str:
        """获取主题切换图标"""
        return "☀️" if self.is_dark else "🌙"

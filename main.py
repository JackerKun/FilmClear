"""
FilmClear - 胶片去灰尘工具
程序入口
"""
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from ui.main_window import MainWindow


def main():
    """主函数"""
    # 高 DPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("FilmClear")
    app.setApplicationVersion("1.2.0")
    app.setOrganizationName("FilmClear")
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "filmclear.png")))
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

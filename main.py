"""FilmClear application entry point."""

import faulthandler
import logging
from pathlib import Path
import sys
from tempfile import gettempdir

from release_check import WORKER_FLAG, fetch_latest_release


MIN_WINDOWS_BUILD = 17763  # Windows 10 version 1809, the minimum supported by Qt 6.


def main():
    if sys.argv[1:2] == [WORKER_FLAG]:
        return fetch_latest_release(sys.argv[2]) if len(sys.argv) == 3 else 2

    # A windowed Windows build has no console. Keep the crash log local.
    log = None
    if sys.platform == "win32":
        try:
            log = (Path(gettempdir()) / "FilmClear-startup.log").open("w", encoding="utf-8", buffering=1)
        except OSError:
            pass
    if log is None:
        return run_gui()

    with log:
        logging.basicConfig(stream=log, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        faulthandler.enable(log)
        try:
            return run_gui()
        except Exception:
            logging.exception("FilmClear failed to start")
            raise
        finally:
            faulthandler.disable()
            logging.shutdown()


def run_gui():
    safe_mode = "--safe-mode" in sys.argv[1:]
    qt_args = [arg for arg in sys.argv if arg != "--safe-mode"]
    logging.info("Loading FilmClear (safe mode: %s)", safe_mode)

    if sys.platform == "win32":
        build = sys.getwindowsversion().build
        if build < MIN_WINDOWS_BUILD:
            # Show the requirement even when Windows cannot load the Qt libraries.
            from ctypes import windll

            logging.error("Unsupported Windows build: %s (requires %s or later)", build, MIN_WINDOWS_BUILD)
            message = ("FilmClear 需要 Windows 10 1809（内部版本 17763）或更新版本。\n\n"
                       f"当前 Windows 内部版本为 {build}。请在较新版本的 Windows 上运行。")
            windll.user32.MessageBoxW(None, message, "FilmClear", 0x10)  # MB_ICONERROR
            return 1

    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(qt_args)
    app.setApplicationName("FilmClear")
    app.setApplicationVersion("1.3.4")
    app.setOrganizationName("FilmClear")
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "filmclear.png")))

    window = MainWindow(restore_workspace=not safe_mode)
    window.show()
    logging.info("Main window shown")
    if not safe_mode:
        QTimer.singleShot(0, window.check_for_updates)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

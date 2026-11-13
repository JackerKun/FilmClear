import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QPushButton

from config import Config
from release_check import MAX_RESPONSE_BYTES, fetch_latest_release
from ui.main_window import MainWindow
from ui.update_checker import UpdateChecker


class UpdateCheckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_for(self, condition):
        deadline = time.monotonic() + 5
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(condition())

    @staticmethod
    def worker_result(tag):
        return lambda result_file: ["-c", f"from pathlib import Path; Path({str(result_file)!r}).write_text({tag!r})"]

    def test_only_newer_final_releases_are_reported(self):
        checker = UpdateChecker("1.3.2")
        results = []
        checker.check_finished.connect(lambda tag, url: results.append((tag, url)))
        for tag, draft, prerelease, expected_tag, has_update in (
            ("v1.3.10", False, False, "v1.3.10", True),
            ("v1.3.2", False, False, "v1.3.2", False),
            ("v1.2.99", False, False, "v1.2.99", False),
            ("v2.0.0", True, False, "", False),
            ("v2.0.0", False, True, "", False),
            ("v2.0.0-rc.1", False, False, "", False),
        ):
            with self.subTest(tag=tag, draft=draft, prerelease=prerelease), \
                    tempfile.TemporaryDirectory() as directory, \
                    patch("urllib.request.build_opener") as build_opener:
                body = json.dumps({"tag_name": tag, "draft": draft, "prerelease": prerelease})
                response = Mock(status=200)
                response.read.return_value = body.encode()
                opened = build_opener.return_value.open.return_value
                opened.__enter__.return_value = response
                destination = Path(directory) / "tag"
                self.assertEqual(fetch_latest_release(destination), 0 if expected_tag else 1)
                response.read.assert_called_once_with(MAX_RESPONSE_BYTES + 1)
                opened.__exit__.assert_called_once()
                results.clear()
                worker_tag = destination.read_text() if destination.exists() else "invalid"
                with patch.object(checker, "_worker_arguments", self.worker_result(worker_tag)):
                    checker.check()
                self.wait_for(lambda: results)
                result_tag, page = results[0]
                self.assertEqual(result_tag, expected_tag)
                self.assertEqual(not page.isEmpty(), has_update)

    def test_worker_exit_and_failed_launch_do_not_announce_an_update(self):
        checker = UpdateChecker("1.3.2")
        finished = []
        checker.check_finished.connect(lambda tag, url: finished.append((tag, url)))
        with patch.object(checker, "_worker_arguments", return_value=["-c", "raise SystemExit(1)"]):
            checker.check()
            self.wait_for(lambda: len(finished) == 1)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "executable", str(Path(directory) / "missing")):
                checker.check()
            self.wait_for(lambda: len(finished) == 2)
        self.assertEqual(finished, [("", QUrl()), ("", QUrl())])

    def test_timed_out_worker_finishes_and_can_be_retried(self):
        checker = UpdateChecker("1.3.2")
        finished = []
        checker.check_finished.connect(lambda tag, url: finished.append((tag, url)))
        with patch("ui.update_checker.TIMEOUT_MS", 150), \
                patch.object(checker, "_worker_arguments", return_value=["-c", "import time; time.sleep(10)"]):
            checker.check()
            self.wait_for(lambda: finished)
        self.assertEqual(finished, [("", QUrl())])
        with patch.object(checker, "_worker_arguments", self.worker_result("v1.3.10")):
            checker.check()
            self.wait_for(lambda: len(finished) == 2)
        self.assertEqual(finished[-1][0], "v1.3.10")

    def test_manual_check_and_github_link(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("ui.main_window.Config", lambda: Config(directory)), \
                patch.object(QApplication, "applicationVersion", return_value="1.3.2"):
            window = MainWindow()
            try:
                help_menu = next(action.menu() for action in window.menuBar().actions() if action.text() == "说明")
                actions = {action.text(): action for action in help_menu.actions()}
                check = actions["检查更新"]
                received = []
                window._update_checker.check_finished.connect(lambda tag, _: received.append(tag))
                with patch("ui.main_window.QDesktopServices.openUrl") as open_url:
                    with patch.object(window._update_checker, "_worker_arguments", self.worker_result("v1.3.10")):
                        window.check_for_updates()
                        check.trigger()
                        self.assertFalse(check.isEnabled())
                    self.assertTrue(window._update_banner.isHidden())
                    self.wait_for(lambda: not window._update_banner.isHidden())
                    self.assertEqual(received, ["v1.3.10"])
                    self.assertIn("发现新版本", window.statusBar().currentMessage())
                    open_url.assert_not_called()
                    buttons = {button.text(): button for button in window._update_banner.findChildren(QPushButton)}
                    buttons["去下载"].click()
                    page = open_url.call_args.args[0]
                    self.assertEqual(page.toString(), "https://github.com/JackerKun/FilmClear/releases/tag/v1.3.10")
                    buttons["稍后"].click()
                    self.assertTrue(window._update_banner.isHidden())
                    actions["打开 GitHub"].trigger()
                    self.assertEqual(open_url.call_args.args[0].toString(), "https://github.com/JackerKun/FilmClear")

                with patch.object(window._update_checker, "_worker_arguments", self.worker_result("v1.3.2")):
                    check.trigger()
                    self.wait_for(check.isEnabled)
                self.assertIn("暂无可用更新", window.statusBar().currentMessage())
                # A native crash in the worker must leave the GUI and retry action usable.
                with patch.object(window._update_checker, "_worker_arguments",
                                  return_value=["-c", "import os; os.abort()"]):
                    check.trigger()
                    self.wait_for(check.isEnabled)
                self.assertIn("暂时无法检查更新", window.statusBar().currentMessage())
                self.assertEqual(received[-1], "")
            finally:
                window.close()

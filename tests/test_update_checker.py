import json
import os
import tempfile
import time
import unittest
from urllib.parse import quote
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QPushButton

from config import Config
from ui.main_window import MainWindow
from ui.update_checker import UpdateChecker


class UpdateCheckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_for(self, condition):
        deadline = time.monotonic() + 1
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(condition())

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
            with self.subTest(tag=tag, draft=draft, prerelease=prerelease):
                body = json.dumps({"tag_name": tag, "draft": draft, "prerelease": prerelease})
                results.clear()
                with patch("ui.update_checker.RELEASE_API", "data:application/json," + quote(body)):
                    checker.check()
                self.wait_for(lambda: results)
                result_tag, page = results[0]
                self.assertEqual(result_tag, expected_tag)
                self.assertEqual(not page.isEmpty(), has_update)

    def test_failed_request_does_not_announce_an_update(self):
        checker = UpdateChecker("1.3.2")
        finished = []
        checker.check_finished.connect(lambda tag, url: finished.append((tag, url)))
        with tempfile.TemporaryDirectory() as directory:
            endpoint = QUrl.fromLocalFile(os.path.join(directory, "missing.json")).toString()
            with patch("ui.update_checker.RELEASE_API", endpoint):
                checker.check()
            self.wait_for(lambda: finished)
        self.assertEqual(finished, [("", QUrl())])

    def test_manual_check_and_github_link(self):
        response = {"tag_name": "v1.3.10", "draft": False, "prerelease": False,
                    "html_url": "https://example.org/untrusted"}
        endpoint = "data:application/json," + quote(json.dumps(response))
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
                    with patch("ui.update_checker.RELEASE_API", endpoint):
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

                response["tag_name"] = "v1.3.2"
                with patch("ui.update_checker.RELEASE_API", "data:application/json," + quote(json.dumps(response))):
                    check.trigger()
                    self.wait_for(check.isEnabled)
                self.assertIn("暂无可用更新", window.statusBar().currentMessage())
                with patch("ui.update_checker.RELEASE_API", "data:application/json,invalid"):
                    check.trigger()
                    self.wait_for(check.isEnabled)
                self.assertIn("暂时无法检查更新", window.statusBar().currentMessage())
            finally:
                window.close()

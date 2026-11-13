"""Run the release check outside the GUI process."""

import logging
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from PySide6.QtCore import QObject, QProcess, QTimer, QUrl, Signal

from release_check import RELEASE_PAGE, WORKER_FLAG, version


TIMEOUT_MS = 12000
LOG = logging.getLogger(__name__)


class UpdateChecker(QObject):
    # An empty tag means the check failed. Only a newer release supplies a URL.
    check_finished = Signal(str, QUrl)

    def __init__(self, current_version, parent=None):
        super().__init__(parent)
        self._current_version = version(current_version)
        self._result_dir = None
        self._process = QProcess(self)
        self._process.finished.connect(self._finish)
        self._process.errorOccurred.connect(self._process_error)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._timeout)

    @staticmethod
    def _worker_arguments(result_file):
        arguments = [WORKER_FLAG, str(result_file)]
        if not getattr(sys, "frozen", False):
            arguments.insert(0, str(Path(__file__).resolve().parents[1] / "main.py"))
        return arguments

    def check(self):
        if self._result_dir is not None:
            return
        try:
            self._result_dir = TemporaryDirectory(prefix="filmclear-update-", ignore_cleanup_errors=True)
        except OSError:
            LOG.exception("Could not create a release check result directory")
            self.check_finished.emit("", QUrl())
            return
        LOG.info("Starting release check worker")
        self._timer.start(TIMEOUT_MS)
        result_file = Path(self._result_dir.name) / "tag"
        self._process.start(sys.executable, self._worker_arguments(result_file))

    def _timeout(self):
        LOG.warning("Release check worker timed out")
        self._process.kill()

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            LOG.warning("Could not start release check worker: %s", self._process.errorString())
            self._finish(-1, QProcess.ExitStatus.CrashExit)

    def _finish(self, exit_code, exit_status):
        if self._result_dir is None:
            return
        self._timer.stop()
        directory, self._result_dir = self._result_dir, None
        tag, page = "", QUrl()
        LOG.info("Release check worker exited: %s (%s)", exit_code, exit_status.name)
        with directory:
            if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0 and self._current_version:
                try:
                    candidate = (Path(directory.name) / "tag").read_text(encoding="ascii")
                except (OSError, UnicodeError):
                    candidate = ""
                candidate_version = version(candidate)
                if candidate_version is not None:
                    tag = candidate
                    if candidate_version > self._current_version:
                        page = QUrl(RELEASE_PAGE + tag)
        self.check_finished.emit(tag, page)

    def stop(self):
        if self._result_dir is not None:
            self._process.kill()
            self._process.waitForFinished(1000)

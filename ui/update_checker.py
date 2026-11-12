"""Check the latest public FilmClear release without blocking the workspace."""

import json
import re

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest


REPOSITORY = "JackerKun/FilmClear"
GITHUB_PROJECT = f"https://github.com/{REPOSITORY}"
RELEASE_API = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
RELEASE_PAGE = f"{GITHUB_PROJECT}/releases/tag/"
TIMEOUT_MS = 10000
VERSION_PATTERN = re.compile(r"v?([0-9]{1,9})\.([0-9]{1,9})\.([0-9]{1,9})")


def _version(text):
    match = VERSION_PATTERN.fullmatch(text) if isinstance(text, str) else None
    return tuple(map(int, match.groups())) if match else None


def _published_release_tag(body):
    try:
        release = json.loads(body)
    except ValueError:
        return None
    if not isinstance(release, dict) or release.get("draft") is not False or release.get("prerelease") is not False:
        return None
    tag = release.get("tag_name")
    return tag if _version(tag) is not None else None


class UpdateChecker(QObject):
    # An empty tag means the check failed. Only a newer release supplies a URL.
    check_finished = Signal(str, QUrl)

    def __init__(self, current_version, parent=None):
        super().__init__(parent)
        self._current_version = _version(current_version)
        self._reply = None
        self._network = QNetworkAccessManager(self)
        self._network.finished.connect(self._finish)

    def check(self):
        if self._reply is not None:
            return
        request = QNetworkRequest(QUrl(RELEASE_API))
        request.setRawHeader(b"Accept", b"application/vnd.github+json")
        request.setRawHeader(b"User-Agent", b"FilmClear")
        request.setAttribute(QNetworkRequest.RedirectPolicyAttribute, QNetworkRequest.SameOriginRedirectPolicy)
        reply = self._reply = self._network.get(request)
        timeout = QTimer(reply)
        timeout.setSingleShot(True)
        timeout.timeout.connect(reply.abort)
        reply.finished.connect(timeout.stop)
        timeout.start(TIMEOUT_MS)

    def _finish(self, reply):
        self._reply = None
        tag, page = "", QUrl()
        if reply.error() == QNetworkReply.NoError and self._current_version is not None:
            tag = _published_release_tag(bytes(reply.readAll())) or ""
            if tag and _version(tag) > self._current_version:
                page = QUrl(RELEASE_PAGE + tag)
        reply.deleteLater()
        self.check_finished.emit(tag, page)

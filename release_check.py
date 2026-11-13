"""Network code for the isolated GitHub release check."""

import json
from pathlib import Path
import re


REPOSITORY = "JackerKun/FilmClear"
GITHUB_PROJECT = f"https://github.com/{REPOSITORY}"
RELEASE_PAGE = f"{GITHUB_PROJECT}/releases/tag/"
WORKER_FLAG = "--filmclear-update-worker"
VERSION_PATTERN = re.compile(r"v?([0-9]{1,9})\.([0-9]{1,9})\.([0-9]{1,9})")
MAX_RESPONSE_BYTES = 1024 * 1024


def version(text):
    match = VERSION_PATTERN.fullmatch(text) if isinstance(text, str) else None
    return tuple(map(int, match.groups())) if match else None


def published_release_tag(body):
    try:
        release = json.loads(body)
    except ValueError:
        return None
    if not isinstance(release, dict) or release.get("draft") is not False or release.get("prerelease") is not False:
        return None
    tag = release.get("tag_name")
    return tag if version(tag) is not None else None


def fetch_latest_release(destination):
    """Write a validated tag for the parent process to read after a clean exit."""
    from http.client import HTTPException
    import ssl
    import sys
    from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, *args):
            return None

    request = Request(f"https://api.github.com/repos/{REPOSITORY}/releases/latest", headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "FilmClear",
    })
    try:
        context = ssl.create_default_context()
        # A python.org macOS build may have no OpenSSL CA bundle installed.
        if sys.platform == "darwin" and context.cert_store_stats()["x509_ca"] == 0:
            context.load_verify_locations("/etc/ssl/cert.pem")
        opener = build_opener(NoRedirect(), HTTPSHandler(context=context))
        with opener.open(request, timeout=8) as response:
            if response.status != 200:
                return 1
            body = response.read(MAX_RESPONSE_BYTES + 1)
        tag = published_release_tag(body) if len(body) <= MAX_RESPONSE_BYTES else None
        if not tag:
            return 1
        Path(destination).write_text(tag, encoding="ascii")
    except (OSError, HTTPException):
        return 3
    return 0

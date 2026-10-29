"""
Timeout guards for the third-party `reolinkapi` library.

`reolinkapi.handlers.rest_handler.Request.post()` issues `requests.post(...)`
with no timeout, and `get_file()` streams with `timeout=(1, None)`. A dead camera
connection therefore blocks the calling thread forever, which on the Pi froze
the single-threaded scheduler loop for days.
"""

import threading
from typing import Any, Callable, Tuple

import requests
from requests.adapters import HTTPAdapter


class _DefaultTimeoutAdapter(HTTPAdapter):
    """HTTPAdapter that applies a default (connect, read) timeout when none is given."""

    def __init__(self, default_timeout: Tuple[float, float], *args, **kwargs):
        self.default_timeout = default_timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self.default_timeout
        return super().send(request, **kwargs)


def make_timeout_session(connect: float = 10.0, read: float = 60.0) -> requests.Session:
    """Session for `reolinkapi.Camera(..., session=...)` so login/search calls can't hang."""
    session = requests.Session()
    adapter = _DefaultTimeoutAdapter((connect, read))
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def call_with_deadline(fn: Callable[..., Any], seconds: float, *args, **kwargs) -> Any:
    """
    Run fn and give up waiting after `seconds`, raising TimeoutError.

    For calls we can't pass a timeout into (e.g. `reolinkapi` `get_file()`). The
    worker is a daemon thread, so a truly stuck call is abandoned rather than
    joined; the caller moves on instead of blocking the scheduler.
    """
    box = {}

    def target():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = e

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"{getattr(fn, '__name__', 'call')} exceeded {seconds:.0f}s deadline")
    if "error" in box:
        raise box["error"]
    return box.get("value")

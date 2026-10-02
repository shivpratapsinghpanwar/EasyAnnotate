"""Request-size limits that act before a body is read.

A handler that counts bytes as it reads an ``UploadFile`` caps nothing:
Starlette has already parsed the multipart body -- spooling it to a temporary
file on disk -- before the handler runs. A 100 GB POST fills the disk first and
is refused afterwards. So the limit lives here, in front of the app, where the
only thing read is the header.
"""

from __future__ import annotations

import json
from typing import Any

#: Room for multipart framing around a file that is itself within the limit.
FRAMING_SLACK = 1024 * 1024

_BODY_METHODS = {b"POST", b"PUT", b"PATCH"}


class BodyLimit:
    """Refuse a request whose declared size exceeds ``limit`` -- before reading it.

    A body with no ``Content-Length`` (chunked) is refused too, with 411:
    without a declared length the only way to enforce a limit is to read the
    body, which is the problem this exists to avoid. Browsers, httpx and curl
    all declare the length of a file upload.
    """

    def __init__(self, app: Any, limit: int) -> None:
        self.app = app
        self.limit = int(limit) + FRAMING_SLACK

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"].encode() not in _BODY_METHODS:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is None:
            if headers.get(b"transfer-encoding", b"").lower() == b"chunked":
                await _refuse(send, 411, "send a Content-Length with uploads")
                return
        elif not declared.isdigit():
            await _refuse(send, 400, "Content-Length is not a number")
            return
        elif int(declared) > self.limit:
            mb = (self.limit - FRAMING_SLACK) / 1e6
            await _refuse(send, 413, f"upload exceeds the {mb:.0f} MB limit")
            return

        await self.app(scope, receive, send)


async def _refuse(send: Any, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                # The client may still be sending; tell it not to reuse a
                # connection with an unread body on it.
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})

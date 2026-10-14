"""The sidecar side of the protocol: read requests, write exactly one reply.

A tool author writes a handler object and calls :func:`serve`. Everything about
framing, correlation ids and error replies is handled here, so a tool cannot
get the part wrong that hangs the editor.

The single rule this module exists to guarantee:

    **every request produces exactly one reply, including when the handler
    raises, and including when the handler raises something that is not an
    Exception.**

The legacy SAM worker broke that rule -- it caught, printed, and looped -- and
the UI blocked forever on a read that would never return.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, BinaryIO, Protocol

from .protocol import ToolError, decode_message, encode_message, fail, ok


class Handler(Protocol):
    """What a tool implements. Every method may raise; none may hang."""

    def describe(self) -> list[dict[str, Any]]: ...
    def health(self, params: dict[str, Any]) -> dict[str, Any]: ...
    def prepare(self, params: dict[str, Any]) -> Any: ...
    def invoke(self, params: dict[str, Any]) -> Any: ...


def serve(handler: Handler, stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> int:
    """Run the request loop until stdin closes. Returns a process exit code."""
    source = stdin if stdin is not None else sys.stdin.buffer
    sink = stdout if stdout is not None else _private_stdout()

    for line in source:
        if not line.strip():
            continue

        # Parsing is inside its own guard because a malformed frame has no id
        # to reply against -- there is no one to answer, so we log and move on
        # rather than dying and taking the healthy calls with us.
        try:
            payload = decode_message(line)
        except ToolError as exc:
            print(f"[tool] dropping unparseable frame: {exc}", file=sys.stderr, flush=True)
            continue

        message_id = payload.get("id")
        if not isinstance(message_id, int):
            print("[tool] dropping frame with no id", file=sys.stderr, flush=True)
            continue

        try:
            result = _dispatch(handler, str(payload.get("method") or ""), payload.get("params") or {})
            reply = ok(message_id, result)
        except BaseException as exc:  # noqa: BLE001 -- the whole point is to always reply
            # BaseException, not Exception: a handler raising KeyboardInterrupt
            # or a bare SystemExit must still produce a reply, or the caller
            # waits for a message that is never coming.
            traceback.print_exc()
            reply = fail(message_id, exc if isinstance(exc, Exception) else ToolError(str(exc)))

        try:
            sink.write(encode_message(reply))
            sink.flush()
        except (BrokenPipeError, OSError):
            # The parent is gone; nothing useful is left to do.
            return 0

    return 0


def _private_stdout() -> BinaryIO:
    """Keep the real stdout for replies alone; everything else goes to stderr.

    Libraries print. Ultralytics writes its download progress bar -- 13 KB of
    carriage returns and no newline -- to stdout, and the next reply was
    glued to the end of it and lost: on a first text prompt the editor said
    "the tool exited". Measured, then fixed here for every library at once:
    Python-level prints and C-level writes to fd 1 both land on stderr,
    which the parent keeps as the tool's log.
    """
    sys.stdout.flush()
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return replies


def _dispatch(handler: Handler, method: str, params: dict[str, Any]) -> Any:
    if method == "describe":
        return handler.describe()
    if method == "health":
        return handler.health(params)
    if method == "prepare":
        handler.prepare(params)
        return {"prepared": params.get("image_id")}
    if method == "invoke":
        return handler.invoke(params)
    raise ToolError(f"unknown method {method!r}", code="unknown_method")

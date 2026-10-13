"""The contract between the editor and a heavy tool.

A "heavy tool" is anything that wants PyTorch and ideally a GPU -- SAM, a
detector running inference. The base package must never import torch, and must
never import a module that imports torch, because:

* ``import torch`` costs 3-8 seconds and roughly 500 MB of RSS. In-process,
  every user pays that at startup, including the ones who never use it.
* A CUDA out-of-memory or a native segfault takes down whatever process it
  happens in. In a sidecar that is the sidecar; the editor greys out one button
  and **the person keeps drawing boxes**.
* The torch x CUDA x numpy version matrix is exactly what makes ``pip install``
  fail. A separate process means a separate environment, which means our
  dependency resolver is never involved in it.

So the boundary is a process, always -- including on a laptop. This module
defines what crosses it, and nothing here imports anything heavier than the
standard library.

The wire is newline-delimited JSON, one object per line. The same messages
travel over a subprocess's stdio and over HTTP to a hosted GPU service, so the
two implementations differ only in transport.

    ->  {"id": 8, "method": "invoke",
         "params": {"tool": "sam2.box", "image_id": "8f3c", "box": [1,2,3,4]}}
    <-  {"id": 8, "ok": true, "result": {"shapes": [...], "ms": 48}}
    <-  {"id": 8, "ok": false, "error": {"code": "not_ready", "retry_after_ms": 800}}

**Every request gets exactly one reply, including when the handler raises.**
That rule has a test, because breaking it is precisely how the legacy SAM
script hung: its worker caught exceptions, printed them, and put nothing on the
queue, while the UI blocked forever on a read inside a mouse callback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

#: Bumped when a change would stop an older peer understanding a newer one.
PROTOCOL_VERSION = 1

#: How long each method may take before the caller gives up. Encoding an image
#: is genuinely slow; answering a click is not, and a slow one is a bug.
DEFAULT_TIMEOUTS = {
    "describe": 5.0,
    "health": 5.0,
    "prepare": 30.0,
    # A tool replies promptly or says not_ready -- except that a native model
    # load (torch.jit.load of CLIP's 338 MB) can hold the GIL for tens of
    # seconds and delay even that reply. Timing out kills and restarts the
    # sidecar, which would throw the loaded models away, so this is generous.
    "invoke": 45.0,
}


class ToolError(Exception):
    """A tool failed. Carries a machine-readable code so callers can branch."""

    def __init__(self, message: str, *, code: str = "error", retry_after_ms: int = 0) -> None:
        super().__init__(message)
        self.code = code
        self.retry_after_ms = retry_after_ms

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.retry_after_ms:
            out["retry_after_ms"] = self.retry_after_ms
        return out


class ToolUnavailable(ToolError):
    """The tool is not installed, or its process will not start.

    Not an error condition for the product: the editor simply does not show the
    button. Manual annotation is never blocked by a missing tool.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, code="unavailable")


class ToolNotReady(ToolError):
    """Correct request, but the tool needs a moment -- usually a cold encode.

    Distinct from a failure on purpose: the UI spins on this one click and
    retries, rather than showing an error the user can do nothing about.
    """

    def __init__(self, message: str, *, retry_after_ms: int = 500) -> None:
        super().__init__(message, code="not_ready", retry_after_ms=retry_after_ms)


@dataclass(frozen=True)
class ToolSpec:
    """What a tool is, described well enough for the UI to render it.

    ``inputs`` is a JSON-schema fragment, so the tools panel builds its controls
    generically -- the same "declare once, render everywhere" the operation
    registry already uses, applied to the second contract. Shipping a new tool
    means shipping a sidecar that advertises a spec; no front-end change.
    """

    name: str                      # 'sam2.box', 'detect.infer'
    title: str
    summary: str
    kind: str = "interactive"      # 'interactive' | 'batch'
    inputs: dict[str, Any] = field(default_factory=dict)
    version: str = "0"
    requires: tuple[str, ...] = ()  # 'gpu', 'prepare'

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "summary": self.summary,
            "kind": self.kind,
            "inputs": dict(self.inputs),
            "version": self.version,
            "requires": list(self.requires),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ToolSpec:
        if not isinstance(raw, dict) or not raw.get("name"):
            raise ToolError(f"a tool spec needs a name, got {raw!r}", code="bad_spec")
        return cls(
            name=str(raw["name"]),
            title=str(raw.get("title") or raw["name"]),
            summary=str(raw.get("summary") or ""),
            kind=str(raw.get("kind") or "interactive"),
            inputs=dict(raw.get("inputs") or {}),
            version=str(raw.get("version") or "0"),
            requires=tuple(str(r) for r in raw.get("requires") or ()),
        )


@dataclass(frozen=True)
class ToolHealth:
    """Whether a tool can be used right now, and why not if it cannot."""

    state: str                     # 'ready' | 'loading' | 'unavailable'
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state, "detail": self.detail}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ToolHealth:
        return cls(state=str(raw.get("state") or "unavailable"), detail=str(raw.get("detail") or ""))


READY = ToolHealth("ready")


class Tool(Protocol):
    """What every backend implements, whatever is behind it."""

    def describe(self) -> list[ToolSpec]: ...

    def health(self, name: str = "") -> ToolHealth: ...

    def prepare(self, name: str, image_id: str, source: Any) -> None:
        """Warm a tool for an image -- the encode-ahead hook.

        Cheap and idempotent by contract, because the editor calls it for the
        current image and the next one every time you navigate.
        """
        ...

    def invoke(self, name: str, request: dict[str, Any]) -> dict[str, Any]: ...


# ------------------------------------------------------------------ framing

def encode_message(payload: dict[str, Any]) -> bytes:
    """One JSON object, one line.

    ``ensure_ascii`` keeps the frame single-byte-safe regardless of what the
    pipe's encoding turns out to be on Windows, and the explicit separators
    guarantee no stray newline can appear inside a frame.
    """
    line = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return (line + "\n").encode("utf-8")


def decode_message(line: str | bytes) -> dict[str, Any]:
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    text = line.strip()
    if not text:
        raise ToolError("empty frame", code="protocol")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolError(f"unparseable frame: {text[:120]!r}", code="protocol") from exc
    if not isinstance(payload, dict):
        raise ToolError(f"a frame must be an object, got {type(payload).__name__}", code="protocol")
    return payload


def request(message_id: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "id": message_id, "method": method, "params": params or {}}


def ok(message_id: int, result: Any) -> dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "id": message_id, "ok": True, "result": result}


def fail(message_id: int, error: BaseException | ToolError) -> dict[str, Any]:
    detail = (
        error.as_dict()
        if isinstance(error, ToolError)
        else {"code": type(error).__name__, "message": str(error)}
    )
    return {"v": PROTOCOL_VERSION, "id": message_id, "ok": False, "error": detail}


def raise_for(payload: dict[str, Any]) -> Any:
    """Turn a reply into a result or an exception of the right class."""
    if payload.get("ok"):
        return payload.get("result")
    error = payload.get("error") or {}
    code = str(error.get("code") or "error")
    message = str(error.get("message") or "the tool failed without saying why")
    if code == "not_ready":
        raise ToolNotReady(message, retry_after_ms=int(error.get("retry_after_ms") or 500))
    if code == "unavailable":
        raise ToolUnavailable(message)
    raise ToolError(message, code=code, retry_after_ms=int(error.get("retry_after_ms") or 0))

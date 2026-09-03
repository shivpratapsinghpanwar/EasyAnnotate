"""Core value types shared by every Data Factory operation.

Nothing in this module prints, calls ``sys.exit`` or touches ``sys.argv``.
Operations return an :class:`OpResult`; adapters (CLI, web API, notebook)
decide how to render it.  That separation is what lets the same code back a
terminal command and an HTTP endpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class Action(str, Enum):
    """What an operation did (or would do) to a single file."""

    MOVE = "move"
    COPY = "copy"
    DELETE = "delete"
    RENAME = "rename"
    WRITE = "write"
    SKIP = "skip"


@dataclass(slots=True)
class FileAction:
    """One file-level change, recorded whether or not it was actually applied.

    In dry-run mode operations build the exact same list of ``FileAction``s and
    simply never hand them to :func:`data_factory.core.fs.apply`.  A preview is
    therefore a faithful description of the real run, not a separate code path.
    """

    action: Action
    src: Path
    dst: Path | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "src": str(self.src),
            "dst": str(self.dst) if self.dst else None,
            "reason": self.reason,
        }


@dataclass
class OpResult:
    """The structured return value of every operation.

    ``counts`` is the headline summary an adapter shows first, ``actions`` is
    the audit trail, and ``data`` carries op-specific payload (a label
    histogram, a written manifest path, a dataset report) that does not fit a
    plain integer.
    """

    op: str
    ok: bool = True
    dry_run: bool = False
    counts: dict[str, int] = field(default_factory=dict)
    actions: list[FileAction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def bump(self, key: str, by: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + by

    def record(self, action: Action, src: Path, dst: Path | None = None, reason: str = "") -> FileAction:
        fa = FileAction(action, src, dst, reason)
        self.actions.append(fa)
        return fa

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def as_dict(self, include_actions: bool = True) -> dict[str, Any]:
        """JSON-serialisable form.  This is what a web API returns verbatim."""
        out: dict[str, Any] = {
            "op": self.op,
            "ok": self.ok,
            "dry_run": self.dry_run,
            "counts": dict(self.counts),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "data": _jsonable(self.data),
        }
        if include_actions:
            out["actions"] = [a.as_dict() for a in self.actions]
        return out


def _jsonable(value: Any) -> Any:
    """Recursively convert Paths (and containers of them) to strings."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


class Progress(Protocol):
    """Callback shape for progress reporting.

    The CLI renders a bar, a web app pushes the numbers over a websocket, a
    test passes nothing at all.
    """

    def __call__(self, done: int, total: int, message: str = "") -> None: ...


class Cancelled(Exception):
    """Raised inside an operation when the caller asked it to stop."""


@dataclass
class Context:
    """Ambient execution settings threaded through every operation.

    ``cancel_check`` exists for the web case: a long dedup over 20k images has
    to be abortable from a browser, and polling a callable is the simplest way
    that works identically in-process and across a task queue.
    """

    dry_run: bool = False
    progress: Progress | None = None
    log: Callable[[str], None] | None = None
    cancel_check: Callable[[], bool] | None = None

    def emit(self, done: int, total: int, message: str = "") -> None:
        if self.progress is not None:
            self.progress(done, total, message)

    def note(self, message: str) -> None:
        if self.log is not None:
            self.log(message)

    def check_cancelled(self) -> None:
        if self.cancel_check is not None and self.cancel_check():
            raise Cancelled("operation cancelled by caller")

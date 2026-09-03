"""A heavy tool running as a child process, spoken to over stdio.

The legacy SAM script found this boundary by accident -- a ``multiprocessing``
worker holding two predictors -- and it was the best idea in either annotation
script. This formalises it, and fixes the three things that made the original
fragile:

* **A blocking read inside the UI callback.** Here a background thread owns the
  pipe and dispatches replies to futures, so no handler ever blocks on a raw
  read, and every wait has a timeout.
* **An untagged queue.** The original had one output queue and no correlation,
  so a stale reply could be consumed by the wrong caller and a real result
  silently dropped. Every message here carries an id and is matched on it.
* **An exception path that never replied.** The original caught, printed, and
  put nothing back, so the UI waited forever. Here the child always answers and
  the parent additionally fails every outstanding call if the child dies.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from .protocol import (
    DEFAULT_TIMEOUTS,
    ToolError,
    ToolHealth,
    ToolSpec,
    ToolUnavailable,
    decode_message,
    encode_message,
    raise_for,
    request,
)

#: Restarts before we stop trying. A tool that dies three times is broken, and
#: retrying forever would hide that while burning CPU.
MAX_RESTARTS = 3
RESTART_BACKOFF = (0.5, 2.0, 5.0)

#: Keep the tail of the child's stderr so a failure can say *why* rather than
#: just "it died" -- the difference between a fixable report and a shrug.
STDERR_LINES = 40


class SidecarClient:
    """Owns one child process and turns its stdio into method calls."""

    def __init__(
        self,
        argv: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeouts: dict[str, float] | None = None,
    ) -> None:
        self.argv = list(argv)
        self.cwd = cwd
        self.env = env
        self.timeouts = {**DEFAULT_TIMEOUTS, **(timeouts or {})}

        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._pending: dict[int, Future] = {}
        self._stderr: list[str] = []
        self._next_id = 0
        self._restarts = 0
        self._closed = False
        self._last_error = ""

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Spawn the child. Raises ToolUnavailable if it will not start."""
        with self._lock:
            if self._closed:
                raise ToolUnavailable("this client has been closed")
            if self._process is not None and self._process.poll() is None:
                return
            self._spawn_locked()

    def _spawn_locked(self) -> None:
        environment = {**os.environ, **(self.env or {})}
        # Unbuffered, or the child's replies sit in a pipe buffer and every
        # call looks like a timeout.
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            self._process = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.cwd) if self.cwd else None,
                env=environment,
                bufsize=0,
            )
        except (OSError, ValueError) as exc:
            self._last_error = str(exc)
            raise ToolUnavailable(f"cannot start {self.argv[0]!r}: {exc}") from None

        self._stderr = []
        self._reader = threading.Thread(target=self._read_stdout, args=(self._process,),
                                        daemon=True, name="dfx-tool-out")
        self._stderr_reader = threading.Thread(target=self._read_stderr, args=(self._process,),
                                               daemon=True, name="dfx-tool-err")
        self._reader.start()
        self._stderr_reader.start()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            process = self._process
            self._process = None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.wait(timeout=2)
        except Exception:
            process.kill()
        finally:
            self._fail_pending(ToolUnavailable("the tool was shut down"))

    @property
    def alive(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    # -- calling -----------------------------------------------------------
    def call(self, method: str, params: dict[str, Any] | None = None,
             *, timeout: float | None = None) -> Any:
        """Send one request and wait for its reply.

        Restarts the child once per call if it has died, so a crashed tool
        recovers on the next click rather than staying broken until a restart.
        """
        if self._closed:
            raise ToolUnavailable("this client has been closed")
        wait = timeout if timeout is not None else self.timeouts.get(method, 15.0)

        # Two attempts, because the child can die *between* the liveness check
        # and the write -- a crash on the previous call is reaped
        # asynchronously, so `poll()` can still say None for a moment while the
        # pipe is already gone. Without the retry the caller sees a confusing
        # "[Errno 22] Invalid argument" for a tool that is perfectly able to
        # restart and serve the request.
        last: Exception | None = None
        for _ in range(2):
            if not self.alive:
                self._restart()          # raises once restarts are exhausted

            message_id, future, payload = self._enqueue(method, params)
            try:
                self._write(payload)
            except ToolUnavailable as exc:
                self._drop(message_id)
                last = exc
                self._kill_child()       # force a fresh child on the next pass
                continue

            try:
                reply = future.result(timeout=wait)
            except TimeoutError:
                self._drop(message_id)
                # A tool that misses its deadline is not trustworthy
                # afterwards: whatever it is still doing is unknown, and a late
                # reply would arrive against an id nobody is waiting for.
                # Restarting is the honest recovery.
                self._kill_child()
                raise ToolError(
                    f"{method} took longer than {wait:g}s", code="timeout"
                ) from None
            return raise_for(reply)

        raise last or ToolUnavailable("the tool could not be reached")

    def _enqueue(self, method: str, params: dict[str, Any] | None):
        with self._lock:
            self._next_id += 1
            message_id = self._next_id
            future: Future = Future()
            self._pending[message_id] = future
        return message_id, future, request(message_id, method, params)

    def _drop(self, message_id: int) -> None:
        with self._lock:
            self._pending.pop(message_id, None)

    def _write(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise ToolUnavailable("the tool is not running")
        try:
            process.stdin.write(encode_message(payload))
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ToolUnavailable(f"lost the tool while writing: {exc}") from None

    # -- reading -----------------------------------------------------------
    def _read_stdout(self, process: subprocess.Popen[bytes]) -> None:
        stream = process.stdout
        assert stream is not None
        try:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    payload = decode_message(line)
                except ToolError:
                    # Junk on stdout is a tool that printed instead of
                    # replying. Ignore the line rather than killing the
                    # session; the pending call will time out and say so.
                    continue
                self._resolve(payload)
        except (ValueError, OSError):
            pass
        finally:
            # The child is gone. Nothing else will ever answer the calls still
            # waiting, so fail them now instead of letting each hit its
            # timeout one after another.
            self._fail_pending(
                ToolUnavailable(f"the tool exited{self._stderr_tail()}")
            )

    def _read_stderr(self, process: subprocess.Popen[bytes]) -> None:
        stream = process.stderr
        assert stream is not None
        try:
            for line in stream:
                text = line.decode("utf-8", errors="replace").rstrip()
                if not text:
                    continue
                self._stderr.append(text)
                del self._stderr[:-STDERR_LINES]
        except (ValueError, OSError):
            pass

    def _resolve(self, payload: dict[str, Any]) -> None:
        message_id = payload.get("id")
        if not isinstance(message_id, int):
            return
        with self._lock:
            future = self._pending.pop(message_id, None)
        # An unknown id is a late reply to something already abandoned. Drop
        # it: matching by id is exactly what stops it poisoning the next call.
        if future is not None and not future.done():
            future.set_result(payload)

    def _fail_pending(self, error: Exception) -> None:
        with self._lock:
            pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(error)

    def _stderr_tail(self) -> str:
        if not self._stderr:
            return ""
        return ": " + " / ".join(self._stderr[-3:])

    def _kill_child(self) -> None:
        with self._lock:
            process, self._process = self._process, None
        if process is not None:
            try:
                process.kill()
            except Exception:
                pass

    def _restart(self) -> None:
        if self._restarts >= MAX_RESTARTS:
            raise ToolUnavailable(
                f"the tool has died {self._restarts} times and will not be restarted"
                f"{self._stderr_tail()}"
            )
        delay = RESTART_BACKOFF[min(self._restarts, len(RESTART_BACKOFF) - 1)]
        self._restarts += 1
        if self._restarts > 1:
            time.sleep(delay)
        with self._lock:
            self._spawn_locked()


class LocalTool:
    """A :class:`Tool` backed by a sidecar process."""

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.client = SidecarClient(argv, **kwargs)
        self._specs: list[ToolSpec] | None = None

    def describe(self) -> list[ToolSpec]:
        if self._specs is None:
            raw = self.client.call("describe")
            self._specs = [ToolSpec.from_dict(item) for item in (raw or [])]
        return list(self._specs)

    def health(self, name: str = "") -> ToolHealth:
        try:
            return ToolHealth.from_dict(self.client.call("health", {"tool": name}) or {})
        except ToolError as exc:
            # Health must never raise: it is what the UI calls to decide
            # whether to show a button at all.
            return ToolHealth("unavailable", str(exc))

    def prepare(self, name: str, image_id: str, source: Any = None) -> None:
        self.client.call("prepare", {"tool": name, "image_id": image_id,
                                     "source": str(source) if source else None})

    def invoke(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.client.call("invoke", {"tool": name, **payload})

    def close(self) -> None:
        self.client.close()


def python_for_tools() -> str:
    """Which interpreter runs the sidecar.

    Deliberately ordered so an explicit choice always wins, and so a tools
    environment kept apart from the main one is found without configuration:

    1. ``DATAFACTORY_TOOLS_PYTHON``
    2. a conventional venv at ``~/.datafactory/tools``
    3. this interpreter -- correct when someone installed the extra into the
       same environment on purpose
    """
    explicit = os.environ.get("DATAFACTORY_TOOLS_PYTHON")
    if explicit:
        return explicit
    candidate = Path.home() / ".datafactory" / "tools"
    binary = candidate / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if binary.is_file():
        return str(binary)
    return sys.executable

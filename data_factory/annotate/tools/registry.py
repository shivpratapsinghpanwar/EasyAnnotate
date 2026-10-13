"""Which heavy tools exist right now, if any.

The rule this module enforces: **a heavy tool's absence is never an error.**
``available()`` returning an empty list is a normal, tested state — it is what
the base wheel does, and the editor must be fully usable in it. Manual
annotation is never blocked by a missing tool.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any

from .local import LocalTool, python_for_tools
from .protocol import Tool, ToolHealth, ToolNotReady, ToolSpec

#: Set to a module name to run that sidecar, e.g. ``data_factory.annotate.tools.fake``.
#: The real SAM tool ships in a separate distribution and names itself here.
TOOLS_MODULE_ENV = "DATAFACTORY_TOOLS_MODULE"

#: Set to "0" to force the base-wheel behaviour even where a tool is installed.
TOOLS_ENABLED_ENV = "DATAFACTORY_TOOLS"
#: What `easyannotate tools install` sets up. Used only when a tools
#: environment exists, so a plain install never spawns a doomed subprocess.
DEFAULT_TOOLS_MODULE = "easyannotate_tools.sidecar"


class NullTool:
    """The default backend: nothing installed, nothing offered."""

    def describe(self) -> list[ToolSpec]:
        return []

    def health(self, name: str = "") -> ToolHealth:
        return ToolHealth("unavailable", "no heavy tools are installed")

    def prepare(self, name: str, image_id: str, source: Any = None) -> None:
        return None

    def invoke(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        from .protocol import ToolUnavailable

        raise ToolUnavailable("no heavy tools are installed")

    def close(self) -> None:
        return None


class ToolRegistry:
    """Discovers a backend once and answers questions about it.

    Discovery is lazy and never raises: the first call decides, and a failure
    to start becomes an empty tool list plus a reason, not an exception in a
    request handler.
    """

    def __init__(self, backend: Tool | None = None) -> None:
        self._backend: Tool | None = backend
        self._resolved = backend is not None
        self._reason = ""

    # -- discovery ---------------------------------------------------------
    def backend(self) -> Tool:
        if not self._resolved:
            self._backend = self._discover()
            self._resolved = True
        assert self._backend is not None
        return self._backend

    def _discover(self) -> Tool:
        if os.environ.get(TOOLS_ENABLED_ENV, "1") in ("0", "false", "no"):
            self._reason = f"disabled by {TOOLS_ENABLED_ENV}"
            return NullTool()

        module = os.environ.get(TOOLS_MODULE_ENV, "").strip()
        interpreter = python_for_tools()
        if not module:
            if interpreter == sys.executable and not os.environ.get("DATAFACTORY_TOOLS_PYTHON"):
                self._reason = "no tools installed -- run: easyannotate tools install"
                return NullTool()
            module = DEFAULT_TOOLS_MODULE

        if not shutil.which(interpreter) and not os.path.isfile(interpreter):
            self._reason = f"interpreter not found: {interpreter}"
            return NullTool()

        tool = LocalTool([interpreter, "-m", module])
        try:
            tool.describe()          # the handshake: if this fails, we have nothing
        except Exception as exc:
            self._reason = f"{module} did not start: {exc}"
            try:
                tool.close()
            except Exception:
                pass
            return NullTool()
        self._reason = ""
        return tool

    # -- queries -----------------------------------------------------------
    def available(self) -> list[ToolSpec]:
        """Every usable tool. Empty is normal."""
        try:
            return self.backend().describe()
        except Exception as exc:
            self._reason = str(exc)
            return []

    def health(self, name: str = "") -> ToolHealth:
        try:
            return self.backend().health(name)
        except Exception as exc:
            return ToolHealth("unavailable", str(exc))

    def as_dict(self) -> dict[str, Any]:
        """What ``GET /api/tools`` returns.

        The UI renders a button only for a tool that is listed *and* healthy,
        so there are never dead buttons and never install nags in the toolbar.
        """
        specs = self.available()
        health = self.health() if specs else ToolHealth("unavailable", self._reason)
        return {
            "tools": [
                {**spec.as_dict(), "health": health.as_dict()}
                for spec in specs
                if health.ready
            ],
            "detail": self._reason or health.detail,
        }

    def prepare(self, name: str, image_id: str, source: Any = None) -> None:
        """Warm a tool for an image. Best-effort by contract.

        Never raises: this is called speculatively for the next image on every
        navigation, and a failed warm-up must not interrupt annotating.
        """
        try:
            self.backend().prepare(name, image_id, source)
        except Exception:
            return

    def invoke(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run a tool. Raises ToolError subclasses, which the router maps to
        HTTP status codes."""
        return self.backend().invoke(name, payload)

    def invoke_waiting(self, name: str, payload: dict[str, Any], *, budget_s: float = 300.0) -> dict[str, Any]:
        """Run a tool, waiting through not_ready replies -- a model loading,
        an image still being worked on -- for up to ``budget_s``. For batch
        work, where nobody is watching a single request. Other errors raise."""
        import time

        deadline = time.time() + budget_s
        while True:
            try:
                return self.backend().invoke(name, payload)
            except ToolNotReady as exc:
                if time.time() >= deadline:
                    raise
                time.sleep(max(0.2, (exc.retry_after_ms or 500) / 1000.0))

    def spec(self, name: str) -> ToolSpec | None:
        return next((s for s in self.available() if s.name == name), None)

    def close(self) -> None:
        backend = self._backend
        self._backend = None
        self._resolved = False
        if backend is not None and hasattr(backend, "close"):
            backend.close()


#: The process-wide registry the service uses.
REGISTRY = ToolRegistry()

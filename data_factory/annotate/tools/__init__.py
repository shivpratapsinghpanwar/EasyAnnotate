"""Heavy tools -- SAM, model inference -- behind a process boundary.

Nothing in this package imports torch, and nothing here imports a module that
does. That is the whole point: the base wheel stays installable in seconds, and
a crashing GPU tool takes down a child process rather than the editor.

See :mod:`data_factory.annotate.tools.protocol` for the contract, and
:mod:`data_factory.annotate.tools.fake` for a reference implementation you can
copy.
"""

from __future__ import annotations

from .protocol import (
    PROTOCOL_VERSION,
    Tool,
    ToolError,
    ToolHealth,
    ToolNotReady,
    ToolSpec,
    ToolUnavailable,
)
from .registry import REGISTRY, NullTool, ToolRegistry

__all__ = [
    "PROTOCOL_VERSION",
    "REGISTRY",
    "NullTool",
    "Tool",
    "ToolError",
    "ToolHealth",
    "ToolNotReady",
    "ToolRegistry",
    "ToolSpec",
    "ToolUnavailable",
]

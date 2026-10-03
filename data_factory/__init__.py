"""Data Factory -- computer-vision dataset preparation, as one library and one CLI.

Importing this package registers every operation and gives you the whole tool
through three names::

    from data_factory import get_operation, run, Context

    result = run("inspect.dataset", {"folder": "./data"})
    print(result.counts, result.data["issues"])

Every operation returns an :class:`~data_factory.core.types.OpResult`, which is
JSON-serialisable via ``result.as_dict()`` -- that is the whole contract a web
backend needs.
"""

from __future__ import annotations

from typing import Any


def _detect_version() -> str:
    """Read the version from installed package metadata.

    Declared once, in pyproject.toml. Hardcoding it here as well guarantees the
    two drift -- which they had, with the CLI reporting 0.2.0 from a 0.3.0
    wheel. The fallback covers running from a source checkout that was never
    installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    # Published as "easyannotate"; "data-factory" is the name it had before,
    # and an editable install made then still answers to it.
    for distribution in ("easyannotate", "data-factory"):
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    return "0.0.0+source"


__version__ = _detect_version()

from .core import ops as _ops  # noqa: F401  -- registers the operations
from .core import registry
from .core.registry import Operation, ParamError
from .core.types import Action, Cancelled, Context, FileAction, OpResult


def get_operation(name: str) -> Operation:
    """Look up one operation by name, e.g. ``clean.dedup``."""
    return registry.get(name)


def list_operations() -> list[Operation]:
    """Every registered operation, ordered by category then name."""
    return registry.all_operations()


def schema() -> dict[str, Any]:
    """Machine-readable description of every operation and its parameters."""
    return registry.schema()


def run(name: str, params: dict[str, Any], ctx: Context | None = None) -> OpResult:
    """Run an operation by name. The one-liner entry point for library users."""
    return registry.get(name).run(params, ctx)


__all__ = [
    "Action",
    "Cancelled",
    "Context",
    "FileAction",
    "OpResult",
    "Operation",
    "ParamError",
    "__version__",
    "get_operation",
    "list_operations",
    "run",
    "schema",
]

"""Terminal front-end: argparse commands, the interactive menu, and rendering.

``main`` is exposed lazily. Importing it eagerly would make
``python -m data_factory.cli.main`` load the module twice and warn about it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_parser", "main"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        # import_module, not `from . import main`: that asks this package for
        # an attribute "main" first -- which is this function again, forever.
        import importlib

        return getattr(importlib.import_module(f"{__name__}.main"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

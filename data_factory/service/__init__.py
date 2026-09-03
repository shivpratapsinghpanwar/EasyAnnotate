"""The hosted HTTP service.

Imported lazily so that ``import data_factory`` never requires FastAPI. Install
the extra to use it::

    pip install "data-factory[web]"
    uvicorn data_factory.service.app:app --port 8000
"""

from __future__ import annotations

from typing import Any

__all__ = ["app", "binding", "jobs", "workspace"]


def __getattr__(name: str) -> Any:
    if name == "app":
        from .app import app

        return app
    if name in ("binding", "jobs", "workspace"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

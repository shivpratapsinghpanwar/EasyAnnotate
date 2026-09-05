"""HTTP routers.

The batch API in ``app.py`` is request/response over folders. Annotation is
stateful CRUD with a latency budget, so it lives here instead of being forced
through the operation registry -- see ``data_factory/annotate/__init__.py`` for
why that boundary exists.
"""

from __future__ import annotations

__all__ = ["annotate", "media", "predictions", "promote", "tools"]

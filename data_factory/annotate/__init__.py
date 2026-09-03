"""Interactive annotation: projects, images, shapes, review.

This package deliberately **does not import fastapi**. The batch operations in
``core/`` answer to one contract -- parameters in, an OpResult out -- which
cannot express a polygon or a per-shape edit. Annotation is stateful CRUD with
sub-100ms latency budgets, so it gets its own contract: plain domain objects
here, HTTP routers in ``service/routers/``.

Keeping FastAPI out means this layer is testable without a TestClient, and the
CLI can drive the same store as the browser.
"""

from __future__ import annotations

__all__ = ["tools"]

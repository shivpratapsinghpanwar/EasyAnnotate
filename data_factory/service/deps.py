"""Shared request dependencies: the key gate, the store, and where files live.

Split out of ``app.py`` so the annotation routers can depend on the same gate
without importing the application module they are mounted into.

**Layout.** The dataset sandboxes live under the workspace root and are
transient -- swept on TTL, zipped for download, addressed by an opaque id.
Annotation projects are not transient, so nothing of theirs is stored in
there::

    ~/.datafactory/
        workspaces/        <- dataset sandboxes, swept, zipped, served
        annotations.db     <- the project database
        projects/<id>/     <- project media

A database file inside a sandbox would eventually be swept out from under a
user, packed into somebody's download, or served by a static route. Putting it
one level up costs nothing and removes all three.
"""

from __future__ import annotations

import os
import secrets
import threading
from pathlib import Path
from typing import Any

from fastapi import Header, HTTPException

from ..annotate.store import LOCAL_OWNER, Store
from .workspace import resolve_workspace_root

#: Read per call rather than captured at import, so a test (or a container
#: restart with new settings) does not need this module reloaded to see it.
def api_key() -> str:
    return os.environ.get("DATAFACTORY_API_KEY", "")


async def require_key(x_api_key: str = Header(default="")) -> None:
    """Optional shared-secret gate.

    Off by default so the tool runs locally with no ceremony. Set
    ``DATAFACTORY_API_KEY`` before exposing it to the internet -- without it,
    anyone can spend your CPU and disk.
    """
    key = api_key()
    if not key:
        return
    # isascii first: compare_digest raises on non-ASCII text -- a 500, not a 401.
    if not x_api_key.isascii() or not secrets.compare_digest(x_api_key, key):
        raise HTTPException(401, "invalid or missing X-API-Key")


_jobs: Any = None


def set_jobs_manager(manager: Any) -> None:
    """The app registers its JobManager here so routers can submit work
    without importing the app module (which imports the routers)."""
    global _jobs
    _jobs = manager


def jobs_manager() -> Any:
    return _jobs


def data_root() -> Path:
    """The directory holding the database and project media."""
    return resolve_workspace_root().parent


def store_path() -> Path:
    return data_root() / "annotations.db"


_stores: dict[Path, Store] = {}
_stores_lock = threading.Lock()


def get_store() -> Store:
    """The annotation store for the configured workspace.

    Cached *by path* rather than as a single global. Tests point
    ``DATAFACTORY_WORKSPACE`` somewhere new between cases, and a plain global
    would keep serving the first one they happened to create.
    """
    path = store_path()
    with _stores_lock:
        existing = _stores.get(path)
        if existing is None:
            existing = Store(path)
            _stores[path] = existing
        return existing


def reset_stores() -> None:
    """Drop cached connections. For tests and for a clean shutdown."""
    with _stores_lock:
        for store in _stores.values():
            store.close()
        _stores.clear()


async def principal() -> str:
    """Who is asking.

    Locally this is always the same answer, and that is the point: every query
    below is already scoped by owner, so Phase 5 replaces this one function
    with a real session lookup instead of revisiting every route.
    """
    return LOCAL_OWNER


def media_root(project_id: str) -> Path:
    """Where a managed project's images are kept."""
    return data_root() / "projects" / project_id


def resolve_media(root: Path, rel_path: str) -> Path:
    """Join a stored relative path onto a root, refusing anything that escapes.

    The client never sends this path -- it names an image id and the path comes
    out of the database. The check is here anyway: a row could have been
    written by an older build, and a symlink planted under the root would
    otherwise redirect the read.
    """
    base = root.resolve()
    target = (base / rel_path).resolve()
    if target != base and base not in target.parents:
        raise HTTPException(400, "that image is outside the project")
    return target

"""Projects: the unit of ownership, and the gate everything else passes.

There is one structural rule here worth stating plainly, because it is what
makes the multi-tenant story in Phase 5 additive rather than a rewrite:

    **Nothing below a project is reachable without first resolving the project
    for an owner.** ``labels``, ``images`` and ``shapes`` all take a
    :class:`Project`, and the only way to get one is :func:`get`, which filters
    by ``owner_id``.

That is deliberately stronger than "remember to add ``WHERE owner_id = ?`` to
every query". A forgotten WHERE clause is a data leak that no test will notice;
a missing function argument does not compile.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from .store import LOCAL_OWNER, NotFound, Store, StoreError, new_id, require_id

#: managed -- the database is the truth, sidecar files come out of export.
#: attached -- a folder the user opened; every save is written through to the
#: sidecar on disk, which is what "my XML gets updated" means.
STORAGE_MODES = ("managed", "attached")

RETENTIONS = ("keep", "ttl")


def attached_allowed() -> bool:
    """Attached storage is a local-only mode.

    On a host, a project pointing at a server-side folder would be exactly the
    arbitrary-path primitive the whole service layer exists to refuse. Hosted
    deployments set ``DATAFACTORY_ALLOW_ATTACHED=0``.
    """
    setting = os.environ.get("DATAFACTORY_ALLOW_ATTACHED", "1").strip().lower()
    return setting not in ("0", "false", "no", "off")


@dataclass(slots=True)
class Project:
    id: str
    owner_id: str
    name: str
    storage: str
    retention: str
    root: str
    next_label_idx: int
    created_at: float
    updated_at: float
    #: Which per-image sidecar format an attached project writes back to:
    #: 'voc', 'labelme', 'yolo', or '' for none. Empty for managed projects,
    #: and empty for attached folders in a collected format -- see attach.py.
    sidecar_format: str = ""

    @property
    def is_attached(self) -> bool:
        return self.storage == "attached"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "storage": self.storage,
            "retention": self.retention,
            "sidecar_format": self.sidecar_format,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _row_to_project(row: Any) -> Project:
    return Project(
        id=row["id"],
        owner_id=row["owner_id"],
        name=row["name"],
        storage=row["storage"],
        retention=row["retention"],
        root=row["root"],
        next_label_idx=row["next_label_idx"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        sidecar_format=row["sidecar_format"],
    )


def create(
    store: Store,
    name: str,
    *,
    owner_id: str = LOCAL_OWNER,
    storage: str = "managed",
    retention: str = "keep",
    root: str = "",
    sidecar_format: str = "",
) -> Project:
    """Start a project. No labels, no schema step -- that is the whole point."""
    clean = (name or "").strip() or "Untitled project"
    if storage not in STORAGE_MODES:
        raise StoreError(f"storage must be one of {', '.join(STORAGE_MODES)}")
    if retention not in RETENTIONS:
        raise StoreError(f"retention must be one of {', '.join(RETENTIONS)}")
    if storage == "attached" and not attached_allowed():
        raise StoreError("attached projects are disabled on this deployment")

    now = time.time()
    project = Project(
        id=new_id(),
        owner_id=owner_id,
        name=clean,
        storage=storage,
        retention=retention,
        root=root if storage == "attached" else "",
        next_label_idx=0,
        created_at=now,
        updated_at=now,
        sidecar_format=sidecar_format if storage == "attached" else "",
    )
    with store.write() as conn:
        conn.execute(
            "INSERT INTO project (id, owner_id, name, storage, retention, root,"
            " next_label_idx, created_at, updated_at, sidecar_format)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                project.id,
                project.owner_id,
                project.name,
                project.storage,
                project.retention,
                project.root,
                0,
                now,
                now,
                project.sidecar_format,
            ),
        )
    return project


def get(store: Store, project_id: str, owner_id: str = LOCAL_OWNER) -> Project:
    """Resolve a project *for this owner*, or raise :class:`NotFound`."""
    require_id(project_id, "project id")
    row = store.one(
        "SELECT * FROM project WHERE id = ? AND owner_id = ?", (project_id, owner_id)
    )
    if row is None:
        raise NotFound("no such project")
    return _row_to_project(row)


def reload(store: Store, project: Project) -> Project:
    """Re-read a project, picking up a moved label counter."""
    return get(store, project.id, project.owner_id)


def listing(store: Store, owner_id: str = LOCAL_OWNER) -> list[Project]:
    rows = store.query(
        "SELECT * FROM project WHERE owner_id = ? ORDER BY updated_at DESC", (owner_id,)
    )
    return [_row_to_project(r) for r in rows]


def rename(store: Store, project: Project, name: str) -> Project:
    clean = (name or "").strip()
    if not clean:
        raise StoreError("a project needs a name")
    with store.write() as conn:
        conn.execute(
            "UPDATE project SET name = ?, updated_at = ? WHERE id = ? AND owner_id = ?",
            (clean, time.time(), project.id, project.owner_id),
        )
    return reload(store, project)


def delete(store: Store, project: Project) -> None:
    """Remove a project and everything under it.

    The cascade is the schema's job (``ON DELETE CASCADE`` plus the
    ``foreign_keys`` pragma), not a hand-written sequence of DELETEs that a
    future table would be forgotten from.
    """
    with store.write() as conn:
        conn.execute(
            "DELETE FROM project WHERE id = ? AND owner_id = ?",
            (project.id, project.owner_id),
        )


def touch(conn: Any, project_id: str) -> None:
    """Mark a project as worked on. Called inside an existing transaction.

    Project mtime is a database column and not the root directory's, because a
    deep write does not bump a parent's mtime -- which is how a project someone
    annotated for a day ends up looking untouched to a sweep.
    """
    conn.execute("UPDATE project SET updated_at = ? WHERE id = ?", (time.time(), project_id))


def audit(conn: Any, project: Project, action: str, detail: str = "") -> None:
    conn.execute(
        "INSERT INTO audit (project_id, owner_id, action, detail, at) VALUES (?, ?, ?, ?, ?)",
        (project.id, project.owner_id, action, detail, time.time()),
    )

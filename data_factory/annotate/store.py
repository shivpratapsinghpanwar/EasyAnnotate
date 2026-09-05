"""The annotation database: one SQLite file, opened through a thin ``Store``.

**Why SQLite.** Files-only has no place to put a class order, a project name or
a review state, and "which images are still unreviewed" degrades into a tree
walk. Postgres cannot be a dependency of a thing installed with ``pip`` and run
on a laptop. SQLite is in the standard library, is ACID, and in WAL mode gives
one writer with many concurrent readers -- which is exactly the shape of
annotation traffic. The same schema then runs locally and hosted, which is what
lets one core serve both tiers.

**Why plain ``sqlite3``.** No ORM. ``Store`` is already the seam: if the hosted
tier ever needs Postgres, it is this class that changes and nothing above it.

**Threads.** FastAPI runs synchronous handlers in a worker threadpool, and a
``sqlite3`` connection must not cross threads. Each thread therefore gets its
own connection, created on first use. WAL is what makes that cheap.

**Migrations.** ``PRAGMA user_version`` is a ladder, not a framework: version 0
is an empty file, and each rung is applied in order inside one transaction. An
older build opening a newer database refuses rather than guessing.
"""

from __future__ import annotations

import re
import secrets
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

#: Bumped when a rung is added to ``MIGRATIONS``.
SCHEMA_VERSION = 3

#: The version-1 DDL lives beside this module rather than in a string, so it can
#: be read, diffed and run through a SQLite client directly.
SCHEMA_FILE = Path(__file__).with_name("schema.sql")

#: Ids are opaque tokens, not row numbers. An id that a client holds is an id a
#: client can try to guess, and sequential integers are trivially enumerable.
ID_BYTES = 12
ID_PATTERN = re.compile(r"^[0-9a-f]{24}$")

#: The principal used when nobody has logged in -- the entire local story.
#: Every query is scoped by owner from day one; locally the owner is just
#: always this. Retrofitting the scoping in Phase 5 would be a rewrite.
LOCAL_OWNER = "local"


class StoreError(Exception):
    """Raised for anything a caller did wrong with the store."""


class NotFound(StoreError):
    """The row does not exist, or does not belong to this owner.

    Deliberately one exception for both cases. Distinguishing "no such project"
    from "not yours" tells a stranger which project ids are real.
    """


class Conflict(StoreError):
    """The row changed underneath a save that was based on an older version."""


def new_id() -> str:
    return secrets.token_hex(ID_BYTES)


def valid_id(value: str) -> bool:
    return bool(ID_PATTERN.match(value or ""))


def require_id(value: str, what: str = "id") -> str:
    """Reject anything that is not one of our ids before it reaches a query.

    Parameterised SQL already stops injection; this stops the subtler thing --
    an id like ``../..`` reaching a path built from it further downstream.
    """
    if not valid_id(value):
        raise NotFound(f"not a valid {what}")
    return value


class Store:
    """A connection to one annotation database, migrated on open."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._migrate()

    # -- connections -------------------------------------------------------
    @property
    def connection(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use."""
        existing = getattr(self._local, "conn", None)
        if existing is not None:
            return existing
        conn = sqlite3.connect(self.path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        # WAL: readers never block the writer and the writer never blocks
        # readers. Without it the editor's polling would stall every save.
        conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL is the documented safe pairing with WAL: durable across a
        # process crash, which is the failure we actually care about.
        conn.execute("PRAGMA synchronous=NORMAL")
        # Off by default in sqlite3, and the composite shape -> label
        # reference is worth nothing unenforced.
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        self._local.conn = conn
        return conn

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """A write transaction. Commits on success, rolls back on anything else.

        The lock serialises writers *in this process* so two threads racing on
        the same file get a Python-level wait instead of a SQLITE_BUSY that
        surfaces to a user as a failed save.
        """
        with self._write_lock:
            conn = self.connection
            try:
                with conn:
                    yield conn
            except sqlite3.IntegrityError as exc:
                raise StoreError(str(exc)) from exc

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self.connection.execute(sql, params))

    def one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        return self.connection.execute(sql, params).fetchone()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- migration ---------------------------------------------------------
    def _migrate(self) -> None:
        conn = self.connection
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise StoreError(
                f"{self.path.name} is schema version {version}; this build "
                f"understands up to {SCHEMA_VERSION}. Upgrade Data Factory to "
                f"open it. (Refusing rather than half-reading it.)"
            )
        if version == SCHEMA_VERSION:
            return
        with self._write_lock:
            with conn:
                for rung in range(version, SCHEMA_VERSION):
                    MIGRATIONS[rung](conn)
                # PRAGMA does not take a parameter, and the value is an int
                # constant of ours -- never anything from outside.
                conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def _to_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))


def _to_v2(conn: sqlite3.Connection) -> None:
    """Record which sidecar format an attached project writes back to.

    Deliberately *not* added to ``schema.sql``. Keeping the base script at
    version 1 means every fresh database climbs the whole ladder, so each
    migration is exercised on every new install rather than only on the one
    upgrade nobody tested.
    """
    conn.execute(
        "ALTER TABLE project ADD COLUMN sidecar_format TEXT NOT NULL DEFAULT ''"
    )


def _to_v3(conn: sqlite3.Connection) -> None:
    """Remember how many shapes a model predicted for each image.

    Without it the agreement metric is quietly wrong in the most common case.
    A person reviewing predictions mostly *deletes* false positives, and a
    deleted row leaves no trace -- so agreement computed from surviving shapes
    would count every deletion as though the prediction had never been made,
    and a model that hallucinated ten boxes an image would score 100%.
    """
    conn.execute("ALTER TABLE image ADD COLUMN predicted_count INTEGER NOT NULL DEFAULT 0")


#: Rung *n* migrates a database at version *n* to version *n+1*.
MIGRATIONS = [_to_v1, _to_v2, _to_v3]


def open_store(root: Path | str) -> Store:
    """Open the annotation database for a workspace root.

    Placed *beside* the dataset sandboxes, never inside one -- a database file
    inside a sandbox would be swept on TTL, zipped into a user's download, and
    served by the static routes.
    """
    return Store(Path(root).parent / "annotations.db")

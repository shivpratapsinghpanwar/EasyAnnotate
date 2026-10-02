"""Images within a project: the list, the ordering, and the resume point.

The editor asks this module three questions constantly -- what images are
there, which one comes next, and how much is left -- so all three are indexed
queries rather than directory walks. That is the reason the image list is
database-backed at all: ``fs.iter_images`` over 20,000 files is fine once at
import and hopeless as a per-request operation.

Paths stored here are POSIX-relative to the project root, never absolute. A row
holding an absolute path is a row that could eventually be joined onto
something a client influences, and it makes a database written on Windows
useless on Linux.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from .projects import Project, touch
from .store import NotFound, Store, StoreError, new_id, require_id

T = TypeVar("T")

#: todo        nothing has happened to it yet
#: predicted   a model wrote shapes onto it, nobody has looked
#: in_progress someone has started
#: reviewed    a person has signed it off -- the only state export trusts
#: skipped     an explicit "I cannot tell", which must never become training data
STATES = ("todo", "predicted", "in_progress", "reviewed", "skipped")

#: States that represent finished human judgement.
DONE_STATES = ("reviewed",)


@dataclass(slots=True)
class Image:
    id: str
    project_id: str
    rel_path: str
    width: int
    height: int
    seq: int
    state: str
    version: int
    created_at: float
    updated_at: float

    @property
    def name(self) -> str:
        return PurePosixPath(self.rel_path).name

    @property
    def stem(self) -> str:
        return PurePosixPath(self.rel_path).stem

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.rel_path,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "seq": self.seq,
            "state": self.state,
            "version": self.version,
        }


def _row(row: Any) -> Image:
    return Image(
        id=row["id"],
        project_id=row["project_id"],
        rel_path=row["rel_path"],
        width=row["width"],
        height=row["height"],
        seq=row["seq"],
        state=row["state"],
        version=row["version"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def clean_rel_path(raw: str) -> str:
    """Normalise a relative path, refusing anything that could escape.

    Separators are normalised because a Windows scan and a Linux scan of the
    same folder should produce the same row. Escapes are *refused* rather than
    stripped -- consistent with the service layer, where quietly sanitising a
    hostile path mostly teaches the sender what to try next.
    """
    text = (raw or "").strip().replace("\\", "/")
    if not text:
        raise StoreError("an image needs a path")
    if text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        raise StoreError(f"image paths must be relative, got {raw!r}")
    parts = PurePosixPath(text).parts
    if any(part == ".." for part in parts):
        raise StoreError(f"image path escapes the project: {raw!r}")
    kept = [p for p in parts if p != "."]
    if not kept:
        raise StoreError("an image needs a path")
    return PurePosixPath(*kept).as_posix()


def _check_state(state: str) -> str:
    if state not in STATES:
        raise StoreError(f"state must be one of {', '.join(STATES)}")
    return state


def add(
    store: Store,
    project: Project,
    rel_path: str,
    *,
    width: int = 0,
    height: int = 0,
    state: str = "todo",
) -> Image:
    """Register one image. Idempotent on path, so a re-scan is harmless."""
    return add_many(store, project, [(rel_path, width, height)], state=state)[0]


def add_many(
    store: Store,
    project: Project,
    entries: Iterable[tuple[str, int, int]],
    *,
    state: str = "todo",
) -> list[Image]:
    """Bulk register. One transaction, because 20,000 commits is a minute.

    ``seq`` is assigned by insertion order and continues from whatever is
    already there, which keeps a second import appended rather than shuffled
    into the middle of someone's half-finished pass.
    """
    _check_state(state)
    cleaned = [(clean_rel_path(p), int(w), int(h)) for p, w, h in entries]
    now = time.time()
    made: list[Image] = []

    with store.write() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) AS top FROM image WHERE project_id = ?",
            (project.id,),
        ).fetchone()
        seq = int(row["top"]) + 1

        for rel_path, width, height in cleaned:
            existing = conn.execute(
                "SELECT * FROM image WHERE project_id = ? AND rel_path = ?",
                (project.id, rel_path),
            ).fetchone()
            if existing is not None:
                # A re-scan may have learned the real dimensions; take those,
                # but never touch state or version -- that is someone's work.
                if width and height and (
                    existing["width"] != width or existing["height"] != height
                ):
                    conn.execute(
                        "UPDATE image SET width = ?, height = ? WHERE id = ?",
                        (width, height, existing["id"]),
                    )
                    existing = conn.execute(
                        "SELECT * FROM image WHERE id = ?", (existing["id"],)
                    ).fetchone()
                made.append(_row(existing))
                continue

            image_id = new_id()
            conn.execute(
                "INSERT INTO image (id, project_id, rel_path, width, height, seq,"
                " state, version, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                (image_id, project.id, rel_path, width, height, seq, state, now, now),
            )
            made.append(
                Image(image_id, project.id, rel_path, width, height, seq, state, 0, now, now)
            )
            seq += 1
        touch(conn, project.id)

    return made


def get(store: Store, project: Project, image_id: str) -> Image:
    require_id(image_id, "image id")
    row = store.one(
        "SELECT * FROM image WHERE id = ? AND project_id = ?", (image_id, project.id)
    )
    if row is None:
        raise NotFound("no such image")
    return _row(row)


def by_path(store: Store, project: Project, rel_path: str) -> Image | None:
    row = store.one(
        "SELECT * FROM image WHERE project_id = ? AND rel_path = ?",
        (project.id, clean_rel_path(rel_path)),
    )
    return _row(row) if row else None


def listing_all(store: Store, project: Project) -> list[Image]:
    """Every image in the project, in order. For import-time matching."""
    return [
        _row(row)
        for row in store.query(
            "SELECT * FROM image WHERE project_id = ? ORDER BY seq", (project.id,)
        )
    ]


class Matcher:
    """Which project image does a label file belong to?

    By path first, and by bare filename stem only when that stem settles it.
    ``train/001.jpg`` and ``val/001.jpg`` share a stem; matching on it alone
    gave both images whichever label came first, and in attached mode the
    next save wrote that image's boxes into the other one's file on disk.

    When several images share the stem, the one sharing the most trailing
    folders wins -- so ``labels/val/001.txt`` still finds
    ``images/val/001.jpg`` -- and a tie is *ambiguous*: recorded in
    :attr:`ambiguous` for the caller to report, never guessed.
    """

    def __init__(self, images: Iterable[Image]) -> None:
        self._by_stem: dict[str, list[Image]] = {}
        for image in images:
            self._by_stem.setdefault(image.stem, []).append(image)
        self.ambiguous: set[str] = set()

    def find(self, name: str) -> Image | None:
        """``name`` is a bare filename or a relative path, extension optional."""
        parts = _match_parts(name)
        if not parts:
            return None
        candidates = self._by_stem.get(parts[-1], [])
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        scored = sorted(
            ((_shared_tail(parts, _match_parts(c.rel_path)), c) for c in candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )
        best, runner_up = scored[0][0], scored[1][0]
        if best > 1 and best > runner_up:
            return scored[0][1]
        self.ambiguous.add(name)
        return None


def assign(images: Iterable[Image], named: Iterable[tuple[str, T]]) -> tuple[dict[str, T], list[str]]:
    """Pair each ``(name, value)`` with the image it belongs to.

    Returns ``{image id: value}`` and the names that could not be placed
    safely -- ambiguous, or a second claim on an image already taken. Those
    are for the caller to report; placing them anyway is how one image's
    labels end up on another.
    """
    matcher = Matcher(images)
    placed: dict[str, T] = {}
    unsure: list[str] = []
    for name, value in named:
        image = matcher.find(name)
        if image is None:
            if name in matcher.ambiguous:
                unsure.append(name)
            continue
        if image.id in placed:
            unsure.append(name)
            continue
        placed[image.id] = value
    return placed, unsure


def item_name(item: Any, root: Path) -> str:
    """The best name a dataset item has for matching: its image's path
    relative to ``root`` when it has one, else the filename it records."""
    image = getattr(item, "image", None)
    if image is not None:
        try:
            return Path(image).resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return item.annotation.filename


def _match_parts(name: str) -> list[str]:
    path = PurePosixPath(str(name).replace("\\", "/"))
    parts = [p for p in path.parts if p not in ("", ".", "/")]
    if parts:
        parts[-1] = PurePosixPath(parts[-1]).stem
    return parts


def _shared_tail(a: list[str], b: list[str]) -> int:
    count = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        count += 1
    return count


def listing(
    store: Store,
    project: Project,
    *,
    state: str = "",
    limit: int = 500,
    offset: int = 0,
) -> list[Image]:
    sql = "SELECT * FROM image WHERE project_id = ?"
    params: list[Any] = [project.id]
    if state:
        sql += " AND state = ?"
        params.append(_check_state(state))
    sql += " ORDER BY seq LIMIT ? OFFSET ?"
    params.extend([max(1, min(int(limit), 5000)), max(0, int(offset))])
    return [_row(r) for r in store.query(sql, tuple(params))]


def resume_point(store: Store, project: Project) -> Image | None:
    """The image to open when a project is opened: the first unfinished one.

    Resume is the default, not a feature behind a button. Closing the browser
    at image 1000 and coming back to image 1 is why people abandon annotation
    tools.
    """
    row = store.one(
        "SELECT * FROM image WHERE project_id = ? AND state NOT IN ('reviewed', 'skipped')"
        " ORDER BY seq LIMIT 1",
        (project.id,),
    )
    if row is None:
        # Everything is done; land on the first image rather than nowhere.
        row = store.one(
            "SELECT * FROM image WHERE project_id = ? ORDER BY seq LIMIT 1", (project.id,)
        )
    return _row(row) if row else None


def neighbour(store: Store, project: Project, image: Image, step: int = 1) -> Image | None:
    """The next or previous image by sequence."""
    if step >= 0:
        row = store.one(
            "SELECT * FROM image WHERE project_id = ? AND seq > ? ORDER BY seq LIMIT 1",
            (project.id, image.seq),
        )
    else:
        row = store.one(
            "SELECT * FROM image WHERE project_id = ? AND seq < ? ORDER BY seq DESC LIMIT 1",
            (project.id, image.seq),
        )
    return _row(row) if row else None


def set_state(store: Store, project: Project, image: Image, state: str) -> Image:
    _check_state(state)
    now = time.time()
    with store.write() as conn:
        conn.execute(
            "UPDATE image SET state = ?, updated_at = ? WHERE id = ? AND project_id = ?",
            (state, now, image.id, project.id),
        )
        touch(conn, project.id)
    image.state = state
    image.updated_at = now
    return image


def set_size(store: Store, project: Project, image: Image, width: int, height: int) -> Image:
    with store.write() as conn:
        conn.execute(
            "UPDATE image SET width = ?, height = ? WHERE id = ? AND project_id = ?",
            (int(width), int(height), image.id, project.id),
        )
    image.width, image.height = int(width), int(height)
    return image


def counts(store: Store, project: Project) -> dict[str, int]:
    """How much is left, by state. The number the progress bar shows."""
    out = {state: 0 for state in STATES}
    for row in store.query(
        "SELECT state, COUNT(*) AS n FROM image WHERE project_id = ? GROUP BY state",
        (project.id,),
    ):
        out[str(row["state"])] = int(row["n"])
    out["total"] = sum(out[s] for s in STATES)
    return out

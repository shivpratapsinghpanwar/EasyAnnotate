"""The ordered label table -- and the correctness bug it exists to fix.

A YOLO label file stores a class *index*, never a name. The exporter today
builds that index table from ``sorted(found)`` (``core/annotations/formats.py``,
``Dataset.labels``), so the mapping is a function of *which class names happen
to be present at export time*. Add one class mid-project and every index at or
after it shifts by one:

    Monday    classes = [Closed, Open]         a file reading `1` means Open
    Friday    classes = [Closed, Handle, Open] the same file now means Handle

Nothing errors. The ``.txt`` files exported on Monday are still on disk, still
readable, and now silently mean something else. A model trained on the mixture
learns nonsense, and the only symptom is that accuracy is worse than it should
be. That is why this table is the first thing built in Phase 1.

Three rules make the index trustworthy:

* **Allocation is monotonic.** Indices come from a per-project counter, not
  from ``max(idx) + 1``, so retiring the newest label and adding another does
  not hand out the retired index.
* **Renaming does not renumber.** A label is an index with a name attached, not
  a name with an index derived from it. Fixing a typo is safe.
* **Retiring does not compact.** A retired label keeps its index forever and
  still occupies its slot in the exported class list, so the indices around it
  do not move.

The counterpart to this module is :func:`class_list`, which is what an export
must use instead of sorting the names it happens to find.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .projects import Project, touch
from .store import NotFound, Store, StoreError

#: Distinct, readable on both light and dark canvas, and safe against the two
#: common colour-vision deficiencies at the top of the list.
#:
#: These are ``#rrggbb``. Do **not** reach for ``core/ops/visualize.PALETTE``
#: here: that one is **BGR**, because OpenCV, and using it in CSS swaps red and
#: blue -- a bug that looks like a design choice and so survives review.
PALETTE = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffe119",
)

#: Long enough for a sentence-ish class name, short enough that a paste
#: accident cannot put a kilobyte in the palette.
MAX_NAME = 120


@dataclass(slots=True)
class Label:
    project_id: str
    idx: int
    name: str
    colour: str
    created_at: float
    retired_at: float | None = None

    @property
    def is_live(self) -> bool:
        return self.retired_at is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "name": self.name,
            "colour": self.colour,
            "retired": not self.is_live,
        }


def _row(row: Any) -> Label:
    return Label(
        project_id=row["project_id"],
        idx=row["idx"],
        name=row["name"],
        colour=row["colour"],
        created_at=row["created_at"],
        retired_at=row["retired_at"],
    )


def clean_name(name: str) -> str:
    """Normalise a name typed into the canvas.

    Whitespace is collapsed rather than merely stripped, because "front door"
    and "front  door" arriving as two classes is a real and very annoying way
    to split a dataset in half.
    """
    text = " ".join((name or "").split())
    if not text:
        raise StoreError("a label needs a name")
    if len(text) > MAX_NAME:
        raise StoreError(f"label names are limited to {MAX_NAME} characters")
    return text


def colour_for(idx: int) -> str:
    return PALETTE[idx % len(PALETTE)]


def listing(store: Store, project: Project, *, include_retired: bool = False) -> list[Label]:
    """Every label, in index order. Index order *is* the class order."""
    sql = "SELECT * FROM label WHERE project_id = ?"
    if not include_retired:
        sql += " AND retired_at IS NULL"
    sql += " ORDER BY idx"
    return [_row(r) for r in store.query(sql, (project.id,))]


def get(store: Store, project: Project, idx: int) -> Label:
    row = store.one(
        "SELECT * FROM label WHERE project_id = ? AND idx = ?", (project.id, int(idx))
    )
    if row is None:
        raise NotFound(f"no label with index {idx}")
    return _row(row)


def by_name(store: Store, project: Project, name: str) -> Label | None:
    """Find a *live* label by name. Retired names do not match."""
    row = store.one(
        "SELECT * FROM label WHERE project_id = ? AND name = ? AND retired_at IS NULL",
        (project.id, clean_name(name)),
    )
    return _row(row) if row else None


def ensure(store: Store, project: Project, name: str, *, colour: str = "") -> Label:
    """Return the label with this name, creating it if it is new.

    This is the "draw a box, type a name, press Enter" path, and it is the
    reason there is no schema step in this product. It is idempotent, so the
    editor can call it on every save without checking first.
    """
    clean = clean_name(name)
    existing = by_name(store, project, clean)
    if existing is not None:
        return existing

    now = time.time()
    with store.write() as conn:
        # Re-read the counter inside the transaction. Two browser tabs both
        # creating a label at once would otherwise both see the same
        # next_label_idx and collide on the primary key.
        row = conn.execute(
            "SELECT next_label_idx FROM project WHERE id = ?", (project.id,)
        ).fetchone()
        if row is None:
            raise NotFound("no such project")
        idx = int(row["next_label_idx"])
        conn.execute(
            "INSERT INTO label (project_id, idx, name, colour, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (project.id, idx, clean, colour or colour_for(idx), now),
        )
        conn.execute(
            "UPDATE project SET next_label_idx = ? WHERE id = ?", (idx + 1, project.id)
        )
        touch(conn, project.id)

    project.next_label_idx = idx + 1
    return Label(
        project_id=project.id,
        idx=idx,
        name=clean,
        colour=colour or colour_for(idx),
        created_at=now,
    )


def rename(store: Store, project: Project, idx: int, name: str) -> Label:
    """Change a label's name. **The index does not move.**

    Every ``.txt`` already exported keeps meaning what it meant; only the
    human-readable name in ``data.yaml`` changes. Fixing "Cosed" to "Closed"
    must never be a data-corrupting act.
    """
    clean = clean_name(name)
    label = get(store, project, idx)
    clash = by_name(store, project, clean)
    if clash is not None and clash.idx != label.idx:
        raise StoreError(f"another label is already called {clean!r}")
    with store.write() as conn:
        conn.execute(
            "UPDATE label SET name = ? WHERE project_id = ? AND idx = ?",
            (clean, project.id, int(idx)),
        )
        touch(conn, project.id)
    label.name = clean
    return label


def recolour(store: Store, project: Project, idx: int, colour: str) -> Label:
    text = (colour or "").strip().lower()
    if not (len(text) == 7 and text[0] == "#" and all(c in "0123456789abcdef" for c in text[1:])):
        raise StoreError("colour must look like #rrggbb")
    label = get(store, project, idx)
    with store.write() as conn:
        conn.execute(
            "UPDATE label SET colour = ? WHERE project_id = ? AND idx = ?",
            (text, project.id, int(idx)),
        )
        touch(conn, project.id)
    label.colour = text
    return label


def retire(store: Store, project: Project, idx: int) -> Label:
    """Hide a label from the palette without deleting it.

    Not a DELETE, for two reasons. Shapes already drawn with it would lose
    their referent, and -- the quiet one -- removing the row would leave a hole
    that a future ``class_list`` might close up, moving every index after it.
    """
    label = get(store, project, idx)
    if not label.is_live:
        return label
    now = time.time()
    with store.write() as conn:
        conn.execute(
            "UPDATE label SET retired_at = ? WHERE project_id = ? AND idx = ?",
            (now, project.id, int(idx)),
        )
        touch(conn, project.id)
    label.retired_at = now
    return label


def restore(store: Store, project: Project, idx: int) -> Label:
    """Un-retire a label, back into its original index."""
    label = get(store, project, idx)
    if label.is_live:
        return label
    clash = by_name(store, project, label.name)
    if clash is not None:
        raise StoreError(
            f"{label.name!r} is in use by index {clash.idx}; rename that one first"
        )
    with store.write() as conn:
        conn.execute(
            "UPDATE label SET retired_at = NULL WHERE project_id = ? AND idx = ?",
            (project.id, int(idx)),
        )
        touch(conn, project.id)
    label.retired_at = None
    return label


def variants(store: Store, project: Project) -> dict[str, list[Label]]:
    """Live labels that differ only by case or spacing, grouped.

    The real door dataset carries exactly this defect -- ``Close``, ``Open``,
    ``open`` and ``close`` as four separate classes -- because two people
    annotated it in different tools. Importing them faithfully is right;
    inventing a merge on the way in would silently change somebody's data. But
    the user has to be *told*, or they will train a four-class model by
    accident and never know why it is confused.
    """
    grouped: dict[str, list[Label]] = {}
    for label in listing(store, project):
        grouped.setdefault(label.name.strip().lower(), []).append(label)
    return {key: found for key, found in grouped.items() if len(found) > 1}


def merge(store: Store, project: Project, source_idx: int, target_idx: int) -> int:
    """Move every shape from one class onto another and retire the source.

    Returns how many shapes moved.

    This is the only safe way to fold ``close`` into ``Close``. Doing it by
    renaming would either be refused (the name is taken) or, if it were
    allowed, leave two live labels with the same name and no way to tell which
    index an export should use. Here the source index is *retired*, so it keeps
    its slot in the class list and nothing after it shifts -- the merged class
    simply has no objects any more.
    """
    source = get(store, project, source_idx)
    target = get(store, project, target_idx)
    if source.idx == target.idx:
        raise StoreError("a class cannot be merged into itself")
    if not target.is_live:
        raise StoreError(f"{target.name!r} is retired; restore it before merging into it")

    now = time.time()
    affected = [
        row["image_id"]
        for row in store.query(
            "SELECT DISTINCT image_id FROM shape WHERE project_id = ? AND label_idx = ?",
            (project.id, source.idx),
        )
    ]
    with store.write() as conn:
        moved = conn.execute(
            "UPDATE shape SET label_idx = ? WHERE project_id = ? AND label_idx = ?",
            (target.idx, project.id, source.idx),
        ).rowcount
        conn.execute(
            "UPDATE label SET retired_at = ? WHERE project_id = ? AND idx = ?",
            (now, project.id, source.idx),
        )
        touch(conn, project.id)
    # Two predictions that were "Close" and "close" were a class conflict;
    # merged, they are not. The review order has to know.
    from . import images as images_mod
    from . import shapes as shapes_mod

    for image_id in affected:
        shapes_mod.refresh_uncertainty(store, project, images_mod.get(store, project, image_id))
    return int(moved)


def class_list(store: Store, project: Project) -> list[str]:
    """The export class list: names positioned at their own indices.

    This is the function an export calls instead of sorting whatever names it
    finds. Two properties matter:

    * Retired labels still occupy their slot, so nothing after them shifts.
    * The list is dense from 0, because YOLO addresses it positionally -- a gap
      would make every later index wrong, which is the very bug this replaces.

    A gap can only appear in a database edited by hand, but filling it costs
    one branch and turns a silent mislabelling into a visible placeholder.
    """
    names = export_names(store, project)
    if not names:
        return []
    highest = max(names)
    return [names.get(i, f"_unused_{i}") for i in range(highest + 1)]


def export_names(store: Store, project: Project) -> dict[int, str]:
    """Index -> the name written into exported files. **Unique, always.**

    Names are unique among live labels only, so a retired class can share its
    name with a live one -- by renaming, or by reusing the name, which is
    allowed on purpose. Writers map shapes to indices *by name*, so a shared
    name silently merged the retired class's shapes into the live one. Live
    labels keep their names; a retired label whose name is already taken is
    written as ``name__retired_<idx>``, visibly and reversibly distinct.
    """
    rows = store.query(
        "SELECT idx, name, retired_at FROM label WHERE project_id = ?"
        " ORDER BY (retired_at IS NOT NULL), idx",
        (project.id,),
    )
    out: dict[int, str] = {}
    taken: set[str] = set()
    for row in rows:
        idx, name = int(row["idx"]), str(row["name"])
        if name in taken:
            name = f"{name}__retired_{idx}"
        taken.add(name)
        out[idx] = name
    return out


def adopt(store: Store, project: Project, names: list[str]) -> list[Label]:
    """Take on an existing class order wholesale, preserving positions.

    Used when a project is started from a dataset that already has a
    ``data.yaml``: those indices are already baked into label files on disk, so
    we inherit them rather than re-deriving an order of our own and
    invalidating every one of them.
    """
    created: list[Label] = []
    now = time.time()
    with store.write() as conn:
        row = conn.execute(
            "SELECT next_label_idx FROM project WHERE id = ?", (project.id,)
        ).fetchone()
        if row is None:
            raise NotFound("no such project")
        if int(row["next_label_idx"]) != 0:
            raise StoreError("a class order can only be adopted before any label exists")
        for idx, raw in enumerate(names):
            clean = clean_name(raw)
            conn.execute(
                "INSERT INTO label (project_id, idx, name, colour, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (project.id, idx, clean, colour_for(idx), now),
            )
            created.append(Label(project.id, idx, clean, colour_for(idx), now))
        conn.execute(
            "UPDATE project SET next_label_idx = ? WHERE id = ?", (len(names), project.id)
        )
        touch(conn, project.id)
    project.next_label_idx = len(names)
    return created

"""Shapes: storage, the save diff, and the bridge back to the neutral model.

Two things here are load-bearing.

**Saves are a diff, not a replace.** The obvious implementation -- delete every
shape for the image and insert what arrived -- loses per-shape history on every
autosave. That history is what Phase 2 is built on: whether a shape came from a
model, and whether a person has since touched it, is the difference between a
prediction and a label. So incoming shapes carry their id, and a save resolves
to updates, inserts and deletes against what is already there.

**Saves are versioned.** Every image carries a version; a save states the
version it was based on. Two tabs open on the same image no longer means the
slower one silently wins -- the second gets a :class:`Conflict` and the server's
current state to reconcile against.

Everything leaves this module as :class:`~data_factory.core.annotations.model.Shape`,
the same type the five format writers already take. Export therefore needs no
new serialiser: it asks for annotations and hands them to the existing writers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.annotations.model import Annotation, Shape
from . import labels as labels_mod
from .images import Image
from .projects import Project, touch
from .store import Conflict, Store, StoreError, new_id

#: human -- somebody drew it. model -- a detector proposed it.
ORIGINS = ("human", "model")


@dataclass(slots=True)
class StoredShape:
    """A shape as it exists in the database: geometry plus provenance."""

    id: str
    shape: Shape
    label_idx: int
    origin: str = "human"
    created_at: float = 0.0
    edited_at: float | None = None

    @property
    def is_model(self) -> bool:
        return self.origin == "model"

    @property
    def is_touched(self) -> bool:
        """A model shape a person has since corrected. Renders solid."""
        return self.edited_at is not None

    def as_dict(self) -> dict[str, Any]:
        payload = self.shape.as_dict()
        payload.update(
            {
                "id": self.id,
                "label_idx": self.label_idx,
                "origin": self.origin,
                "edited": self.is_touched,
            }
        )
        return payload


def _geometry(row: Any) -> tuple[tuple[int, int, int, int], list[tuple[int, int]] | None]:
    bbox = (int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"]))
    raw = row["polygon"]
    if not raw:
        return bbox, None
    try:
        points = json.loads(raw)
    except (TypeError, ValueError):
        # A corrupt polygon column should degrade to a box, not lose the
        # object entirely. The box columns are always authoritative.
        return bbox, None
    return bbox, [(int(x), int(y)) for x, y in points]


def _stored(row: Any, names: dict[int, str]) -> StoredShape:
    bbox, polygon = _geometry(row)
    idx = int(row["label_idx"])
    return StoredShape(
        id=row["id"],
        shape=Shape(
            label=names.get(idx, f"class_{idx}"),
            bbox=bbox,
            polygon=polygon,
            pose=row["pose"],
            truncated=int(row["truncated"]),
            difficult=int(row["difficult"]),
            occluded=int(row["occluded"]),
            confidence=row["confidence"],
            kind=str(row["kind"] or "") if "kind" in row.keys() else "",
        ),
        label_idx=idx,
        origin=row["origin"],
        created_at=float(row["created_at"]),
        edited_at=row["edited_at"],
    )


def _name_map(store: Store, project: Project) -> dict[int, str]:
    return {
        int(r["idx"]): str(r["name"])
        for r in store.query("SELECT idx, name FROM label WHERE project_id = ?", (project.id,))
    }


def listing(
    store: Store, project: Project, image: Image, *, names: dict[int, str] | None = None
) -> list[StoredShape]:
    names = names if names is not None else _name_map(store, project)
    rows = store.query(
        "SELECT * FROM shape WHERE image_id = ? ORDER BY created_at, id", (image.id,)
    )
    return [_stored(r, names) for r in rows]


def annotation_for(store: Store, project: Project, image: Image) -> Annotation:
    """Everything known about one image, as the format writers want it."""
    return Annotation(
        filename=image.name,
        width=image.width,
        height=image.height,
        # Export names, which are unique: the writers map shapes to class
        # indices by name, and a retired class sharing a live one's name
        # would otherwise be merged into it in the output.
        shapes=[
            s.shape
            for s in listing(store, project, image, names=labels_mod.export_names(store, project))
        ],
    )


@dataclass(slots=True)
class SaveResult:
    version: int
    inserted: int = 0
    updated: int = 0
    deleted: int = 0
    labels_created: list[str] = field(default_factory=list)
    #: The stored id of each submitted shape, **in submission order**.
    #:
    #: This is what lets a client turn its local drafts into real shapes after
    #: a save. Without it the next save would send those shapes with no id
    #: again and they would insert a second time -- every autosave silently
    #: doubling the boxes on the image. Order is the mapping, because it is the
    #: only thing both sides already agree on.
    ids: list[str] = field(default_factory=list)


def _same_geometry(stored: StoredShape, shape: Shape) -> bool:
    return stored.shape.bbox == shape.bbox and stored.shape.polygon == shape.polygon


def save(
    store: Store,
    project: Project,
    image: Image,
    incoming: list[tuple[str | None, Shape]],
    *,
    base_version: int | None = None,
    origin: str = "human",
    advance_state: bool = True,
) -> SaveResult:
    """Apply the editor's shape list to one image.

    ``incoming`` is ``(id or None, shape)`` pairs: an id means "this is the
    shape you already have", ``None`` means "this one is new". Anything in the
    database and absent from the list is deleted.

    Labels are created on demand from the shape's name -- this is the whole
    "draw, type a name, Enter" promise, and the reason there is no schema step.
    """
    if origin not in ORIGINS:
        raise StoreError(f"origin must be one of {', '.join(ORIGINS)}")

    if base_version is not None and int(base_version) != image.version:
        raise _moved_on(base_version, image.version)

    # Labels first, outside the shape transaction. `ensure` takes the write
    # lock itself, and creating a label is a legitimate side effect of a save
    # even when the geometry turns out to be a no-op.
    #
    # Only names that *changed* are resolved. The editor sends names, not
    # indices, and re-resolving an untouched shape's name would move every
    # shape of a retired class into the live class sharing its name -- or
    # recreate the class -- on the next autosave of its image.
    created: list[str] = []
    idx_for: dict[str, int] = {}
    before_edit = {s.id: s for s in listing(store, project, image)}
    for shape_id, shape in incoming:
        if shape.label in idx_for or _keeps_class(before_edit.get(shape_id), shape):
            continue
        before = labels_mod.by_name(store, project, shape.label)
        label = labels_mod.ensure(store, project, shape.label)
        idx_for[shape.label] = label.idx
        # `ensure` normalises whitespace, so match on the resolved name.
        idx_for[label.name] = label.idx
        if before is None:
            created.append(label.name)

    seen: set[str] = set()
    now = time.time()

    with store.write() as conn:
        # Check and read again *inside* the transaction. The check above is
        # only a fast path: between it and here another save -- a second tab,
        # or an autosave racing a navigate-flush -- can commit, and a diff
        # computed against what it replaced would resurrect deleted shapes or
        # drop new ones without anyone being told.
        row = conn.execute(
            "SELECT version, state FROM image WHERE id = ?", (image.id,)
        ).fetchone()
        if row is None:
            raise StoreError("that image no longer exists")
        image.version, image.state = int(row["version"]), row["state"]
        if base_version is not None and int(base_version) != image.version:
            raise _moved_on(base_version, image.version)
        existing = {s.id: s for s in listing(store, project, image)}
        result = SaveResult(version=image.version, labels_created=created)

        for position, (shape_id, raw) in enumerate(incoming):
            # Server-side clamping, always. A browser can send a box dragged
            # off the edge of the image, and every format writer downstream
            # assumes coordinates are inside the picture.
            shape = raw.clamped(image.width, image.height) if image.width and image.height else raw
            before = existing.get(shape_id) if shape_id else None
            if _keeps_class(before, shape):
                label_idx = before.label_idx
            elif shape.label in idx_for or labels_mod.clean_name(shape.label) in idx_for:
                label_idx = idx_for.get(shape.label, idx_for.get(labels_mod.clean_name(shape.label)))
            else:
                # Its class changed between the read above and this lock:
                # another save got in. Ask the client to reload, never guess.
                raise _moved_on(base_version, image.version)
            polygon = (
                json.dumps([[int(x), int(y)] for x, y in shape.polygon])
                if shape.polygon
                else None
            )
            xmin, ymin, xmax, ymax = shape.bbox

            prior = existing.get(shape_id) if shape_id else None
            if prior is not None:
                seen.add(prior.id)
                # A model shape a person has moved becomes their shape: it
                # stops rendering dashed and starts counting as a correction.
                edited = prior.edited_at
                if prior.is_model and not _same_geometry(prior, shape):
                    edited = now
                conn.execute(
                    "UPDATE shape SET label_idx = ?, xmin = ?, ymin = ?, xmax = ?,"
                    " ymax = ?, polygon = ?, pose = ?, truncated = ?, difficult = ?,"
                    " occluded = ?, confidence = ?, edited_at = ?, kind = ? WHERE id = ?",
                    (
                        label_idx, xmin, ymin, xmax, ymax, polygon, shape.pose,
                        shape.truncated, shape.difficult, shape.occluded,
                        shape.confidence, edited, shape.shape_kind, prior.id,
                    ),
                )
                result.updated += 1
                result.ids.append(prior.id)
            else:
                fresh = new_id()
                conn.execute(
                    "INSERT INTO shape (id, image_id, project_id, label_idx, xmin,"
                    " ymin, xmax, ymax, polygon, origin, confidence, pose, truncated,"
                    " difficult, occluded, created_at, kind)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        fresh, image.id, project.id, label_idx, xmin, ymin, xmax,
                        ymax, polygon, origin, shape.confidence, shape.pose,
                        shape.truncated, shape.difficult, shape.occluded,
                        # Distinct per shape, in submission order. Listing
                        # sorts by created_at, and shapes saved in one burst
                        # would otherwise come back in random id order --
                        # reshuffling what is on top of what after a reload.
                        now + position * 1e-6,
                        shape.shape_kind,
                    ),
                )
                result.inserted += 1
                result.ids.append(fresh)

        for gone in set(existing) - seen:
            conn.execute("DELETE FROM shape WHERE id = ?", (gone,))
            result.deleted += 1

        result.version = image.version + 1
        state = image.state
        if advance_state and state in ("todo", "predicted"):
            state = "in_progress"
        updated = conn.execute(
            "UPDATE image SET version = ?, state = ?, updated_at = ?"
            " WHERE id = ? AND version = ?",
            (result.version, state, now, image.id, image.version),
        )
        if updated.rowcount != 1:  # unreachable under the write lock; kept as the backstop
            raise _moved_on(image.version, None)
        touch(conn, project.id)

    image.version = result.version
    image.state = state
    refresh_uncertainty(store, project, image)
    return result


#: Two predictions of different classes overlapping this much are the model
#: arguing with itself about what the object is.
CONFLICT_IOU = 0.7


def uncertainty_of(stored: list[StoredShape]) -> float:
    """How much review an image's predictions need. Higher: look sooner.

    Only untouched model shapes count -- what a person drew or corrected is
    settled. Each scores its ambiguity, 1 at confidence 0.5 falling to 0 at
    0 or 1, so a wall of 0.95s is quick work and a handful of 0.5s is not.
    Every pair of overlapping predictions with different classes adds 1:
    class confusion is the commonest systematic error and the most
    expensive one to train on.
    """
    live = [s for s in stored if s.is_model and not s.is_touched]
    score = 0.0
    for s in live:
        if s.shape.confidence is not None:
            score += 1.0 - abs(2.0 * float(s.shape.confidence) - 1.0)
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            if a.label_idx != b.label_idx and _iou(a.shape.bbox, b.shape.bbox) > CONFLICT_IOU:
                score += 1.0
    return round(score, 4)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def refresh_uncertainty(store: Store, project: Project, image: Image) -> float:
    score = uncertainty_of(listing(store, project, image))
    with store.write() as conn:
        conn.execute("UPDATE image SET uncertainty = ? WHERE id = ?", (score, image.id))
    image.uncertainty = score
    return score


def _keeps_class(before: StoredShape | None, shape: Shape) -> bool:
    """An existing shape submitted under the name it already had."""
    return before is not None and before.shape.label == shape.label


def _moved_on(had: int | None, now: int | None) -> Conflict:
    current = "" if now is None else f", it is now {now}"
    return Conflict(
        f"this image moved on (you had version {had}{current}). "
        "Reload and reapply your change."
    )


def replace_all(
    store: Store,
    project: Project,
    image: Image,
    shapes: list[Shape],
    *,
    origin: str = "human",
    base_version: int | None = None,
) -> SaveResult:
    """Save a list of plain shapes, treating all of them as new.

    The import path -- predictions, or a folder of existing sidecar files --
    where nothing has an id yet.
    """
    return save(
        store,
        project,
        image,
        [(None, s) for s in shapes],
        base_version=base_version,
        origin=origin,
        advance_state=False,
    )


def label_histogram(store: Store, project: Project) -> dict[str, int]:
    """Objects per class, by name. Feeds the class-balance panel."""
    names = _name_map(store, project)
    out: dict[str, int] = {}
    for row in store.query(
        "SELECT label_idx, COUNT(*) AS n FROM shape WHERE project_id = ? GROUP BY label_idx",
        (project.id,),
    ):
        idx = int(row["label_idx"])
        out[names.get(idx, f"class_{idx}")] = int(row["n"])
    return out


def agreement(store: Store, project: Project) -> dict[str, Any]:
    """How often the model was right, as a headline number.

    Counts model shapes that survived review untouched against those a person
    moved or deleted. This is the figure that tells you when to retrain and
    when to stop annotating by hand.

    **Scoped to reviewed images.** A prediction nobody has looked at is not
    evidence of agreement -- it is evidence of nothing. Counting the whole
    project would mean importing a fresh run made the model look better,
    because every new unexamined guess would score as an agreement, and the
    number would climb fastest exactly when the least work had been done.
    """
    # The denominator is what the model *predicted* on reviewed images, not
    # what survived -- see the predicted_count column. Deleting a false
    # positive is the commonest correction there is, and counting only
    # survivors would score a model that hallucinates boxes at 100%.
    predicted = store.one(
        "SELECT COALESCE(SUM(predicted_count), 0) AS n FROM image"
        " WHERE project_id = ? AND state = 'reviewed'",
        (project.id,),
    )
    total = int(predicted["n"] or 0)

    row = store.one(
        "SELECT COUNT(*) AS surviving,"
        " SUM(CASE WHEN s.edited_at IS NOT NULL THEN 1 ELSE 0 END) AS edited"
        " FROM shape s JOIN image i ON i.id = s.image_id"
        " WHERE s.project_id = ? AND s.origin = 'model' AND i.state = 'reviewed'",
        (project.id,),
    )
    surviving = int(row["surviving"] or 0)
    edited = int(row["edited"] or 0)
    deleted = max(0, total - surviving)
    corrected = edited + deleted

    pending = store.one(
        "SELECT COALESCE(SUM(predicted_count), 0) AS n FROM image"
        " WHERE project_id = ? AND state != 'reviewed'",
        (project.id,),
    )
    return {
        "model_shapes": total,
        "corrected": corrected,
        "adjusted": edited,
        "deleted": deleted,
        "unreviewed": int(pending["n"] or 0),
        "agreement": round(max(0.0, 1.0 - (corrected / total)), 4) if total else None,
    }

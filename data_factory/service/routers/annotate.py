"""Annotation routes: projects, labels, images and shapes.

Thin on purpose. Every rule that matters -- ownership, label index allocation,
coordinate clamping, save conflicts -- lives in ``data_factory/annotate/`` and
is tested without an HTTP client. What is left here is translation: domain
exceptions to status codes, and JSON to domain objects.

The one rule this layer owns is the same one the batch API owns: **a client
never names a path.** It names a project id and an image id; every path is
derived server-side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...annotate import images as images_mod
from ...annotate import labels as labels_mod
from ...annotate import projects as projects_mod
from ...annotate import shapes as shapes_mod
from ...annotate.store import Conflict, NotFound, Store, StoreError, valid_id
from ...core.annotations.model import AnnotationError, Shape
from ..deps import get_store, media_root, principal, require_key

router = APIRouter(prefix="/api", tags=["annotate"], dependencies=[Depends(require_key)])

#: One save carries one image's shapes. A cap turns "a client sent us 40
#: million boxes" from an out-of-memory event into a 400.
MAX_SHAPES_PER_SAVE = 5000

#: Images registered in a single scan request.
MAX_IMAGES_PER_CALL = 50_000


# ------------------------------------------------------------------ wire

class ProjectIn(BaseModel):
    name: str = Field(default="Untitled project", max_length=200)


class LabelIn(BaseModel):
    name: str = Field(max_length=labels_mod.MAX_NAME)


class LabelPatch(BaseModel):
    name: str | None = Field(default=None, max_length=labels_mod.MAX_NAME)
    colour: str | None = Field(default=None, max_length=7)


class StatePatch(BaseModel):
    state: str


#: Shape fields the editor may not send, and which a save must therefore keep.
KEPT_ATTRIBUTES = ("pose", "truncated", "difficult", "occluded")


class ShapesIn(BaseModel):
    version: int | None = Field(
        default=None,
        description="The image version this edit was based on. Omit to force.",
    )
    shapes: list[dict[str, Any]] = Field(default_factory=list)
    advance_state: bool = True


# -------------------------------------------------------------- plumbing

def _fail(exc: Exception) -> HTTPException:
    """Map a domain exception onto a status code.

    ``NotFound`` covers both "no such project" and "not yours" deliberately --
    telling them apart would tell a stranger which ids are real.
    """
    if isinstance(exc, NotFound):
        return HTTPException(404, str(exc))
    if isinstance(exc, Conflict):
        return HTTPException(409, str(exc))
    if isinstance(exc, (StoreError, AnnotationError)):
        return HTTPException(400, str(exc))
    return HTTPException(400, "that request could not be completed")


def _project(store: Store, project_id: str, owner: str):
    try:
        return projects_mod.get(store, project_id, owner)
    except StoreError as exc:
        raise _fail(exc) from None


def _image(store: Store, project, image_id: str):
    try:
        return images_mod.get(store, project, image_id)
    except StoreError as exc:
        raise _fail(exc) from None


def _sync_disk(store: Store, project, image_ids) -> list[str]:
    """Write changed images back to an attached folder's own label files."""
    if not project.is_attached:
        return []
    from ...annotate import attach

    return attach.sync(store, project, image_ids)


def attach_images_using(store: Store, project, idx: int) -> list[str]:
    if not project.is_attached:
        return []
    from ...annotate import attach

    return attach.images_using(store, project, idx)


def _with_disk(payload: dict[str, Any], problems: list[str]) -> dict[str, Any]:
    if problems:
        payload["sidecar_errors"] = problems
    return payload


def _image_payload(store: Store, project, image) -> dict[str, Any]:
    stored = shapes_mod.listing(store, project, image)
    return {
        **image.as_dict(),
        "shapes": [s.as_dict() for s in stored],
    }


# -------------------------------------------------------------- projects

@router.post("/projects", status_code=201)
def create_project(
    body: ProjectIn,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    """Start a project. No labels, no task, no organisation -- just a name."""
    try:
        project = projects_mod.create(store, body.name, owner_id=owner)
    except StoreError as exc:
        raise _fail(exc) from None
    media_root(project.id).mkdir(parents=True, exist_ok=True)
    return project.as_dict()


@router.get("/projects")
def list_projects(
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    return {"projects": [p.as_dict() for p in projects_mod.listing(store, owner)]}


@router.get("/projects/{project_id}")
def project_detail(
    project_id: str,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    """Everything the editor needs to open a project in one round trip."""
    project = _project(store, project_id, owner)
    resume = images_mod.resume_point(store, project)
    return {
        **project.as_dict(),
        "labels": [label.as_dict() for label in labels_mod.listing(store, project)],
        "counts": images_mod.counts(store, project),
        "resume": resume.as_dict() if resume else None,
        "agreement": shapes_mod.agreement(store, project),
    }


@router.patch("/projects/{project_id}")
def rename_project(
    project_id: str,
    body: ProjectIn,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    project = _project(store, project_id, owner)
    try:
        return projects_mod.rename(store, project, body.name).as_dict()
    except StoreError as exc:
        raise _fail(exc) from None


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(
    project_id: str,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> None:
    project = _project(store, project_id, owner)
    projects_mod.delete(store, project)
    # Media is removed after the rows, so a crash between the two leaves
    # orphaned files rather than rows pointing at nothing.
    import shutil

    shutil.rmtree(media_root(project.id), ignore_errors=True)


# ---------------------------------------------------------------- labels

@router.get("/projects/{project_id}/labels")
def list_labels(
    project_id: str,
    include_retired: bool = False,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    project = _project(store, project_id, owner)
    return {
        "labels": [
            label.as_dict()
            for label in labels_mod.listing(store, project, include_retired=include_retired)
        ],
        # The export order, including retired slots. Shown so nobody has to
        # guess why index 3 is missing from the palette.
        "class_list": labels_mod.class_list(store, project),
        # Classes differing only by case or spacing. Surfaced rather than
        # merged: folding them silently would change somebody's data.
        "variants": {
            key: [label.idx for label in found]
            for key, found in labels_mod.variants(store, project).items()
        },
    }


@router.post("/projects/{project_id}/labels", status_code=201)
def create_label(
    project_id: str,
    body: LabelIn,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    """Create a label by naming it. Idempotent, so the editor can just call it."""
    project = _project(store, project_id, owner)
    try:
        return labels_mod.ensure(store, project, body.name).as_dict()
    except StoreError as exc:
        raise _fail(exc) from None


@router.patch("/projects/{project_id}/labels/{idx}")
def patch_label(
    project_id: str,
    idx: int,
    body: LabelPatch,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    """Rename or recolour. Neither moves the index."""
    project = _project(store, project_id, owner)
    try:
        label = labels_mod.get(store, project, idx)
        problems: list[str] = []
        if body.name is not None:
            label = labels_mod.rename(store, project, idx, body.name)
            problems = _sync_disk(store, project, attach_images_using(store, project, idx))
        if body.colour is not None:
            label = labels_mod.recolour(store, project, idx, body.colour)
        return _with_disk(label.as_dict(), problems)
    except StoreError as exc:
        raise _fail(exc) from None


@router.delete("/projects/{project_id}/labels/{idx}")
def retire_label(
    project_id: str,
    idx: int,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    """Retire a label. Never a DELETE: shapes still reference it, and its
    index must stay occupied so later classes do not shift."""
    project = _project(store, project_id, owner)
    try:
        retired = labels_mod.retire(store, project, idx)
    except StoreError as exc:
        raise _fail(exc) from None
    problems = _sync_disk(store, project, attach_images_using(store, project, idx))
    return _with_disk(retired.as_dict(), problems)


class MergeIn(BaseModel):
    into: int


@router.post("/projects/{project_id}/labels/{idx}/merge")
def merge_label(
    project_id: str,
    idx: int,
    body: MergeIn,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    """Fold one class into another -- the fix for case-variant duplicates.

    The source index is retired, not deleted, so it keeps its slot and no
    later class shifts.
    """
    project = _project(store, project_id, owner)
    affected = attach_images_using(store, project, idx)
    try:
        moved = labels_mod.merge(store, project, idx, body.into)
    except StoreError as exc:
        raise _fail(exc) from None
    problems = _sync_disk(store, project, affected)
    return _with_disk({
        "moved": moved,
        "labels": [label.as_dict() for label in labels_mod.listing(store, project)],
        "class_list": labels_mod.class_list(store, project),
    }, problems)


@router.post("/projects/{project_id}/labels/{idx}/restore")
def restore_label(
    project_id: str,
    idx: int,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    project = _project(store, project_id, owner)
    try:
        restored = labels_mod.restore(store, project, idx)
    except StoreError as exc:
        raise _fail(exc) from None
    problems = _sync_disk(store, project, attach_images_using(store, project, idx))
    return _with_disk(restored.as_dict(), problems)


# ---------------------------------------------------------------- images

@router.get("/projects/{project_id}/images")
def list_images(
    project_id: str,
    state: str = "",
    limit: int = 500,
    offset: int = 0,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    project = _project(store, project_id, owner)
    try:
        found = images_mod.listing(store, project, state=state, limit=limit, offset=offset)
    except StoreError as exc:
        raise _fail(exc) from None
    return {
        "images": [i.as_dict() for i in found],
        "counts": images_mod.counts(store, project),
        "offset": offset,
    }


@router.get("/projects/{project_id}/resume")
def resume(
    project_id: str,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    """Where to carry on. Opening a project lands here, always."""
    project = _project(store, project_id, owner)
    image = images_mod.resume_point(store, project)
    if image is None:
        return {"image": None}
    return {"image": _image_payload(store, project, image)}


@router.get("/projects/{project_id}/images/{image_id}")
def image_detail(
    project_id: str,
    image_id: str,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    project = _project(store, project_id, owner)
    image = _image(store, project, image_id)
    payload = _image_payload(store, project, image)
    nxt = images_mod.neighbour(store, project, image, 1)
    prv = images_mod.neighbour(store, project, image, -1)
    payload["next"] = nxt.id if nxt else None
    payload["previous"] = prv.id if prv else None
    return payload


@router.patch("/projects/{project_id}/images/{image_id}")
def patch_image(
    project_id: str,
    image_id: str,
    body: StatePatch,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    project = _project(store, project_id, owner)
    image = _image(store, project, image_id)
    try:
        return images_mod.set_state(store, project, image, body.state).as_dict()
    except StoreError as exc:
        raise _fail(exc) from None


# ---------------------------------------------------------------- shapes

@router.put("/projects/{project_id}/images/{image_id}/shapes")
def save_shapes(
    project_id: str,
    image_id: str,
    body: ShapesIn,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    """Save one image's shapes.

    Labels named here that do not exist yet are created -- that is the whole
    "draw a box, type a name, Enter" flow, and why there is no schema step.

    A 409 carries the server's current state so the editor can reconcile
    against it rather than asking the user to redo the work.
    """
    project = _project(store, project_id, owner)
    image = _image(store, project, image_id)

    if len(body.shapes) > MAX_SHAPES_PER_SAVE:
        raise HTTPException(
            413, f"an image can hold at most {MAX_SHAPES_PER_SAVE:,} shapes in one save"
        )

    # A client that omits an attribute means "unchanged", not "reset". Older
    # editors sent none of these, and every save cleared VOC's difficult,
    # truncated, occluded and pose -- into the user's own XML in attached mode.
    stored = {s.id: s.shape for s in shapes_mod.listing(store, project, image)}
    incoming: list[tuple[str | None, Shape]] = []
    try:
        for raw in body.shapes:
            raw_id = raw.get("id")
            shape_id = raw_id if isinstance(raw_id, str) and valid_id(raw_id) else None
            before = stored.get(shape_id) if shape_id else None
            if before is not None:
                raw = {
                    **{key: getattr(before, key) for key in KEPT_ATTRIBUTES if key not in raw},
                    **raw,
                }
            incoming.append((shape_id, Shape.from_dict(raw)))
    except AnnotationError as exc:
        raise _fail(exc) from None

    try:
        saved = shapes_mod.save(
            store,
            project,
            image,
            incoming,
            base_version=body.version,
            advance_state=body.advance_state,
        )
    except Conflict as exc:
        raise HTTPException(
            409,
            detail={
                "message": str(exc),
                "current": _image_payload(store, project, _image(store, project, image_id)),
            },
        ) from None
    except StoreError as exc:
        raise _fail(exc) from None

    # Write-through, for a folder opened in place. The database already has the
    # edit, so a folder that has gone read-only degrades to "saved, but not
    # written to disk" with a reason -- never to a lost edit.
    written = ""
    sidecar_error = ""
    if project.is_attached:
        from ...annotate import attach

        try:
            target = attach.write_through(store, project, image)
            written = target.name if target else ""
        except StoreError as exc:
            sidecar_error = str(exc)

    return {
        "version": saved.version,
        "inserted": saved.inserted,
        "updated": saved.updated,
        "deleted": saved.deleted,
        "labels_created": saved.labels_created,
        "sidecar": written,
        "sidecar_error": sidecar_error,
        # In submission order, so a client can turn its local drafts into real
        # shapes. Without this the next save would re-send them with no id and
        # insert them a second time.
        "ids": saved.ids,
        "state": image.state,
        "labels": [label.as_dict() for label in labels_mod.listing(store, project)],
    }


# ---------------------------------------------------------------- review

@router.post("/projects/{project_id}/images/{image_id}/accept")
def accept_image(
    project_id: str,
    image_id: str,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    """Take this image's predictions as correct and mark it reviewed.

    What ``Space`` calls. The shapes keep ``origin='model'`` -- an accepted
    prediction is still a prediction, and rewriting that away would empty out
    the agreement metric's denominator.
    """
    from ...annotate import review

    project = _project(store, project_id, owner)
    image = _image(store, project, image_id)
    confirmed = review.accept_all(store, project, image)
    problems = _sync_disk(store, project, [image.id])
    nxt = images_mod.neighbour(store, project, image, 1)
    return _with_disk({
        "confirmed": confirmed,
        "state": image.state,
        "next": nxt.id if nxt else None,
        "counts": images_mod.counts(store, project),
        "agreement": shapes_mod.agreement(store, project),
    }, problems)


@router.get("/projects/{project_id}/runs")
def list_runs(
    project_id: str,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    from ...annotate import review

    project = _project(store, project_id, owner)
    return {"runs": review.runs(store, project)}


# ---------------------------------------------------------------- summary

@router.get("/projects/{project_id}/summary")
def summary(
    project_id: str,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    """Class balance and model agreement -- the two numbers worth watching."""
    project = _project(store, project_id, owner)
    return {
        "counts": images_mod.counts(store, project),
        "classes": shapes_mod.label_histogram(store, project),
        "agreement": shapes_mod.agreement(store, project),
        "class_list": labels_mod.class_list(store, project),
    }


def register_images(
    store: Store,
    project,
    root: Path,
    entries: list[tuple[str, int, int]],
) -> list[Any]:
    """Shared by promotion and by the attached-folder scan."""
    if len(entries) > MAX_IMAGES_PER_CALL:
        raise HTTPException(413, f"at most {MAX_IMAGES_PER_CALL:,} images at a time")
    return images_mod.add_many(store, project, entries)

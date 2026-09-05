"""Loading a model's output back onto a project.

The upload half of the review loop. A client sends a file -- a zip of YOLO
``.txt``, VOC ``.xml`` or LabelMe ``.json``, or a single COCO json in either
the dataset or the detection-results form -- and never a path. The same
zip-slip, zip-bomb and member-count guards the dataset upload uses apply here,
because an archive from a training run is no more trustworthy than any other.

The response is a **matching report**, and it is the point of the endpoint.
Predictions match images by filename stem; when they do not line up the result
is zero shapes imported and no error anywhere, which looks exactly like a model
that predicted nothing. Saying so plainly is the difference between a feature
that works and one people quietly stop using.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ...annotate import projects as projects_mod
from ...annotate import review
from ...annotate.store import Store, StoreError
from ..deps import get_store, principal, require_key
from ..workspace import MAX_UPLOAD_BYTES, Workspace, WorkspaceError, extract_zip

router = APIRouter(prefix="/api", tags=["annotate"], dependencies=[Depends(require_key)])

#: Anything else is not a shape this endpoint knows how to read, and guessing
#: from bytes would be worse than saying so.
ALLOWED_SUFFIXES = (".zip", ".json")


@router.post("/projects/{project_id}/predictions", status_code=201)
async def import_predictions(
    project_id: str,
    file: UploadFile = File(...),
    name: str = Form(default=""),
    min_confidence: float = Form(default=0.0),
    overwrite_reviewed: bool = Form(default=False),
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    try:
        project = projects_mod.get(store, project_id, owner)
    except StoreError as exc:
        raise HTTPException(404, str(exc)) from None

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            400,
            f"send a .zip of label files or a .json of predictions, not "
            f"{suffix or 'a file with no extension'}",
        )
    if not (0.0 <= min_confidence <= 1.0):
        raise HTTPException(400, "min_confidence must be between 0 and 1")

    holding = Path(tempfile.mkdtemp(prefix="predictions-"))
    try:
        payload = holding / "upload" / (Path(file.filename or "upload").name)
        payload.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        with payload.open("wb") as handle:
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "that file is too large")
                handle.write(chunk)
        if written == 0:
            raise HTTPException(400, "that file was empty")

        if suffix == ".zip":
            # Rooted at the temp directory so extract_zip's containment check
            # is against somewhere disposable, not the real workspace.
            sandbox = Workspace(holding / "unpacked")
            target = sandbox.root / "run"
            try:
                extract_zip(payload, target, workspace=sandbox)
            except WorkspaceError as exc:
                raise HTTPException(400, str(exc)) from None
            source = target
        else:
            source = payload

        try:
            report = review.import_predictions(
                store,
                project,
                source,
                name=name or Path(file.filename or "").stem,
                min_confidence=min_confidence,
                overwrite_reviewed=overwrite_reviewed,
            )
        except StoreError as exc:
            raise HTTPException(400, str(exc)) from None
    finally:
        shutil.rmtree(holding, ignore_errors=True)

    from ...annotate import images as images_mod
    from ...annotate import labels as labels_mod
    from ...annotate import shapes as shapes_mod

    report["counts"] = images_mod.counts(store, project)
    report["agreement"] = shapes_mod.agreement(store, project)
    report["labels"] = [label.as_dict() for label in labels_mod.listing(store, project)]
    return report

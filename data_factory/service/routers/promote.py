"""Turning an uploaded dataset into an annotation project.

This is the funnel. A stranger drops a zip on the front page, looks at what the
tool says about it, clicks "Start annotating", and only *then* is there a
project worth asking them to sign in for. Nothing before this point required an
account, a schema, or a decision.

It is also the migration path. The browser annotator at annotationfree4u.com
exports Pascal VOC, and this reads VOC, so "upload what you already have and
carry on" works on day one rather than after a converter is written.

**What happens to state on import.** Shapes that arrive with a confidence score
came from a detector and land as ``predicted`` -- unreviewed, and excluded from
export until a person has looked. Shapes without one were drawn by a human in
some other tool, so they land as ``reviewed``. That is the same distinction the
whole review layer rests on, applied at the door.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...annotate import images as images_mod
from ...annotate import labels as labels_mod
from ...annotate import projects as projects_mod
from ...annotate import shapes as shapes_mod
from ...annotate.store import Store, StoreError
from ...core import fs, imaging
from ...core.annotations import formats, yolo
from ...core.annotations.model import AnnotationError
from ..deps import get_store, media_root, principal, require_key
from ..workspace import Workspace, WorkspaceError, resolve_workspace_root

router = APIRouter(prefix="/api", tags=["annotate"], dependencies=[Depends(require_key)])

#: A promotion copies files. Past this many, it is not a browser upload any
#: more and belongs on the command line.
MAX_IMAGES = 50_000


class PromoteIn(BaseModel):
    name: str = Field(default="", max_length=200)
    import_annotations: bool = True


@router.post("/datasets/{dataset_id}/promote", status_code=201)
def promote(
    dataset_id: str,
    body: PromoteIn,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    """Create a project from an uploaded dataset."""
    # Built per request rather than held at module scope: the workspace root is
    # an environment variable, and a cached one would outlive a change to it.
    workspace = Workspace(resolve_workspace_root())
    try:
        dataset = workspace.get(dataset_id)
    except WorkspaceError as exc:
        raise HTTPException(404, str(exc)) from None

    source = dataset.source
    files = sorted(p for p in fs.iter_images(source, recursive=True))
    if not files:
        raise HTTPException(400, "that dataset has no images to annotate")
    if len(files) > MAX_IMAGES:
        raise HTTPException(
            413,
            f"{len(files):,} images is past what a browser upload should carry. "
            f"Use 'datafactory annotate' on the folder directly.",
        )

    project = projects_mod.create(store, body.name or f"Dataset {dataset_id[:8]}", owner_id=owner)
    root = media_root(project.id)
    root.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    loaded = None
    if body.import_annotations:
        try:
            loaded = formats.read_dataset(source, "auto", recursive=True)
            warnings.extend(loaded.warnings[:10])
        except AnnotationError as exc:
            # A folder of bare images is a completely normal thing to annotate.
            warnings.append(f"no annotations imported: {exc}")

    # Adopt an existing class order before anything else. If this dataset came
    # with a data.yaml, its indices are already baked into .txt files that
    # exist in the world, and inventing our own order would invalidate them.
    if loaded is not None and loaded.classes:
        try:
            labels_mod.adopt(store, project, list(loaded.classes))
        except StoreError as exc:
            warnings.append(f"could not adopt the existing class order: {exc}")
    elif yolo.find_data_yaml(source) is not None:
        warnings.append("a data.yaml was present but its class list could not be read")

    by_stem = {}
    if loaded is not None:
        for item in loaded.items:
            by_stem.setdefault(item.stem, item.annotation)

    copied = 0
    registered: list[tuple[str, int, int]] = []
    for path in files:
        rel = path.relative_to(source).as_posix()
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(path, target)
            copied += 1
        except OSError as exc:
            warnings.append(f"{rel}: could not copy ({exc})")
            continue
        try:
            # Header read, EXIF honoured, no decode -- 20,000 images stays fast.
            width, height = imaging.image_size(target)
        except Exception:
            width = height = 0
        registered.append((rel, width, height))

    made = images_mod.add_many(store, project, registered)

    imported_shapes = 0
    predicted = 0
    reviewed = 0
    for image in made:
        annotation = by_stem.get(Path(image.rel_path).stem)
        if annotation is None or not annotation.shapes:
            continue
        # A score means a detector produced it. No score means a person did.
        from_model = any(s.confidence is not None for s in annotation.shapes)
        try:
            shapes_mod.replace_all(
                store,
                project,
                image,
                list(annotation.shapes),
                origin="model" if from_model else "human",
            )
        except StoreError as exc:
            warnings.append(f"{image.rel_path}: {exc}")
            continue
        imported_shapes += len(annotation.shapes)
        state = "predicted" if from_model else "reviewed"
        images_mod.set_state(store, project, image, state)
        if from_model:
            predicted += 1
        else:
            reviewed += 1

    unmatched = len(made) - (predicted + reviewed)
    if loaded is not None and loaded.items and unmatched:
        # Never silent. Importing zero shapes because the filenames did not
        # line up is the single most common way this goes wrong.
        warnings.append(
            f"{predicted + reviewed} image(s) matched an annotation file, "
            f"{unmatched} did not and start empty"
        )

    return {
        **project.as_dict(),
        "images": len(made),
        "images_copied": copied,
        "shapes_imported": imported_shapes,
        "reviewed": reviewed,
        "predicted": predicted,
        "labels": [label.as_dict() for label in labels_mod.listing(store, project)],
        "warnings": warnings,
    }

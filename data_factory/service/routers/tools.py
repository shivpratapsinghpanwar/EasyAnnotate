"""Heavy tools, over HTTP: what is available, and running one.

The rule this router exists to keep, stated in the plan and worth repeating
because everything here follows from it:

    **A heavy tool's absence is never an error.**

``GET /api/tools`` returning ``{"tools": []}`` is a normal, tested state -- it
is what a plain ``pip install data-factory`` produces, and the editor is fully
usable in it. Buttons appear only for a tool that is both listed *and* healthy,
so there are no dead controls and no install nags in the toolbar.

Anything model-shaped runs in a separate process (see
``annotate/tools/local.py``). A CUDA out-of-memory or a native segfault kills
that process, the registry reports the tool unavailable, and **the person
carries on drawing boxes**. That is the entire reason for the boundary.
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
from ...annotate.store import Store, StoreError
from ...annotate.tools import postprocess
from ...annotate.tools.protocol import ToolError, ToolNotReady, ToolUnavailable
from ...annotate.tools.registry import REGISTRY
from ..deps import get_store, media_root, principal, require_key, resolve_media

router = APIRouter(prefix="/api", tags=["tools"], dependencies=[Depends(require_key)])


class InvokeIn(BaseModel):
    image_id: str
    #: Whatever the tool's declared input schema asks for -- a box, a set of
    #: click points. Passed through rather than modelled here, because the
    #: whole point of a declared spec is that a new tool needs no server change.
    params: dict[str, Any] = Field(default_factory=dict)
    label: str = ""


@router.get("/tools")
def list_tools() -> dict[str, Any]:
    """Every usable tool. An empty list is the normal, healthy default."""
    return REGISTRY.as_dict()


@router.post("/projects/{project_id}/tools/{name}")
def invoke_tool(
    project_id: str,
    name: str,
    body: InvokeIn,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    """Run a tool against one image and turn its answer into shapes.

    The mask-to-polygon step is ours, not the tool's: a model returns a raster
    mask, and what the editor needs is an outline with a manageable number of
    draggable vertices. See ``annotate/tools/postprocess.py``.
    """
    try:
        project = projects_mod.get(store, project_id, owner)
        image = images_mod.get(store, project, body.image_id)
    except StoreError as exc:
        raise HTTPException(404, str(exc)) from None

    root = (
        Path(project.root)
        if project.is_attached and project.root
        else media_root(project.id)
    )
    path = resolve_media(root, image.rel_path)
    if not path.is_file():
        raise HTTPException(404, f"{image.rel_path} is no longer on disk")

    payload = {
        **body.params,
        "image_id": image.id,
        "image_path": str(path),
        "width": image.width,
        "height": image.height,
    }

    try:
        result = REGISTRY.invoke(name, payload)
    except ToolNotReady as exc:
        # Not an error: the sidecar is still encoding this image. The client
        # is told when to come back rather than being made to guess.
        raise HTTPException(
            503,
            detail={"message": str(exc), "retry_after_ms": getattr(exc, "retry_after_ms", 500)},
        ) from None
    except ToolUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    except ToolError as exc:
        raise HTTPException(400, str(exc)) from None
    except Exception:
        # A tool crashing must not surface a traceback, and must not look like
        # the editor breaking.
        raise HTTPException(502, f"the {name!r} tool failed; drawing by hand still works") from None

    label = (body.label or "").strip()
    made = _shapes_from(result, label, image.width, image.height)
    if not made:
        return {"shapes": [], "detail": "the tool returned nothing for that input"}

    if label:
        try:
            labels_mod.ensure(store, project, label)
        except StoreError as exc:
            raise HTTPException(400, str(exc)) from None

    return {
        "shapes": [
            {**shape.as_dict(), "origin": "model"}
            for shape in made
        ],
        "detail": result.get("detail", "") if isinstance(result, dict) else "",
    }


def _shapes_from(result: Any, label: str, width: int, height: int) -> list[Any]:
    """Read a tool's reply into shapes, whichever form it used.

    A tool may answer with a mask, with polygons, or with boxes. All three are
    reasonable things for a model to produce, and none of them should require
    a different endpoint.
    """
    if not isinstance(result, dict):
        return []
    made = []
    name = label or "object"

    from ...core.annotations.model import AnnotationError, Shape

    # Ready-made shapes: the most natural form for a tool to answer in, and
    # what the reference sidecar uses. Parsed through Shape.from_dict because
    # a sidecar is a separate process -- possibly a third party's -- and its
    # output is no more trusted than a browser's.
    for raw in result.get("shapes") or []:
        if not isinstance(raw, dict):
            continue
        payload = dict(raw)
        if label:
            payload["label"] = label
        payload.setdefault("label", name)
        try:
            shape = Shape.from_dict(payload)
        except AnnotationError:
            continue
        if shape.polygon and len(shape.polygon) > 3:
            shape.polygon = postprocess.simplify_polygon(shape.polygon)
        made.append(shape.clamped(width, height) if width and height else shape)

    mask = result.get("mask")
    if mask is not None:
        shape = postprocess.mask_to_shape(
            mask, name, width, height, confidence=result.get("score")
        )
        if shape is not None:
            made.append(shape)

    for polygon in result.get("polygons") or []:
        points = [(int(x), int(y)) for x, y in polygon]
        if len(points) < 3:
            continue
        simplified = postprocess.simplify_polygon(points)
        xs = [p[0] for p in simplified]
        ys = [p[1] for p in simplified]
        made.append(
            Shape(
                label=name,
                bbox=(min(xs), min(ys), max(xs), max(ys)),
                polygon=simplified,
                confidence=result.get("score"),
            ).clamped(width, height)
        )

    for box in result.get("boxes") or []:
        try:
            x1, y1, x2, y2 = (int(v) for v in box[:4])
        except (TypeError, ValueError):
            continue
        made.append(
            Shape(label=name, bbox=(x1, y1, x2, y2), confidence=result.get("score")).clamped(
                width, height
            )
        )

    return made

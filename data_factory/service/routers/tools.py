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
from ...annotate.tools.protocol import ToolError, ToolNotReady, ToolUnavailable
from ...annotate.tools.registry import REGISTRY
from ...annotate.tools.results import shapes_from_result as _shapes_from
from ..deps import get_store, media_root, principal, require_key, resolve_media

router = APIRouter(prefix="/api", tags=["tools"], dependencies=[Depends(require_key)])


class InvokeIn(BaseModel):
    image_id: str
    #: Whatever the tool's declared input schema asks for -- a box, a set of
    #: click points. Passed through rather than modelled here, because the
    #: whole point of a declared spec is that a new tool needs no server change.
    params: dict[str, Any] = Field(default_factory=dict)
    label: str = ""


class BatchIn(BaseModel):
    #: The tool's own inputs -- for a prompt tool, the text.
    params: dict[str, Any] = Field(default_factory=dict)
    label: str = ""
    #: Which images to run over. Reviewed images are never touched here:
    #: a person's finished work is not something a model gets to redo.
    states: list[str] = Field(default_factory=lambda: ["todo"])
    limit: int = 2000


@router.post("/projects/{project_id}/tools/{name}/batch")
def invoke_tool_batch(
    project_id: str,
    name: str,
    body: BatchIn,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    """Run a tool over many images as a job; its answers become predictions.

    This is "type a class name, get every instance to review": each image's
    results go through review.set_predictions, so they arrive dashed, with
    confidence, counted by the agreement metric, and never exported until
    somebody has looked at them. The same path a model run imported from a
    file takes -- a tool is just a model run that happens here.
    """
    import secrets

    from ...annotate import review
    from ...core.types import OpResult
    from ..deps import jobs_manager

    try:
        project = projects_mod.get(store, project_id, owner)
    except StoreError as exc:
        raise HTTPException(404, str(exc)) from None
    jobs = jobs_manager()
    if jobs is None:
        raise HTTPException(503, "no job runner in this process")
    wanted = [s for s in body.states if s in images_mod.STATES and s != "reviewed"]
    if not wanted:
        raise HTTPException(
            400, "states must name unreviewed states: todo, predicted, in_progress, skipped"
        )
    if name not in {spec.name for spec in REGISTRY.available()}:
        raise HTTPException(404, f"no tool called {name!r} is installed")
    _refuse_local_only(name)
    _refuse_reserved(body.params)

    targets: list[Any] = []
    for state in wanted:
        targets.extend(images_mod.listing(store, project, state=state, limit=body.limit))
    targets = targets[: body.limit]
    root = Path(project.root) if project.is_attached and project.root else media_root(project.id)
    label = (body.label or "").strip()
    spec = next(s for s in REGISTRY.available() if s.name == name)
    needs_prepare = "prepare" in (spec.requires or ())

    def work(context) -> OpResult:
        result = OpResult(op=f"tool:{name}")
        predicted = shapes_total = failed = 0
        for index, image in enumerate(targets):
            context.check_cancelled()
            context.emit(index, len(targets), image.name)
            path = resolve_media(root, image.rel_path)
            if not path.is_file():
                failed += 1
                continue
            payload = {
                **body.params,
                "image_id": image.id,
                "image_path": str(path),
                "width": image.width,
                "height": image.height,
            }
            if needs_prepare:
                try:
                    REGISTRY.prepare(name, image.id, path)
                except Exception:
                    pass  # the invoke below reports the real problem
            raw = _invoke_patiently(name, payload)
            if raw is None:
                failed += 1
                continue
            made = _shapes_from(raw, label, image.width, image.height)
            # Re-read: a person may have reviewed this image while the batch
            # ran, and their work is not the batch's to replace.
            try:
                current = images_mod.get(store, project, image.id)
                if current.state not in wanted:
                    continue
                review.set_predictions(store, project, current, made, base_version=current.version)
            except StoreError:
                continue
            predicted += 1
            shapes_total += len(made)
        context.emit(len(targets), len(targets), "done")
        result.data = {
            "images": len(targets),
            "predicted": predicted,
            "shapes": shapes_total,
            "failed": failed,
            "agreement": shapes_mod.agreement(store, project),
        }
        if failed:
            result.warnings.append(f"{failed} image(s) could not be processed")
        return result

    job = jobs.submit(secrets.token_hex(8), f"tool:{name}", project.id, work)
    return job.as_dict()


def _invoke_patiently(name: str, payload: dict[str, Any]) -> Any:
    """A tool still loading its weights says not_ready; wait for it. Anything
    else wrong with one image is that image's problem, not the job's."""
    try:
        return REGISTRY.invoke_waiting(name, payload, budget_s=120.0)
    except Exception:
        return None


#: Keys the server sets itself. A client's copy is refused, not overwritten:
#: a request that tries to name the tool, a model file or an image path is
#: not a mistake worth tolerating.
RESERVED_PARAMS = frozenset({"tool", "model", "source", "image_path", "image_id", "width", "height"})


def _refuse_reserved(params: dict[str, Any]) -> None:
    clash = sorted(RESERVED_PARAMS.intersection(params))
    if clash:
        raise HTTPException(400, f"these are set by the server, not the request: {', '.join(clash)}")


def _refuse_local_only(name: str) -> None:
    """A "model" tool loads a model file, and a model file is code: loading a
    .pt unpickles it. Over HTTP that would let a client name any file to run
    on the server. Like attached folders, these run from the command line
    only -- `easyannotate predict` -- where the path comes from a person
    typing it into their own terminal."""
    spec = REGISTRY.spec(name)
    if spec is not None and spec.kind == "model":
        raise HTTPException(
            403, f"{name!r} runs a model file, which is code; use `easyannotate predict` instead"
        )


class PrepareIn(BaseModel):
    image_id: str


@router.post("/projects/{project_id}/tools/{name}/prepare")
def prepare_tool(
    project_id: str,
    name: str,
    body: PrepareIn,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> dict[str, Any]:
    """Warm a tool for the image that is open: encode-ahead.

    Best-effort by contract and always quick -- the editor fires this on
    every navigation once a SAM tool is in use, and a failure here must
    never be something the person sees.
    """
    _refuse_local_only(name)
    if REGISTRY.spec(name) is None:
        raise HTTPException(404, f"no tool called {name!r} is installed")
    try:
        project = projects_mod.get(store, project_id, owner)
        image = images_mod.get(store, project, body.image_id)
    except StoreError as exc:
        raise HTTPException(404, str(exc)) from None
    root = Path(project.root) if project.is_attached and project.root else media_root(project.id)
    path = resolve_media(root, image.rel_path)
    if path.is_file():
        REGISTRY.prepare(name, image.id, path)
    return {"ok": True}


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
    _refuse_local_only(name)
    _refuse_reserved(body.params)
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

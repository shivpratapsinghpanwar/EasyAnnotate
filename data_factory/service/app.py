"""The hosted HTTP service.

A deployable API rather than the illustrative one in ``examples/``. The
difference is entirely about trust: this one never accepts a filesystem path,
sandboxes every dataset, caps uploads, expires old work, and keeps tracebacks
out of responses.

Run it::

    pip install "data-factory[web]"
    uvicorn data_factory.service.app:app --host 0.0.0.0 --port 8000

Configuration, all optional, all environment variables::

    DATAFACTORY_WORKSPACE     where datasets live (default ~/.datafactory/workspaces)
    DATAFACTORY_TTL_HOURS     how long an anonymous dataset survives (default 24)
    DATAFACTORY_ORIGINS       comma-separated CORS origins (default: same-origin only)
    DATAFACTORY_API_KEY       if set, every /api call must send X-API-Key
    DATAFACTORY_MAX_UPLOAD_MB overrides the upload cap

The endpoints are shaped for the flow a browser actually performs: upload a
zip, look at what it is, run steps against it, download the result.
"""

from __future__ import annotations

import os
import secrets
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import data_factory as df

from ..core import registry
from ..core.annotations import formats
from ..core.annotations.model import AnnotationError
from ..core.registry import ParamError
from . import binding
from . import jobs as jobs_module
from .workspace import (
    MAX_UPLOAD_BYTES,
    Workspace,
    WorkspaceError,
    extract_zip,
    make_zip,
    resolve_workspace_root,
)

WORKSPACE = Workspace(
    resolve_workspace_root(),
    ttl_seconds=int(float(os.environ.get("DATAFACTORY_TTL_HOURS", "24")) * 3600),
)
JOBS = jobs_module.JobManager()
MAX_UPLOAD = int(os.environ.get("DATAFACTORY_MAX_UPLOAD_MB", "0")) * 1024 * 1024 or MAX_UPLOAD_BYTES
API_KEY = os.environ.get("DATAFACTORY_API_KEY", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One sweep at boot clears whatever a previous container left behind; a
    # real deployment should also run it on a timer or a cron sidecar.
    WORKSPACE.sweep()
    yield
    JOBS.shutdown()


app = FastAPI(
    title="Data Factory",
    version=df.__version__,
    summary="Dataset preparation for computer vision: convert, clean, augment, analyse.",
    lifespan=lifespan,
)

_origins = [o.strip() for o in os.environ.get("DATAFACTORY_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["X-API-Key", "Content-Type"],
    )


async def require_key(x_api_key: str = Header(default="")) -> None:
    """Optional shared-secret gate.

    Off by default so the tool runs locally with no ceremony. Set
    DATAFACTORY_API_KEY before exposing it to the internet -- without it,
    anyone can spend your CPU and disk.
    """
    if not API_KEY:
        return
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "invalid or missing X-API-Key")


# ----------------------------------------------------------------- models

class RunRequest(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)
    input: str | None = Field(default=None, description="'source', or an earlier output name")
    output_name: str = Field(default="", description="what to call the result")
    dry_run: bool = False


# ----------------------------------------------------------------- meta

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "version": df.__version__, "operations": len(registry.all_operations())}


@app.get("/api/operations", dependencies=[Depends(require_key)])
def list_operations() -> dict[str, Any]:
    """Every operation, described so a front-end can generate its own forms.

    Path parameters are omitted: the client cannot set them, so rendering an
    input for one would only produce a request the service refuses.
    """
    return binding.public_schema()


@app.get("/api/operations/{name}", dependencies=[Depends(require_key)])
def describe_operation(name: str) -> dict[str, Any]:
    try:
        return binding.describe_operation(registry.get(name))
    except KeyError:
        raise HTTPException(404, f"unknown operation {name!r}") from None


# ----------------------------------------------------------------- datasets

@app.post("/api/datasets", dependencies=[Depends(require_key)], status_code=201)
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a zipped dataset and get back an opaque id.

    The id is the only handle a client ever holds; it never learns a path.
    """
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(415, "upload a .zip archive of your dataset")

    dataset = WORKSPACE.create()
    staged = Path(tempfile.mkdtemp()) / "upload.zip"
    written = 0
    try:
        with staged.open("wb") as handle:
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > MAX_UPLOAD:
                    raise HTTPException(
                        413, f"upload exceeds the {MAX_UPLOAD / 1e6:.0f} MB limit"
                    )
                handle.write(chunk)
        stats = extract_zip(staged, dataset.source, workspace=WORKSPACE)
    except HTTPException:
        WORKSPACE.delete(dataset.id)
        raise
    except WorkspaceError as exc:
        WORKSPACE.delete(dataset.id)
        raise HTTPException(400, str(exc)) from None
    finally:
        shutil.rmtree(staged.parent, ignore_errors=True)

    return {"id": dataset.id, "uploaded": stats, **_describe(dataset)}


@app.get("/api/datasets/{dataset_id}", dependencies=[Depends(require_key)])
def dataset_info(dataset_id: str) -> dict[str, Any]:
    return {"id": dataset_id, **_describe(_dataset(dataset_id))}


@app.delete("/api/datasets/{dataset_id}", dependencies=[Depends(require_key)], status_code=204)
def delete_dataset(dataset_id: str) -> None:
    _dataset(dataset_id)
    WORKSPACE.delete(dataset_id)


@app.get("/api/datasets/{dataset_id}/download", dependencies=[Depends(require_key)])
def download(dataset_id: str, output: str = "") -> FileResponse:
    """Zip an output (or the source) for the browser to save."""
    dataset = _dataset(dataset_id)
    try:
        source = binding.resolve_input(dataset, output or "source", WORKSPACE)
        archive = make_zip(source, dataset.root / "download" / f"{output or 'source'}.zip",
                           workspace=WORKSPACE)
    except (WorkspaceError, binding.BindingError) as exc:
        raise HTTPException(404, str(exc)) from None
    return FileResponse(
        archive,
        media_type="application/zip",
        filename=f"{output or 'dataset'}-{dataset_id[:8]}.zip",
    )


# ----------------------------------------------------------------- running

@app.post("/api/datasets/{dataset_id}/run/{name}", dependencies=[Depends(require_key)])
def run_operation(dataset_id: str, name: str, request: RunRequest) -> dict[str, Any]:
    """Start an operation. Returns a job id immediately.

    Validation happens synchronously so a bad request fails now, with the same
    message the CLI would print, rather than as a failed job thirty seconds
    later.
    """
    dataset = _dataset(dataset_id)
    try:
        operation = registry.get(name)
    except KeyError:
        raise HTTPException(404, f"unknown operation {name!r}") from None

    try:
        bound, output_dir = binding.bind(
            operation,
            request.params,
            dataset=dataset,
            workspace=WORKSPACE,
            input_ref=request.input,
            output_name=request.output_name,
        )
    except binding.BindingError as exc:
        raise HTTPException(400, str(exc)) from None
    except ParamError as exc:
        raise HTTPException(422, str(exc)) from None
    except WorkspaceError as exc:
        raise HTTPException(400, str(exc)) from None

    dry_run = request.dry_run
    job_id = secrets.token_hex(8)

    def work(context):
        context.dry_run = dry_run
        result = operation.func(bound, context)
        # Provenance is recorded by Operation.run, which we bypass to keep the
        # already-bound parameters, so record it explicitly here.
        _record(operation.name, bound, result)
        return result

    job = JOBS.submit(
        job_id,
        operation.name,
        dataset_id,
        work,
        output_name=output_dir.name if output_dir else "",
    )
    return job.as_dict()


@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_key)])
def job_status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.as_dict()


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(require_key)])
def cancel_job(job_id: str) -> dict[str, Any]:
    """Cooperative cancellation: the run stops at its next checkpoint."""
    if not JOBS.cancel(job_id):
        raise HTTPException(409, "that job has already finished")
    return {"id": job_id, "state": "cancelling"}


@app.get("/api/datasets/{dataset_id}/jobs", dependencies=[Depends(require_key)])
def dataset_jobs(dataset_id: str) -> dict[str, Any]:
    _dataset(dataset_id)
    return {"jobs": [j.as_dict() for j in JOBS.for_dataset(dataset_id)]}


# ----------------------------------------------------------------- helpers

STATIC = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The built-in UI.

    It builds its own forms from ``/api/operations``, so an operation added to
    the Python package appears in the browser with no front-end change -- the
    same generation the CLI and the menu get.
    """
    return FileResponse(STATIC / "index.html", media_type="text/html")


def _dataset(dataset_id: str):
    try:
        return WORKSPACE.get(dataset_id)
    except WorkspaceError as exc:
        raise HTTPException(404, str(exc)) from None


def _describe(dataset) -> dict[str, Any]:
    """What the dataset is, as far as we can tell, for the UI to show."""
    payload: dict[str, Any] = {
        "inputs": binding.available_inputs(dataset),
        "size_bytes": dataset.size_bytes(),
        "expires_in_seconds": max(
            0, int(dataset.created_at() + WORKSPACE.ttl_seconds - time.time())
        ),
    }
    try:
        loaded = formats.read_dataset(dataset.source, "auto", recursive=True)
        payload["format"] = loaded.source_format
        payload["stats"] = loaded.stats()
        payload["classes"] = loaded.labels
        payload["warnings"] = loaded.warnings[:10]
    except AnnotationError as exc:
        # Not fatal: plenty of useful operations work on a folder of images
        # with no annotations at all.
        payload["format"] = None
        payload["detection_error"] = str(exc)
    return payload


def _record(name: str, bound: dict[str, Any], result) -> None:
    try:
        from ..core import provenance

        provenance.record(name, bound, result, version=df.__version__)
    except Exception:
        pass

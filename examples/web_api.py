"""A minimal FastAPI wrapper over Data Factory.

Demonstrates the point of the core/adapter split: this file contains no dataset
logic at all. It exposes the operation registry as an API, and a front-end can
render a form for any operation -- including ones added later -- from
``GET /operations`` without this file changing.

    pip install fastapi uvicorn
    uvicorn examples.web_api:app --reload

Endpoints
    GET  /operations           every operation and its parameters (form schema)
    GET  /operations/{name}    one operation
    POST /run/{name}           run it, returns the OpResult as JSON
    POST /jobs/{name}          start it in the background, returns a job id
    GET  /jobs/{job_id}        progress, and the result once finished
    POST /jobs/{job_id}/cancel ask a running job to stop

NOTE: this is an example, not a hardened service. Anything that accepts a
filesystem path from a browser needs authentication and path allow-listing
before it goes anywhere near a network.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

import data_factory as df
from data_factory import Context
from data_factory.core.registry import ParamError

app = FastAPI(title="Data Factory API", version=df.__version__)


class RunRequest(BaseModel):
    params: dict[str, Any] = {}
    dry_run: bool = False


# ---------------------------------------------------------------- discovery

@app.get("/operations")
def list_operations() -> dict[str, Any]:
    """The whole tool, described. A UI builds its forms from this."""
    return df.schema()


@app.get("/operations/{name}")
def describe_operation(name: str) -> dict[str, Any]:
    try:
        return df.get_operation(name).as_dict()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None


# ---------------------------------------------------------- synchronous run

@app.post("/run/{name}")
def run_operation(name: str, request: RunRequest) -> dict[str, Any]:
    """Run and wait. Fine for audits; use /jobs for anything long."""
    try:
        operation = df.get_operation(name)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None

    try:
        # Validation is the core's, not ours -- identical to the CLI's.
        bound = operation.bind(request.params)
    except ParamError as exc:
        raise HTTPException(422, str(exc)) from None

    if operation.destructive_for(bound) and not request.dry_run:
        # Make the caller opt in explicitly; a stray POST should not delete data.
        if not request.params.get("confirm"):
            raise HTTPException(
                409,
                f"{name} modifies files. Re-send with dry_run=true to preview, "
                "or confirm=true to proceed.",
            )

    result = operation.run(request.params, Context(dry_run=request.dry_run))
    return result.as_dict(include_actions=False)


# --------------------------------------------------------------- background

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


@app.post("/jobs/{name}")
def start_job(name: str, request: RunRequest, background: BackgroundTasks) -> dict[str, str]:
    """Start a long operation and return immediately with a job id."""
    try:
        operation = df.get_operation(name)
        operation.bind(request.params)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None
    except ParamError as exc:
        raise HTTPException(422, str(exc)) from None

    job_id = uuid.uuid4().hex
    with _LOCK:
        _JOBS[job_id] = {"state": "queued", "done": 0, "total": 0, "message": "", "cancel": False}

    def progress(done: int, total: int, message: str = "") -> None:
        with _LOCK:
            _JOBS[job_id].update(done=done, total=total, message=message)

    def cancelled() -> bool:
        with _LOCK:
            return _JOBS[job_id]["cancel"]

    def work() -> None:
        with _LOCK:
            _JOBS[job_id]["state"] = "running"
        ctx = Context(dry_run=request.dry_run, progress=progress, cancel_check=cancelled)
        try:
            result = operation.run(request.params, ctx)
            with _LOCK:
                _JOBS[job_id].update(state="finished", result=result.as_dict(include_actions=False))
        except Exception as exc:  # surfaced to the caller rather than swallowed
            with _LOCK:
                _JOBS[job_id].update(state="failed", error=str(exc))

    background.add_task(work)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, str]:
    """Cooperative cancellation -- the operation stops at its next checkpoint."""
    with _LOCK:
        if job_id not in _JOBS:
            raise HTTPException(404, "unknown job")
        _JOBS[job_id]["cancel"] = True
    return {"status": "cancelling"}

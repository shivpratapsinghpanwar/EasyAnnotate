"""Background jobs with progress and cancellation.

Dataset work outlives an HTTP request -- augmenting 3,000 images takes minutes,
and a browser will not hold a connection open for that. So a run returns a job
id immediately and the client polls.

This is a single-process, in-memory manager, which is the right size for one
container serving a modest site. It is deliberately shaped so that swapping in
Redis or Celery later means replacing this file and nothing else: the rest of
the service only ever calls ``submit``, ``get`` and ``cancel``.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.types import Cancelled, Context, OpResult

#: Runs at once. Dataset work is CPU-bound and already multi-threaded inside
#: each operation, so letting several run concurrently mostly adds contention.
MAX_CONCURRENT = 2

#: Finished jobs kept for polling before being evicted.
MAX_REMEMBERED = 500


@dataclass
class Job:
    id: str
    operation: str
    dataset_id: str
    state: str = "queued"           # queued | running | finished | failed | cancelled
    done: int = 0
    total: int = 0
    message: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    result: dict[str, Any] | None = None
    error: str = ""
    output_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        elapsed = (self.finished_at or time.time()) - (self.started_at or self.created_at)
        return {
            "id": self.id,
            "operation": self.operation,
            "dataset": self.dataset_id,
            "state": self.state,
            "progress": {
                "done": self.done,
                "total": self.total,
                "message": self.message,
                # A percentage the UI can render without dividing by zero.
                "percent": round(100 * self.done / self.total, 1) if self.total else None,
            },
            "elapsed_seconds": round(elapsed, 1),
            "output": self.output_name or None,
            "result": self.result,
            "error": self.error or None,
        }


class JobManager:
    """Runs operations off the request thread, with progress and cancellation."""

    def __init__(self, max_concurrent: int = MAX_CONCURRENT) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="dfx-job")
        self._lock = threading.Lock()
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._cancelled: set[str] = set()

    # -- api ---------------------------------------------------------------
    def submit(
        self,
        job_id: str,
        operation: str,
        dataset_id: str,
        work: Callable[[Context], OpResult],
        *,
        output_name: str = "",
    ) -> Job:
        job = Job(id=job_id, operation=operation, dataset_id=dataset_id, output_name=output_name)
        with self._lock:
            self._jobs[job_id] = job
            self._evict_locked()
        self._pool.submit(self._run, job, work)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state in ("finished", "failed", "cancelled"):
                return False
            self._cancelled.add(job_id)
            if job.state == "queued":
                # Never started, so nothing will poll the cancel flag for us.
                job.state = "cancelled"
                job.finished_at = time.time()
            return True

    def for_dataset(self, dataset_id: str) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.dataset_id == dataset_id]

    # -- internals ---------------------------------------------------------
    def _run(self, job: Job, work: Callable[[Context], OpResult]) -> None:
        with self._lock:
            if job.id in self._cancelled:
                job.state = "cancelled"
                job.finished_at = time.time()
                return
            job.state = "running"
            job.started_at = time.time()

        def progress(done: int, total: int, message: str = "") -> None:
            with self._lock:
                job.done, job.total, job.message = done, total, message

        def cancelled() -> bool:
            with self._lock:
                return job.id in self._cancelled

        context = Context(progress=progress, cancel_check=cancelled)
        try:
            result = work(context)
            with self._lock:
                # Actions are the per-file audit trail: useful in a terminal,
                # far too large to ship to a browser for a 10,000-file run.
                job.result = result.as_dict(include_actions=False)
                job.state = "finished" if result.ok else "failed"
                if not result.ok:
                    job.error = "; ".join(result.errors) or "the operation reported failure"
                job.finished_at = time.time()
        except Cancelled:
            with self._lock:
                job.state = "cancelled"
                job.finished_at = time.time()
        except Exception as exc:
            # The traceback goes to the log, not to the client: it can carry
            # host paths and package layout that a public API should not leak.
            traceback.print_exc()
            with self._lock:
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = time.time()
        finally:
            with self._lock:
                self._cancelled.discard(job.id)

    def _evict_locked(self) -> None:
        while len(self._jobs) > MAX_REMEMBERED:
            for key, job in self._jobs.items():
                if job.state in ("finished", "failed", "cancelled"):
                    del self._jobs[key]
                    break
            else:
                return

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def output_exists(path: Path | None) -> bool:
    return bool(path and path.is_dir() and any(path.iterdir()))

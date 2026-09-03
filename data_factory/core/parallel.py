"""Parallel map for the decode-bound stages.

Perceptual hashing 2,783 images took 67 seconds, and essentially all of it was
JPEG decode -- one core busy, the rest idle. Decoding is embarrassingly
parallel, so this is the piece that was missing rather than a smarter
algorithm; the BK-tree already fixed the comparison cost.

Threads, not processes. OpenCV and Pillow release the GIL around decode, so
threads get real parallelism here, and they avoid pickling a few thousand image
paths and hash objects across a process boundary. A process pool would also
break the Windows entry-point rules for a library that gets imported by a web
server.

Two properties matter more than the speed:

*Order is preserved.* Dedup keeps the first image it sees, so results that
depended on which thread finished first would not be reproducible. Results come
back in input order regardless of completion order.

*Cancellation and progress still work.* Both are polled from the calling
thread as results arrive, so a web request can still abort a long run.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from .types import Context

T = TypeVar("T")
R = TypeVar("R")

#: Past this, threads contend on memory bandwidth rather than adding speed, and
#: a 32-core machine should not open 32 large JPEGs at once.
MAX_WORKERS = 8


def resolve_workers(requested: int) -> int:
    """Turn a ``--workers`` value into an actual thread count.

    0 means "decide for me": one per core, capped, and always leaving the
    machine usable.
    """
    if requested and requested > 0:
        return max(1, min(int(requested), 32))
    cores = os.cpu_count() or 2
    return max(1, min(cores - 1, MAX_WORKERS))


def map_ordered(
    items: Sequence[T],
    work: Callable[[T], R],
    *,
    ctx: Context,
    workers: int = 0,
    label: Callable[[T], str] | None = None,
    on_error: Callable[[T, Exception], None] | None = None,
) -> Iterator[tuple[T, R | None]]:
    """Apply ``work`` to every item, yielding ``(item, result)`` in input order.

    Items whose work raised yield ``None`` after ``on_error`` is called, so a
    single corrupt file cannot abort a scan of twenty thousand.
    """
    total = len(items)
    if total == 0:
        return

    count = resolve_workers(workers)
    if count == 1 or total == 1:
        # Serial path: identical semantics, no thread pool overhead, and a much
        # easier thing to step through in a debugger.
        for position, item in enumerate(items, start=1):
            ctx.check_cancelled()
            ctx.emit(position, total, label(item) if label else "")
            try:
                yield item, work(item)
            except Exception as exc:
                if on_error is not None:
                    on_error(item, exc)
                yield item, None
        return

    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(work, item) for item in items]
        for position, (item, future) in enumerate(zip(items, futures), start=1):
            try:
                ctx.check_cancelled()
            except BaseException:
                # Stop the queue draining into work nobody will read. Already
                # running tasks finish; the rest never start.
                for pending in futures[position - 1 :]:
                    pending.cancel()
                raise
            ctx.emit(position, total, label(item) if label else "")
            try:
                yield item, future.result()
            except Exception as exc:
                if on_error is not None:
                    on_error(item, exc)
                yield item, None


def collect(
    items: Sequence[T],
    work: Callable[[T], R],
    *,
    ctx: Context,
    workers: int = 0,
    label: Callable[[T], str] | None = None,
    on_error: Callable[[T, Exception], None] | None = None,
) -> list[tuple[T, R]]:
    """:func:`map_ordered` with the failures dropped, as a list."""
    return [
        (item, value)
        for item, value in map_ordered(
            items, work, ctx=ctx, workers=workers, label=label, on_error=on_error
        )
        if value is not None
    ]


WORKERS_HELP = (
    "Parallel worker threads for image decoding (0 = choose automatically)."
)

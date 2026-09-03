"""Dataset lineage: what was done to this folder, and with what settings.

Six weeks after a dataset is built, the question that matters is "what is this,
exactly" -- was it augmented, at what seed, was it resized before or after the
split, which source folders went into the merge. Without a record that question
is unanswerable, and the usual answer is to rebuild the dataset from scratch
and hope.

So every operation that writes a dataset appends one line to
``<output>/.datafactory/history.jsonl``. It is recorded in
:meth:`Operation.run`, not by the operations themselves, which means no
operation can forget to do it and no new operation has to remember.

JSON Lines rather than a single JSON document, because appending must not
require reading and rewriting the file -- a build that dies halfway still
leaves a readable, truthful history of what it managed to do.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .types import OpResult, _jsonable

HISTORY_DIR = ".datafactory"
HISTORY_FILE = "history.jsonl"

#: Parameters naming where output goes. Whichever is present decides which
#: folder gets the history entry.
OUTPUT_KEYS = ("output", "quarantine")

#: Never copy these into a record -- they are noise, or they are large.
SKIP_KEYS = frozenset({"workers", "quality", "recursive"})


def history_path(folder: Path) -> Path:
    return Path(folder) / HISTORY_DIR / HISTORY_FILE


def record(
    op_name: str,
    params: dict[str, Any],
    result: OpResult,
    *,
    version: str = "",
) -> Path | None:
    """Append one entry to the output folder's history. Returns the file written.

    Silent no-op when the run was a dry run, failed, or produced no output
    folder -- there is nothing to describe the provenance *of*.
    """
    if result.dry_run or not result.ok:
        return None
    folder = _output_folder(params)
    if folder is None or not folder.is_dir():
        return None

    entry = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "op": op_name,
        "version": version,
        "params": _describe(params),
        "counts": dict(result.counts),
        "warnings": len(result.warnings),
        "produced": _jsonable(result.data.get("format")) or None,
    }

    target = history_path(folder)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        # Provenance is valuable but never worth failing a completed build for.
        return None
    return target


def read(folder: Path) -> list[dict[str, Any]]:
    """Read a folder's history, skipping any line that got truncated."""
    path = history_path(folder)
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _output_folder(params: dict[str, Any]) -> Path | None:
    for key in OUTPUT_KEYS:
        value = params.get(key)
        if isinstance(value, Path):
            return value
    return None


def _describe(params: dict[str, Any]) -> dict[str, Any]:
    """The settings worth keeping, as JSON.

    Defaults are kept as well as overrides: "what seed was this augmented at"
    has to be answerable without also knowing what the default was in the
    version that built it.
    """
    described: dict[str, Any] = {}
    for key, value in params.items():
        if key in SKIP_KEYS or value is None:
            continue
        if isinstance(value, Path):
            described[key] = _short(value)
        elif isinstance(value, (list, tuple)):
            described[key] = [_short(v) if isinstance(v, Path) else v for v in value]
        else:
            described[key] = _jsonable(value)
    return described


def _short(path: Path) -> str:
    """Absolute paths, but relative to the working directory when that is
    shorter -- a history full of D:/long/absolute/prefixes is unreadable."""
    text = str(path)
    try:
        relative = os.path.relpath(path, Path.cwd())
    except (ValueError, OSError):
        return text
    return relative if len(relative) < len(text) and not relative.startswith("..") else text

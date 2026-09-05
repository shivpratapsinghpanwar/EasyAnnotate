"""Filesystem helpers, and the single choke point through which every
mutation passes.

Operations never call ``shutil`` directly. They record a :class:`FileAction`
and hand it to :func:`perform`, which is a no-op under ``--dry-run``. That is
why a preview is guaranteed to match the real run: there is only one code path,
and the only thing dry-run changes is whether the last line executes.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from .types import Action, Context, FileAction

IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"})
DEFAULT_LABEL_EXT = ".xml"


def write_text_atomic(
    path: Path, text: str, *, encoding: str = "utf-8", newline: str | None = None
) -> Path:
    """Write a file so a reader never sees it half-written.

    The obvious ``path.write_text(...)`` truncates the file first and then
    fills it. Two things go wrong with that, and both are silent:

    * **A crash or a full disk mid-write leaves a truncated file.** For an XML
      or JSON annotation that means an unparseable label -- and the original
      contents are already gone, so there is nothing to fall back to.
    * **Two concurrent writers interleave.** The annotation editor writes
      through to sidecar files on every save; a second tab, or an export
      running at the same time, would produce a mangled file rather than
      one of the two intended ones.

    Writing to a temporary file in the same directory and then renaming makes
    the swap atomic on both POSIX and Windows: a reader sees either the whole
    old file or the whole new one. Same directory matters -- ``os.replace``
    across filesystems is not atomic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``newline`` defaults to None so this matches Path.write_text exactly:
    # swapping in an atomic write must not also quietly change line endings in
    # every label file the tool has ever produced. Callers that already emit
    # their own terminators -- the csv module writes \r\n itself -- pass "",
    # which is the difference between a valid CSV and one with a blank line
    # between every row on Windows.
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        newline=newline,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            # Without this the rename can land before the contents reach disk,
            # which on a power loss gives an empty file where one used to be.
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


#: How long to keep retrying a rename that Windows refuses. Deliberately short:
#: this is for transient holds, and anything longer is a real problem that
#: should surface rather than hang a save.
_REPLACE_ATTEMPTS = 10
_REPLACE_PAUSE = 0.05


def _replace_with_retry(temporary: Path, path: Path) -> None:
    """``os.replace``, retried past Windows' transient sharing failures.

    On POSIX the rename is unconditional. On Windows it goes through
    ``MoveFileEx``, which fails with ``PermissionError`` (``WinError 5``) if
    *anything* currently has the destination open without ``FILE_SHARE_DELETE``
    -- another thread reading it, an editor, a backup agent, or an antivirus
    scanner that opened the file precisely because we just wrote it.

    Found by the concurrency test: two threads writing and reading the same
    file made this fail within a few hundred iterations. Left unhandled it
    would surface as an annotation save that failed for no visible reason, on
    a machine with antivirus, intermittently -- the worst kind of bug to be
    told about second-hand.
    """
    last: OSError | None = None
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:  # Windows only, and transient
            last = exc
            time.sleep(_REPLACE_PAUSE * (attempt + 1))
    assert last is not None
    raise last


def is_image(path: Path, exts: Iterable[str] = IMAGE_EXTS) -> bool:
    return path.suffix.lower() in set(exts)


def iter_images(
    root: Path, *, recursive: bool = False, exts: Iterable[str] = IMAGE_EXTS
) -> Iterator[Path]:
    """Yield image files under ``root`` in a stable, reproducible order.

    Sorting matters more than it looks: dedup keeps the *first* file it sees, so
    an unstable directory order would make results differ between runs.
    """
    extset = {e.lower() for e in exts}
    walker = root.rglob("*") if recursive else root.glob("*")
    for path in sorted(walker):
        if path.is_file() and path.suffix.lower() in extset:
            yield path


def iter_files(
    root: Path, *, recursive: bool = False, exts: Iterable[str] | None = None
) -> Iterator[Path]:
    extset = {e.lower() for e in exts} if exts else None
    walker = root.rglob("*") if recursive else root.glob("*")
    for path in sorted(walker):
        if not path.is_file():
            continue
        if extset is None or path.suffix.lower() in extset:
            yield path


def ensure_dir(path: Path, ctx: Context) -> Path:
    if not ctx.dry_run:
        path.mkdir(parents=True, exist_ok=True)
    return path


def unique_path(dst: Path, *, taken: set[Path] | None = None) -> Path:
    """Return a destination that collides with nothing.

    ``taken`` lets a dry run reserve names it has not actually created yet, so
    the preview does not show three files all landing on the same path.
    """
    reserved = taken if taken is not None else set()
    if not dst.exists() and dst not in reserved:
        return dst
    stem, suffix, parent = dst.stem, dst.suffix, dst.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists() and candidate not in reserved:
            return candidate
        counter += 1


def perform(action: FileAction, ctx: Context) -> None:
    """Apply one recorded action. Does nothing when ``ctx.dry_run`` is set."""
    if ctx.dry_run or action.action is Action.SKIP:
        return
    if action.action is Action.DELETE:
        action.src.unlink(missing_ok=True)
        return
    if action.dst is None:
        raise ValueError(f"{action.action.value} action requires a destination")
    action.dst.parent.mkdir(parents=True, exist_ok=True)
    if action.action in (Action.MOVE, Action.RENAME):
        shutil.move(str(action.src), str(action.dst))
    elif action.action is Action.COPY:
        shutil.copy2(str(action.src), str(action.dst))


def sha256_file(path: Path, chunk_size: int = 1 << 16) -> str:
    """Content fingerprint used for exact-duplicate detection."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def relative_to_safe(path: Path, root: Path) -> Path:
    """``path.relative_to(root)`` that degrades to the bare name off-tree."""
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)

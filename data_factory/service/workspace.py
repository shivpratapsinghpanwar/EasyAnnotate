"""Sandboxed per-dataset storage for the hosted service.

The example API shipped with this project takes filesystem paths straight from
the request body. That is fine on a laptop and catastrophic on a public host:
``{"folder": "/etc"}`` is an arbitrary-read primitive, and ``{"output": "/"}``
is worse. Nothing in this module trusts a client-supplied path.

Two rules, enforced here rather than remembered at each call site:

1. A client never names a directory. It names a **dataset id** -- an opaque
   token this module issued -- and every path is derived from that.
2. Every derived path is resolved and checked to be inside the workspace root
   before it is used, so symlinks and ``..`` cannot escape.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

#: Ids are opaque and unguessable: possession of the id is the only
#: authorisation the anonymous flow has, so it must not be enumerable.
ID_BYTES = 16
ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

#: Upload limits. A zip bomb is a small file that expands to fill the disk, so
#: the compressed size alone is not a sufficient guard.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024        # 2 GB compressed
MAX_EXTRACTED_BYTES = 8 * 1024 * 1024 * 1024     # 8 GB uncompressed
MAX_MEMBERS = 200_000
MAX_RATIO = 200                                  # uncompressed / compressed

#: Anonymous datasets are transient. Anything older than this is swept.
DEFAULT_TTL_SECONDS = 24 * 60 * 60

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


class WorkspaceError(Exception):
    """Raised for anything a client did wrong with storage."""


@dataclass(frozen=True)
class Dataset:
    """One sandboxed dataset directory."""

    id: str
    root: Path

    @property
    def source(self) -> Path:
        """Where the uploaded dataset lives. Operations read from here."""
        return self.root / "source"

    @property
    def outputs(self) -> Path:
        """Parent of every generated dataset. Operations write under here."""
        return self.root / "outputs"

    def output(self, name: str) -> Path:
        """A named output directory, with the name sanitised to one segment."""
        return self.outputs / safe_segment(name)

    def created_at(self) -> float:
        try:
            return self.root.stat().st_mtime
        except OSError:
            return 0.0

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())


class Workspace:
    """The root under which every dataset sandbox lives."""

    def __init__(self, root: Path, *, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds

    # -- lifecycle ---------------------------------------------------------
    def create(self) -> Dataset:
        dataset_id = secrets.token_hex(ID_BYTES)
        dataset = Dataset(dataset_id, self.root / dataset_id)
        dataset.source.mkdir(parents=True, exist_ok=True)
        dataset.outputs.mkdir(parents=True, exist_ok=True)
        return dataset

    def get(self, dataset_id: str) -> Dataset:
        """Resolve an id to a sandbox, rejecting anything that is not one.

        The pattern check matters more than it looks: without it, an id of
        ``..`` would resolve to the workspace parent.
        """
        if not ID_PATTERN.match(dataset_id or ""):
            raise WorkspaceError("not a valid dataset id")
        root = (self.root / dataset_id).resolve()
        self.assert_inside(root)
        if not root.is_dir():
            raise WorkspaceError("unknown dataset -- it may have expired")
        return Dataset(dataset_id, root)

    def delete(self, dataset_id: str) -> None:
        dataset = self.get(dataset_id)
        shutil.rmtree(dataset.root, ignore_errors=True)

    def sweep(self) -> int:
        """Remove datasets past their TTL. Returns how many were removed."""
        cutoff = time.time() - self.ttl_seconds
        removed = 0
        for child in self.root.iterdir():
            if not child.is_dir() or not ID_PATTERN.match(child.name):
                continue
            try:
                if child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        return removed

    def datasets(self) -> Iterator[Dataset]:
        for child in sorted(self.root.iterdir()):
            if child.is_dir() and ID_PATTERN.match(child.name):
                yield Dataset(child.name, child)

    # -- the security boundary --------------------------------------------
    def assert_inside(self, path: Path) -> Path:
        """Fail unless ``path`` really is under the workspace root.

        Uses the resolved path, so a symlink planted inside a sandbox cannot
        redirect a write to somewhere else on the host.
        """
        resolved = Path(path).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceError("path escapes the workspace")
        return resolved


def safe_segment(name: str) -> str:
    """Reduce arbitrary text to one filesystem-safe path segment."""
    cleaned = SAFE_NAME.sub("_", (name or "").strip())[:64].strip("._")
    return cleaned or "output"


# ------------------------------------------------------------------ uploads

def extract_zip(archive: Path, destination: Path, *, workspace: Workspace) -> dict[str, int]:
    """Unpack an uploaded zip safely.

    Guards, in the order they matter:

    * **Zip slip** -- a member named ``../../etc/cron.d/x`` writes outside the
      target. Every member is resolved and checked against the destination.
    * **Zip bomb** -- a few kB expanding to fill the disk. Both the total
      uncompressed size and the per-member compression ratio are capped.
    * **Member count** -- a million tiny files is its own denial of service.
    """
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = workspace.assert_inside(destination)

    written = 0
    total = 0
    skipped = 0

    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if len(members) > MAX_MEMBERS:
                raise WorkspaceError(f"archive holds more than {MAX_MEMBERS:,} entries")

            declared = sum(m.file_size for m in members)
            if declared > MAX_EXTRACTED_BYTES:
                raise WorkspaceError(
                    f"archive expands to {declared / 1e9:.1f} GB, over the "
                    f"{MAX_EXTRACTED_BYTES / 1e9:.0f} GB limit"
                )

            for member in members:
                if member.is_dir():
                    continue
                name = member.filename.replace("\\", "/")
                if name.startswith("/") or ".." in Path(name).parts:
                    skipped += 1
                    continue
                # Some archives carry junk; keeping it would confuse format
                # detection more than dropping it loses anything.
                if Path(name).name.startswith("._") or "__MACOSX/" in name:
                    skipped += 1
                    continue

                target = (resolved_destination / name).resolve()
                if resolved_destination not in target.parents and target != resolved_destination:
                    skipped += 1
                    continue

                if member.compress_size and member.file_size / max(1, member.compress_size) > MAX_RATIO:
                    raise WorkspaceError(f"{name}: suspicious compression ratio, refusing to extract")

                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as src, target.open("wb") as dst:
                    # Copy in chunks and stop at the cap, rather than trusting
                    # the header's declared size, which an attacker controls.
                    while chunk := src.read(1 << 20):
                        total += len(chunk)
                        if total > MAX_EXTRACTED_BYTES:
                            raise WorkspaceError("archive exceeded the uncompressed size limit")
                        dst.write(chunk)
                written += 1
    except zipfile.BadZipFile as exc:
        raise WorkspaceError(f"not a readable zip archive ({exc})") from exc

    if written == 0:
        raise WorkspaceError("the archive contained no usable files")
    return {"files": written, "bytes": total, "skipped": skipped}


def make_zip(source: Path, archive: Path, *, workspace: Workspace) -> Path:
    """Bundle a result directory for download."""
    workspace.assert_inside(source)
    if not source.is_dir():
        raise WorkspaceError("nothing to download yet")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(source).as_posix())
    return archive


def resolve_workspace_root() -> Path:
    """Where datasets live, overridable for containers and tests."""
    configured = os.environ.get("DATAFACTORY_WORKSPACE")
    if configured:
        return Path(configured)
    return Path.home() / ".datafactory" / "workspaces"

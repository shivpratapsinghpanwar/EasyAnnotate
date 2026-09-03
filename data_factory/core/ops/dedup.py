"""Duplicate removal -- exact and perceptual, single- or cross-folder.

This replaces two scripts that had diverged: one did cross-folder SHA-256 and
carried the sidecar XML along, the other did single-folder perceptual hashing
and had the XML-moving code commented out. Both behaviours are options here, so
a near-duplicate can never again be dropped while its annotation is left
behind.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import fs, parallel
from ..bktree import BKTree
from ..pairs import PairIndex, sidecar_for
from ..registry import Param, operation
from ..types import Action, Context, OpResult

#: Perceptual algorithms, resolved lazily so importing the package does not
#: require Pillow/imagehash unless a perceptual run is actually requested.
PERCEPTUAL = ("phash", "dhash", "ahash", "whash")

PARAMS = (
    Param("folders", "dirs", "Folder(s) to scan. Several are compared against each other.",
          required=True, must_exist=True),
    Param("method", "choice", "How duplicates are identified.",
          default="exact", choices=("exact",) + PERCEPTUAL),
    Param("threshold", "int", "Max Hamming distance for perceptual matches (0 = identical).",
          default=8, minimum=0, maximum=64),
    Param("hash_size", "int", "Perceptual hash size; higher is stricter and slower.",
          default=8, minimum=4, maximum=32),
    Param("action", "choice", "What to do with the duplicates that are found.",
          default="move", choices=("move", "delete", "report")),
    Param("quarantine", "dir", "Where moved duplicates go. Defaults to <first folder>/_duplicates."),
    Param("keep", "choice", "Which member of a duplicate group to keep.",
          default="annotated", choices=("annotated", "first", "largest")),
    Param("recursive", "bool", "Include sub-folders.", default=False),
    Param("move_labels", "bool", "Move or delete the matching sidecar label too.", default=True),
    Param("workers", "int", parallel.WORKERS_HELP, default=0, minimum=0, maximum=32),
)


@operation(
    "clean.dedup",
    title="Remove duplicate images",
    category="clean",
    summary="Find exact or visually similar duplicates and quarantine them, sidecar labels included.",
    params=PARAMS,
    destructive=True,
    safe_when=lambda p: p.get("action") == "report",
    notes=(
        "method=exact hashes file bytes: fast, catches only byte-identical copies.\n"
        "method=phash and friends compare appearance, so re-compressed, resized or\n"
        "lightly cropped copies are caught too -- tune with --threshold (0 is\n"
        "identical, 8 is a good default, above ~12 starts merging genuinely\n"
        "different images).\n\n"
        "keep=annotated is the safe default: when a duplicate group contains both a\n"
        "labelled and an unlabelled copy, the labelled one survives."
    ),
)
def dedup(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="clean.dedup", dry_run=ctx.dry_run)
    folders: list[Path] = params["folders"]
    method: str = params["method"]
    recursive: bool = params["recursive"]
    action: str = params["action"]

    quarantine = params["quarantine"] or (folders[0] / "_duplicates")

    # Gather every image across every folder, remembering which folder it came
    # from so quarantined files can keep their origin in the path.
    entries: list[tuple[Path, Path]] = []
    for folder in folders:
        index = PairIndex.scan(folder, recursive=recursive)
        for pair in index.with_images():
            assert pair.image is not None
            entries.append((folder, pair.image))

    total = len(entries)
    # Seed the headline counters so they are present even at zero. A consumer
    # reading result.as_dict() should never have to guess whether a missing key
    # means "none found" or "this operation does not report that".
    result.counts["scanned"] = total
    result.counts["unique"] = 0
    result.counts["duplicates"] = 0
    if params["action"] == "delete":
        result.counts["files_removed"] = 0
    elif params["action"] == "move":
        result.counts["files_moved"] = 0
    if total == 0:
        result.warnings.append("no images found in the supplied folder(s)")
        return result

    hasher = _exact_hasher() if method == "exact" else _perceptual_hasher(method, params["hash_size"])
    threshold = 0 if method == "exact" else int(params["threshold"])

    groups: dict[Any, list[tuple[Path, Path]]] = defaultdict(list)
    tree: BKTree | None = None
    representatives: dict[Any, Any] = {}

    if method != "exact":
        tree = BKTree(lambda a, b: a - b)

    # Hashing is pure and decode-bound, so it runs across threads; grouping is
    # order-dependent (the first image seen is the one kept) and stays serial.
    # map_ordered yields in input order, so the two phases agree.
    def note_failure(entry: tuple[Path, Path], exc: Exception) -> None:
        result.warnings.append(f"skipped {entry[1].name}: {exc}")
        result.bump("unreadable")

    hashed = parallel.map_ordered(
        entries,
        lambda entry: hasher(entry[1]),
        ctx=ctx,
        workers=params["workers"],
        label=lambda entry: f"hashing {entry[1].name}",
        on_error=note_failure,
    )

    for (folder, image), digest in hashed:
        if digest is None:
            continue

        if method == "exact":
            groups[digest].append((folder, image))
            continue

        assert tree is not None
        match = tree.nearest(digest, threshold)
        if match is None:
            tree.add(digest)
            representatives[_key(digest)] = digest
            groups[_key(digest)].append((folder, image))
        else:
            groups[_key(match[1])].append((folder, image))

    # Every group with more than one member yields duplicates.
    duplicate_groups: dict[str, list[str]] = {}
    taken: set[Path] = set()

    for members in groups.values():
        if len(members) < 2:
            result.bump("unique")
            continue
        keeper, dropped = _choose_keeper(members, params["keep"], params["move_labels"])
        result.bump("unique")
        duplicate_groups[str(keeper[1])] = [str(p) for _, p in dropped]

        for folder, image in dropped:
            result.bump("duplicates")
            targets = [image]
            if params["move_labels"]:
                label = sidecar_for(image)
                if label.exists():
                    targets.append(label)

            for target in targets:
                if action == "report":
                    result.record(Action.SKIP, target, None, f"duplicate of {keeper[1].name}")
                    continue
                if action == "delete":
                    file_action = result.record(
                        Action.DELETE, target, None, f"duplicate of {keeper[1].name}"
                    )
                else:
                    relative = fs.relative_to_safe(target, folder)
                    destination = fs.unique_path(quarantine / folder.name / relative, taken=taken)
                    taken.add(destination)
                    file_action = result.record(
                        Action.MOVE, target, destination, f"duplicate of {keeper[1].name}"
                    )
                fs.perform(file_action, ctx)
                result.bump("files_removed" if action == "delete" else "files_moved")

    result.data["groups"] = duplicate_groups
    result.data["method"] = method
    result.data["threshold"] = threshold
    if action == "move":
        result.data["quarantine"] = quarantine
    return result


def _key(digest: Any) -> str:
    return str(digest)


def _choose_keeper(
    members: list[tuple[Path, Path]], policy: str, consider_labels: bool
) -> tuple[tuple[Path, Path], list[tuple[Path, Path]]]:
    """Pick the survivor of a duplicate group; the rest are dropped."""
    if policy == "annotated" and consider_labels:
        # Prefer a copy that already has an annotation -- discarding the
        # labelled twin and keeping a bare image throws away real work.
        annotated = [m for m in members if sidecar_for(m[1]).exists()]
        if annotated:
            keeper = annotated[0]
            return keeper, [m for m in members if m is not keeper]
    if policy == "largest":
        keeper = max(members, key=lambda m: m[1].stat().st_size)
        return keeper, [m for m in members if m is not keeper]
    keeper = members[0]
    return keeper, list(members[1:])


def _exact_hasher() -> Callable[[Path], str]:
    return fs.sha256_file


def _perceptual_hasher(method: str, hash_size: int) -> Callable[[Path], Any]:
    try:
        import imagehash
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            f"perceptual dedup needs Pillow and imagehash installed ({exc})"
        ) from exc

    functions = {
        "phash": imagehash.phash,
        "dhash": imagehash.dhash,
        "ahash": imagehash.average_hash,
        "whash": imagehash.whash,
    }
    function = functions[method]

    def compute(path: Path) -> Any:
        with Image.open(path) as image:
            image.load()
            return function(image, hash_size=hash_size)

    return compute

"""Dataset health, and train/validation leakage.

``inspect.dataset`` answers "is this folder well-formed". These two answer the
harder question: "will training on this produce a model whose score means
anything".
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import fs, stats
from ..annotations import formats
from ..annotations.model import AnnotationError
from ..bktree import BKTree
from ..registry import Param, operation
from ..types import Context, OpResult

HEALTH_PARAMS = (
    Param("folder", "dir", "Dataset to assess.", required=True, must_exist=True),
    Param(
        "source_format",
        "choice",
        "Input format; 'auto' sniffs the layout.",
        default="auto",
        choices=("auto",) + formats.FORMATS,
    ),
    Param("recursive", "bool", "Include sub-folders.", default=True),
    Param("top_classes", "int", "How many classes to list in the summary.", default=25, minimum=1),
)


@operation(
    "inspect.health",
    title="Dataset health check",
    category="inspect",
    summary="Class balance, object sizes, geometry and composition -- with ranked findings and fixes.",
    params=HEALTH_PARAMS,
    notes=(
        "Read-only, and works on any supported format.\n"
        "\n"
        "What it is actually checking for, and why each matters:\n"
        "  class imbalance      a 50x gap means the rare class is barely learned\n"
        "  tiny objects         under ~0.05% of the frame, they vanish on resize\n"
        "  case-variant classes 'Open' and 'open' train as two separate classes\n"
        "  zero-area boxes      crash some trainers, are ignored by others\n"
        "  background share     too many object-free images biases toward silence\n"
        "  mixed resolutions    usually harmless, occasionally a sign of trouble\n"
        "\n"
        "Every threshold is a named constant in core/stats.py with its reasoning\n"
        "written down -- they are judgement calls, not laws, and you should\n"
        "disagree with them when your problem warrants it."
    ),
)
def health(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="inspect.health", dry_run=ctx.dry_run)
    folder: Path = params["folder"]

    try:
        dataset = formats.read_dataset(
            folder, params["source_format"], recursive=params["recursive"], ctx=ctx
        )
    except AnnotationError as exc:
        result.fail(str(exc))
        return result

    ctx.emit(1, 1, "analysing")
    payload = stats.analyse(dataset)
    totals = payload["totals"]

    result.counts.update(
        {
            "images": totals["images"],
            "objects": totals["objects"],
            "classes": totals["classes"],
            "backgrounds": totals["backgrounds"],
            "findings": len(payload["findings"]),
            "errors": sum(1 for f in payload["findings"] if f["severity"] == "error"),
            "warnings_found": sum(1 for f in payload["findings"] if f["severity"] == "warning"),
        }
    )
    result.warnings.extend(dataset.warnings[:10])
    result.data.update(
        {
            "format": dataset.source_format,
            "findings": payload["findings"],
            "per_class": payload["per_class"][: params["top_classes"]],
            "balance": payload["balance"],
            "object_sizes": payload["object_sizes"],
            "relative_area": payload["relative_area"],
            "objects_per_image": payload["objects_per_image"],
            "image_sizes": payload["image_sizes"],
            "splits": payload["splits"],
            "geometry": payload["geometry"],
        }
    )
    return result


LEAKAGE_PARAMS = (
    Param("folder", "dir", "Split dataset to check (containing train/, val/, ...).",
          required=True, must_exist=True),
    Param("splits", "strs", "Split folder names (blank = every top-level folder).", default=None),
    Param(
        "method",
        "choice",
        "exact compares file bytes; phash and friends compare appearance.",
        default="phash",
        choices=("exact", "phash", "dhash", "ahash"),
    ),
    Param(
        "threshold",
        "int",
        "Perceptual distance counted as the same image (0 = identical).",
        default=6,
        minimum=0,
        maximum=32,
    ),
    Param("hash_size", "int", "Perceptual hash size.", default=8, minimum=4, maximum=32),
    Param("max_examples", "int", "How many leaked pairs to list.", default=50, minimum=1),
)


@operation(
    "inspect.leakage",
    title="Train/validation leakage",
    category="inspect",
    summary="Find near-duplicate images shared between splits, which silently inflate your score.",
    params=LEAKAGE_PARAMS,
    notes=(
        "The most expensive bug in applied computer vision, and the hardest to\n"
        "see: if the same photo -- or a burst frame taken a tenth of a second\n"
        "apart, or a re-saved copy -- appears in both train and val, the model\n"
        "has memorised the answer and your validation score is measuring recall\n"
        "of the training set. Scores look excellent right up until deployment.\n"
        "\n"
        "It happens by accident constantly: video frames split randomly, a\n"
        "dataset merged with an older copy of itself, images duplicated during\n"
        "collection and then split.\n"
        "\n"
        "Read-only -- it reports, it never deletes. The fix is to re-split with\n"
        "the duplicates grouped (dedup first with clean.dedup, then split), not\n"
        "to remove the val copy and keep going.\n"
        "\n"
        "method=exact only catches byte-identical files; the default phash also\n"
        "catches re-encoded, resized and lightly-cropped copies, which is where\n"
        "most real leakage lives."
    ),
)
def leakage(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="inspect.leakage", dry_run=ctx.dry_run)
    folder: Path = params["folder"]

    groups = _split_groups(folder, params["splits"])
    if len(groups) < 2:
        result.fail(
            f"need at least two splits to compare, found {len(groups)} under {folder}. "
            f"Point this at the parent folder holding train/ and val/, or name them "
            f"with --splits train,val"
        )
        return result

    method = params["method"]
    try:
        fingerprint = _fingerprinter(method, params["hash_size"])
    except RuntimeError as exc:
        result.fail(str(exc))
        return result

    total = sum(len(paths) for paths in groups.values())
    hashes: dict[str, list[tuple[Any, Path]]] = {}
    position = 0
    unreadable = 0

    for split, paths in groups.items():
        entries: list[tuple[Any, Path]] = []
        for path in paths:
            position += 1
            ctx.check_cancelled()
            ctx.emit(position, total, f"{split}/{path.name}")
            try:
                entries.append((fingerprint(path), path))
            except Exception as exc:
                unreadable += 1
                if unreadable <= 5:
                    result.warnings.append(f"{path.name}: unreadable ({exc})")
        hashes[split] = entries

    names = sorted(groups)
    matches: list[dict[str, Any]] = []
    per_pair: dict[str, int] = {}
    leaked_files: set[Path] = set()

    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            found = (
                _exact_matches(hashes[left], hashes[right])
                if method == "exact"
                else _near_matches(hashes[left], hashes[right], params["threshold"])
            )
            per_pair[f"{left} <-> {right}"] = len(found)
            for distance, a, b in found:
                leaked_files.add(b)
                if len(matches) < params["max_examples"]:
                    matches.append(
                        {
                            "distance": distance,
                            "splits": f"{left} <-> {right}",
                            left: str(a),
                            right: str(b),
                        }
                    )

    leaked = sum(per_pair.values())
    smallest = min((len(v) for v in groups.values()), default=1)
    result.counts.update(
        {
            "images": total,
            "splits": len(groups),
            "leaked_pairs": leaked,
            "affected_images": len(leaked_files),
            "unreadable": unreadable,
        }
    )
    result.data.update(
        {
            "method": method,
            "threshold": params["threshold"] if method != "exact" else 0,
            "per_split": {k: len(v) for k, v in groups.items()},
            "per_pair": per_pair,
            "examples": matches,
        }
    )

    if leaked:
        share = len(leaked_files) / max(1, smallest)
        result.warnings.append(
            f"{leaked} leaked pair(s) across splits, touching {len(leaked_files)} image(s) "
            f"-- around {share:.0%} of the smallest split. Validation scores from this "
            f"dataset are optimistic."
        )
        result.warnings.append(
            "Fix by de-duplicating before splitting: clean.dedup --method "
            f"{method if method != 'exact' else 'phash'}, then dataset.split."
        )
    return result


# ------------------------------------------------------------------ helpers

def _split_groups(folder: Path, wanted: list[str] | None) -> dict[str, list[Path]]:
    """Images grouped by split folder.

    Looks one and two levels down, so both ``train/`` and the YOLO
    ``images/train/`` layout are recognised without being told which it is.
    """
    groups: dict[str, list[Path]] = {}
    names = set(wanted or [])

    candidates: list[Path] = [p for p in sorted(folder.iterdir()) if p.is_dir()]
    images_dir = folder / "images"
    if images_dir.is_dir():
        nested = [p for p in sorted(images_dir.iterdir()) if p.is_dir()]
        if nested:
            candidates = nested

    for directory in candidates:
        if names and directory.name not in names:
            continue
        found = list(fs.iter_images(directory, recursive=True))
        if found:
            groups[directory.name] = found
    return groups


def _fingerprinter(method: str, hash_size: int) -> Callable[[Path], Any]:
    if method == "exact":
        return fs.sha256_file
    try:
        import imagehash
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(f"perceptual comparison needs Pillow and imagehash ({exc})") from exc

    function = {
        "phash": imagehash.phash,
        "dhash": imagehash.dhash,
        "ahash": imagehash.average_hash,
    }[method]

    def compute(path: Path) -> Any:
        with Image.open(path) as image:
            image.load()
            return function(image, hash_size=hash_size)

    return compute


def _exact_matches(
    left: list[tuple[Any, Path]], right: list[tuple[Any, Path]]
) -> list[tuple[int, Path, Path]]:
    index: dict[Any, Path] = {}
    for digest, path in left:
        index.setdefault(digest, path)
    return [(0, index[digest], path) for digest, path in right if digest in index]


def _near_matches(
    left: list[tuple[Any, Path]], right: list[tuple[Any, Path]], threshold: int
) -> list[tuple[int, Path, Path]]:
    """Radius search rather than all-pairs.

    Comparing a 3,000-image train split against a 400-image val split pairwise
    is 1.2M hash comparisons; the BK-tree turns each lookup into a handful.
    """
    tree: BKTree = BKTree(lambda a, b: a[0] - b[0])
    for entry in left:
        tree.add(entry)

    found: list[tuple[int, Path, Path]] = []
    for digest, path in right:
        nearest = tree.nearest((digest, path), threshold)
        if nearest is not None:
            distance, (_, other) = nearest
            found.append((distance, other, path))
    return found

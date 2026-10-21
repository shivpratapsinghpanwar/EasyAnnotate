"""Dataset arrangement: train/val/test splitting, and sorting by class.

The README listed train/test splitting as a supported capability but no script
ever implemented it. It is implemented here, and stratified by default -- an
unstratified random split of an imbalanced set (the door dataset has ~200 hard
negatives against ~3500 positives) can easily land almost every negative in one
side and skew the evaluation.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .. import fs
from ..annotations import voc
from ..annotations.model import AnnotationError
from ..pairs import Pair, PairIndex
from ..registry import Param, operation
from ..types import Action, Context, OpResult

SPLIT_PARAMS = (
    Param("folder", "dir", "Folder of image/label pairs to split.", required=True, must_exist=True),
    Param("output", "dir", "Where the split folders are created.", required=True),
    Param("ratios", "strs", "Split sizes, in order, e.g. 0.8,0.1,0.1", default="0.8,0.1,0.1"),
    Param("names", "strs", "Names for each split.", default="train,val,test"),
    Param("mode", "choice", "Copy into the split, or move.", default="copy", choices=("copy", "move")),
    Param("seed", "int", "Random seed, so the same split is reproducible.", default=42),
    Param("stratify", "bool", "Keep class proportions equal across splits.", default=True),
    Param("layout", "choice", "Folder shape inside each split.",
          default="flat", choices=("flat", "images_labels")),
    Param("include", "choice", "Split complete pairs only, or all images.",
          default="pairs", choices=("pairs", "all")),
    Param("recursive", "bool", "Include sub-folders when scanning.", default=False),
    Param("label_ext", "str", "Sidecar label extension.", default=".xml"),
)


@operation(
    "dataset.split",
    title="Split into train / val / test",
    category="dataset",
    summary="Divide pairs into reproducible, class-balanced splits without separating a label from its image.",
    params=SPLIT_PARAMS,
    destructive=False,
    notes=(
        "Stratification groups by the set of classes in each image (an image with\n"
        "both Open and Close is its own stratum, and background images are theirs),\n"
        "then splits each group by the given ratios. Every split therefore sees the\n"
        "same class mix as the whole set.\n\n"
        "seed makes the split reproducible: same folder plus same seed gives the\n"
        "same partition, which is what makes model comparisons meaningful."
    ),
)
def split(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="dataset.split", dry_run=ctx.dry_run)
    folder: Path = params["folder"]
    output: Path = params["output"]

    try:
        ratios = [float(r) for r in params["ratios"]]
    except ValueError:
        result.fail(f"ratios must be numbers, got {params['ratios']}")
        return result
    names = list(params["names"])
    if len(ratios) != len(names):
        result.fail(f"got {len(ratios)} ratios but {len(names)} names; they must match")
        return result
    if any(r < 0 for r in ratios) or sum(ratios) <= 0:
        result.fail("ratios must be non-negative and sum to more than zero")
        return result
    ratios = [r / sum(ratios) for r in ratios]

    index = PairIndex.scan(folder, recursive=params["recursive"], label_ext=params["label_ext"])
    items = index.complete() if params["include"] == "pairs" else index.with_images()
    result.counts["scanned"] = len(index)
    result.counts["selected"] = len(items)
    if not items:
        result.warnings.append("nothing to split")
        return result

    strata = _stratify(items, params["stratify"], result)
    rng = random.Random(int(params["seed"]))
    buckets: dict[str, list[Pair]] = {name: [] for name in names}

    for key in sorted(strata):
        group = sorted(strata[key], key=lambda p: p.key)
        rng.shuffle(group)
        for name, chunk in zip(names, _partition(len(group), ratios)):
            buckets[name].extend(group[chunk[0] : chunk[1]])

    verb = Action.COPY if params["mode"] == "copy" else Action.MOVE
    per_split_labels: dict[str, dict[str, int]] = {}

    for name in names:
        members = sorted(buckets[name], key=lambda p: p.key)
        result.counts[name] = len(members)
        label_counts: dict[str, int] = defaultdict(int)

        # Create the folder even when the split is empty. Downstream steps
        # address splits by name, and a missing folder turns an empty split into
        # a hard failure three commands later.
        if params["layout"] == "flat":
            fs.ensure_dir(output / name, ctx)
        else:
            fs.ensure_dir(output / name / "images", ctx)
            fs.ensure_dir(output / name / "labels", ctx)

        if not members:
            result.warnings.append(
                f"split '{name}' is empty -- with {len(items)} item(s) across "
                f"{len(strata)} stratum/strata, its share rounded to zero"
            )

        for position, pair in enumerate(members, start=1):
            ctx.check_cancelled()
            ctx.emit(position, len(members), f"{name}: {pair.stem}")
            for path in pair.files:
                destination = _destination(output, name, path, params["layout"], params["label_ext"])
                fs.perform(result.record(verb, path, destination, f"split={name}"), ctx)
            result.bump(f"{name}_files", len(pair.files))
            for label in _labels_of(pair):
                label_counts[label] += 1

        per_split_labels[name] = dict(sorted(label_counts.items()))

    result.data["per_split_labels"] = per_split_labels
    result.data["ratios"] = dict(zip(names, ratios))
    result.data["seed"] = int(params["seed"])
    result.data["output"] = output
    return result


def _destination(output: Path, split_name: str, path: Path, layout: str, label_ext: str) -> Path:
    if layout == "flat":
        return output / split_name / path.name
    bucket = "labels" if path.suffix.lower() == label_ext.lower() else "images"
    return output / split_name / bucket / path.name


def _labels_of(pair: Pair) -> list[str]:
    if pair.label is None:
        return []
    try:
        return sorted(voc.read(pair.label).label_set) or ["__background__"]
    except AnnotationError:
        return []


def _stratify(items: list[Pair], enabled: bool, result: OpResult) -> dict[str, list[Pair]]:
    """Group pairs by their class signature, or into one bucket if disabled."""
    if not enabled:
        return {"__all__": list(items)}
    groups: dict[str, list[Pair]] = defaultdict(list)
    for pair in items:
        labels = _labels_of(pair)
        groups["+".join(labels) if labels else "__unlabelled__"].append(pair)
    result.data["strata"] = {k: len(v) for k, v in sorted(groups.items())}
    return groups


def _partition(count: int, ratios: list[float]) -> list[tuple[int, int]]:
    """Slice boundaries for ``count`` items, with the remainder going first.

    Largest-remainder allocation keeps tiny strata from vanishing: with 1 item
    and 0.8/0.1/0.1, the item lands in train rather than being rounded away.
    """
    exact = [count * r for r in ratios]
    sizes = [int(value) for value in exact]
    leftover = count - sum(sizes)
    order = sorted(range(len(ratios)), key=lambda i: exact[i] - sizes[i], reverse=True)
    for i in range(leftover):
        sizes[order[i % len(order)]] += 1

    bounds: list[tuple[int, int]] = []
    cursor = 0
    for size in sizes:
        bounds.append((cursor, cursor + size))
        cursor += size
    return bounds


SORT_PARAMS = (
    Param("folder", "dir", "Folder of image/label pairs.", required=True, must_exist=True),
    Param("output", "dir", "Where the class folders are created.", required=True),
    Param("mode", "choice", "Copy into class folders, or move.",
          default="copy", choices=("copy", "move")),
    Param("mixed_name", "str", "Folder name for images holding more than one class.",
          default="mixed"),
    Param("background_name", "str", "Folder name for images with no objects.",
          default="background"),
    Param("recursive", "bool", "Include sub-folders.", default=False),
    Param("label_ext", "str", "Sidecar label extension.", default=".xml"),
)


@operation(
    "dataset.sort",
    title="Sort pairs into class folders",
    category="dataset",
    summary="Group image/label pairs into one folder per class, with mixed and background buckets.",
    params=SORT_PARAMS,
    notes=(
        "A fast way to eyeball class balance and spot mislabelled images: browse\n"
        "the Open folder and anything that is not an open door stands out."
    ),
)
def sort_by_class(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="dataset.sort", dry_run=ctx.dry_run)
    folder: Path = params["folder"]
    output: Path = params["output"]
    verb = Action.COPY if params["mode"] == "copy" else Action.MOVE

    index = PairIndex.scan(folder, recursive=params["recursive"], label_ext=params["label_ext"])
    pairs = index.complete()
    result.counts["scanned"] = len(index)
    result.counts["sorted"] = 0

    for position, pair in enumerate(pairs, start=1):
        ctx.check_cancelled()
        ctx.emit(position, len(pairs), pair.stem)
        assert pair.label is not None
        try:
            annotation = voc.read(pair.label)
        except AnnotationError as exc:
            result.warnings.append(f"{pair.label.name}: {exc}")
            result.bump("unreadable")
            continue

        labels = sorted(annotation.label_set)
        if not labels:
            bucket = params["background_name"]
        elif len(labels) == 1:
            bucket = labels[0]
        else:
            bucket = params["mixed_name"]

        for path in pair.files:
            fs.perform(result.record(verb, path, output / bucket / path.name, f"class={bucket}"), ctx)
        result.bump("sorted")
        result.bump(f"class:{bucket}")

    result.data["output"] = output
    return result

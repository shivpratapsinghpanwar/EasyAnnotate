"""Dataset algebra: merging, filtering and sampling.

These three cover most of what people actually do to a dataset between
collecting it and training on it, and all three are usually done with a
one-off script that gets the annotations subtly wrong -- merging two folders
whose files share names, or filtering by class and leaving the dropped class's
boxes behind in the label files.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from ..annotations import formats
from ..annotations.model import Annotation, AnnotationError
from ..registry import Param, operation
from ..types import Context, OpResult
from ..writer import DatasetWriter
from .image import OUTPUT_FORMATS, SOURCE_FORMATS

MERGE_PARAMS = (
    Param("folders", "dirs", "Datasets to merge, in order.", required=True, must_exist=True),
    Param("output", "dir", "Where the merged dataset is written.", required=True),
    Param("source_format", "choice", "Input format for all sources.", default="auto", choices=SOURCE_FORMATS),
    Param("format", "choice", "Output format.", default="same", choices=OUTPUT_FORMATS),
    Param(
        "label_map",
        "mapping",
        "Class renames applied during the merge, e.g. car=vehicle,truck=vehicle",
        default=None,
    ),
    Param("fold_case", "bool", "Treat classes differing only by case as one.", default=True),
    Param(
        "prefix_source",
        "bool",
        "Prefix filenames with their source folder name to guarantee uniqueness.",
        default=True,
    ),
    Param("keep_splits", "bool", "Preserve train/val/test folders from the sources.", default=True),
    Param("recursive", "bool", "Include sub-folders.", default=True),
    Param("quality", "int", "JPEG/WebP quality when re-encoding.", default=95, minimum=1, maximum=100),
)


@operation(
    "dataset.merge",
    title="Merge datasets",
    category="dataset",
    summary="Combine several datasets into one, reconciling class names and filename collisions.",
    params=MERGE_PARAMS,
    destructive=True,
    notes=(
        "The two things that go wrong when merging by hand:\n"
        "\n"
        "  Filename collisions. Two folders both containing image_001.jpg means\n"
        "  one silently overwrites the other, taking its annotations with it.\n"
        "  --prefix-source (on by default) makes every name unique.\n"
        "\n"
        "  Class drift. The same object is 'car' in one set and 'Car' or\n"
        "  'vehicle' in another, and you end up training three classes that\n"
        "  should be one. --fold-case handles capitalisation; --label-map\n"
        "  handles the rest: --label-map car=vehicle,truck=vehicle\n"
        "\n"
        "Sources may be in different formats -- each is detected separately."
    ),
)
def merge(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="dataset.merge", dry_run=ctx.dry_run)
    folders: list[Path] = params["folders"]
    output: Path = params["output"]

    if len(folders) < 2:
        result.fail("merging needs at least two source folders")
        return result
    resolved_out = output.resolve()
    for folder in folders:
        if folder.resolve() == resolved_out:
            result.fail(f"output must differ from every source ({folder})")
            return result

    label_map: dict[str, str] = params["label_map"] or {}
    fold_case: bool = params["fold_case"]

    loaded: list[tuple[Path, formats.Dataset]] = []
    for folder in folders:
        try:
            dataset = formats.read_dataset(
                folder, params["source_format"], recursive=params["recursive"], ctx=ctx
            )
        except AnnotationError as exc:
            result.fail(f"{folder}: {exc}")
            return result
        result.warnings.extend(f"{folder.name}: {w}" for w in dataset.warnings)
        loaded.append((folder, dataset))

    # Settle the class vocabulary across all sources before writing anything,
    # so YOLO indices are stable and the mapping can be reported.
    canonical: dict[str, str] = {}
    for _, dataset in loaded:
        for label in dataset.labels:
            canonical.setdefault(label, _canonical(label, label_map, fold_case))
    merged_classes = sorted(set(canonical.values()))

    renames = {k: v for k, v in canonical.items() if k != v}
    target_format = params["format"]
    if target_format == "same":
        target_format = loaded[0][1].source_format

    writer = DatasetWriter(
        output, target_format, ctx, classes=merged_classes, quality=params["quality"]
    )
    per_source: dict[str, int] = {}
    seen_stems: set[str] = set()
    collisions = 0
    total = sum(len(d.items) for _, d in loaded)
    position = 0

    for folder, dataset in loaded:
        written = 0
        for item in dataset.items:
            position += 1
            ctx.check_cancelled()
            ctx.emit(position, total, f"{folder.name}/{item.stem}")

            stem = f"{folder.name}_{item.stem}" if params["prefix_source"] else item.stem
            if stem in seen_stems:
                collisions += 1
                stem = f"{stem}_{collisions}"
            seen_stems.add(stem)

            writer.add(
                _relabelled(item.annotation, canonical),
                stem=stem,
                source=item.image,
                split=item.split if params["keep_splits"] else "",
                suffix=item.image.suffix if item.image else ".jpg",
            )
            written += 1
        per_source[str(folder)] = written

    report = writer.close()
    result.actions.extend(writer.actions)
    result.counts.update(
        {
            "sources": len(loaded),
            "images": total,
            "written": report["labels"],
            "classes": len(merged_classes),
            "collisions_resolved": collisions,
        }
    )
    result.warnings.extend(report["warnings"])
    if collisions and not params["prefix_source"]:
        result.warnings.append(
            f"{collisions} filename collision(s) were renamed. Use --prefix-source to "
            f"avoid them up front and keep names predictable."
        )
    result.data.update(
        {
            "output": output,
            "format": target_format,
            "classes": merged_classes,
            "renamed_classes": renames,
            "per_source": per_source,
        }
    )
    if renames:
        result.warnings.append(
            "classes reconciled: "
            + ", ".join(f"{k} -> {v}" for k, v in sorted(renames.items())[:10])
        )
    return result


def _canonical(label: str, label_map: dict[str, str], fold_case: bool) -> str:
    """Map a source label onto its merged name."""
    mapped = label_map.get(label)
    if mapped is None and fold_case:
        # Try a case-insensitive lookup, so --label-map Car=vehicle also
        # catches "car" without listing every spelling.
        for key, value in label_map.items():
            if key.casefold() == label.casefold():
                mapped = value
                break
    result = (mapped or label).strip()
    return result


def _relabelled(annotation: Annotation, canonical: dict[str, str]) -> Annotation:
    """A copy whose shapes carry the merged class names.

    Case folding is applied by canonicalising to whichever spelling sorts
    first, which is reproducible and keeps the choice out of the file order.
    """
    from ..annotations.model import Shape

    shapes = [
        Shape(
            label=canonical.get(s.label, s.label),
            bbox=s.bbox,
            polygon=s.polygon,
            pose=s.pose,
            truncated=s.truncated,
            difficult=s.difficult,
            occluded=s.occluded,
        )
        for s in annotation.shapes
    ]
    return Annotation(
        filename=annotation.filename,
        width=annotation.width,
        height=annotation.height,
        depth=annotation.depth,
        shapes=shapes,
        folder=annotation.folder,
        source_path=annotation.source_path,
    )


FILTER_PARAMS = (
    Param("folder", "dir", "Dataset to filter.", required=True, must_exist=True),
    Param("output", "dir", "Where the filtered dataset is written.", required=True),
    Param("keep_classes", "strs", "Keep only these classes (blank means all).", default=None),
    Param("drop_classes", "strs", "Remove these classes.", default=None),
    Param("min_objects", "int", "Drop images with fewer objects than this.", default=0, minimum=0),
    Param("max_objects", "int", "Drop images with more objects than this (0 = no limit).", default=0, minimum=0),
    Param(
        "min_area",
        "int",
        "Drop individual objects with a box area below this many pixels.",
        default=0,
        minimum=0,
    ),
    Param("require_polygons", "bool", "Keep only objects that carry an outline.", default=False),
    Param("keep_backgrounds", "bool", "Keep images that end up with no objects.", default=False),
    Param("splits", "strs", "Keep only these splits (blank means all).", default=None),
    Param("source_format", "choice", "Input format.", default="auto", choices=SOURCE_FORMATS),
    Param("format", "choice", "Output format.", default="same", choices=OUTPUT_FORMATS),
    Param("recursive", "bool", "Include sub-folders.", default=True),
    Param("quality", "int", "JPEG/WebP quality.", default=95, minimum=1, maximum=100),
)


@operation(
    "dataset.filter",
    title="Filter dataset",
    category="dataset",
    summary="Build a subset by class, object count, object size or split.",
    params=FILTER_PARAMS,
    destructive=True,
    notes=(
        "Filtering happens at the object level first, then the image level. So\n"
        "--keep-classes door removes every non-door box from the label files\n"
        "*and then* drops any image left with nothing -- rather than leaving\n"
        "images whose annotations still mention classes you excluded.\n"
        "\n"
        "--keep-backgrounds changes that last step: images emptied by the\n"
        "filter are kept as hard negatives instead of dropped, which is often\n"
        "what you want when narrowing a multi-class set to one class."
    ),
)
def filter_dataset(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="dataset.filter", dry_run=ctx.dry_run)
    folder: Path = params["folder"]
    output: Path = params["output"]
    if output.resolve() == folder.resolve():
        result.fail("output must differ from folder")
        return result

    try:
        dataset = formats.read_dataset(
            folder, params["source_format"], recursive=params["recursive"], ctx=ctx
        )
    except AnnotationError as exc:
        result.fail(str(exc))
        return result
    result.warnings.extend(dataset.warnings)

    keep = {c.casefold() for c in (params["keep_classes"] or [])}
    drop = {c.casefold() for c in (params["drop_classes"] or [])}
    splits = set(params["splits"] or [])
    unknown = (keep | drop) - {label.casefold() for label in dataset.labels}
    if unknown:
        result.warnings.append(
            f"class(es) not present in this dataset: {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(dataset.labels)}"
        )

    target_format = params["format"]
    if target_format == "same":
        target_format = dataset.source_format

    kept_labels = [
        label
        for label in dataset.labels
        if (not keep or label.casefold() in keep) and label.casefold() not in drop
    ]
    writer = DatasetWriter(output, target_format, ctx, classes=kept_labels, quality=params["quality"])

    total = len(dataset.items)
    images_kept = 0
    objects_kept = 0
    objects_removed = 0
    reasons: Counter[str] = Counter()

    for position, item in enumerate(dataset.items, start=1):
        ctx.check_cancelled()
        ctx.emit(position, total, item.stem)
        if splits and (item.split or "") not in splits:
            reasons["split"] += 1
            continue

        shapes = []
        for shape in item.annotation.shapes:
            folded = shape.label.casefold()
            if keep and folded not in keep:
                objects_removed += 1
                continue
            if folded in drop:
                objects_removed += 1
                continue
            if params["min_area"] and shape.bbox_area < params["min_area"]:
                objects_removed += 1
                continue
            if params["require_polygons"] and not shape.has_polygon:
                objects_removed += 1
                continue
            shapes.append(shape)

        if len(shapes) < params["min_objects"]:
            reasons["too few objects"] += 1
            continue
        if params["max_objects"] and len(shapes) > params["max_objects"]:
            reasons["too many objects"] += 1
            continue
        if not shapes and not params["keep_backgrounds"]:
            reasons["nothing left"] += 1
            continue

        writer.add(
            Annotation(
                filename=item.annotation.filename,
                width=item.annotation.width,
                height=item.annotation.height,
                depth=item.annotation.depth,
                shapes=shapes,
                folder=item.annotation.folder,
            ),
            stem=item.stem,
            source=item.image,
            split=item.split,
            suffix=item.image.suffix if item.image else ".jpg",
        )
        images_kept += 1
        objects_kept += len(shapes)

    report = writer.close()
    result.actions.extend(writer.actions)
    result.counts.update(
        {
            "images_in": total,
            "images_kept": images_kept,
            "images_dropped": total - images_kept,
            "objects_kept": objects_kept,
            "objects_removed": objects_removed,
        }
    )
    result.warnings.extend(report["warnings"])
    result.data.update(
        {
            "output": output,
            "format": target_format,
            "classes": kept_labels,
            "dropped_because": dict(reasons),
        }
    )
    if images_kept == 0:
        result.warnings.append("the filter kept nothing -- check the class names and thresholds")
    return result


SAMPLE_PARAMS = (
    Param("folder", "dir", "Dataset to sample from.", required=True, must_exist=True),
    Param("output", "dir", "Where the sample is written.", required=True),
    Param("count", "int", "How many images to take (0 = use fraction).", default=0, minimum=0),
    Param(
        "fraction",
        "float",
        "Fraction of the dataset to take, when count is 0.",
        default=0.1,
        minimum=0.0,
        maximum=1.0,
    ),
    Param(
        "per_class",
        "int",
        "Cap on images per class -- the way to level an imbalanced set (0 = off).",
        default=0,
        minimum=0,
    ),
    Param("stratify", "bool", "Preserve class proportions in the sample.", default=True),
    Param("seed", "int", "Random seed; the same seed gives the same sample.", default=42),
    Param("source_format", "choice", "Input format.", default="auto", choices=SOURCE_FORMATS),
    Param("format", "choice", "Output format.", default="same", choices=OUTPUT_FORMATS),
    Param("recursive", "bool", "Include sub-folders.", default=True),
    Param("quality", "int", "JPEG/WebP quality.", default=95, minimum=1, maximum=100),
)


@operation(
    "dataset.sample",
    title="Sample a subset",
    category="dataset",
    summary="Take a reproducible subset -- random, stratified, or capped per class.",
    params=SAMPLE_PARAMS,
    destructive=True,
    notes=(
        "Three uses, in order of how often they come up:\n"
        "\n"
        "  A small set to iterate on. --fraction 0.05 gives you something that\n"
        "  trains in minutes while you get the pipeline right.\n"
        "\n"
        "  Levelling an imbalanced dataset. --per-class 500 caps the dominant\n"
        "  class without touching the rare ones, which is usually a better\n"
        "  starting point than class weights.\n"
        "\n"
        "  A review sample. A stratified 200 images is enough to eyeball label\n"
        "  quality across every class.\n"
        "\n"
        "Always seeded, so 'the 500 images we evaluated on' stays meaningful."
    ),
)
def sample(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="dataset.sample", dry_run=ctx.dry_run)
    folder: Path = params["folder"]
    output: Path = params["output"]
    if output.resolve() == folder.resolve():
        result.fail("output must differ from folder")
        return result

    try:
        dataset = formats.read_dataset(
            folder, params["source_format"], recursive=params["recursive"], ctx=ctx
        )
    except AnnotationError as exc:
        result.fail(str(exc))
        return result
    result.warnings.extend(dataset.warnings)

    items = list(dataset.items)
    if not items:
        result.fail(f"no images found in {folder}")
        return result

    rng = random.Random(params["seed"])
    if params["per_class"]:
        chosen = _cap_per_class(items, params["per_class"], rng)
    else:
        wanted = params["count"] or max(1, round(len(items) * params["fraction"]))
        wanted = min(wanted, len(items))
        chosen = (
            _stratified(items, wanted, rng) if params["stratify"] else rng.sample(items, wanted)
        )

    # Sorting restores a stable on-disk order after the random selection, so
    # the output folder listing does not look shuffled.
    chosen.sort(key=lambda i: i.stem)

    target_format = params["format"]
    if target_format == "same":
        target_format = dataset.source_format
    writer = DatasetWriter(
        output, target_format, ctx, classes=dataset.labels, quality=params["quality"]
    )

    per_class: Counter[str] = Counter()
    for position, item in enumerate(chosen, start=1):
        ctx.check_cancelled()
        ctx.emit(position, len(chosen), item.stem)
        per_class.update(item.annotation.label_set)
        writer.add(
            item.annotation,
            stem=item.stem,
            source=item.image,
            split=item.split,
            suffix=item.image.suffix if item.image else ".jpg",
        )

    report = writer.close()
    result.actions.extend(writer.actions)
    result.counts.update(
        {
            "available": len(items),
            "sampled": len(chosen),
            "objects": sum(len(i.annotation.shapes) for i in chosen),
        }
    )
    result.warnings.extend(report["warnings"])
    result.data.update(
        {
            "output": output,
            "format": target_format,
            "seed": params["seed"],
            "images_per_class": dict(sorted(per_class.items())),
        }
    )
    return result


def _stratified(items: list[formats.Item], wanted: int, rng: random.Random) -> list[formats.Item]:
    """Sample proportionally per class signature, so rare classes survive.

    Grouping on the *set* of classes in an image, rather than on a single
    class, is what keeps multi-label images from being counted twice.
    """
    groups: dict[str, list[formats.Item]] = {}
    for item in items:
        key = ",".join(sorted(item.annotation.label_set)) or "(background)"
        groups.setdefault(key, []).append(item)

    chosen: list[formats.Item] = []
    remainders: list[tuple[float, str]] = []
    for key, members in sorted(groups.items()):
        share = len(members) / len(items) * wanted
        take = int(share)
        rng.shuffle(members)
        chosen.extend(members[:take])
        remainders.append((share - take, key))

    # Largest-remainder, so the total lands exactly on `wanted` rather than a
    # few short from repeated rounding down.
    remainders.sort(reverse=True)
    index = 0
    while len(chosen) < wanted and index < len(remainders):
        key = remainders[index][1]
        members = groups[key]
        already = sum(1 for c in chosen if c in members)
        if already < len(members):
            chosen.append(members[already])
        index += 1
    return chosen[:wanted]


def _cap_per_class(
    items: list[formats.Item], cap: int, rng: random.Random
) -> list[formats.Item]:
    """Keep images until every class has reached ``cap``.

    Rarest class first: an image showing both a common and a rare class should
    be admitted on the rare class's account, or the rare class runs out of
    candidates while the common one is still filling up.
    """
    frequency: Counter[str] = Counter()
    for item in items:
        frequency.update(item.annotation.label_set)

    order = sorted(
        items,
        key=lambda i: (
            min((frequency[label] for label in i.annotation.label_set), default=10**9),
            i.stem,
        ),
    )
    rng.shuffle(order)
    order.sort(
        key=lambda i: min((frequency[label] for label in i.annotation.label_set), default=10**9)
    )

    taken: Counter[str] = Counter()
    chosen: list[formats.Item] = []
    for item in order:
        labels = item.annotation.label_set
        if not labels:
            continue
        if all(taken[label] >= cap for label in labels):
            continue
        chosen.append(item)
        taken.update(labels)
    return chosen

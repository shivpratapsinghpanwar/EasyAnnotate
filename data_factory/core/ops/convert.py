"""Dataset format conversion -- any supported format to any other."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..annotations import formats
from ..annotations.model import AnnotationError
from ..registry import Param, operation
from ..types import Context, OpResult

FORMAT_CHOICES = ("auto",) + formats.FORMATS

CONVERT_PARAMS = (
    Param("folder", "dir", "Dataset to convert.", required=True, must_exist=True),
    Param("output", "dir", "Where the converted dataset is written.", required=True),
    Param(
        "to",
        "choice",
        "Target format.",
        required=True,
        choices=formats.FORMATS,
    ),
    Param(
        "source_format",
        "choice",
        "Source format; 'auto' sniffs the folder layout.",
        default="auto",
        choices=FORMAT_CHOICES,
    ),
    Param(
        "classes",
        "strs",
        "Explicit class order, e.g. 'Closed,Open'. Fixes YOLO indices in place.",
        default=None,
    ),
    Param("recursive", "bool", "Include sub-folders.", default=True),
    Param("copy_images", "bool", "Copy the image files alongside the labels.", default=True),
    Param(
        "segmentation",
        "bool",
        "Keep polygon outlines where the target format supports them.",
        default=True,
    ),
)


@operation(
    "convert.dataset",
    title="Convert dataset format",
    category="convert",
    summary="Convert between Pascal VOC, YOLO, COCO, LabelMe and CSV in any direction.",
    params=CONVERT_PARAMS,
    destructive=True,
    notes=(
        "Formats meet in the middle at a neutral annotation model, so every\n"
        "direction works: voc->yolo, coco->voc, labelme->coco, and so on.\n"
        "\n"
        "Things worth knowing before you run it:\n"
        "  * YOLO stores a class *index*, so the class list lives in data.yaml.\n"
        "    Reading a YOLO folder without one names classes class_0, class_1...\n"
        "  * COCO RLE / iscrowd masks cannot become polygons without a mask\n"
        "    decoder; those objects keep their boxes and the run warns.\n"
        "  * CSV cannot hold outlines, so converting a segmentation dataset to\n"
        "    csv is lossy -- it warns rather than dropping them silently.\n"
        "  * Splits are preserved when the source layout names them (a train/\n"
        "    sub-folder, or instances_val.json).\n"
        "\n"
        "About 'classes'. Without it, the YOLO class order is derived by sorting\n"
        "the names found in *this* folder. That is reproducible but not stable:\n"
        "add a class next month and every index at or after it shifts, so the\n"
        ".txt files you exported earlier quietly start meaning something else.\n"
        "Pass the same 'classes' list on every export and the indices hold. Any\n"
        "class found but not listed is appended at the end, never inserted."
    ),
)
def convert_dataset(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="convert.dataset", dry_run=ctx.dry_run)
    folder: Path = params["folder"]
    output: Path = params["output"]
    target: str = params["to"]

    try:
        dataset = formats.read_dataset(
            folder, params["source_format"], recursive=params["recursive"], ctx=ctx
        )
    except AnnotationError as exc:
        result.fail(str(exc))
        return result

    result.warnings.extend(dataset.warnings)
    if not dataset.items:
        result.fail(f"no annotated images found in {folder}")
        return result

    if output.resolve() == folder.resolve():
        result.fail("output must differ from folder -- converting in place would overwrite the source")
        return result

    # An explicit class order pins the YOLO indices. Applied after reading so
    # it overrides a data.yaml too: the caller's declared order wins over
    # whatever the source folder happened to carry.
    declared: list[str] | None = params.get("classes")
    if declared:
        order = [name.strip() for name in declared if name.strip()]
        found = {label for item in dataset.items for label in item.annotation.label_set}
        unlisted = sorted(found - set(order))
        if unlisted:
            result.warnings.append(
                f"{len(unlisted)} class(es) present but not in 'classes' were appended "
                f"after the declared order: {', '.join(unlisted)}"
            )
        unused = [name for name in order if name not in found]
        if unused:
            # Keeping them is the point -- a class with no objects this month
            # still has to hold its index for the files exported last month.
            result.warnings.append(
                f"{len(unused)} declared class(es) have no objects here but keep "
                f"their index: {', '.join(unused)}"
            )
        dataset.classes = order + unlisted

    stats = dataset.stats()
    total = len(dataset.items)
    ctx.emit(0, total, f"{dataset.source_format} -> {target}")

    report = formats.write_dataset(
        dataset,
        output,
        target,
        ctx=ctx,
        copy_images=params["copy_images"],
        segmentation=params["segmentation"],
        on_action=lambda action: result.actions.append(action),
    )
    ctx.emit(total, total, "written")

    result.counts.update(
        {
            "images": stats["images"],
            "objects": stats["objects"],
            "classes": stats["classes"],
            "backgrounds": stats["backgrounds"],
            "labels_written": int(report["written"]),
            "images_copied": int(report["images"]),
        }
    )
    result.warnings.extend(str(w) for w in report["warnings"])  # type: ignore[union-attr]
    if stats["missing_images"]:
        result.warnings.append(
            f"{stats['missing_images']} annotation(s) had no matching image file; "
            f"their labels were still converted"
        )

    result.data.update(
        {
            "source_format": dataset.source_format,
            "target_format": target,
            "output": output,
            "classes": dataset.labels,
        }
    )
    return result


DETECT_PARAMS = (
    Param("folder", "dir", "Dataset folder to identify.", required=True, must_exist=True),
)


@operation(
    "convert.detect",
    title="Identify dataset format",
    category="convert",
    summary="Work out what annotation format a folder is in, and summarise it.",
    params=DETECT_PARAMS,
    notes=(
        "Read-only. Useful before converting something you were handed: it\n"
        "reports the detected format, the class list, and how many objects\n"
        "carry polygons -- which is what decides whether a conversion is lossy."
    ),
)
def detect_format(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="convert.detect", dry_run=ctx.dry_run)
    folder: Path = params["folder"]

    try:
        detected = formats.detect(folder)
    except AnnotationError as exc:
        result.fail(str(exc))
        return result

    try:
        dataset = formats.read_dataset(folder, detected, recursive=True, ctx=ctx)
    except AnnotationError as exc:
        # Detection succeeded but the contents did not parse -- still useful.
        result.data["format"] = detected
        result.fail(str(exc))
        return result

    result.counts.update(dataset.stats())
    result.warnings.extend(dataset.warnings)
    result.data.update(
        {
            "format": detected,
            "classes": dataset.labels,
            "splits": sorted({i.split for i in dataset.items if i.split}),
            "convertible_to": [f for f in formats.FORMATS if f != detected],
        }
    )
    return result

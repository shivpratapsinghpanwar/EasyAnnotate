"""Dataset health reports -- the "what shape is this data actually in" command.

No original script did this. Every check here corresponds to a defect that has
really occurred in this repository: case-variant classes that split a class in
two, polygons the converter could not read, labels whose image is gone, and
boxes whose coordinates fall outside the image they annotate.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..annotations import coco as coco_format
from ..annotations import voc
from ..annotations.model import AnnotationError
from ..pairs import PairIndex
from ..registry import Param, operation
from ..types import Context, OpResult

PARAMS = (
    Param("folder", "dir", "Dataset folder to audit.", required=True, must_exist=True),
    Param("recursive", "bool", "Include sub-folders.", default=True),
    Param("check_images", "bool",
          "Open every image to verify it decodes and matches the size in its label.",
          default=False),
    Param("check_duplicates", "choice", "Also scan for duplicate images.",
          default="none", choices=("none", "exact", "phash")),
    Param("threshold", "int", "Hamming distance for phash duplicate detection.",
          default=8, minimum=0, maximum=64),
    Param("label_ext", "str", "Sidecar label extension.", default=".xml"),
)


@operation(
    "inspect.dataset",
    title="Audit a dataset",
    category="inspect",
    summary="One report: pairing, classes, geometry problems, corrupt files, and optional duplicates.",
    params=PARAMS,
    notes=(
        "Read-only -- it changes nothing, so it is always safe to run first.\n\n"
        "check_images decodes every file, which is slow on a big set but is the only\n"
        "way to catch truncated JPEGs and labels whose <size> disagrees with the\n"
        "actual image (a common cause of boxes that look right in the XML and land\n"
        "in the wrong place at training time)."
    ),
)
def inspect_dataset(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="inspect.dataset", dry_run=ctx.dry_run)
    folder: Path = params["folder"]

    index = PairIndex.scan(folder, recursive=params["recursive"], label_ext=params["label_ext"])
    result.counts.update(index.summary())

    labels: Counter[str] = Counter()
    corrupt: list[dict[str, str]] = []
    geometry: list[dict[str, Any]] = []
    size_mismatch: list[dict[str, Any]] = []
    unreadable_images: list[str] = []
    backgrounds = 0
    polygon_objects = 0
    total_objects = 0

    pairs = index.complete()
    for position, pair in enumerate(pairs, start=1):
        ctx.check_cancelled()
        ctx.emit(position, len(pairs), pair.stem)
        assert pair.image is not None and pair.label is not None

        try:
            annotation = voc.read(pair.label)
        except AnnotationError as exc:
            corrupt.append({"file": str(pair.label), "error": str(exc)})
            continue

        if annotation.is_background:
            backgrounds += 1

        width, height = annotation.width, annotation.height
        if params["check_images"]:
            actual = _image_size(pair.image)
            if actual is None:
                unreadable_images.append(str(pair.image))
            elif (width, height) != actual and width > 0 and height > 0:
                size_mismatch.append(
                    {"file": str(pair.image), "label_says": [width, height], "actual": list(actual)}
                )
                width, height = actual

        for shape in annotation.shapes:
            total_objects += 1
            labels[shape.label] += 1
            if shape.has_polygon:
                polygon_objects += 1
            problem = _geometry_problem(shape, width, height)
            if problem:
                geometry.append(
                    {"file": str(pair.label), "label": shape.label,
                     "bbox": list(shape.bbox), "problem": problem}
                )

    folded: dict[str, list[str]] = {}
    for label in labels:
        folded.setdefault(label.strip().casefold(), []).append(label)
    variants = {k: sorted(v) for k, v in folded.items() if len(v) > 1}

    result.counts.update(
        {
            "objects": total_objects,
            "classes": len(labels),
            "background_images": backgrounds,
            "polygon_objects": polygon_objects,
            "bbox_only_objects": total_objects - polygon_objects,
            "corrupt_labels": len(corrupt),
            "geometry_problems": len(geometry),
        }
    )
    if params["check_images"]:
        result.counts["unreadable_images"] = len(unreadable_images)
        result.counts["size_mismatches"] = len(size_mismatch)

    result.data["per_label"] = dict(sorted(labels.items()))
    result.data["case_variants"] = variants
    result.data["corrupt_labels"] = corrupt[:200]
    result.data["geometry_problems"] = geometry[:200]
    result.data["size_mismatches"] = size_mismatch[:200]
    result.data["unreadable_images"] = unreadable_images[:200]

    if params["check_duplicates"] != "none":
        result.data["duplicates"] = _duplicate_scan(index, params, result, ctx)

    result.data["issues"] = _issues(result, index, variants, corrupt, geometry, size_mismatch)
    return result


def _geometry_problem(shape: Any, width: int, height: int) -> str | None:
    xmin, ymin, xmax, ymax = shape.bbox
    if xmax <= xmin or ymax <= ymin:
        return "zero or negative area"
    if xmin < 0 or ymin < 0:
        return "negative coordinates"
    if width > 0 and height > 0 and (xmax > width or ymax > height):
        return f"extends past the {width}x{height} image"
    if shape.polygon is not None and 0 < len(shape.polygon) < 3:
        return f"polygon has only {len(shape.polygon)} point(s)"
    return None


def _duplicate_scan(
    index: PairIndex, params: dict[str, Any], result: OpResult, ctx: Context
) -> dict[str, Any]:
    """Reuse the dedup machinery in report-only mode.

    Bound through the registry rather than by hand: that fills in defaults for
    every parameter we do not care about here, so adding one to clean.dedup
    cannot silently break this call site.
    """
    from ..registry import get as get_operation

    method = params["check_duplicates"]
    preview = Context(dry_run=True, progress=ctx.progress, cancel_check=ctx.cancel_check)
    dedup_op = get_operation("clean.dedup")
    found = dedup_op.func(
        dedup_op.bind(
            {
                "folders": [index.root],
                "method": method,
                "threshold": int(params["threshold"]),
                "action": "report",
                "recursive": params["recursive"],
            }
        ),
        preview,
    )
    result.counts["duplicate_images"] = found.counts.get("duplicates", 0)
    return {"method": method, "count": found.counts.get("duplicates", 0), "groups": found.data.get("groups", {})}


def _issues(
    result: OpResult,
    index: PairIndex,
    variants: dict[str, list[str]],
    corrupt: list[dict[str, str]],
    geometry: list[dict[str, Any]],
    size_mismatch: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Turn raw findings into a prioritised, human-readable list."""
    issues: list[dict[str, str]] = []

    def add(severity: str, headline: str, fix: str) -> None:
        issues.append({"severity": severity, "issue": headline, "fix": fix})

    if variants:
        joined = "; ".join(" / ".join(v) for v in variants.values())
        add("high", f"classes differing only by case: {joined}",
            "datafactory labels.rename --folder <dir> --fold-case")
    if corrupt:
        add("high", f"{len(corrupt)} label file(s) could not be parsed",
            "inspect the files listed in data.corrupt_labels and re-annotate or delete them")
    if index.label_only():
        add("medium", f"{len(index.label_only())} label(s) have no matching image",
            "datafactory clean.orphans --folder <dir> --scope labels")
    if index.image_only():
        add("low", f"{len(index.image_only())} image(s) have no label",
            "datafactory clean.unannotated --folder <dir> --target <to_annotate>")
    if geometry:
        add("medium", f"{len(geometry)} object(s) have out-of-bounds or degenerate boxes",
            "review data.geometry_problems; these usually come from resizing images after annotating")
    if size_mismatch:
        add("high", f"{len(size_mismatch)} label(s) record a different size than the image",
            "the coordinates are relative to the old size and will be wrong; re-annotate or rescale")
    if result.counts.get("duplicate_images"):
        add("medium", f"{result.counts['duplicate_images']} duplicate image(s)",
            "datafactory clean.dedup --folders <dir> --method phash")
    if not issues:
        add("none", "no problems found", "")
    return issues


COCO_PARAMS = (
    Param("file", "file", "Path to an instances.json.", required=True, must_exist=True),
)


@operation(
    "inspect.coco",
    title="Audit a COCO annotations file",
    category="inspect",
    summary="Summarise an existing instances.json and flag case-collided categories.",
    params=COCO_PARAMS,
    notes="Read-only. Use it to check an export before handing the dataset to training.",
)
def inspect_coco(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="inspect.coco", dry_run=ctx.dry_run)
    path: Path = params["file"]
    try:
        document = coco_format.load(path)
    except Exception as exc:
        result.fail(f"{path.name}: {exc}")
        return result

    summary = coco_format.summarise(document)
    result.counts.update(
        {
            "images": summary["images"],
            "annotations": summary["annotations"],
            "categories": summary["categories"],
            "with_segmentation": summary["with_segmentation"],
            "background_images": summary["background_images"],
        }
    )
    result.data["per_category"] = summary["per_category"]
    result.data["case_collisions"] = summary["case_collisions"]
    if summary["case_collisions"]:
        joined = "; ".join(" / ".join(v) for v in summary["case_collisions"].values())
        result.warnings.append(
            f"categories differing only by case: {joined} -- "
            "re-export with case_policy=casefold, or fix the source labels first"
        )
    return result


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None

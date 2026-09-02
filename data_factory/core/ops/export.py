"""Export a Pascal VOC folder to COCO.

Two behaviour changes from the original converter, both deliberate:

* Categories fold case by default, so ``Open`` and ``open`` are one class. The
  original produced four classes for Project_Door V3 where there are two.
* ``area`` is the true polygon area when a mask exists, not the box area.

Both are reported in the result, so the export never changes your data silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import fs
from ..annotations import coco as coco_format
from ..annotations import voc
from ..annotations.model import AnnotationError
from ..pairs import PairIndex
from ..registry import Param, operation
from ..types import Action, Context, OpResult

PARAMS = (
    Param("folder", "dir", "Folder of Pascal VOC image/label pairs.", required=True, must_exist=True),
    Param("output", "dir", "Dataset root to create.", required=True),
    Param("splits", "strs",
          "Sub-folders to export as separate splits, e.g. train,val,test. Empty = one dataset."),
    Param("copy_images", "bool", "Copy images into <output>/images.", default=True),
    Param("case_policy", "choice", "Whether labels differing only by case are one class.",
          default="casefold", choices=("casefold", "exact")),
    Param("map", "mapping", "Class renames applied during export, as old=new pairs."),
    Param("include_backgrounds", "bool",
          "Include object-free images as hard negatives.", default=True),
    Param("description", "str", "Description written into the COCO info block.",
          default="Data Factory export"),
    Param("recursive", "bool", "Scan sub-folders when not using splits.", default=False),
    Param("label_ext", "str", "Sidecar label extension.", default=".xml"),
)


@operation(
    "export.coco",
    title="Export Pascal VOC to COCO",
    category="export",
    summary="Build instances.json (with segmentation where polygons exist) and gather the images.",
    params=PARAMS,
    notes=(
        "Polygons from either annotation tool are exported as COCO segmentation --\n"
        "including the SAM tool's <pt> dialect, which the old converter could not\n"
        "read and silently exported as empty masks.\n\n"
        "Run labels.list first: if it reports case variants, decide whether you want\n"
        "case_policy=casefold (merge them) or to fix the labels at source with\n"
        "labels.rename --fold-case."
    ),
)
def to_coco(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="export.coco", dry_run=ctx.dry_run)
    folder: Path = params["folder"]
    output: Path = params["output"]
    splits = list(params["splits"] or [])

    targets: list[tuple[str, Path]]
    if splits:
        targets = [(name, folder / name) for name in splits]
        missing = [name for name, path in targets if not path.is_dir()]
        if missing:
            result.fail(
                f"split folder(s) not found under {folder}: {', '.join(missing)}. "
                f"Present: {', '.join(sorted(p.name for p in folder.iterdir() if p.is_dir())) or 'none'}"
            )
            return result
    else:
        targets = [("", folder)]

    for split_name, source in targets:
        ctx.check_cancelled()
        _export_one(split_name, source, output, params, result, ctx)

    result.data["output"] = output
    return result


def _export_one(
    split_name: str,
    source: Path,
    output: Path,
    params: dict[str, Any],
    result: OpResult,
    ctx: Context,
) -> None:
    prefix = f"{split_name}_" if split_name else ""
    image_dir = output / (f"images/{split_name}" if split_name else "images")
    ann_path = output / "annotations" / (f"instances_{split_name}.json" if split_name else "instances.json")

    builder = coco_format.CocoBuilder(
        description=params["description"],
        case_policy=params["case_policy"],
        label_map=params["map"] or {},
    )

    index = PairIndex.scan(
        source,
        recursive=params["recursive"] and not split_name,
        label_ext=params["label_ext"],
    )
    pairs = index.complete()
    result.counts[f"{prefix}scanned"] = len(index)

    for position, pair in enumerate(pairs, start=1):
        ctx.check_cancelled()
        ctx.emit(position, len(pairs), f"{split_name or 'dataset'}: {pair.stem}")
        assert pair.image is not None and pair.label is not None

        try:
            annotation = voc.read(pair.label)
        except AnnotationError as exc:
            result.warnings.append(f"{pair.label.name}: {exc}")
            result.bump(f"{prefix}unreadable")
            continue

        if annotation.is_background and not params["include_backgrounds"]:
            result.bump(f"{prefix}backgrounds_skipped")
            continue

        # The XML's <size> is not always trustworthy; fall back to the real
        # image so COCO never records a zero-dimension entry.
        if annotation.width <= 0 or annotation.height <= 0:
            size = _image_size(pair.image)
            if size is None:
                result.warnings.append(f"{pair.image.name}: cannot determine image size, skipped")
                result.bump(f"{prefix}unsized")
                continue
            annotation.width, annotation.height = size

        builder.add(annotation, file_name=pair.image.name)

        if params["copy_images"]:
            destination = image_dir / pair.image.name
            fs.perform(result.record(Action.COPY, pair.image, destination, "coco image"), ctx)
            result.bump(f"{prefix}images_copied")

    if not ctx.dry_run:
        builder.write(ann_path)
    result.record(Action.WRITE, source, ann_path, "coco annotations")

    for key, value in builder.stats().items():
        result.counts[f"{prefix}{key}"] = value

    merges = builder.categories.merges()
    if merges:
        result.data.setdefault("category_merges", {})[split_name or "dataset"] = merges
        result.warnings.append(
            f"{split_name or 'dataset'}: merged case-variant classes -- "
            + "; ".join(f"{name} <- {'/'.join(v)}" for name, v in merges.items())
        )
    result.data.setdefault("annotations_file", {})[split_name or "dataset"] = ann_path


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None

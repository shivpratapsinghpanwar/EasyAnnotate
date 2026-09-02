"""Label inspection and bulk relabelling across Pascal VOC files."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from .. import fs
from ..annotations import voc
from ..annotations.model import AnnotationError
from ..registry import Param, operation
from ..types import Action, Context, OpResult

LIST_PARAMS = (
    Param("folder", "dir", "Folder of label files to summarise.", required=True, must_exist=True),
    Param("recursive", "bool", "Include sub-folders.", default=True),
    Param("label_ext", "str", "Label file extension.", default=".xml"),
)


@operation(
    "labels.list",
    title="Summarise labels",
    category="labels",
    summary="Count classes and objects, flag unreadable files, and surface case-variant classes.",
    params=LIST_PARAMS,
    notes=(
        "Read this before any export. The case_variants section is the one that\n"
        "bites: labels that differ only in capitalisation become separate classes\n"
        "in COCO and silently halve your per-class training data."
    ),
)
def list_labels(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="labels.list", dry_run=ctx.dry_run)
    folder: Path = params["folder"]
    files = list(fs.iter_files(folder, recursive=params["recursive"], exts=[params["label_ext"]]))

    counter: Counter[str] = Counter()
    images_per_label: Counter[str] = Counter()
    corrupt: list[dict[str, str]] = []
    backgrounds = 0
    polygon_objects = 0

    for position, path in enumerate(files, start=1):
        ctx.check_cancelled()
        ctx.emit(position, len(files), path.name)
        try:
            annotation = voc.read(path)
        except AnnotationError as exc:
            corrupt.append({"file": str(path), "error": str(exc)})
            continue

        if annotation.is_background:
            backgrounds += 1
        for shape in annotation.shapes:
            counter[shape.label] += 1
            if shape.has_polygon:
                polygon_objects += 1
        for label in annotation.label_set:
            images_per_label[label] += 1

    # Group spellings that differ only by case, which is what breaks exports.
    folded: dict[str, list[str]] = {}
    for label in counter:
        folded.setdefault(label.strip().casefold(), []).append(label)
    variants = {key: sorted(names) for key, names in folded.items() if len(names) > 1}

    result.counts.update(
        {
            "label_files": len(files),
            "readable": len(files) - len(corrupt),
            "corrupt": len(corrupt),
            "objects": sum(counter.values()),
            "classes": len(counter),
            "background_images": backgrounds,
            "polygon_objects": polygon_objects,
            "bbox_only_objects": sum(counter.values()) - polygon_objects,
        }
    )
    result.data["per_label"] = dict(sorted(counter.items()))
    result.data["images_per_label"] = dict(sorted(images_per_label.items()))
    result.data["case_variants"] = variants
    result.data["corrupt_files"] = corrupt
    if variants:
        result.warnings.append(
            "labels differing only by case: "
            + "; ".join(" / ".join(v) for v in variants.values())
        )
    return result


RENAME_PARAMS = (
    Param("folder", "dir", "Folder of label files to edit.", required=True, must_exist=True),
    Param("map", "mapping", "Renames as old=new pairs, e.g. planter_View=Plantar_View."),
    Param("fold_case", "bool",
          "Also unify labels that differ only by case onto their most common spelling.",
          default=False),
    Param("drop", "strs", "Class names to remove entirely from the labels."),
    Param("recursive", "bool", "Include sub-folders.", default=True),
    Param("label_ext", "str", "Label file extension.", default=".xml"),
)


@operation(
    "labels.rename",
    title="Rename or drop label classes",
    category="labels",
    summary="Apply a class rename map across every label file, optionally unifying case variants.",
    params=RENAME_PARAMS,
    destructive=True,
    notes=(
        "fold_case fixes case drift at the source rather than papering over it at\n"
        "export time: every spelling collapses onto whichever is most common, so\n"
        "Open/open become one class in the XML itself.\n\n"
        "drop removes objects of a class outright. An image left with no objects\n"
        "stays as a valid background/hard-negative label rather than being deleted."
    ),
)
def rename_labels(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="labels.rename", dry_run=ctx.dry_run)
    folder: Path = params["folder"]
    files = list(fs.iter_files(folder, recursive=params["recursive"], exts=[params["label_ext"]]))
    mapping: dict[str, str] = dict(params["map"] or {})
    drop = {d.strip() for d in (params["drop"] or []) if d.strip()}

    if params["fold_case"]:
        mapping = {**_case_folding_map(files), **mapping}
        result.data["case_folding"] = {
            old: new for old, new in mapping.items() if old.casefold() == new.casefold()
        }

    if not mapping and not drop:
        result.warnings.append("nothing to do: supply --map, --drop, or --fold-case")
        return result

    renamed = Counter[str]()
    dropped = Counter[str]()

    for position, path in enumerate(files, start=1):
        ctx.check_cancelled()
        ctx.emit(position, len(files), path.name)
        try:
            tree = ET.parse(path)
        except (ET.ParseError, OSError) as exc:
            result.warnings.append(f"{path.name}: unreadable ({exc})")
            result.bump("unreadable")
            continue

        root = tree.getroot()
        changed = False
        for obj in list(root.findall("object")):
            node = obj.find("name")
            if node is None or node.text is None:
                continue
            current = node.text.strip()
            if current in drop:
                root.remove(obj)
                dropped[current] += 1
                changed = True
                continue
            replacement = mapping.get(current)
            if replacement and replacement != current:
                node.text = replacement
                renamed[f"{current} -> {replacement}"] += 1
                changed = True

        if changed:
            if not ctx.dry_run:
                tree.write(path, encoding="utf-8")
            result.record(Action.WRITE, path, path, "labels updated")
            result.bump("files_changed")

    result.counts["label_files"] = len(files)
    result.counts["objects_renamed"] = sum(renamed.values())
    result.counts["objects_dropped"] = sum(dropped.values())
    result.data["renamed"] = dict(renamed)
    result.data["dropped"] = dict(dropped)
    return result


def _case_folding_map(files: list[Path]) -> dict[str, str]:
    """Map every case-variant spelling onto the most common one."""
    spellings: dict[str, Counter[str]] = {}
    for path in files:
        try:
            annotation = voc.read(path)
        except AnnotationError:
            continue
        for shape in annotation.shapes:
            spellings.setdefault(shape.label.casefold(), Counter())[shape.label] += 1

    mapping: dict[str, str] = {}
    for variants in spellings.values():
        if len(variants) < 2:
            continue
        # Most frequent spelling wins; alphabetical order breaks ties so the
        # result does not depend on directory traversal order.
        winner = sorted(variants.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for spelling in variants:
            if spelling != winner:
                mapping[spelling] = winner
    return mapping

"""Orphan handling: images without labels, labels without images.

Three original scripts covered slices of this -- one deleted both kinds, one
quarantined both, one moved just the unannotated images to a "to annotate"
folder. They are the same operation with different ``scope`` and ``action``, so
they are one operation here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import fs
from ..pairs import PairIndex
from ..registry import Param, operation
from ..types import Action, Context, OpResult

PARAMS = (
    Param("folder", "dir", "Folder to inspect.", required=True, must_exist=True),
    Param("scope", "choice", "Which orphans to act on.",
          default="both", choices=("both", "images", "labels")),
    Param("action", "choice", "What to do with them.",
          default="move", choices=("move", "delete", "report")),
    Param("target", "dir", "Destination for moved orphans. Defaults to <folder>/_orphans."),
    Param("recursive", "bool", "Include sub-folders.", default=False),
    Param("label_ext", "str", "Sidecar label extension.", default=".xml"),
)


@operation(
    "clean.orphans",
    title="Handle unpaired images and labels",
    category="clean",
    summary="Quarantine, delete, or just report images with no label and labels with no image.",
    params=PARAMS,
    destructive=True,
    safe_when=lambda p: p.get("action") == "report",
    notes=(
        "An image with no label is not necessarily junk -- it is usually just work\n"
        "not done yet. scope=images with action=move is the 'set aside what still\n"
        "needs annotating' workflow.\n\n"
        "A label with no image, by contrast, is always dead weight.\n\n"
        "action=report changes nothing and is the safe way to see the damage first."
    ),
)
def orphans(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="clean.orphans", dry_run=ctx.dry_run)
    folder: Path = params["folder"]
    action: str = params["action"]
    scope: str = params["scope"]
    target: Path = params["target"] or (folder / "_orphans")

    index = PairIndex.scan(folder, recursive=params["recursive"], label_ext=params["label_ext"])
    result.counts.update(index.summary())
    # Present even at zero, so the shape of the result never varies.
    result.counts["images_without_labels_handled"] = 0
    result.counts["labels_without_images_handled"] = 0

    jobs: list[tuple[Path, str]] = []
    if scope in ("both", "images"):
        jobs += [(p.image, "images_without_labels") for p in index.image_only() if p.image]
    if scope in ("both", "labels"):
        jobs += [(p.label, "labels_without_images") for p in index.label_only() if p.label]

    taken: set[Path] = set()
    for position, (path, bucket) in enumerate(sorted(jobs), start=1):
        ctx.check_cancelled()
        ctx.emit(position, len(jobs), f"orphan {path.name}")

        if action == "report":
            result.record(Action.SKIP, path, None, bucket)
        elif action == "delete":
            file_action = result.record(Action.DELETE, path, None, bucket)
            fs.perform(file_action, ctx)
        else:
            relative = fs.relative_to_safe(path, folder)
            destination = fs.unique_path(target / bucket / relative, taken=taken)
            taken.add(destination)
            file_action = result.record(Action.MOVE, path, destination, bucket)
            fs.perform(file_action, ctx)
        result.bump(bucket + "_handled")

    result.data["action"] = action
    if action == "move" and jobs:
        result.data["target"] = target
    return result


SPLIT_PARAMS = (
    Param("folder", "dir", "Folder holding mixed annotated and unannotated images.",
          required=True, must_exist=True),
    Param("target", "dir", "Where unannotated images are moved.", required=True),
    Param("mode", "choice", "Move them out, or copy them.",
          default="move", choices=("move", "copy")),
    Param("recursive", "bool", "Include sub-folders.", default=False),
    Param("label_ext", "str", "Sidecar label extension.", default=".xml"),
)


@operation(
    "clean.unannotated",
    title="Separate images still needing annotation",
    category="clean",
    summary="Move or copy every image that has no label file into its own folder.",
    params=SPLIT_PARAMS,
    destructive=True,
    notes="The 'what is left to annotate' queue. Use mode=copy to leave the source intact.",
)
def unannotated(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="clean.unannotated", dry_run=ctx.dry_run)
    folder: Path = params["folder"]
    target: Path = params["target"]
    verb = Action.MOVE if params["mode"] == "move" else Action.COPY

    index = PairIndex.scan(folder, recursive=params["recursive"], label_ext=params["label_ext"])
    pending = [p for p in index.image_only() if p.image]
    result.counts["scanned"] = len(index)
    result.counts["already_annotated"] = len(index.complete())

    taken: set[Path] = set()
    for position, pair in enumerate(pending, start=1):
        ctx.check_cancelled()
        assert pair.image is not None
        ctx.emit(position, len(pending), pair.image.name)
        relative = fs.relative_to_safe(pair.image, folder)
        destination = fs.unique_path(target / relative, taken=taken)
        taken.add(destination)
        fs.perform(result.record(verb, pair.image, destination, "no label file"), ctx)
        result.bump("unannotated")

    result.data["target"] = target
    return result

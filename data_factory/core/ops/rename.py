"""Sequential renaming that keeps image and label locked together.

Two fixes over the original script. It renamed each pair through a temporary
name individually, which still collides when the new name belongs to a file
later in the queue; renaming now happens in two global phases so any
permutation is safe. And it left the ``<filename>`` inside the XML pointing at
the old image, which quietly breaks every tool that trusts that field --
the label is now retargeted as part of the rename.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .. import fs
from ..pairs import PairIndex
from ..registry import Param, operation
from ..types import Action, Context, FileAction, OpResult

PARAMS = (
    Param("folder", "dir", "Folder whose files are renamed in place.", required=True, must_exist=True),
    Param("prefix", "str", "Name prefix, e.g. Door -> Door0001.jpg", required=True),
    Param("start", "int", "First number in the sequence.", default=1, minimum=0),
    Param("pad", "int", "Zero-pad numbers to this width (0 = no padding).",
          default=4, minimum=0, maximum=12),
    Param("include", "choice", "Rename complete pairs only, or every image.",
          default="pairs", choices=("pairs", "all")),
    Param("update_label_filename", "bool",
          "Rewrite <filename>/<path> inside the label to match the new image name.",
          default=True),
    Param("label_ext", "str", "Sidecar label extension.", default=".xml"),
)


@operation(
    "clean.rename",
    title="Rename images and labels in sequence",
    category="clean",
    summary="Give every pair a clean sequential name, renaming image and label together.",
    params=PARAMS,
    destructive=True,
    notes=(
        "Renaming happens in two passes (everything to temporary names, then to\n"
        "final names), so a sequence that reuses names already in the folder --\n"
        "renumbering Door5..Door9 as Door1..Door5, say -- cannot clobber anything.\n\n"
        "Always preview with --dry-run first: this rewrites the whole folder."
    ),
)
def rename(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="clean.rename", dry_run=ctx.dry_run)
    folder: Path = params["folder"]
    prefix: str = params["prefix"]
    pad: int = params["pad"]
    label_ext: str = params["label_ext"]

    index = PairIndex.scan(folder, label_ext=label_ext)
    candidates = index.complete() if params["include"] == "pairs" else index.with_images()
    result.counts["scanned"] = len(index)
    result.counts["selected"] = len(candidates)
    skipped = len(index) - len(candidates)
    if skipped:
        result.counts["skipped"] = skipped

    if not candidates:
        result.warnings.append("nothing to rename with the current include= setting")
        return result

    # Reserve every name we are about to free, so conflict detection only fires
    # on files that are genuinely outside this rename.
    involved = {path for pair in candidates for path in pair.files}
    plan: list[tuple[Path, Path]] = []
    renames: list[tuple[Path, Path, Path | None, Path | None]] = []

    number = int(params["start"])
    for pair in candidates:
        ctx.check_cancelled()
        base = f"{prefix}{number:0{pad}d}" if pad else f"{prefix}{number}"
        number += 1

        assert pair.image is not None
        new_image = folder / f"{base}{pair.image.suffix.lower()}"
        new_label = folder / f"{base}{label_ext}" if pair.label else None

        conflict = next(
            (p for p in (new_image, new_label) if p and p.exists() and p not in involved),
            None,
        )
        if conflict is not None:
            result.warnings.append(
                f"{pair.stem}: target {conflict.name} already exists and is not part of this rename"
            )
            result.bump("conflicts")
            continue

        plan.append((pair.image, new_image))
        if pair.label and new_label:
            plan.append((pair.label, new_label))
        renames.append((pair.image, new_image, pair.label, new_label))

    # Phase 1: park everything under a temporary name so the final names are all
    # free regardless of how the sequence overlaps the existing one.
    staged: list[Path] = []
    for position, (source, _) in enumerate(plan):
        ctx.check_cancelled()
        ctx.emit(position + 1, len(plan) * 2, f"staging {source.name}")
        temporary = source.parent / f".dfx-rename-{position}{source.suffix}"
        fs.perform(FileAction(Action.RENAME, source, temporary), ctx)
        staged.append(temporary)

    # Phase 2: move into place, and record the logical rename for the audit trail.
    for position, ((source, destination), temporary) in enumerate(zip(plan, staged)):
        ctx.check_cancelled()
        ctx.emit(len(plan) + position + 1, len(plan) * 2, f"renaming {destination.name}")
        fs.perform(FileAction(Action.RENAME, temporary, destination), ctx)
        result.record(Action.RENAME, source, destination, "sequential rename")
        result.bump("files_renamed")

    if params["update_label_filename"]:
        for _, new_image, old_label, new_label in renames:
            if old_label is None or new_label is None:
                continue
            if _retarget_label(new_label, new_image.name, ctx):
                result.bump("labels_retargeted")

    result.counts["pairs_renamed"] = len(renames)
    result.data["first"] = plan[0][1].name if plan else None
    result.data["last"] = plan[-1][1].name if plan else None
    return result


def _retarget_label(label_path: Path, image_name: str, ctx: Context) -> bool:
    """Point a VOC label's filename/path fields at the renamed image.

    Edited in place with ElementTree rather than round-tripped through the model
    so any extra fields a third-party tool wrote are preserved untouched.
    """
    if ctx.dry_run:
        return True
    try:
        tree = ET.parse(label_path)
    except (ET.ParseError, OSError):
        return False
    root = tree.getroot()
    changed = False
    for tag in ("filename", "path"):
        node = root.find(tag)
        if node is not None:
            node.text = image_name
            changed = True
    if changed:
        tree.write(label_path, encoding="utf-8")
    return changed

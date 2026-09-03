"""Image preprocessing that keeps the annotations correct.

Resizing a dataset is the step people most often do with a throwaway script,
and it is the step that most often corrupts a dataset -- because the boxes have
to be rescaled by exactly the same factor, and a letterboxed image needs the
padding offset added too. Doing it here means the geometry is handled once, by
:mod:`data_factory.core.imaging`, and every operation inherits it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import imaging, parallel
from ..annotations import formats
from ..annotations.model import AnnotationError
from ..registry import Param, operation
from ..types import Context, OpResult
from ..writer import DatasetWriter

#: ``same`` writes the dataset back out in the format it arrived in.
OUTPUT_FORMATS = ("same",) + formats.FORMATS
SOURCE_FORMATS = ("auto",) + formats.FORMATS

RESIZE_PARAMS = (
    Param("folder", "dir", "Dataset to resize.", required=True, must_exist=True),
    Param("output", "dir", "Where the resized dataset is written.", required=True),
    Param("width", "int", "Target width in pixels.", required=True, minimum=1),
    Param("height", "int", "Target height in pixels.", required=True, minimum=1),
    Param(
        "mode",
        "choice",
        "How to fit the image to the target size.",
        default="letterbox",
        choices=("letterbox", "stretch", "crop", "fit"),
    ),
    Param("source_format", "choice", "Input format.", default="auto", choices=SOURCE_FORMATS),
    Param("format", "choice", "Output format.", default="same", choices=OUTPUT_FORMATS),
    Param("quality", "int", "JPEG/WebP quality.", default=95, minimum=1, maximum=100),
    Param("recursive", "bool", "Include sub-folders.", default=True),
    Param("skip_larger", "bool", "Leave images already smaller than the target alone.", default=False),
    Param("workers", "int", parallel.WORKERS_HELP, default=0, minimum=0, maximum=32),
)


@operation(
    "image.resize",
    title="Resize dataset",
    category="image",
    summary="Resize every image to a fixed size, rescaling boxes and polygons to match.",
    params=RESIZE_PARAMS,
    destructive=True,
    notes=(
        "Fit modes, and when each is right:\n"
        "  letterbox  scale to fit, pad the rest. Nothing distorted, nothing\n"
        "             lost. The safe default, and what YOLO does internally.\n"
        "  stretch    fill the frame exactly, distorting aspect ratio. Fine\n"
        "             when every source image already has one shape.\n"
        "  crop       scale to fill, cut the overflow from the edges. Keeps\n"
        "             the aspect ratio but can cut objects at the border.\n"
        "  fit        scale the long edge to fit, leave the short edge free.\n"
        "             Output sizes vary; use when the trainer handles that.\n"
        "\n"
        "Objects reduced to under two pixels are dropped, because a one-pixel\n"
        "box is noise that a model will only learn to distrust."
    ),
)
def resize(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="image.resize", dry_run=ctx.dry_run)
    target_w, target_h = params["width"], params["height"]
    mode = params["mode"]

    def build(width: int, height: int):
        if params["skip_larger"] and width <= target_w and height <= target_h:
            return None
        return imaging.make_resize(width, height, target_w, target_h, mode)

    return _apply_geometry(
        params, ctx, result, build, label=f"{target_w}x{target_h} ({mode})"
    )


ORIENT_PARAMS = (
    Param("folder", "dir", "Dataset to straighten.", required=True, must_exist=True),
    Param("output", "dir", "Where the corrected dataset is written.", required=True),
    Param("source_format", "choice", "Input format.", default="auto", choices=SOURCE_FORMATS),
    Param("format", "choice", "Output format.", default="same", choices=OUTPUT_FORMATS),
    Param("quality", "int", "JPEG/WebP quality.", default=95, minimum=1, maximum=100),
    Param("recursive", "bool", "Include sub-folders.", default=True),
    Param("workers", "int", parallel.WORKERS_HELP, default=0, minimum=0, maximum=32),
)


@operation(
    "image.autoorient",
    title="Apply EXIF orientation",
    category="image",
    summary="Bake rotation from the EXIF tag into the pixels and strip it, so every reader agrees.",
    params=ORIENT_PARAMS,
    destructive=True,
    notes=(
        "A phone photo is often stored sideways with a tag saying 'rotate me'.\n"
        "Pillow and most annotation GUIs honour that tag; OpenCV and most\n"
        "training loaders ignore it. So the person drawing the boxes sees an\n"
        "upright image, the trainer sees a sideways one, and the labels land in\n"
        "the wrong place -- with nothing anywhere reporting an error.\n"
        "\n"
        "This rewrites the pixels in their displayed orientation, rotates the\n"
        "annotations with them, and leaves no tag behind to disagree about.\n"
        "Run it first, before anything else touches the images."
    ),
)
def autoorient(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="image.autoorient", dry_run=ctx.dry_run)
    rotated = 0

    def build(width: int, height: int):
        return None  # decided per file below, since it depends on the EXIF tag

    dataset, target_format = _load(params, ctx, result)
    if dataset is None:
        return result

    writer = DatasetWriter(
        params["output"],
        target_format,
        ctx,
        classes=dataset.labels,
        quality=params["quality"],
    )
    total = len(dataset.items)
    for position, item in enumerate(dataset.items, start=1):
        ctx.check_cancelled()
        ctx.emit(position, total, item.stem)
        if item.image is None:
            continue
        orientation = imaging.exif_orientation(item.image)
        if orientation in (0, 1):
            writer.add(
                item.annotation,
                stem=item.stem,
                source=item.image,
                split=item.split,
                suffix=item.image.suffix,
            )
            continue
        try:
            array = imaging.load(item.image)
        except OSError as exc:
            result.warnings.append(f"{item.image.name}: {exc}")
            continue
        transform = imaging.exif_transform(orientation, array.shape[1], array.shape[0])
        writer.add(
            imaging.transform_annotation(item.annotation, transform),
            stem=item.stem,
            image=imaging.apply_exif(array, orientation),
            source=item.image,
            split=item.split,
            suffix=item.image.suffix,
        )
        rotated += 1

    report = writer.close()
    result.actions.extend(writer.actions)
    result.counts.update(
        {"images": total, "rotated": rotated, "unchanged": total - rotated, "written": report["labels"]}
    )
    result.warnings.extend(dataset.warnings)
    result.warnings.extend(report["warnings"])
    result.data.update({"output": params["output"], "format": target_format})
    if not rotated:
        result.warnings.append("no image carried a non-upright EXIF orientation")
    return result


TILE_PARAMS = (
    Param("folder", "dir", "Dataset to tile.", required=True, must_exist=True),
    Param("output", "dir", "Where the tiles are written.", required=True),
    Param("tile_width", "int", "Tile width in pixels.", default=640, minimum=32),
    Param("tile_height", "int", "Tile height in pixels.", default=640, minimum=32),
    Param(
        "overlap",
        "float",
        "Fraction each tile overlaps its neighbour (0-0.9).",
        default=0.2,
        minimum=0.0,
        maximum=0.9,
    ),
    Param(
        "min_visibility",
        "float",
        "Fraction of an object that must survive the cut for it to be kept.",
        default=0.3,
        minimum=0.0,
        maximum=1.0,
    ),
    Param("keep_empty", "bool", "Keep tiles containing no objects, as hard negatives.", default=False),
    Param("source_format", "choice", "Input format.", default="auto", choices=SOURCE_FORMATS),
    Param("format", "choice", "Output format.", default="same", choices=OUTPUT_FORMATS),
    Param("quality", "int", "JPEG/WebP quality.", default=95, minimum=1, maximum=100),
    Param("recursive", "bool", "Include sub-folders.", default=True),
    Param("workers", "int", parallel.WORKERS_HELP, default=0, minimum=0, maximum=32),
)


@operation(
    "image.tile",
    title="Slice into tiles",
    category="image",
    summary="Cut large images into an overlapping grid, reassigning annotations to each tile.",
    params=TILE_PARAMS,
    destructive=True,
    notes=(
        "The fix for small objects in big images. A detector resizes a 4000px\n"
        "frame down to 640 before it ever looks at it, so a 40-pixel object\n"
        "becomes 6 pixels and is effectively invisible. Tiling keeps the native\n"
        "resolution by making each tile its own training image.\n"
        "\n"
        "overlap exists because an object sitting on a tile boundary would\n"
        "otherwise be cut in half in every tile it appears in. At 0.2 an object\n"
        "up to a fifth of a tile wide always lands whole somewhere.\n"
        "\n"
        "min_visibility discards the slivers: without it you teach the model\n"
        "that three pixels of a door is a door.\n"
        "\n"
        "Expect the image count to multiply. A 4000x3000 source at 640 with 0.2\n"
        "overlap yields around 40 tiles."
    ),
)
def tile(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="image.tile", dry_run=ctx.dry_run)
    dataset, target_format = _load(params, ctx, result)
    if dataset is None:
        return result

    tile_w, tile_h = params["tile_width"], params["tile_height"]
    overlap = params["overlap"]
    keep_empty = params["keep_empty"]

    writer = DatasetWriter(
        params["output"], target_format, ctx, classes=dataset.labels, quality=params["quality"]
    )
    total = len(dataset.items)
    tiles_written = 0
    tiles_empty = 0
    objects_kept = 0

    for position, item in enumerate(dataset.items, start=1):
        ctx.check_cancelled()
        ctx.emit(position, total, item.stem)
        if item.image is None:
            continue
        try:
            array = imaging.load(item.image)
        except OSError as exc:
            result.warnings.append(f"{item.image.name}: {exc}")
            continue

        height, width = array.shape[:2]
        for index, (x, y, w, h) in enumerate(
            imaging.tile_windows(width, height, tile_w, tile_h, overlap)
        ):
            crop = imaging.Crop(out_size=(w, h), left=x, top=y)
            annotation = imaging.transform_annotation(
                item.annotation, crop, min_visibility=params["min_visibility"]
            )
            if annotation.is_background and not keep_empty:
                tiles_empty += 1
                continue
            writer.add(
                annotation,
                stem=f"{item.stem}_t{index:03d}",
                image=crop.image(array),
                source=item.image,
                split=item.split,
                suffix=item.image.suffix,
            )
            tiles_written += 1
            objects_kept += len(annotation.shapes)

    report = writer.close()
    result.actions.extend(writer.actions)
    result.counts.update(
        {
            "source_images": total,
            "tiles_written": tiles_written,
            "tiles_skipped_empty": tiles_empty,
            "objects": objects_kept,
        }
    )
    result.warnings.extend(dataset.warnings)
    result.warnings.extend(report["warnings"])
    result.data.update({"output": params["output"], "format": target_format})
    if tiles_written == 0:
        result.warnings.append(
            "no tiles kept -- every tile came out empty. Use --keep-empty, or "
            "check the tile size is smaller than the images."
        )
    return result


# ------------------------------------------------------------------ shared

def _load(params: dict[str, Any], ctx: Context, result: OpResult):
    """Read the source dataset and settle the output format."""
    folder: Path = params["folder"]
    output: Path = params["output"]
    if output.resolve() == folder.resolve():
        result.fail("output must differ from folder -- these operations rewrite pixels")
        return None, ""
    try:
        dataset = formats.read_dataset(
            folder, params["source_format"], recursive=params["recursive"], ctx=ctx
        )
    except AnnotationError as exc:
        result.fail(str(exc))
        return None, ""
    if not dataset.items:
        result.fail(f"no images found in {folder}")
        return None, ""
    chosen = params["format"]
    return dataset, (dataset.source_format if chosen == "same" else chosen)


def _apply_geometry(params, ctx, result, build, *, label: str) -> OpResult:
    """Run a per-image geometric transform over a whole dataset.

    Decoding, transforming and re-encoding all happen on worker threads; only
    the disk write and the writer's bookkeeping stay on the calling thread,
    which keeps the output order -- and therefore COCO image ids and YOLO class
    indices -- independent of which thread finished first.
    """
    dataset, target_format = _load(params, ctx, result)
    if dataset is None:
        return result

    writer = DatasetWriter(
        params["output"], target_format, ctx, classes=dataset.labels, quality=params["quality"]
    )
    quality = params["quality"]
    total = len(dataset.items)
    changed = 0
    dropped = 0

    def transform_one(item):
        """Pure: no shared state, so it is safe on a worker thread."""
        if item.image is None:
            return None
        array = imaging.load(item.image)
        height, width = array.shape[:2]
        transform = build(width, height)
        if transform is None:
            return ("passthrough", item.annotation, None)
        moved = imaging.transform_annotation(item.annotation, transform)
        encoded = imaging.encode(transform.image(array), item.image.suffix, quality=quality)
        return ("changed", moved, encoded)

    def note_failure(item, exc: Exception) -> None:
        result.warnings.append(f"{item.image.name if item.image else item.stem}: {exc}")

    for item, outcome in parallel.map_bounded(
        dataset.items,
        transform_one,
        ctx=ctx,
        workers=params["workers"],
        label=lambda item: f"{item.stem} -> {label}",
        on_error=note_failure,
    ):
        if outcome is None:
            continue
        kind, annotation, encoded = outcome
        if kind == "passthrough":
            writer.add(
                annotation,
                stem=item.stem,
                source=item.image,
                split=item.split,
                suffix=item.image.suffix,
            )
            continue

        dropped += max(0, len(item.annotation.shapes) - len(annotation.shapes))
        writer.add(
            annotation,
            stem=item.stem,
            encoded=encoded,
            source=item.image,
            split=item.split,
            suffix=item.image.suffix,
        )
        changed += 1

    report = writer.close()
    result.actions.extend(writer.actions)
    result.counts.update(
        {"images": total, "resized": changed, "unchanged": total - changed, "objects_dropped": dropped}
    )
    result.warnings.extend(dataset.warnings)
    result.warnings.extend(report["warnings"])
    result.data.update({"output": params["output"], "format": target_format})
    if dropped:
        result.warnings.append(
            f"{dropped} object(s) became smaller than {imaging.MIN_OBJECT_PIXELS}px "
            f"and were dropped"
        )
    return result

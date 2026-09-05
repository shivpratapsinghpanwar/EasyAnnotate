"""Model-in-the-loop review: getting predictions back in, and judging them.

The loop this closes: annotate a few hundred images by hand, train, run
inference on the rest, **load those predictions back in**, fix the wrong ones,
export, retrain. Each turn costs less than the one before, and the only part
that was missing is this one.

Three rules, each of which exists because the obvious alternative fails
quietly:

**Matching is reported, never assumed.** Predictions are matched to images by
filename stem, and the import returns exactly how many matched and what did
not. Silently importing zero shapes because ``img_001.txt`` met ``IMG_001.jpg``
is the single most common way this feature disappoints people, and it looks
identical to a model that predicted nothing.

**An import replaces the previous run's predictions, not your work.** Model
shapes from an earlier run are cleared; hand-drawn shapes, and model shapes a
person has already corrected, are kept. Otherwise a second run doubles every
box on the image.

**Reviewed images are left alone by default.** Dropping fresh predictions onto
an image somebody already signed off would quietly un-finish it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..core.annotations import coco as coco_fmt
from ..core.annotations import formats, yolo
from ..core.annotations.model import Annotation, AnnotationError
from . import images as images_mod
from . import labels as labels_mod
from . import projects as projects_mod
from . import shapes as shapes_mod
from .store import Store, StoreError, new_id

#: States an import will not touch unless explicitly told to.
PROTECTED_STATES = ("reviewed",)


class ReviewError(StoreError):
    """Raised when predictions cannot be read or matched."""


def read_predictions(
    source: Path,
    *,
    project_classes: list[str] | None = None,
    sizes: dict[str, tuple[int, int]] | None = None,
) -> tuple[list[Annotation], list[str]]:
    """Read a folder or file of model output into annotations.

    Handles the shapes real tools actually emit:

    * a folder of YOLO ``.txt`` with a trailing confidence column
    * a folder of VOC ``.xml`` or LabelMe ``.json``
    * a COCO dataset json
    * a COCO **detection results** json -- a bare list, which is what
      torchvision, mmdetection and Ultralytics evaluation scripts write

    ``sizes`` maps a filename stem to the image's ``(width, height)`` and comes
    from the project. It is needed because **a prediction folder has no images
    in it** -- an inference run writes labels, not pictures -- and YOLO
    coordinates are normalised, so without the size there is nothing to
    multiply by. The ordinary dataset reader pairs labels with images and
    therefore cannot read this at all; that is why it is not simply reused.
    """
    notes: list[str] = []
    source = Path(source)
    sizes = sizes or {}

    if source.is_file() and source.suffix.lower() == ".json":
        return _read_json(source, project_classes, notes)

    if not source.is_dir():
        raise ReviewError(f"{source} is not a folder or a .json file")

    # A results json sitting alone in a folder is the common case when someone
    # zips up "the output", so look for it before anything else.
    for candidate in sorted(source.rglob("*.json")):
        if not candidate.is_file():
            continue
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if coco_fmt.is_detection_results(document):
            notes.append(f"read COCO detection results from {candidate.name}")
            return (
                coco_fmt.results_to_annotations(
                    document, names=_names_for(project_classes)
                ),
                notes,
            )

    label_files = [p for p in sorted(source.rglob("*.txt")) if p.is_file()]
    if label_files:
        return _read_yolo_labels(source, label_files, project_classes, sizes, notes)

    try:
        dataset = formats.read_dataset(source, "auto", recursive=True)
    except AnnotationError as exc:
        raise ReviewError(f"could not read predictions: {exc}") from exc

    notes.extend(dataset.warnings[:10])
    notes.append(f"read {dataset.source_format} predictions")
    return [item.annotation for item in dataset.items], notes


def _read_yolo_labels(
    source: Path,
    label_files: list[Path],
    project_classes: list[str] | None,
    sizes: dict[str, tuple[int, int]],
    notes: list[str],
) -> tuple[list[Annotation], list[str]]:
    """Read a folder of bare YOLO ``.txt`` predictions.

    The class list comes from a ``data.yaml`` if the run shipped one, and
    otherwise from the project -- which is the right guess, because a model
    trained from this project was trained on this project's class order. When
    neither is available the names come out as ``class_0``, visibly wrong
    rather than quietly attributed to the wrong class.
    """
    yaml_path = yolo.find_data_yaml(source)
    if yaml_path is not None:
        try:
            classes, _ = yolo.read_data_yaml(yaml_path)
            notes.append(f"class order read from {yaml_path.name}")
        except AnnotationError:
            classes = yolo.ClassList(list(project_classes or []))
            notes.append("data.yaml was unreadable; used the project's class order")
    elif project_classes:
        classes = yolo.ClassList(list(project_classes))
        notes.append("no data.yaml in the predictions; used the project's class order")
    else:
        classes = yolo.ClassList([])
        notes.append(
            "no data.yaml and no classes in the project, so classes are named "
            "class_0, class_1 ... -- rename them after importing"
        )

    annotations: list[Annotation] = []
    unmeasured = 0
    for path in label_files:
        stem = path.stem
        width, height = sizes.get(stem, (0, 0))

        if stem not in sizes:
            # This file matches no image in the project. It still has to appear
            # in the result, empty, so the caller can *report* it as unmatched
            # -- dropping it here would turn "your filenames do not line up"
            # into "no predictions were found", which sends people looking in
            # entirely the wrong place.
            annotations.append(
                Annotation(filename=f"{stem}.jpg", width=0, height=0, shapes=[])
            )
            continue

        if not (width and height):
            # The image is ours but we never managed to measure it. Normalised
            # coordinates cannot be turned into pixels without a size, and
            # guessing one would put every box in the top-left corner.
            unmeasured += 1
            continue

        try:
            annotations.append(
                yolo.read(path, classes, width, height, filename=f"{stem}.jpg")
            )
        except AnnotationError as exc:
            notes.append(f"{path.name}: {exc}")

    if unmeasured:
        notes.append(
            f"{unmeasured} image(s) have no recorded size, so their predictions "
            f"could not be scaled and were skipped"
        )
    notes.append("read YOLO predictions")
    return annotations, notes


def _read_json(
    path: Path, project_classes: list[str] | None, notes: list[str]
) -> tuple[list[Annotation], list[str]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReviewError(f"{path.name}: not readable JSON ({exc})") from exc

    if coco_fmt.is_detection_results(document):
        notes.append("read COCO detection results (a bare list of predictions)")
        return (
            coco_fmt.results_to_annotations(document, names=_names_for(project_classes)),
            notes,
        )

    try:
        annotations = coco_fmt.to_annotations(document, unsupported_out=notes)
    except AnnotationError as exc:
        raise ReviewError(str(exc)) from exc
    notes.append("read a COCO dataset document")
    return annotations, notes


def _names_for(project_classes: list[str] | None) -> dict[Any, str]:
    """Map COCO category ids onto the project's class names.

    A detection-results file carries category *ids* and no names. Ultralytics
    and torchvision both number them from 0 in the order of the training class
    list, which for a model trained from this project is the project's own
    order -- so that is what we assume, and the matching report is what catches
    it when the assumption is wrong.
    """
    if not project_classes:
        return {}
    mapping: dict[Any, str] = {}
    for index, name in enumerate(project_classes):
        mapping[index] = name
        # COCO's own convention numbers categories from 1.
        mapping.setdefault(index + 1, name)
    return mapping


def import_predictions(
    store: Store,
    project: projects_mod.Project,
    source: Path,
    *,
    name: str = "",
    min_confidence: float = 0.0,
    overwrite_reviewed: bool = False,
) -> dict[str, Any]:
    """Load a model's output onto a project's images.

    Returns a matching report. Read it: the interesting number is usually the
    one that did *not* match.
    """
    class_list = labels_mod.class_list(store, project)
    by_stem = images_mod.by_stem(store, project)
    # A prediction folder holds labels, not images, so the sizes YOLO needs to
    # denormalise its coordinates have to come from the project.
    sizes = {stem: (image.width, image.height) for stem, image in by_stem.items()}

    annotations, notes = read_predictions(
        source, project_classes=class_list, sizes=sizes
    )
    if not annotations:
        raise ReviewError(
            "no predictions were found in that file or folder. "
            + ("; ".join(notes) if notes else "")
        )
    matched = 0
    skipped_reviewed = 0
    unmatched_files: list[str] = []
    imported = 0
    dropped_low = 0
    touched: list[images_mod.Image] = []

    for annotation in annotations:
        stem = Path(annotation.filename).stem
        image = by_stem.get(stem)
        if image is None:
            unmatched_files.append(annotation.filename)
            continue
        if image.state in PROTECTED_STATES and not overwrite_reviewed:
            skipped_reviewed += 1
            continue

        keep = []
        for shape in annotation.shapes:
            if shape.confidence is not None and shape.confidence < min_confidence:
                dropped_low += 1
                continue
            keep.append(shape)

        # Predictions arrive in the source's own coordinate space. If the
        # source did not say how big the image was, trust the project's size --
        # otherwise clamping would squash every box into the top-left corner.
        set_predictions(store, project, image, keep)
        matched += 1
        imported += len(keep)
        touched.append(image)

    run_id = new_id()
    with store.write() as conn:
        conn.execute(
            "INSERT INTO model_run (id, project_id, name, source, matched, unmatched, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                project.id,
                name or Path(source).name or "model run",
                str(Path(source).name),
                matched,
                len(unmatched_files),
                time.time(),
            ),
        )
        projects_mod.touch(conn, project.id)

    unmatched_images = len(by_stem) - matched - skipped_reviewed
    if unmatched_files:
        notes.append(
            f"{len(unmatched_files)} prediction file(s) matched no image, "
            f"for example: {', '.join(unmatched_files[:5])}"
        )
    if matched == 0:
        notes.append(
            "nothing matched. Predictions are matched to images by filename "
            "stem, so check that the names line up."
        )
    if skipped_reviewed:
        notes.append(
            f"{skipped_reviewed} image(s) were already reviewed and were left "
            f"untouched; pass overwrite_reviewed to include them"
        )
    if dropped_low:
        notes.append(f"{dropped_low} prediction(s) below the confidence threshold were dropped")

    return {
        "run_id": run_id,
        "matched": matched,
        "unmatched_files": len(unmatched_files),
        "unmatched_images": max(0, unmatched_images),
        "shapes_imported": imported,
        "skipped_reviewed": skipped_reviewed,
        "dropped_below_threshold": dropped_low,
        "notes": notes,
    }


def set_predictions(
    store: Store,
    project: projects_mod.Project,
    image: images_mod.Image,
    predicted: list[Any],
) -> None:
    """Replace this image's untouched model shapes with a new run's.

    Kept: everything a person drew, and every model shape a person has since
    corrected -- that correction is human work now. Cleared: model shapes
    nobody has looked at, which is what stops a second run doubling the boxes.
    """
    existing = shapes_mod.listing(store, project, image)
    keep = [s for s in existing if not s.is_model or s.is_touched]

    incoming: list[tuple[str | None, Any]] = [(s.id, s.shape) for s in keep]
    incoming.extend((None, shape) for shape in predicted)

    # advance_state stays off: a prediction is not somebody starting work.
    shapes_mod.save(
        store, project, image, incoming, base_version=None,
        origin="model", advance_state=False,
    )

    # Record what was predicted, so a later deletion still counts as a
    # correction. Without this, removing a false positive would look identical
    # to it never having been predicted.
    kept_model = sum(1 for s in keep if s.is_model)
    with store.write() as conn:
        conn.execute(
            "UPDATE image SET predicted_count = ? WHERE id = ?",
            (len(predicted) + kept_model, image.id),
        )

    if image.state == "todo":
        images_mod.set_state(store, project, image, "predicted")


def accept_all(
    store: Store,
    project: projects_mod.Project,
    image: images_mod.Image,
) -> int:
    """Take this image's predictions as correct: mark it reviewed.

    This is what ``Space`` does, and it is the throughput argument -- on a
    model that is mostly right, a thousand images is fifteen minutes rather
    than a day.

    Note what it deliberately does *not* do: it leaves ``origin`` alone. The
    first version of this flipped accepted shapes to ``human``, which felt
    tidy and quietly destroyed the agreement metric -- the denominator is
    "model shapes a person has reviewed", and rewriting them out of existence
    means agreement can only ever be computed over shapes nobody has looked at.
    An accepted prediction is still a prediction; what changed is that someone
    has now vouched for it, and that is what the image state records.
    """
    now = time.time()
    with store.write() as conn:
        confirmed = conn.execute(
            "SELECT COUNT(*) AS n FROM shape WHERE image_id = ? AND origin = 'model'",
            (image.id,),
        ).fetchone()["n"]
        conn.execute(
            "UPDATE image SET state = 'reviewed', updated_at = ? WHERE id = ?",
            (now, image.id),
        )
        projects_mod.touch(conn, project.id)
    image.state = "reviewed"
    return int(confirmed)


def runs(store: Store, project: projects_mod.Project) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "matched": row["matched"],
            "unmatched": row["unmatched"],
            "created_at": row["created_at"],
        }
        for row in store.query(
            "SELECT * FROM model_run WHERE project_id = ? ORDER BY created_at DESC",
            (project.id,),
        )
    ]

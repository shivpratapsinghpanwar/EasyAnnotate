"""COCO instances.json construction.

The category handling here is deliberate. The original converter keyed
categories on the raw XML string, so a dataset whose labels drifted between
``Open`` and ``open`` came out with four classes instead of two -- exactly what
happened to Project_Door V3 (Close/Open/open/close split 1341/1385/645/500).
The builder now folds case by default and reports every merge it made, so the
drift is fixed *and* visible rather than silently baked into the export.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..fs import write_text_atomic
from .model import Annotation, AnnotationError, Point, Shape

#: How to decide whether two label spellings are the same class.
CASE_POLICIES = ("exact", "casefold")


@dataclass
class CategoryTable:
    """Assigns stable category ids, optionally folding case-variant spellings."""

    case_policy: str = "casefold"
    label_map: dict[str, str] = field(default_factory=dict)

    _ids: dict[str, int] = field(default_factory=dict, init=False)
    _names: dict[str, str] = field(default_factory=dict, init=False)
    _variants: dict[str, set[str]] = field(default_factory=dict, init=False)

    def key_for(self, label: str) -> str:
        renamed = self.label_map.get(label, label).strip()
        return renamed.casefold() if self.case_policy == "casefold" else renamed

    def id_for(self, label: str) -> int:
        renamed = self.label_map.get(label, label).strip()
        key = self.key_for(label)
        if key not in self._ids:
            self._ids[key] = len(self._ids) + 1
            # First spelling encountered becomes the exported display name; the
            # scan is sorted, so this is reproducible across runs.
            self._names[key] = renamed
            self._variants[key] = set()
        self._variants[key].add(renamed)
        return self._ids[key]

    def categories(self) -> list[dict[str, Any]]:
        return [
            {"id": self._ids[key], "name": self._names[key], "supercategory": "none"}
            for key in sorted(self._ids, key=lambda k: self._ids[k])
        ]

    def merges(self) -> dict[str, list[str]]:
        """Classes that absorbed more than one spelling, for reporting."""
        return {
            self._names[key]: sorted(variants)
            for key, variants in self._variants.items()
            if len(variants) > 1
        }


class CocoBuilder:
    """Accumulates images and annotations into a COCO dictionary."""

    def __init__(
        self,
        description: str = "Data Factory export",
        version: str = "1.0",
        case_policy: str = "casefold",
        label_map: dict[str, str] | None = None,
    ) -> None:
        if case_policy not in CASE_POLICIES:
            raise ValueError(f"case_policy must be one of {CASE_POLICIES}")
        self.description = description
        self.version = version
        self.categories = CategoryTable(case_policy=case_policy, label_map=label_map or {})
        self.images: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self.background_images = 0
        self._next_image_id = 1
        self._next_ann_id = 1

    def add(self, annotation: Annotation, file_name: str | None = None) -> int:
        """Add one annotated image; returns its COCO image id."""
        image_id = self._next_image_id
        self._next_image_id += 1
        self.images.append(
            {
                "id": image_id,
                "file_name": file_name or annotation.filename,
                "width": annotation.width,
                "height": annotation.height,
            }
        )
        if annotation.is_background:
            # Hard negatives belong in the export: an image entry with no
            # annotations is how COCO expresses "nothing here, on purpose".
            self.background_images += 1
        for shape in annotation.shapes:
            self._add_shape(shape, image_id)
        return image_id

    def _add_shape(self, shape: Shape, image_id: int) -> None:
        xmin, ymin, xmax, ymax = shape.bbox
        entry = {
            "id": self._next_ann_id,
            "image_id": image_id,
            "category_id": self.categories.id_for(shape.label),
            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
            "segmentation": [shape.flat_polygon()] if shape.has_polygon else [],
            # Polygon area is the true mask area; falling back to the box
            # keeps bbox-only objects comparable in COCO eval.
            "area": round(shape.polygon_area(), 2),
            "iscrowd": 0,
        }
        if shape.is_point:
            # COCO has no point type; a visible keypoint with a one-pixel box
            # is the form its keypoint evaluators and viewers both accept.
            entry["bbox"] = [xmin, ymin, 1, 1]
            entry["area"] = 1
            entry["keypoints"] = [xmin, ymin, 2]
            entry["num_keypoints"] = 1
        self.annotations.append(entry)
        self._next_ann_id += 1

    def build(self) -> dict[str, Any]:
        return {
            "info": {"description": self.description, "version": self.version},
            "licenses": [],
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories.categories(),
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, json.dumps(self.build(), indent=2))
        return path

    def stats(self) -> dict[str, int]:
        return {
            "images": len(self.images),
            "annotations": len(self.annotations),
            "categories": len(self.categories.categories()),
            "background_images": self.background_images,
        }


def load(path: Path) -> dict[str, Any]:
    """Read an existing instances.json."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _score(ann: dict[str, Any]) -> float | None:
    """A detection's confidence, if it has one."""
    raw = ann.get("score")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def is_detection_results(document: Any) -> bool:
    """Whether this is the *detection results* form rather than a dataset.

    torchvision, mmdetection and Ultralytics all evaluate against a bare JSON
    **list** of ``{image_id, category_id, bbox, score}`` -- no ``images``, no
    ``categories``. It is what you get out of a model, and it is therefore the
    exact file someone will try to load back in for review.
    """
    return isinstance(document, list)


def results_to_annotations(
    results: list[dict[str, Any]],
    *,
    names: dict[Any, str] | None = None,
    sizes: dict[Any, tuple[int, int]] | None = None,
    filenames: dict[Any, str] | None = None,
) -> list[Annotation]:
    """Read the detection-results form into annotations.

    Everything a dataset document would have carried -- image file names,
    sizes, category names -- is absent here, because the file is designed to be
    read *alongside* the dataset it was evaluated on. Callers pass what they
    know; anything missing degrades to a usable stand-in rather than an error,
    because refusing to show a person their model's output on a technicality
    helps nobody.
    """
    names = names or {}
    sizes = sizes or {}
    filenames = filenames or {}
    by_image: dict[Any, Annotation] = {}

    for entry in results:
        if not isinstance(entry, dict):
            continue
        image_id = entry.get("image_id")
        bbox = entry.get("bbox") or [0, 0, 0, 0]
        try:
            x, y, w, h = (float(v) for v in bbox[:4])
        except (TypeError, ValueError):
            continue

        annotation = by_image.get(image_id)
        if annotation is None:
            width, height = sizes.get(image_id, (0, 0))
            annotation = Annotation(
                filename=filenames.get(image_id, f"{image_id}"),
                width=width,
                height=height,
                shapes=[],
            )
            by_image[image_id] = annotation

        category = entry.get("category_id")
        segmentation = entry.get("segmentation")
        polygon = (
            _first_polygon(segmentation)
            if isinstance(segmentation, list) and segmentation
            else None
        )
        annotation.shapes.append(
            Shape(
                label=names.get(category, f"class_{category}"),
                bbox=(int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))),
                polygon=polygon,
                confidence=_score(entry),
            )
        )

    for annotation in by_image.values():
        if (annotation.width <= 0 or annotation.height <= 0) and annotation.shapes:
            annotation.width = max(s.bbox[2] for s in annotation.shapes) + 1
            annotation.height = max(s.bbox[3] for s in annotation.shapes) + 1

    return [by_image[k] for k in sorted(by_image, key=lambda k: str(by_image[k].filename))]


def to_annotations(
    coco: dict[str, Any], *, unsupported_out: list[str] | None = None
) -> list[Annotation]:
    """Explode a COCO document back into one :class:`Annotation` per image.

    This is what makes COCO a two-way format rather than an export-only dead
    end. Images carrying no annotations are preserved as backgrounds, since in
    COCO that absence is meaningful rather than incidental.

    RLE segmentations (``iscrowd=1``, or a dict-valued ``segmentation``) cannot
    become a polygon without a mask decoder, so the box is kept and the fact is
    reported through ``unsupported_out`` rather than passed off as a polygon.
    """
    if is_detection_results(coco):
        # Previously this raised AttributeError deep inside a dict lookup,
        # which told the user nothing about what they had actually handed over.
        raise AnnotationError(
            "this is a COCO *detection results* file (a bare list of "
            "predictions), not a dataset. It has no images or categories of "
            "its own -- load it against the project it was run on, with "
            "results_to_annotations, or import it as predictions."
        )
    if not isinstance(coco, dict):
        raise AnnotationError(
            f"a COCO document must be an object, got {type(coco).__name__}"
        )

    names = {c["id"]: str(c.get("name", f"class_{c['id']}")) for c in coco.get("categories", [])}
    by_image: dict[int, Annotation] = {}

    for image in coco.get("images", []):
        by_image[image["id"]] = Annotation(
            filename=Path(str(image.get("file_name", ""))).name,
            width=int(image.get("width") or 0),
            height=int(image.get("height") or 0),
            shapes=[],
        )

    rle_count = 0
    for ann in coco.get("annotations", []):
        annotation = by_image.get(ann.get("image_id"))
        if annotation is None:
            continue
        bbox = ann.get("bbox") or [0, 0, 0, 0]
        try:
            x, y, w, h = (float(v) for v in bbox[:4])
        except (TypeError, ValueError):
            continue

        segmentation = ann.get("segmentation")
        polygon: list[Point] | None = None
        if isinstance(segmentation, dict) or ann.get("iscrowd"):
            rle_count += 1
        elif isinstance(segmentation, list) and segmentation:
            polygon = _first_polygon(segmentation)

        annotation.shapes.append(
            Shape(
                label=names.get(ann.get("category_id"), f"class_{ann.get('category_id')}"),
                bbox=(
                    int(round(x)),
                    int(round(y)),
                    int(round(x + w)),
                    int(round(y + h)),
                ),
                polygon=polygon,
                # 'score' is what every detector writes and no ground-truth
                # file has. Carrying it is what lets the review layer tell a
                # prediction from a label rather than trusting a filename.
                confidence=_score(ann),
            )
        )

    if rle_count and unsupported_out is not None:
        unsupported_out.append(
            f"{rle_count} RLE/crowd segmentation(s) kept as bounding boxes only"
        )

    # Recover sizes the document omitted, so a YOLO export downstream is not
    # left dividing by zero.
    for annotation in by_image.values():
        if (annotation.width <= 0 or annotation.height <= 0) and annotation.shapes:
            annotation.width = max(s.bbox[2] for s in annotation.shapes) + 1
            annotation.height = max(s.bbox[3] for s in annotation.shapes) + 1

    return [by_image[k] for k in sorted(by_image, key=lambda k: by_image[k].filename)]


def _first_polygon(segmentation: list[Any]) -> list[Point] | None:
    """Take the outer ring. Multi-part masks lose their holes, which is the
    honest limit of a single-polygon model -- and what every VOC/YOLO consumer
    downstream can represent anyway."""
    for part in segmentation:
        if isinstance(part, list) and len(part) >= 6:
            values = [float(v) for v in part]
            return [
                (int(round(values[i])), int(round(values[i + 1])))
                for i in range(0, len(values) - 1, 2)
            ]
    return None


def find_annotation_files(folder: Path) -> list[Path]:
    """Locate COCO json files under the layouts exporters actually produce."""
    candidates: list[Path] = []
    for pattern in ("_annotations.coco.json", "instances*.json", "*.json"):
        for path in sorted(folder.glob(pattern)):
            if path.is_file() and path not in candidates:
                candidates.append(path)
        if candidates:
            break
    annotations_dir = folder / "annotations"
    if annotations_dir.is_dir():
        for path in sorted(annotations_dir.glob("*.json")):
            if path not in candidates:
                candidates.append(path)
    return candidates


def summarise(coco: dict[str, Any]) -> dict[str, Any]:
    """Headline numbers for an existing COCO file, including case collisions."""
    categories: Iterable[dict[str, Any]] = coco.get("categories", [])
    by_name = {c["id"]: c["name"] for c in categories}
    counts: dict[str, int] = {}
    segmented = 0
    for ann in coco.get("annotations", []):
        counts[by_name.get(ann.get("category_id"), "?")] = (
            counts.get(by_name.get(ann.get("category_id"), "?"), 0) + 1
        )
        if ann.get("segmentation"):
            segmented += 1

    folded: dict[str, list[str]] = {}
    for name in by_name.values():
        folded.setdefault(name.strip().casefold(), []).append(name)
    collisions = {k: sorted(v) for k, v in folded.items() if len(v) > 1}

    image_ids_with_anns = {a.get("image_id") for a in coco.get("annotations", [])}
    return {
        "images": len(coco.get("images", [])),
        "annotations": len(coco.get("annotations", [])),
        "categories": len(by_name),
        "per_category": dict(sorted(counts.items())),
        "with_segmentation": segmented,
        "background_images": len(coco.get("images", [])) - len(image_ids_with_anns),
        "case_collisions": collisions,
    }

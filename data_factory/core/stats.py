"""Dataset statistics, and the judgements drawn from them.

Counting objects is easy and not very useful. What decides whether a dataset
trains well is the *shape* of those counts: whether one class has forty times
another's examples, whether the objects are so small they vanish at the model's
input resolution, whether half the images are a different size from the rest.

Everything here is pure measurement plus explicitly-stated thresholds. The
thresholds are the arguable part, so they are named constants with a reason
attached rather than magic numbers buried in a conditional.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .annotations.formats import Dataset

#: COCO's own small/medium/large split, in pixels of box area. Using the same
#: boundaries means our "small object" agrees with everyone's evaluation metric.
SMALL_AREA = 32 * 32
MEDIUM_AREA = 96 * 96

#: A box under this fraction of the frame survives a 640px resize as roughly a
#: single pixel, which no detector can learn.
TINY_FRACTION = 0.0005

#: Above this ratio between the commonest and rarest class, the rare class is
#: usually learned badly no matter how good the model is.
IMBALANCE_WARN = 10.0
IMBALANCE_SEVERE = 50.0

#: Fewer examples than this and a class is not really trained, it is memorised.
MIN_EXAMPLES_PER_CLASS = 50

#: Hard negatives are valuable, but past this share the model mostly learns to
#: predict nothing.
BACKGROUND_WARN = 0.5


@dataclass
class ClassStat:
    name: str
    objects: int = 0
    images: int = 0
    polygons: int = 0
    areas: list[float] = field(default_factory=list)

    def as_dict(self, total_objects: int) -> dict[str, Any]:
        return {
            "class": self.name,
            "objects": self.objects,
            "images": self.images,
            "polygons": self.polygons,
            "share": round(self.objects / total_objects, 4) if total_objects else 0.0,
            "median_area_px": int(_median(self.areas)) if self.areas else 0,
        }


@dataclass
class Finding:
    """One judgement about the dataset, with what to do about it."""

    severity: str  # "error" | "warning" | "note"
    issue: str
    detail: str
    fix: str

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "issue": self.issue,
            "detail": self.detail,
            "fix": self.fix,
        }


_SEVERITY_ORDER = {"error": 0, "warning": 1, "note": 2}


def analyse(dataset: Dataset) -> dict[str, Any]:
    """Measure a loaded dataset and return the full statistics payload."""
    per_class: dict[str, ClassStat] = {}
    image_sizes: Counter[tuple[int, int]] = Counter()
    objects_per_image: Counter[int] = Counter()
    relative_areas: list[float] = []
    aspect_ratios: list[float] = []
    size_buckets = Counter({"tiny": 0, "small": 0, "medium": 0, "large": 0})
    per_split: Counter[str] = Counter()

    total_objects = 0
    backgrounds = 0
    polygon_objects = 0
    out_of_bounds = 0
    degenerate = 0

    for item in dataset.items:
        annotation = item.annotation
        per_split[item.split or "(none)"] += 1
        if annotation.width > 0 and annotation.height > 0:
            image_sizes[(annotation.width, annotation.height)] += 1
        objects_per_image[len(annotation.shapes)] += 1
        if annotation.is_background:
            backgrounds += 1

        frame_area = float(max(1, annotation.width * annotation.height))
        for shape in annotation.shapes:
            total_objects += 1
            stat = per_class.setdefault(shape.label, ClassStat(shape.label))
            stat.objects += 1
            area = float(shape.bbox_area)
            stat.areas.append(area)
            if shape.has_polygon:
                stat.polygons += 1
                polygon_objects += 1

            relative_areas.append(area / frame_area)
            if shape.height > 0:
                aspect_ratios.append(shape.width / shape.height)
            if shape.width <= 0 or shape.height <= 0:
                degenerate += 1
            xmin, ymin, xmax, ymax = shape.bbox
            if (
                xmin < 0
                or ymin < 0
                or (annotation.width and xmax > annotation.width)
                or (annotation.height and ymax > annotation.height)
            ):
                out_of_bounds += 1

            if area / frame_area < TINY_FRACTION:
                size_buckets["tiny"] += 1
            elif area < SMALL_AREA:
                size_buckets["small"] += 1
            elif area < MEDIUM_AREA:
                size_buckets["medium"] += 1
            else:
                size_buckets["large"] += 1

        for label in annotation.label_set:
            per_class[label].images += 1

    ordered = sorted(per_class.values(), key=lambda s: (-s.objects, s.name))
    payload: dict[str, Any] = {
        "totals": {
            "images": len(dataset.items),
            "objects": total_objects,
            "classes": len(per_class),
            "backgrounds": backgrounds,
            "polygon_objects": polygon_objects,
            "images_with_polygons": sum(1 for i in dataset.items if i.annotation.has_polygons),
        },
        "per_class": [s.as_dict(total_objects) for s in ordered],
        "object_sizes": dict(size_buckets),
        "relative_area": _distribution(relative_areas),
        "aspect_ratio": _distribution(aspect_ratios),
        "objects_per_image": {
            "mean": round(total_objects / len(dataset.items), 2) if dataset.items else 0.0,
            "max": max(objects_per_image) if objects_per_image else 0,
            "histogram": {str(k): v for k, v in sorted(objects_per_image.items())[:20]},
        },
        "image_sizes": {
            "distinct": len(image_sizes),
            "most_common": [
                {"size": f"{w}x{h}", "images": n} for (w, h), n in image_sizes.most_common(5)
            ],
        },
        "splits": dict(sorted(per_split.items())),
        "geometry": {"out_of_bounds": out_of_bounds, "degenerate": degenerate},
        "source_format": dataset.source_format,
    }
    payload["findings"] = [f.as_dict() for f in judge(payload, dataset)]
    payload["balance"] = _balance(ordered)
    return payload


def judge(payload: dict[str, Any], dataset: Dataset) -> list[Finding]:
    """Turn the measurements into ranked, actionable findings."""
    findings: list[Finding] = []
    totals = payload["totals"]
    classes: list[dict[str, Any]] = payload["per_class"]

    if totals["objects"] == 0:
        findings.append(
            Finding(
                "error",
                "No objects at all",
                f"{totals['images']} image(s) loaded, none carrying a single annotation.",
                "Check the source format is right -- convert.detect will tell you what "
                "the folder actually is.",
            )
        )
        return findings

    # -- class balance
    if len(classes) >= 2:
        biggest, smallest = classes[0], classes[-1]
        ratio = biggest["objects"] / max(1, smallest["objects"])
        if ratio >= IMBALANCE_SEVERE:
            findings.append(
                Finding(
                    "error",
                    "Severe class imbalance",
                    f"{biggest['class']} has {biggest['objects']} objects, "
                    f"{smallest['class']} has {smallest['objects']} -- a {ratio:.0f}x gap.",
                    "Collect more of the rare class, or cap the common one with "
                    "dataset.sample --per-class. Class weighting in the trainer helps "
                    "but does not substitute for examples.",
                )
            )
        elif ratio >= IMBALANCE_WARN:
            findings.append(
                Finding(
                    "warning",
                    "Class imbalance",
                    f"{biggest['class']} outnumbers {smallest['class']} by {ratio:.0f}x.",
                    "Expect weaker recall on the rare class. Report per-class metrics "
                    "rather than a single mAP.",
                )
            )

    thin = [c for c in classes if c["objects"] < MIN_EXAMPLES_PER_CLASS]
    if thin:
        findings.append(
            Finding(
                "warning",
                "Classes with very few examples",
                ", ".join(f"{c['class']} ({c['objects']})" for c in thin[:6])
                + (f" and {len(thin) - 6} more" if len(thin) > 6 else ""),
                f"Under about {MIN_EXAMPLES_PER_CLASS} objects a class is memorised, "
                f"not learned. Collect more, or merge it into a neighbouring class "
                f"with labels.rename.",
            )
        )

    # -- case-variant classes: the bug that started this whole tool
    folded: dict[str, list[str]] = {}
    for entry in classes:
        folded.setdefault(entry["class"].strip().casefold(), []).append(entry["class"])
    collisions = {k: v for k, v in folded.items() if len(v) > 1}
    if collisions:
        findings.append(
            Finding(
                "error",
                "Classes differing only by capitalisation",
                "; ".join(f"{' / '.join(sorted(v))}" for v in collisions.values()),
                "These train as separate classes and split your data in half. "
                "Fix with labels.rename --fold-case.",
            )
        )

    # -- object size
    sizes = payload["object_sizes"]
    tiny_share = sizes["tiny"] / totals["objects"]
    if tiny_share > 0.2:
        findings.append(
            Finding(
                "warning",
                "Many objects are tiny relative to the frame",
                f"{sizes['tiny']} of {totals['objects']} objects ({tiny_share:.0%}) cover "
                f"under {TINY_FRACTION:.2%} of their image.",
                "At a 640px input these are a pixel or two across. Use image.tile to "
                "keep native resolution, or train at a higher input size.",
            )
        )

    # -- geometry
    geometry = payload["geometry"]
    if geometry["degenerate"]:
        findings.append(
            Finding(
                "error",
                "Zero-area boxes",
                f"{geometry['degenerate']} object(s) have no width or no height.",
                "These crash some trainers and are ignored by others. Delete them, or "
                "re-annotate those images.",
            )
        )
    if geometry["out_of_bounds"]:
        findings.append(
            Finding(
                "warning",
                "Boxes outside the image bounds",
                f"{geometry['out_of_bounds']} object(s) extend past the image edge.",
                "Usually harmless -- most trainers clip -- but it often means the "
                "annotations were made against a different image size. Worth a look.",
            )
        )

    # -- backgrounds
    background_share = totals["backgrounds"] / max(1, totals["images"])
    if background_share > BACKGROUND_WARN:
        findings.append(
            Finding(
                "warning",
                "Mostly background images",
                f"{totals['backgrounds']} of {totals['images']} images ({background_share:.0%}) "
                f"contain no objects.",
                "Some hard negatives are good (5-15% is typical). This many pushes the "
                "model towards predicting nothing.",
            )
        )
    elif totals["backgrounds"] == 0:
        findings.append(
            Finding(
                "note",
                "No hard negatives",
                "Every image contains at least one object.",
                "Adding some object-free images of the same scenes usually cuts false "
                "positives noticeably.",
            )
        )

    # -- image sizes
    if payload["image_sizes"]["distinct"] > 1:
        findings.append(
            Finding(
                "note",
                "Mixed image sizes",
                f"{payload['image_sizes']['distinct']} distinct resolutions.",
                "Fine for most trainers, which resize anyway. Run image.resize if you "
                "want a uniform set you can inspect.",
            )
        )

    # -- splits
    splits = {k: v for k, v in payload["splits"].items() if k != "(none)"}
    if not splits:
        findings.append(
            Finding(
                "note",
                "No train/val split",
                "The dataset is one undivided folder.",
                "Split it with dataset.split before training, and check for leakage "
                "afterwards with inspect.leakage.",
            )
        )

    if dataset.warnings:
        findings.append(
            Finding(
                "warning",
                "Problems while reading the dataset",
                "; ".join(dataset.warnings[:3])
                + (f" (+{len(dataset.warnings) - 3} more)" if len(dataset.warnings) > 3 else ""),
                "Files that fail to parse are silently absent from training. Fix or "
                "remove them.",
            )
        )

    findings.sort(key=lambda f: _SEVERITY_ORDER.get(f.severity, 3))
    return findings


def _balance(classes: Iterable[ClassStat]) -> dict[str, Any]:
    counts = [c.objects for c in classes]
    if not counts:
        return {"ratio": 0.0, "gini": 0.0}
    ratio = max(counts) / max(1, min(counts))
    return {"ratio": round(ratio, 2), "gini": round(_gini(counts), 3)}


def _gini(values: list[int]) -> float:
    """0 = perfectly balanced classes, approaching 1 = one class dominates."""
    if not values or sum(values) == 0:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    cumulative = sum((i + 1) * v for i, v in enumerate(ordered))
    return (2 * cumulative) / (n * sum(ordered)) - (n + 1) / n


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 5),
        "p25": round(_quantile(ordered, 0.25), 5),
        "median": round(_quantile(ordered, 0.5), 5),
        "p75": round(_quantile(ordered, 0.75), 5),
        "max": round(ordered[-1], 5),
        "mean": round(sum(ordered) / len(ordered), 5),
    }


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _median(values: list[float]) -> float:
    return _quantile(sorted(values), 0.5) if values else 0.0

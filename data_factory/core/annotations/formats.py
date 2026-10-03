"""One dataset, five formats, read and written through a single interface.

This is the module that turns a pile of readers and writers into a converter.
Because everything meets in the middle at :class:`Annotation`, supporting N
formats costs N readers and N writers -- not the N-squared converters a
pairwise design would need. Adding CreateML tomorrow means one file, and it
immediately converts to and from all four existing formats.

The layouts recognised are the ones exporters actually produce, not the ones
the specifications describe:

    voc       ``x.jpg`` + ``x.xml`` side by side, or Annotations/ + JPEGImages/
    yolo      ``images/`` + ``labels/`` + ``data.yaml``, or flat .txt sidecars
    coco      ``_annotations.coco.json``, or ``annotations/instances_*.json``
    labelme   ``x.jpg`` + ``x.json`` side by side
    csv       a single ``_annotations.csv``
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .. import fs, parallel
from ..pairs import PairIndex
from ..types import Action, Context
from . import coco as coco_fmt
from . import csvfmt, labelme, voc, yolo
from . import dota
from .model import Annotation, AnnotationError

FORMATS = ("voc", "yolo", "coco", "labelme", "csv", "dota")

#: Formats able to carry a polygon outline. Converting out of one of these into
#: a box-only format is lossy, and the converter says so rather than quietly
#: flattening the masks.
SEGMENTATION_FORMATS = frozenset({"voc", "yolo", "coco", "labelme"})


@dataclass(slots=True)
class Item:
    """One image and what is known about it."""

    annotation: Annotation
    image: Path | None = None
    split: str = ""

    @property
    def stem(self) -> str:
        source = self.image.stem if self.image else Path(self.annotation.filename).stem
        return source


@dataclass
class Dataset:
    """A whole dataset, loaded into the neutral model."""

    items: list[Item] = field(default_factory=list)
    source_format: str = ""
    classes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        if self.classes:
            return self.classes
        found: set[str] = set()
        for item in self.items:
            found |= item.annotation.label_set
        return sorted(found)

    def stats(self) -> dict[str, int]:
        objects = sum(len(i.annotation.shapes) for i in self.items)
        return {
            "images": len(self.items),
            "objects": objects,
            "classes": len(self.labels),
            "backgrounds": sum(1 for i in self.items if i.annotation.is_background),
            "with_polygons": sum(1 for i in self.items if i.annotation.has_polygons),
            "missing_images": sum(1 for i in self.items if i.image is None),
        }


# ------------------------------------------------------------- detection

def detect(folder: Path) -> str:
    """Guess the format of a folder by looking for each layout's marker.

    Ordered by how distinctive the evidence is: a ``data.yaml`` or a COCO json
    is conclusive, while loose ``.json`` files need their contents inspected to
    tell LabelMe from COCO.
    """
    folder = Path(folder)
    if yolo.find_data_yaml(folder):
        return "yolo"
    if (folder / yolo.LABEL_DIR).is_dir():
        # DOTA uses the same images/ + labels/ layout; only the lines differ.
        return "dota" if _looks_like_dota(folder) else "yolo"

    for path in coco_fmt.find_annotation_files(folder):
        if _looks_like_coco(path):
            return "coco"

    counts = {
        "voc": _count(folder, ".xml"),
        "labelme": _count(folder, ".json"),
        "yolo": _count(folder, ".txt"),
        "csv": _count(folder, ".csv"),
    }
    if counts["voc"]:
        return "voc"
    if counts["labelme"]:
        return "labelme"
    if counts["csv"]:
        return "csv"
    if counts["yolo"]:
        # Both are bare .txt files; DOTA names its class in the ninth field,
        # where YOLO has nothing but numbers.
        return "dota" if _looks_like_dota(folder) else "yolo"
    raise AnnotationError(
        f"cannot tell what format {folder} is -- found no .xml, .txt, .json or "
        f".csv labels. Pass the format explicitly."
    )


def _looks_like_dota(folder: Path) -> bool:
    for path in folder.rglob("*.txt"):
        if not path.is_file() or path.name == "classes.txt":
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if not parts or line.startswith(dota.HEADER_PREFIXES):
                continue
            if len(parts) < 9:
                return False
            try:
                float(parts[8])
            except ValueError:
                return True
            return False
    return False


def _count(folder: Path, suffix: str, limit: int = 1) -> int:
    seen = 0
    for path in folder.rglob(f"*{suffix}"):
        if path.is_file() and path.name != "data.yaml":
            seen += 1
            if seen >= limit:
                return seen
    return seen


def _looks_like_coco(path: Path) -> bool:
    """Peek at the head of the file rather than parsing a 200MB json."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    return '"categories"' in head or '"annotations"' in head


# ---------------------------------------------------------------- reading

def read_dataset(
    folder: Path,
    fmt: str = "auto",
    *,
    recursive: bool = True,
    ctx: Context | None = None,
) -> Dataset:
    """Load a folder into the neutral model, whatever format it is in."""
    folder = Path(folder)
    resolved = detect(folder) if fmt in ("", "auto") else fmt
    if resolved not in FORMATS:
        raise AnnotationError(f"unknown format {resolved!r}; known: {', '.join(FORMATS)}")

    reader = {
        "voc": _read_voc,
        "yolo": _read_yolo,
        "coco": _read_coco,
        "labelme": _read_labelme,
        "csv": _read_csv,
        "dota": _read_dota,
    }[resolved]
    dataset = reader(folder, recursive, ctx or Context())
    dataset.source_format = resolved
    return dataset


def _read_sidecars(
    folder: Path, recursive: bool, ext: str, parse
) -> Dataset:
    """Shared body for the two sidecar formats (VOC and LabelMe)."""
    dataset = Dataset()
    index = PairIndex.scan(folder, recursive=recursive, label_ext=ext)
    for pair in index:
        if pair.label is None:
            if pair.image is not None:
                # An unlabelled image is a real dataset member; carry it as a
                # background rather than dropping it on the floor.
                dataset.items.append(
                    Item(annotation=_empty_annotation(pair.image), image=pair.image)
                )
            continue
        try:
            annotation = parse(pair.label)
        except AnnotationError as exc:
            dataset.warnings.append(str(exc))
            continue
        if pair.image is not None and (annotation.width <= 0 or annotation.height <= 0):
            annotation.width, annotation.height = _size_of(pair.image, dataset)
        dataset.items.append(Item(annotation=annotation, image=pair.image, split=_split_of(pair.key)))
    return dataset


def _read_voc(folder: Path, recursive: bool, ctx: Context) -> Dataset:
    return _read_sidecars(folder, recursive, ".xml", voc.read)


def _read_labelme(folder: Path, recursive: bool, ctx: Context) -> Dataset:
    skipped: list[str] = []

    def parse(path: Path) -> Annotation:
        return labelme.read(path, skipped_out=skipped)

    dataset = _read_sidecars(folder, recursive, ".json", parse)
    if skipped:
        dataset.warnings.append(
            f"LabelMe shape types not representable as polygons were skipped: "
            f"{', '.join(sorted(set(skipped)))}"
        )
    return dataset


def _read_yolo(folder: Path, recursive: bool, ctx: Context) -> Dataset:
    dataset = Dataset()
    yaml_path = yolo.find_data_yaml(folder)
    if yaml_path is not None:
        classes, _ = yolo.read_data_yaml(yaml_path)
        dataset.classes = list(classes.names)
    else:
        # Without a class list the indices are meaningless, so say so loudly
        # and fall back to placeholder names rather than inventing a mapping.
        classes = yolo.ClassList([])
        dataset.warnings.append(
            "no data.yaml found -- classes will be named class_0, class_1, ... "
            "Point --classes at the right file to keep the real names."
        )

    # YOLO stores coordinates normalised, so every label needs its image's
    # dimensions before it can be read at all -- one header read per image.
    # Serial, that is the dominant cost of loading a large YOLO dataset.
    images = list(fs.iter_images(folder, recursive=recursive))
    sizes = dict(
        parallel.collect(
            images,
            _image_size,
            ctx=ctx,
            label=lambda image: image.name,
            on_error=lambda image, exc: dataset.warnings.append(
                f"{image.name}: cannot read image size ({exc})"
            ),
        )
    )

    for image in images:
        label_path = yolo.label_path_for(image, folder)
        width, height = sizes.get(image, (0, 0))
        if width <= 0:
            continue
        if not label_path.is_file():
            dataset.items.append(
                Item(annotation=Annotation(filename=image.name, width=width, height=height),
                     image=image)
            )
            continue
        try:
            annotation = yolo.read(
                label_path, classes, width, height, filename=image.name
            )
        except AnnotationError as exc:
            dataset.warnings.append(str(exc))
            continue
        dataset.items.append(
            Item(
                annotation=annotation,
                image=image,
                split=_split_of(str(fs.relative_to_safe(image, folder)).replace("\\", "/")),
            )
        )
    if not dataset.classes:
        dataset.classes = list(classes.names)
    return dataset


def _read_dota(folder: Path, recursive: bool, ctx: Context) -> Dataset:
    """Like YOLO -- one .txt beside or under labels/ per image -- but the class
    is named in the file, so there is no list to find."""
    dataset = Dataset()
    images = list(fs.iter_images(folder, recursive=recursive))
    sizes = dict(
        parallel.collect(
            images,
            _image_size,
            ctx=ctx,
            label=lambda image: image.name,
            on_error=lambda image, exc: dataset.warnings.append(
                f"{image.name}: cannot read image size ({exc})"
            ),
        )
    )
    for image in images:
        label_path = yolo.label_path_for(image, folder)
        width, height = sizes.get(image, (0, 0))
        if width <= 0:
            continue
        if not label_path.is_file():
            dataset.items.append(
                Item(annotation=Annotation(filename=image.name, width=width, height=height),
                     image=image)
            )
            continue
        try:
            annotation = dota.read(label_path, width, height, filename=image.name)
        except AnnotationError as exc:
            dataset.warnings.append(str(exc))
            continue
        dataset.items.append(
            Item(
                annotation=annotation,
                image=image,
                split=_split_of(str(fs.relative_to_safe(image, folder)).replace("\\", "/")),
            )
        )
    return dataset


def _read_coco(folder: Path, recursive: bool, ctx: Context) -> Dataset:
    dataset = Dataset()
    files = [p for p in coco_fmt.find_annotation_files(folder) if _looks_like_coco(p)]
    if not files:
        raise AnnotationError(f"no COCO annotation json found under {folder}")

    images_by_name = _image_index(folder)
    unsupported: list[str] = []
    for path in files:
        document = coco_fmt.load(path)
        split = _split_of(path.stem)
        for annotation in coco_fmt.to_annotations(document, unsupported_out=unsupported):
            image = _pick_image(images_by_name, annotation.filename, split)
            dataset.items.append(Item(annotation=annotation, image=image, split=split))
        for category in document.get("categories", []):
            name = str(category.get("name", ""))
            if name and name not in dataset.classes:
                dataset.classes.append(name)
    dataset.warnings.extend(unsupported)
    return dataset


def _read_csv(folder: Path, recursive: bool, ctx: Context) -> Dataset:
    dataset = Dataset()
    candidates = sorted(folder.rglob("*.csv")) if recursive else sorted(folder.glob("*.csv"))
    if not candidates:
        raise AnnotationError(f"no .csv annotation file found under {folder}")

    images_by_name = _image_index(folder)
    for path in candidates:
        for annotation in csvfmt.read(path):
            image = _pick_image(images_by_name, annotation.filename)
            if image is not None and annotation.width <= 0:
                annotation.width, annotation.height = _size_of(image, dataset)
            dataset.items.append(Item(annotation=annotation, image=image))
    return dataset


def _image_index(folder: Path) -> dict[str, list[Path]]:
    """Every image under a folder, findable by both name and stem.

    COCO and CSV reference images by name only, so the file could be anywhere
    in the tree -- and in a split dataset ``a.jpg`` legitimately exists in both
    ``train/`` and ``val/``. Every candidate is kept so the caller can pick the
    one in the right split rather than whichever sorted first.
    """
    index: dict[str, list[Path]] = {}
    for image in fs.iter_images(folder, recursive=True):
        index.setdefault(image.name, []).append(image)
        if image.stem != image.name:
            index.setdefault(image.stem, []).append(image)
    return index


def _pick_image(
    index: dict[str, list[Path]], filename: str, split: str = ""
) -> Path | None:
    """Resolve a referenced filename to an actual file, preferring its split.

    Without the split preference, a COCO dataset whose train and val folders
    share a filename would attach val's annotations to train's image -- wrong
    in a way nothing downstream could detect.
    """
    candidates = index.get(filename) or index.get(Path(filename).name) or index.get(
        Path(filename).stem
    )
    if not candidates:
        return None
    if split:
        for candidate in candidates:
            if split in {part.lower() for part in candidate.parts}:
                return candidate
    return candidates[0]


def _empty_annotation(image: Path) -> Annotation:
    try:
        width, height = _image_size(image)
    except Exception:
        width, height = 0, 0
    return Annotation(filename=image.name, width=width, height=height, shapes=[])


def _size_of(image: Path, dataset: Dataset) -> tuple[int, int]:
    try:
        return _image_size(image)
    except Exception as exc:
        dataset.warnings.append(f"{image.name}: cannot read image size ({exc})")
        return 0, 0


def _image_size(path: Path) -> tuple[int, int]:
    from ..imaging import image_size

    return image_size(path)


def _split_of(key: str) -> str:
    """Recover train/val/test from a path or filename, when it is there."""
    lowered = key.lower().replace("\\", "/")
    for name in ("train", "valid", "val", "test"):
        if f"/{name}/" in f"/{lowered}/" or lowered.endswith(f"_{name}"):
            return "val" if name == "valid" else name
    return ""


# ---------------------------------------------------------------- writing

def write_dataset(
    dataset: Dataset,
    output: Path,
    fmt: str,
    *,
    ctx: Context,
    copy_images: bool = True,
    segmentation: bool = True,
    on_action=None,
) -> dict[str, object]:
    """Serialise a dataset into ``fmt`` under ``output``.

    Image files are copied through the usual :func:`fs.perform` choke point, so
    ``--dry-run`` reports the entire conversion without writing a byte. Label
    files are written directly, but only ever inside ``output``, and never at
    all in dry-run mode.
    """
    if fmt not in FORMATS:
        raise AnnotationError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")

    output = Path(output)
    report: dict[str, object] = {"format": fmt, "written": 0, "images": 0, "warnings": []}
    warnings: list[str] = report["warnings"]  # type: ignore[assignment]

    if segmentation and fmt not in SEGMENTATION_FORMATS:
        polygons = sum(1 for i in dataset.items if i.annotation.has_polygons)
        if polygons:
            warnings.append(
                f"{fmt} cannot store outlines: {polygons} image(s) keep their boxes "
                f"but lose their polygons"
            )

    writer = {
        "voc": _write_voc,
        "yolo": _write_yolo,
        "coco": _write_coco,
        "labelme": _write_labelme,
        "csv": _write_csv,
        "dota": _write_dota,
    }[fmt]
    writer(dataset, output, ctx, report, copy_images, segmentation, on_action)
    return report


def _copy_image(
    item: Item, destination: Path, ctx: Context, report: dict, on_action
) -> None:
    if item.image is None or not destination:
        return
    action = fs.FileAction(Action.COPY, item.image, destination, "convert")
    fs.perform(action, ctx)
    report["images"] = int(report["images"]) + 1
    if on_action is not None:
        on_action(action)


def _split_dir(base: Path, item: Item) -> Path:
    return base / item.split if item.split else base


def _write_voc(dataset, output, ctx, report, copy_images, segmentation, on_action) -> None:
    for item in dataset.items:
        folder = _split_dir(output, item)
        fs.ensure_dir(folder, ctx)
        stem = item.stem
        if copy_images and item.image is not None:
            _copy_image(item, folder / f"{stem}{item.image.suffix}", ctx, report, on_action)
        if not ctx.dry_run:
            voc.write(item.annotation, folder / f"{stem}.xml")
        report["written"] = int(report["written"]) + 1


def _write_labelme(dataset, output, ctx, report, copy_images, segmentation, on_action) -> None:
    for item in dataset.items:
        folder = _split_dir(output, item)
        fs.ensure_dir(folder, ctx)
        stem = item.stem
        if copy_images and item.image is not None:
            _copy_image(item, folder / f"{stem}{item.image.suffix}", ctx, report, on_action)
        if not ctx.dry_run:
            labelme.write(item.annotation, folder / f"{stem}.json")
        report["written"] = int(report["written"]) + 1


def _write_yolo(dataset, output, ctx, report, copy_images, segmentation, on_action) -> None:
    # Seed the class list from the dataset so indices follow the declared order
    # rather than whichever image happened to be processed first.
    classes = yolo.ClassList(list(dataset.labels))
    splits: dict[str, str] = {}

    for item in dataset.items:
        split = item.split or "train"
        splits.setdefault(split, f"{yolo.IMAGE_DIR}/{split}")
        image_dir = output / yolo.IMAGE_DIR / split
        label_dir = output / yolo.LABEL_DIR / split
        fs.ensure_dir(image_dir, ctx)
        fs.ensure_dir(label_dir, ctx)
        stem = item.stem
        if copy_images and item.image is not None:
            _copy_image(item, image_dir / f"{stem}{item.image.suffix}", ctx, report, on_action)
        if not ctx.dry_run:
            yolo.write(item.annotation, label_dir / f"{stem}.txt", classes, segmentation=segmentation)
        report["written"] = int(report["written"]) + 1

    if not ctx.dry_run:
        yolo.write_data_yaml(output / yolo.DATA_YAML, classes, splits=splits, root=".")
    report["classes"] = list(classes.names)


def _write_dota(dataset, output, ctx, report, copy_images, segmentation, on_action) -> None:
    for item in dataset.items:
        split = item.split or "train"
        image_dir = output / yolo.IMAGE_DIR / split
        label_dir = output / yolo.LABEL_DIR / split
        fs.ensure_dir(image_dir, ctx)
        fs.ensure_dir(label_dir, ctx)
        stem = item.stem
        if copy_images and item.image is not None:
            _copy_image(item, image_dir / f"{stem}{item.image.suffix}", ctx, report, on_action)
        if not ctx.dry_run:
            dota.write(item.annotation, label_dir / f"{stem}.txt")
        report["written"] = int(report["written"]) + 1
    report["classes"] = list(dataset.labels)


def _write_coco(dataset, output, ctx, report, copy_images, segmentation, on_action) -> None:
    by_split: dict[str, list[Item]] = {}
    for item in dataset.items:
        by_split.setdefault(item.split or "train", []).append(item)

    for split, items in sorted(by_split.items()):
        builder = coco_fmt.CocoBuilder(description=f"Data Factory export ({split})")
        image_dir = output / split
        fs.ensure_dir(image_dir, ctx)
        for item in items:
            name = f"{item.stem}{item.image.suffix if item.image else '.jpg'}"
            builder.add(item.annotation, file_name=name)
            if copy_images and item.image is not None:
                _copy_image(item, image_dir / name, ctx, report, on_action)
            report["written"] = int(report["written"]) + 1
        if not ctx.dry_run:
            fs.ensure_dir(output / "annotations", ctx)
            builder.write(output / "annotations" / f"instances_{split}.json")
        merges = builder.categories.merges()
        if merges:
            report["warnings"].append(  # type: ignore[union-attr]
                "case-variant classes merged: "
                + "; ".join(f"{k} <- {', '.join(v)}" for k, v in merges.items())
            )


def _write_csv(dataset, output, ctx, report, copy_images, segmentation, on_action) -> None:
    fs.ensure_dir(output, ctx)
    image_dir = output / "images"
    annotations: list[Annotation] = []
    for item in dataset.items:
        annotations.append(item.annotation)
        if copy_images and item.image is not None:
            fs.ensure_dir(image_dir, ctx)
            _copy_image(
                item, image_dir / f"{item.stem}{item.image.suffix}", ctx, report, on_action
            )
        report["written"] = int(report["written"]) + 1
    if not ctx.dry_run:
        csvfmt.write(annotations, output / "_annotations.csv")


def label_extension(fmt: str) -> str:
    """The sidecar extension a format uses, or empty for the collected ones."""
    return {"voc": ".xml", "labelme": ".json", "yolo": ".txt", "dota": ".txt"}.get(fmt, "")


def iter_formats() -> Iterable[str]:
    return FORMATS

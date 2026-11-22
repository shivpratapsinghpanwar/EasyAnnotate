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

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import fs, parallel
from ..pairs import PairIndex
from ..types import Action, Context
from . import coco as coco_fmt
from . import csvfmt, labelme, voc, yolo
from . import dota
from .model import Annotation, AnnotationError

FORMATS = ("voc", "yolo", "coco", "labelme", "csv", "dota")

#: Formats able to carry a polygon outline. Writing an outline to any other
#: flattens it to its box, and the converter says so, with a count, rather
#: than quietly dropping the masks. CSV is here: it has a polygon column.
SEGMENTATION_FORMATS = frozenset({"voc", "yolo", "coco", "labelme", "csv"})

#: Formats with somewhere to put a detector's score.
SCORE_FORMATS = frozenset({"voc", "coco"})

#: What a format does with a kind of shape it has no form of its own for.
#: A kind that is absent is written as itself. This table describes the
#: writers; tests/test_converter_fixes.py holds every row to what they do.
_NO_FORM: dict[str, dict[str, str]] = {
    "voc": {"line": "boxed", "point": "flattened"},
    "yolo": {"line": "boxed", "point": "left out"},
    "coco": {"line": "boxed"},
    "dota": {"line": "left out", "point": "left out"},
}


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
    return _sniff(Path(folder))[0]


def _sniff(folder: Path) -> tuple[str, dict[Path, Any]]:
    """:func:`detect`, plus the COCO document it had to parse to decide.

    Telling COCO from LabelMe means parsing the file, and an instances.json
    can be hundreds of megabytes. Handing the parsed document on lets
    :func:`read_dataset` read it once rather than twice.
    """
    if yolo.find_data_yaml(folder):
        return "yolo", {}
    if (folder / yolo.LABEL_DIR).is_dir():
        # DOTA uses the same images/ + labels/ layout; only the lines differ.
        return ("dota" if _looks_like_dota(folder) else "yolo"), {}

    for path in coco_fmt.find_annotation_files(folder):
        document = _coco_document(path)
        if document is not None:
            return "coco", {path: document}

    return _sniff_by_extension(folder), {}


def _sniff_by_extension(folder: Path) -> str:
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


def _in_order(paths: Iterable[Path], root: Path) -> list[Path]:
    """Files in the one order they are read in, on every platform.

    By path under the dataset folder, compared one folder name at a time, by
    Unicode code point -- so ``B.jpg`` comes before ``a.jpg``, as it always
    has on Linux and macOS. Sorting the ``Path`` objects themselves is not
    that: on Windows they compare without regard to case, so the same folder
    was read in a different order there, and a COCO or CSV made from it
    numbered and listed its images differently.
    """
    return sorted(paths, key=lambda path: fs.relative_to_safe(path, root).parts)


def _looks_like_dota(folder: Path) -> bool:
    for path in _in_order(folder.rglob("*.txt"), folder):
        if not path.is_file() or path.name == "classes.txt":
            continue
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
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


def _coco_document(path: Path) -> Any | None:
    """The parsed file if it is a COCO dataset, otherwise ``None``.

    This used to peek at the first 4096 characters for the word "annotations"
    or "categories", to avoid parsing a large file. A COCO file may list its
    images first, and forty of them fill that window -- so this tool's own
    export of 71 images was taken for LabelMe and read back, without an error,
    as one empty image. A wrong answer that looks like success costs more than
    a second or two of parsing, so the file is parsed and its keys decide.
    """
    try:
        document = coco_fmt.load(path)
    except AnnotationError:
        return None
    return document if coco_fmt.is_dataset(document) else None


# ---------------------------------------------------------------- reading

def read_dataset(
    folder: Path,
    fmt: str = "auto",
    *,
    recursive: bool = True,
    ctx: Context | None = None,
    classes: list[str] | None = None,
) -> Dataset:
    """Load a folder into the neutral model, whatever format it is in.

    ``classes`` is the caller's class names in number order. It is used for
    one thing: naming the class numbers of a YOLO folder that has no
    ``data.yaml`` or ``classes.txt`` of its own.
    """
    folder = Path(folder)
    parsed: dict[Path, Any] = {}
    if fmt in ("", "auto"):
        resolved, parsed = _sniff(folder)
    else:
        resolved = fmt
    if resolved not in FORMATS:
        raise AnnotationError(f"unknown format {resolved!r}; known: {', '.join(FORMATS)}")

    context = ctx or Context()
    if resolved == "coco":
        dataset = _read_coco(folder, recursive, context, parsed)
    elif resolved == "yolo":
        dataset = _read_yolo(folder, recursive, context, classes)
    else:
        reader = {
            "voc": _read_voc,
            "labelme": _read_labelme,
            "csv": _read_csv,
            "dota": _read_dota,
        }[resolved]
        dataset = reader(folder, recursive, context)
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
                    Item(
                        annotation=_empty_annotation(pair.image),
                        image=pair.image,
                        split=_split_of(pair.key),
                    )
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


def _report_dropped(dataset: Dataset, dropped: list[str], what: str) -> None:
    """One counted line per reason a reader could not keep an object."""
    for reason, count in sorted(Counter(dropped).items()):
        dataset.warnings.append(f"{count} {what} have {reason} and were skipped")


def _read_voc(folder: Path, recursive: bool, ctx: Context) -> Dataset:
    dropped: list[str] = []

    def parse(path: Path) -> Annotation:
        return voc.read(path, dropped_out=dropped)

    dataset = _read_sidecars(folder, recursive, ".xml", parse)
    _report_dropped(dataset, dropped, "VOC object(s)")
    return dataset


def _read_labelme(folder: Path, recursive: bool, ctx: Context) -> Dataset:
    skipped: list[str] = []
    dropped: list[str] = []

    def parse(path: Path) -> Annotation:
        return labelme.read(path, skipped_out=skipped, dropped_out=dropped)

    dataset = _read_sidecars(folder, recursive, ".json", parse)
    _report_dropped(dataset, dropped, "LabelMe shape(s)")
    if skipped:
        dataset.warnings.append(
            f"LabelMe shape types not representable as polygons were skipped: "
            f"{', '.join(sorted(set(skipped)))}"
        )
    return dataset


def _read_yolo(
    folder: Path, recursive: bool, ctx: Context, class_names: list[str] | None = None
) -> Dataset:
    dataset = Dataset()
    yaml_path = yolo.find_data_yaml(folder)
    txt_path = None if yaml_path is not None else yolo.find_classes_txt(folder)
    if yaml_path is not None:
        classes, _ = yolo.read_data_yaml(yaml_path)
        named_by = yaml_path.name
    elif txt_path is not None:
        classes = yolo.read_classes_txt(txt_path)
        named_by = txt_path.name
    elif class_names:
        # The caller knows what the numbers mean and the folder does not say:
        # the names are taken in number order, and the run says that it did.
        classes = yolo.ClassList(list(class_names))
        named_by = "'classes'"
        dataset.warnings.append(
            "no data.yaml or classes.txt found -- class numbers were named from "
            "'classes' in order: "
            + ", ".join(f"{index} = {name}" for index, name in enumerate(classes.names))
        )
    else:
        # Without a class list the numbers are all there is, so say so loudly
        # and name each class after its number rather than inventing a mapping.
        classes = yolo.ClassList([])
        named_by = ""
        dataset.warnings.append(
            "no data.yaml or classes.txt found -- classes are named class_0, "
            "class_1, ... after their numbers. To give them their real names, "
            "pass 'classes' with the names in number order (the first is class "
            "0), or add a data.yaml or classes.txt."
        )
    names = list(classes.names)
    unnamed: list[int] = []

    # YOLO stores coordinates normalised, so every label needs its image's
    # dimensions before it can be read at all -- one header read per image.
    # Serial, that is the dominant cost of loading a large YOLO dataset.
    images = _in_order(fs.iter_images(folder, recursive=recursive), folder)
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
        split = _split_of(fs.relative_to_safe(image, folder).as_posix())
        if not label_path.is_file():
            dataset.items.append(
                Item(annotation=Annotation(filename=image.name, width=width, height=height),
                     image=image, split=split)
            )
            continue
        try:
            annotation = yolo.read(
                label_path, classes, width, height, filename=image.name, unnamed_out=unnamed
            )
        except AnnotationError as exc:
            dataset.warnings.append(str(exc))
            continue
        dataset.items.append(Item(annotation=annotation, image=image, split=split))

    # A class number nothing names keeps its place: the list is filled up to
    # it with class_N. Leaving the list short let the writer hand that class
    # the next free number (5 became 2), and sorting the placeholders as text
    # put class_10 before class_2 -- either way a YOLO folder converted to
    # YOLO came out renumbered.
    top = max(unnamed, default=-1)
    if top >= len(names):
        if named_by:
            numbers = ", ".join(str(n) for n in sorted(set(unnamed)))
            dataset.warnings.append(
                f"{len(unnamed)} object(s) use class number(s) {named_by} does not "
                f"name ({numbers}); they are named class_N and keep their numbers"
            )
        names.extend(f"class_{index}" for index in range(len(names), top + 1))
    dataset.classes = names
    return dataset


def _read_dota(folder: Path, recursive: bool, ctx: Context) -> Dataset:
    """Like YOLO -- one .txt beside or under labels/ per image -- but the class
    is named in the file, so there is no list to find."""
    dataset = Dataset()
    images = _in_order(fs.iter_images(folder, recursive=recursive), folder)
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
        split = _split_of(fs.relative_to_safe(image, folder).as_posix())
        if not label_path.is_file():
            dataset.items.append(
                Item(annotation=Annotation(filename=image.name, width=width, height=height),
                     image=image, split=split)
            )
            continue
        try:
            annotation = dota.read(label_path, width, height, filename=image.name)
        except AnnotationError as exc:
            dataset.warnings.append(str(exc))
            continue
        dataset.items.append(Item(annotation=annotation, image=image, split=split))
    return dataset


def _read_coco(
    folder: Path, recursive: bool, ctx: Context, parsed: dict[Path, Any] | None = None
) -> Dataset:
    """Read every COCO dataset file under the folder.

    ``parsed`` holds documents the format sniff already loaded. Files are
    handled one at a time so two large split files are never both in memory.
    """
    dataset = Dataset()
    parsed = dict(parsed or {})
    images_by_name: dict[str, list[Path]] | None = None
    unsupported: list[str] = []
    problems: list[str] = []
    read = 0

    for path in coco_fmt.find_annotation_files(folder):
        document = parsed.pop(path, None)
        if document is None:
            try:
                document = coco_fmt.load(path)
            except AnnotationError as exc:
                # Named, and kept: if nothing else here is COCO this is the
                # error; if something is, it is a warning beside the result.
                problems.append(str(exc))
                continue
        if not coco_fmt.is_dataset(document):
            if coco_fmt.is_detection_results(document):
                problems.append(
                    f"{path.name}: a bare list of detection results, not a dataset "
                    f"(import it as predictions instead)"
                )
            continue

        read += 1
        if images_by_name is None:
            images_by_name = _image_index(folder)
        split = _coco_split(path.stem)
        try:
            annotations = coco_fmt.to_annotations(document, unsupported_out=unsupported)
            names = [
                str(category.get("name") or "")
                for category in document.get("categories") or []
                if isinstance(category, dict)
            ]
        except AnnotationError as exc:
            raise AnnotationError(f"{path.name}: {exc}") from exc
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            # Valid JSON in a shape nobody anticipated. The specific cases are
            # handled where they arise; this is what keeps the rest from
            # reaching the user as a traceback with no file name in it.
            raise AnnotationError(
                f"{path.name}: not a readable COCO document ({type(exc).__name__}: {exc})"
            ) from exc
        for annotation in annotations:
            image = _pick_image(images_by_name, annotation.filename, split)
            dataset.items.append(Item(annotation=annotation, image=image, split=split))
        for name in names:
            if name and name not in dataset.classes:
                dataset.classes.append(name)

    if not read:
        detail = f": {'; '.join(problems[:3])}" if problems else ""
        raise AnnotationError(f"no COCO annotation json found under {folder}{detail}")
    dataset.warnings.extend(problems)
    dataset.warnings.extend(unsupported)
    return dataset


def _read_csv(folder: Path, recursive: bool, ctx: Context) -> Dataset:
    dataset = Dataset()
    candidates = _in_order(folder.rglob("*.csv") if recursive else folder.glob("*.csv"), folder)
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
    for image in _in_order(fs.iter_images(folder, recursive=True), folder):
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


_SPLIT_NAMES = ("train", "valid", "val", "test")


def _split_of(relative: str) -> str:
    """The split a file belongs to: a folder named train, val, valid or test
    on its path inside the dataset.

    ``relative`` is the file's path under the dataset folder; its last
    component is the file itself and is never looked at. Matching the name
    too sent ``photo_test.xml`` to test/ and ``train.xml`` to train/.
    """
    folders = relative.lower().replace("\\", "/").split("/")[:-1]
    for name in _SPLIT_NAMES:
        if name in folders:
            return "val" if name == "valid" else name
    return ""


def _coco_split(stem: str) -> str:
    """The split a COCO annotation file is for. COCO is the one layout that
    names the split in a file name: ``instances_val.json``."""
    lowered = stem.lower()
    for name in _SPLIT_NAMES:
        if lowered.endswith(f"_{name}"):
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

    warnings.extend(_write_losses(dataset, fmt, segmentation))

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


def _write_losses(dataset: Dataset, fmt: str, segmentation: bool) -> list[str]:
    """What this conversion cannot keep, counted and in plain words.

    Worked out from the dataset rather than reported back by the writers, so
    a dry run -- which never calls a writer -- says exactly what the real run
    will. Each line is one kind of loss with one number, so the handful of
    warnings a caller shows is the whole story.
    """
    tally: Counter[str] = Counter()
    renamed: dict[str, str] = {}
    no_form = _NO_FORM.get(fmt, {})
    for item in dataset.items:
        annotation = item.annotation
        for shape in annotation.shapes:
            kind = shape.shape_kind
            if kind in no_form:
                tally[f"{kind} {no_form[kind]}"] += 1
                if no_form[kind] == "left out":
                    continue
            elif fmt == "yolo":
                tally[
                    "yolo-"
                    + yolo.fate(shape, annotation.width, annotation.height, segmentation=segmentation)
                ] += 1
            elif fmt == "dota":
                # A rotated box is four corners, which is what DOTA stores.
                if segmentation and shape.has_polygon and kind != "obb":
                    tally["outline boxed"] += 1
            if shape.confidence is not None and fmt not in SCORE_FORMATS:
                tally["score"] += 1
            if fmt == "dota" and dota.class_name(shape.label) != shape.label:
                renamed[shape.label] = dota.class_name(shape.label)

    texts = {
        "outline boxed": "outline(s) were written as their bounding boxes: dota stores "
        "four-corner boxes, not outlines",
        "line boxed": f"line(s) were written as their bounding boxes: {fmt} has no line shape",
        "line left out": f"line(s) were not written: {fmt} has no line shape",
        "point flattened": f"point(s) were written as boxes with no area: {fmt} has no point shape",
        "point left out": f"point(s) were not written: {fmt} has no point shape",
        "yolo-clipped": "box(es) reach outside their image and were cut to the part inside it",
        "yolo-dropped": "box(es) have no area inside their image and were not written",
        "yolo-outline": "outline(s) have points outside their image, which were moved onto its edge",
        "score": f"detector score(s) were not written: {fmt} has no place for them",
    }
    notes = [f"{tally[key]} {text}" for key, text in texts.items() if tally[key]]
    if renamed:
        pairs = ", ".join(f"{old!r} -> {new!r}" for old, new in sorted(renamed.items()))
        notes.append(
            f"{len(renamed)} class name(s) hold spaces, which DOTA uses to separate "
            f"fields, and were written with hyphens: {pairs}"
        )
    return notes


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


def _class_order(dataset: Dataset) -> list[str]:
    """Every class in the dataset, in the one order ids are handed out in:
    the declared classes, then any other label found, sorted by name."""
    order = list(dataset.labels)
    found = {label for item in dataset.items for label in item.annotation.label_set}
    return order + sorted(found - set(order))


def _write_coco(dataset, output, ctx, report, copy_images, segmentation, on_action) -> None:
    by_split: dict[str, list[Item]] = {}
    for item in dataset.items:
        by_split.setdefault(item.split or "train", []).append(item)

    # One table for the whole dataset, filled before any file is written. A
    # table per split numbered classes by first appearance in that split, so
    # id 1 could be one class in instances_train.json and another in
    # instances_val.json -- and a model trained on one scored against the
    # other without a single error.
    categories = coco_fmt.CategoryTable()
    for label in _class_order(dataset):
        categories.id_for(label)

    for split, items in sorted(by_split.items()):
        builder = coco_fmt.CocoBuilder(
            description=f"Data Factory export ({split})", categories=categories
        )
        image_dir = output / split
        fs.ensure_dir(image_dir, ctx)
        for item in items:
            if item.image is not None:
                name = f"{item.stem}{item.image.suffix}"
            else:
                # No file to ask, so the label's own name stands. Forcing
                # ".jpg" here renamed a.png to a file that does not exist.
                name = item.annotation.filename or f"{item.stem}.jpg"
            builder.add(item.annotation, file_name=name)
            if copy_images and item.image is not None:
                _copy_image(item, image_dir / name, ctx, report, on_action)
            report["written"] = int(report["written"]) + 1
        if not ctx.dry_run:
            fs.ensure_dir(output / "annotations", ctx)
            builder.write(output / "annotations" / f"instances_{split}.json")
    merges = categories.merges()
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

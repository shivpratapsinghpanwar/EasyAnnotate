"""Streaming dataset writer.

:func:`data_factory.core.annotations.formats.write_dataset` takes a whole
dataset in memory, which is fine for conversion -- the source already had to be
loaded. It is the wrong shape for the operations that *generate* images:
augmenting a 4,000-image set five ways would mean holding 20,000 decoded frames
before a single byte reached the disk.

This writer takes one item at a time and flushes as it goes. The formats that
genuinely cannot stream -- COCO and CSV are single files -- accumulate only the
annotations, never the pixels, so the memory cost is a few hundred bytes per
image rather than a few megabytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import fs, imaging
from .annotations import coco as coco_fmt
from .annotations import csvfmt, formats, labelme, voc, yolo
from .annotations.model import Annotation
from .types import Action, Context, FileAction


class DatasetWriter:
    """Writes annotations and images incrementally in a chosen format.

    ``close()`` finalises whatever the format needs written once -- a
    ``data.yaml``, an ``instances_*.json``, a ``_annotations.csv`` -- and
    returns a small report.
    """

    def __init__(
        self,
        output: Path,
        fmt: str,
        ctx: Context,
        *,
        classes: list[str] | None = None,
        segmentation: bool = True,
        quality: int = 95,
    ) -> None:
        if fmt not in formats.FORMATS:
            raise ValueError(f"unknown format {fmt!r}; known: {', '.join(formats.FORMATS)}")
        self.output = Path(output)
        self.fmt = fmt
        self.ctx = ctx
        self.segmentation = segmentation
        self.quality = quality
        # Seeding the class list up front keeps YOLO indices stable and
        # independent of the order images happen to be processed in.
        self.classes = yolo.ClassList(list(classes or []))
        self.images_written = 0
        self.labels_written = 0
        self.warnings: list[str] = []
        self.actions: list[FileAction] = []
        self._splits: dict[str, str] = {}
        self._collected: dict[str, list[tuple[Annotation, str]]] = {}

    # -- one item ----------------------------------------------------------
    def add(
        self,
        annotation: Annotation,
        *,
        stem: str,
        image: np.ndarray | None = None,
        encoded: bytes | None = None,
        source: Path | None = None,
        split: str = "",
        suffix: str = ".jpg",
    ) -> None:
        """Write one image and its label.

        Pass ``image`` for a freshly computed frame, ``encoded`` when the
        caller already compressed it on a worker thread, or ``source`` alone to
        copy an untouched file straight through -- re-encoding an image nothing
        changed would cost quality for nothing.
        """
        suffix = suffix or ".jpg"
        image_dir, label_dir = self._directories(split)
        name = f"{stem}{suffix}"

        if encoded is not None or image is not None:
            if not self.ctx.dry_run:
                fs.ensure_dir(image_dir, self.ctx)
                if encoded is not None:
                    imaging.write_bytes(encoded, image_dir / name)
                else:
                    assert image is not None
                    imaging.save(image, image_dir / name, quality=self.quality)
            self.actions.append(
                FileAction(Action.WRITE, source or (image_dir / name), image_dir / name, "generated")
            )
            self.images_written += 1
        elif source is not None:
            fs.ensure_dir(image_dir, self.ctx)
            action = FileAction(Action.COPY, source, image_dir / name, "unchanged")
            fs.perform(action, self.ctx)
            self.actions.append(action)
            self.images_written += 1

        stored = Annotation(
            filename=name,
            width=annotation.width,
            height=annotation.height,
            depth=annotation.depth,
            shapes=annotation.shapes,
            folder=annotation.folder,
            source_path=annotation.source_path,
        )

        if self.fmt in ("coco", "csv"):
            self._collected.setdefault(split or "train", []).append((stored, name))
        elif not self.ctx.dry_run:
            fs.ensure_dir(label_dir, self.ctx)
            self._write_sidecar(stored, label_dir, stem)
        self.labels_written += 1
        if split:
            self._splits.setdefault(split, f"{yolo.IMAGE_DIR}/{split}")

    def _write_sidecar(self, annotation: Annotation, label_dir: Path, stem: str) -> None:
        if self.fmt == "voc":
            voc.write(annotation, label_dir / f"{stem}.xml")
        elif self.fmt == "labelme":
            labelme.write(annotation, label_dir / f"{stem}.json")
        elif self.fmt == "yolo":
            yolo.write(
                annotation, label_dir / f"{stem}.txt", self.classes, segmentation=self.segmentation
            )

    def _directories(self, split: str) -> tuple[Path, Path]:
        """Where images and labels go, which is the one thing formats disagree on."""
        if self.fmt == "yolo":
            leaf = split or "train"
            return (
                self.output / yolo.IMAGE_DIR / leaf,
                self.output / yolo.LABEL_DIR / leaf,
            )
        if self.fmt == "coco":
            base = self.output / (split or "train")
            return base, base
        if self.fmt == "csv":
            base = self.output / "images"
            return base, base
        base = self.output / split if split else self.output
        return base, base

    # -- finish ------------------------------------------------------------
    def close(self) -> dict[str, Any]:
        if not self.ctx.dry_run:
            if self.fmt == "yolo":
                yolo.write_data_yaml(
                    self.output / yolo.DATA_YAML,
                    self.classes,
                    splits=self._splits or {"train": f"{yolo.IMAGE_DIR}/train"},
                    root=".",
                )
            elif self.fmt == "coco":
                self._write_coco()
            elif self.fmt == "csv":
                self._write_csv()

        return {
            "format": self.fmt,
            "images": self.images_written,
            "labels": self.labels_written,
            "classes": list(self.classes.names),
            "warnings": list(self.warnings),
        }

    def _write_coco(self) -> None:
        for split, entries in sorted(self._collected.items()):
            builder = coco_fmt.CocoBuilder(description=f"Data Factory ({split})")
            for annotation, name in entries:
                builder.add(annotation, file_name=name)
            fs.ensure_dir(self.output / "annotations", self.ctx)
            builder.write(self.output / "annotations" / f"instances_{split}.json")
            merges = builder.categories.merges()
            if merges:
                self.warnings.append(
                    "case-variant classes merged: "
                    + "; ".join(f"{k} <- {', '.join(v)}" for k, v in merges.items())
                )

    def _write_csv(self) -> None:
        rows: list[Annotation] = []
        for _, entries in sorted(self._collected.items()):
            rows.extend(annotation for annotation, _ in entries)
        fs.ensure_dir(self.output, self.ctx)
        csvfmt.write(rows, self.output / "_annotations.csv")

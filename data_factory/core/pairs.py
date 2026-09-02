"""The image + sidecar-label pair.

Nearly every original script re-implemented this: loop the XMLs, try each image
extension in turn, remember to move both halves together. Getting it wrong is
how datasets end up with orphans. It lives here once, and every operation
consumes it, so "keep the pair together" stops being a thing anyone can forget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from . import fs


@dataclass(slots=True)
class Pair:
    """One dataset item: an image, its label file, or a half of either."""

    key: str
    """Unique identity: the stem, prefixed by sub-folder when scanning recursively."""

    stem: str
    image: Path | None = None
    label: Path | None = None

    @property
    def complete(self) -> bool:
        return self.image is not None and self.label is not None

    @property
    def image_only(self) -> bool:
        """An image with no annotation -- unlabelled, not necessarily junk."""
        return self.image is not None and self.label is None

    @property
    def label_only(self) -> bool:
        """An annotation whose image is gone -- always junk."""
        return self.image is None and self.label is not None

    @property
    def files(self) -> list[Path]:
        return [p for p in (self.image, self.label) if p is not None]

    @property
    def anchor(self) -> Path:
        """Whichever half exists, for reporting and sorting."""
        anchor = self.image or self.label
        assert anchor is not None, "a Pair must hold at least one file"
        return anchor


class PairIndex:
    """A scanned folder, resolved into pairs.

    Scanning is done once and reused; operations that need both the image and
    the label (rename, split, dedup, sort) then cannot desynchronise them.
    """

    def __init__(self, root: Path, pairs: list[Pair]) -> None:
        self.root = root
        self.pairs = pairs

    # -- construction ------------------------------------------------------
    @classmethod
    def scan(
        cls,
        root: Path,
        *,
        recursive: bool = False,
        image_exts: Iterable[str] = fs.IMAGE_EXTS,
        label_ext: str = fs.DEFAULT_LABEL_EXT,
    ) -> "PairIndex":
        extset = {e.lower() for e in image_exts}
        label_ext = label_ext.lower()
        found: dict[str, Pair] = {}

        for path in fs.iter_files(root, recursive=recursive):
            suffix = path.suffix.lower()
            is_img = suffix in extset
            is_label = suffix == label_ext
            if not (is_img or is_label):
                continue

            # Key on sub-folder + stem so a recursive scan of train/ and val/
            # does not fuse two distinct images that happen to share a name.
            relative = fs.relative_to_safe(path, root)
            key = str(relative.parent / relative.stem).replace("\\", "/")
            pair = found.get(key)
            if pair is None:
                pair = Pair(key=key, stem=path.stem)
                found[key] = pair

            if is_img:
                # Deterministic winner when both cat.jpg and cat.png exist, so
                # the un-chosen twin surfaces as a duplicate rather than
                # silently shadowing whichever the OS listed first.
                if pair.image is None or _ext_rank(path, extset) < _ext_rank(pair.image, extset):
                    pair.image = path
            else:
                pair.label = path

        return cls(root, [found[k] for k in sorted(found)])

    # -- views -------------------------------------------------------------
    def complete(self) -> list[Pair]:
        return [p for p in self.pairs if p.complete]

    def image_only(self) -> list[Pair]:
        return [p for p in self.pairs if p.image_only]

    def label_only(self) -> list[Pair]:
        return [p for p in self.pairs if p.label_only]

    def with_images(self) -> list[Pair]:
        return [p for p in self.pairs if p.image is not None]

    def summary(self) -> dict[str, int]:
        return {
            "pairs_total": len(self.pairs),
            "complete": len(self.complete()),
            "images_without_labels": len(self.image_only()),
            "labels_without_images": len(self.label_only()),
        }

    def __iter__(self) -> Iterator[Pair]:
        return iter(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)


def _ext_rank(path: Path, extset: set[str]) -> int:
    """Rank extensions by the canonical order so the choice is reproducible."""
    order = [e for e in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp") if e in extset]
    suffix = path.suffix.lower()
    return order.index(suffix) if suffix in order else len(order)


def sidecar_for(image: Path, label_ext: str = fs.DEFAULT_LABEL_EXT) -> Path:
    """The label path an image would have, whether or not it exists."""
    return image.with_suffix(label_ext)

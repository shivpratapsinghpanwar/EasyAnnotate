"""Shared fixtures: small synthetic datasets that carry real defects.

Images are generated from seeded noise rather than flat colour. A flat image has
almost no DCT structure, so every flat image perceptually hashes the same and
dedup tests would pass or fail for the wrong reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

IMAGE_SIZE = (120, 90)  # width, height


def _structured(seed: int, size: tuple[int, int]) -> np.ndarray:
    """A seeded gradient with a few solid blocks on it.

    Deliberately not random noise. Perceptual hashing looks at the low-frequency
    DCT coefficients, and noise has essentially none -- two noise images hash
    arbitrarily far apart, so a dedup test built on noise would prove nothing.
    Structure is what phash is designed to compare, and what real photos have.
    """
    rng = np.random.default_rng(seed)
    width, height = size
    _, columns = np.mgrid[0:height, 0:width]

    start = rng.integers(0, 256, 3).astype(np.float64)
    end = rng.integers(0, 256, 3).astype(np.float64)
    ramp = (columns / max(width - 1, 1))[..., None]
    canvas = start * (1 - ramp) + end * ramp

    for _ in range(3):
        left = int(rng.integers(0, width // 2))
        top = int(rng.integers(0, height // 2))
        right = min(width, left + int(rng.integers(width // 8, width // 2)))
        bottom = min(height, top + int(rng.integers(height // 8, height // 2)))
        canvas[top:bottom, left:right] = rng.integers(0, 256, 3).astype(np.float64)

    return canvas.astype(np.uint8)


def make_image(path: Path, seed: int, size: tuple[int, int] = IMAGE_SIZE) -> Path:
    """Write a deterministic, structured image that hashes distinctly per seed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_structured(seed, size)).save(path)
    return path


def make_similar_image(path: Path, seed: int, shift: int = 10) -> Path:
    """A visually near-identical variant: same scene, brighter and recompressed.

    This is the realistic near-duplicate -- the same photo saved twice by
    different tools -- which exact hashing misses and perceptual hashing catches.
    """
    array = _structured(seed, IMAGE_SIZE).astype(np.int16) + shift
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(path, quality=60)
    return path


def write_voc(
    path: Path,
    label: str,
    bbox: tuple[int, int, int, int] | None = (10, 10, 60, 70),
    polygon: list[tuple[int, int]] | None = None,
    size: tuple[int, int] = IMAGE_SIZE,
    dialect: str = "indexed",
) -> Path:
    """Write a Pascal VOC label in either polygon dialect.

    dialect="indexed" is the polygon annotator's <x1>/<y1> form.
    dialect="pt" is the SAM tool's <pt><x>/<y> form.
    """
    body = ""
    if bbox is not None:
        body += (
            f"<bndbox><xmin>{bbox[0]}</xmin><ymin>{bbox[1]}</ymin>"
            f"<xmax>{bbox[2]}</xmax><ymax>{bbox[3]}</ymax></bndbox>"
        )
    if polygon:
        if dialect == "indexed":
            points = "".join(
                f"<x{i}>{x}</x{i}><y{i}>{y}</y{i}>" for i, (x, y) in enumerate(polygon, 1)
            )
        else:
            points = "".join(f"<pt><x>{x}</x><y>{y}</y></pt>" for x, y in polygon)
        body += f"<polygon>{points}</polygon>"

    path.parent.mkdir(parents=True, exist_ok=True)
    # <folder>/<filename>/<path> mirror what the polygon annotator actually
    # writes, so retargeting on rename is exercised against a realistic file.
    path.write_text(
        f"<annotation><folder>images</folder>"
        f"<filename>{path.stem}.jpg</filename><path>{path.stem}.jpg</path>"
        f"<size><width>{size[0]}</width><height>{size[1]}</height><depth>3</depth></size>"
        f"<object><name>{label}</name>{body}</object></annotation>",
        encoding="utf-8",
    )
    return path


def write_background(path: Path, size: tuple[int, int] = IMAGE_SIZE) -> Path:
    """A valid, deliberately object-free label -- a hard negative."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"<annotation><folder>images</folder>"
        f"<filename>{path.stem}.jpg</filename><path>{path.stem}.jpg</path>"
        f"<size><width>{size[0]}</width><height>{size[1]}</height></size></annotation>",
        encoding="utf-8",
    )
    return path


def make_pair(folder: Path, stem: str, label: str, seed: int, **kwargs) -> tuple[Path, Path]:
    image = make_image(folder / f"{stem}.jpg", seed)
    annotation = write_voc(folder / f"{stem}.xml", label, **kwargs)
    return image, annotation


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A small dataset carrying every defect the real Project_Door set had.

    Contents: case-variant classes, both polygon dialects, a hard negative, an
    exact duplicate, an orphan image, an orphan label, and a corrupt XML.
    """
    folder = tmp_path / "data"
    folder.mkdir()

    make_pair(folder, "a", "Open", 1, polygon=[(10, 10), (60, 10), (60, 70)])
    make_pair(folder, "b", "open", 2, bbox=None, polygon=[(5, 5), (50, 5), (50, 55)], dialect="pt")
    make_pair(folder, "c", "Close", 3)
    make_pair(folder, "d", "close", 4)

    make_image(folder / "e.jpg", 5)
    write_background(folder / "e.xml")

    # An exact byte-for-byte duplicate of a.jpg, also annotated.
    (folder / "dup.jpg").write_bytes((folder / "a.jpg").read_bytes())
    write_voc(folder / "dup.xml", "Open")

    make_image(folder / "lonely_image.jpg", 7)
    write_voc(folder / "lonely_label.xml", "Open")
    (folder / "corrupt.xml").write_text("<annotation><object><name>Open", encoding="utf-8")
    make_image(folder / "corrupt.jpg", 8)

    return folder


@pytest.fixture
def clean_dataset(tmp_path: Path) -> Path:
    """Six well-formed pairs across two classes, plus one background."""
    folder = tmp_path / "clean"
    folder.mkdir()
    for index in range(3):
        make_pair(folder, f"open{index}", "Open", 100 + index)
    for index in range(3):
        make_pair(folder, f"close{index}", "Close", 200 + index)
    make_image(folder / "bg.jpg", 300)
    write_background(folder / "bg.xml")
    return folder


@pytest.fixture(autouse=True)
def _no_real_tools(monkeypatch, tmp_path_factory):
    """Never discover a developer's real tools environment from a test.

    Discovery is zero-config on purpose -- a venv at ~/.datafactory/tools is
    enough -- which means a test that lists tools would otherwise start the
    real sidecar, load PyTorch, and take a minute. Tests that want a sidecar
    name the fake one explicitly.
    """
    monkeypatch.setenv("DATAFACTORY_TOOLS_HOME", str(tmp_path_factory.mktemp("no-tools")))

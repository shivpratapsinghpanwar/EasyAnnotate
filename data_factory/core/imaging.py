"""Image I/O and annotation-aware geometry.

Resizing, tiling and augmentation all face the same problem: move the pixels,
and the boxes and polygons have to move with them or the dataset is quietly
ruined. Solving that once, here, is why those three operations are small.

A :class:`Transform` knows two things -- how to redraw the image, and where a
single point lands afterwards. :func:`transform_annotation` does the rest: it
maps every polygon vertex, rebuilds each box from its four transformed corners
(so rotation stays correct rather than degenerating into a box of a box), clips
to the new frame, and drops objects that fell out of view.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .annotations.model import Annotation, Point, Shape

#: Pixels below this in either dimension make an object worthless for training.
MIN_OBJECT_PIXELS = 2


# --------------------------------------------------------------------- I/O

def image_size(path: Path) -> tuple[int, int]:
    """``(width, height)`` read from the header, without decoding the pixels.

    Auditing 7,000 images should not cost 7,000 full JPEG decodes.
    """
    from PIL import Image  # local import: keeps module import cheap

    with Image.open(path) as im:
        return im.size


def load(path: Path) -> np.ndarray:
    """Decode to a BGR array, tolerating non-ASCII paths on Windows.

    ``cv2.imread`` silently returns None for paths it cannot encode, which on a
    Windows box with accented folder names looks exactly like a corrupt file.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise OSError(f"empty file: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"not a decodable image: {path}")
    return image


def save(image: np.ndarray, path: Path, *, quality: int = 95) -> Path:
    """Encode and write, again via a buffer so Unicode paths survive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    params: list[int] = []
    if suffix in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif suffix == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    elif suffix == ".png":
        # PNG is lossless; map the 0-100 dial onto compression effort instead.
        params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, 9 - int(quality) // 12))]
    ok, buffer = cv2.imencode(suffix, image, params)
    if not ok:
        raise OSError(f"could not encode {path.name}")
    buffer.tofile(str(path))
    return path


def exif_orientation(path: Path) -> int:
    """The EXIF orientation tag (1 = upright), or 1 when absent."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            exif = im.getexif()
        return int(exif.get(0x0112, 1) or 1)
    except Exception:
        return 1


# -------------------------------------------------------------- transforms

@dataclass(slots=True)
class Transform:
    """Base transform: identity, and the interface the others fill in.

    ``out_size`` is the frame the result lives in, which is what annotations
    are clipped against.
    """

    out_size: tuple[int, int]

    def image(self, im: np.ndarray) -> np.ndarray:
        return im

    def point(self, x: float, y: float) -> tuple[float, float]:
        return x, y

    def describe(self) -> str:
        return type(self).__name__.lower()


@dataclass(slots=True)
class Identity(Transform):
    """Pixels change, geometry does not -- every photometric augmentation."""


@dataclass(slots=True)
class Resize(Transform):
    """Independent x/y scaling. Distorts aspect ratio; cheap and predictable."""

    scale_x: float = 1.0
    scale_y: float = 1.0

    def image(self, im: np.ndarray) -> np.ndarray:
        w, h = self.out_size
        # Shrinking and growing want different filters; INTER_AREA avoids the
        # aliasing that makes downscaled thin objects vanish.
        interp = cv2.INTER_AREA if (w < im.shape[1] or h < im.shape[0]) else cv2.INTER_LINEAR
        return cv2.resize(im, (w, h), interpolation=interp)

    def point(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale_x, y * self.scale_y


@dataclass(slots=True)
class Letterbox(Transform):
    """Uniform scale plus padding -- aspect ratio preserved, nothing cropped."""

    scale: float = 1.0
    pad_x: int = 0
    pad_y: int = 0
    colour: tuple[int, int, int] = (114, 114, 114)

    def image(self, im: np.ndarray) -> np.ndarray:
        w, h = self.out_size
        inner_w = max(1, int(round(im.shape[1] * self.scale)))
        inner_h = max(1, int(round(im.shape[0] * self.scale)))
        interp = cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(im, (inner_w, inner_h), interpolation=interp)
        canvas = np.full((h, w, 3), self.colour, dtype=np.uint8)
        canvas[self.pad_y : self.pad_y + inner_h, self.pad_x : self.pad_x + inner_w] = resized
        return canvas

    def point(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.pad_x, y * self.scale + self.pad_y


@dataclass(slots=True)
class Crop(Transform):
    """Take a rectangular window. Backs both tiling and random-crop zoom."""

    left: int = 0
    top: int = 0

    def image(self, im: np.ndarray) -> np.ndarray:
        w, h = self.out_size
        window = im[self.top : self.top + h, self.left : self.left + w]
        if window.shape[0] != h or window.shape[1] != w:
            # Windows running off the edge are padded rather than shrunk, so
            # every tile in a grid comes out the same size.
            padded = np.zeros((h, w, 3), dtype=im.dtype)
            padded[: window.shape[0], : window.shape[1]] = window
            return padded
        return window

    def point(self, x: float, y: float) -> tuple[float, float]:
        return x - self.left, y - self.top


@dataclass(slots=True)
class FlipH(Transform):
    def image(self, im: np.ndarray) -> np.ndarray:
        return cv2.flip(im, 1)

    def point(self, x: float, y: float) -> tuple[float, float]:
        return self.out_size[0] - 1 - x, y


@dataclass(slots=True)
class FlipV(Transform):
    def image(self, im: np.ndarray) -> np.ndarray:
        return cv2.flip(im, 0)

    def point(self, x: float, y: float) -> tuple[float, float]:
        return x, self.out_size[1] - 1 - y


@dataclass(slots=True)
class Rot90(Transform):
    """Lossless 90-degree rotation. ``turns`` is counter-clockwise quarters."""

    turns: int = 1
    src_size: tuple[int, int] = (0, 0)

    def image(self, im: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(im, k=self.turns % 4))

    def point(self, x: float, y: float) -> tuple[float, float]:
        w, h = self.src_size
        turns = self.turns % 4
        for _ in range(turns):
            x, y = y, w - 1 - x
            w, h = h, w
        return x, y


@dataclass(slots=True)
class Rotate(Transform):
    """Arbitrary rotation about the centre, expanding the frame to fit."""

    degrees: float = 0.0
    matrix: Any = None

    def image(self, im: np.ndarray) -> np.ndarray:
        w, h = self.out_size
        return cv2.warpAffine(
            im, self.matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114)
        )

    def point(self, x: float, y: float) -> tuple[float, float]:
        m = self.matrix
        return (
            m[0, 0] * x + m[0, 1] * y + m[0, 2],
            m[1, 0] * x + m[1, 1] * y + m[1, 2],
        )


def make_rotate(width: int, height: int, degrees: float) -> Rotate:
    """Build a rotation that keeps the whole image, growing the canvas."""
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, degrees, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(round(height * sin + width * cos))
    new_h = int(round(height * cos + width * sin))
    matrix[0, 2] += new_w / 2.0 - centre[0]
    matrix[1, 2] += new_h / 2.0 - centre[1]
    return Rotate(out_size=(new_w, new_h), degrees=degrees, matrix=matrix)


def make_resize(
    width: int, height: int, target_w: int, target_h: int, mode: str = "letterbox"
) -> Transform:
    """Pick the transform for a target size under the requested fit mode.

    ``stretch`` fills the frame and distorts, ``letterbox`` pads, ``crop``
    fills by scaling up and cutting the overflow (centre-anchored), and ``fit``
    scales the long edge only, leaving the other free.
    """
    if mode == "stretch":
        return Resize(
            out_size=(target_w, target_h),
            scale_x=target_w / width,
            scale_y=target_h / height,
        )
    if mode == "fit":
        scale = min(target_w / width, target_h / height)
        out_w, out_h = max(1, round(width * scale)), max(1, round(height * scale))
        return Resize(out_size=(out_w, out_h), scale_x=out_w / width, scale_y=out_h / height)
    if mode == "crop":
        scale = max(target_w / width, target_h / height)
        scaled_w, scaled_h = max(1, round(width * scale)), max(1, round(height * scale))
        left = (scaled_w - target_w) // 2
        top = (scaled_h - target_h) // 2
        return _Chain(
            out_size=(target_w, target_h),
            steps=(
                Resize(out_size=(scaled_w, scaled_h), scale_x=scale, scale_y=scale),
                Crop(out_size=(target_w, target_h), left=left, top=top),
            ),
        )
    scale = min(target_w / width, target_h / height)
    inner_w, inner_h = max(1, round(width * scale)), max(1, round(height * scale))
    return Letterbox(
        out_size=(target_w, target_h),
        scale=scale,
        pad_x=(target_w - inner_w) // 2,
        pad_y=(target_h - inner_h) // 2,
    )


@dataclass(slots=True)
class _Chain(Transform):
    """Two or more transforms applied in order, as one."""

    steps: tuple[Transform, ...] = ()

    def image(self, im: np.ndarray) -> np.ndarray:
        for step in self.steps:
            im = step.image(im)
        return im

    def point(self, x: float, y: float) -> tuple[float, float]:
        for step in self.steps:
            x, y = step.point(x, y)
        return x, y


def chain(*steps: Transform) -> Transform:
    return _Chain(out_size=steps[-1].out_size, steps=steps)


# ------------------------------------------------- annotation-aware mapping

def transform_annotation(
    annotation: Annotation,
    transform: Transform,
    *,
    min_visibility: float = 0.25,
    filename: str | None = None,
) -> Annotation:
    """Move every shape through ``transform`` and clip it to the new frame.

    ``min_visibility`` is the fraction of an object's original area that must
    survive clipping for it to be kept. Without it, tiling produces slivers --
    two pixels of a door at the edge of a tile, labelled "door" -- which teach
    a model that almost nothing is a door.
    """
    out_w, out_h = transform.out_size
    kept: list[Shape] = []

    for shape in annotation.shapes:
        moved = _transform_shape(shape, transform)
        if moved is None:
            continue
        before = max(1.0, float(shape.bbox_area))
        clipped = moved.clamped(out_w, out_h)
        if clipped.width < MIN_OBJECT_PIXELS or clipped.height < MIN_OBJECT_PIXELS:
            continue
        # Compare against the *transformed* area, so a legitimate downscale is
        # not mistaken for the object having been cropped away.
        expected = max(1.0, float(moved.bbox_area))
        if expected / before > 0.0 and clipped.bbox_area / expected < min_visibility:
            continue
        kept.append(clipped)

    return Annotation(
        filename=filename or annotation.filename,
        width=out_w,
        height=out_h,
        depth=annotation.depth,
        shapes=kept,
        folder=annotation.folder,
        source_path=annotation.source_path,
    )


def _transform_shape(shape: Shape, transform: Transform) -> Shape | None:
    polygon: list[Point] | None = None
    if shape.has_polygon:
        assert shape.polygon is not None
        polygon = [_ipoint(transform.point(x, y)) for x, y in shape.polygon]

    # Rebuild the box from all four transformed corners. Transforming only the
    # two opposite corners is right for scaling and wrong for rotation.
    xmin, ymin, xmax, ymax = shape.bbox
    corners = [
        transform.point(xmin, ymin),
        transform.point(xmax, ymin),
        transform.point(xmax, ymax),
        transform.point(xmin, ymax),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    box = (
        int(math.floor(min(xs))),
        int(math.floor(min(ys))),
        int(math.ceil(max(xs))),
        int(math.ceil(max(ys))),
    )
    return Shape(
        label=shape.label,
        bbox=box,
        polygon=polygon,
        pose=shape.pose,
        truncated=shape.truncated,
        difficult=shape.difficult,
        occluded=shape.occluded,
    )


def _ipoint(pair: tuple[float, float]) -> Point:
    return int(round(pair[0])), int(round(pair[1]))


# ------------------------------------------------------------ photometric

def adjust_brightness(im: np.ndarray, factor: float) -> np.ndarray:
    return cv2.convertScaleAbs(im, alpha=1.0, beta=(factor - 1.0) * 255.0 * 0.5)


def adjust_contrast(im: np.ndarray, factor: float) -> np.ndarray:
    mean = float(im.mean())
    return np.clip((im.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def adjust_saturation(im: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_hue(im: np.ndarray, degrees: float) -> np.ndarray:
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.int16)
    # OpenCV packs hue into 0-179, so a degree is half a unit.
    hsv[..., 0] = (hsv[..., 0] + int(degrees / 2.0)) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def blur(im: np.ndarray, radius: float) -> np.ndarray:
    k = max(1, int(radius) * 2 + 1)
    return cv2.GaussianBlur(im, (k, k), 0)


def add_noise(im: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, amount * 255.0, im.shape)
    return np.clip(im.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def to_grayscale(im: np.ndarray) -> np.ndarray:
    """Grey pixels, still three channels -- models expect a 3-channel input."""
    return cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def cutout(
    im: np.ndarray, count: int, fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """Erase random rectangles. Teaches a model not to rely on one landmark."""
    out = im.copy()
    h, w = out.shape[:2]
    box_w = max(1, int(w * fraction))
    box_h = max(1, int(h * fraction))
    for _ in range(count):
        x = int(rng.integers(0, max(1, w - box_w)))
        y = int(rng.integers(0, max(1, h - box_h)))
        out[y : y + box_h, x : x + box_w] = int(rng.integers(0, 256))
    return out


def apply_exif(im: np.ndarray, orientation: int) -> np.ndarray:
    """Bake an EXIF orientation into the pixels."""
    if orientation == 2:
        return cv2.flip(im, 1)
    if orientation == 3:
        return cv2.rotate(im, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(im, 0)
    if orientation == 5:
        return cv2.flip(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), 1)
    if orientation == 6:
        return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    if orientation == 8:
        return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return im


def exif_transform(orientation: int, width: int, height: int) -> Transform:
    """The geometry matching :func:`apply_exif`, for moving the annotations."""
    if orientation in (1,):
        return Identity(out_size=(width, height))
    if orientation == 2:
        return FlipH(out_size=(width, height))
    if orientation == 3:
        return Rot90(out_size=(width, height), turns=2, src_size=(width, height))
    if orientation == 4:
        return FlipV(out_size=(width, height))
    if orientation == 6:
        return Rot90(out_size=(height, width), turns=3, src_size=(width, height))
    if orientation == 8:
        return Rot90(out_size=(height, width), turns=1, src_size=(width, height))
    if orientation == 5:
        return chain(
            Rot90(out_size=(height, width), turns=3, src_size=(width, height)),
            FlipH(out_size=(height, width)),
        )
    if orientation == 7:
        return chain(
            Rot90(out_size=(height, width), turns=1, src_size=(width, height)),
            FlipH(out_size=(height, width)),
        )
    return Identity(out_size=(width, height))


def tile_windows(
    width: int, height: int, tile_w: int, tile_h: int, overlap: float
) -> Iterable[tuple[int, int, int, int]]:
    """Grid of overlapping windows covering the image.

    Overlap matters: an object straddling a tile boundary is cut in half by a
    naive grid, and with enough overlap it lands whole in at least one tile.
    """
    step_x = max(1, int(tile_w * (1.0 - overlap)))
    step_y = max(1, int(tile_h * (1.0 - overlap)))
    xs = list(range(0, max(1, width - tile_w + 1), step_x))
    ys = list(range(0, max(1, height - tile_h + 1), step_y))
    # Always include a window flush with the right/bottom edge, or the last
    # strip of a non-divisible image is never seen.
    if xs[-1] + tile_w < width:
        xs.append(max(0, width - tile_w))
    if ys[-1] + tile_h < height:
        ys.append(max(0, height - tile_h))
    for y in ys:
        for x in xs:
            yield x, y, tile_w, tile_h

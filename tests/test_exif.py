"""EXIF orientation, which decides whether a box lands on the object.

A phone photo is commonly stored sideways with a tag saying "rotate me".
Browsers honour that tag. PIL's ``Image.size``, OpenCV, and every batch
operation in this package do not. So a person draws a box on the *displayed*
image and we record it against the *stored* dimensions -- rotated 90 degrees
from the pixels a trainer will load, with nothing anywhere reporting an error.

Not currently live: all 3,000 sampled Project_Door images are orientation 1.
It arrives with the first phone photo, which is exactly what a public
annotation tool receives. These tests build the tagged images the real data
has never contained.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import piexif
import pytest
from PIL import Image

from data_factory.core import imaging

pytest.importorskip("piexif", reason="dev-only: pip install piexif")

#: Portrait stored as landscape, which is the common phone case.
STORED = (200, 100)          # width, height as written to disk
TRANSPOSING = (5, 6, 7, 8)   # tags that swap the axes when displayed


def write_tagged(path: Path, orientation: int, size: tuple[int, int] = STORED) -> Path:
    """A real JPEG carrying a real EXIF orientation tag."""
    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, : width // 2] = (220, 40, 40)     # left half red, so a flip is visible
    array[: height // 2, :] += 20              # top slightly brighter

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=95)
    exif = piexif.dump({"0th": {piexif.ImageIFD.Orientation: orientation}})

    path.parent.mkdir(parents=True, exist_ok=True)
    piexif.insert(exif, buffer.getvalue(), str(path))
    return path


class TestSizeAgreesWithPixels:
    """The invariant: whatever image_size() reports, load() delivers.

    OpenCV's imdecode applies the orientation tag, so load() returns upright
    pixels. If image_size() reported the stored frame instead, a VOC <size>
    would be transposed relative to the image the annotator drew on -- silently.
    """

    @pytest.mark.parametrize("orientation", range(1, 9))
    def test_reported_size_matches_the_decoded_array(self, tmp_path: Path, orientation: int):
        path = write_tagged(tmp_path / f"o{orientation}.jpg", orientation)
        array = imaging.load(path)
        assert imaging.image_size(path) == (array.shape[1], array.shape[0]), orientation

    @pytest.mark.parametrize("orientation", TRANSPOSING)
    def test_transposing_tags_are_reported_rotated(self, tmp_path: Path, orientation: int):
        path = write_tagged(tmp_path / f"t{orientation}.jpg", orientation)
        assert imaging.stored_size(path) == STORED
        assert imaging.image_size(path) == (STORED[1], STORED[0])

    @pytest.mark.parametrize("orientation", [1, 2, 3, 4])
    def test_non_transposing_tags_keep_the_frame(self, tmp_path: Path, orientation: int):
        """Mirrors and a half turn move pixels but not width and height."""
        path = write_tagged(tmp_path / f"n{orientation}.jpg", orientation)
        assert imaging.image_size(path) == STORED

    def test_we_agree_with_pillow(self, tmp_path: Path):
        """Cross-checked against the library whose behaviour browsers match."""
        from PIL import ImageOps

        for orientation in range(1, 9):
            path = write_tagged(tmp_path / f"p{orientation}.jpg", orientation)
            with Image.open(path) as im:
                expected = ImageOps.exif_transpose(im).size
            assert imaging.image_size(path) == expected, orientation

    def test_an_untagged_image_is_unaffected(self, tmp_path: Path):
        path = tmp_path / "plain.png"
        imaging.save(np.zeros((60, 90, 3), dtype=np.uint8), path)
        assert imaging.image_size(path) == imaging.stored_size(path) == (90, 60)


class TestIngest:
    @pytest.mark.parametrize("orientation", TRANSPOSING)
    def test_ingest_returns_the_upright_size(self, tmp_path: Path, orientation: int):
        path = write_tagged(tmp_path / f"i{orientation}.jpg", orientation)
        assert imaging.ingest(path) == (STORED[1], STORED[0])

    @pytest.mark.parametrize("orientation", TRANSPOSING)
    def test_after_ingest_the_tag_is_gone_and_everything_agrees(
        self, tmp_path: Path, orientation: int
    ):
        """The point of baking: one coordinate space, forever after."""
        path = write_tagged(tmp_path / f"b{orientation}.jpg", orientation)
        imaging.ingest(path)
        assert imaging.exif_orientation(path) in (0, 1)
        assert imaging.stored_size(path) == imaging.image_size(path)
        array = imaging.load(path)
        assert imaging.stored_size(path) == (array.shape[1], array.shape[0])

    def test_an_upright_image_is_left_byte_for_byte_alone(self, tmp_path: Path):
        """Re-encoding a JPEG that needs nothing done costs quality for free."""
        path = write_tagged(tmp_path / "upright.jpg", 1)
        before = path.read_bytes()
        assert imaging.ingest(path) == STORED
        assert path.read_bytes() == before

    def test_ingest_can_write_somewhere_else(self, tmp_path: Path):
        source = write_tagged(tmp_path / "src.jpg", 6)
        target = tmp_path / "out" / "dst.jpg"
        assert imaging.ingest(source, target) == (STORED[1], STORED[0])
        assert target.is_file()
        # The source is never modified when a destination is given: it still
        # holds the un-baked frame and its original tag.
        assert imaging.stored_size(source) == STORED
        assert imaging.exif_orientation(source) == 6

    def test_an_upright_image_is_copied_intact_to_a_destination(self, tmp_path: Path):
        source = write_tagged(tmp_path / "src.jpg", 1)
        target = tmp_path / "out" / "dst.jpg"
        imaging.ingest(source, target)
        assert target.read_bytes() == source.read_bytes()

    def test_the_pixels_actually_move(self, tmp_path: Path):
        """Orientation 6 is a 90-degree clockwise turn. The red half must end
        up on top, not on the left."""
        path = write_tagged(tmp_path / "rot.jpg", 6)
        imaging.ingest(path)
        array = imaging.load(path)
        height, width = array.shape[:2]
        top = array[: height // 2].mean(axis=(0, 1))
        bottom = array[height // 2 :].mean(axis=(0, 1))
        # OpenCV is BGR, so channel 2 is red.
        assert top[2] > bottom[2] + 40, "the red half did not rotate to the top"


class TestAnnotationsLandOnTheObject:
    """The end-to-end property, and the only one that actually matters.

    A person draws a box on what the browser shows. We record it against
    image_size(). A trainer loads pixels via load(). All three must be the same
    coordinate space -- if they are, a box over a red region stays over the red
    region for every orientation tag in the spec.
    """

    @pytest.mark.parametrize("orientation", range(1, 9))
    def test_a_box_in_reported_space_covers_the_right_pixels(
        self, tmp_path: Path, orientation: int
    ):
        from data_factory.core.annotations.model import Annotation, Shape

        path = write_tagged(tmp_path / f"m{orientation}.jpg", orientation)
        width, height = imaging.image_size(path)
        array = imaging.load(path)
        assert (array.shape[1], array.shape[0]) == (width, height), orientation

        # Find the red region in displayed space, and box exactly it. OpenCV is
        # BGR, so channel 2 carries red.
        red = array[..., 2].astype(int) - array[..., 0].astype(int) > 80
        ys, xs = np.nonzero(red)
        shape = Shape(label="red", bbox=(int(xs.min()), int(ys.min()),
                                         int(xs.max()), int(ys.max())))
        annotation = Annotation(filename=path.name, width=width, height=height, shapes=[shape])

        # A box recorded this way must survive a real VOC round trip and still
        # sit on the object.
        from data_factory.core.annotations import voc

        back = voc.read(voc.write(annotation, tmp_path / f"m{orientation}.xml"))
        assert (back.width, back.height) == (width, height), orientation

        xmin, ymin, xmax, ymax = back.shapes[0].bbox
        patch = array[ymin : ymax + 1, xmin : xmax + 1]
        assert patch.size, f"empty patch for orientation {orientation}"
        assert red[ymin : ymax + 1, xmin : xmax + 1].mean() > 0.9, orientation

    @pytest.mark.parametrize("orientation", TRANSPOSING)
    def test_ingesting_first_changes_nothing_about_where_the_box_goes(
        self, tmp_path: Path, orientation: int
    ):
        """Baking is an optimisation, not a correction -- it must be a no-op
        as far as coordinates are concerned."""
        path = write_tagged(tmp_path / f"k{orientation}.jpg", orientation)
        before = imaging.load(path)
        imaging.ingest(path)
        after = imaging.load(path)
        assert before.shape == after.shape
        # Re-encoded at quality 95, so compare loosely rather than exactly.
        assert float(np.abs(before.astype(int) - after.astype(int)).mean()) < 6.0

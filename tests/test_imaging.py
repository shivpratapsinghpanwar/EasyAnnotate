"""Geometry. If these are wrong, every image operation silently corrupts data.

The point of each test is that the annotation lands where the pixels landed.
A resize test that only checks the output size proves nothing -- the failure
mode is boxes that stay put while the image shrinks underneath them.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_factory.core import imaging
from data_factory.core.annotations.model import Annotation, Shape


def square(size=(100, 100), bbox=(10, 20, 50, 60), polygon=None) -> Annotation:
    return Annotation(
        filename="x.jpg",
        width=size[0],
        height=size[1],
        shapes=[Shape(label="thing", bbox=bbox, polygon=polygon)],
    )


def canvas(width=100, height=100) -> np.ndarray:
    """A distinguishable image: a bright block in the top-left quadrant."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[0 : height // 2, 0 : width // 2] = 255
    return array


class TestResize:
    def test_stretch_scales_the_box_by_the_same_factor(self):
        transform = imaging.make_resize(100, 100, 50, 200, "stretch")
        moved = imaging.transform_annotation(square(), transform)
        assert moved.width, moved.height == (50, 200)
        # x halved, y doubled -- exactly what happened to the pixels.
        assert moved.shapes[0].bbox == (5, 40, 25, 120)

    def test_letterbox_preserves_aspect_and_offsets_by_the_padding(self):
        # 100x50 into a 100x100 frame: scale 1.0, padded 25px top and bottom.
        transform = imaging.make_resize(100, 50, 100, 100, "letterbox")
        moved = imaging.transform_annotation(square(size=(100, 50), bbox=(10, 10, 50, 40)), transform)
        assert moved.width, moved.height == (100, 100)
        assert moved.shapes[0].bbox == (10, 35, 50, 65)

    def test_letterbox_pixels_and_geometry_agree(self):
        """The bright quadrant must still contain the box after letterboxing."""
        image = canvas(100, 50)
        transform = imaging.make_resize(100, 50, 100, 100, "letterbox")
        out = transform.image(image)
        assert out.shape[:2] == (100, 100)
        moved = imaging.transform_annotation(square(size=(100, 50), bbox=(5, 5, 45, 20)), transform)
        xmin, ymin, xmax, ymax = moved.shapes[0].bbox
        # Every pixel the box covers should still be the bright block.
        assert out[ymin:ymax, xmin:xmax].mean() > 200

    def test_fit_leaves_the_short_edge_free(self):
        transform = imaging.make_resize(200, 100, 100, 100, "fit")
        assert transform.out_size == (100, 50)

    def test_crop_fills_the_frame_exactly(self):
        transform = imaging.make_resize(200, 100, 100, 100, "crop")
        assert transform.out_size == (100, 100)
        assert transform.image(canvas(200, 100)).shape[:2] == (100, 100)


class TestFlips:
    def test_horizontal_flip_mirrors_the_box(self):
        transform = imaging.FlipH(out_size=(100, 100))
        moved = imaging.transform_annotation(square(bbox=(10, 20, 30, 60)), transform)
        # x -> 99 - x, and the corners swap roles, so the box stays well-formed.
        assert moved.shapes[0].bbox == (69, 20, 89, 60)

    def test_flip_is_its_own_inverse(self):
        transform = imaging.FlipH(out_size=(100, 100))
        once = imaging.transform_annotation(square(), transform)
        twice = imaging.transform_annotation(once, transform)
        assert twice.shapes[0].bbox == square().shapes[0].bbox

    def test_flipped_pixels_and_box_stay_together(self):
        image = canvas()
        transform = imaging.FlipH(out_size=(100, 100))
        out = transform.image(image)
        moved = imaging.transform_annotation(square(bbox=(5, 5, 45, 45)), transform)
        xmin, ymin, xmax, ymax = moved.shapes[0].bbox
        assert out[ymin:ymax, xmin:xmax].mean() > 200


class TestRotation:
    def test_quarter_turn_swaps_the_frame(self):
        transform = imaging.Rot90(out_size=(50, 100), turns=1, src_size=(100, 50))
        moved = imaging.transform_annotation(square(size=(100, 50), bbox=(10, 5, 30, 25)), transform)
        assert (moved.width, moved.height) == (50, 100)
        assert moved.shapes[0].width > 0 and moved.shapes[0].height > 0

    def test_rot90_pixels_and_geometry_agree(self):
        image = canvas(100, 50)
        transform = imaging.Rot90(out_size=(50, 100), turns=1, src_size=(100, 50))
        out = transform.image(image)
        assert out.shape[:2] == (100, 50)
        moved = imaging.transform_annotation(square(size=(100, 50), bbox=(2, 2, 45, 22)), transform)
        xmin, ymin, xmax, ymax = moved.shapes[0].bbox
        assert out[ymin:ymax, xmin:xmax].mean() > 200

    def test_four_quarter_turns_return_to_the_start(self):
        original = square(size=(100, 60), bbox=(10, 10, 40, 30))
        current = original
        size = (100, 60)
        for _ in range(4):
            out = (size[1], size[0])
            transform = imaging.Rot90(out_size=out, turns=1, src_size=size)
            current = imaging.transform_annotation(current, transform)
            size = out
        assert current.shapes[0].bbox == original.shapes[0].bbox

    def test_arbitrary_rotation_expands_the_canvas(self):
        transform = imaging.make_rotate(100, 100, 45)
        # A square rotated 45 degrees needs a frame sqrt(2) times wider.
        assert transform.out_size[0] > 130

    def test_rotation_uses_all_four_corners(self):
        """Transforming only two corners gives a visibly too-small box."""
        transform = imaging.make_rotate(100, 100, 45)
        moved = imaging.transform_annotation(square(bbox=(40, 40, 60, 60)), transform)
        box = moved.shapes[0]
        # A 20x20 box rotated 45 degrees has an axis-aligned extent of ~28px.
        assert 26 <= box.width <= 31, box.bbox
        assert 26 <= box.height <= 31, box.bbox


class TestCropAndVisibility:
    def test_crop_translates_coordinates(self):
        transform = imaging.Crop(out_size=(50, 50), left=20, top=20)
        moved = imaging.transform_annotation(square(bbox=(30, 30, 50, 50)), transform)
        assert moved.shapes[0].bbox == (10, 10, 30, 30)

    def test_object_cropped_entirely_away_is_dropped(self):
        transform = imaging.Crop(out_size=(30, 30), left=60, top=60)
        moved = imaging.transform_annotation(square(bbox=(0, 0, 20, 20)), transform)
        assert moved.shapes == []

    def test_sliver_below_min_visibility_is_dropped(self):
        # Only the leftmost 2px of a 40px-wide box survive this window.
        transform = imaging.Crop(out_size=(50, 50), left=38, top=20)
        moved = imaging.transform_annotation(
            square(bbox=(0, 20, 40, 60)), transform, min_visibility=0.5
        )
        assert moved.shapes == []

    def test_the_same_sliver_is_kept_when_visibility_is_relaxed(self):
        transform = imaging.Crop(out_size=(50, 50), left=38, top=20)
        moved = imaging.transform_annotation(
            square(bbox=(0, 20, 40, 60)), transform, min_visibility=0.0
        )
        assert len(moved.shapes) == 1


class TestPolygons:
    def test_every_vertex_moves(self):
        polygon = [(10, 10), (30, 10), (30, 30), (10, 30)]
        transform = imaging.make_resize(100, 100, 50, 50, "stretch")
        moved = imaging.transform_annotation(square(polygon=polygon), transform)
        assert moved.shapes[0].polygon == [(5, 5), (15, 5), (15, 15), (5, 15)]

    def test_box_is_rebuilt_from_the_moved_polygon_bounds(self):
        polygon = [(10, 10), (30, 10), (30, 30), (10, 30)]
        transform = imaging.FlipH(out_size=(100, 100))
        moved = imaging.transform_annotation(
            square(bbox=(10, 10, 30, 30), polygon=polygon), transform
        )
        shape = moved.shapes[0]
        xs = [p[0] for p in shape.polygon]
        assert min(xs) == shape.bbox[0] and max(xs) == shape.bbox[2]


class TestChain:
    def test_two_transforms_compose(self):
        combined = imaging.chain(
            imaging.FlipH(out_size=(100, 100)),
            imaging.Crop(out_size=(70, 100), left=25, top=0),
        )
        moved = imaging.transform_annotation(square(bbox=(10, 10, 30, 30)), combined)
        # Flip puts it at 69..89, the crop then subtracts 25.
        assert moved.shapes[0].bbox == (44, 10, 64, 30)

    def test_a_composed_box_is_still_clipped_to_the_final_frame(self):
        """The clip happens once, against the last transform's frame."""
        combined = imaging.chain(
            imaging.FlipH(out_size=(100, 100)),
            imaging.Crop(out_size=(50, 100), left=25, top=0),
        )
        moved = imaging.transform_annotation(square(bbox=(10, 10, 30, 30)), combined)
        assert moved.shapes[0].bbox == (44, 10, 49, 30)

    def test_chained_pixels_match_chained_geometry(self):
        image = canvas()
        combined = imaging.chain(
            imaging.FlipH(out_size=(100, 100)),
            imaging.Crop(out_size=(60, 60), left=40, top=0),
        )
        out = combined.image(image)
        moved = imaging.transform_annotation(square(bbox=(5, 5, 45, 45)), combined)
        xmin, ymin, xmax, ymax = moved.shapes[0].bbox
        assert out[ymin:ymax, xmin:xmax].mean() > 200


class TestTiling:
    def test_windows_cover_the_whole_image(self):
        windows = list(imaging.tile_windows(1000, 1000, 400, 400, 0.0))
        right = max(x + w for x, _, w, _ in windows)
        bottom = max(y + h for _, y, _, h in windows)
        assert right >= 1000 and bottom >= 1000

    def test_overlap_produces_more_windows(self):
        none = len(list(imaging.tile_windows(1000, 1000, 400, 400, 0.0)))
        some = len(list(imaging.tile_windows(1000, 1000, 400, 400, 0.5)))
        assert some > none

    def test_a_tile_smaller_than_the_image_still_yields_one_window(self):
        windows = list(imaging.tile_windows(100, 100, 640, 640, 0.2))
        assert len(windows) == 1

    def test_crop_pads_a_window_running_off_the_edge(self):
        crop = imaging.Crop(out_size=(60, 60), left=60, top=60)
        assert crop.image(canvas()).shape[:2] == (60, 60)


class TestPhotometric:
    def test_brightness_moves_the_mean(self):
        image = np.full((10, 10, 3), 100, dtype=np.uint8)
        assert imaging.adjust_brightness(image, 1.5).mean() > image.mean()
        assert imaging.adjust_brightness(image, 0.5).mean() < image.mean()

    def test_grayscale_equalises_the_channels(self):
        out = imaging.to_grayscale(canvas())
        assert out.shape[2] == 3
        assert np.array_equal(out[..., 0], out[..., 2])

    def test_noise_is_reproducible_for_a_seed(self):
        image = np.full((20, 20, 3), 128, dtype=np.uint8)
        a = imaging.add_noise(image, 0.1, np.random.default_rng(7))
        b = imaging.add_noise(image, 0.1, np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_cutout_erases_something(self):
        image = np.full((50, 50, 3), 200, dtype=np.uint8)
        out = imaging.cutout(image, 3, 0.3, np.random.default_rng(1))
        assert not np.array_equal(out, image)


class TestExif:
    @pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_pixel_and_geometry_transforms_agree_on_output_size(self, orientation):
        """The declared frame must match what apply_exif actually produces."""
        image = canvas(80, 40)
        out = imaging.apply_exif(image, orientation)
        transform = imaging.exif_transform(orientation, 80, 40)
        assert (out.shape[1], out.shape[0]) == transform.out_size, orientation

    def test_upright_orientation_changes_nothing(self):
        image = canvas()
        assert np.array_equal(imaging.apply_exif(image, 1), image)


class TestIO:
    def test_roundtrip_through_save_and_load(self, tmp_path):
        image = canvas(64, 48)
        path = tmp_path / "out.png"
        imaging.save(image, path)
        assert imaging.load(path).shape == image.shape

    def test_image_size_reads_the_header_only(self, tmp_path):
        path = tmp_path / "out.png"
        imaging.save(canvas(64, 48), path)
        assert imaging.image_size(path) == (64, 48)

    def test_loading_a_non_image_raises_rather_than_returning_none(self, tmp_path):
        path = tmp_path / "not.jpg"
        path.write_text("this is not an image", encoding="utf-8")
        with pytest.raises(OSError):
            imaging.load(path)

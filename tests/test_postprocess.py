"""Turning a model's mask into a shape a person can actually edit.

The legacy SAM script wrote every boundary pixel as a polygon vertex. A 512x512
mask has a boundary thousands of pixels long, which is why its XML ran to
hundreds of kilobytes per image and why dragging a point in it was hopeless.
These tests are mostly about that: the outline has to be recognisable *and*
small enough to edit.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_factory.annotate.tools import postprocess
from data_factory.annotate.tools.postprocess import MaskError


def rectangle_mask(width=200, height=160, box=(40, 30, 150, 120)) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 1
    return mask


def disc_mask(width=256, height=256, radius=90) -> np.ndarray:
    """A circle -- the worst case for vertex count, since every boundary pixel
    is a corner as far as a naive tracer is concerned."""
    rows, columns = np.ogrid[:height, :width]
    centre_y, centre_x = height // 2, width // 2
    return (((rows - centre_y) ** 2 + (columns - centre_x) ** 2) <= radius**2).astype(np.uint8)


# ------------------------------------------------------------------ basics

def test_a_rectangle_becomes_four_corners():
    polygon = postprocess.mask_to_polygon(rectangle_mask())
    assert polygon is not None
    assert len(polygon) == 4, f"a rectangle should simplify to 4 points, got {len(polygon)}"


def test_the_outline_lands_where_the_mask_is():
    polygon = postprocess.mask_to_polygon(rectangle_mask(box=(40, 30, 150, 120)))
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    assert min(xs) == pytest.approx(40, abs=2)
    assert min(ys) == pytest.approx(30, abs=2)
    assert max(xs) == pytest.approx(149, abs=2)
    assert max(ys) == pytest.approx(119, abs=2)


def test_an_empty_mask_returns_nothing_rather_than_raising():
    """Clicking on background is a normal thing to do."""
    assert postprocess.mask_to_polygon(np.zeros((50, 50), dtype=np.uint8)) is None


def test_a_speck_is_not_an_object():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:12, 10:12] = 1
    assert postprocess.mask_to_polygon(mask) is None


def test_a_boolean_mask_works():
    assert postprocess.mask_to_polygon(rectangle_mask().astype(bool)) is not None


def test_a_float_probability_mask_is_thresholded():
    mask = rectangle_mask().astype(np.float32) * 0.9
    assert postprocess.mask_to_polygon(mask) is not None
    # Everything below 0.5 is background, so a uniformly weak mask is nothing.
    assert postprocess.mask_to_polygon(rectangle_mask().astype(np.float32) * 0.2) is None


def test_a_channel_axis_is_tolerated():
    mask = rectangle_mask()[..., None]
    assert postprocess.mask_to_polygon(mask) is not None
    batched = rectangle_mask()[None, ...]
    assert postprocess.mask_to_polygon(batched) is not None


def test_a_mask_of_the_wrong_shape_is_refused_clearly():
    with pytest.raises(MaskError):
        postprocess.mask_to_polygon(np.zeros((4, 4, 3), dtype=np.uint8))


# ------------------------------------------------- the point of the module

def test_a_circle_does_not_become_a_thousand_vertices():
    """The failure the legacy tool actually had."""
    polygon = postprocess.mask_to_polygon(disc_mask())
    assert polygon is not None
    assert len(polygon) <= postprocess.MAX_VERTICES
    # And it is still round enough to recognise.
    assert len(polygon) >= 8


def test_the_vertex_cap_is_honoured_even_when_asked_for_a_tiny_epsilon():
    polygon = postprocess.mask_to_polygon(disc_mask(), epsilon=0.00001, max_vertices=20)
    assert polygon is not None
    assert len(polygon) <= 20


def test_relaxing_the_cap_keeps_the_shape_closed_not_truncated():
    """Truncating a point list would leave a straight line where the object
    was, which reads as a bug. Relaxing epsilon keeps it recognisable."""
    coarse = postprocess.mask_to_polygon(disc_mask(), max_vertices=8)
    assert coarse is not None and len(coarse) >= 3
    xs = [p[0] for p in coarse]
    ys = [p[1] for p in coarse]
    # Still spans most of the disc in both directions.
    assert max(xs) - min(xs) > 120
    assert max(ys) - min(ys) > 120


def test_simplification_is_scale_free():
    """A fixed pixel epsilon would over-simplify small objects and
    under-simplify large ones; datasets contain both."""
    small = postprocess.mask_to_polygon(disc_mask(width=64, height=64, radius=20))
    large = postprocess.mask_to_polygon(disc_mask(width=1024, height=1024, radius=400))
    assert small is not None and large is not None
    assert abs(len(small) - len(large)) <= 12, (len(small), len(large))


def test_only_the_largest_blob_is_kept():
    """One click makes one shape. Silently producing four is worse."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[10:100, 10:100] = 1     # the big one
    mask[150:170, 150:170] = 1   # a stray
    polygon = postprocess.mask_to_polygon(mask)
    xs = [p[0] for p in polygon]
    assert max(xs) < 130, "the stray blob was included"


# ------------------------------------------------------------------ shapes

def test_a_mask_becomes_a_clamped_labelled_shape():
    shape = postprocess.mask_to_shape(rectangle_mask(), "Door", 200, 160, confidence=0.8)
    assert shape is not None
    assert shape.label == "Door"
    assert shape.confidence == pytest.approx(0.8)
    assert shape.has_polygon
    xmin, ymin, xmax, ymax = shape.bbox
    assert 0 <= xmin and 0 <= ymin
    assert xmax < 200 and ymax < 160


def test_a_mask_running_off_the_edge_is_clamped():
    mask = np.ones((100, 100), dtype=np.uint8)
    shape = postprocess.mask_to_shape(mask, "Door", 100, 100)
    assert shape is not None
    assert all(0 <= x < 100 and 0 <= y < 100 for x, y in shape.polygon)


def test_an_empty_mask_makes_no_shape():
    assert postprocess.mask_to_shape(np.zeros((40, 40), dtype=np.uint8), "x", 40, 40) is None


def test_an_already_polygonal_prediction_is_simplified_too():
    """Models that emit polygons directly are just as capable of emitting two
    thousand vertices, and they land in the same editor."""
    import math

    dense = [
        (int(128 + 90 * math.cos(t / 400 * 2 * math.pi)),
         int(128 + 90 * math.sin(t / 400 * 2 * math.pi)))
        for t in range(400)
    ]
    simplified = postprocess.simplify_polygon(dense)
    assert len(simplified) <= postprocess.MAX_VERTICES
    assert len(simplified) >= 8


def test_a_triangle_is_left_alone():
    triangle = [(0, 0), (50, 0), (25, 40)]
    assert postprocess.simplify_polygon(triangle) == triangle


# -------------------------------------------------- absence is not an error

def test_the_tool_list_is_empty_by_default_and_says_why():
    """A plain `pip install data-factory` has no heavy tools, and that is the
    normal, fully-usable state -- not a failure."""
    from data_factory.annotate.tools.registry import ToolRegistry

    payload = ToolRegistry().as_dict()
    assert payload["tools"] == []
    assert payload["detail"]


def test_invoking_a_tool_with_none_installed_raises_unavailable():
    from data_factory.annotate.tools.protocol import ToolUnavailable
    from data_factory.annotate.tools.registry import NullTool

    with pytest.raises(ToolUnavailable):
        NullTool().invoke("sam2.box", {})


def test_preparing_a_missing_tool_is_silent():
    """prepare() is called speculatively on every navigation; it must never
    interrupt annotating."""
    from data_factory.annotate.tools.registry import ToolRegistry

    assert ToolRegistry().prepare("sam2.box", "abc") is None

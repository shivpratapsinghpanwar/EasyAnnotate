"""The wire format between the annotation editor and the store.

Everything the browser sends passes through `Shape.from_dict`, so this is
untrusted-input parsing as much as it is serialisation. Two properties matter:
a round trip loses nothing, and a malformed payload fails loudly rather than
becoming a half-built shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_factory.core.annotations import voc
from data_factory.core.annotations.model import (
    WIRE_VERSION,
    Annotation,
    AnnotationError,
    Shape,
)

SHAPES = [
    Shape(label="Open", bbox=(10, 20, 60, 80)),
    Shape(label="Close", bbox=(0, 0, 5, 5), polygon=[(0, 0), (5, 0), (5, 5), (0, 5)]),
    Shape(label="door", bbox=(3, 4, 9, 9), pose="Frontal", truncated=1, difficult=1, occluded=1),
    Shape(label="car", bbox=(1, 1, 2, 2), confidence=0.8125),
    Shape(label="edge", bbox=(0, 0, 0, 0)),
]


class TestShapeRoundTrip:
    @pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.label + str(s.bbox))
    def test_a_shape_survives_a_round_trip_exactly(self, shape: Shape):
        assert Shape.from_dict(shape.as_dict()) == shape

    def test_every_field_is_carried(self):
        """Named individually, because a silently dropped flag is the failure
        this test exists to catch."""
        shape = Shape(
            label="thing", bbox=(1, 2, 3, 4), polygon=[(1, 2), (3, 2), (3, 4)],
            pose="Rear", truncated=1, difficult=1, occluded=1, confidence=0.5,
        )
        payload = shape.as_dict()
        for key in ("label", "bbox", "polygon", "pose", "truncated", "difficult",
                    "occluded", "confidence"):
            assert key in payload, key
        assert Shape.from_dict(payload) == shape

    def test_the_payload_is_json_safe(self):
        import json

        shape = Shape(label="x", bbox=(1, 2, 3, 4), polygon=[(1, 2), (3, 4), (5, 6)])
        assert Shape.from_dict(json.loads(json.dumps(shape.as_dict()))) == shape

    def test_confidence_defaults_to_none_for_hand_drawn_shapes(self):
        """None is how the review UI distinguishes a human label from a
        prediction, so the default matters."""
        assert Shape(label="x", bbox=(0, 0, 1, 1)).confidence is None

    def test_clamped_carries_confidence_through(self):
        """clamped() rebuilds the shape, so it is where a new field gets lost."""
        shape = Shape(label="x", bbox=(-5, -5, 500, 500), confidence=0.42)
        clamped = shape.clamped(100, 100)
        assert clamped.confidence == 0.42
        assert clamped.bbox == (0, 0, 99, 99)


class TestShapeValidation:
    def test_a_box_dragged_backwards_is_normalised_not_rejected(self):
        """Dragging up-and-left is ordinary mouse behaviour; the corners just
        arrive swapped."""
        shape = Shape.from_dict({"label": "x", "bbox": [60, 80, 10, 20]})
        assert shape.bbox == (10, 20, 60, 80)

    def test_floats_from_the_browser_are_rounded(self):
        shape = Shape.from_dict({"label": "x", "bbox": [10.4, 20.6, 60.5, 80.2]})
        assert shape.bbox == (10, 21, 60, 80)

    def test_bools_are_accepted_for_the_voc_flags(self):
        """A browser sends true/false; VOC stores 0/1."""
        shape = Shape.from_dict(
            {"label": "x", "bbox": [0, 0, 1, 1], "truncated": True, "difficult": False}
        )
        assert shape.truncated == 1 and shape.difficult == 0

    def test_an_empty_polygon_becomes_none(self):
        assert Shape.from_dict({"label": "x", "bbox": [0, 0, 1, 1], "polygon": []}).polygon is None

    @pytest.mark.parametrize(
        "payload,message",
        [
            ({"bbox": [0, 0, 1, 1]}, "non-empty label"),
            ({"label": "  ", "bbox": [0, 0, 1, 1]}, "non-empty label"),
            ({"label": "x"}, "four numbers"),
            ({"label": "x", "bbox": [0, 0, 1]}, "four numbers"),
            ({"label": "x", "bbox": [0, 0, 1, "a"]}, "non-number"),
            ({"label": "x", "bbox": [0, 0, 1, 1], "polygon": "nope"}, "list of points"),
            # match= is a regex, so the literal brackets need escaping.
            ({"label": "x", "bbox": [0, 0, 1, 1], "polygon": [[1, 2, 3]]}, r"must be \[x, y\]"),
            ({"label": "x", "bbox": [0, 0, 1, 1], "polygon": [["a", 2]]}, "non-number"),
            ({"label": "x", "bbox": [0, 0, 1, 1], "confidence": "high"}, "must be a number"),
        ],
    )
    def test_malformed_payloads_fail_loudly(self, payload, message):
        with pytest.raises(AnnotationError, match=message):
            Shape.from_dict(payload)

    def test_a_non_object_is_refused(self):
        with pytest.raises(AnnotationError, match="must be an object"):
            Shape.from_dict(["not", "a", "shape"])


class TestAnnotationRoundTrip:
    def test_an_annotation_survives_a_round_trip(self):
        annotation = Annotation(
            filename="a.jpg", width=120, height=90, depth=3,
            shapes=list(SHAPES), folder="pictures",
        )
        back = Annotation.from_dict(annotation.as_dict())
        assert back.filename == annotation.filename
        assert (back.width, back.height, back.depth) == (120, 90, 3)
        assert back.folder == "pictures"
        assert back.shapes == annotation.shapes

    def test_a_background_survives(self):
        """Zero shapes is meaningful -- a hard negative, not an empty result."""
        back = Annotation.from_dict(Annotation(filename="bg.jpg", width=10, height=10).as_dict())
        assert back.is_background

    def test_the_payload_carries_its_wire_version(self):
        assert Annotation(filename="a.jpg", width=1, height=1).as_dict()["wire"] == WIRE_VERSION

    def test_a_newer_wire_version_is_refused_rather_than_silently_degraded(self):
        payload = Annotation(filename="a.jpg", width=1, height=1).as_dict()
        payload["wire"] = WIRE_VERSION + 1
        with pytest.raises(AnnotationError, match="Upgrade Data Factory"):
            Annotation.from_dict(payload)


class TestThroughTheFileFormats:
    def test_wire_form_survives_a_voc_round_trip(self, tmp_path: Path):
        """The editor's payload and the file on disk must agree. Confidence is
        deliberately excluded -- it is review state, not a VOC field."""
        annotation = Annotation(
            filename="a.jpg", width=120, height=90,
            shapes=[
                Shape(label="Open", bbox=(10, 20, 60, 80), truncated=1, difficult=1),
                Shape(label="Close", bbox=(5, 5, 40, 40),
                      polygon=[(5, 5), (40, 5), (40, 40), (5, 40)]),
            ],
        )
        path = voc.write(Annotation.from_dict(annotation.as_dict()), tmp_path / "a.xml")
        back = voc.read(path)

        assert [s.label for s in back.shapes] == ["Open", "Close"]
        assert [s.bbox for s in back.shapes] == [(10, 20, 60, 80), (5, 5, 40, 40)]
        assert back.shapes[0].truncated == 1 and back.shapes[0].difficult == 1
        assert back.shapes[1].polygon == [(5, 5), (40, 5), (40, 40), (5, 40)]
        # And the whole thing survives one more trip through the wire form.
        assert Annotation.from_dict(back.as_dict()).shapes == back.shapes

"""Model-in-the-loop review: importing predictions, and judging them.

The tests that matter most here are about *not lying*. A prediction import
that matches nothing produces zero shapes and no error, which is
indistinguishable from a model that predicted nothing -- so the matching
report is tested harder than the happy path. Likewise the agreement metric:
a number that flatters the model is worse than no number.
"""

from __future__ import annotations

import json

import pytest
from conftest import make_image, write_voc

from data_factory.annotate import images, labels, projects, review, shapes
from data_factory.annotate.store import Store, StoreError
from data_factory.core.annotations import coco as coco_fmt
from data_factory.core.annotations import voc, yolo
from data_factory.core.annotations.model import Annotation, AnnotationError, Shape


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "annotations.db")
    yield s
    s.close()


@pytest.fixture()
def project(store):
    """Three images, two hand-drawn classes, nothing reviewed yet."""
    made = projects.create(store, "Doors")
    labels.adopt(store, made, ["Close", "Open"])
    for index in range(3):
        images.add(store, made, f"img{index}.jpg", width=200, height=200)
    return made


def yolo_run(folder, rows, names=("Close", "Open")):
    """A folder of YOLO predictions, as --save-conf writes them."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data.yaml").write_text(
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)), encoding="utf-8"
    )
    for stem, lines in rows.items():
        (folder / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


# ----------------------------------------------------- reading model output

class TestFormatsCarryConfidence:
    def test_yolo_reads_the_confidence_column(self, tmp_path):
        """`cls cx cy w h conf` is what --save-conf writes, and it used to
        raise outright -- the one file people most want to load."""
        path = tmp_path / "a.txt"
        path.write_text("1 0.5 0.5 0.2 0.2 0.87\n", encoding="utf-8")
        annotation = yolo.read(path, yolo.ClassList(["Close", "Open"]), 100, 100)
        assert len(annotation.shapes) == 1
        assert annotation.shapes[0].label == "Open"
        assert annotation.shapes[0].confidence == pytest.approx(0.87)

    def test_yolo_still_reads_a_plain_box(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        annotation = yolo.read(path, yolo.ClassList(["Close"]), 100, 100)
        assert annotation.shapes[0].confidence is None

    def test_yolo_reads_a_segmentation_line_with_confidence(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("0 0.1 0.1 0.4 0.1 0.25 0.4 0.9\n", encoding="utf-8")
        annotation = yolo.read(path, yolo.ClassList(["Close"]), 100, 100)
        assert annotation.shapes[0].polygon is not None
        assert annotation.shapes[0].confidence == pytest.approx(0.9)

    def test_a_genuinely_malformed_line_still_raises(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("0 0.5 0.5\n", encoding="utf-8")
        with pytest.raises(AnnotationError):
            yolo.read(path, yolo.ClassList(["Close"]), 100, 100)

    def test_coco_reads_a_score(self):
        document = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10],
                 "score": 0.62, "iscrowd": 0},
            ],
            "categories": [{"id": 1, "name": "Open"}],
        }
        annotations = coco_fmt.to_annotations(document)
        assert annotations[0].shapes[0].confidence == pytest.approx(0.62)

    def test_a_coco_results_list_is_recognised(self):
        """torchvision, mmdet and Ultralytics all emit a bare list."""
        results = [{"image_id": 7, "category_id": 0, "bbox": [1, 2, 10, 20], "score": 0.4}]
        assert coco_fmt.is_detection_results(results)
        annotations = coco_fmt.results_to_annotations(results, names={0: "Open"})
        assert annotations[0].shapes[0].label == "Open"
        assert annotations[0].shapes[0].confidence == pytest.approx(0.4)

    def test_a_results_list_given_to_the_dataset_reader_says_what_it_is(self):
        """It used to fail with AttributeError from inside a dict lookup."""
        with pytest.raises(AnnotationError, match="detection results"):
            coco_fmt.to_annotations([{"image_id": 1}])

    def test_voc_round_trips_a_score(self, tmp_path):
        path = tmp_path / "a.xml"
        voc.write(
            Annotation(filename="a.jpg", width=100, height=100,
                       shapes=[Shape(label="Open", bbox=(1, 1, 10, 10), confidence=0.5)]),
            path,
        )
        assert "<score>" in path.read_text(encoding="utf-8")
        assert voc.read(path).shapes[0].confidence == pytest.approx(0.5)

    def test_a_hand_drawn_voc_file_gains_no_score_element(self, tmp_path):
        """Ground truth must not acquire a fake confidence of 1.0."""
        path = tmp_path / "a.xml"
        voc.write(
            Annotation(filename="a.jpg", width=100, height=100,
                       shapes=[Shape(label="Open", bbox=(1, 1, 10, 10))]),
            path,
        )
        assert "<score>" not in path.read_text(encoding="utf-8")
        assert voc.read(path).shapes[0].confidence is None


# ------------------------------------------------------------- the import

class TestImport:
    def test_predictions_land_on_the_matching_images(self, store, project, tmp_path):
        folder = yolo_run(tmp_path / "run", {
            "img0": ["1 0.5 0.5 0.2 0.2 0.9"],
            "img1": ["0 0.5 0.5 0.3 0.3 0.7"],
        })
        report = review.import_predictions(store, project, folder)

        assert report["matched"] == 2
        assert report["shapes_imported"] == 2
        image = images.by_path(store, project, "img0.jpg")
        stored = shapes.listing(store, project, image)
        assert stored[0].origin == "model"
        assert stored[0].shape.confidence == pytest.approx(0.9)
        assert image.state == "predicted" or images.get(store, project, image.id).state == "predicted"

    def test_names_that_do_not_line_up_are_reported_not_swallowed(
        self, store, project, tmp_path
    ):
        """The dominant failure mode. Zero shapes and no error looks exactly
        like a model that predicted nothing."""
        folder = yolo_run(tmp_path / "run", {"IMG_0": ["0 0.5 0.5 0.2 0.2 0.9"]})
        report = review.import_predictions(store, project, folder)

        assert report["matched"] == 0
        assert report["unmatched_files"] == 1
        assert any("nothing matched" in note for note in report["notes"])
        assert any("matched no image" in note for note in report["notes"])

    def test_a_second_run_replaces_the_first_rather_than_doubling_it(
        self, store, project, tmp_path
    ):
        first = yolo_run(tmp_path / "r1", {"img0": ["0 0.5 0.5 0.2 0.2 0.5"]})
        review.import_predictions(store, project, first)
        second = yolo_run(tmp_path / "r2", {"img0": ["0 0.4 0.4 0.2 0.2 0.8"]})
        review.import_predictions(store, project, second)

        image = images.by_path(store, project, "img0.jpg")
        stored = shapes.listing(store, project, image)
        assert len(stored) == 1, "a second run doubled the boxes"
        assert stored[0].shape.confidence == pytest.approx(0.8)

    def test_hand_drawn_shapes_survive_an_import(self, store, project, tmp_path):
        image = images.by_path(store, project, "img0.jpg")
        shapes.replace_all(store, project, image, [Shape(label="Open", bbox=(1, 1, 30, 30))])

        folder = yolo_run(tmp_path / "run", {"img0": ["0 0.5 0.5 0.2 0.2 0.9"]})
        review.import_predictions(store, project, folder)

        stored = shapes.listing(store, project, images.get(store, project, image.id))
        origins = sorted(s.origin for s in stored)
        assert origins == ["human", "model"], "the hand-drawn shape was lost"

    def test_a_corrected_model_shape_survives_a_later_run(self, store, project, tmp_path):
        """Once a person has moved it, it is their work, not the model's."""
        folder = yolo_run(tmp_path / "r1", {"img0": ["0 0.5 0.5 0.2 0.2 0.5"]})
        review.import_predictions(store, project, folder)

        image = images.by_path(store, project, "img0.jpg")
        stored = shapes.listing(store, project, image)
        moved = Shape(label="Close", bbox=(5, 5, 60, 60))
        shapes.save(store, project, image, [(stored[0].id, moved)], base_version=image.version)
        assert shapes.listing(store, project, image)[0].is_touched

        second = yolo_run(tmp_path / "r2", {"img0": ["0 0.9 0.9 0.05 0.05 0.8"]})
        review.import_predictions(store, project, second)

        after = shapes.listing(store, project, images.get(store, project, image.id))
        assert any(s.is_touched for s in after), "the correction was thrown away"

    def test_reviewed_images_are_left_alone(self, store, project, tmp_path):
        image = images.by_path(store, project, "img0.jpg")
        images.set_state(store, project, image, "reviewed")

        folder = yolo_run(tmp_path / "run", {"img0": ["0 0.5 0.5 0.2 0.2 0.9"]})
        report = review.import_predictions(store, project, folder)

        assert report["skipped_reviewed"] == 1
        assert report["matched"] == 0
        assert shapes.listing(store, project, image) == []
        assert any("already reviewed" in note for note in report["notes"])

    def test_reviewed_images_can_be_included_deliberately(self, store, project, tmp_path):
        image = images.by_path(store, project, "img0.jpg")
        images.set_state(store, project, image, "reviewed")
        folder = yolo_run(tmp_path / "run", {"img0": ["0 0.5 0.5 0.2 0.2 0.9"]})

        report = review.import_predictions(store, project, folder, overwrite_reviewed=True)
        assert report["matched"] == 1

    def test_a_confidence_threshold_drops_the_weak_ones(self, store, project, tmp_path):
        folder = yolo_run(tmp_path / "run", {
            "img0": ["0 0.5 0.5 0.2 0.2 0.95", "1 0.2 0.2 0.1 0.1 0.10"],
        })
        report = review.import_predictions(store, project, folder, min_confidence=0.5)
        assert report["shapes_imported"] == 1
        assert report["dropped_below_threshold"] == 1

    def test_a_coco_results_file_imports(self, store, project, tmp_path):
        """Category ids with no names, matched against the project's order."""
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps([
                {"image_id": "img0", "category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.8},
            ]),
            encoding="utf-8",
        )
        report = review.import_predictions(store, project, path)
        assert report["matched"] == 1
        image = images.by_path(store, project, "img0.jpg")
        assert shapes.listing(store, project, image)[0].shape.label == "Open"

    def test_an_empty_prediction_set_is_refused_clearly(self, store, project, tmp_path):
        empty = tmp_path / "nothing"
        empty.mkdir()
        with pytest.raises(StoreError):
            review.import_predictions(store, project, empty)

    def test_the_run_is_recorded(self, store, project, tmp_path):
        folder = yolo_run(tmp_path / "run", {"img0": ["0 0.5 0.5 0.2 0.2 0.9"]})
        review.import_predictions(store, project, folder, name="epoch-40")
        recorded = review.runs(store, project)
        assert len(recorded) == 1
        assert recorded[0]["name"] == "epoch-40"
        assert recorded[0]["matched"] == 1


# ---------------------------------------------------------------- agreement

class TestAgreement:
    def test_unreviewed_predictions_do_not_count_as_agreement(
        self, store, project, tmp_path
    ):
        """Otherwise importing a fresh run makes the model look better, and
        the number climbs fastest when the least work has been done."""
        folder = yolo_run(tmp_path / "run", {
            "img0": ["0 0.5 0.5 0.2 0.2 0.9"],
            "img1": ["0 0.5 0.5 0.2 0.2 0.9"],
        })
        review.import_predictions(store, project, folder)

        figure = shapes.agreement(store, project)
        assert figure["agreement"] is None
        assert figure["unreviewed"] == 2

    def test_accepting_an_image_counts_its_predictions_as_agreement(
        self, store, project, tmp_path
    ):
        folder = yolo_run(tmp_path / "run", {"img0": ["0 0.5 0.5 0.2 0.2 0.9"]})
        review.import_predictions(store, project, folder)

        image = images.by_path(store, project, "img0.jpg")
        review.accept_all(store, project, image)

        figure = shapes.agreement(store, project)
        assert figure["model_shapes"] == 1
        assert figure["corrected"] == 0
        assert figure["agreement"] == 1.0

    def test_accepting_keeps_the_shape_a_prediction(self, store, project, tmp_path):
        """Flipping origin to 'human' on accept would empty the denominator."""
        folder = yolo_run(tmp_path / "run", {"img0": ["0 0.5 0.5 0.2 0.2 0.9"]})
        review.import_predictions(store, project, folder)
        image = images.by_path(store, project, "img0.jpg")
        review.accept_all(store, project, image)

        stored = shapes.listing(store, project, images.get(store, project, image.id))
        assert stored[0].origin == "model"

    def test_moving_a_box_counts_as_a_correction(self, store, project, tmp_path):
        folder = yolo_run(tmp_path / "run", {"img0": ["0 0.5 0.5 0.2 0.2 0.9"]})
        review.import_predictions(store, project, folder)

        image = images.by_path(store, project, "img0.jpg")
        stored = shapes.listing(store, project, image)
        shapes.save(
            store, project, image,
            [(stored[0].id, Shape(label="Close", bbox=(1, 1, 50, 50)))],
            base_version=image.version,
        )
        images.set_state(store, project, images.get(store, project, image.id), "reviewed")

        figure = shapes.agreement(store, project)
        assert figure["adjusted"] == 1
        assert figure["agreement"] == 0.0

    def test_deleting_a_false_positive_counts_as_a_correction(
        self, store, project, tmp_path
    ):
        """The commonest correction there is. Counting only surviving shapes
        would score a model that hallucinates boxes at 100%."""
        folder = yolo_run(tmp_path / "run", {
            "img0": ["0 0.5 0.5 0.2 0.2 0.9", "1 0.2 0.2 0.1 0.1 0.8"],
        })
        review.import_predictions(store, project, folder)

        image = images.by_path(store, project, "img0.jpg")
        stored = shapes.listing(store, project, image)
        # Keep one, delete the other.
        shapes.save(
            store, project, image,
            [(stored[0].id, stored[0].shape)],
            base_version=image.version,
        )
        images.set_state(store, project, images.get(store, project, image.id), "reviewed")

        figure = shapes.agreement(store, project)
        assert figure["model_shapes"] == 2
        assert figure["deleted"] == 1
        assert figure["corrected"] == 1
        assert figure["agreement"] == 0.5

    def test_a_model_that_predicted_nothing_useful_scores_zero(
        self, store, project, tmp_path
    ):
        folder = yolo_run(tmp_path / "run", {
            "img0": ["0 0.1 0.1 0.05 0.05 0.9", "1 0.2 0.2 0.05 0.05 0.9"],
        })
        review.import_predictions(store, project, folder)
        image = images.by_path(store, project, "img0.jpg")
        shapes.save(store, project, image, [], base_version=image.version)
        images.set_state(store, project, images.get(store, project, image.id), "reviewed")

        assert shapes.agreement(store, project)["agreement"] == 0.0


# ------------------------------------------------------------------ schema

def test_the_ladder_adds_the_predicted_count_column(tmp_path):
    from data_factory.annotate.store import SCHEMA_VERSION

    store = Store(tmp_path / "fresh.db")
    assert store.one("PRAGMA user_version")[0] == SCHEMA_VERSION
    columns = [r["name"] for r in store.query("PRAGMA table_info(image)")]
    assert "predicted_count" in columns
    store.close()

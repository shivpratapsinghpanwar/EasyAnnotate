"""The annotation HTTP layer.

Two kinds of test here. The functional ones follow the flow the product
promises -- drop a folder, draw a box, name the class as you draw it, close the
browser, come back to the right image. The rest are refusals, written in the
same style as ``test_service.py``, because these routes reach files and are
about to reach other people's projects.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="optional: pip install 'data-factory[web]'")

from fastapi.testclient import TestClient

from data_factory.annotate.store import LOCAL_OWNER


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.delenv("DATAFACTORY_API_KEY", raising=False)

    import importlib

    from data_factory.service import app as app_module
    from data_factory.service import deps

    deps.reset_stores()
    importlib.reload(app_module)
    with TestClient(app_module.app) as test_client:
        test_client.app_module = app_module
        yield test_client
    deps.reset_stores()


def zip_of(folder: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(folder).as_posix())
    return buffer.getvalue()


def upload(client, folder: Path) -> str:
    response = client.post(
        "/api/datasets", files={"file": ("d.zip", zip_of(folder), "application/zip")}
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def make_project(client, name="Doors") -> str:
    response = client.post("/api/projects", json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def box(label, xmin, ymin, xmax, ymax, **extra):
    return {"label": label, "bbox": [xmin, ymin, xmax, ymax], **extra}


# ------------------------------------------------------------- the flow

class TestAnnotatingFlow:
    def test_a_project_starts_with_nothing_to_configure(self, client):
        payload = client.post("/api/projects", json={"name": "Doors"}).json()
        detail = client.get(f"/api/projects/{payload['id']}").json()
        assert detail["labels"] == [], "no schema step: labels arrive by drawing"
        assert detail["counts"]["total"] == 0

    def test_drawing_a_box_creates_its_label(self, client, clean_dataset):
        dataset_id = upload(client, clean_dataset)
        project = client.post(
            f"/api/datasets/{dataset_id}/promote", json={"import_annotations": False}
        ).json()
        image = client.get(f"/api/projects/{project['id']}/resume").json()["image"]

        saved = client.put(
            f"/api/projects/{project['id']}/images/{image['id']}/shapes",
            json={"version": image["version"], "shapes": [box("Ajar", 5, 5, 40, 40)]},
        )
        assert saved.status_code == 200, saved.text
        body = saved.json()
        assert body["labels_created"] == ["Ajar"]
        assert body["inserted"] == 1
        assert [label["name"] for label in body["labels"]] == ["Ajar"]

    def test_a_saved_shape_comes_back(self, client, clean_dataset):
        dataset_id = upload(client, clean_dataset)
        project = client.post(
            f"/api/datasets/{dataset_id}/promote", json={"import_annotations": False}
        ).json()
        image = client.get(f"/api/projects/{project['id']}/resume").json()["image"]
        client.put(
            f"/api/projects/{project['id']}/images/{image['id']}/shapes",
            json={"version": 0, "shapes": [box("Open", 5, 5, 40, 40)]},
        )
        again = client.get(f"/api/projects/{project['id']}/images/{image['id']}").json()
        assert len(again["shapes"]) == 1
        assert again["shapes"][0]["label"] == "Open"
        assert again["shapes"][0]["origin"] == "human"
        assert again["version"] == 1

    def test_resume_lands_on_the_first_unfinished_image(self, client, clean_dataset):
        dataset_id = upload(client, clean_dataset)
        project = client.post(
            f"/api/datasets/{dataset_id}/promote", json={"import_annotations": False}
        ).json()
        pid = project["id"]

        first = client.get(f"/api/projects/{pid}/resume").json()["image"]
        client.patch(f"/api/projects/{pid}/images/{first['id']}", json={"state": "reviewed"})

        second = client.get(f"/api/projects/{pid}/resume").json()["image"]
        assert second["id"] != first["id"], "resume must move on, not restart"
        assert second["seq"] > first["seq"]

    def test_shapes_are_clamped_to_the_image_server_side(self, client, clean_dataset):
        dataset_id = upload(client, clean_dataset)
        project = client.post(
            f"/api/datasets/{dataset_id}/promote", json={"import_annotations": False}
        ).json()
        image = client.get(f"/api/projects/{project['id']}/resume").json()["image"]

        client.put(
            f"/api/projects/{project['id']}/images/{image['id']}/shapes",
            json={"version": 0, "shapes": [box("Open", -50, -50, 99999, 99999)]},
        )
        stored = client.get(
            f"/api/projects/{project['id']}/images/{image['id']}"
        ).json()["shapes"][0]
        xmin, ymin, xmax, ymax = stored["bbox"]
        assert xmin >= 0 and ymin >= 0
        assert xmax < image["width"] and ymax < image["height"]

    def test_a_polygon_round_trips(self, client, clean_dataset):
        dataset_id = upload(client, clean_dataset)
        project = client.post(
            f"/api/datasets/{dataset_id}/promote", json={"import_annotations": False}
        ).json()
        image = client.get(f"/api/projects/{project['id']}/resume").json()["image"]
        polygon = [[10, 10], [50, 10], [30, 45]]
        client.put(
            f"/api/projects/{project['id']}/images/{image['id']}/shapes",
            json={"version": 0, "shapes": [box("Open", 10, 10, 50, 45, polygon=polygon)]},
        )
        stored = client.get(
            f"/api/projects/{project['id']}/images/{image['id']}"
        ).json()["shapes"][0]
        assert stored["polygon"] == polygon

    def test_editing_keeps_shape_identity(self, client, clean_dataset):
        """A save is a diff. Moving a box must not delete and recreate it."""
        dataset_id = upload(client, clean_dataset)
        project = client.post(
            f"/api/datasets/{dataset_id}/promote", json={"import_annotations": False}
        ).json()
        pid, image = project["id"], client.get(f"/api/projects/{project['id']}/resume").json()["image"]

        client.put(
            f"/api/projects/{pid}/images/{image['id']}/shapes",
            json={"version": 0, "shapes": [box("Open", 5, 5, 40, 40)]},
        )
        stored = client.get(f"/api/projects/{pid}/images/{image['id']}").json()["shapes"][0]

        moved = dict(stored)
        moved["bbox"] = [9, 9, 44, 44]
        result = client.put(
            f"/api/projects/{pid}/images/{image['id']}/shapes",
            json={"version": 1, "shapes": [moved]},
        ).json()

        assert result["updated"] == 1 and result["inserted"] == 0 and result["deleted"] == 0
        after = client.get(f"/api/projects/{pid}/images/{image['id']}").json()["shapes"][0]
        assert after["id"] == stored["id"], "the shape kept its identity"


# -------------------------------------------------------------- conflicts

class TestConcurrentEdits:
    def test_a_stale_save_is_refused_with_the_current_state(self, client, clean_dataset):
        dataset_id = upload(client, clean_dataset)
        project = client.post(
            f"/api/datasets/{dataset_id}/promote", json={"import_annotations": False}
        ).json()
        pid = project["id"]
        image = client.get(f"/api/projects/{pid}/resume").json()["image"]

        client.put(
            f"/api/projects/{pid}/images/{image['id']}/shapes",
            json={"version": 0, "shapes": [box("Open", 5, 5, 40, 40)]},
        )
        # A second tab still believes it is on version 0.
        late = client.put(
            f"/api/projects/{pid}/images/{image['id']}/shapes",
            json={"version": 0, "shapes": [box("Close", 1, 1, 20, 20)]},
        )
        assert late.status_code == 409
        detail = late.json()["detail"]
        assert "version" in detail["message"]
        # The response carries what the server actually has, so the editor can
        # reconcile rather than making someone redo the work.
        assert detail["current"]["shapes"][0]["label"] == "Open"

    def test_omitting_the_version_forces_the_save(self, client, clean_dataset):
        dataset_id = upload(client, clean_dataset)
        project = client.post(
            f"/api/datasets/{dataset_id}/promote", json={"import_annotations": False}
        ).json()
        pid = project["id"]
        image = client.get(f"/api/projects/{pid}/resume").json()["image"]
        client.put(
            f"/api/projects/{pid}/images/{image['id']}/shapes",
            json={"version": 0, "shapes": [box("Open", 5, 5, 40, 40)]},
        )
        forced = client.put(
            f"/api/projects/{pid}/images/{image['id']}/shapes",
            json={"shapes": [box("Close", 1, 1, 20, 20)]},
        )
        assert forced.status_code == 200


# ----------------------------------------------------------------- labels

class TestLabels:
    def test_renaming_a_label_does_not_move_its_index(self, client):
        pid = make_project(client)
        client.post(f"/api/projects/{pid}/labels", json={"name": "Cosed"})
        client.post(f"/api/projects/{pid}/labels", json={"name": "Open"})
        client.patch(f"/api/projects/{pid}/labels/0", json={"name": "Closed"})

        payload = client.get(f"/api/projects/{pid}/labels").json()
        assert payload["class_list"] == ["Closed", "Open"]

    def test_retiring_a_label_keeps_its_export_slot(self, client):
        pid = make_project(client)
        for name in ("Closed", "Handle", "Open"):
            client.post(f"/api/projects/{pid}/labels", json={"name": name})
        client.delete(f"/api/projects/{pid}/labels/1")

        payload = client.get(f"/api/projects/{pid}/labels").json()
        assert [label["name"] for label in payload["labels"]] == ["Closed", "Open"]
        assert payload["class_list"] == ["Closed", "Handle", "Open"]

    def test_creating_a_label_twice_is_idempotent(self, client):
        pid = make_project(client)
        first = client.post(f"/api/projects/{pid}/labels", json={"name": "Open"}).json()
        second = client.post(f"/api/projects/{pid}/labels", json={"name": "  Open "}).json()
        assert first["idx"] == second["idx"] == 0

    def test_a_colour_that_is_not_a_colour_is_refused(self, client):
        pid = make_project(client)
        client.post(f"/api/projects/{pid}/labels", json={"name": "Open"})
        response = client.patch(
            f"/api/projects/{pid}/labels/0", json={"colour": "javascript:alert(1)"}
        )
        assert response.status_code in (400, 422)

    def test_label_colours_are_rgb_not_bgr(self, client):
        """A guard against reusing visualize.PALETTE, which is BGR for OpenCV
        and would render every class with red and blue swapped."""
        from data_factory.annotate.labels import PALETTE
        from data_factory.core.ops.visualize import PALETTE as BGR

        overlap = {c.lower() for c in PALETTE} & {
            "#%02x%02x%02x" % tuple(c) for c in BGR
        }
        assert not overlap


# ------------------------------------------------------------- promotion

class TestPromotion:
    def test_promoting_imports_the_existing_annotations(self, client, clean_dataset):
        dataset_id = upload(client, clean_dataset)
        payload = client.post(f"/api/datasets/{dataset_id}/promote", json={}).json()

        assert payload["images"] == 7
        assert payload["shapes_imported"] == 6
        assert sorted(label["name"] for label in payload["labels"]) == ["Close", "Open"]

    def test_hand_made_annotations_arrive_reviewed(self, client, clean_dataset):
        """They have no confidence score, so a person drew them somewhere else."""
        dataset_id = upload(client, clean_dataset)
        payload = client.post(f"/api/datasets/{dataset_id}/promote", json={}).json()
        assert payload["reviewed"] == 6
        assert payload["predicted"] == 0

    def test_a_folder_of_bare_images_promotes_fine(self, client, tmp_path):
        from conftest import make_image

        folder = tmp_path / "bare"
        folder.mkdir()
        for i in range(3):
            make_image(folder / f"{i}.jpg", i)

        dataset_id = upload(client, folder)
        payload = client.post(f"/api/datasets/{dataset_id}/promote", json={}).json()
        assert payload["images"] == 3
        assert payload["shapes_imported"] == 0
        assert any("no annotations imported" in w for w in payload["warnings"])

    def test_an_unmatched_annotation_is_reported_not_swallowed(self, client, dataset):
        dataset_id = upload(client, dataset)
        payload = client.post(f"/api/datasets/{dataset_id}/promote", json={}).json()
        assert any("did not and start empty" in w for w in payload["warnings"])

    def test_promoting_an_empty_dataset_is_refused(self, client, tmp_path):
        folder = tmp_path / "labels_only"
        folder.mkdir()
        (folder / "a.xml").write_text("<annotation></annotation>", encoding="utf-8")
        dataset_id = upload(client, folder)
        response = client.post(f"/api/datasets/{dataset_id}/promote", json={})
        assert response.status_code == 400


# ----------------------------------------------------------------- media

class TestMedia:
    def test_the_image_and_its_thumbnail_are_served(self, client, clean_dataset):
        dataset_id = upload(client, clean_dataset)
        project = client.post(f"/api/datasets/{dataset_id}/promote", json={}).json()
        image = client.get(f"/api/projects/{project['id']}/resume").json()["image"]

        full = client.get(f"/api/projects/{project['id']}/images/{image['id']}/file")
        assert full.status_code == 200
        assert full.headers["content-type"] == "image/jpeg"

        thumb = client.get(f"/api/projects/{project['id']}/images/{image['id']}/thumb")
        assert thumb.status_code == 200
        assert thumb.headers["content-type"] == "image/jpeg"

    def test_a_small_image_is_served_as_its_own_thumbnail(self, client, clean_dataset):
        """The fixtures are 120x90. Re-encoding those costs a JPEG generation
        and produces a *larger* file, so the work is skipped entirely."""
        dataset_id = upload(client, clean_dataset)
        project = client.post(f"/api/datasets/{dataset_id}/promote", json={}).json()
        image = client.get(f"/api/projects/{project['id']}/resume").json()["image"]

        full = client.get(f"/api/projects/{project['id']}/images/{image['id']}/file")
        thumb = client.get(f"/api/projects/{project['id']}/images/{image['id']}/thumb")
        assert thumb.content == full.content

    def test_a_large_image_is_actually_downscaled(self, client, tmp_path):
        from conftest import make_image
        from data_factory.core import imaging

        folder = tmp_path / "big"
        folder.mkdir()
        make_image(folder / "big.jpg", 1, size=(1200, 900))

        dataset_id = upload(client, folder)
        project = client.post(f"/api/datasets/{dataset_id}/promote", json={}).json()
        image = client.get(f"/api/projects/{project['id']}/resume").json()["image"]

        thumb = client.get(f"/api/projects/{project['id']}/images/{image['id']}/thumb")
        assert thumb.status_code == 200

        out = tmp_path / "thumb.jpg"
        out.write_bytes(thumb.content)
        height, width = imaging.load(out).shape[:2]
        assert max(width, height) <= 200
        assert len(thumb.content) < (folder / "big.jpg").stat().st_size

    def test_a_thumbnail_is_cached_and_matches_on_the_second_call(self, client, clean_dataset):
        dataset_id = upload(client, clean_dataset)
        project = client.post(f"/api/datasets/{dataset_id}/promote", json={}).json()
        image = client.get(f"/api/projects/{project['id']}/resume").json()["image"]
        url = f"/api/projects/{project['id']}/images/{image['id']}/thumb"
        assert client.get(url).content == client.get(url).content


# ------------------------------------------------------------- the refusals

class TestSecurity:
    @pytest.mark.parametrize(
        "bad_id", ["..", "%2e%2e", "%2e%2e%2f", "not-hex", "a" * 23, "a" * 25, "/etc/passwd"]
    )
    def test_project_ids_that_are_not_ids_are_rejected(self, client, bad_id):
        response = client.get(f"/api/projects/{bad_id}")
        assert response.status_code in (400, 404, 405), f"{bad_id} -> {response.status_code}"

    def test_another_owners_project_is_not_visible(self, client, clean_dataset):
        """The scoping is real, not just present in the signature."""
        from data_factory.service.deps import principal

        pid = make_project(client, "alice's work")

        client.app_module.app.dependency_overrides[principal] = lambda: "bob"
        try:
            assert client.get(f"/api/projects/{pid}").status_code == 404
            assert client.get("/api/projects").json()["projects"] == []
            assert client.delete(f"/api/projects/{pid}").status_code == 404
        finally:
            client.app_module.app.dependency_overrides.pop(principal, None)

        # And it is still there for its actual owner.
        assert client.get(f"/api/projects/{pid}").status_code == 200

    def test_an_image_from_another_project_is_not_reachable(self, client, clean_dataset):
        dataset_id = upload(client, clean_dataset)
        mine = client.post(f"/api/datasets/{dataset_id}/promote", json={}).json()
        image = client.get(f"/api/projects/{mine['id']}/resume").json()["image"]

        other = make_project(client, "other")
        response = client.get(f"/api/projects/{other}/images/{image['id']}")
        assert response.status_code == 404

    def test_a_shape_with_no_label_is_refused(self, client, clean_dataset):
        dataset_id = upload(client, clean_dataset)
        project = client.post(
            f"/api/datasets/{dataset_id}/promote", json={"import_annotations": False}
        ).json()
        image = client.get(f"/api/projects/{project['id']}/resume").json()["image"]
        response = client.put(
            f"/api/projects/{project['id']}/images/{image['id']}/shapes",
            json={"version": 0, "shapes": [{"label": "  ", "bbox": [1, 1, 2, 2]}]},
        )
        assert response.status_code == 400

    def test_a_malformed_bbox_is_refused(self, client, clean_dataset):
        dataset_id = upload(client, clean_dataset)
        project = client.post(
            f"/api/datasets/{dataset_id}/promote", json={"import_annotations": False}
        ).json()
        image = client.get(f"/api/projects/{project['id']}/resume").json()["image"]
        response = client.put(
            f"/api/projects/{project['id']}/images/{image['id']}/shapes",
            json={"version": 0, "shapes": [{"label": "Open", "bbox": ["a", 1, 2, 3]}]},
        )
        assert response.status_code == 400

    def test_the_annotation_database_is_not_inside_a_dataset_sandbox(self, client):
        """It must never be swept on TTL, zipped into a download, or served."""
        from data_factory.service.deps import store_path

        workspace = Path(client.app_module.WORKSPACE.root)
        assert workspace not in store_path().parents
        assert store_path().parent == workspace.parent

    def test_an_api_key_gates_the_annotation_routes_too(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws2"))
        monkeypatch.setenv("DATAFACTORY_API_KEY", "s3cret")

        import importlib

        from data_factory.service import app as app_module
        from data_factory.service import deps

        deps.reset_stores()
        importlib.reload(app_module)
        with TestClient(app_module.app) as guarded:
            assert guarded.post("/api/projects", json={"name": "x"}).status_code == 401
            allowed = guarded.post(
                "/api/projects", json={"name": "x"}, headers={"X-API-Key": "s3cret"}
            )
            assert allowed.status_code == 201
        deps.reset_stores()


# ------------------------------------------------------------------ export

def test_a_project_exports_through_the_operation(client, clean_dataset, tmp_path):
    """The one bridge back to the batch world, end to end over HTTP + CLI."""
    from data_factory.core import registry
    from data_factory.core.types import Context
    from data_factory.service.deps import store_path

    import data_factory.core.ops  # noqa: F401

    dataset_id = upload(client, clean_dataset)
    project = client.post(f"/api/datasets/{dataset_id}/promote", json={}).json()

    operation = registry.get("annotate.export")
    result = operation.func(
        operation.bind(
            {
                "database": str(store_path()),
                "project": project["id"],
                "output": str(tmp_path / "exported"),
                "to": "yolo",
                "owner": LOCAL_OWNER,
                "copy_images": False,
            }
        ),
        Context(),
    )
    assert result.ok, result.errors
    assert result.counts["objects"] == 6
    assert (tmp_path / "exported" / "data.yaml").is_file()

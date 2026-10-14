"""Heavy tools over HTTP, with and without one installed.

The most important test in this file is the one where nothing is installed.
``GET /api/tools`` returning an empty list is what a plain
``pip install data-factory`` produces, the editor has to be fully usable in
that state, and it is the state every other test in the suite runs in.

The rest drive the *real* fake sidecar as a separate process, so the boundary
-- framing, correlation, timeouts, crashes -- is exercised end to end without
installing PyTorch.
"""

from __future__ import annotations

import io
import zipfile

import pytest
from conftest import make_image

pytest.importorskip("fastapi", reason="optional: pip install 'data-factory[web]'")

from fastapi.testclient import TestClient

FAKE_MODULE = "data_factory.annotate.tools.fake"


def build_client(tmp_path, monkeypatch, *, tools: str = ""):
    monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.delenv("DATAFACTORY_API_KEY", raising=False)
    if tools:
        monkeypatch.setenv("DATAFACTORY_TOOLS_MODULE", tools)
    else:
        monkeypatch.delenv("DATAFACTORY_TOOLS_MODULE", raising=False)

    import importlib

    from data_factory.annotate.tools.registry import REGISTRY
    from data_factory.service import app as app_module
    from data_factory.service import deps

    deps.reset_stores()
    REGISTRY.close()          # forget any backend a previous test discovered
    importlib.reload(app_module)
    client = TestClient(app_module.app)
    client.app_module = app_module
    return client


@pytest.fixture
def plain(tmp_path, monkeypatch):
    """No tools installed -- the default everyone gets."""
    with build_client(tmp_path, monkeypatch) as client:
        yield client
    from data_factory.annotate.tools.registry import REGISTRY
    from data_factory.service import deps

    REGISTRY.close()
    deps.reset_stores()


@pytest.fixture
def with_fake(tmp_path, monkeypatch):
    with build_client(tmp_path, monkeypatch, tools=FAKE_MODULE) as client:
        yield client
    from data_factory.annotate.tools.registry import REGISTRY
    from data_factory.service import deps

    REGISTRY.close()
    deps.reset_stores()


def seed(client, tmp_path, name="tools") -> tuple[str, str]:
    """A promoted project with one image. Returns (project_id, image_id)."""
    folder = tmp_path / name
    folder.mkdir(exist_ok=True)
    make_image(folder / "a.jpg", 3, size=(320, 240))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.write(folder / "a.jpg", "a.jpg")

    dataset = client.post(
        "/api/datasets", files={"file": ("d.zip", buffer.getvalue(), "application/zip")}
    ).json()
    project = client.post(f"/api/datasets/{dataset['id']}/promote", json={}).json()
    image = client.get(f"/api/projects/{project['id']}/resume").json()["image"]
    return project["id"], image["id"]


# ------------------------------------------------- absence is not an error

class TestNothingInstalled:
    def test_the_tool_list_is_empty_and_says_why(self, plain):
        payload = plain.get("/api/tools").json()
        assert payload["tools"] == []
        assert payload["detail"], "an empty list should still explain itself"

    def test_invoking_a_tool_is_a_503_not_a_500(self, plain, tmp_path):
        project_id, image_id = seed(plain, tmp_path)
        response = plain.post(
            f"/api/projects/{project_id}/tools/sam2.box",
            json={"image_id": image_id, "params": {"box": [1, 1, 40, 40]}},
        )
        assert response.status_code == 503

    def test_the_editor_still_works_with_no_tools(self, plain, tmp_path):
        """The whole point of the boundary."""
        project_id, image_id = seed(plain, tmp_path)
        saved = plain.put(
            f"/api/projects/{project_id}/images/{image_id}/shapes",
            json={"version": 0, "shapes": [{"label": "Door", "bbox": [5, 5, 60, 60]}]},
        )
        assert saved.status_code == 200
        assert saved.json()["inserted"] == 1


# ------------------------------------------------------ with a real sidecar

class TestWithASidecar:
    def test_the_tool_is_listed(self, with_fake):
        payload = with_fake.get("/api/tools").json()
        names = [tool["name"] for tool in payload["tools"]]
        assert "fake.box" in names, payload
        assert payload["tools"][0]["health"]["state"] == "ready"

    def test_a_cold_image_returns_not_ready_with_a_retry_hint(self, with_fake, tmp_path):
        """Exactly how the real SAM tool behaves before an image is encoded:
        not an error, just not yet."""
        project_id, image_id = seed(with_fake, tmp_path)
        response = with_fake.post(
            f"/api/projects/{project_id}/tools/fake.box",
            json={"image_id": image_id, "params": {"box": [10, 10, 80, 80]}, "label": "Door"},
        )
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["retry_after_ms"] > 0

    def test_a_prepared_image_returns_shapes(self, with_fake, tmp_path):
        from data_factory.annotate.tools.registry import REGISTRY

        project_id, image_id = seed(with_fake, tmp_path)
        REGISTRY.prepare("fake.box", image_id)

        response = with_fake.post(
            f"/api/projects/{project_id}/tools/fake.box",
            json={"image_id": image_id, "params": {"box": [10, 10, 80, 80]}, "label": "Door"},
        )
        assert response.status_code == 200, response.text
        shapes = response.json()["shapes"]
        assert len(shapes) == 1
        assert shapes[0]["label"] == "Door"
        assert shapes[0]["origin"] == "model"
        assert shapes[0]["polygon"], "a tool shape should carry its outline"

    def test_the_label_is_created_by_using_it(self, with_fake, tmp_path):
        from data_factory.annotate.tools.registry import REGISTRY

        project_id, image_id = seed(with_fake, tmp_path)
        REGISTRY.prepare("fake.box", image_id)
        with_fake.post(
            f"/api/projects/{project_id}/tools/fake.box",
            json={"image_id": image_id, "params": {"box": [10, 10, 80, 80]}, "label": "Hinge"},
        )
        labels = with_fake.get(f"/api/projects/{project_id}/labels").json()["labels"]
        assert [label["name"] for label in labels] == ["Hinge"]

    def test_a_tool_that_raises_does_not_leak_a_traceback(self, with_fake, tmp_path):
        from data_factory.annotate.tools.registry import REGISTRY

        project_id, image_id = seed(with_fake, tmp_path)
        REGISTRY.prepare("fake.box", image_id)
        response = with_fake.post(
            f"/api/projects/{project_id}/tools/fake.box",
            json={"image_id": image_id, "params": {"box": [1, 1, 9, 9], "behaviour": "raise"}},
        )
        assert response.status_code in (400, 502)
        assert "Traceback" not in response.text
        assert "blew up" not in response.text or response.status_code == 400

    def test_a_tool_that_dies_leaves_the_editor_working(self, with_fake, tmp_path):
        """A CUDA OOM kills the sidecar, not the annotation session."""
        from data_factory.annotate.tools.registry import REGISTRY

        project_id, image_id = seed(with_fake, tmp_path)
        REGISTRY.prepare("fake.box", image_id)

        with_fake.post(
            f"/api/projects/{project_id}/tools/fake.box",
            json={"image_id": image_id, "params": {"box": [1, 1, 9, 9], "behaviour": "exit"}},
        )
        # Whatever that returned, drawing must still work.
        saved = with_fake.put(
            f"/api/projects/{project_id}/images/{image_id}/shapes",
            json={"version": 0, "shapes": [{"label": "Door", "bbox": [5, 5, 60, 60]}]},
        )
        assert saved.status_code == 200
        assert saved.json()["inserted"] == 1

    def test_an_unknown_image_is_refused(self, with_fake, tmp_path):
        project_id, _ = seed(with_fake, tmp_path)
        response = with_fake.post(
            f"/api/projects/{project_id}/tools/fake.box",
            json={"image_id": "0" * 24, "params": {"box": [1, 1, 9, 9]}},
        )
        assert response.status_code == 404

    def test_another_owners_project_is_refused(self, with_fake, tmp_path):
        from data_factory.service.deps import principal

        project_id, image_id = seed(with_fake, tmp_path)
        with_fake.app_module.app.dependency_overrides[principal] = lambda: "bob"
        try:
            response = with_fake.post(
                f"/api/projects/{project_id}/tools/fake.box",
                json={"image_id": image_id, "params": {"box": [1, 1, 9, 9]}},
            )
            assert response.status_code == 404
        finally:
            with_fake.app_module.app.dependency_overrides.pop(principal, None)


# --------------------------------------------------------- shape conversion

class TestShapeConversion:
    def test_a_mask_reply_becomes_a_simplified_polygon(self):
        import numpy as np

        from data_factory.service.routers.tools import _shapes_from

        mask = np.zeros((120, 160), dtype=np.uint8)
        mask[20:100, 30:130] = 1
        made = _shapes_from({"mask": mask, "score": 0.7}, "Door", 160, 120)
        assert len(made) == 1
        assert made[0].label == "Door"
        assert len(made[0].polygon) == 4
        assert made[0].confidence == pytest.approx(0.7)

    def test_a_box_reply_becomes_a_box(self):
        from data_factory.service.routers.tools import _shapes_from

        made = _shapes_from({"boxes": [[1, 2, 30, 40]]}, "Door", 100, 100)
        assert made[0].bbox == (1, 2, 30, 40)
        assert not made[0].has_polygon

    def test_a_dense_polygon_reply_is_simplified(self):
        import math

        from data_factory.service.routers.tools import _shapes_from

        dense = [
            [int(120 + 80 * math.cos(t / 300 * 2 * math.pi)),
             int(120 + 80 * math.sin(t / 300 * 2 * math.pi))]
            for t in range(300)
        ]
        made = _shapes_from({"polygons": [dense]}, "Door", 240, 240)
        assert len(made[0].polygon) <= 80, "an unsimplified outline reached the editor"

    def test_a_malformed_shape_is_dropped_not_fatal(self):
        """A sidecar may be a third party's code; its output is untrusted."""
        from data_factory.service.routers.tools import _shapes_from

        made = _shapes_from(
            {"shapes": [{"label": "", "bbox": [1, 1, 2, 2]}, "not a dict", {"bbox": "nope"}]},
            "",
            100,
            100,
        )
        assert made == []

    def test_a_reply_that_is_not_a_dict_is_ignored(self):
        from data_factory.service.routers.tools import _shapes_from

        assert _shapes_from(["surprise"], "Door", 100, 100) == []

"""The hosted service, and especially the boundary a public host depends on.

Most of these are security tests. The functional ones matter, but the ones that
matter *most* are the refusals: a dataset-preparation service that accepts a
filesystem path from a browser is an arbitrary-read primitive, and a zip
extractor that trusts member names is an arbitrary-write one.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="optional: pip install 'data-factory[web]'")

from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    """A service whose workspace is a throwaway directory."""
    monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.delenv("DATAFACTORY_API_KEY", raising=False)

    import importlib

    from data_factory.service import app as app_module

    importlib.reload(app_module)
    with TestClient(app_module.app) as test_client:
        test_client.app_module = app_module
        yield test_client


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


def wait_for(client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["state"] in ("finished", "failed", "cancelled"):
            return payload
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


class TestSecurity:
    """The refusals. Each of these is a hole if it stops holding."""

    def test_a_path_parameter_is_refused_outright(self, client, clean_dataset: Path):
        """Not sanitised -- refused. Sanitising invites an encoding arms race."""
        dataset_id = upload(client, clean_dataset)
        response = client.post(
            f"/api/datasets/{dataset_id}/run/convert.dataset",
            json={"params": {"folder": "/etc", "to": "yolo"}},
        )
        assert response.status_code == 400
        assert "cannot be set directly" in response.json()["detail"]

    def test_an_output_path_cannot_be_redirected(self, client, clean_dataset: Path):
        dataset_id = upload(client, clean_dataset)
        response = client.post(
            f"/api/datasets/{dataset_id}/run/convert.dataset",
            json={"params": {"output": "/tmp/anywhere", "to": "yolo"}},
        )
        assert response.status_code == 400

    @pytest.mark.parametrize(
        "bad_id",
        # Forms that actually reach the handler. Plain "../.." is deliberately
        # not here: an HTTP client normalises it away before the request is
        # sent, so asserting on it would test httpx, not this service. The
        # percent-encoded form does arrive intact, and is the real case.
        ["..", "%2e%2e", "%2e%2e%2f", "not-hex", "a" * 31, "a" * 33,
         "0123456789ABCDEF0123456789abcdef", "/etc/passwd"],
    )
    def test_dataset_ids_that_are_not_ids_are_rejected(self, client, bad_id):
        response = client.get(f"/api/datasets/{bad_id}")
        assert response.status_code in (400, 404, 405), f"{bad_id} -> {response.status_code}"

    def test_a_traversal_id_cannot_read_outside_the_workspace(self, client, tmp_path: Path):
        """The id pattern is what stops '..' resolving to the workspace parent."""
        secret = Path(client.app_module.WORKSPACE.root).parent / "secret.txt"
        secret.write_text("do not serve me", encoding="utf-8")
        for attempt in ("%2e%2e", "%2e%2e%2f%2e%2e", ".."):
            response = client.get(f"/api/datasets/{attempt}/download")
            assert response.status_code in (400, 404), attempt
            assert b"do not serve me" not in response.content

    def test_zip_slip_members_are_skipped(self, client, tmp_path: Path):
        """A member named ../../evil.txt must not be written outside."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr("../../escaped.txt", "nope")
            bundle.writestr("../escaped2.txt", "nope")
            bundle.writestr("fine.jpg", b"\xff\xd8\xff\xe0jpegish")
        response = client.post(
            "/api/datasets", files={"file": ("d.zip", buffer.getvalue(), "application/zip")}
        )
        assert response.status_code == 201
        assert response.json()["uploaded"]["skipped"] == 2
        assert response.json()["uploaded"]["files"] == 1

        workspace = Path(client.app_module.WORKSPACE.root)
        assert not list(workspace.parent.glob("escaped*.txt"))
        assert not list(workspace.rglob("escaped*.txt"))

    def test_an_absolute_member_path_is_skipped(self, client):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr("/absolute.txt", "nope")
            bundle.writestr("ok.jpg", b"data")
        response = client.post(
            "/api/datasets", files={"file": ("d.zip", buffer.getvalue(), "application/zip")}
        )
        assert response.json()["uploaded"]["skipped"] == 1

    def test_a_non_zip_upload_is_refused(self, client):
        response = client.post(
            "/api/datasets", files={"file": ("x.tar", b"not a zip", "application/x-tar")}
        )
        assert response.status_code == 415

    def test_a_corrupt_zip_is_refused_cleanly(self, client):
        response = client.post(
            "/api/datasets", files={"file": ("d.zip", b"PK\x03\x04garbage", "application/zip")}
        )
        assert response.status_code == 400
        assert "zip" in response.json()["detail"].lower()

    def test_an_empty_archive_is_refused(self, client):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w"):
            pass
        response = client.post(
            "/api/datasets", files={"file": ("d.zip", buffer.getvalue(), "application/zip")}
        )
        assert response.status_code == 400

    def test_the_api_key_gate_works_when_configured(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws2"))
        monkeypatch.setenv("DATAFACTORY_API_KEY", "s3cret")
        import importlib

        from data_factory.service import app as app_module

        importlib.reload(app_module)
        with TestClient(app_module.app) as guarded:
            assert guarded.get("/api/operations").status_code == 401
            assert guarded.get("/api/operations", headers={"X-API-Key": "wrong"}).status_code == 401
            assert guarded.get("/api/operations", headers={"X-API-Key": "s3cret"}).status_code == 200
            # health stays open, so a load balancer can probe it.
            assert guarded.get("/api/health").status_code == 200


class TestDiscovery:
    def test_health_reports_the_operation_count(self, client):
        payload = client.get("/api/health").json()
        assert payload["ok"] is True
        assert payload["operations"] >= 27

    def test_the_schema_hides_path_parameters(self, client):
        """A client cannot set them, so offering a field would be a lie."""
        payload = client.get("/api/operations").json()
        for operation in payload["operations"]:
            for param in operation["params"]:
                assert param["kind"] not in ("dir", "file", "path", "dirs"), operation["name"]

    def test_an_operation_declares_whether_it_writes(self, client):
        payload = client.get("/api/operations/convert.dataset").json()
        assert payload["writes_output"] is True
        assert client.get("/api/operations/inspect.health").json()["writes_output"] is False

    def test_unknown_operation_is_404(self, client):
        assert client.get("/api/operations/nope.nope").status_code == 404


class TestDatasetFlow:
    def test_upload_detects_the_format_and_classes(self, client, clean_dataset: Path):
        response = client.post(
            "/api/datasets", files={"file": ("d.zip", zip_of(clean_dataset), "application/zip")}
        )
        assert response.status_code == 201
        payload = response.json()
        assert payload["format"] == "voc"
        assert payload["stats"]["images"] == 7
        assert set(payload["classes"]) == {"Open", "Close"}
        assert payload["expires_in_seconds"] > 0

    def test_run_convert_then_download_the_result(self, client, clean_dataset: Path):
        dataset_id = upload(client, clean_dataset)
        started = client.post(
            f"/api/datasets/{dataset_id}/run/convert.dataset",
            json={"params": {"to": "yolo"}, "output_name": "yolo"},
        )
        assert started.status_code == 200, started.text
        finished = wait_for(client, started.json()["id"])
        assert finished["state"] == "finished", finished
        assert finished["result"]["counts"]["images"] == 7

        download = client.get(f"/api/datasets/{dataset_id}/download", params={"output": "yolo"})
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(download.content)) as bundle:
            names = bundle.namelist()
        assert "data.yaml" in names
        assert any(n.endswith(".txt") for n in names)

    def test_the_output_of_one_step_feeds_the_next(self, client, clean_dataset: Path):
        """Chaining without the client ever seeing a path."""
        dataset_id = upload(client, clean_dataset)
        first = client.post(
            f"/api/datasets/{dataset_id}/run/image.resize",
            json={"params": {"width": 64, "height": 64}, "output_name": "small"},
        ).json()
        assert wait_for(client, first["id"])["state"] == "finished"

        second = client.post(
            f"/api/datasets/{dataset_id}/run/inspect.health",
            json={"input": "small"},
        ).json()
        result = wait_for(client, second["id"])
        assert result["state"] == "finished"
        assert result["result"]["counts"]["images"] == 7

        listing = client.get(f"/api/datasets/{dataset_id}").json()
        assert "small" in listing["inputs"]

    def test_an_unknown_input_name_is_a_clear_error(self, client, clean_dataset: Path):
        dataset_id = upload(client, clean_dataset)
        response = client.post(
            f"/api/datasets/{dataset_id}/run/inspect.health",
            json={"input": "never-made"},
        )
        assert response.status_code == 400
        assert "no step has produced" in response.json()["detail"]

    def test_a_bad_parameter_fails_immediately_not_as_a_job(self, client, clean_dataset: Path):
        """Validation is synchronous, so the client learns now, not in 30s."""
        dataset_id = upload(client, clean_dataset)
        response = client.post(
            f"/api/datasets/{dataset_id}/run/convert.dataset",
            json={"params": {"to": "not-a-format"}},
        )
        assert response.status_code == 422
        assert "not one of" in response.json()["detail"]

    def test_deleting_a_dataset_removes_it(self, client, clean_dataset: Path):
        dataset_id = upload(client, clean_dataset)
        assert client.delete(f"/api/datasets/{dataset_id}").status_code == 204
        assert client.get(f"/api/datasets/{dataset_id}").status_code == 404

    def test_a_folder_of_bare_images_still_uploads(self, client, tmp_path: Path):
        """No annotations at all is a legitimate starting point."""
        from conftest import make_image

        folder = tmp_path / "bare"
        for index in range(3):
            make_image(folder / f"{index}.jpg", index)
        response = client.post(
            "/api/datasets", files={"file": ("d.zip", zip_of(folder), "application/zip")}
        )
        assert response.status_code == 201
        assert response.json()["format"] is None
        assert "detection_error" in response.json()


class TestJobs:
    def test_a_failing_operation_reports_failed_not_crashed(self, client, clean_dataset: Path):
        dataset_id = upload(client, clean_dataset)
        started = client.post(
            f"/api/datasets/{dataset_id}/run/augment.dataset",
            json={"params": {"flip_horizontal": False, "brightness": 0.0, "contrast": 0.0}},
        ).json()
        finished = wait_for(client, started["id"])
        assert finished["state"] == "failed"
        assert "no augmentations" in finished["error"]

    def test_jobs_are_listed_per_dataset(self, client, clean_dataset: Path):
        dataset_id = upload(client, clean_dataset)
        started = client.post(
            f"/api/datasets/{dataset_id}/run/inspect.health", json={}
        ).json()
        wait_for(client, started["id"])
        listing = client.get(f"/api/datasets/{dataset_id}/jobs").json()
        assert len(listing["jobs"]) == 1
        assert listing["jobs"][0]["operation"] == "inspect.health"

    def test_cancelling_a_finished_job_is_a_conflict(self, client, clean_dataset: Path):
        dataset_id = upload(client, clean_dataset)
        started = client.post(
            f"/api/datasets/{dataset_id}/run/inspect.health", json={}
        ).json()
        wait_for(client, started["id"])
        assert client.post(f"/api/jobs/{started['id']}/cancel").status_code == 409

    def test_unknown_job_is_404(self, client):
        assert client.get("/api/jobs/deadbeef").status_code == 404

    def test_the_audit_trail_is_not_shipped_to_the_browser(self, client, clean_dataset: Path):
        """Fine in a terminal, far too large for a 10,000-file run over HTTP."""
        dataset_id = upload(client, clean_dataset)
        started = client.post(
            f"/api/datasets/{dataset_id}/run/convert.dataset", json={"params": {"to": "coco"}}
        ).json()
        finished = wait_for(client, started["id"])
        assert "actions" not in finished["result"]


class TestExpiry:
    def test_the_sweeper_removes_expired_datasets(self, client, clean_dataset: Path):
        dataset_id = upload(client, clean_dataset)
        workspace = client.app_module.WORKSPACE
        workspace.ttl_seconds = -1  # everything is already expired
        assert workspace.sweep() >= 1
        assert client.get(f"/api/datasets/{dataset_id}").status_code == 404

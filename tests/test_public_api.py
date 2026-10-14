"""`import easyannotate`: the headless library surface."""

from __future__ import annotations


import pytest

from conftest import make_image, write_voc


@pytest.fixture
def ea(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws"))
    import easyannotate
    from easyannotate import project as project_mod

    project_mod._stores.clear()
    yield easyannotate
    for store in project_mod._stores.values():
        store.close()
    project_mod._stores.clear()


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "doors"
    for index in range(3):
        make_image(root / f"d{index}.jpg", 50 + index, size=(120, 90))
    write_voc(root / "d0.xml", "Open", bbox=(10, 10, 60, 70), size=(120, 90))
    return root


def test_a_script_opens_a_folder_and_exports_reviewed_work(ea, folder, tmp_path):
    project = ea.open_folder(folder)
    assert project.counts()["total"] == 3 and project.counts()["reviewed"] == 1
    assert project.classes() == ["Open"]
    assert project.root == folder.resolve()
    assert ea.database_path() == tmp_path / "ws" / ".." / "annotations.db" or ea.database_path().name == "annotations.db"

    result = project.export(tmp_path / "out", to="yolo")
    assert result.ok, result.errors
    labels = sorted(p.name for p in (tmp_path / "out").rglob("*.txt"))
    assert labels == ["d0.txt"]  # reviewed only

    again = ea.open_folder(folder)
    assert again.id == project.id
    assert [p.id for p in ea.list_projects()] == [project.id]
    assert ea.open_project(project.id).name == project.name


def test_predictions_go_through_the_matching_report(ea, folder, tmp_path):
    project = ea.open_folder(folder)
    run = tmp_path / "run"
    run.mkdir()
    (run / "data.yaml").write_text("names:\n  0: Open\n", encoding="utf-8")
    (run / "d1.txt").write_text("0 0.5 0.5 0.2 0.2 0.6\n", encoding="utf-8")
    report = project.import_predictions(run)
    assert report["matched"] == 1
    queue = project.images(state="predicted", order="uncertain")
    assert [i["name"] for i in queue] == ["d1.jpg"]
    assert queue[0]["uncertainty"] == pytest.approx(0.8)


def test_operations_run_from_the_library(ea, folder):
    result = ea.run("labels.list", {"folder": folder})
    assert result.ok
    assert "inspect.health" in [op.name for op in ea.list_operations()]


def test_python_dash_m_is_the_command(tmp_path):
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-m", "easyannotate", "--version"], capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.startswith("easyannotate ")

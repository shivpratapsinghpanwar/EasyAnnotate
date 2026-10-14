"""`import easyannotate`: the headless library surface."""

from __future__ import annotations


from pathlib import Path

import pytest

from conftest import make_image, write_voc


@pytest.fixture
def ea(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws"))
    import easyannotate
    from easyannotate import project as project_mod

    project_mod._stores.clear()
    yield easyannotate
    easyannotate.close()


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
    assert ea.database_path().resolve() == (tmp_path / "annotations.db").resolve()

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


def test_review_findings_stay_fixed(ea, folder, tmp_path, monkeypatch):
    """What the independent review of the first version found."""
    import threading

    project = ea.open_folder(folder)
    # A state as a plain string is one state, not its letters.
    assert project.export(tmp_path / "a", states="reviewed").ok
    # Formats are validated like every other caller's, and `~` is expanded.
    with pytest.raises(ea.ParamError):
        project.export(tmp_path / "b", to="bogus")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    assert project.export("~/exported").ok
    assert (tmp_path / "home" / "exported").is_dir()
    assert not (Path.cwd() / "~").exists()
    # Provenance is written, as for ea.run().
    assert (tmp_path / "home" / "exported" / ".datafactory" / "history.jsonl").is_file()
    # Errors are catchable from the public surface.
    with pytest.raises(ea.Error):
        ea.open_project("0" * 32)

    # The store is opened once however many threads ask first.
    from easyannotate import project as project_mod

    ea.close()
    failures = []

    def first_touch():
        try:
            ea.open_folder(folder).counts()
        except Exception as exc:  # pragma: no cover - the failure being guarded
            failures.append(exc)

    threads = [threading.Thread(target=first_touch) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == [] and len(project_mod._stores) == 1


def test_images_reach_past_one_page(ea):
    """The store pages at 5,000; a script asking for everything gets everything."""
    from data_factory.annotate import images as images_mod
    from data_factory.annotate import projects as projects_mod
    from easyannotate import project as project_mod

    store = project_mod._store()
    record = projects_mod.create(store, "many")
    images_mod.add_many(store, record, [(f"i{n:05d}.jpg", 10, 10) for n in range(5200)])
    project = ea.open_project(record.id)
    assert project.counts()["total"] == 5200
    assert len(project.images()) == 5200
    assert len(project.images(limit=10, offset=5195)) == 5

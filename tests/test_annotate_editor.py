"""The editor, driven in a real browser.

Everything else in this suite tests Python. The editor is the product, and a
canvas tool with no automated test is one that quietly stops working -- so
these drive the actual page: drag a box, name a class, save, reload, and check
the work came back.

Skipped when Playwright or its browser is not installed, which keeps the suite
runnable on a fresh machine with nothing but ``pip install -e '.[web,dev]'``.
Install it with::

    pip install playwright && playwright install chromium
"""

from __future__ import annotations

import os
import socket
import threading
import time
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="optional: pip install 'data-factory[web]'")
pytest.importorskip("playwright", reason="optional: pip install playwright")
pytest.importorskip("uvicorn", reason="optional: pip install 'data-factory[web]'")

import httpx
import uvicorn
from playwright.sync_api import sync_playwright

from conftest import make_image, write_voc


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        try:
            launched = pw.chromium.launch()
        except Exception as exc:  # no browser downloaded
            pytest.skip(f"chromium unavailable: {exc}")
        yield launched
        launched.close()


@pytest.fixture
def service(tmp_path, monkeypatch):
    """The real app on a real port, against a throwaway workspace."""
    monkeypatch.setenv("DATAFACTORY_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.delenv("DATAFACTORY_API_KEY", raising=False)

    import importlib

    from data_factory.service import app as app_module
    from data_factory.service import deps

    deps.reset_stores()
    importlib.reload(app_module)

    port = free_port()
    config = uvicorn.Config(app_module.app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        pytest.skip("the service did not start")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=15)
    deps.reset_stores()


def make_project(base: str, folder: Path, name="test") -> dict:
    archive = folder.parent / f"{name}.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in sorted(folder.iterdir()):
            if path.is_file():
                bundle.write(path, path.name)
    with httpx.Client(base_url=base, timeout=60) as client:
        dataset = client.post(
            "/api/datasets",
            files={"file": ("d.zip", archive.read_bytes(), "application/zip")},
        ).json()
        return client.post(f"/api/datasets/{dataset['id']}/promote", json={}).json()


def open_editor(browser, base: str, project_id: str, errors: list[str]):
    page = browser.new_page(viewport={"width": 1280, "height": 820})
    page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
    page.on(
        "console",
        lambda msg: errors.append(f"console.error: {msg.text}") if msg.type == "error" else None,
    )
    page.goto(f"{base}/annotate?project={project_id}", wait_until="networkidle")
    page.wait_for_function("window.__editor && window.__editor.image", timeout=30_000)
    page.wait_for_timeout(400)
    return page


def drag(page, x1, y1, x2, y2):
    rect = page.locator("#canvas").bounding_box()
    page.mouse.move(rect["x"] + x1, rect["y"] + y1)
    page.mouse.down()
    page.mouse.move(rect["x"] + x2, rect["y"] + y2, steps=10)
    page.mouse.up()
    page.wait_for_timeout(250)


@pytest.fixture
def bare_images(tmp_path):
    folder = tmp_path / "bare"
    folder.mkdir()
    for index in range(3):
        make_image(folder / f"p{index}.jpg", 40 + index, size=(900, 700))
    return folder


@pytest.fixture
def annotated_images(tmp_path):
    folder = tmp_path / "annotated"
    folder.mkdir()
    for index in range(3):
        make_image(folder / f"a{index}.jpg", 60 + index, size=(800, 600))
    write_voc(folder / "a0.xml", "Open", size=(800, 600))
    return folder


def test_a_box_can_be_drawn_and_named_with_nothing_configured(
    browser, service, bare_images
):
    """The product claim, end to end: no schema step, no task, no labels.

    If this ever needs a configuration step before the first box, the whole
    competitive position against CVAT is gone.
    """
    errors: list[str] = []
    project = make_project(service, bare_images, "bare")
    assert project["labels"] == [], "a fresh project must start with no classes"

    page = open_editor(browser, service, project["id"], errors)
    try:
        assert page.evaluate("window.__editor.rail.labels.length") == 0

        drag(page, 380, 260, 620, 470)

        # The geometry is held, not discarded, and the name box takes focus.
        assert page.evaluate("!!window.__editor.pendingGeometry")
        assert page.evaluate(
            "document.activeElement === document.querySelector"
            "('[data-role=\"label-input\"]')"
        )

        page.keyboard.type("Closed")
        page.keyboard.press("Enter")
        page.wait_for_timeout(900)

        shapes = page.evaluate(
            "[...window.__editor.scene.shapes.values()].map(s => [s.label, s.labelIdx])"
        )
        assert shapes == [["Closed", 0]]
        assert page.inner_text("#status") == "saved"
    finally:
        page.close()
    assert not errors, errors


def test_work_survives_a_reload(browser, service, bare_images):
    errors: list[str] = []
    project = make_project(service, bare_images, "reload")
    page = open_editor(browser, service, project["id"], errors)
    try:
        drag(page, 300, 240, 520, 420)
        page.keyboard.type("Open")
        page.keyboard.press("Enter")
        page.wait_for_timeout(900)

        drag(page, 560, 250, 700, 400)
        page.wait_for_timeout(900)
        assert page.evaluate("window.__editor.scene.order.length") == 2

        # Drafts must have taken their server ids, or the next save inserts
        # them a second time and every autosave doubles the boxes.
        assert (
            page.evaluate(
                "window.__editor.scene.order.filter(i => i.startsWith('draft-')).length"
            )
            == 0
        )

        page.reload(wait_until="networkidle")
        page.wait_for_function("window.__editor && window.__editor.image", timeout=30_000)
        page.wait_for_timeout(500)
        assert page.evaluate("window.__editor.scene.order.length") == 2
    finally:
        page.close()
    assert not errors, errors


def test_undo_removes_a_shape_and_the_removal_is_saved(browser, service, bare_images):
    errors: list[str] = []
    project = make_project(service, bare_images, "undo")
    page = open_editor(browser, service, project["id"], errors)
    try:
        drag(page, 300, 240, 520, 420)
        page.keyboard.type("Open")
        page.keyboard.press("Enter")
        page.wait_for_timeout(900)
        drag(page, 560, 250, 700, 400)
        page.wait_for_timeout(900)

        page.keyboard.press("Control+z")
        page.wait_for_timeout(900)
        assert page.evaluate("window.__editor.scene.order.length") == 1

        page.reload(wait_until="networkidle")
        page.wait_for_function("window.__editor && window.__editor.image", timeout=30_000)
        page.wait_for_timeout(500)
        assert page.evaluate("window.__editor.scene.order.length") == 1
    finally:
        page.close()
    assert not errors, errors


def test_a_polygon_can_be_drawn(browser, service, bare_images):
    errors: list[str] = []
    project = make_project(service, bare_images, "poly")
    page = open_editor(browser, service, project["id"], errors)
    try:
        page.click('[data-role="label-input"]')
        page.keyboard.type("Frame")
        page.keyboard.press("Enter")
        page.wait_for_timeout(600)
        # Naming a class must hand the keyboard back, so P reaches the canvas
        # rather than being typed into the name field.
        assert page.evaluate(
            "document.activeElement !== document.querySelector"
            "('[data-role=\"label-input\"]')"
        ), "focus stayed in the name field after creating a class"

        page.keyboard.press("p")
        rect = page.locator("#canvas").bounding_box()
        for dx, dy in [(220, 220), (360, 230), (330, 380), (200, 360)]:
            page.mouse.click(rect["x"] + dx, rect["y"] + dy)
            page.wait_for_timeout(70)
        page.keyboard.press("Enter")
        page.wait_for_timeout(900)

        polygons = page.evaluate(
            "[...window.__editor.scene.shapes.values()].filter(s => s.polygon).length"
        )
        assert polygons == 1
    finally:
        page.close()
    assert not errors, errors


def test_resume_opens_the_first_unfinished_image(browser, service, annotated_images):
    """a0 arrives reviewed because its VOC file was hand-made, so the editor
    must open a1 -- not restart at the beginning."""
    errors: list[str] = []
    project = make_project(service, annotated_images, "resume")
    page = open_editor(browser, service, project["id"], errors)
    try:
        assert page.evaluate("window.__editor.image.name") == "a1.jpg"
    finally:
        page.close()
    assert not errors, errors


def test_marking_reviewed_advances(browser, service, bare_images):
    errors: list[str] = []
    project = make_project(service, bare_images, "advance")
    page = open_editor(browser, service, project["id"], errors)
    try:
        first = page.evaluate("window.__editor.image.id")
        page.keyboard.press("r")
        page.wait_for_timeout(1200)
        assert page.evaluate("window.__editor.image.id") != first
    finally:
        page.close()
    assert not errors, errors


def test_the_front_page_leads_into_the_editor(browser, service, annotated_images, tmp_path):
    """The funnel: upload, look at what it is, start annotating. No account,
    no schema, no decisions before the first box."""
    errors: list[str] = []
    archive = tmp_path / "funnel.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in sorted(annotated_images.iterdir()):
            bundle.write(path, path.name)

    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
    try:
        page.goto(service, wait_until="networkidle")
        page.set_input_files("input[type=file]", str(archive))
        page.wait_for_selector("#infoCard:not(.hide)", timeout=60_000)

        page.click("#btnAnnotate")
        page.wait_for_function("window.__editor && window.__editor.image", timeout=60_000)
        assert "/annotate?project=" in page.url
        # The class order came from the upload, not from sorting what we found.
        assert page.evaluate("window.__editor.rail.labels.map(l => l.name)") == ["Open"]
    finally:
        page.close()
    assert not errors, errors


def test_drawing_in_the_browser_updates_the_xml_on_disk(browser, service, tmp_path):
    """Attached mode's whole promise, proved through the real editor.

    Not a copy, not on export: the .xml that was already in the folder is
    rewritten while you draw.
    """
    import xml.etree.ElementTree as ET

    from data_factory.annotate import attach
    from data_factory.service.deps import get_store

    folder = tmp_path / "in_place"
    folder.mkdir()
    for index in range(3):
        make_image(folder / f"d{index}.jpg", 70 + index, size=(400, 300))
    write_voc(folder / "d0.xml", "Open", size=(400, 300))

    summary = attach.open_folder(get_store(), folder)
    project = summary["project"]
    assert summary["sidecar_format"] == "voc"

    errors: list[str] = []
    page = open_editor(browser, service, project.id, errors)
    try:
        # d0 imported as reviewed, so the editor resumes on d1 -- which has no
        # label file at all yet.
        assert page.evaluate("window.__editor.image.name") == "d1.jpg"
        target = folder / "d1.xml"
        assert not target.exists()

        # The imported class 'Open' already exists, so the box uses it.
        drag(page, 300, 220, 520, 400)
        page.wait_for_timeout(1400)

        assert target.is_file(), "no sidecar was written to the folder"
        names = [node.text for node in ET.parse(target).getroot().iter("name")]
        assert names == ["Open"]

        # Now re-class it, which must rewrite the same file rather than
        # appending to it or leaving it stale.
        page.click('[data-role="label-input"]')
        page.keyboard.type("Hinge")
        page.keyboard.press("Enter")
        page.wait_for_timeout(700)

        rect = page.locator("#canvas").bounding_box()
        page.mouse.click(rect["x"] + 400, rect["y"] + 300)
        page.wait_for_timeout(300)
        assert page.evaluate("window.__editor.scene.selection.size") == 1
        page.keyboard.press("2")  # index 1 -> Hinge
        page.wait_for_timeout(1400)

        names = [node.text for node in ET.parse(target).getroot().iter("name")]
        assert names == ["Hinge"], "the sidecar was not rewritten on re-class"
    finally:
        page.close()
    assert not errors, errors


def test_the_review_loop_end_to_end(browser, service, bare_images, tmp_path):
    """Import predictions, look at them, accept, and watch agreement move.

    The whole Phase 2 argument in one test: a model's output comes back in,
    a person confirms it in one keystroke, and the headline number reflects
    what actually happened.
    """
    import zipfile

    errors: list[str] = []
    project = make_project(service, bare_images, "loop")
    pid = project["id"]

    # Two boxes per image, as `yolo --save-conf` writes them: one confident,
    # one not.
    run = tmp_path / "run"
    run.mkdir()
    (run / "data.yaml").write_text("names:\n  0: Door\n", encoding="utf-8")
    for index in range(3):
        (run / f"p{index}.txt").write_text(
            "0 0.5 0.5 0.3 0.3 0.95\n0 0.15 0.15 0.06 0.06 0.20\n", encoding="utf-8"
        )
    archive = tmp_path / "run.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in sorted(run.iterdir()):
            bundle.write(path, path.name)

    with httpx.Client(base_url=service, timeout=120) as client:
        report = client.post(
            f"/api/projects/{pid}/predictions",
            files={"file": ("run.zip", archive.read_bytes(), "application/zip")},
        ).json()
    assert report["matched"] == 3, report
    assert report["shapes_imported"] == 6

    page = open_editor(browser, service, pid, errors)
    try:
        assert page.evaluate("window.__editor.scene.order.length") == 2
        assert page.evaluate(
            "[...window.__editor.scene.shapes.values()].every(s => s.origin === 'model')"
        )
        # The review panel appears only when there is something to review.
        assert page.evaluate("!document.getElementById('review').hidden")

        # The slider hides the weak box without deleting it.
        page.evaluate("""() => {
          const slider = document.getElementById('conf');
          slider.value = '50';
          slider.dispatchEvent(new Event('input'));
        }""")
        page.wait_for_timeout(200)
        shown = page.evaluate(
            "[...window.__editor.scene.shapes.values()].filter(s =>"
            " window.__editor.visible(s)).length"
        )
        assert shown == 1, "the confidence filter did not hide the weak prediction"
        assert page.evaluate("window.__editor.scene.order.length") == 2, "it deleted instead"

        # Accept: the visible one is kept, the hidden one is a rejection.
        page.keyboard.press("a")
        page.wait_for_timeout(1800)

        with httpx.Client(base_url=service, timeout=60) as client:
            summary = client.get(f"/api/projects/{pid}/summary").json()
        assert summary["counts"]["reviewed"] == 1
        agreement = summary["agreement"]
        assert agreement["model_shapes"] == 2
        assert agreement["deleted"] == 1, agreement
        assert agreement["agreement"] == 0.5
    finally:
        page.close()
    assert not errors, errors


def test_the_status_never_claims_unsaved_when_nothing_changed(
    browser, service, bare_images
):
    """A save indicator that is wrong even once is one nobody trusts again."""
    errors: list[str] = []
    project = make_project(service, bare_images, "status")
    page = open_editor(browser, service, project["id"], errors)
    try:
        drag(page, 300, 240, 520, 420)
        page.keyboard.type("Open")
        page.keyboard.press("Enter")
        page.wait_for_timeout(900)

        page.mouse.click(page.locator("#canvas").bounding_box()["x"] + 400,
                         page.locator("#canvas").bounding_box()["y"] + 330)
        page.wait_for_timeout(200)
        # Re-picking the class it already has changes nothing.
        page.keyboard.press("1")
        page.wait_for_timeout(900)
        assert page.inner_text("#status") == "saved"
    finally:
        page.close()
    assert not errors, errors

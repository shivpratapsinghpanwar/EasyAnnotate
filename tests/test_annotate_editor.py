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


def _shape_boxes(page):
    return page.evaluate(
        "window.__editor.scene.order.map(id => window.__editor.scene.get(id).bbox)"
    )


def test_undo_still_works_after_autosave_renames_the_shape(browser, service, bare_images):
    """Autosave swaps a draft id for the server's. Undo must follow the rename.

    The commands recorded the id the shape had when it was drawn; once the
    save renamed it, Ctrl+Z on a move silently did nothing.
    """
    errors: list[str] = []
    project = make_project(service, bare_images, "undo-reid")
    page = open_editor(browser, service, project["id"], errors)
    try:
        drag(page, 300, 240, 520, 420)
        page.keyboard.type("Open")
        page.keyboard.press("Enter")
        page.wait_for_function("window.__editor.scene.order.length === 1")
        drawn = _shape_boxes(page)

        # Move it before the first autosave, so the move is recorded against
        # the draft id -- then let the save rename the shape underneath it.
        # Click first: a drag across a just-drawn box draws, it does not move.
        rect = page.locator("#canvas").bounding_box()
        page.mouse.click(rect["x"] + 400, rect["y"] + 330)
        drag(page, 400, 330, 470, 390)
        assert page.evaluate("window.__editor.scene.order[0]").startswith("draft-")
        page.wait_for_timeout(900)
        assert not page.evaluate("window.__editor.scene.order[0]").startswith("draft-")
        assert _shape_boxes(page) != drawn

        page.keyboard.press("Control+z")
        page.wait_for_timeout(900)
        assert _shape_boxes(page) == drawn
    finally:
        page.close()
    assert not errors, errors


def test_a_restored_shape_keeps_one_id(browser, service, bare_images):
    """Undoing a delete re-sends an id the server no longer has.

    The server inserts it under a fresh id. The client must adopt that id, or
    every later save churns the shape into a new row and its history is lost.
    """
    errors: list[str] = []
    project = make_project(service, bare_images, "restore")
    page = open_editor(browser, service, project["id"], errors)
    try:
        drag(page, 300, 240, 520, 420)
        page.keyboard.type("Open")
        page.keyboard.press("Enter")
        page.wait_for_timeout(900)

        page.keyboard.press("Delete")
        page.wait_for_timeout(900)
        page.keyboard.press("Control+z")
        page.wait_for_timeout(900)

        image_id = page.evaluate("window.__editor.scene.imageId")
        with httpx.Client(base_url=service, timeout=30) as client:
            stored = client.get(f"/api/projects/{project['id']}/images/{image_id}").json()
        assert [s["id"] for s in stored["shapes"]] == page.evaluate(
            "window.__editor.scene.order"
        )
    finally:
        page.close()
    assert not errors, errors


def test_edits_made_during_a_slow_save_are_kept(browser, service, bare_images):
    """A save still in flight when the next autosave fires.

    Both used to carry the same base version: the second got a 409, and the
    conflict path reloaded the server copy over the user's newest box.
    """
    errors: list[str] = []
    project = make_project(service, bare_images, "slow")
    page = open_editor(browser, service, project["id"], errors)
    try:
        page.evaluate(
            """() => {
              const api = window.__editor.api;
              const real = api.saveShapes.bind(api);
              api.saveShapes = (...args) =>
                new Promise((r) => setTimeout(r, 1500)).then(() => real(...args));
            }"""
        )
        drag(page, 300, 240, 520, 420)
        page.keyboard.type("Open")
        page.keyboard.press("Enter")
        page.wait_for_timeout(700)  # first save is now in flight
        drag(page, 560, 250, 700, 400)
        page.wait_for_timeout(6000)

        assert page.evaluate("window.__editor.scene.order.length") == 2
        assert page.inner_text("#status") == "saved"

        page.reload(wait_until="networkidle")
        page.wait_for_function("window.__editor && window.__editor.image", timeout=30_000)
        page.wait_for_timeout(500)
        assert page.evaluate("window.__editor.scene.order.length") == 2
    finally:
        page.close()
    assert not errors, errors


# ------------------------------------------------- M1: editing from the keys


def _draw_named_box(page, name="Open", box=(300, 240, 520, 420)):
    drag(page, *box)
    if page.evaluate("window.__editor.rail.labels.length") == 0:
        page.keyboard.type(name)
        page.keyboard.press("Enter")
    page.wait_for_function("window.__editor.scene.order.length >= 1")
    page.wait_for_timeout(900)


def test_arrow_keys_nudge_and_a_held_key_is_one_undo(browser, service, bare_images):
    errors: list[str] = []
    project = make_project(service, bare_images, "nudge")
    page = open_editor(browser, service, project["id"], errors)
    try:
        _draw_named_box(page)
        before = _shape_boxes(page)[0]

        for _ in range(5):
            page.keyboard.press("ArrowRight")
        page.keyboard.press("Shift+ArrowDown")
        after = _shape_boxes(page)[0]
        assert after == [before[0] + 5, before[1] + 10, before[2] + 5, before[3] + 10]

        page.keyboard.press("Control+z")
        assert _shape_boxes(page)[0] == before

        page.wait_for_timeout(900)
        page.reload(wait_until="networkidle")
        page.wait_for_function("window.__editor && window.__editor.image", timeout=30_000)
        page.wait_for_timeout(500)
        assert _shape_boxes(page)[0] == before
    finally:
        page.close()
    assert not errors, errors


def test_tab_cycles_and_paste_carries_shapes_to_the_next_image(browser, service, bare_images):
    errors: list[str] = []
    project = make_project(service, bare_images, "clip")
    page = open_editor(browser, service, project["id"], errors)
    try:
        _draw_named_box(page)
        drag(page, 560, 250, 700, 400)
        page.wait_for_timeout(900)
        page.keyboard.press("Escape")

        page.keyboard.press("Tab")
        first = page.evaluate("[...window.__editor.scene.selection]")
        page.keyboard.press("Tab")
        second = page.evaluate("[...window.__editor.scene.selection]")
        assert len(first) == len(second) == 1 and first != second

        page.keyboard.press("Control+a")
        page.keyboard.press("Control+c")
        boxes = sorted(_shape_boxes(page))
        page.keyboard.press("n")
        page.wait_for_function("window.__editor.image.name === 'p1.jpg'")
        page.keyboard.press("Control+v")
        page.wait_for_timeout(900)
        assert sorted(_shape_boxes(page)) == boxes  # in place, on the new image
        assert page.inner_text("#status") == "saved"
    finally:
        page.close()
    assert not errors, errors


def test_guides_show_over_empty_canvas_only(browser, service, bare_images):
    errors: list[str] = []
    project = make_project(service, bare_images, "guides")
    page = open_editor(browser, service, project["id"], errors)
    try:
        rect = page.locator("#canvas").bounding_box()
        page.mouse.move(rect["x"] + 100, rect["y"] + 100)
        assert page.evaluate("window.__editor.wantsCrosshair()")
        _draw_named_box(page)
        page.mouse.move(rect["x"] + 400, rect["y"] + 330)  # over the new box
        page.wait_for_timeout(100)
        # A drag here would draw a nested box, so the guides stay.
        assert page.evaluate("window.__editor.wantsCrosshair()")
        page.mouse.click(rect["x"] + 400, rect["y"] + 330)  # now it is selected
        page.mouse.move(rect["x"] + 405, rect["y"] + 335)
        page.wait_for_timeout(100)
        # A drag here moves it: guides would only be noise.
        assert not page.evaluate("window.__editor.wantsCrosshair()")
    finally:
        page.close()
    assert not errors, errors


# ------------------------------------------------- M1: classes and visibility


def test_typing_a_case_variant_picks_the_existing_class(browser, service, bare_images):
    """How the door set got Close/close/Open/open. Typing 'open' while 'Open'
    exists must reuse it, not make a second class."""
    errors: list[str] = []
    project = make_project(service, bare_images, "case")
    page = open_editor(browser, service, project["id"], errors)
    try:
        _draw_named_box(page, "Open")
        page.keyboard.press("Escape")
        drag(page, 560, 250, 700, 400)  # drawn as Open, then re-classed below
        rect = page.locator("#canvas").bounding_box()
        page.mouse.click(rect["x"] + 630, rect["y"] + 320)  # select it explicitly
        page.keyboard.press("/")
        page.keyboard.type("open")
        page.keyboard.press("Enter")
        page.wait_for_timeout(900)
        assert page.evaluate("window.__editor.rail.labels.map(l => l.name)") == ["Open"]

        page.keyboard.press("/")
        page.keyboard.type("Closed")
        page.keyboard.press("Enter")
        page.wait_for_timeout(900)
        names = page.evaluate("window.__editor.rail.labels.map(l => l.name)")
        assert names == ["Open", "Closed"]
        # Made while a shape was selected, so the selection took the new class.
        assert sorted(page.evaluate(
            "[...window.__editor.scene.shapes.values()].map(s => s.label)"
        )) == ["Closed", "Open"]
    finally:
        page.close()
    assert not errors, errors


def test_a_hidden_class_is_kept_by_accept_all(browser, service, bare_images):
    """accept-all deletes what the confidence slider hides. A class hidden only
    to declutter the view must not count as rejected."""
    errors: list[str] = []
    project = make_project(service, bare_images, "hide")
    page = open_editor(browser, service, project["id"], errors)
    try:
        _draw_named_box(page, "Open")
        page.locator(".label-item [data-role='eye']").first.click()
        page.wait_for_timeout(100)  # the list redraws on the next frame
        assert page.evaluate("window.__editor.visible([...window.__editor.scene.shapes.values()][0])") is False
        assert page.text_content("[data-role='shape-count']").strip() == "0 of 1 shown"

        page.keyboard.press("a")
        page.wait_for_timeout(900)
        with httpx.Client(base_url=service, timeout=30) as client:
            first = client.get(f"/api/projects/{project['id']}/images").json()["images"][0]
            stored = client.get(f"/api/projects/{project['id']}/images/{first['id']}").json()
        assert len(stored["shapes"]) == 1
    finally:
        page.close()
    assert not errors, errors


def test_the_shape_list_selects_what_is_clicked(browser, service, bare_images):
    errors: list[str] = []
    project = make_project(service, bare_images, "list")
    page = open_editor(browser, service, project["id"], errors)
    try:
        _draw_named_box(page, "Open")
        drag(page, 560, 250, 700, 400)
        page.wait_for_timeout(900)
        page.keyboard.press("Escape")
        assert page.locator(".shape-item").count() == 2
        page.locator(".shape-item").nth(0).click()
        assert page.evaluate("[...window.__editor.scene.selection]") == [
            page.evaluate("window.__editor.scene.order[0]")
        ]
    finally:
        page.close()
    assert not errors, errors


# ------------------------------------------------------------ M1: at scale


def test_the_filmstrip_reaches_past_two_thousand_images(browser, service, tmp_path):
    """The strip fetched one page of 2,000 and showed nothing beyond it. The
    door set has 15,514 images."""
    import shutil

    from data_factory.annotate import attach
    from data_factory.service.deps import get_store

    folder = tmp_path / "many"
    folder.mkdir()
    seed = make_image(folder / "img00000.jpg", 5, size=(64, 48))
    for index in range(1, 2500):
        shutil.copyfile(seed, folder / f"img{index:05d}.jpg")
    project = attach.open_folder(get_store(), folder)["project"]

    errors: list[str] = []
    page = open_editor(browser, service, project.id, errors)
    try:
        assert page.evaluate("window.__editor.strip.total") == 2500
        # Only the rows on screen exist -- not 2,500 nodes.
        assert page.locator(".strip-item").count() < 80

        page.evaluate(
            "const l = document.querySelector('[data-role=strip-list]'); l.scrollTop = l.scrollHeight"
        )
        page.wait_for_selector(".strip-item[data-id] >> text=img02499.jpg", timeout=10_000)
        page.locator(".strip-item >> text=img02499.jpg").click()
        page.wait_for_function("window.__editor.image && window.__editor.image.name === 'img02499.jpg'")

        page.select_option("[data-role=strip-filter]", "reviewed")
        page.wait_for_timeout(300)
        assert page.locator(".strip-item").count() == 0
        page.keyboard.press("r")
        page.wait_for_function("window.__editor.strip.total === 1", timeout=10_000)
        page.wait_for_selector(".strip-item >> text=img02499.jpg", timeout=10_000)
    finally:
        page.close()
    assert not errors, errors


# --------------------------------------------- M1: attributes and the view


def _stored_shapes(service, project_id, image_id):
    with httpx.Client(base_url=service, timeout=30) as client:
        return client.get(f"/api/projects/{project_id}/images/{image_id}").json()["shapes"]


def test_flags_are_set_from_the_panel_saved_and_undoable(browser, service, bare_images):
    errors: list[str] = []
    project = make_project(service, bare_images, "flags")
    page = open_editor(browser, service, project["id"], errors)
    try:
        _draw_named_box(page)
        image_id = page.evaluate("window.__editor.scene.imageId")
        assert page.is_visible("#attributes")

        page.check("#attributes [data-attr=occluded]")
        page.wait_for_timeout(900)
        assert _stored_shapes(service, project["id"], image_id)[0]["occluded"] == 1

        page.keyboard.press("Control+z")
        page.wait_for_timeout(900)
        assert _stored_shapes(service, project["id"], image_id)[0]["occluded"] == 0
    finally:
        page.close()
    assert not errors, errors


def test_brightness_is_applied_and_remembered(browser, service, bare_images):
    errors: list[str] = []
    project = make_project(service, bare_images, "look")
    page = open_editor(browser, service, project["id"], errors)
    try:
        page.locator("#brightness").fill("180")
        assert page.evaluate("window.__editor.imageFilter()") == "brightness(180%) contrast(100%)"
        page.reload(wait_until="networkidle")
        page.wait_for_function("window.__editor && window.__editor.image", timeout=30_000)
        assert page.evaluate("window.__editor.imageFilter()") == "brightness(180%) contrast(100%)"
    finally:
        page.close()
    assert not errors, errors


# ------------------------------------------------- M1: the drawing flow


def test_choosing_the_next_class_leaves_the_last_box_alone(browser, service, bare_images):
    """Pick a class, draw; pick the next, draw. A new box stays selected so it
    can be resized -- but picking the next class must not re-class it."""
    errors: list[str] = []
    project = make_project(service, bare_images, "flow")
    page = open_editor(browser, service, project["id"], errors)
    try:
        _draw_named_box(page, "Door")
        page.keyboard.press("/")
        page.keyboard.type("Handle")
        page.keyboard.press("Enter")
        page.wait_for_timeout(600)
        assert page.evaluate(
            "[...window.__editor.scene.shapes.values()].map(s => s.label)"
        ) == ["Door"]
        assert page.evaluate("window.__editor.rail.current.name") == "Handle"
    finally:
        page.close()
    assert not errors, errors


def test_a_box_can_be_drawn_inside_another(browser, service, bare_images):
    """A handle on a door: dragging inside an existing box draws a new one.
    It used to move the door."""
    errors: list[str] = []
    project = make_project(service, bare_images, "nested")
    page = open_editor(browser, service, project["id"], errors)
    try:
        _draw_named_box(page, "Door", (300, 200, 600, 520))
        door = _shape_boxes(page)[0]
        drag(page, 520, 340, 570, 390)
        page.wait_for_timeout(900)
        boxes = _shape_boxes(page)
        assert len(boxes) == 2 and boxes[0] == door

        # A click selects; then dragging that shape moves it.
        rect = page.locator("#canvas").bounding_box()
        page.mouse.click(rect["x"] + 330, rect["y"] + 230)
        drag(page, 330, 230, 360, 260)
        assert _shape_boxes(page)[0] != door
    finally:
        page.close()
    assert not errors, errors

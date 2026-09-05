# Plan: Data Factory → an annotation platform that beats CVAT on ease of use

> **Status.** Phase 0 is complete and committed. **Phase 1 is built and working**
> — the ordered label table, the annotation store, the HTTP layer, and a
> hand-written canvas editor with boxes, polygons, undo, autosave and resume.
> 518 tests pass, 8 of them driving the editor in a real browser.
>
> The class-index bug described under "Risks" as the worst on the list is fixed,
> with the export→add-class→export regression test the plan asked for first.
>
> Phase 2 (model-in-the-loop review) is next. Its storage already exists: shapes
> carry `origin` and `confidence`, and the agreement metric is computed.
>
> Counts written below (327 tests, 27 operations) are from before Phase 1;
> current figures are 518 and 28. The design reasoning is unchanged.

## Context

We have a solid dataset **preparation** engine: 27 operations, 5 annotation formats read and
written, 327 tests, a sandboxed FastAPI service, and a schema-driven web UI. What it cannot do
is the thing CVAT exists for — **annotate**. Today labelling happens in two OpenCV scripts that
cannot edit a shape once drawn, cannot zoom, never load what they previously wrote, and in one
case delete two annotation files on a stray keypress.

The goal is one product, delivered two ways (free local package, paid hosted), where a person
opens a folder and starts drawing. CVAT makes you create an organisation, then a project, then
labels, then a task, then jobs before you can draw one box. That gap is the entire competitive
position.

The workflow the owner wants at the centre of it: **annotate ~1000 images by hand → train a
model → run inference on unseen images → load those predictions back in → fix the wrong boxes by
dragging or redrawing → export → retrain.** Each turn of that loop costs less than the one
before. Manual annotation is Phase 1 on purpose: SAM cannot decide whether a door is *open or
closed*, and hand-drawn polygons must always remain available for when a model produces bad
masks.

### The rule this plan is built around

> **If a feature requires the user to configure something before they can draw a box, it does
> not ship.**

Everything below is subordinate to that.

---

## Architecture decisions

### 1. Annotation is a *second contract*, alongside the operation registry — not inside it

The registry's whole value is one property: `Operation.bind()` gives a CLI, a YAML recipe and an
HTTP POST byte-identical validation (`core/registry.py:195`). That holds only because every
`Param` kind has a meaningful string form. A polygon has no CLI rendering, so adding a `json`
param kind would make that claim false.

Annotation is stateful CRUD with sub-100ms latency budgets. It gets plain FastAPI routers over a
store. **One** bridge goes back to the batch world: a registered `annotate.export` operation,
which is genuinely folder-in/folder-out and reuses the existing `JobManager` and progress UI
unchanged.

New package `data_factory/annotate/` **must not import fastapi** — that keeps the domain testable
without a TestClient and lets the CLI drive the same store.

### 2. SQLite, one file, beside the workspace

Not files-only (no place for project name, class order, or review state; "which images are
unreviewed" becomes a tree walk). Not Postgres (a pip wheel cannot require a server). SQLite is
stdlib, ACID, and WAL gives one writer + many readers — exactly the annotation access pattern.
The same schema runs local and hosted, which is what makes one core serve both tiers.

Plain `sqlite3` with a thin `Store` class and a `PRAGMA user_version` migration ladder. **No
SQLAlchemy** — `Store` is already the seam if hosted ever needs Postgres.

`<workspace_root>/index.db`, *beside* the dataset sandboxes, never inside one — so it is never
zipped by `make_zip` and never served.

### 3. Two storage modes, one write path

- **managed** (upload, and all hosted): the DB is the truth; sidecar files are produced by export.
- **attached** (local "open this folder"): DB is canonical for the session, but every save is
  **write-through** to the sidecar XML/JSON. This is what delivers "my XML gets updated with the
  box I just drew" while keeping reads fast.

Attached is local-only; hosted sets `DATAFACTORY_ALLOW_ATTACHED=0`.

### 4. Heavy tools live behind a process boundary — always, including locally

The base wheel never imports torch, and never imports a module that does. `import torch` costs
3–8s and ~500MB RSS; a CUDA OOM or native segfault would kill the whole server. In a sidecar it
kills the sidecar, the editor says "SAM unavailable", **and the user keeps drawing boxes**.

One `Tool` protocol, NDJSON messages, three implementations behind it: `NullTool` (base wheel),
`LocalSidecar` (subprocess, user installs the separate `data-factory-tools` distribution),
`RemoteTool` (hosted GPU service). Same messages over stdio and HTTP.

**A heavy tool's absence is never an error.** `GET /api/tools` returning `{"tools": []}` is a
normal, tested state. Buttons render only when a tool is listed *and* healthy.

### 5. The editor is hand-written ES modules — with one rule: no `innerHTML`

No build step, so the wheel stays self-contained and `pip install` works offline forever. The
problem with the current UI is not vanilla JS, it is `innerHTML` string concatenation with inline
`onclick` calling globals — which is why `refreshInputs()` (`static/index.html:365`) has to
capture and restore field values by hand.

The editor uses `<template>` + `cloneNode` + `textContent` + `addEventListener`. A canvas needs a
retained scene model, not a VDOM — you never diff a canvas. It lives on its own route with its
own module graph, so swapping in a framework later touches nothing else.

---

## Phases

### Phase 0 — De-risk (4 small independent spikes, no UI commitment)

1. **`Shape.as_dict`/`from_dict` + `Shape.confidence`.** Round-trip property test through VOC.
   `Annotation.as_dict()` (`core/annotations/model.py:132`) is **verified dead code** — zero
   callers repo-wide — and emits `polygon_points` as a *count*. Replace it, don't parallel it.
2. **Canvas spike.** 4000×3000 image, 300 shapes, pan/zoom/drag with real hit-testing at 60fps.
   If vanilla canvas cannot do it, decision 5 changes — find out in a day, not in month three.
3. **EXIF at ingest.** *Verified: not currently live* — all 3,000 sampled V3/V2 images have
   orientation tag 1. But `imaging.image_size()` uses PIL's `im.size`, which is the **stored**
   size and ignores the tag, while browsers auto-apply it. The first phone photo a public tool
   receives draws every box 90° off. Bake orientation on ingest; test with 20 real phone photos.
4. **Fake sidecar.** subprocess + NDJSON, a tool that sleeps 200ms. Prove request/reply matching
   by id, timeouts, crash-restart, and empty tool list with nothing installed. **No torch.**

### Phase 1 — Manual annotation, end to end

The product. Boxes and polygons, both hand-drawn.

**Create**
```
data_factory/annotate/{__init__,store,projects,labels,wire,paths,media}.py
data_factory/annotate/schema.sql
data_factory/service/routers/{annotate,media}.py
data_factory/core/ops/annotate_export.py
data_factory/service/static/annotate.html
data_factory/service/static/annotate/{app,api,scene,viewport,render,hit,undo,labels,filmstrip}.js
data_factory/service/static/annotate/tools/{select,box,polygon}.js
```

**Change**
- `service/workspace.py` — `.keep` marker honoured by `sweep()`; `.created` file so
  `created_at()` stops using root mtime (deep writes don't bump it, so a project annotated for
  25h is currently sweep-eligible); `sweep(*, keep=None)` keyword-only so existing calls and
  tests are untouched.
- `service/app.py` — include routers; **cache `_describe()`**, which today runs `size_bytes()`
  (full tree walk) *plus* `formats.read_dataset(recursive=True)` on every GET (`app.py:302`).
  O(dataset) per request; an editor cannot poll that.
- `core/registry.py` — one optional field `Operation.hidden_from_web: bool = False`.
- `service/binding.py` — filter it in `public_schema()` only, so `annotate.export` stays in
  `datafactory list` but does not render as a form in the batch panel (`renderOps()` has no
  filter today).
- `core/annotations/model.py` — the serialisers from Phase 0.
- `core/annotations/{voc,labelme}.py` — atomic write (temp + `os.replace`). Both currently do
  `mkdir` + `write_text`; two concurrent saves silently lose one. Fixes existing ops too.

**Reuse directly** (all already pure and synchronous — usable from a request handler):
- `core/annotations/model.py` — `Shape`, `Annotation`, **`Shape.clamped(w,h)`** is exactly the
  server-side validation every save needs.
- `core/annotations/{voc,yolo,coco,labelme,csvfmt}.py` — read/write, unchanged.
- `core/imaging.py` — `image_size` (header only, no decode), `load` (Windows-unicode-safe),
  `encode` (bytes, no disk — the thumbnail primitive), `make_resize(...)` returning a `Transform`
  with **both** `.image(im)` and `.point(x,y)`, `exif_orientation`/`apply_exif`/`exif_transform`.
- `service/workspace.py` — `Workspace.assert_inside` stays the **sole** path authority.
- `service/jobs.py` — `JobManager` unchanged, for export.

**Schema essentials** (full DDL in `schema.sql`):
`project` (+ `retention` = keep|ttl, `storage` = managed|attached, `owner_id` default `'local'`),
`label` (`(project_id, idx)` — **idx immutable, never reused, never reordered**), `image`
(opaque id, `rel_path` POSIX-relative, oriented w/h, `seq`, `state`, `version`), `shape`
(one row per shape, `origin`, `confidence`, `edited_at`), `model_run`, `audit`, plus
`user`/`session`/`member` created-but-unused.

> **Every store query is scoped `WHERE owner_id = ?` from day one**, and every router takes
> `principal = Depends(auth)` even locally, where it returns a fixed local principal.
> Retrofitting that in Phase 5 would be a rewrite; writing it now costs a WHERE clause.

**Class-index stability is a correctness fix, not a nicety.** `formats._write_yolo` builds
`yolo.ClassList(list(dataset.labels))` fresh per export and `Dataset.labels` falls back to
`sorted(found)` — so adding one class mid-project shifts every index and silently re-labels
previously exported `.txt` files. The ordered `label` table fixes it; `annotate.export` takes an
explicit `classes` param. **Its output param must be named `output`** —
`binding.OUTPUT_NAMES` is a hardcoded `frozenset({"output","quarantine"})`.

**Interaction design (this is the product):**
- Drop folder → drawing. **Labels are created while annotating**: draw, type a name, Enter.
  No schema step, ever.
- Left-drag = new box in the current class. No mode to enter. `P` = polygon; click vertices,
  Enter/double-click/click-first to close; every vertex then draggable, Alt+click deletes one,
  edge midpoint inserts one.
- **Number keys pick the class**, always. Largest throughput affordance in any annotator.
- Autosave debounced 400ms + on navigate. No Save button (`Ctrl+S` flushes anyway).
- **Resume is the default** — opening a project lands on the first `todo` image.
- Wheel = zoom at cursor, space/middle-drag = pan, `F` fit, `0` 1:1. Neither legacy tool has zoom.
- `Ctrl+D` copies the previous image's shapes. For video frames, worth more than SAM.
- `?` overlay lists every shortcut. CVAT's shortcuts are a wiki page.
- **Undo is a command stack** (do/undo pairs, depth 200), client-side, so it survives a failed
  save — saves are diffs against `version`, `409` returns server state to reconcile.

**Colour:** store `#rrggbb` per label in the DB, assigned server-side. Do **not** reuse
`ops/visualize.PALETTE` in the front-end — it is **BGR** and would swap red and blue in CSS.

**Done when:** someone annotates 1000 images cold with no documentation, closes the browser,
reopens, and lands on image 1001; exporting to YOLO twice with a class added in between produces
*identical indices* for the original classes; 327 existing tests green plus ~60 new.

### Phase 2 — Model-in-the-loop review (the killer feature)

**Confidence is a genuine gap** — nothing in the codebase models it.

**Change**
- `core/annotations/coco.py` — read `ann["score"]`; support the **COCO detection-results form**
  (a bare JSON *list*, what torchvision/mmdet/Ultralytics eval scripts actually emit).
  `to_annotations` calls `.get("categories")` and would raise on a list today.
- `core/annotations/yolo.py` — accept a 6th column (`cls cx cy w h conf`), what `--save-conf`
  writes. `yolo.read` currently raises.
- `core/annotations/voc.py` — optional `<score>`, written only for `origin='model'`.

**Create** `annotate/review.py`, `POST /api/projects/{id}/predictions`, dashed rendering,
confidence slider, `Space`-to-accept, agreement metrics.

- Import matches by **filename stem** and returns a **matching report**
  (`842 matched, 158 unmatched: …`). Silently importing zero shapes because names didn't line up
  is the dominant failure mode and must be impossible.
- **Model shapes are dashed, human shapes solid** — reads at any zoom, survives colour-blindness.
  Opacity scales with confidence. Dragging a model box sets `edited_at` and **turns it solid**.
- **`Space` accepts everything visible, marks reviewed, advances.** 1000 images ≈ 15 minutes.
- Confidence slider filters client-side, no round trip.
- `image.state`: `todo → predicted → in_progress → reviewed`, plus **`skipped`** (an explicit
  "I can't tell", which must never silently become training data).
- **Export defaults to `reviewed` only.** Exporting unreviewed model output back into training is
  how a model poisons itself, silently. Overrides are explicit and named.
- Headline project metric: **model agreement %** — it tells you when to retrain and when to stop.

**Done when:** import 1000 predictions, review at >3 images/min, export reviewed-only, and the
agreement figure moves as you correct things.

### Phase 3 — Heavy tools, SAM first

**Create** `annotate/tools/{protocol,registry,local,remote,null,postprocess}.py`, `/api/tools`,
a generic tools panel, and the separate **`data-factory-tools`** distribution.

- **Separate distribution, not an extra.** An extra still puts torch in the *same* environment,
  reviving the version-matrix problem. Discovery order: `DATAFACTORY_TOOLS_PYTHON` →
  entry-point group → `~/.datafactory/tools/` → unavailable.
- **The current `sam` extra is broken and must be fixed or deleted** — `pyproject.toml:58` pins
  `sam2>=1.0`, which is not published under that name; and the legacy script passes an absolute
  Windows path to `build_sam2`, which Hydra rejects.
- **Encode-ahead**, taken wholesale from the legacy tool (its one good idea): `prepare()` current
  + next on image open, LRU of embeddings in the sidecar, click → cached decode ≈ 50ms. Never
  block a handler on a cold prepare — return `not_ready` + `retry_after_ms`.
- **Every request gets exactly one reply, including on exception** — a rule with a test. The
  legacy worker's `except: print(...)` with no reply is why its UI deadlocked.
- **Post-processing is ours**, in the base package (pure OpenCV): contours →
  **`cv2.approxPolyDP`** → vertex cap → `Shape.clamped`. The legacy tool has *no* simplification,
  which is why its XML runs to hundreds of KB per image.
- Manual polygon drawing stays available always, for when a model gives bad masks.

**Done when:** with the sidecar absent everything works and `/api/tools` is empty; with it
present, box→mask returns a *simplified* polygon in <150ms warm; killing it mid-session degrades
to manual with a visible reason.

### Phase 4 — Inference in the loop

`detect.infer` as a **batch** tool over the same protocol — point at a checkpoint, run over
unreviewed images, write a `model_run`. Closes the owner's loop without leaving the tool. Ship
`recipes/model_loop.yaml` chaining `annotate.export` → `convert.dataset` → the analyze ops.

### Phase 5 — Hosted

Users, sessions, tokens, quotas, per-tenant GPU service. Additive, because `owner_id` scoping
was written in Phase 1. Password hashing via `hashlib.scrypt` — **stdlib, no new dependency**.

**Two resource classes, two authorisation rules.** Today possession of a 32-hex id grants read
*and delete* (`app.py:181`). Defensible for a 24-hour anonymous upload; not for a project someone
spent 40 hours annotating. Anonymous datasets keep id-as-capability; projects require a principal.

**Done when:** two accounts cannot see each other's projects (tested as a *refusal*, in the style
of the existing security tests) and the local path still requires no login.

---

## Compatibility

- **The 27 operations are untouched.** One optional `Operation` field with a behaviour-preserving
  default, plus one new op.
- **The anonymous upload flow is unchanged.** New: a "Start annotating" button that *promotes* an
  anonymous dataset into a project, flipping retention to `keep`. **Promotion is the conversion
  funnel** — a stranger uploads a zip, clicks annotate, and only then is asked to sign in.
- **`static/index.html` is not refactored.** Its `/api/operations` form generator is good and is
  not the bottleneck.
- **annotationfree4u.com**: keep the browser annotator live; ship the new editor alongside;
  redirect once it has resume, zoom and edit-existing-shape. The old tool exports VOC and we read
  VOC, so "upload your existing work and keep going" works on day one.
- **Honesty:** "images stay in the browser" stops being true for the *hosted* editor. Say so
  plainly on that page. `datafactory serve` stays the honest answer, and that is a feature.

---

## Critical files

| File | Why |
|---|---|
| `data_factory/core/annotations/model.py` | `Shape`/`Annotation` serialisers, `confidence` |
| `data_factory/service/workspace.py` | `.keep`, `.created`, `sweep(keep=)`, cache sweep |
| `data_factory/service/app.py` | routers, cached `_describe` |
| `data_factory/service/binding.py` | `hidden_from_web` filter; `OUTPUT_NAMES` landmine |
| `data_factory/core/annotations/formats.py` | class-index stability at export |
| `data_factory/core/registry.py` | one optional `Operation` field |
| `data_factory/core/annotations/{coco,yolo,voc}.py` | confidence ingest (Phase 2) |

---

## Risks, ranked

1. **Editor feel.** Everything else is plumbing. Spike in Phase 0, then dogfood on the real
   1000-image door set *before* building the review layer on top.
2. **Class index stability.** The worst bug on the list — it corrupts work already exported.
   Build the ordered label table first, with the export→add-class→export regression test on day one.
3. **Coordinate space / EXIF.** Silent and data-corrupting. Not live today (verified), but it
   arrives with the first phone photo. Handle at ingest before any public launch.
4. **The heavy-tool boundary.** Cheap now, expensive later. The fake sidecar settles it in Phase 0.
5. **Scope creep toward CVAT.** Tasks, jobs, assignees, orgs — the moment those arrive, the only
   advantage is gone.

---

## Verification

**Per phase**
- `pytest` — 327 existing must stay green throughout; each phase adds its own.
- Phase 1: new `tests/test_annotate_store.py`, `test_annotate_api.py`, `test_export_classes.py`.
- Security tests written as **refusals**, matching `tests/test_service.py` (path params refused,
  cross-owner access refused, traversal ids refused).

**End to end, on the real door dataset**
1. `datafactory serve`, open a folder of V3 images in attached mode.
2. Annotate 50 images by hand — boxes and polygons. Close the browser mid-image; reopen; confirm
   it resumes on the right image with the work intact, and that the **sidecar XML on disk** was
   updated in place.
3. Export to YOLO. Add a class. Export again. Assert original class indices are unchanged.
4. Feed the existing 3,686-image V3 predictions in as a `model_run`; confirm the matching report
   is accurate; review with `Space`; confirm agreement % moves as boxes are corrected.
5. Export reviewed-only → `convert.dataset` → `inspect.health` → `inspect.leakage`. Expect zero
   errors, as the current pipeline already achieves.
6. With no sidecar installed: confirm `/api/tools` is empty and the editor is fully functional.

**Performance gates**
- Image open to first paint < 300ms warm.
- Pan/zoom/drag at 60fps with 300 shapes on a 4000×3000 image.
- Save round trip < 100ms.
- `GET /api/projects/{id}/images` for 20,000 images < 200ms (this is why `_describe` gets cached
  and why the image list is DB-backed rather than an `fs.iter_images` scan).

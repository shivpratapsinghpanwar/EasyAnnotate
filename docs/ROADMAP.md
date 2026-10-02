# Roadmap: from 52 to 75

> Written 2 October 2026, after Phases 0–4 of `PLAN.md`. `PLAN.md` explains how
> the system is built; this file says what to build next, and in what order.

## Where we stand

A weighted scorecard (0–10 per dimension, weights sum to 100) against CVAT,
Label Studio, Roboflow, X-AnyLabeling and Labelme. EasyAnnotate is scored on
what runs today, not on what is designed.

| Dimension (weight) | Now | Target | CVAT | Roboflow | X-AnyLabeling |
|---|---|---|---|---|---|
| Time to first box (8) | 9 | 10 | 3 | 7 | 7 |
| Core editor (13) | 6 | 8 | 8 | 8 | 7 |
| Annotation types (7) | 3 | 6 | 10 | 7 | 9 |
| AI assist (13) | 2 | 8 | 8 | 9 | 10 |
| Model-in-the-loop review (12) | 7 | 9 | 6 | 7 | 4 |
| Data correctness & formats (10) | 8 | 9 | 8 | 9 | 8 |
| Dataset prep & health (8) | 9 | 9 | 3 | 8 | 3 |
| Performance at scale (6) | 5 | 7 | 7 | 7 | 6 |
| Video (4) | 1 | 2 | 9 | 6 | 8 |
| Collaboration (5) | 1 | 1 | 9 | 9 | 1 |
| Privacy / local / cost (4) | 10 | 10 | 7 | 3 | 10 |
| Maturity & distribution (10) | 1 | 6 | 10 | 10 | 8 |
| **Total / 100** | **52** | **75** | **73** | **79** | **69** |

## The position

Not "a CVAT replacement". **The best local tool for one ML engineer making a
detector better:** open a folder, review the model's mistakes fast, get a
dataset out that is correct by construction, retrain. Nobody in the field
treats that loop as the product; we already have its hardest parts (stable
class indices, reviewed-only export, agreement metric, dataset health).

Collaboration and video stay deliberately low. Chasing them is how the
"no configuration before the first box" rule dies.

## Order, and why

Trust first, because a tool that loses work cannot be recommended at any score.
Then the editor essentials, because the first public impression is the editor.
Then ship — a product nobody can install scores 1 on maturity forever, and real
users will re-rank everything after this point. Then AI assist and the loop,
which are the bulk of the points.

### M0 — Trust: nothing is lost, nothing is wrong *(correctness 8→9, perf 5→6)*

1. Save race: version check and the existing-shape read happen outside the
   write lock (`annotate/shapes.py`). Check inside `BEGIN IMMEDIATE`, and
   guard the version `UPDATE` itself.
2. Stem matching: `train/001.jpg` and `val/001.jpg` collide, first wins, and
   attached mode writes the wrong shapes into the second file. Match on the
   relative path first; fall back to stem only when unique; report ambiguity.
3. `overwrite_reviewed` leaves a re-predicted image `reviewed`, so raw model
   output reaches the reviewed-only export. Send it back to `predicted`.
4. Attached write-through runs only on shape save. Accept, label rename /
   retire / merge and prediction import leave the files on disk stale.
5. Renaming a label to a retired label's name duplicates a class-list name and
   silently merges two classes on export.
6. Docker: the annotation DB and project media land outside `/data`
   (`data_root()` is the workspace's parent). Lost on restart.
7. Upload size is enforced after Starlette has spooled the body to disk; sync
   extraction and import run on the event loop and freeze every save.
8. Decompression bombs: cap decoded pixels for thumbnails and media.
9. Migrations: `executescript` commits implicitly; make each step atomic.

**Done when** each has a test that fails on the old code, and the full suite is green.

### M1 — Editor essentials *(editor 6→8, perf →7)*

Crosshair guides while drawing · arrow-key nudge (Shift ×10) · Tab to cycle
overlapping shapes · hide/lock per shape and class, with a shape list ·
class palette with type-to-search beyond ten classes · copy/paste within and
across images · brightness/contrast · trackpad two-finger pan, pinch zoom ·
virtualised filmstrip with paging and a state filter (removes the 2,000-image
cap) · shape attributes (occluded, truncated, free tags) — the honest fix for
`Open/open`-style class splits.

**Gate:** the owner annotates 200 real door images. Their notes outrank this list.

### M2 — Ship v0.1 *(maturity 1→5, first box 9→10)*

Settle the name (repo, package and domain are three names today) · claim it on
PyPI · CI on GitHub Actions including the Playwright editor tests ·
`easyannotate demo` opens a bundled sample project in the browser in one
command · run the Dockerfile, the wheel and the PyInstaller build for real ·
README with a 20-second GIF and an honest comparison page · make the repo public.

### M3 — AI assist *(AI 2→7)*

A real sidecar distribution behind the existing protocol, installed by the
tool itself into its own venv (`easyannotate tools install sam`), never into
ours:

- Click and box prompts → mask → our simplified polygon (SAM 2.1 / SAM 3).
- **Text prompt → every instance, as predictions in the review queue** (SAM 3
  concepts or YOLO-World). This is where our review loop beats everyone's AI
  button: the output arrives dashed, filterable, one key to accept.
- CPU path (ONNX MobileSAM/EfficientSAM) so it works without CUDA.
- Encode-ahead for the next image, as planned in `PLAN.md` Phase 3.

### M4 — Close the loop *(loop 7→9, AI 7→8)*

`detect.infer`: any Ultralytics `.pt` over unreviewed images into a
`model_run` · "train on reviewed" via Ultralytics in the sidecar ·
**review queues ordered by likely error** — low confidence, model/label
disagreement, class confusion, probable missed objects. That last one is what
FiftyOne and Lightly sell separately; here it is the default order of work.

### M5 — Breadth *(types 3→6, video 1→2)*

Rotated boxes (YOLO-OBB, DOTA) · keypoints (YOLO-pose, COCO keypoints) ·
image-level classification tags · keyframe interpolation across frame
sequences. Each through the existing neutral model, so every format and every
dataset operation gets it.

## Rules that do not change

- Nothing ships that needs configuration before the first box.
- Every bug fix lands with a test that failed before it.
- Heavy dependencies stay behind the process boundary. The base wheel never imports torch.
- Export stays reviewed-only by default.

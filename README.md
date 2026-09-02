# Data Factory

Dataset preparation for computer vision — cleaning, arranging, verifying and
exporting — as **one tool** with a library, a CLI, an interactive menu, and a
YAML pipeline runner over a single shared core.

*Author: Shivpratap Singh Panwar (Kody Technolab Public Limited). Originally
created 22 July 2026 as a folder of task-specific scripts; consolidated into
this package.*

---

## Why one tool

The original `Power_tools/` folder held fifteen scripts. Each hardcoded its
paths at the top, each re-implemented "find the image that matches this XML",
and they had quietly drifted apart — one moved sidecar labels alongside
duplicates, another had that code commented out. Same job, different answers.

Everything here runs through one core. Pairing an image with its label, previewing
before writing, and reporting what happened are solved once, so no operation can
forget them.

```
data_factory/
  core/            pure logic — no printing, no argparse, no hardcoded paths
    types.py         OpResult / FileAction / Context (progress + cancellation)
    registry.py      every operation declares its own parameters
    pairs.py         the image + sidecar abstraction all 15 scripts re-invented
    fs.py            the single choke point every file mutation passes through
    bktree.py        metric tree behind fast perceptual dedup
    annotations/     neutral model + Pascal VOC reader/writer + COCO builder
    ops/             the operations themselves
  cli/             argparse commands, interactive menu, terminal rendering
  pipeline.py      YAML/JSON recipes
```

An operation declares its parameters **once**. The CLI subcommand, the menu
prompts, the recipe validator and the web form schema are all generated from that
declaration — adding an operation means writing one file, and it appears
everywhere.

---

## Install

```bash
uv sync                 # or: pip install -e .
datafactory --version
```

Python 3.12+. The core needs only numpy, Pillow, OpenCV, imagehash and PyYAML —
deliberately no PyTorch, so a CI box or web backend can install it cheaply. The
SAM-assisted annotator is an optional extra: `pip install -e ".[sam]"`.

---

## Three ways to drive it

**1. Interactive menu** — run it bare. Every path is a prompt, and destructive
operations preview before they ask.

```bash
datafactory
```

**2. Command line** — for scripting and repeatability.

```bash
datafactory list                          # every operation
datafactory explain clean.dedup           # full help for one

datafactory inspect.dataset --folder ./data
datafactory clean.dedup --folders ./a ./b --method phash --threshold 8 --dry-run
datafactory export.coco --folder ./voc --output ./coco
```

**3. Recipes** — a dataset build written down, so it is reproducible.

```bash
datafactory run recipes/voc_to_coco.yaml --set root=D:/data --set out=D:/build --dry-run
```

---

## Operations

| Operation | What it does |
|---|---|
| `inspect.dataset` | Full audit: pairing, classes, geometry, corrupt files, duplicates |
| `inspect.coco` | Summarise an `instances.json`, flag case-collided categories |
| `labels.list` | Count classes and objects, surface case-variant class names |
| `labels.rename` | Bulk rename/drop classes; `--fold-case` unifies case drift |
| `clean.dedup` | Exact or perceptual duplicates, sidecar labels moved too |
| `clean.orphans` | Report, quarantine or delete unpaired images and labels |
| `clean.unannotated` | Move images with no label into an "still to annotate" queue |
| `clean.rename` | Sequential renaming, pair-safe, retargets the label's filename |
| `dataset.split` | Reproducible, stratified train/val/test split |
| `dataset.sort` | Group pairs into one folder per class |
| `verify.visualize` | Draw boxes and polygons onto copies, colour-coded by class |
| `export.coco` | Pascal VOC → COCO, with segmentation and hard negatives |
| `collect.frames` | Video (or a folder of videos) → numbered JPEG frames |
| `collect.webcam` | Capture stills from a camera, manually or on a timer |

Operations marked `!` in `datafactory list` write to disk. All of them accept
`--dry-run`.

---

## Three ideas worth knowing

**Pairs.** An image and its `.xml` are one thing. Every operation moves, renames
and splits them together — that is a property of the core, not something each
operation remembers to do.

**Dry run is the same code path.** Operations build their full list of
`FileAction`s either way; `--dry-run` only stops the last step from executing. A
preview is therefore a faithful description of the real run, not an
approximation of one.

**Results are data.** Every operation returns an `OpResult` with counts, an audit
trail, warnings and structured payload. The terminal renderer is one consumer of
that; a web API is another.

---

## As a library, and in a web app

```python
import data_factory as df

result = df.run("inspect.dataset", {"folder": "./data"})
print(result.counts["objects"])
for issue in result.data["issues"]:
    print(issue["severity"], issue["issue"], "->", issue["fix"])

payload = result.as_dict()          # JSON-serialisable, ready to return over HTTP
```

Long runs report progress and can be cancelled — designed for a request that
outlives a single HTTP round trip:

```python
from data_factory import Context

ctx = Context(
    dry_run=False,
    progress=lambda done, total, msg: socket.send({"done": done, "total": total}),
    cancel_check=lambda: job.is_cancelled(),
)
df.run("clean.dedup", {"folders": ["./data"], "method": "phash"}, ctx)
```

A front-end can build its own forms without importing Python at all:

```bash
datafactory schema --output schema.json
```

That file describes every operation, every parameter, its kind (`dir`, `int`,
`choice`, …), defaults, and whether it is destructive. See
`examples/web_api.py` for a working FastAPI wrapper in about eighty lines.

---

## Adding an operation

One file, one decorator. It then appears in the CLI, the menu, the schema and
the recipe validator with no further wiring.

```python
from ..registry import Param, operation
from ..types import Context, OpResult

@operation(
    "clean.mything",
    title="My thing",
    category="clean",
    summary="One line shown in listings.",
    params=(Param("folder", "dir", "Folder to work on.", required=True, must_exist=True),),
    destructive=True,
    safe_when=lambda p: p.get("action") == "report",
)
def mything(params: dict, ctx: Context) -> OpResult:
    result = OpResult(op="clean.mything", dry_run=ctx.dry_run)
    ...
    return result
```

Then add it to the imports in `data_factory/core/ops/__init__.py`.

---

## Where the old scripts went

| Original script | Now |
|---|---|
| `CrossFolder_Duplicate_Removal_WithXML.py` | `clean.dedup --method exact` |
| `Remove_Duplicates_OneFolderOnly.py` | `clean.dedup --method phash` |
| `del_orphan_img_xml.py` | `clean.orphans --action delete` |
| `rename_labels_andFindOrphans.py` | `clean.orphans` + `labels.rename` |
| `Move_pairs.py` | `clean.unannotated` |
| `rename_files_and_xmls.py` | `clean.rename` |
| `Find_label_PVOC.py` | `labels.list` |
| `video_to_frames.py` | `collect.frames` |
| `live_webcame_image_saver.txt` (unfinished) | `collect.webcam` |
| `coco_dataset_generators.py` | `export.coco` |
| `draw_BB_from_xml_toNewFolder.py` | `verify.visualize --draw bbox` |
| `Crosscheck_polygon_coco_data.py` | `verify.visualize --split-by-type` |
| `Crosscheck_finishedData_pascalvoc.py` | `dataset.sort` + `verify.visualize` |

The annotation GUIs (`Polygon_Advanced_Annotation_Tool.py`,
`Sam2.1_segmentation.py`) stay as they are — they are interactive OpenCV apps,
not batch operations.

### Bugs fixed along the way

- **COCO exported four classes where there were two.** Categories were keyed on
  the raw XML string, so `Open` and `open` became separate classes — splitting
  Project_Door V3 as 1341/1385/645/500. Export now folds case by default and
  *reports* every merge; `labels.rename --fold-case` fixes it at the source.
- **SAM polygons exported as empty masks.** The two annotation tools write
  different polygon XML (`<x1>` vs `<pt><x>`) and the converter only read one.
  The reader now accepts both.
- **Renaming could clobber files.** Renaming through per-file temporary names
  still collides when a target belongs to a file later in the queue. It is now
  done in two global passes.
- **Renaming left labels pointing at the old image.** The `<filename>` and
  `<path>` fields inside the XML are now retargeted.
- **Perceptual dedup was O(n²).** ~112 million comparisons on a 15k-image set.
  A BK-tree cuts it to roughly logarithmic per lookup.
- **Duplicate removal could discard the annotated copy.** `keep=annotated` now
  keeps the labelled twin by default.
- **COCO `area` was always the box area,** even for polygons. It is now the true
  polygon area.
- **Train/test splitting was documented but never implemented.** It exists now,
  stratified and seeded.

---

## Tests

```bash
uv run pytest          # 114 tests
```

Fixtures build a small dataset carrying every defect the real one had: case
variant classes, both polygon dialects, a hard negative, an exact duplicate,
orphans of both kinds and a corrupt XML.

---

## Datasets in this repository

`Projects/Project_Door/` holds the door-state dataset (V3: 3686 images, 3871
objects, 2726 with segmentation) and `Models/Sam2.1_tiny/` the SAM 2.1 tiny
checkpoint. Neither is tracked by git.

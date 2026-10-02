# Working on this repo

Read this first. It is the state of play, not documentation — `README.md` is the
documentation.

## What this is

A computer-vision dataset tool that has grown into an **annotation platform**.
Repo `EasyAnnotate`, Python package `data_factory`, intended home
annotationfree4u.com. (Three names for one product; unresolved.)

The competitive position, which everything is subordinate to:

> **If a feature requires the user to configure something before they can draw a
> box, it does not ship.**

CVAT makes you create an organisation, a project, labels, a task and jobs before
the first box. Removing that is the entire product argument.

---

## Git identity

- Commits use the repo-local identity
  `Shivpratap Singh Panwar <106022228+shivpratapsinghpanwar@users.noreply.github.com>`.
  Never change git configuration outside this repository.
- Commits carry the owner's name only: never a co-author line.
- Never ask for or accept a token or password in chat.
- Machine-specific notes live in `CLAUDE.local.md`, which is never committed.

---

## State

**Phases 0–4 of `docs/PLAN.md` are done, and M0 ("trust") of
`docs/ROADMAP.md`.** 637 tests pass, including 13 browser tests (install
playwright to run them). **The roadmap is the plan of record now** — a scored
path from 52 to 75 against CVAT, Roboflow, X-AnyLabeling and others.

Commits carry the owner's name only: never a Claude co-author line.

Working today: format conversion between 5 formats; cleaning; annotation-aware
geometry; analysis; a sandboxed HTTP service; the **annotation editor** (boxes,
polygons, undo, autosave, resume); **attached projects** (`datafactory annotate
<folder>` edits a folder's own label files in place); the **model loop** (import
predictions, confidence filter, one-key accept, agreement metric); and the
**heavy-tool process boundary** with mask→polygon simplification.

**The correctness fix everything rests on:** YOLO class indices come from a
per-project ordered label table with a monotonic counter. Renaming keeps the
index, retiring keeps the slot, new classes append. It used to be
`sorted(found)`, which silently re-labelled already-exported `.txt` files.
`tests/test_export_classes.py` guards it. Do not regress this.

## Not done / unverified

- **PyInstaller binary** (`packaging/`) and **PyPI publish** — written, never
  run. The Dockerfile was in this category too and turned out to be broken
  (a shell character-class bug in `*.whl[web]`). Expect the same here: **test
  these by running them, not by reading them.**
- **Phase 5 (hosted)** — accounts, quotas, per-tenant isolation. Not started.
  `owner_id` scoping already exists throughout, so it is additive.
- **`detect.infer`** — training/inference still happens outside the tool. This
  is probably the highest-value remaining feature; the loop is what is
  monetisable, not the annotator.
- **The user has not yet used the editor properly.** The plan ranks "editor
  feel" as risk #1. Their feedback outranks new features.

---

## Rules the code keeps

- `core/` is pure: no printing, no argparse, no HTTP imports.
- `annotate/` must not import fastapi.
- **Clients never send filesystem paths.** Path-kind params are *refused, not
  sanitised*. Attached projects are CLI-only for exactly this reason.
- Every operation declares its params once (`core/registry.py`); CLI, menu,
  recipe validator and web form are all generated from that.
- Editor JS: no `innerHTML`. `<template>` + `cloneNode` + `textContent`.
- `annotate/tools/` = model sidecars. `static/annotate/tools/` = drawing tools.
  `static/annotate/assist.js` = the sidecar panel. These were all called
  "tools" once; do not re-merge them.
- Export defaults to **reviewed images only**. Feeding unreviewed model output
  back into training is silent self-poisoning.

## Running it

```bash
.\.venv\Scripts\activate
datafactory serve                 # http://localhost:8765
datafactory annotate <folder>     # edit a folder's labels in place
python -m pytest                  # 637; 13 browser tests skip without playwright
```

Real data lives in `Projects/Project_Door/` (15,514 images, gitignored). The V3
set carries a genuine defect: case-variant classes `Close/close/Open/open`.

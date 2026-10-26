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

**Phases 0–4 of `docs/PLAN.md` are done; of `docs/ROADMAP.md`, M0 (trust),
M1 (editor essentials), M2 (shipping: `easyannotate` CLI, demo, wheel and
binary run), M2.5 (desktop window), M3 (AI assist: SAM box/click, YOLO-World
text prompts, batch prompt → predictions, SAM encode-ahead), M4 (the loop:
`easyannotate train/predict`, uncertainty-ordered review queue) and the
rotated-box/line/point half of M5.** 788 tests pass, including 32 browser tests
(install playwright). `docs/bench_scale.py` measures 20k images: all gates pass.

**Three front ends, as the owner asked (Oct 2026):** (1) the browser bound to
this computer -- the token-fenced local link opens folders from the front
page; (2) headless -- the public `easyannotate` library (`ea.open_folder`)
and `serve --headless`; (3) the desktop app, which carries uv and both
wheels so "Install AI tools" works with no Python, plus Train/Predict in the
editor. The showcase site is `site/`, live at
https://shivpratapsinghpanwar.github.io/EasyAnnotate/ (`pages.yml`).

AI tools live in `tools/` as the separate `easyannotate-tools` distribution,
installed into `~/.datafactory/tools` by `easyannotate tools install` (from
PyPI, pinned to the editor's version; `--source .` from a checkout). Tests set `DATAFACTORY_TOOLS_HOME` to an empty dir so they
never start the real sidecar. The real models were driven through the
editor in a browser from cold (CLIP's 338 MB download included: 46 s to the
first text-prompt result), and `easyannotate train` ran for real on CPU (it
found a crash the fake trainer could not -- keep running these by hand after
touching `tools/`; they are outside pytest on purpose). **The roadmap is the plan of record now** — a scored
path from 52 to 75 against CVAT, Roboflow, X-AnyLabeling and others.

**GPU, measured on this laptop (RTX 2050 4 GB, driver 529.04):** PyTorch's
cu126 build reports CUDA available and fails every allocation; 2.7.1+cu118
works. `hardware.plan()` now picks builds by each toolkit's own driver and
the installer runs real work on the device before trusting it. SAM 0.044 s
a click, a new text prompt ~1 s (after caching CLIP), training via
`loop.train` on device 0. Ultralytics prints to stdout: the sidecar keeps
fd 1 for replies (`serve._private_stdout`) -- never undo that.

Commits carry the owner's name only: never a Claude co-author line.

Working today: format conversion between 6 formats (VOC, YOLO incl. OBB, COCO,
LabelMe, CSV, DOTA); five shape kinds (box, polygon, rotated box, line, point); cleaning; annotation-aware
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

- **CI (`ci.yml`) is green** (Linux and Windows, wheel, Docker). **v0.4.0 is
  released** by `release.yml` (tag `v0.4.0`): desktop zips for Windows, macOS
  and Linux, each smoke-tested in CI, unsigned. macOS produces a folder, not
  a `.app`.
- **The repository is public** (since 2026-10-15). **Both packages are on
  PyPI**, published from `release.yml` by trusted publishing (no token
  anywhere), each from its own job and GitHub environment, approved by the
  owner: job `pypi` → environment `pypi` → project `easyannotate`; job
  `pypi-tools` → environment `pypi-tools` → project `easyannotate-tools`.
  Keep one publisher per PyPI project, naming that environment. A
  publisher added on a project's page can only upload to that
  project; a new project needs a *pending* publisher from the account page.
  The wheel (`uv build`) installs into a clean venv and runs; the
  PyInstaller binary (`pyinstaller packaging/easyannotate.spec`) boots and
  serves the editor. Docker is CI-tested but has not run on this machine.
  Rule unchanged: **test these by running them, not by reading them.**
- **Phase 5 (hosted)** — accounts, quotas, per-tenant isolation. Not started.
  `owner_id` scoping already exists throughout, so it is additive.
- **Train/Predict buttons exist** (local only) but have not been clicked
  through with the real trainer; `loop.train` on the GPU has, from Python.
- **The user has not yet used the editor properly.** The plan ranks "editor
  feel" as risk #1. Their feedback outranks new features.

---

## Rules the code keeps

- `core/` is pure: no printing, no argparse, no HTTP imports.
- `annotate/` must not import fastapi.
- **Clients never send filesystem paths.** Path-kind params are *refused, not
  sanitised*. The one exception is the fenced local route below.
- **One route takes a path: `POST /api/local/open`** (`routers/local.py`),
  and only behind all the fences: a per-launch token (minted by the CLI or
  desktop launcher, sent in the URL `#fragment`, then as `X-Local-Token`;
  never inherited, never passed to a child process), a strictly parsed
  localhost `Host`, a loopback client address, an `Origin` equal to the
  server's own, and local mode (never with an API key, a non-localhost
  bind, or `DATAFACTORY_ALLOW_ATTACHED=0`). While bound to loopback,
  `LocalHostFence` refuses non-local Hosts on **every** route (DNS
  rebinding reached attached folders through the others). It returns a
  project id. Operations still never take paths from clients -- running ops
  in place on arbitrary folders from a browser is deliberately not offered.
- **No client names a model file.** A `.pt` is a pickle: loading it runs
  it. Tools of kind `"model"` are refused by every tools route and hidden from
  the editor. The one HTTP path that loads a model is the token-fenced
  `/api/local/train|predict`, and it loads only the project's own latest
  model from `models_home()` -- the page sends a project id, never a path.
  `easyannotate predict --model` is the only way to name a file.
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
easyannotate demo                 # sample project in the browser
easyannotate serve                # http://localhost:8765
easyannotate annotate <folder>    # edit a folder's labels in place
easyannotate train <folder>       # reviewed -> model -> predictions on the rest
python -m pytest                  # 788; 32 browser tests skip without playwright
```

Real data lives under `Projects/` (the door dataset, 15,514 images, gitignored). The V3
set carries a genuine defect: case-variant classes `Close/close/Open/open`.

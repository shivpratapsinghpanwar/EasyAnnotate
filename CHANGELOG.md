# Changelog

## 0.4.0 — first public release

The tool is now called **EasyAnnotate**: `pip install easyannotate`,
`easyannotate demo`. The Python package stays `data_factory`, and
`datafactory` still works as a command.

### Editor
- Guides while drawing, with the box size in image pixels beside the cursor.
- Draw anywhere — inside other boxes too. Click selects; drag a selected shape
  to move it, its handles to resize.
- Arrow keys nudge (Shift ×10); Tab cycles shapes; Ctrl+A/C/V; paste carries
  shapes to the next image.
- Find a class by typing (`/`). Typing a case variant picks the existing class
  instead of making a second one.
- Hide a class while you work; `H` hides every shape. A list of the shapes on
  the image. Occluded / truncated / difficult flags. Brightness and contrast.
- The image strip is virtual and paged: no 2,000-image cap, filter by state.
- Trackpad: two fingers pan, pinch zooms.
- Next and previous images are decoded ahead of time, off the main thread.

### AI assist
- `easyannotate tools install` sets up a separate environment with PyTorch
  and Ultralytics; the editor discovers it with no configuration.
- SAM 2.1 outlines from a dragged box or a click; YOLO-World finds every
  instance of a typed class on one image or, as a job, on every unreviewed
  image -- the results land in the review queue as predictions.
- A tool still loading or still computing answers "not ready" and keeps
  working; the editor retries quietly. CPU-only machines work.

### Annotation types
- Points (`K`, one click), lines (`L`, click points, Enter) and rotated
  boxes (`O`: drag, then turn by the handle or `[` `]`), alongside boxes and
  polygons. Corner-dragging a rotated box keeps it a rectangle.
- Rotated boxes export as YOLO-OBB and DOTA and read back from both; DOTA
  is a new format throughout (convert, inspect, export). LabelMe writes and
  reads `point`, `linestrip` and `rotation`; COCO writes a point as a
  visible keypoint; CSV carries a `kind` column.
- Shapes saved in one burst used to come back in random order after a
  reload, reshuffling what was on top. Order is now the order they were made.

### Correctness
- Two saves racing could both pass the version check and silently drop or
  resurrect shapes. Saves are serialised in the editor and checked inside the
  write transaction on the server.
- Labels were matched to images by filename alone; `train/001.jpg` and
  `val/001.jpg` collided. Matching is by path first, and ambiguity is reported
  rather than guessed.
- Autosave renamed shapes under the undo stack, so Ctrl+Z after a save did
  nothing. Fixed.
- Editor saves reset VOC's `difficult`, `truncated`, `occluded` and `pose`.
  Fixed.
- A retired class sharing a live class's name was merged into it on save and
  on export. Fixed.
- Re-predicting a reviewed image left it marked reviewed, so raw model output
  reached the reviewed-only export. It now returns to *predicted*.
- In attached mode only shape saves reached the files on disk. Accepting,
  importing predictions and every class change now write through too.

### Service
- `POST /api/projects/{id}/export`: a hosted user can download their dataset
  in any format. Reviewed images only unless asked; skipped images never.
  The editor has an Export button for it.
- Oversized uploads are refused on the declared length before any body is
  read; archive extraction and prediction import run off the event loop.
- Thumbnails refuse decompression bombs and decode large JPEGs reduced.
- The Docker image keeps its database inside the volume.

### Packaging
- `easyannotate demo`: a generated sample project with labelled, predicted and
  untouched images, importing its predictions through the real YOLO path.
- CI runs the suite, including the browser tests, on Linux and Windows;
  installs the wheel into an empty environment; boots the Docker image.

## 0.3.0 and earlier

Data Factory: format conversion, cleaning, augmentation, analysis, the HTTP
service, the first annotation editor and the model loop. See `docs/PLAN.md`.

# easyannotate-tools

The model-backed half of EasyAnnotate, kept in its own environment:

- **SAM 2.1** — drag a box or click a point, get an editable outline.
- **Text-prompted detection (YOLO-World)** — type `handle`, get every handle
  on the image, or on every unreviewed image, as predictions to review.

```bash
easyannotate tools install      # makes ~/.datafactory/tools and installs this
easyannotate tools status       # what the editor will find
```

Weights download on first use (SAM 2.1 tiny ≈ 150 MB, YOLO-World small ≈ 25 MB)
into Ultralytics' cache. A GPU is used when PyTorch can see one; everything
works on CPU, slower.

The editor never imports this. It starts `python -m easyannotate_tools.sidecar`
from the tools environment and speaks newline-delimited JSON to it; if the
process is absent or dies, the editor keeps working without it.

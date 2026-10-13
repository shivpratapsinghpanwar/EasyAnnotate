# Deploying and distributing EasyAnnotate

Four ways people will get this, in the order they matter.

---

## 1. `pip` / `uv` — the developer path

```bash
pip install easyannotate                 # everything
uv tool install easyannotate             # isolated, on PATH, no venv to manage
uvx easyannotate demo                    # run once, install nothing
```

Then:

```bash
easyannotate list
easyannotate inspect.health --folder ./data
easyannotate serve
```

**Why the default install is headless.** The dependency is
`opencv-python-headless`, not `opencv-python`. The ordinary OpenCV wheel links
against `libGL.so.1` and a GLib stack, which slim containers, Cloud Run,
Lambda and most CI images do not have — so it installs fine and then throws
`ImportError: libGL.so.1: cannot open shared object file` at the first import,
on exactly the machines a backend runs on. Only `collect.webcam`'s preview
window needs a GUI, and it degrades to timer capture with a warning when the
headless build is present. If you want the preview:

```bash
pip install "easyannotate[gui]"          # replaces the headless wheel
```

Do not install both into one environment; they provide the same `cv2` module.

---

## 2. Docker — the hosting path

```bash
docker build -t easyannotate .
docker run -d -p 8000:8000 \
  -e DATAFACTORY_API_KEY="$(openssl rand -hex 24)" \
  -e DATAFACTORY_ORIGINS="https://annotationfree4u.com" \
  -v easyannotate-data:/data \
  easyannotate
```

The image is small because the core has no PyTorch dependency — that is the
payoff for keeping SAM in an optional extra.

### Configuration

| Variable | Default | What it does |
|---|---|---|
| `DATAFACTORY_WORKSPACE` | `~/.easyannotate/workspaces` | where uploads live. The annotation database and project media go in its **parent**, so keep it one level inside your volume (the image uses `/data/workspaces`) |
| `DATAFACTORY_TTL_HOURS` | `24` | how long an anonymous dataset survives |
| `DATAFACTORY_API_KEY` | *(unset)* | if set, `/api/*` requires `X-API-Key` |
| `DATAFACTORY_ORIGINS` | *(unset)* | comma-separated CORS origins |
| `DATAFACTORY_MAX_UPLOAD_MB` | `2048` | upload cap |

### Before it faces the internet

- [ ] **Set `DATAFACTORY_API_KEY`.** Without it, anyone can spend your CPU and
      fill your disk. This is the single most important line here.
- [ ] **Put it behind a reverse proxy** (Caddy, nginx, Cloudflare) for TLS and
      rate limiting. The app does no rate limiting itself.
- [ ] **Cap the upload size at the proxy too** — `client_max_body_size` in
      nginx. The app's own cap only applies once the body reaches it.
- [ ] **Run the TTL sweep on a timer.** It runs at boot only; add a cron or a
      sidecar calling `Workspace.sweep()`, or restart daily.
- [ ] **Watch disk.** Uploads are large and expiry is time-based, not
      quota-based. Alert well before full.
- [ ] **One worker.** Jobs live in process memory, so a second uvicorn worker
      would answer polls for jobs it has never seen. Scale up the box, or move
      `JobManager` to Redis first.

### Getting work out

Hosted users export from the editor's **Export…** button, which calls
`POST /api/projects/{id}/export` — the same `annotate.export` operation the
CLI runs, as a job, into a throwaway sandbox, then
`GET /api/datasets/{sandbox}/download?output=export`. Reviewed images only
unless asked; skipped images never.

### What is already handled

The service never accepts a filesystem path — clients name an opaque dataset
id and every path is derived server-side. Uploads are guarded against zip slip,
zip bombs and member-count floods; tracebacks stay in the log rather than the
response; the audit trail is not shipped to the browser. There are 32 tests
covering these specifically, most of them asserting a refusal.

---

## 3. Desktop — `easyannotate app`

```bash
pip install "easyannotate[desktop]"
easyannotate app                   # its own window, over the same local server
easyannotate app --browser         # the same, in your browser instead
easyannotate serve                 # server only: opens http://localhost:8765
```

A local server plus a native window (pywebview — WebView2 on Windows, WebKit
on macOS and Linux), and when no window can be made, the system browser. The
real argument: **it is the same code that runs on the hosted service**, so
there is one thing to maintain rather than a desktop build and a web build that
drift apart.

One thing the window can do that a browser never may: **Open a folder…** uses
the operating system's folder dialog, and the page is handed a project id. No
filesystem path ever crosses HTTP, so the sandbox rule the service relies on
still holds.

`serve` binds to `127.0.0.1` by default, so it is not reachable from the
network. Binding to `0.0.0.0` without `--api-key` prints a warning.

### A true double-clickable build

```bash
pip install pyinstaller "easyannotate[desktop]"
pyinstaller packaging/easyannotate.spec
```

Produces `dist/EasyAnnotate/EasyAnnotate.exe` (or the platform equivalent),
which opens the window. **Run it, do not only build it**: with
`EASYANNOTATE_NO_WINDOW=1` it takes the browser path, which is how CI smokes
it headlessly. Notes that will save you a day:

- The bundle is ~290 MB, mostly numpy, OpenCV and the .NET bridge pywebview
  needs on Windows. That is normal.
- Windows Defender can briefly lock a freshly copied DLL while scanning it,
  and PyInstaller then fails with `PermissionError` on `ucrtbase.dll`.
  Rebuild; or exclude the `build/` folder from real-time scanning.
- The version is read from package metadata; the spec freezes the
  `easyannotate` dist-info in with `copy_metadata`, or the binary reports
  `0.0.0+source`.
- Windows SmartScreen warns on an unsigned binary. Code signing needs a
  certificate (~$70–300/yr, or Azure Trusted Signing); until then, publish the
  SHA-256 and tell people to expect the warning.
- macOS needs notarisation (an Apple Developer account, $99/yr) or users get
  "damaged and can't be opened".

### Releases

Pushing a tag `v*` runs `.github/workflows/release.yml`: it builds the wheel
and a desktop bundle on Windows, macOS and Linux, smoke-launches each, and
attaches them to a GitHub Release. **PyPI is a manual step on purpose** —
`uv publish` from the `wheel` artifact, by the owner.

---

## 4. As a library

```python
import data_factory as df

result = df.run("inspect.health", {"folder": "./data"})
for finding in result.data["findings"]:
    print(finding["severity"], finding["issue"], "->", finding["fix"])
```

---

## Wiring it to annotationfree4u.com

The site annotates; this prepares. They are the two halves of one workflow, and
both are local-first — the site keeps images in the browser, and this keeps
them on your machine unless you deliberately upload.

**The lowest-friction integration** is to keep the annotator exactly as it is
and add a "Prepare your dataset" page that is an iframe or a link to the hosted
service. The annotator exports Pascal VOC; the service reads Pascal VOC. No
data model has to be shared.

**A tighter one**, worth doing once the above is live: after the annotator
exports, offer "Convert to YOLO / COCO" as a one-click button that POSTs the
zip to `/api/datasets` and then `/api/datasets/{id}/run/convert.dataset`. That
turns the annotator's single export format into five with about thirty lines of
front-end code.

Keep the marketing claim accurate: images stay in the browser **in the
annotator**. The moment someone uploads a zip to the preparation service, that
is a server round trip. Say so plainly on that page — the honesty is worth more
than the claim, and the CLI is there for anyone who wants zero upload.

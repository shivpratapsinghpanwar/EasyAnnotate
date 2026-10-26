"""Record the README films from the real editor, real models, real photos.

    docs/demo.gif / .mp4           draw, SAM, text prompts on every image, review
    docs/segmentation.gif / .mp4   SAM masks: one box or one click per object

Not a mock-up: it boots the service on the CC0 photos in docs/demo_photos/,
drives the page with Playwright, and writes the frames it saw. The AI steps
are the real SAM 2.1 and YOLO-World, so the tools must be installed:

    easyannotate tools install --source .        # once
    uv run --with imageio-ffmpeg python docs/make_demo_gif.py [demo|segmentation|all]

ffmpeg comes from imageio-ffmpeg (or PATH): its two-pass palette is what lets a
GIF of photographs look like photographs at a size a README can carry.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHOTOS = HERE / "demo_photos"
WIDTH, HEIGHT, SCALE = 1280, 760, 1.5     # CSS viewport; screenshots at 1.5x
OUT_WIDTH = 960                            # what a README shows

work = Path(tempfile.mkdtemp(prefix="easyannotate-gif-"))
os.environ["DATAFACTORY_WORKSPACE"] = str(work / "ws")
os.environ.pop("DATAFACTORY_TOOLS_HOME", None)   # the real tools, not a test's empty home

import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from data_factory.annotate import attach  # noqa: E402
from data_factory.annotate.tools.registry import REGISTRY  # noqa: E402
from data_factory.service.app import app  # noqa: E402
from data_factory.service.deps import get_store  # noqa: E402

#: A visible pointer. Screenshots never include the real one, and a demo
#: without a cursor is boxes appearing out of nowhere.
CURSOR = """
(() => {
  const install = () => {
    if (document.getElementById("__demo_cursor")) return;
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", "22"); svg.setAttribute("height", "28"); svg.setAttribute("viewBox", "0 0 22 28");
    const path = document.createElementNS(ns, "path");
    path.setAttribute("d", "M2 2 L2 22 L7.5 17 L11.5 26 L15 24.5 L11 15.8 L18.5 15.8 Z");
    path.setAttribute("fill", "#ffffff"); path.setAttribute("stroke", "#111111");
    path.setAttribute("stroke-width", "1.6"); path.setAttribute("stroke-linejoin", "round");
    svg.appendChild(path);
    const ring = document.createElement("div");
    ring.id = "__demo_ring";
    Object.assign(ring.style, {position: "absolute", left: "-14px", top: "-14px", width: "28px", height: "28px",
      borderRadius: "50%", border: "2px solid #4d9bff", opacity: "0", transform: "scale(.4)",
      transition: "transform .25s ease-out, opacity .25s ease-out"});
    const cursor = document.createElement("div");
    cursor.id = "__demo_cursor";
    Object.assign(cursor.style, {position: "fixed", left: "0", top: "0", zIndex: "2147483647",
      pointerEvents: "none", transform: "translate(640px, 380px)", filter: "drop-shadow(0 1px 1.5px rgba(0,0,0,.5))"});
    cursor.append(ring, svg);
    document.documentElement.appendChild(cursor);
    addEventListener("mousemove", (e) => {
      cursor.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
    }, true);
    addEventListener("mousedown", () => { ring.style.opacity = "1"; ring.style.transform = "scale(1)"; }, true);
    addEventListener("mouseup", () => { ring.style.opacity = "0"; ring.style.transform = "scale(.4)"; }, true);
  };
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", install); else install();
})();
"""


def ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        found = shutil.which("ffmpeg")
        if not found:
            sys.exit("needs ffmpeg: uv run --with imageio-ffmpeg python docs/make_demo_gif.py")
        return found


class Film:
    """Screenshots with how long each is held, then encoded once at the end."""

    def __init__(self, page, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.page, self.folder, self.shots = page, folder, []
        self.x, self.y = WIDTH / 2, HEIGHT / 2

    def shoot(self, hold_ms: int) -> None:
        path = self.folder / f"{len(self.shots):04d}.png"
        self.page.screenshot(path=str(path))
        self.shots.append((path, hold_ms))

    def glide(self, x: float, y: float, frames: int = 7, ms: int = 40) -> None:
        """Move the pointer in visible steps, easing out."""
        x0, y0 = self.x, self.y
        for i in range(1, frames + 1):
            t = 1 - (1 - i / frames) ** 2
            self.page.mouse.move(x0 + (x - x0) * t, y0 + (y - y0) * t)
            self.shoot(ms)
        self.x, self.y = x, y

    def click(self, x: float, y: float, hold_ms: int = 160) -> None:
        self.glide(x, y)
        self.page.mouse.down()
        self.shoot(90)
        self.page.mouse.up()
        self.shoot(hold_ms)

    def drag(self, x1: float, y1: float, x2: float, y2: float, frames: int = 9) -> None:
        self.glide(x1, y1)
        self.page.mouse.down()
        self.shoot(80)
        for i in range(1, frames + 1):
            t = i / frames
            self.page.mouse.move(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
            self.shoot(45)
        self.page.mouse.up()
        self.x, self.y = x2, y2

    def type(self, text: str, per: int = 2) -> None:
        for i, ch in enumerate(text):
            self.page.keyboard.type(ch)
            if (i + 1) % per == 0 or i == len(text) - 1:
                self.shoot(70)

    def center_of(self, selector: str) -> tuple[float, float]:
        box = self.page.locator(selector).first.bounding_box()
        return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

    def shapes(self) -> int:
        return self.page.evaluate("window.__editor.scene.order.length")

    def wait_for_shapes(self, more_than: int) -> None:
        self.page.wait_for_function(f"window.__editor.scene.order.length > {more_than}", timeout=120_000)


def open_editor(page, film: Film, project_id: str) -> None:
    page.goto(f"http://127.0.0.1:8798/annotate?project={project_id}", wait_until="networkidle")
    page.wait_for_function("window.__editor && window.__editor.image")
    page.wait_for_function("!document.getElementById('tools').hidden", timeout=120_000)
    page.wait_for_timeout(800)
    film.page.mouse.move(film.x, film.y)
    film.shoot(1500)


def new_class(page, film: Film, name: str) -> None:
    """No schema step: "/", type the name, Enter -- it exists."""
    page.keyboard.press("/")
    film.shoot(150)
    film.type(name)
    page.keyboard.press("Enter")
    film.shoot(200)


def sam(page, film: Film, *, box=None, point=None, hold_ms: int = 700) -> None:
    """One SAM call through the armed tool, filmed until its mask is drawn."""
    before = film.shapes()
    if box:
        film.drag(*box)
    else:
        film.click(*point, hold_ms=60)
    film.shoot(150)
    film.wait_for_shapes(before)
    page.wait_for_timeout(250)
    film.shoot(hold_ms)


def record_segmentation(page, film: Film, project_id: str) -> None:
    open_editor(page, film, project_id)

    # A car from one rough box.
    new_class(page, film, "car")
    film.click(*film.center_of("#tools button[data-tool='sam.box']"), hold_ms=150)
    sam(page, film, box=(330, 252, 1024, 576), hold_ms=250)
    film.glide(90, 470, frames=6)  # off the photo: the editor's guides follow the pointer
    film.shoot(1400)

    # Next photo: three dogs from one click each, three people from a box each.
    film.click(*film.center_of(".strip-item:nth-child(2)"), hold_ms=100)
    page.wait_for_function("window.__editor.image && window.__editor.image.name.includes('sidewalk-dogs')")
    page.wait_for_timeout(500)
    film.shoot(900)
    new_class(page, film, "dog")
    film.click(*film.center_of("#tools button[data-tool='sam.point']"), hold_ms=150)
    for point in [(268, 520), (976, 508), (878, 500)]:
        sam(page, film, point=point, hold_ms=500)
    new_class(page, film, "person")
    film.click(*film.center_of("#tools button[data-tool='sam.box']"), hold_ms=150)
    for box in [(506, 270, 650, 586), (640, 254, 770, 636), (758, 280, 898, 636)]:
        sam(page, film, box=box, hold_ms=450)
    film.glide(90, 640, frames=5)
    film.shoot(2400)


def record_demo(page, film: Film, project_id: str) -> None:
    open_editor(page, film, project_id)

    # 1. No setup: name a class by typing it, then draw. The dog.
    page.keyboard.press("/")
    film.shoot(200)
    film.type("dog")
    page.keyboard.press("Enter")
    film.shoot(250)
    film.drag(476, 494, 607, 625)
    page.wait_for_timeout(300)
    film.shoot(1100)

    # 2. SAM: drag a rough box around the man, get his outline.
    page.keyboard.press("/")
    film.type("person")
    page.keyboard.press("Enter")
    film.shoot(200)
    film.click(*film.center_of("#tools button[data-tool='sam.box']"), hold_ms=200)
    before = film.shapes()
    film.drag(578, 266, 690, 586)
    film.shoot(200)
    film.wait_for_shapes(before)
    page.wait_for_timeout(250)
    film.shoot(1500)

    # 3. Text prompts on every unreviewed image, as a job.
    film.click(*film.center_of("#tools [data-role=prompt-text]"), hold_ms=120)
    film.type("person, dog, car", per=3)
    film.click(*film.center_of("#tools [data-role=prompt-all]"), hold_ms=150)
    film.glide(90, 640, frames=5)  # off the panel, which re-flows when the job ends
    started = time.time()
    while True:
        message = page.evaluate("document.getElementById('progress').textContent")
        if message.startswith("found ") or message.startswith("search "):
            break
        if time.time() - started > 300:
            raise RuntimeError(f"batch prompt stuck at: {message}")
        film.shoot(260)
        page.wait_for_timeout(700)
    page.wait_for_timeout(300)
    film.shoot(1700)

    # 4. Review the model's work, least sure first.
    film.click(*film.center_of("[data-role=strip-filter]"), hold_ms=80)
    page.select_option("[data-role=strip-filter]", "predicted:uncertain")
    page.wait_for_timeout(500)
    film.shoot(300)
    film.click(*film.center_of(".strip-item"), hold_ms=100)
    page.wait_for_function("!document.getElementById('review').hidden", timeout=30_000)
    page.wait_for_timeout(500)
    film.shoot(1500)

    # The slider hides weak guesses: drag it to about 40 %.
    track = page.locator("#conf").bounding_box()
    y = track["y"] + track["height"] / 2
    film.drag(track["x"] + 4, y, track["x"] + 4 + (track["width"] - 8) * 0.4, y, frames=8)
    page.wait_for_timeout(250)
    film.shoot(1500)

    # Accept (the A key, shown as its button) saves the image and opens the
    # next one. Twice, quickly; then rest the pointer off the photo.
    for _ in range(2):
        current = page.evaluate("window.__editor.image.id")
        film.click(*film.center_of("#btn-accept"), hold_ms=60)
        page.wait_for_function(f"window.__editor.image && window.__editor.image.id !== '{current}'", timeout=30_000)
        page.wait_for_timeout(450)
        film.shoot(1300)
    film.glide(90, 640, frames=5)
    film.shoot(1600)


def encode(film: Film, gif: Path, mp4: Path) -> None:
    exe = ffmpeg()
    listing = film.folder / "frames.txt"
    lines = []
    for path, hold in film.shots:
        lines += [f"file '{path.name}'", f"duration {hold / 1000:.3f}"]
    lines.append(f"file '{film.shots[-1][0].name}'")  # the concat demuxer drops the last duration otherwise
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    common = [exe, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing)]
    # Palette from the frames that changed, Bayer dither (stable between
    # frames, so unchanged regions compress), and only the changed rectangle
    # redrawn per frame.
    subprocess.run(common + [
        "-vf", f"scale={OUT_WIDTH}:-1:flags=lanczos,split[a][b];"
               "[a]palettegen=max_colors=256:stats_mode=diff[p];"
               "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
        "-loop", "0", str(gif),
    ], check=True, cwd=film.folder)
    subprocess.run(common + [
        "-vf", "scale=1280:-2:flags=lanczos,format=yuv420p", "-fps_mode", "vfr",
        "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-movflags", "+faststart", str(mp4),
    ], check=True, cwd=film.folder)


#: name -> (photos in strip order, what happens on film)
FILMS = {
    "demo": (["walking-the-dog", "crosswalk", "sidewalk-dogs", "dog-walkers",
              "cafe-corner", "parked-cars", "play-street", "police-car"], record_demo),
    "segmentation": (["police-car", "sidewalk-dogs", "dog-walkers", "cafe-corner",
                      "walking-the-dog", "play-street"], record_segmentation),
}


def project_for(name: str, photos: list[str]):
    folder = work / name
    folder.mkdir(parents=True)
    for i, photo in enumerate(photos):
        shutil.copy(PHOTOS / f"{photo}.jpg", folder / f"{i + 1:02d}-{photo}.jpg")
    return attach.open_folder(get_store(), folder, name="street")["project"]


def main() -> None:
    wanted = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(FILMS) if wanted == "all" else [wanted]
    if any(name not in FILMS for name in names):
        sys.exit(f"which film? {' | '.join(FILMS)} | all")
    projects = {name: project_for(name, FILMS[name][0]) for name in names}

    if {"sam.box", "detect.text"} - {spec.name for spec in REGISTRY.available()}:
        sys.exit("the AI tools are not installed: easyannotate tools install")
    # Load both models before the first frame: the films are about using them.
    warm = str(PHOTOS / "parked-cars.jpg")
    REGISTRY.invoke_waiting("sam.box", {"image_id": "warm", "image_path": warm, "box": [400, 300, 1200, 900],
                                        "width": 1600, "height": 1200}, budget_s=900)
    REGISTRY.invoke_waiting("detect.text", {"image_id": "warm", "image_path": warm, "text": "car",
                                            "width": 1600, "height": 1200}, budget_s=900)

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8798, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)

    films = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for name in names:
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=SCALE)
            page.add_init_script(CURSOR)
            films[name] = Film(page, work / "frames" / name)
            FILMS[name][1](page, films[name], projects[name].id)
            page.close()
        browser.close()
    server.should_exit = True
    REGISTRY.close()

    for name, film in films.items():
        gif, mp4 = HERE / f"{name}.gif", HERE / f"{name}.mp4"
        encode(film, gif, mp4)
        seconds = sum(hold for _, hold in film.shots) / 1000
        print(f"wrote {gif.name} ({gif.stat().st_size / 1e6:.1f} MB) and {mp4.name} "
              f"({mp4.stat().st_size / 1e6:.1f} MB): {len(film.shots)} frames, {seconds:.1f} s")


if __name__ == "__main__":
    main()

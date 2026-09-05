// The editor: wiring, keyboard, and autosave.
//
// Everything with an opinion lives in the modules this imports. What is here
// is the loop, the shortcuts, and the rules about when work reaches the
// server.
//
// Two of those rules are the product:
//
//   * There is no Save button. Edits are saved 400ms after you stop, and
//     before you navigate. A Save button is a way to lose an afternoon.
//   * Opening a project resumes where you left off. Not at image 1.

import { Api, ApiError, Conflict } from "./api.js";
import { Filmstrip } from "./filmstrip.js";
import { LabelRail } from "./labels.js";
import { draw } from "./render.js";
import { Scene, cloneShape, draftId, isDraft } from "./scene.js";
import { AssistPanel } from "./assist.js";
import { UndoStack, addMany, addShape, changeLabel, removeShape } from "./undo.js";
import { Viewport, buildMips } from "./viewport.js";
// tools/ are the *drawing* tools. assist.js is the panel for optional
// model-backed helpers -- two different things that were both called
// "tools" until the import block made the collision obvious.
import * as boxTool from "./tools/box.js";
import * as polygonTool from "./tools/polygon.js";
import * as selectTool from "./tools/select.js";

const SAVE_DELAY = 400;

class Editor {
  constructor() {
    this.api = new Api();
    this.scene = new Scene();
    this.view = new Viewport();
    this.undo = new UndoStack();

    this.canvas = document.getElementById("canvas");
    this.ctx = this.canvas.getContext("2d", { alpha: false });
    this.status = document.getElementById("status");
    this.imageName = document.getElementById("image-name");
    this.progress = document.getElementById("progress");
    this.help = document.getElementById("help");

    this.projectId = new URLSearchParams(location.search).get("project") || "";
    this.project = null;
    this.images = [];
    this.image = null;
    this.mips = null;
    this.saveTimer = null;
    this.saving = null;
    this.pendingGeometry = null;
    this.spacePanning = false;
    // Client-side only: filtering predictions is a slider, and a round trip
    // per drag would make it feel broken.
    this.minConfidence = 0;

    this.tools = {
      select: selectTool.create(this),
      box: boxTool.create(this),
      polygon: polygonTool.create(this),
    };
    this.tool = this.tools.select;

    this.rail = new LabelRail(document.getElementById("labels"), {
      onPick: () => this.applyPendingGeometry(),
      onCreate: (name) => this.createLabel(name),
      onRename: (idx, name) => this.renameLabel(idx, name),
      onRetire: (idx, name) => this.retireLabel(idx, name),
    });

    this.assistPanel = new AssistPanel(document.getElementById("tools"), {
      onArm: (tool) => {
        this.armedTool = tool;
        this.say(tool ? `${tool.title}: drag a box around the object` : "");
      },
    });
    this.armedTool = null;

    this.strip = new Filmstrip(document.getElementById("filmstrip"), {
      onOpen: (id) => this.openImage(id),
      thumbUrl: (id) => this.api.thumbUrl(this.projectId, id),
    });

    this.bindCanvas();
    this.bindKeyboard();
    this.bindChrome();
  }

  // -- lifecycle ----------------------------------------------------------

  async boot() {
    if (!this.projectId) {
      this.say("no project chosen -- open one from the main page");
      return;
    }
    try {
      this.project = await this.api.project(this.projectId);
    } catch (error) {
      this.say(describe(error));
      return;
    }
    document.title = `${this.project.name} — Data Factory`;
    document.getElementById("project-name").textContent = this.project.name;

    this.rail.setLabels(this.project.labels || []);
    await this.loadImageList();

    this.updateReviewPanel(this.project.agreement);

    // Heavy tools are optional. An empty list hides the panel entirely --
    // never a greyed-out button, never an install nag.
    try {
      this.assistPanel.setTools(await this.api.tools());
    } catch {
      this.assistPanel.setTools({ tools: [] });
    }

    const resume = this.project.resume;
    if (resume) await this.openImage(resume.id);
    else this.say("this project has no images yet");
  }

  async loadImageList() {
    const payload = await this.api.images(this.projectId, { limit: 2000 });
    this.images = payload.images;
    this.strip.setImages(this.images);
    this.strip.setCounts(payload.counts);
  }

  async openImage(imageId) {
    // Never navigate away from unsaved work. This is the single most likely
    // place to lose an edit, so it is awaited rather than fired and forgotten.
    await this.flushSave();

    let payload;
    try {
      payload = await this.api.image(this.projectId, imageId);
    } catch (error) {
      this.say(describe(error));
      return;
    }

    this.image = payload;
    this.scene.load(payload);
    this.undo.clear();
    this.view.setImageSize(payload.width, payload.height);
    this.strip.setCurrent(imageId);
    this.imageName.textContent = payload.name;
    this.setStatusDot("saved");
    this.cancelTool();

    await this.loadPixels(imageId);
    this.view.fit(this.canvas.getBoundingClientRect());
    this.updateReviewPanel(null);
    this.scene.invalidate();
    this.say(`${payload.name} — ${payload.shapes.length} shape(s)`);
  }

  loadPixels(imageId) {
    return new Promise((resolve) => {
      const img = new Image();
      img.decoding = "async";
      img.onload = () => {
        // Mips are built once per image, at load. See viewport.js for why.
        this.mips = buildMips(img);
        // The database is the authority on size, but if a row predates the
        // EXIF work its dimensions could disagree with the pixels. Trust what
        // decoded, or every box on this image is drawn in the wrong space.
        if (img.naturalWidth && img.naturalHeight) {
          this.view.setImageSize(img.naturalWidth, img.naturalHeight);
          this.scene.width = img.naturalWidth;
          this.scene.height = img.naturalHeight;
        }
        this.scene.invalidate();
        resolve();
      };
      img.onerror = () => {
        this.mips = null;
        this.say("that image could not be loaded — its shapes are still editable");
        resolve();
      };
      img.src = this.api.imageUrl(this.projectId, imageId);
    });
  }

  // -- saving -------------------------------------------------------------

  queueSave() {
    clearTimeout(this.saveTimer);
    this.setStatusDot("pending");
    this.saveTimer = setTimeout(() => this.save(), SAVE_DELAY);
  }

  async flushSave() {
    clearTimeout(this.saveTimer);
    if (this.saving) await this.saving.catch(() => {});
    if (this.scene.changed) await this.save();
  }

  async save() {
    if (!this.image) return;
    if (!this.scene.changed) {
      // Nothing to send. Say so, rather than leaving "unsaved" on screen --
      // an indicator that is wrong even once is one nobody trusts again, and
      // this is the only thing telling people their work is safe.
      this.setStatusDot("saved");
      return;
    }
    const imageId = this.scene.imageId;
    const submitted = this.scene.toWire();
    const localIds = this.scene.order.slice();

    this.setStatusDot("saving");
    this.saving = this.api
      .saveShapes(this.projectId, imageId, this.scene.version, submitted)
      .then((result) => {
        // Only reconcile if we are still on the image we saved. Navigating
        // mid-flight is normal, and applying the result to a different image
        // would corrupt it.
        if (this.scene.imageId !== imageId) return;
        this.scene.version = result.version;
        this.scene.changed = false;
        (result.ids || []).forEach((serverId, index) => {
          const local = localIds[index];
          if (local && isDraft(local)) this.scene.reid(local, serverId);
        });
        if (result.labels) this.rail.setLabels(result.labels);
        if (result.state) this.strip.setState(imageId, result.state);
        this.setStatusDot("saved");
        this.scene.invalidate();
      })
      .catch(async (error) => {
        if (error instanceof Conflict) {
          // Somebody else moved this image. Take their version rather than
          // overwrite it, and say so plainly.
          this.scene.load(error.current);
          this.undo.clear();
          this.setStatusDot("conflict");
          this.say("this image was changed elsewhere — reloaded the newer version");
          return;
        }
        this.setStatusDot("error");
        this.say(describe(error));
      })
      .finally(() => {
        this.saving = null;
      });

    await this.saving;
  }

  setStatusDot(state) {
    this.status.dataset.state = state;
    this.status.textContent = {
      pending: "unsaved",
      saving: "saving…",
      saved: "saved",
      conflict: "reloaded",
      error: "not saved",
    }[state] || "";
  }

  // -- labels -------------------------------------------------------------

  currentLabel() {
    return this.rail.current;
  }

  async createLabel(name) {
    try {
      await this.api.createLabel(this.projectId, name);
      const payload = await this.api.labels(this.projectId);
      this.rail.setLabels(payload.labels);
      const made = payload.labels.find((l) => l.name === name.trim());
      if (made) this.rail.pick(made);
      // Hand the keyboard back to the canvas: naming a class is part of
      // drawing, and the next keystroke belongs to the drawing.
      this.rail.releaseInput();
    } catch (error) {
      this.say(describe(error));
    }
  }

  async renameLabel(idx, name) {
    // Renaming a class onto one that already exists is how people try to merge
    // duplicates -- and the real door set has four of them (Close/close/
    // Open/open). Refusing outright leaves them stuck, so offer the merge they
    // were reaching for. It is destructive, so it is asked, not assumed.
    const clash = this.rail.labels.find(
      (label) => label.name.toLowerCase() === name.trim().toLowerCase() && label.idx !== idx
    );
    if (clash) {
      const from = this.rail.labels.find((label) => label.idx === idx);
      const ok = confirm(
        `"${clash.name}" already exists.\n\n` +
          `Merge "${from ? from.name : idx}" into it? Every shape using it moves ` +
          `to "${clash.name}", and class index ${idx} is retired.\n\n` +
          `Class indices already exported do not change.`
      );
      if (!ok) return;
      try {
        const result = await this.api.mergeLabel(this.projectId, idx, clash.idx);
        this.rail.setLabels(result.labels);
        await this.openImage(this.image.id);
        this.say(`merged — ${result.moved} shape(s) moved to "${clash.name}"`);
      } catch (error) {
        this.say(describe(error));
      }
      return;
    }

    try {
      await this.api.renameLabel(this.projectId, idx, name);
      const payload = await this.api.labels(this.projectId);
      this.rail.setLabels(payload.labels);
      // Shapes carry the name for display, so refresh them from the index.
      for (const shape of this.scene.shapes.values()) {
        const label = payload.labels.find((l) => l.idx === shape.labelIdx);
        if (label) shape.label = label.name;
      }
      this.scene.invalidate();
      this.say(`renamed to "${name}" — the class index did not move`);
    } catch (error) {
      this.say(describe(error));
    }
  }

  async retireLabel(idx, name) {
    if (!confirm(`Hide "${name}" from the palette? Shapes already drawn with it keep it.`)) {
      return;
    }
    try {
      await this.api.retireLabel(this.projectId, idx);
      const payload = await this.api.labels(this.projectId);
      this.rail.setLabels(payload.labels);
      this.say(`"${name}" retired — its class index stays reserved`);
    } catch (error) {
      this.say(describe(error));
    }
  }

  // A shape was drawn before any class existed. Hold the geometry, ask for a
  // name, and apply it the moment one arrives -- rather than discarding the
  // drag and telling the user to go and configure something.
  promptForLabel(geometry) {
    this.pendingGeometry = geometry;
    this.say("name this class, then press Enter");
    this.rail.focusInput();
  }

  applyPendingGeometry() {
    const geometry = this.pendingGeometry;
    const label = this.currentLabel();
    if (!geometry || !label) return;
    this.pendingGeometry = null;
    const shape = this.scene.clampShape({
      id: draftId(),
      label: label.name,
      labelIdx: label.idx,
      bbox: geometry.bbox,
      polygon: geometry.polygon,
      origin: "human",
      confidence: null,
      edited: false,
    });
    this.undo.run(addShape(this.scene, shape));
    this.queueSave();
  }

  assignClassToSelection(label) {
    const selected = this.scene.selected();
    if (!selected.length || !label) return false;
    let changed = 0;
    for (const shape of selected) {
      if (shape.labelIdx === label.idx) continue;
      this.undo.run(
        changeLabel(
          this.scene,
          shape.id,
          { label: shape.label, labelIdx: shape.labelIdx },
          { label: label.name, labelIdx: label.idx }
        )
      );
      this.markEdited(shape);
      changed += 1;
    }
    // Re-picking the class a shape already has is a no-op, not an edit. Saving
    // anyway would mark the image unsaved for something that changed nothing.
    if (changed) this.queueSave();
    return true;
  }

  // A person touching a model shape makes it theirs: it stops rendering
  // dashed, and starts counting as a correction in the agreement figure.
  markEdited(shape) {
    if (shape && shape.origin === "model" && !shape.edited) {
      shape.edited = true;
      this.scene.invalidate();
    }
  }

  onSelectionChanged() {
    const selected = this.scene.selected();
    if (selected.length === 1) {
      const label = this.rail.labels.find((l) => l.idx === selected[0].labelIdx);
      if (label) this.rail.pick(label);
    }
  }

  // -- canvas -------------------------------------------------------------

  bindCanvas() {
    const resize = () => {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.canvas.width = Math.round(rect.width * dpr);
      this.canvas.height = Math.round(rect.height * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.scene.invalidate();
    };
    new ResizeObserver(resize).observe(this.canvas);
    resize();

    this.canvas.addEventListener("pointerdown", (event) => {
      this.canvas.setPointerCapture(event.pointerId);
      const [ix, iy] = this.view.toImage(event.offsetX, event.offsetY);

      // Middle button and held space always pan, whatever tool is active.
      if (event.button === 1 || this.spacePanning) {
        this.panning = { x: event.offsetX, y: event.offsetY };
        return;
      }
      // Left-drag on empty canvas with the select tool draws a box: no mode
      // to enter, which is the entire interaction argument of this product.
      if (
        event.button === 0 &&
        this.tool === this.tools.select &&
        !this.hitSomething(ix, iy)
      ) {
        this.tools.box.pointerdown(event, ix, iy);
        this.drawingBox = true;
        return;
      }
      this.tool.pointerdown(event, ix, iy);
    });

    this.canvas.addEventListener("pointermove", (event) => {
      const [ix, iy] = this.view.toImage(event.offsetX, event.offsetY);
      if (this.panning) {
        this.view.panBy(event.offsetX - this.panning.x, event.offsetY - this.panning.y);
        this.panning = { x: event.offsetX, y: event.offsetY };
        this.scene.invalidate();
        return;
      }
      if (this.drawingBox) {
        this.tools.box.pointermove(event, ix, iy);
        return;
      }
      this.tool.pointermove(event, ix, iy);
    });

    const release = (event) => {
      if (this.panning) {
        this.panning = null;
        return;
      }
      if (this.drawingBox) {
        if (this.armedTool) {
          const draft = this.tools.box.draft();
          this.tools.box.cancel();
          this.drawingBox = false;
          if (draft) this.runTool(this.armedTool, draft.bbox);
          return;
        }
        this.tools.box.pointerup(event);
        this.drawingBox = false;
        return;
      }
      this.tool.pointerup(event);
    };
    this.canvas.addEventListener("pointerup", release);
    this.canvas.addEventListener("pointercancel", release);

    this.canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        this.view.zoomAt(event.offsetX, event.offsetY, Math.exp(-event.deltaY * 0.0015));
        this.scene.invalidate();
      },
      { passive: false }
    );

    // Right-click deletes what is under the cursor. Faster than select-then-
    // Delete, and the undo stack makes it safe.
    this.canvas.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      const [ix, iy] = this.view.toImage(event.offsetX, event.offsetY);
      const hit = this.hitSomething(ix, iy);
      if (hit) {
        this.undo.run(removeShape(this.scene, hit));
        this.queueSave();
      }
    });

    requestAnimationFrame(() => this.frame());
  }

  hitSomething(ix, iy) {
    for (let i = this.scene.order.length - 1; i >= 0; i--) {
      const shape = this.scene.get(this.scene.order[i]);
      if (!shape || !this.visible(shape)) continue;
      const [x1, y1, x2, y2] = shape.bbox;
      if (ix >= x1 && ix <= x2 && iy >= y1 && iy <= y2) return shape.id;
    }
    return null;
  }

  frame() {
    if (this.scene.dirty) {
      this.scene.dirty = false;
      const rect = this.canvas.getBoundingClientRect();
      const draft = this.drawingBox ? this.tools.box.draft() : this.tool.draft();
      draw(this.ctx, rect, this.scene, this.view, this.mips, {
        colourFor: (shape) => this.rail.colourFor(shape),
        visible: (shape) => this.visible(shape),
        draft,
      });
    }
    requestAnimationFrame(() => this.frame());
  }

  // -- keyboard -----------------------------------------------------------

  bindKeyboard() {
    window.addEventListener("keydown", (event) => {
      if (isTypingTarget(event.target)) return;

      if (event.key === " " && !event.repeat) {
        this.spacePanning = true;
        this.canvas.style.cursor = "grab";
        event.preventDefault();
        return;
      }

      if (this.tool.key && this.tool.key(event)) {
        event.preventDefault();
        return;
      }

      const ctrl = event.ctrlKey || event.metaKey;

      if (ctrl && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) this.undo.redo();
        else this.undo.undo();
        this.scene.invalidate();
        this.queueSave();
        return;
      }
      if (ctrl && event.key.toLowerCase() === "y") {
        event.preventDefault();
        this.undo.redo();
        this.scene.invalidate();
        this.queueSave();
        return;
      }
      if (ctrl && event.key.toLowerCase() === "s") {
        // There is no Save button, but muscle memory deserves an answer.
        event.preventDefault();
        this.flushSave();
        return;
      }
      if (ctrl && event.key.toLowerCase() === "d") {
        event.preventDefault();
        this.copyPrevious();
        return;
      }

      if (/^[0-9]$/.test(event.key)) {
        event.preventDefault();
        const digit = Number(event.key);
        const before = this.rail.current;
        if (this.rail.pickByNumber(digit)) {
          // With a selection, a number key re-classes it. Without one, it
          // chooses what the next box will be.
          if (!this.assignClassToSelection(this.rail.current) && before !== this.rail.current) {
            this.say(`drawing "${this.rail.current.name}"`);
          }
        }
        return;
      }

      switch (event.key.toLowerCase()) {
        case "escape":
          this.assistPanel.disarm();
          this.cancelTool();
          this.scene.clearSelection();
          break;
        case "delete":
        case "backspace": {
          const selected = [...this.scene.selection];
          if (!selected.length) break;
          event.preventDefault();
          for (const id of selected) this.undo.run(removeShape(this.scene, id));
          this.queueSave();
          break;
        }
        case "p":
          this.setTool("polygon");
          break;
        case "v":
          this.setTool("select");
          break;
        case "b":
          this.setTool("box");
          break;
        case "f":
          this.view.fit(this.canvas.getBoundingClientRect());
          this.scene.invalidate();
          break;
        case "0":
          this.view.actual(this.canvas.getBoundingClientRect());
          this.scene.invalidate();
          break;
        case "n":
          this.step(1);
          break;
        case "e":
          this.step(-1);
          break;
        case "r":
          this.markReviewedAndAdvance();
          break;
        case "a":
          // Not Space, which is already held-to-pan. Dual-binding the biggest
          // key on the keyboard would mean panning on a predicted image
          // silently accepted every box on it.
          this.acceptAll();
          break;
        case "s":
          this.markState("skipped");
          break;
        default:
          break;
      }

      if (event.key === "?" || (event.key === "/" && event.shiftKey)) {
        this.help.hidden = !this.help.hidden;
      }
    });

    window.addEventListener("keyup", (event) => {
      if (event.key === " ") {
        this.spacePanning = false;
        this.panning = null;
        this.canvas.style.cursor = "default";
      }
    });

    // A close with unsaved work is the one failure people never forgive.
    window.addEventListener("beforeunload", (event) => {
      if (this.scene.changed) {
        event.preventDefault();
        event.returnValue = "";
      }
    });
  }

  bindChrome() {
    document.getElementById("btn-help").addEventListener("click", () => {
      this.help.hidden = !this.help.hidden;
    });
    this.help.addEventListener("click", () => {
      this.help.hidden = true;
    });
    document.getElementById("btn-fit").addEventListener("click", () => {
      this.view.fit(this.canvas.getBoundingClientRect());
      this.scene.invalidate();
    });
    document.getElementById("btn-prev").addEventListener("click", () => this.step(-1));
    document.getElementById("btn-next").addEventListener("click", () => this.step(1));
    document.getElementById("btn-review").addEventListener("click", () =>
      this.markReviewedAndAdvance()
    );
    for (const button of document.querySelectorAll("[data-tool]")) {
      button.addEventListener("click", () => this.setTool(button.dataset.tool));
    }

    const slider = document.getElementById("conf");
    slider.addEventListener("input", () => {
      this.minConfidence = Number(slider.value) / 100;
      document.getElementById("conf-value").textContent = `${slider.value}%`;
      this.scene.invalidate();
    });

    document.getElementById("btn-accept").addEventListener("click", () => this.acceptAll());

    const picker = document.getElementById("predictions-file");
    document.getElementById("btn-predictions").addEventListener("click", () => picker.click());
    picker.addEventListener("change", () => {
      const file = picker.files && picker.files[0];
      picker.value = "";
      if (file) this.loadPredictions(file);
    });
  }

  // -- review -------------------------------------------------------------

  // A shape hidden by the confidence slider is not deleted, only unseen: the
  // filter is a way of looking, not an edit. Accepting an image is what turns
  // what you can see into what you keep.
  visible(shape) {
    if (shape.confidence == null) return true;
    return shape.confidence >= this.minConfidence;
  }

  hasPredictions() {
    for (const shape of this.scene.shapes.values()) {
      if (shape.origin === "model" && !shape.edited) return true;
    }
    return false;
  }

  updateReviewPanel(agreement) {
    const panel = document.getElementById("review");
    panel.hidden = !this.hasPredictions() && !(agreement && agreement.model_shapes);
    if (!agreement) return;
    const label = document.getElementById("agreement");
    if (agreement.agreement == null) {
      label.textContent = agreement.unreviewed
        ? `${agreement.unreviewed} prediction(s) not yet reviewed`
        : "";
      return;
    }
    label.textContent =
      `model agreement ${Math.round(agreement.agreement * 100)}% ` +
      `(${agreement.corrected} of ${agreement.model_shapes} corrected)`;
  }

  async acceptAll() {
    if (!this.image) return;
    // Anything the slider is hiding is a rejection, so it has to be removed
    // before the accept -- otherwise "accept what I can see" silently keeps
    // what you could not.
    const hidden = [...this.scene.shapes.values()].filter((s) => !this.visible(s));
    if (hidden.length) {
      for (const shape of hidden) this.undo.run(removeShape(this.scene, shape.id));
      await this.flushSave();
    }
    try {
      const result = await this.api.accept(this.projectId, this.image.id);
      this.strip.setState(this.image.id, "reviewed");
      this.strip.setCounts(result.counts);
      this.updateReviewPanel(result.agreement);
      this.say(
        `accepted ${result.confirmed} prediction(s)` +
          (hidden.length ? `, rejected ${hidden.length}` : "")
      );
      if (result.next) await this.openImage(result.next);
      else this.say("that was the last image");
    } catch (error) {
      this.say(describe(error));
    }
  }

  // -- heavy tools ---------------------------------------------------------

  async runTool(tool, box) {
    if (!this.image) return;
    const label = this.currentLabel();
    this.say(`${tool.title}…`);

    let result;
    try {
      result = await this.api.runTool(this.projectId, tool.name, {
        imageId: this.image.id,
        params: { box },
        label: label ? label.name : "",
      });
    } catch (error) {
      // A tool that is warming up is not a failure. Say when to try again
      // rather than making the user guess, and never lose the box they drew.
      if (error.status === 503 && error.payload && error.payload.detail) {
        const detail = error.payload.detail;
        const wait = detail.retry_after_ms || 500;
        this.say(`${tool.title} is warming up — try again in ${Math.ceil(wait / 100) / 10}s`);
        return;
      }
      this.say(`${describe(error)} — drawing by hand still works`);
      return;
    }

    const made = result.shapes || [];
    if (!made.length) {
      this.say(result.detail || "the tool found nothing there");
      return;
    }

    // Added as one undo step, so a bad suggestion costs one Ctrl+Z.
    const shapes = made.map((raw) =>
      this.scene.clampShape({
        id: draftId(),
        label: raw.label,
        labelIdx: label ? label.idx : -1,
        bbox: raw.bbox.slice(),
        polygon: raw.polygon ? raw.polygon.map((p) => p.slice()) : null,
        origin: "model",
        confidence: raw.confidence ?? null,
        edited: false,
      })
    );
    for (const shape of shapes) this.scene.add(shape);
    this.undo.push(addMany(this.scene, shapes));
    this.queueSave();

    const points = shapes[0].polygon ? shapes[0].polygon.length : 0;
    this.say(
      `${tool.title}: ${shapes.length} shape(s)` +
        (points ? `, ${points} points` : "") +
        " — drag a vertex to correct it"
    );
  }

  async loadPredictions(file) {
    this.say(`reading ${file.name}…`);
    try {
      const report = await this.api.importPredictions(this.projectId, file, {
        minConfidence: 0,
      });
      this.rail.setLabels(report.labels || this.rail.labels);
      this.strip.setCounts(report.counts);
      await this.loadImageList();
      if (this.image) await this.openImage(this.image.id);
      this.updateReviewPanel(report.agreement);

      // The matching report is the point. Importing zero shapes because the
      // filenames did not line up looks exactly like a model that predicted
      // nothing, so it is stated rather than left to be inferred.
      const parts = [
        `${report.matched} image(s) matched, ${report.shapes_imported} prediction(s) loaded`,
      ];
      if (report.unmatched_files) parts.push(`${report.unmatched_files} file(s) matched nothing`);
      if (report.skipped_reviewed) parts.push(`${report.skipped_reviewed} already reviewed, skipped`);
      this.say(parts.join(" · "));
      if (report.matched === 0) {
        alert(
          "Nothing matched.\n\n" +
            "Predictions are matched to images by filename stem, so check the " +
            "names line up.\n\n" +
            (report.notes || []).join("\n")
        );
      }
    } catch (error) {
      this.say(describe(error));
    }
  }

  setTool(name) {
    const tool = this.tools[name];
    if (!tool) return;
    this.cancelTool();
    this.tool = tool;
    for (const button of document.querySelectorAll("[data-tool]")) {
      if (button.dataset.tool === name) button.dataset.current = "true";
      else delete button.dataset.current;
    }
    this.canvas.style.cursor = name === "select" ? "default" : "crosshair";
    this.say(`${name} tool`);
  }

  cancelTool() {
    for (const tool of Object.values(this.tools)) tool.cancel();
    this.drawingBox = false;
    this.scene.invalidate();
  }

  // -- navigation ---------------------------------------------------------

  async step(direction) {
    if (!this.image) return;
    const target = direction > 0 ? this.image.next : this.image.previous;
    if (!target) {
      this.say(direction > 0 ? "that was the last image" : "that was the first image");
      return;
    }
    await this.openImage(target);
  }

  async markReviewedAndAdvance() {
    if (!this.image) return;
    await this.markState("reviewed");
    await this.step(1);
  }

  async markState(state) {
    if (!this.image) return;
    await this.flushSave();
    try {
      await this.api.setState(this.projectId, this.image.id, state);
      this.strip.setState(this.image.id, state);
      const payload = await this.api.images(this.projectId, { limit: 1 });
      this.strip.setCounts(payload.counts);
      this.say(`marked ${state}`);
    } catch (error) {
      this.say(describe(error));
    }
  }

  // Copy the previous image's shapes onto this one. For video frames this is
  // worth more than any model: consecutive frames differ by a few pixels.
  async copyPrevious() {
    if (!this.image || !this.image.previous) {
      this.say("no previous image to copy from");
      return;
    }
    try {
      const previous = await this.api.image(this.projectId, this.image.previous);
      if (!previous.shapes.length) {
        this.say("the previous image has no shapes");
        return;
      }
      const copies = previous.shapes.map((raw) =>
        this.scene.clampShape({
          id: draftId(),
          label: raw.label,
          labelIdx: raw.label_idx,
          bbox: raw.bbox.slice(),
          polygon: raw.polygon ? raw.polygon.map((p) => p.slice()) : null,
          origin: "human",
          confidence: null,
          edited: false,
        })
      );
      for (const shape of copies) this.scene.add(cloneShape(shape));
      this.undo.push(addMany(this.scene, copies));
      this.queueSave();
      this.say(`copied ${copies.length} shape(s) from ${previous.name}`);
    } catch (error) {
      this.say(describe(error));
    }
  }

  say(message) {
    this.progress.textContent = message;
  }
}

function isTypingTarget(node) {
  if (!node) return false;
  const tag = node.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || node.isContentEditable;
}

function describe(error) {
  if (error instanceof ApiError && error.status === 0) {
    return "cannot reach the server — your work is still here, retrying will save it";
  }
  return error && error.message ? error.message : String(error);
}

const editor = new Editor();
editor.boot();
window.__editor = editor;

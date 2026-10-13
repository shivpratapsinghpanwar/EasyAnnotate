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
import { hitTest } from "./hit.js";
import { draw } from "./render.js";
import { Scene, attributesOf, centroid, cloneShape, draftId, rebuildBox, rotatePoints } from "./scene.js";
import { ShapeList } from "./shapelist.js";
import { AssistPanel } from "./assist.js";
import {
  UndoStack,
  addMany,
  addShape,
  changeAttributes,
  changeGeometry,
  changeLabel,
  changeMany,
  removeShape,
  snapshot,
} from "./undo.js";
import { Viewport, decodePixels, releaseMips } from "./viewport.js";
// tools/ are the *drawing* tools. assist.js is the panel for optional
// model-backed helpers -- two different things that were both called
// "tools" until the import block made the collision obvious.
import * as boxTool from "./tools/box.js";
import * as lineTool from "./tools/line.js";
import * as obbTool from "./tools/obb.js";
import * as pointTool from "./tools/point.js";
import * as polygonTool from "./tools/polygon.js";
import * as selectTool from "./tools/select.js";

const SAVE_DELAY = 400;
const PIXEL_CACHE = 3;

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
    this.image = null;
    this.mips = null;
    this.pixelCache = new Map();
    this.opening = 0;
    this.saveTimer = null;
    this.saving = null;
    this.pendingGeometry = null;
    // True while the selection is only "the shape just drawn". It keeps its
    // handles, so it can be resized at once, but it does not capture a class
    // pick -- choosing the class for the *next* box must not re-class the
    // last one -- and a drag across it draws rather than moves.
    this.selectionFromDrawing = false;
    this.pressed = null;
    this.spacePanning = false;
    // Client-side only: filtering predictions is a slider, and a round trip
    // per drag would make it feel broken.
    this.minConfidence = 0;

    this.tools = {
      select: selectTool.create(this),
      box: boxTool.create(this),
      polygon: polygonTool.create(this),
      line: lineTool.create(this),
      point: pointTool.create(this),
      obb: obbTool.create(this),
    };
    this.tool = this.tools.select;

    // Hiding is a way of looking, kept per project in this browser only: it
    // changes nothing anyone else sees and nothing that is exported.
    this.hiddenClasses = new Set(readPreference(`hidden:${this.projectId}`, []));
    this.hideAll = false;
    this.look = readPreference(`look:${this.projectId}`, { brightness: 100, contrast: 100 });

    this.rail = new LabelRail(document.getElementById("labels"), {
      // Picking a class draws a held shape if there is one, and otherwise
      // re-classes the selection -- the same as its number key.
      onPick: (label) => {
        if (this.pendingGeometry) this.applyPendingGeometry();
        else this.assignClassToSelection(label);
      },
      onToggleHidden: (idx) => this.toggleClassHidden(idx),
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

    this.shapeList = new ShapeList(document.getElementById("shapes"), {
      onSelect: (id, additive) => {
        const shape = this.scene.get(id);
        if (shape && !this.visible(shape)) {
          this.say("that shape is hidden — show its class to edit it");
          return;
        }
        this.scene.select(id, { additive });
        this.onSelectionChanged();
      },
      onHover: (id) => {
        this.scene.hover = id;
        this.scene.invalidate();
      },
      colourFor: (shape) => this.rail.colourFor(shape),
    });

    this.strip = new Filmstrip(document.getElementById("filmstrip"), {
      onOpen: (id) => this.openImage(id),
      thumbUrl: (id) => this.api.thumbUrl(this.projectId, id),
      fetchPage: (query) => this.api.images(this.projectId, query),
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
    document.title = `${this.project.name} — EasyAnnotate`;
    document.getElementById("project-name").textContent = this.project.name;

    this.rail.setLabels(this.project.labels || []);
    this.rail.setHidden(this.hiddenClasses);
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

  // Only the counts: the strip fetches its own rows, a page at a time.
  async loadImageList() {
    const payload = await this.api.images(this.projectId, { limit: 1 });
    this.strip.reset(payload.counts);
  }

  async openImage(imageId) {
    // Holding N fires opens faster than they finish. Each takes a ticket and
    // only the newest may touch the screen -- otherwise whichever request
    // lands last wins, and one image's pixels sit under another's shapes.
    const ticket = ++this.opening;

    // Never navigate away from unsaved work. This is the single most likely
    // place to lose an edit, so it is awaited rather than fired and forgotten.
    await this.flushSave();
    if (ticket !== this.opening) return;

    let payload;
    let pixels;
    try {
      // Shapes and pixels together, and the pixels are usually decoded
      // already: the previous open prefetched its neighbours.
      [payload, pixels] = await Promise.all([
        this.api.image(this.projectId, imageId),
        this.pixels(imageId),
      ]);
    } catch (error) {
      if (ticket === this.opening) this.say(describe(error));
      return;
    }
    if (ticket !== this.opening) return;

    // Everything below swaps in one go, so there is never a frame that shows
    // the old picture with the new shapes.
    this.image = payload;
    this.scene.load(payload);
    this.undo.clear();
    this.pendingGeometry = null;
    this.mips = pixels ? pixels.mips : null;
    // The database is the authority on size, but if a row predates the EXIF
    // work its dimensions could disagree with the pixels. Trust what decoded,
    // or every box on this image is drawn in the wrong space.
    if (pixels && pixels.width && pixels.height) {
      this.scene.width = pixels.width;
      this.scene.height = pixels.height;
    }
    this.view.setImageSize(this.scene.width, this.scene.height);
    this.view.fit(this.canvas.getBoundingClientRect());
    this.strip.setCurrent(imageId);
    this.imageName.textContent = payload.name;
    this.setStatusDot("saved");
    this.cancelTool();
    this.updateReviewPanel(null);
    this.scene.invalidate();
    this.say(
      pixels
        ? `${payload.name} — ${payload.shapes.length} shape(s)`
        : "that image could not be loaded — its shapes are still editable",
    );

    // Warm both neighbours: N is the common case, E the second most common.
    for (const neighbour of [payload.next, payload.previous]) {
      if (neighbour) this.pixels(neighbour);
    }
  }

  // Decoded pixels by image id, least recently used first. Kept small on
  // purpose: each entry is a full-resolution image plus its mips, and three
  // covers the previous, current and next image.
  pixels(imageId) {
    let entry = this.pixelCache.get(imageId);
    if (entry) {
      this.pixelCache.delete(imageId);
    } else {
      entry = decodePixels(this.api.imageUrl(this.projectId, imageId));
      // A failure is not cached: the next attempt may well succeed.
      entry.then((result) => {
        if (!result && this.pixelCache.get(imageId) === entry) this.pixelCache.delete(imageId);
      });
    }
    this.pixelCache.set(imageId, entry);
    while (this.pixelCache.size > PIXEL_CACHE) {
      const oldest = this.pixelCache.keys().next().value;
      const evicted = this.pixelCache.get(oldest);
      this.pixelCache.delete(oldest);
      evicted.then((result) => {
        if (result && result.mips !== this.mips) releaseMips(result.mips);
      });
    }
    return entry;
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
    // One save at a time. Two in flight would both carry the same base
    // version, the second would 409, and the conflict path would reload the
    // server copy over the user's newest edit.
    while (this.saving) await this.saving;
    if (!this.image) return;
    if (!this.scene.changed) {
      // Nothing to send. Say so, rather than leaving "unsaved" on screen --
      // an indicator that is wrong even once is one nobody trusts again, and
      // this is the only thing telling people their work is safe.
      this.setStatusDot("saved");
      return;
    }
    const imageId = this.scene.imageId;
    const revision = this.scene.revision;
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
        // The server may assign a new id to any shape, not only drafts -- an
        // undone delete comes back under an id it no longer has. Adopt every
        // one, or the next save inserts that shape a second time.
        (result.ids || []).forEach((serverId, index) => {
          const local = localIds[index];
          if (local && local !== serverId) this.scene.reid(local, serverId);
        });
        if (result.labels) this.rail.setLabels(result.labels);
        if (result.state) this.strip.setState(imageId, result.state);
        this.scene.invalidate();
        if (this.scene.revision === revision) {
          this.scene.changed = false;
          this.setStatusDot("saved");
        } else {
          // Edited while the request was in flight: that work is not saved
          // yet, so neither the flag nor the dot may say it is.
          this.queueSave();
        }
      })
      .catch(async (error) => {
        if (this.scene.imageId !== imageId) {
          this.say(`an earlier image failed to save: ${describe(error)}`);
          return;
        }
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
      if (made) {
        this.rail.pick(made);
        // Replace the "name this class" prompt: a hint still asking for a
        // name after it was given reads as the name not having taken.
        const position = this.rail.labels.indexOf(made);
        const key = position >= 0 && position < 10 ? ` · key ${(position + 1) % 10}` : "";
        this.say(`drawing "${made.name}"${key}`);
      }
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
      kind: geometry.kind || (geometry.polygon ? "polygon" : "box"),
      origin: "human",
      confidence: null,
      edited: false,
    });
    this.undo.run(addShape(this.scene, shape));
    this.drew();
    this.queueSave();
  }

  drew() {
    this.selectionFromDrawing = true;
  }

  assignClassToSelection(label) {
    if (this.selectionFromDrawing) return false;
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
    this.selectionFromDrawing = false;
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
      // Right-click belongs to the context-menu delete. Letting it through
      // to the tools added a polygon vertex or started a box underneath it.
      if (event.button !== 0) return;
      // Left-drag with the select tool draws a box: no mode to enter, which
      // is the entire interaction argument of this product. That includes a
      // drag starting on a shape you have not selected -- a handle on a
      // door, a plate on a car -- because nested objects are the common case,
      // and having to switch tools to draw inside a box was the first thing
      // to break the flow. A click there still selects; a drag on a shape
      // you *have* selected moves it.
      if (this.tool === this.tools.select) {
        const hit = hitTest(this.scene, ix, iy, this.view.scale, (s) => this.visible(s));
        if (!hit) {
          this.tools.box.pointerdown(event, ix, iy);
          this.drawingBox = true;
          return;
        }
        const owned = this.scene.selection.has(hit.id) && !this.selectionFromDrawing;
        if (hit.part === "body" && !owned && !event.shiftKey && !event.altKey) {
          this.pressed = { id: hit.id, sx: event.offsetX, sy: event.offsetY, ix, iy };
          return;
        }
      }
      this.tool.pointerdown(event, ix, iy);
    });

    this.canvas.addEventListener("pointermove", (event) => {
      const [ix, iy] = this.view.toImage(event.offsetX, event.offsetY);
      this.pointer = { x: event.offsetX, y: event.offsetY };
      if (this.wantsCrosshair()) this.scene.invalidate();
      if (this.panning) {
        this.view.panBy(event.offsetX - this.panning.x, event.offsetY - this.panning.y);
        this.panning = { x: event.offsetX, y: event.offsetY };
        this.scene.invalidate();
        return;
      }
      if (this.pressed) {
        // Past a few pixels a press on a shape becomes a drawn box.
        const { sx, sy } = this.pressed;
        if (Math.hypot(event.offsetX - sx, event.offsetY - sy) > 4) {
          this.tools.box.pointerdown(event, this.pressed.ix, this.pressed.iy);
          this.drawingBox = true;
          this.pressed = null;
          this.tools.box.pointermove(event, ix, iy);
        }
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
      if (this.pressed) {
        // It never moved: a click, so it selects.
        const { id } = this.pressed;
        this.pressed = null;
        this.scene.select(id);
        this.onSelectionChanged();
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
    this.canvas.addEventListener("pointerleave", () => {
      this.pointer = null;
      this.scene.invalidate();
    });
    this.canvas.addEventListener("pointerup", release);
    this.canvas.addEventListener("pointercancel", release);

    this.canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        // Firefox reports lines, not pixels, and its zoom was a crawl.
        const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 400 : 1;
        const dx = event.deltaX * unit;
        const dy = event.deltaY * unit;
        if (event.ctrlKey) {
          // A trackpad pinch arrives as a ctrl+wheel with small deltas.
          this.view.zoomAt(event.offsetX, event.offsetY, Math.exp(-dy * 0.01));
        } else if (looksLikeTrackpad(event, dx, dy)) {
          // Two fingers move the picture, as in every map and design tool;
          // zooming on them made a trackpad unusable.
          this.view.panBy(-dx, -dy);
        } else {
          this.view.zoomAt(event.offsetX, event.offsetY, Math.exp(-dy * 0.0015));
        }
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

  // The same test the select tool uses. A bbox-only check here treated the
  // outer half of every resize handle as empty canvas -- grabbing a corner
  // started a new box instead -- and the hollow of a concave polygon as solid.
  hitSomething(ix, iy) {
    const hit = hitTest(this.scene, ix, iy, this.view.scale, (s) => this.visible(s));
    return hit ? hit.id : null;
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
        crosshair: this.wantsCrosshair() ? this.pointer : null,
        imageFilter: this.imageFilter(),
      });
    }
    this.updateAttributePanel();
    this.shapeList.update(
      this.scene,
      (shape) => this.visible(shape),
      `${this.hideAll}:${[...this.hiddenClasses].join(",")}:${this.minConfidence}`
    );
    requestAnimationFrame(() => this.frame());
  }

  // Guides show whenever the next left-drag would draw: over empty canvas,
  // with a drawing tool, or mid-draw. Over a shape they would only be noise.
  wantsCrosshair() {
    if (!this.pointer || this.panning || this.spacePanning) return false;
    if (this.drawingBox || this.tool !== this.tools.select) return true;
    // Wherever a drag would draw: empty canvas, or a shape not selected.
    const hover = this.scene.hover;
    return hover == null || !this.scene.selection.has(hover) || this.selectionFromDrawing;
  }

  // -- editing from the keyboard -------------------------------------------

  // Move the selection by whole image pixels: 1, or 10 with Shift. A held
  // key extends one undo step rather than filling the stack with repeats.
  nudge(dx, dy) {
    const selected = this.scene.selected().filter((s) => this.visible(s));
    if (!selected.length) return false;
    const now = performance.now();
    const ids = selected.map((s) => s.id).sort().join(",");
    const last = this.lastNudge;
    const extending =
      last && last.ids === ids && now - last.at < 700 && this.undo.past.at(-1) === last.command;

    const entries = extending ? last.command.entries : [];
    for (const shape of selected) {
      let entry = entries.find((e) => this.scene.resolve(e.id) === shape.id);
      if (!entry) {
        entry = { id: shape.id, before: snapshot(shape), after: null };
        entries.push(entry);
      }
      shape.bbox = shape.bbox.map((v, i) => v + (i % 2 === 0 ? dx : dy));
      if (shape.polygon) shape.polygon = shape.polygon.map(([x, y]) => [x + dx, y + dy]);
      this.scene.clampShape(shape);
      entry.after = snapshot(shape);
      this.markEdited(shape);
    }
    this.scene.touch();
    if (extending) {
      last.at = now;
    } else {
      const command = changeMany(this.scene, entries);
      this.undo.push(command);
      this.lastNudge = { ids, at: now, command };
    }
    this.queueSave();
    return true;
  }

  // Turn the selected polygons and rotated boxes about their centres.
  rotateSelection(degrees) {
    const selected = this.scene
      .selected()
      .filter((s) => s.polygon && s.polygon.length >= 3 && this.visible(s));
    if (!selected.length) return false;
    for (const shape of selected) {
      const before = snapshot(shape);
      const [cx, cy] = centroid(shape.polygon);
      shape.polygon = rotatePoints(shape.polygon, cx, cy, (degrees * Math.PI) / 180);
      rebuildBox(shape);
      this.scene.clampShape(shape);
      this.undo.push(changeGeometry(this.scene, shape.id, before, snapshot(shape)));
      this.markEdited(shape);
    }
    this.scene.touch();
    this.queueSave();
    return true;
  }

  // Tab walks the visible shapes. The only way to reach a small box buried
  // under a big one without moving the big one.
  cycleSelection(direction) {
    const ids = this.scene.order.filter((id) => this.visible(this.scene.get(id)));
    if (!ids.length) return;
    const current = this.scene.selection.size === 1 ? [...this.scene.selection][0] : null;
    const at = current ? ids.indexOf(current) : -1;
    const next = ids[(at + direction + ids.length) % ids.length] ?? ids[0];
    this.scene.select(next);
    this.onSelectionChanged();
    const shape = this.scene.get(next);
    const score = shape.confidence != null ? ` · ${Math.round(shape.confidence * 100)}%` : "";
    this.say(`${shape.label}${score}`);
  }

  copySelection() {
    const selected = this.scene.selected();
    if (!selected.length) return;
    this.clipboard = { imageId: this.scene.imageId, shapes: selected.map(cloneShape) };
    this.say(`copied ${selected.length} shape(s) — Ctrl+V pastes, on this image or another`);
  }

  // Pasting onto another image lands in place -- the point is carrying a box
  // to the next frame. On the same image it is offset, or the copy would sit
  // exactly on its original and look like nothing happened.
  pasteClipboard() {
    if (!this.clipboard || !this.clipboard.shapes.length || !this.image) return;
    const offset = this.clipboard.imageId === this.scene.imageId ? 10 : 0;
    const pasted = this.clipboard.shapes.map((source) =>
      this.scene.clampShape({
        ...cloneShape(source),
        id: draftId(),
        bbox: source.bbox.map((v) => v + offset),
        polygon: source.polygon ? source.polygon.map(([x, y]) => [x + offset, y + offset]) : null,
        origin: "human",
        confidence: null,
        edited: false,
      })
    );
    for (const shape of pasted) this.scene.add(shape);
    this.undo.push(addMany(this.scene, pasted));
    this.scene.clearSelection();
    for (const shape of pasted) this.scene.select(shape.id, { additive: true });
    this.drew();
    this.queueSave();
    this.say(`pasted ${pasted.length} shape(s)`);
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
      if (ctrl && event.key.toLowerCase() === "c" && this.scene.selection.size) {
        event.preventDefault();
        this.copySelection();
        return;
      }
      if (ctrl && event.key.toLowerCase() === "v") {
        event.preventDefault();
        this.pasteClipboard();
        return;
      }
      if (ctrl && event.key.toLowerCase() === "a") {
        event.preventDefault();
        this.scene.clearSelection();
        for (const id of this.scene.order) {
          if (this.visible(this.scene.get(id))) this.scene.select(id, { additive: true });
        }
        this.onSelectionChanged();
        return;
      }
      if (event.key === "Tab") {
        event.preventDefault();
        this.cycleSelection(event.shiftKey ? -1 : 1);
        return;
      }
      if (event.key.startsWith("Arrow")) {
        const step = event.shiftKey ? 10 : 1;
        const [dx, dy] = {
          ArrowLeft: [-step, 0],
          ArrowRight: [step, 0],
          ArrowUp: [0, -step],
          ArrowDown: [0, step],
        }[event.key] || [0, 0];
        event.preventDefault();
        // With nothing selected, arrows move the view instead.
        if (!this.nudge(dx, dy)) {
          this.view.panBy(-dx * 5, -dy * 5);
          this.scene.invalidate();
        }
        return;
      }

      // Actual size is Shift+0, by physical key so it works on any layout.
      // Plain 0 is class ten; it cannot be both.
      if (event.shiftKey && event.code === "Digit0") {
        event.preventDefault();
        this.view.actual(this.canvas.getBoundingClientRect());
        this.scene.invalidate();
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
          this.pendingGeometry = null;
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
        case "k":
          this.setTool("point");
          break;
        case "l":
          this.setTool("line");
          break;
        case "o":
          this.setTool("obb");
          break;
        case "[":
        case "]": {
          const step = (event.shiftKey ? 15 : 1) * (event.key === "[" ? -1 : 1);
          if (this.rotateSelection(step)) event.preventDefault();
          break;
        }
        case "f":
          this.view.fit(this.canvas.getBoundingClientRect());
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
        case "h":
          // See the bare image -- the fastest check of whether a box is tight.
          this.hideAll = !this.hideAll;
          this.dropHiddenFromSelection();
          this.scene.invalidate();
          this.say(this.hideAll ? "shapes hidden — H to show them" : "shapes shown");
          break;
        case "/":
          event.preventDefault();
          this.rail.focusInput();
          break;
        default:
          break;
      }

      if (event.key === "?" || (event.key === "/" && event.shiftKey)) {
        this.help.hidden = !this.help.hidden;
      }
    });

    // Alt-Tab with space held never delivers the keyup, and every click
    // afterwards would pan. Losing focus releases everything held.
    window.addEventListener("blur", () => {
      this.spacePanning = false;
      this.panning = null;
      this.canvas.style.cursor = this.tool === this.tools.select ? "default" : "crosshair";
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

    for (const key of ["brightness", "contrast"]) {
      const control = document.getElementById(key);
      control.value = String(this.look[key]);
      control.addEventListener("input", () => {
        this.look = { ...this.look, [key]: Number(control.value) };
        writePreference(`look:${this.projectId}`, this.look);
        this.scene.invalidate();
      });
      control.addEventListener("change", () => control.blur());
      // Double-click puts it back, the quickest way out of a bad adjustment.
      control.addEventListener("dblclick", () => {
        control.value = "100";
        control.dispatchEvent(new Event("input"));
      });
    }

    for (const box of document.querySelectorAll("#attributes [data-attr]")) {
      box.addEventListener("change", () => {
        this.setAttribute(box.dataset.attr, box.checked);
        box.blur();
      });
    }

    const slider = document.getElementById("conf");
    slider.addEventListener("input", () => {
      this.minConfidence = Number(slider.value) / 100;
      document.getElementById("conf-value").textContent = `${slider.value}%`;
      this.scene.invalidate();
    });
    // Hand the keyboard back once the drag ends. A focused range input counts
    // as a typing target, so every shortcut went dead until the next click.
    slider.addEventListener("change", () => slider.blur());

    document.getElementById("btn-accept").addEventListener("click", () => this.acceptAll());
    document.getElementById("btn-export").addEventListener("click", () => this.exportProject());

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
  imageFilter() {
    const { brightness, contrast } = this.look;
    if (brightness === 100 && contrast === 100) return "";
    return `brightness(${brightness}%) contrast(${contrast}%)`;
  }

  // -- attributes -----------------------------------------------------------

  // Shown for a selection, so the flags of what you are looking at are one
  // click away -- and set on every selected shape at once.
  updateAttributePanel() {
    const selected = this.scene.selected();
    const signature = `${this.scene.revision}|${selected.map((s) => s.id).join(",")}`;
    if (signature === this.attributeSignature) return;
    this.attributeSignature = signature;
    const panel = document.getElementById("attributes");
    panel.hidden = selected.length === 0;
    for (const box of panel.querySelectorAll("[data-attr]")) {
      const values = selected.map((s) => Number((s.attributes || {})[box.dataset.attr] || 0));
      box.checked = values.length > 0 && values.every(Boolean);
      box.indeterminate = values.some(Boolean) && !box.checked;
    }
  }

  setAttribute(key, on) {
    const selected = this.scene.selected();
    if (!selected.length) return;
    for (const shape of selected) {
      const before = { ...(shape.attributes || {}) };
      const after = { ...before, [key]: on ? 1 : 0 };
      if (Number(before[key] || 0) === after[key]) continue;
      this.undo.run(changeAttributes(this.scene, shape.id, before, after));
      this.markEdited(shape);
    }
    this.queueSave();
  }

  passesConfidence(shape) {
    if (shape.confidence == null) return true;
    return shape.confidence >= this.minConfidence;
  }

  // What is drawn and what can be grabbed. Three different reasons to hide,
  // and only the confidence slider is a judgement on the shape -- see
  // acceptAll, which must never treat a hidden *class* as rejected.
  visible(shape) {
    if (!shape || this.hideAll) return false;
    if (this.hiddenClasses.has(shape.labelIdx)) return false;
    return this.passesConfidence(shape);
  }

  toggleClassHidden(idx) {
    if (this.hiddenClasses.has(idx)) this.hiddenClasses.delete(idx);
    else this.hiddenClasses.add(idx);
    writePreference(`hidden:${this.projectId}`, [...this.hiddenClasses]);
    this.rail.setHidden(this.hiddenClasses);
    this.dropHiddenFromSelection();
    this.scene.invalidate();
  }

  // Never leave something selected that cannot be seen: Delete would remove
  // it, and arrows would move it, with no feedback at all.
  dropHiddenFromSelection() {
    for (const id of [...this.scene.selection]) {
      if (!this.visible(this.scene.get(id))) this.scene.selection.delete(id);
    }
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
    // Only the confidence slider rejects. A class hidden to declutter the
    // view is still wanted, and deleting it here would be silent data loss.
    const hidden = [...this.scene.shapes.values()].filter((s) => !this.passesConfidence(s));
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

  // -- export ---------------------------------------------------------------

  // The download is the point of all of this. Reviewed images only, unless
  // asked -- and skipped ones never, which is a promise, not a default.
  async exportProject() {
    if (!this.projectId) return;
    await this.flushSave();
    const to = document.getElementById("export-format").value;
    const everything = document.getElementById("export-all").checked;
    const button = document.getElementById("btn-export");
    button.disabled = true;
    this.say(`exporting as ${to.toUpperCase()}, ${everything ? "every image except skipped" : "reviewed images only"}…`);
    try {
      const job = await this.api.exportProject(this.projectId, {
        to,
        states: everything ? ["todo", "predicted", "in_progress", "reviewed"] : ["reviewed"],
      });
      let status = job;
      while (!["finished", "failed", "cancelled"].includes(status.state)) {
        await pause(600);
        status = await this.api.job(job.id);
      }
      if (status.state !== "finished") {
        this.say(`export ${status.state}: ${status.error || "no detail"}`);
        return;
      }
      this.say("export ready — downloading");
      window.location = job.download;
    } catch (error) {
      this.say(describe(error));
    } finally {
      button.disabled = false;
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
        kind: raw.kind || (raw.polygon ? "polygon" : "box"),
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
    this.say(
      {
        select: "select tool",
        box: "box tool: drag",
        polygon: "polygon tool: click points, Enter to close",
        line: "line tool: click points, Enter to finish",
        point: "point tool: click",
        obb: "rotated box: drag, then turn it with the handle or [ ]",
      }[name] || `${name} tool`
    );
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
          kind: raw.kind || (raw.polygon ? "polygon" : "box"),
          origin: "human",
          confidence: null,
          edited: false,
          attributes: attributesOf(raw),
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

// Mouse wheels step in large whole notches with no sideways component;
// trackpads scroll in small, often fractional, two-axis deltas. A guess, but
// a wrong guess costs one scroll, and always zooming made trackpads useless.
function looksLikeTrackpad(event, dx, dy) {
  if (event.deltaMode !== 0) return false;
  if (dx !== 0) return true;
  return Math.abs(dy) < 40 && !Number.isInteger(dy);
}

// Per-browser conveniences. Storage can be missing or refuse (private
// windows, blocked site data); the editor must work identically without it.
function readPreference(key, fallback) {
  try {
    const raw = localStorage.getItem(`easyannotate:${key}`);
    return raw == null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function writePreference(key, value) {
  try {
    localStorage.setItem(`easyannotate:${key}`, JSON.stringify(value));
  } catch {
    // Not persisted; still applied for this session.
  }
}

function pause(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isTypingTarget(node) {
  if (!node) return false;
  const tag = node.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || node.isContentEditable;
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

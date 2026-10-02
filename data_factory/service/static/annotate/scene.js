// The retained scene: what is on this image, what is selected, what changed.
//
// A canvas is never diffed, so there is no virtual DOM here and no framework.
// What there is instead is one authoritative model that the renderer reads and
// the tools mutate, plus a dirty flag so we only redraw when something moved.
//
// Shapes keep their server id. That is what makes a save a diff rather than a
// delete-and-recreate, which in turn is what lets a model shape remember that
// it came from a model after somebody nudges it.

let temporary = 0;

// Client-side ids for shapes the server has never seen. Deliberately not in
// the server's id format, so `save` can tell "new" from "existing" without a
// separate flag that could drift out of step.
export function draftId() {
  temporary += 1;
  return `draft-${temporary}`;
}

export const isDraft = (id) => typeof id === "string" && id.startsWith("draft-");

export class Scene {
  constructor() {
    this.shapes = new Map();
    this.order = [];
    this.selection = new Set();
    this.hover = null;
    this.dirty = true;
    this.changed = false;
    // Bumped on every edit. A save compares it before and after the round
    // trip: if it moved, the user kept working while the request was in
    // flight, and "saved" would be a lie.
    this.revision = 0;
    // old id -> new id, for every shape the server renamed. Undo commands
    // hold the id a shape had when they were recorded; without this, the
    // first autosave after a drag would turn Ctrl+Z into a silent no-op.
    this.aliases = new Map();
    this.imageId = null;
    this.version = 0;
    this.width = 0;
    this.height = 0;
  }

  invalidate() {
    this.dirty = true;
  }

  // Mark the model as differing from what the server holds. Separate from
  // `dirty`, which is only about pixels: panning is dirty but not changed, and
  // autosaving on a pan would be absurd.
  touch() {
    this.revision += 1;
    this.changed = true;
    this.dirty = true;
  }

  load(payload) {
    this.shapes.clear();
    this.order = [];
    this.selection.clear();
    this.hover = null;
    this.aliases.clear();
    this.imageId = payload.id;
    this.version = payload.version;
    this.width = payload.width;
    this.height = payload.height;
    this.changed = false;

    for (const raw of payload.shapes || []) {
      const shape = {
        id: raw.id,
        label: raw.label,
        labelIdx: raw.label_idx,
        bbox: raw.bbox.slice(),
        polygon: raw.polygon ? raw.polygon.map((p) => p.slice()) : null,
        origin: raw.origin || "human",
        confidence: raw.confidence,
        edited: !!raw.edited,
      };
      this.shapes.set(shape.id, shape);
      this.order.push(shape.id);
    }
    this.dirty = true;
  }

  add(shape) {
    if (!shape.id) shape.id = draftId();
    this.shapes.set(shape.id, shape);
    this.order.push(shape.id);
    this.touch();
    return shape;
  }

  // The current id of a shape known by any id it has ever had. A live id wins
  // over an alias, so a shape re-inserted under its old id is still found.
  resolve(id) {
    let current = id;
    for (let hops = 0; hops < 16 && !this.shapes.has(current); hops++) {
      const next = this.aliases.get(current);
      if (next === undefined) break;
      current = next;
    }
    return current;
  }

  has(id) {
    return this.shapes.has(this.resolve(id));
  }

  remove(id) {
    id = this.resolve(id);
    if (!this.shapes.has(id)) return null;
    const shape = this.shapes.get(id);
    this.shapes.delete(id);
    this.order = this.order.filter((each) => each !== id);
    this.selection.delete(id);
    if (this.hover === id) this.hover = null;
    this.touch();
    return shape;
  }

  // Re-insert at a known position. Undo needs this: putting a deleted shape
  // back on top of the stack would change what the user sees, so it goes back
  // exactly where it was.
  insertAt(shape, index) {
    this.shapes.set(shape.id, shape);
    this.order.splice(Math.max(0, Math.min(index, this.order.length)), 0, shape.id);
    this.touch();
    return shape;
  }

  indexOf(id) {
    return this.order.indexOf(this.resolve(id));
  }

  // Give a shape the id the server assigned it, keeping its place, its
  // selection and everything pointing at it. Called after a save, to turn
  // drafts into real shapes -- otherwise the next save re-sends them with no
  // id and they insert all over again.
  reid(oldId, newId) {
    const shape = this.shapes.get(oldId);
    if (!shape || oldId === newId) return;
    this.aliases.set(oldId, newId);
    shape.id = newId;
    this.shapes.delete(oldId);
    this.shapes.set(newId, shape);
    const index = this.order.indexOf(oldId);
    if (index >= 0) this.order[index] = newId;
    if (this.selection.delete(oldId)) this.selection.add(newId);
    if (this.hover === oldId) this.hover = newId;
  }

  get(id) {
    return this.shapes.get(this.resolve(id));
  }

  select(id, { additive = false } = {}) {
    if (!additive) this.selection.clear();
    if (id) this.selection.add(id);
    this.dirty = true;
  }

  clearSelection() {
    if (this.selection.size === 0) return;
    this.selection.clear();
    this.dirty = true;
  }

  selected() {
    return [...this.selection].map((id) => this.shapes.get(id)).filter(Boolean);
  }

  // Keep every coordinate inside the image. The server clamps too -- this is
  // so the user sees the box stop at the edge rather than watching it snap
  // back after a save.
  clampShape(shape) {
    const maxX = Math.max(0, this.width - 1);
    const maxY = Math.max(0, this.height - 1);
    const cx = (v) => Math.max(0, Math.min(Math.round(v), maxX));
    const cy = (v) => Math.max(0, Math.min(Math.round(v), maxY));
    let [x1, y1, x2, y2] = shape.bbox;
    if (x2 < x1) [x1, x2] = [x2, x1];
    if (y2 < y1) [y1, y2] = [y2, y1];
    shape.bbox = [cx(x1), cy(y1), cx(x2), cy(y2)];
    if (shape.polygon) {
      shape.polygon = shape.polygon.map(([x, y]) => [cx(x), cy(y)]);
      rebuildBox(shape);
    }
    return shape;
  }

  toWire() {
    return this.order.map((id) => {
      const shape = this.shapes.get(id);
      return {
        // A draft id is ours, not the server's; sending it would be asking the
        // server to update a row that does not exist.
        id: isDraft(shape.id) ? undefined : shape.id,
        label: shape.label,
        bbox: shape.bbox,
        polygon: shape.polygon,
        confidence: shape.confidence ?? null,
      };
    });
  }
}

// A polygon's box is derived, never independent. Letting them drift is how a
// shape ends up selectable somewhere it is not drawn.
export function rebuildBox(shape) {
  if (!shape.polygon || shape.polygon.length < 2) return shape;
  const xs = shape.polygon.map((p) => p[0]);
  const ys = shape.polygon.map((p) => p[1]);
  shape.bbox = [
    Math.round(Math.min(...xs)),
    Math.round(Math.min(...ys)),
    Math.round(Math.max(...xs)),
    Math.round(Math.max(...ys)),
  ];
  return shape;
}

export function cloneShape(shape) {
  return {
    ...shape,
    bbox: shape.bbox.slice(),
    polygon: shape.polygon ? shape.polygon.map((p) => p.slice()) : null,
  };
}

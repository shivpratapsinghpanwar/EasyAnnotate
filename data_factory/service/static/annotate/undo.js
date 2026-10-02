// Undo as a command stack: explicit do/undo pairs, client-side.
//
// Not snapshots of the scene. Snapshots of a thousand-shape image are large,
// and more importantly they cannot survive a failed save -- a snapshot taken
// before a save that the server rejected would restore state the server never
// had. A command knows how to reverse itself, so the stack stays valid
// whatever happened to the network.
//
// Depth is capped. An unbounded stack in a session that runs for a day is a
// slow memory leak that shows up as "the tab got sluggish".

export const MAX_DEPTH = 200;

export class UndoStack {
  constructor(depth = MAX_DEPTH) {
    this.depth = depth;
    this.past = [];
    this.future = [];
  }

  get canUndo() {
    return this.past.length > 0;
  }

  get canRedo() {
    return this.future.length > 0;
  }

  // `command` is { label, do(), undo() }. Runs it, then records it.
  run(command) {
    command.do();
    this.push(command);
    return command;
  }

  // For actions already applied by a tool as the user dragged -- recording
  // after the fact keeps the drag itself free of bookkeeping.
  push(command) {
    this.past.push(command);
    if (this.past.length > this.depth) this.past.shift();
    // A new action makes any redo branch unreachable, which is what everyone
    // expects and what avoids having to explain a tree to the user.
    this.future.length = 0;
  }

  undo() {
    const command = this.past.pop();
    if (!command) return null;
    command.undo();
    this.future.push(command);
    return command;
  }

  redo() {
    const command = this.future.pop();
    if (!command) return null;
    command.do();
    this.past.push(command);
    return command;
  }

  clear() {
    this.past.length = 0;
    this.future.length = 0;
  }
}

// -- the commands the editor actually uses ---------------------------------

import { cloneShape } from "./scene.js";

export function addShape(scene, shape) {
  return {
    label: "add shape",
    do: () => {
      if (!scene.has(shape.id)) scene.insertAt(shape, scene.order.length);
      scene.select(shape.id);
    },
    undo: () => scene.remove(shape.id),
  };
}

export function removeShape(scene, id) {
  const shape = cloneShape(scene.get(id));
  const index = scene.indexOf(id);
  return {
    label: "delete shape",
    do: () => scene.remove(id),
    // Back where it was, not on top: re-stacking it would silently change
    // which shape wins a hit test.
    undo: () => scene.insertAt(cloneShape(shape), index),
  };
}

// Geometry changes are recorded as before/after copies. A box drag produces
// one command on pointerup, not one per mouse move.
export function changeGeometry(scene, id, before, after) {
  return {
    label: "move shape",
    do: () => apply(scene, id, after),
    undo: () => apply(scene, id, before),
  };
}

export function changeLabel(scene, id, before, after) {
  return {
    label: "change class",
    do: () => {
      const shape = scene.get(id);
      if (!shape) return;
      shape.label = after.label;
      shape.labelIdx = after.labelIdx;
      scene.touch();
    },
    undo: () => {
      const shape = scene.get(id);
      if (!shape) return;
      shape.label = before.label;
      shape.labelIdx = before.labelIdx;
      scene.touch();
    },
  };
}

export function addMany(scene, shapes) {
  const copies = shapes.map(cloneShape);
  return {
    label: "copy shapes",
    do: () => {
      for (const shape of copies) {
        if (!scene.has(shape.id)) scene.insertAt(cloneShape(shape), scene.order.length);
      }
    },
    undo: () => {
      for (const shape of copies) scene.remove(shape.id);
    },
  };
}

function apply(scene, id, state) {
  const shape = scene.get(id);
  if (!shape) return;
  shape.bbox = state.bbox.slice();
  shape.polygon = state.polygon ? state.polygon.map((p) => p.slice()) : null;
  scene.touch();
}

export function snapshot(shape) {
  return {
    bbox: shape.bbox.slice(),
    polygon: shape.polygon ? shape.polygon.map((p) => p.slice()) : null,
  };
}

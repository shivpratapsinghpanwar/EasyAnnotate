// Select, move, resize, and edit polygon vertices.
//
// The default tool, and the one that has to feel right. Everything here was
// proven in the Phase 0 spike at 60fps with 300 shapes.
//
// The rule that keeps undo sane: a drag mutates the shape live, for feedback,
// but records exactly one command on pointerup. One undo per gesture, not one
// per mouse move.

import { hitTest } from "../hit.js";
import { centroid, rebuildBox, rotatePoints } from "../scene.js";
import { changeGeometry, snapshot } from "../undo.js";

export const name = "select";
export const cursor = "default";

export function create(editor) {
  let drag = null;

  return {
    name,

    pointerdown(event, ix, iy) {
      const { scene, view } = editor;
      const hit = hitTest(scene, ix, iy, view.scale, (s) => editor.visible(s));

      if (!hit) {
        if (!event.shiftKey) scene.clearSelection();
        // Dragging empty space pans. No modifier, no mode -- the same gesture
        // people already expect from every map.
        drag = { pan: true, x: event.offsetX, y: event.offsetY };
        return;
      }

      if (!scene.selection.has(hit.id)) {
        scene.select(hit.id, { additive: event.shiftKey });
        editor.onSelectionChanged();
      }

      const shape = scene.get(hit.id);

      if (hit.part === "rotate" && shape.polygon) {
        const [cx, cy] = centroid(shape.polygon);
        drag = {
          id: shape.id,
          part: "rotate",
          ix,
          iy,
          before: snapshot(shape),
          centre: [cx, cy],
          startAngle: Math.atan2(iy - cy, ix - cx),
        };
        return;
      }

      if (hit.part.startsWith("edge-") && shape.polygon) {
        // Alt on an edge midpoint inserts a vertex there. Without a modifier
        // the edge just moves the shape, so a stray click near an edge does
        // not silently add geometry.
        if (event.altKey) {
          const index = Number(hit.part.slice(5));
          const before = snapshot(shape);
          const [ax, ay] = shape.polygon[index];
          const [bx, by] = shape.polygon[(index + 1) % shape.polygon.length];
          shape.polygon.splice(index + 1, 0, [(ax + bx) / 2, (ay + by) / 2]);
          rebuildBox(shape);
          scene.touch();
          editor.undo.push(changeGeometry(scene, shape.id, before, snapshot(shape)));
          drag = { id: shape.id, part: `vertex-${index + 1}`, ix, iy, before: snapshot(shape) };
          return;
        }
        drag = { id: shape.id, part: "body", ix, iy, before: snapshot(shape) };
        return;
      }

      if (hit.part.startsWith("vertex-") && event.altKey && shape.polygon) {
        // Alt-click deletes a vertex -- but never below a triangle, which
        // would leave an outline that cannot enclose anything.
        const index = Number(hit.part.slice(7));
        // A line needs two points; a polygon three; a rotated box keeps four.
        const minimum = shape.kind === "line" ? 2 : 3;
        if (shape.kind !== "obb" && shape.polygon.length > minimum) {
          const before = snapshot(shape);
          shape.polygon.splice(index, 1);
          rebuildBox(shape);
          scene.touch();
          editor.undo.push(changeGeometry(scene, shape.id, before, snapshot(shape)));
        } else {
          editor.say(
            shape.kind === "obb"
              ? "a rotated box always has four corners"
              : `a ${shape.kind === "line" ? "line needs at least two" : "polygon needs at least three"} points`
          );
        }
        return;
      }

      drag = { id: hit.id, part: hit.part, ix, iy, before: snapshot(shape) };
    },

    pointermove(event, ix, iy) {
      const { scene, view } = editor;

      if (drag && drag.pan) {
        view.panBy(event.offsetX - drag.x, event.offsetY - drag.y);
        drag.x = event.offsetX;
        drag.y = event.offsetY;
        scene.invalidate();
        return;
      }

      if (!drag) {
        const hit = hitTest(scene, ix, iy, view.scale, (s) => editor.visible(s));
        const id = hit ? hit.id : null;
        if (id !== scene.hover) {
          scene.hover = id;
          scene.invalidate();
        }
        return;
      }

      const shape = scene.get(drag.id);
      if (!shape) return;
      const dx = ix - drag.ix;
      const dy = iy - drag.iy;

      if (drag.part === "body") {
        shape.bbox = shape.bbox.map((v, i) => v + (i % 2 === 0 ? dx : dy));
        if (shape.polygon) shape.polygon = shape.polygon.map(([x, y]) => [x + dx, y + dy]);
      } else if (drag.part === "rotate") {
        // Rotation is applied to the shape as it was when the drag began, so
        // a long drag does not accumulate rounding from every mouse move.
        const [cx, cy] = drag.centre;
        const angle = Math.atan2(iy - cy, ix - cx) - drag.startAngle;
        shape.polygon = rotatePoints(drag.before.polygon, cx, cy, angle);
        rebuildBox(shape);
      } else if (drag.part.startsWith("handle-")) {
        resize(shape, Number(drag.part.slice(7)), dx, dy);
      } else if (drag.part.startsWith("vertex-") && shape.polygon) {
        const index = Number(drag.part.slice(7));
        if (shape.kind === "obb") moveCorner(shape, index, ix, iy);
        else shape.polygon[index] = [ix, iy];
        rebuildBox(shape);
      }

      drag.ix = ix;
      drag.iy = iy;
      scene.touch();
    },

    pointerup() {
      if (drag && !drag.pan && drag.id) {
        const shape = editor.scene.get(drag.id);
        if (shape) {
          editor.scene.clampShape(shape);
          const after = snapshot(shape);
          // Only record a command if something actually moved. A click that
          // merely selects should not consume an undo step.
          if (JSON.stringify(after) !== JSON.stringify(drag.before)) {
            editor.undo.push(changeGeometry(editor.scene, drag.id, drag.before, after));
            editor.markEdited(shape);
            editor.queueSave();
          }
        }
      }
      drag = null;
    },

    cancel() {
      drag = null;
    },

    draft() {
      return null;
    },
  };
}

// Dragging one corner of a rotated box keeps it a rectangle: the opposite
// corner stays put and the two neighbours slide along their edges. Without
// this a corner drag would turn the box into an arbitrary quadrilateral, and
// every OBB format would then be lying about what it holds.
function moveCorner(shape, index, ix, iy) {
  const p = shape.polygon;
  const [ox, oy] = p[(index + 2) % 4];
  const u = unitVector(p[(index + 1) % 4], [ox, oy]);
  const v = unitVector(p[(index + 3) % 4], [ox, oy]);
  const a = (ix - ox) * u[0] + (iy - oy) * u[1];
  const b = (ix - ox) * v[0] + (iy - oy) * v[1];
  p[(index + 1) % 4] = [ox + a * u[0], oy + a * u[1]];
  p[(index + 3) % 4] = [ox + b * v[0], oy + b * v[1]];
  p[index] = [ox + a * u[0] + b * v[0], oy + a * u[1] + b * v[1]];
}

function unitVector([x, y], [fromX, fromY]) {
  const dx = x - fromX;
  const dy = y - fromY;
  const len = Math.hypot(dx, dy) || 1;
  return [dx / len, dy / len];
}

// Handle order matches `handlesOf`: four corners, then four edge midpoints.
// A handle dragged past its opposite side flips the box, which is normal and
// gets normalised by `clampShape` on release.
function resize(shape, handle, dx, dy) {
  const box = shape.bbox.slice();
  if (handle === 0 || handle === 3 || handle === 6) box[0] += dx;
  if (handle === 1 || handle === 2 || handle === 7) box[2] += dx;
  if (handle === 0 || handle === 1 || handle === 4) box[1] += dy;
  if (handle === 2 || handle === 3 || handle === 5) box[3] += dy;

  if (shape.polygon) {
    // Resizing a polygon scales its outline within the new box, so the shape
    // deforms with the handle instead of the box drifting away from the
    // outline it is supposed to describe.
    const [ox1, oy1, ox2, oy2] = shape.bbox;
    const ow = Math.max(1, ox2 - ox1);
    const oh = Math.max(1, oy2 - oy1);
    const nw = box[2] - box[0];
    const nh = box[3] - box[1];
    shape.polygon = shape.polygon.map(([x, y]) => [
      box[0] + ((x - ox1) / ow) * nw,
      box[1] + ((y - oy1) / oh) * nh,
    ]);
  }
  shape.bbox = box;
}

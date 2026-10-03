// Drawing a polygon: click vertices, close to finish.
//
// Three ways to close, because people reach for different ones and being
// wrong about which costs a whole outline: Enter, double-click, or clicking
// the first point again. Escape abandons; Backspace removes the last vertex,
// which is what you want after a misplaced click on point 40.

import { draftId, rebuildBox } from "../scene.js";
import { addShape } from "../undo.js";

export const name = "polygon";
export const cursor = "crosshair";

//: Screen pixels within which a click counts as "on the first point".
const CLOSE_RADIUS = 12;

export function create(editor) {
  let points = [];
  let cursorAt = null;

  function finish() {
    if (points.length < 3) {
      editor.say("a polygon needs at least three points");
      return;
    }
    const outline = points.map(([x, y]) => [Math.round(x), Math.round(y)]);
    points = [];
    cursorAt = null;

    const label = editor.currentLabel();
    const geometry = rebuildBox({ polygon: outline, bbox: [0, 0, 0, 0] });

    if (!label) {
      editor.promptForLabel({ bbox: geometry.bbox, polygon: outline });
      return;
    }

    const shape = editor.scene.clampShape({
      id: draftId(),
      label: label.name,
      labelIdx: label.idx,
      bbox: geometry.bbox,
      polygon: outline,
      origin: "human",
      confidence: null,
      edited: false,
    });
    editor.undo.run(addShape(editor.scene, shape));
    editor.drew();
    editor.queueSave();
  }

  return {
    name,

    pointerdown(event, ix, iy) {
      if (event.detail >= 2) {
        finish();
        return;
      }
      if (points.length >= 3) {
        const [fx, fy] = points[0];
        const near =
          Math.hypot(fx - ix, fy - iy) * editor.view.scale <= CLOSE_RADIUS;
        if (near) {
          finish();
          return;
        }
      }
      points.push([ix, iy]);
      editor.scene.invalidate();
    },

    pointermove(event, ix, iy) {
      if (!points.length) return;
      cursorAt = [ix, iy];
      editor.scene.invalidate();
    },

    pointerup() {},

    key(event) {
      if (event.key === "Enter") {
        finish();
        return true;
      }
      if (event.key === "Backspace" && points.length) {
        points.pop();
        editor.scene.invalidate();
        return true;
      }
      if (event.key === "Escape" && points.length) {
        this.cancel();
        return true;
      }
      return false;
    },

    cancel() {
      points = [];
      cursorAt = null;
      editor.scene.invalidate();
    },

    draft() {
      if (!points.length) return null;
      return { kind: "polygon", points, cursor: cursorAt };
    },
  };
}

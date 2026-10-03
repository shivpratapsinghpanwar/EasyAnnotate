// A polyline: click the vertices, Enter or double-click to finish.
//
// Lane markings, seams, cables, edges -- things with length and no area.
// Same gestures as the polygon tool, minus closing: there is no first point
// to click back on, and two points are already a line.

import { draftId, rebuildBox } from "../scene.js";
import { addShape } from "../undo.js";

export const name = "line";
export const cursor = "crosshair";

export function create(editor) {
  let points = [];
  let cursorAt = null;

  function finish() {
    if (points.length < 2) {
      editor.say("a line needs at least two points");
      return;
    }
    const outline = points.map(([x, y]) => [Math.round(x), Math.round(y)]);
    points = [];
    cursorAt = null;

    const label = editor.currentLabel();
    const geometry = rebuildBox({ polygon: outline, bbox: [0, 0, 0, 0] });
    if (!label) {
      editor.promptForLabel({ bbox: geometry.bbox, polygon: outline, kind: "line" });
      return;
    }
    const shape = editor.scene.clampShape({
      id: draftId(),
      label: label.name,
      labelIdx: label.idx,
      bbox: geometry.bbox,
      polygon: outline,
      kind: "line",
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
      return { kind: "line", points, cursor: cursorAt };
    },
  };
}

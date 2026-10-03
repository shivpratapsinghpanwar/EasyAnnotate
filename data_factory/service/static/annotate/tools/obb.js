// A rotated box: drag it out like a box, then turn it.
//
// Drawn upright first, because that is one gesture and most objects start
// near upright anyway; the rotation handle above the box, or [ and ], do the
// turning. Stored as a four-corner polygon with kind "obb" -- which is
// exactly what YOLO-OBB and DOTA write, so export is a straight copy.

import { draftId } from "../scene.js";
import { addShape } from "../undo.js";

export const name = "obb";
export const cursor = "crosshair";

const MIN_SIZE = 3;

export function create(editor) {
  let start = null;
  let current = null;

  return {
    name,

    pointerdown(event, ix, iy) {
      start = [ix, iy];
      current = [ix, iy];
    },

    pointermove(event, ix, iy) {
      if (!start) return;
      current = [ix, iy];
      editor.scene.invalidate();
    },

    pointerup() {
      if (!start || !current) {
        start = current = null;
        return;
      }
      const [ax, ay] = start;
      const [bx, by] = current;
      start = current = null;
      if (Math.abs(bx - ax) < MIN_SIZE || Math.abs(by - ay) < MIN_SIZE) {
        editor.scene.invalidate();
        return;
      }
      const x1 = Math.round(Math.min(ax, bx));
      const y1 = Math.round(Math.min(ay, by));
      const x2 = Math.round(Math.max(ax, bx));
      const y2 = Math.round(Math.max(ay, by));
      // Clockwise from the top-left, so edge 0 is the top: the rotation
      // handle hangs off it, and DOTA readers expect a consistent order.
      const corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];

      const label = editor.currentLabel();
      if (!label) {
        editor.promptForLabel({ bbox: [x1, y1, x2, y2], polygon: corners, kind: "obb" });
        return;
      }
      const shape = editor.scene.clampShape({
        id: draftId(),
        label: label.name,
        labelIdx: label.idx,
        bbox: [x1, y1, x2, y2],
        polygon: corners,
        kind: "obb",
        origin: "human",
        confidence: null,
        edited: false,
      });
      editor.undo.run(addShape(editor.scene, shape));
      editor.drew();
      editor.queueSave();
      editor.say("drag the handle above the box to rotate it · [ and ] turn it from the keyboard");
    },

    cancel() {
      start = current = null;
      editor.scene.invalidate();
    },

    draft() {
      if (!start || !current) return null;
      const [ax, ay] = start;
      const [bx, by] = current;
      return { kind: "box", bbox: [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)] };
    },
  };
}

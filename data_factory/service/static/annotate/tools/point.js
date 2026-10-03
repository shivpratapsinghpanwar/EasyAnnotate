// A point: one click.
//
// Keypoints, landmarks, "the thing is here" -- the fastest annotation there
// is, so it must cost exactly one click and nothing else. A point is stored as
// a degenerate box (x, y, x, y) with kind "point", which keeps every writer
// that only knows boxes from having to learn anything.

import { draftId } from "../scene.js";
import { addShape } from "../undo.js";

export const name = "point";
export const cursor = "crosshair";

export function create(editor) {
  return {
    name,

    pointerdown(event, ix, iy) {
      const x = Math.round(ix);
      const y = Math.round(iy);
      const label = editor.currentLabel();
      if (!label) {
        editor.promptForLabel({ bbox: [x, y, x, y], polygon: null, kind: "point" });
        return;
      }
      const shape = editor.scene.clampShape({
        id: draftId(),
        label: label.name,
        labelIdx: label.idx,
        bbox: [x, y, x, y],
        polygon: null,
        kind: "point",
        origin: "human",
        confidence: null,
        edited: false,
      });
      editor.undo.run(addShape(editor.scene, shape));
      editor.drew();
      editor.queueSave();
    },

    pointermove() {},
    pointerup() {},
    cancel() {},
    draft() {
      return null;
    },
  };
}

// Drawing a box.
//
// There is no mode to enter. Left-drag on empty canvas draws a box in the
// current class -- that is the whole interaction, and it is the reason this
// tool is usually never selected explicitly: `select` hands off to it when a
// drag starts on empty space with a class chosen.

import { draftId } from "../scene.js";
import { addShape } from "../undo.js";

export const name = "box";
export const cursor = "crosshair";

//: Below this, in image pixels, it was a click and not a drag. Without it,
//: every stray click leaves a zero-area box that exports as a broken label.
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
      const [x1, y1] = start;
      const [x2, y2] = current;
      start = current = null;

      if (Math.abs(x2 - x1) < MIN_SIZE || Math.abs(y2 - y1) < MIN_SIZE) {
        editor.scene.invalidate();
        return;
      }

      const label = editor.currentLabel();
      if (!label) {
        // Nothing is lost -- the geometry is remembered and applied as soon as
        // a name is typed, so the user is never told to go and configure
        // something first.
        editor.promptForLabel({
          bbox: normalise(x1, y1, x2, y2),
          polygon: null,
        });
        return;
      }

      const shape = editor.scene.clampShape({
        id: draftId(),
        label: label.name,
        labelIdx: label.idx,
        bbox: normalise(x1, y1, x2, y2),
        polygon: null,
        origin: "human",
        confidence: null,
        edited: false,
      });
      editor.undo.run(addShape(editor.scene, shape));
      editor.drew();
      editor.queueSave();
    },

    cancel() {
      start = current = null;
      editor.scene.invalidate();
    },

    draft() {
      if (!start || !current) return null;
      return { kind: "box", bbox: normalise(start[0], start[1], current[0], current[1]) };
    },
  };
}

function normalise(x1, y1, x2, y2) {
  return [
    Math.round(Math.min(x1, x2)),
    Math.round(Math.min(y1, y2)),
    Math.round(Math.max(x1, x2)),
    Math.round(Math.max(y1, y2)),
  ];
}

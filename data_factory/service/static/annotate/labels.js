// The class rail: pick a class, or make one by typing its name.
//
// This is the component the whole product argument rests on. CVAT makes you
// define labels before a task exists; here a label is created by using it, and
// the rail is just a record of what you have used so far.
//
// Number keys pick a class. It is the largest throughput affordance in any
// annotator -- with a hand on the number row and a hand on the mouse, class
// switching costs nothing.

import { readableOn } from "./render.js";

export class LabelRail {
  constructor(root, { onPick, onCreate, onRename, onRetire }) {
    this.root = root;
    this.list = root.querySelector('[data-role="label-list"]');
    this.template = root.querySelector('[data-role="label-item"]');
    this.input = root.querySelector('[data-role="label-input"]');
    this.onPick = onPick;
    this.onCreate = onCreate;
    this.onRename = onRename;
    this.onRetire = onRetire;
    this.labels = [];
    this.current = null;

    this.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        const name = this.input.value.trim();
        if (name) {
          this.input.value = "";
          this.onCreate(name);
        }
        event.preventDefault();
      }
      if (event.key === "Escape") {
        this.input.value = "";
        this.input.blur();
      }
      // Number keys must reach the canvas, not this field.
      event.stopPropagation();
    });
  }

  focusInput() {
    this.input.focus();
    this.input.select();
  }

  // Naming a class is a step *inside* drawing, not a separate mode. Leaving
  // focus in the text field means the next keystroke -- P for polygon, or a
  // number key -- is typed into the box instead of reaching the canvas, and
  // the tool appears to be broken.
  releaseInput() {
    this.input.value = "";
    this.input.blur();
  }

  setLabels(labels) {
    this.labels = labels;
    if (this.current) {
      // Keep the selection pointing at the same *index*, not the same object:
      // a rename must not silently change which class you are drawing.
      this.current = labels.find((l) => l.idx === this.current.idx) || null;
    }
    if (!this.current && labels.length) this.current = labels[0];
    this.render();
  }

  pick(label) {
    this.current = label || null;
    this.render();
    this.onPick(this.current);
  }

  // 1-9 then 0 map to the first ten classes, in index order.
  pickByNumber(digit) {
    const position = digit === 0 ? 9 : digit - 1;
    const label = this.labels[position];
    if (label) this.pick(label);
    return !!label;
  }

  colourFor(shape) {
    const label =
      this.labels.find((l) => l.idx === shape.labelIdx) ||
      this.labels.find((l) => l.name === shape.label);
    return label ? label.colour : "#8a8a8a";
  }

  render() {
    this.list.replaceChildren();
    this.labels.forEach((label, position) => {
      const node = this.template.content.firstElementChild.cloneNode(true);
      const swatch = node.querySelector('[data-role="swatch"]');
      const name = node.querySelector('[data-role="name"]');
      const key = node.querySelector('[data-role="key"]');
      const retire = node.querySelector('[data-role="retire"]');

      swatch.style.background = label.colour;
      swatch.style.color = readableOn(label.colour);
      swatch.textContent = String(label.idx);
      name.textContent = label.name;
      key.textContent = position < 10 ? String(position === 9 ? 0 : position + 1) : "";
      node.dataset.idx = String(label.idx);
      if (this.current && this.current.idx === label.idx) node.dataset.current = "true";

      node.addEventListener("click", (event) => {
        if (event.target === retire) return;
        this.pick(label);
      });
      // Double-click a name to fix a typo. The index does not move, so every
      // file already exported keeps meaning what it meant.
      name.addEventListener("dblclick", () => {
        const next = prompt(`Rename "${label.name}" to:`, label.name);
        if (next && next.trim() && next.trim() !== label.name) {
          this.onRename(label.idx, next.trim());
        }
      });
      retire.addEventListener("click", (event) => {
        event.stopPropagation();
        this.onRetire(label.idx, label.name);
      });

      this.list.append(node);
    });
  }
}

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
  constructor(root, { onPick, onCreate, onRename, onRetire, onToggleHidden }) {
    this.root = root;
    this.list = root.querySelector('[data-role="label-list"]');
    this.template = root.querySelector('[data-role="label-item"]');
    this.input = root.querySelector('[data-role="label-input"]');
    this.onPick = onPick;
    this.onCreate = onCreate;
    this.onRename = onRename;
    this.onRetire = onRetire;
    this.onToggleHidden = onToggleHidden || (() => {});
    this.labels = [];
    this.current = null;
    this.hidden = new Set();
    // What is typed filters the list, and arrows walk what is left. Past ten
    // classes the number row runs out, and scrolling a list for "Handle" is
    // slower than typing "han".
    this.filter = "";
    this.highlight = -1;

    this.input.addEventListener("input", () => {
      this.filter = this.input.value.trim().toLowerCase();
      this.highlight = -1;
      this.render();
    });

    this.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        this.commit(this.input.value.trim());
      } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const matches = this.matches();
        if (matches.length) {
          const step = event.key === "ArrowDown" ? 1 : -1;
          this.highlight = (this.highlight + step + matches.length) % matches.length;
          this.render();
        }
      } else if (event.key === "Escape") {
        this.releaseInput();
      }
      // Number keys must reach the canvas, not this field.
      event.stopPropagation();
    });
  }

  matches() {
    if (!this.filter) return this.labels;
    return this.labels.filter((l) => l.name.toLowerCase().includes(this.filter));
  }

  // Enter in the class box. An existing class wins over making a new one --
  // **matched without regard to case**. Typing "open" while "Open" exists and
  // getting a second class is precisely how the door set came to hold
  // Close/close/Open/open, and every export after that splits one class in
  // two. Making a case variant is still possible: rename one afterwards.
  commit(name) {
    if (!name) return;
    const matches = this.matches();
    const exact = this.labels.find((l) => l.name.toLowerCase() === name.toLowerCase());
    const chosen = exact || (this.highlight >= 0 ? matches[this.highlight] : null);
    if (chosen) {
      this.releaseInput();
      this.pick(chosen);
      return;
    }
    this.releaseInput();
    this.onCreate(name);
  }

  setHidden(hidden) {
    this.hidden = new Set(hidden);
    this.render();
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
    if (this.filter || this.highlight >= 0) {
      this.filter = "";
      this.highlight = -1;
      this.render();
    }
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

  // The index a name maps to, or -1 for a class the server has not made yet.
  indexFor(name) {
    const label = this.labels.find((l) => l.name.toLowerCase() === String(name || "").toLowerCase());
    return label ? label.idx : -1;
  }

  colourFor(shape) {
    const label =
      this.labels.find((l) => l.idx === shape.labelIdx) ||
      this.labels.find((l) => l.name === shape.label);
    return label ? label.colour : "#8a8a8a";
  }

  render() {
    this.list.replaceChildren();
    const matches = this.matches();
    const highlighted = this.highlight >= 0 ? matches[this.highlight] : null;
    this.labels.forEach((label, position) => {
      const node = this.template.content.firstElementChild.cloneNode(true);
      const swatch = node.querySelector('[data-role="swatch"]');
      const name = node.querySelector('[data-role="name"]');
      const key = node.querySelector('[data-role="key"]');
      const retire = node.querySelector('[data-role="retire"]');
      const eye = node.querySelector('[data-role="eye"]');

      swatch.style.background = label.colour;
      swatch.style.color = readableOn(label.colour);
      swatch.textContent = String(label.idx);
      name.textContent = label.name;
      key.textContent = position < 10 ? String(position === 9 ? 0 : position + 1) : "";
      node.dataset.idx = String(label.idx);
      if (this.current && this.current.idx === label.idx) node.dataset.current = "true";
      if (this.filter && !matches.includes(label)) node.hidden = true;
      if (highlighted === label) node.dataset.highlight = "true";
      if (this.hidden.has(label.idx)) {
        node.dataset.hidden = "true";
        eye.title = `Show "${label.name}"`;
      } else {
        eye.title = `Hide "${label.name}" while you work`;
      }

      node.addEventListener("click", (event) => {
        if (event.target === retire || event.target === eye) return;
        this.pick(label);
      });
      eye.addEventListener("click", (event) => {
        event.stopPropagation();
        this.onToggleHidden(label.idx);
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

// The shapes on this image, as a list.
//
// A canvas answers "what is here" by looking; a list answers it by reading.
// The list is how you find the 4px box hiding under a big one, see that an
// image holds eleven predictions when you can make out nine, or check what
// class something is without hovering over it.
//
// Rendered from the scene whenever it changes, never edited directly: the
// scene stays the one source of truth.

import { readableOn } from "./render.js";

export class ShapeList {
  constructor(root, { onSelect, onHover, colourFor }) {
    this.root = root;
    this.list = root.querySelector('[data-role="shape-list"]');
    this.count = root.querySelector('[data-role="shape-count"]');
    this.template = root.querySelector('[data-role="shape-item"]');
    this.onSelect = onSelect;
    this.onHover = onHover;
    this.colourFor = colourFor;
    this.signature = "";
  }

  // Cheap to call every frame: it only rebuilds when something it shows moved.
  update(scene, visible, extra = "") {
    const signature = [
      scene.imageId,
      scene.revision,
      scene.order.join(","),
      [...scene.selection].join(","),
      extra,
    ].join("|");
    if (signature === this.signature) return;
    this.signature = signature;
    this.render(scene, visible);
  }

  render(scene, visible) {
    this.list.replaceChildren();
    let shown = 0;
    scene.order.forEach((id) => {
      const shape = scene.get(id);
      if (!shape) return;
      const node = this.template.content.firstElementChild.cloneNode(true);
      const swatch = node.querySelector('[data-role="swatch"]');
      const name = node.querySelector('[data-role="name"]');
      const meta = node.querySelector('[data-role="meta"]');

      const colour = this.colourFor(shape);
      swatch.style.background = colour;
      swatch.style.color = readableOn(colour);
      name.textContent = shape.label;
      // A prediction nobody has touched says how sure the model was; a
      // corrected or hand-drawn shape has nothing to qualify.
      meta.textContent =
        shape.origin === "model" && !shape.edited && shape.confidence != null
          ? `${Math.round(shape.confidence * 100)}%`
          : "";
      if (scene.selection.has(shape.id)) node.dataset.current = "true";
      if (!visible(shape)) node.dataset.hidden = "true";
      else shown += 1;

      node.addEventListener("click", (event) => this.onSelect(shape.id, event.shiftKey));
      node.addEventListener("mouseenter", () => this.onHover(shape.id));
      node.addEventListener("mouseleave", () => this.onHover(null));
      this.list.append(node);
    });
    const total = scene.order.length;
    this.count.textContent = total === shown ? `${total}` : `${shown} of ${total} shown`;
  }
}

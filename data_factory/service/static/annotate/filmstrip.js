// The image list down the side: where you are, and how much is left.
//
// Thumbnails load lazily through an IntersectionObserver rather than all at
// once. A 20,000-image project would otherwise open twenty thousand requests
// on the first paint, and the browser would spend the next minute doing
// nothing useful.
//
// State is shown as a colour bar rather than an icon, because the useful
// question is "how much of this strip is green", answered by glancing rather
// than reading.

export const STATE_COLOURS = {
  todo: "#3a3a3a",
  predicted: "#b08900",
  in_progress: "#3b7dd8",
  reviewed: "#3cb44b",
  skipped: "#8a4b8a",
};

export class Filmstrip {
  constructor(root, { onOpen, thumbUrl }) {
    this.root = root;
    this.list = root.querySelector('[data-role="strip-list"]');
    this.template = root.querySelector('[data-role="strip-item"]');
    this.counts = root.querySelector('[data-role="strip-counts"]');
    this.onOpen = onOpen;
    this.thumbUrl = thumbUrl;
    this.nodes = new Map();
    this.currentId = null;

    this.observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const img = entry.target;
          if (!img.dataset.src) continue;
          img.src = img.dataset.src;
          delete img.dataset.src;
          this.observer.unobserve(img);
        }
      },
      { root: this.list, rootMargin: "300px" }
    );
  }

  setImages(images) {
    this.list.replaceChildren();
    this.nodes.clear();

    for (const image of images) {
      const node = this.template.content.firstElementChild.cloneNode(true);
      const img = node.querySelector("img");
      const name = node.querySelector('[data-role="strip-name"]');
      const bar = node.querySelector('[data-role="strip-state"]');

      img.dataset.src = this.thumbUrl(image.id);
      img.alt = image.name;
      // A broken thumbnail is a normal finding in a real dataset -- a corrupt
      // file is one of the things inspect.health looks for -- so it must not
      // leave a browser's broken-image glyph implying the tool failed.
      img.addEventListener("error", () => {
        img.remove();
        node.dataset.broken = "true";
      });
      name.textContent = image.name;
      name.title = image.path;
      bar.style.background = STATE_COLOURS[image.state] || STATE_COLOURS.todo;
      node.dataset.id = image.id;
      node.dataset.state = image.state;

      node.addEventListener("click", () => this.onOpen(image.id));
      this.observer.observe(img);
      this.list.append(node);
      this.nodes.set(image.id, node);
    }
    if (this.currentId) this.setCurrent(this.currentId);
  }

  setCurrent(imageId) {
    for (const [id, node] of this.nodes) {
      if (id === imageId) node.dataset.current = "true";
      else delete node.dataset.current;
    }
    this.currentId = imageId;
    const node = this.nodes.get(imageId);
    if (node) node.scrollIntoView({ block: "nearest" });
  }

  setState(imageId, state) {
    const node = this.nodes.get(imageId);
    if (!node) return;
    node.dataset.state = state;
    const bar = node.querySelector('[data-role="strip-state"]');
    if (bar) bar.style.background = STATE_COLOURS[state] || STATE_COLOURS.todo;
  }

  setCounts(counts) {
    const done = counts.reviewed || 0;
    const total = counts.total || 0;
    const percent = total ? Math.round((done / total) * 100) : 0;
    this.counts.textContent = `${done} / ${total} reviewed (${percent}%)`;
  }
}

// The image list down the side: where you are, and how much is left.
//
// Virtual, because projects are large. A spacer as tall as every row would be
// gives the scrollbar its true length; only the rows on screen exist in the
// DOM, and their data arrives a page at a time as you scroll. The first
// version built a node per image and stopped at 2,000 -- the door set alone
// has 15,514.
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

export const ROW_HEIGHT = 41;
const PAGE = 200;
const OVERSCAN = 8;
// Rows kept built while off screen, so scrolling back does not refetch or
// re-decode thumbnails. Bounded: an unbounded cache is a slow leak.
const NODE_CACHE = 400;

export class Filmstrip {
  constructor(root, { onOpen, thumbUrl, fetchPage }) {
    this.root = root;
    this.list = root.querySelector('[data-role="strip-list"]');
    this.template = root.querySelector('[data-role="strip-item"]');
    this.counts = root.querySelector('[data-role="strip-counts"]');
    this.filterBox = root.querySelector('[data-role="strip-filter"]');
    this.onOpen = onOpen;
    this.thumbUrl = thumbUrl;
    this.fetchPage = fetchPage;

    // The filter box holds "state" or "state:order". Kept apart here because
    // counts are per state, while the order only changes what comes first.
    this.filter = "";
    this.order = "";
    this.total = 0;
    this.lastCounts = null;
    this.pages = new Map(); // page number -> images[] (or a pending promise)
    this.nodes = new Map(); // image id -> row node, LRU by insertion order
    this.currentId = null;
    this.generation = 0;

    this.spacer = document.createElement("div");
    this.spacer.style.position = "relative";
    this.list.replaceChildren(this.spacer);

    let queued = false;
    this.list.addEventListener("scroll", () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        this.draw();
      });
    });
    new ResizeObserver(() => this.draw()).observe(this.list);

    if (this.filterBox) {
      this.filterBox.addEventListener("change", () => {
        this.setFilter(this.filterBox.value);
        this.filterBox.blur();
      });
    }
  }

  // -- data ---------------------------------------------------------------

  // Forget every loaded page and start over -- after an import, or when the
  // filter changes what the list contains.
  reset(counts) {
    this.generation += 1;
    this.pages.clear();
    if (counts) this.setCounts(counts);
    this.draw();
  }

  setFilter(value) {
    const [state, order] = String(value || "").split(":");
    this.filter = state || "";
    this.order = order || "";
    this.list.scrollTop = 0;
    this.reset(this.lastCounts);
    if (this.currentId) this.reveal(this.currentId);
  }

  imageAt(index) {
    const page = this.pages.get(Math.floor(index / PAGE));
    return Array.isArray(page) ? page[index % PAGE] || null : null;
  }

  ensurePage(number) {
    if (this.pages.has(number) || number * PAGE >= this.total) return;
    const generation = this.generation;
    const pending = this.fetchPage({ state: this.filter, order: this.order, offset: number * PAGE, limit: PAGE })
      .then((payload) => {
        // A filter change or reset while this was in flight makes it stale.
        if (generation !== this.generation) return;
        this.pages.set(number, payload.images);
        this.draw();
      })
      .catch(() => {
        if (generation === this.generation) this.pages.delete(number);
      });
    this.pages.set(number, pending);
  }

  // -- drawing ------------------------------------------------------------

  draw() {
    this.spacer.style.height = `${this.total * ROW_HEIGHT}px`;
    const top = this.list.scrollTop;
    const height = this.list.clientHeight || 600;
    const first = Math.max(0, Math.floor(top / ROW_HEIGHT) - OVERSCAN);
    const last = Math.min(this.total - 1, Math.ceil((top + height) / ROW_HEIGHT) + OVERSCAN);

    const wanted = [];
    for (let index = first; index <= last; index++) {
      this.ensurePage(Math.floor(index / PAGE));
      const image = this.imageAt(index);
      if (image) wanted.push([index, image]);
    }

    const keep = new Set();
    for (const [index, image] of wanted) {
      const node = this.nodeFor(image);
      node.style.top = `${index * ROW_HEIGHT}px`;
      keep.add(node);
      if (node.parentNode !== this.spacer) this.spacer.append(node);
    }
    for (const child of [...this.spacer.children]) {
      if (!keep.has(child)) child.remove();
    }
  }

  nodeFor(image) {
    let node = this.nodes.get(image.id);
    if (node) {
      // Refresh LRU position, and pick up a state that changed meanwhile.
      this.nodes.delete(image.id);
      this.nodes.set(image.id, node);
      this.paintState(node, image.state);
      return node;
    }
    node = this.template.content.firstElementChild.cloneNode(true);
    const img = node.querySelector("img");
    const name = node.querySelector('[data-role="strip-name"]');

    img.src = this.thumbUrl(image.id);
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
    node.dataset.id = image.id;
    node.style.position = "absolute";
    node.style.left = "0";
    node.style.right = "0";
    this.paintState(node, image.state);
    if (image.id === this.currentId) node.dataset.current = "true";
    node.addEventListener("click", () => this.onOpen(image.id));

    this.nodes.set(image.id, node);
    while (this.nodes.size > NODE_CACHE) {
      const [oldest, stale] = this.nodes.entries().next().value;
      if (stale.isConnected) break; // never evict a row on screen
      this.nodes.delete(oldest);
    }
    return node;
  }

  paintState(node, state) {
    node.dataset.state = state;
    const bar = node.querySelector('[data-role="strip-state"]');
    if (bar) bar.style.background = STATE_COLOURS[state] || STATE_COLOURS.todo;
  }

  // -- the open image ---------------------------------------------------------

  setCurrent(imageId, position) {
    const previous = this.nodes.get(this.currentId);
    if (previous) delete previous.dataset.current;
    this.currentId = imageId;
    const node = this.nodes.get(imageId);
    if (node) node.dataset.current = "true";
    if (position != null) this.scrollToIndex(position);
    else this.reveal(imageId);
  }

  // Scroll so the open image is on screen, asking the server where it is when
  // it is not in a page already loaded.
  async reveal(imageId) {
    for (const [number, page] of this.pages) {
      if (!Array.isArray(page)) continue;
      const at = page.findIndex((image) => image.id === imageId);
      if (at >= 0) {
        this.lastPosition = number * PAGE + at;
        this.scrollToIndex(number * PAGE + at);
        return;
      }
    }
    try {
      const payload = await this.fetchPage({ state: this.filter, order: this.order, offset: 0, limit: 1, locate: imageId });
      if (payload.position != null && imageId === this.currentId) {
        this.lastPosition = payload.position;
        this.scrollToIndex(payload.position);
      }
    } catch {
      // Not knowing where it is only costs the scroll.
    }
  }

  scrollToIndex(index) {
    const top = index * ROW_HEIGHT;
    const view = this.list.clientHeight || 600;
    if (top < this.list.scrollTop || top + ROW_HEIGHT > this.list.scrollTop + view) {
      this.list.scrollTop = Math.max(0, top - view / 2 + ROW_HEIGHT / 2);
    }
    this.draw();
  }

  // Whether navigation should follow this list rather than file order: any
  // filter or ordering means the person chose a queue to work through.
  get isQueue() {
    return Boolean(this.filter || this.order);
  }

  // The image beside `imageId` in this queue. When the current image has
  // just left it -- accepted, so no longer "predicted" -- the one that slid
  // into its place is next, so `lastPosition` remembers where it was.
  async neighbour(imageId, direction) {
    const query = { state: this.filter, order: this.order };
    const located = await this.fetchPage({ ...query, offset: 0, limit: 1, locate: imageId });
    let offset;
    if (located.position != null) {
      this.lastPosition = located.position;
      offset = located.position + direction;
    } else {
      offset = (this.lastPosition ?? 0) + (direction > 0 ? 0 : -1);
    }
    if (offset < 0) return null;
    const page = await this.fetchPage({ ...query, offset, limit: 1 });
    return page.images && page.images.length ? page.images[0].id : null;
  }

  setState(imageId, state) {
    for (const page of this.pages.values()) {
      if (!Array.isArray(page)) continue;
      const image = page.find((each) => each.id === imageId);
      if (image) image.state = state;
    }
    const node = this.nodes.get(imageId);
    if (node) this.paintState(node, state);
  }

  setCounts(counts) {
    if (!counts) return;
    this.lastCounts = counts;
    const total = this.filter ? counts[this.filter] || 0 : counts.total || 0;
    if (total !== this.total) {
      // A different length means rows moved -- an image left this filter.
      // Pages already loaded are no longer where they claim to be.
      if (this.filter) {
        this.generation += 1;
        this.pages.clear();
      }
      this.total = total;
    }
    const done = counts.reviewed || 0;
    const all = counts.total || 0;
    const percent = all ? Math.round((done / all) * 100) : 0;
    this.counts.textContent = `${done} / ${all} reviewed (${percent}%)`;
    this.draw();
  }
}

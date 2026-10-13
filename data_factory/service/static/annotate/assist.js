// The Assist panel: whatever model-backed tools are installed, rendered
// generically.
//
// Named 'assist', not 'tools', because the editor already has tools/ --
// select, box, polygon -- and those are what a person draws with. These
// are optional helpers that may not be installed at all.
//
// There is no per-tool front-end code here and there must never be. A tool
// ships a spec (name, title, summary, input schema) and this renders a button
// for it -- the same "declare once, render everywhere" the operation registry
// already does for the batch panel.
//
// The panel is **absent** when no tool is installed, which is the normal case.
// Not greyed out, not an upsell: absent. A control you cannot use is worse
// than no control, and a plain `pip install data-factory` is a complete,
// working annotator.

export class AssistPanel {
  constructor(root, { onArm, onPrompt }) {
    this.root = root;
    this.list = root.querySelector('[data-role="tool-list"]');
    this.template = root.querySelector('[data-role="tool-item"]');
    this.promptTemplate = root.querySelector('[data-role="prompt-item"]');
    this.onArm = onArm;
    this.onPrompt = onPrompt || (() => {});
    this.tools = [];
    this.armed = null;
  }

  setTools(payload) {
    this.tools = ((payload && payload.tools) || []).filter((tool) => tool.kind !== "model");
    this.root.hidden = this.tools.length === 0;
    this.list.replaceChildren();

    for (const tool of this.tools) {
      // A model-file tool is command-line only (`easyannotate predict`):
      // loading one runs it, so the browser never gets a button for it.
      if (tool.kind === "model") continue;
      if (tool.kind === "prompt") {
        this.list.append(this.promptNode(tool));
        continue;
      }
      const node = this.template.content.firstElementChild.cloneNode(true);
      const button = node.querySelector("button");
      button.textContent = tool.title || tool.name;
      button.title = tool.summary || "";
      button.dataset.tool = tool.name;
      button.addEventListener("click", () => this.arm(tool));
      this.list.append(node);
    }
  }

  // A prompt tool takes words, not a gesture: "find every handle". It runs on
  // this image, or on every unreviewed image as a job -- which is where typing
  // a class name turns into a review queue instead of a drawing session.
  promptNode(tool) {
    const node = this.promptTemplate.content.firstElementChild.cloneNode(true);
    const text = node.querySelector('[data-role="prompt-text"]');
    const here = node.querySelector('[data-role="prompt-image"]');
    const all = node.querySelector('[data-role="prompt-all"]');
    node.querySelector('[data-role="prompt-title"]').textContent = tool.title || tool.name;
    text.title = tool.summary || "";
    text.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        this.onPrompt(tool, text.value.trim(), "image");
      }
      if (event.key === "Escape") text.blur();
      event.stopPropagation();
    });
    here.addEventListener("click", () => this.onPrompt(tool, text.value.trim(), "image"));
    all.addEventListener("click", () => this.onPrompt(tool, text.value.trim(), "all"));
    return node;
  }

  // Arming is a toggle, not a mode you can get stuck in: clicking the armed
  // tool again disarms it, and Escape does too.
  arm(tool) {
    this.armed = this.armed && this.armed.name === tool.name ? null : tool;
    for (const button of this.list.querySelectorAll("button")) {
      if (this.armed && button.dataset.tool === this.armed.name) {
        button.dataset.current = "true";
      } else {
        delete button.dataset.current;
      }
    }
    this.onArm(this.armed);
  }

  disarm() {
    if (this.armed) this.arm(this.armed);
  }

  // Whether this tool wants a box dragged out before it runs. Read from the
  // declared input schema rather than the tool's name, so a tool nobody here
  // has heard of still works.
  wantsBox(tool) {
    const properties = (tool && tool.inputs && tool.inputs.properties) || {};
    return Object.prototype.hasOwnProperty.call(properties, "box");
  }

  wantsPoints(tool) {
    const properties = (tool && tool.inputs && tool.inputs.properties) || {};
    return Object.prototype.hasOwnProperty.call(properties, "points");
  }
}

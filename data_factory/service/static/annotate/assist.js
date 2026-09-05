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
  constructor(root, { onArm }) {
    this.root = root;
    this.list = root.querySelector('[data-role="tool-list"]');
    this.template = root.querySelector('[data-role="tool-item"]');
    this.onArm = onArm;
    this.tools = [];
    this.armed = null;
  }

  setTools(payload) {
    this.tools = (payload && payload.tools) || [];
    this.root.hidden = this.tools.length === 0;
    this.list.replaceChildren();

    for (const tool of this.tools) {
      const node = this.template.content.firstElementChild.cloneNode(true);
      const button = node.querySelector("button");
      button.textContent = tool.title || tool.name;
      button.title = tool.summary || "";
      button.dataset.tool = tool.name;
      button.addEventListener("click", () => this.arm(tool));
      this.list.append(node);
    }
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
}

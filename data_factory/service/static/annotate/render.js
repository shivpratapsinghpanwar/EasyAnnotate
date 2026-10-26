// Drawing the canvas. Carried over from the Phase 0 spike, which measured it
// at 0.6ms per frame with 300 shapes on a 4000x3000 image.
//
// Two things here are measurements, not preferences:
//
//   * At full resolution use the plain three-argument drawImage. The explicit
//     source-rectangle form costs 15ms against 4.8ms for identical output,
//     because it drops a fast blit path.
//   * Below full resolution, draw from a mip. Asking the browser to resample
//     12 megapixels down to 1600px every frame costs 29ms.
//
// And one that is a design decision: model shapes are dashed, human shapes
// solid. That reads at any zoom and survives colour-vision deficiency, which a
// difference in hue would not -- and hue is already spent on the class.

import { handlesOf, rotationHandle } from "./hit.js";
import { mipFor } from "./viewport.js";

const BACKGROUND = "#11100f";
const HOVER = "#ffffff";
const HANDLE_SIZE = 5;

export function draw(ctx, rect, scene, view, mips, options = {}) {
  ctx.fillStyle = BACKGROUND;
  ctx.fillRect(0, 0, rect.width, rect.height);

  ctx.save();
  ctx.translate(view.tx, view.ty);
  ctx.scale(view.scale, view.scale);

  if (mips && mips.length) {
    // Brightness and contrast touch the picture only, never the outlines --
    // a box must stay the same colour however dark the scene is made.
    if (options.imageFilter) ctx.filter = options.imageFilter;
    // Pick by *device* pixels. Choosing by CSS pixels on a 2x screen draws
    // from a mip at half the resolution the display can show -- every fine
    // edge comes out soft, exactly where a tight box needs to see it.
    const dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
    const mip = mipFor(mips, view.width * view.scale * dpr);
    if (mip.width === view.width) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(mip, 0, 0);
    } else {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(mip, 0, 0, mip.width, mip.height, 0, 0, view.width, view.height);
    }
    ctx.filter = "none";
  }

  // Every stroke is divided by the scale so it stays two screen pixels wide at
  // any zoom. This is the difference between a canvas editor and a zoomed
  // picture of one.
  const unit = 1 / view.scale;

  const visible = options.visible || (() => true);

  for (const id of scene.order) {
    const shape = scene.shapes.get(id);
    if (!shape) continue;
    // The confidence slider hides, it does not delete. A hidden shape is
    // still in the scene and still saved -- filtering is a way of looking.
    if (!visible(shape)) continue;
    const colour = options.colourFor ? options.colourFor(shape) : "#4363d8";
    const selected = scene.selection.has(id);
    const hovered = scene.hover === id;

    ctx.strokeStyle = hovered && !selected ? HOVER : colour;
    ctx.lineWidth = (selected ? 3 : 2) * unit;
    // A model shape a person has already corrected is theirs now, so it stops
    // being dashed -- that is the visual feedback for "this one is done".
    const dashed = shape.origin === "model" && !shape.edited && !selected;
    ctx.setLineDash(dashed ? [8 * unit, 6 * unit] : []);
    // Confidence fades the outline, so a wall of low-confidence guesses reads
    // as noise rather than as ground truth.
    ctx.globalAlpha =
      shape.confidence == null ? 1 : 0.35 + 0.65 * Math.max(0, Math.min(1, shape.confidence));

    const kind = shape.kind || (shape.polygon && shape.polygon.length >= 3 ? "polygon" : "box");
    if (kind === "point") {
      drawPoint(ctx, shape.bbox, colour, unit, selected);
    } else if (kind === "line" && shape.polygon && shape.polygon.length >= 2) {
      tracePath(ctx, shape.polygon, false);
      ctx.stroke();
      if (selected) drawVertices(ctx, shape.polygon, colour, unit);
    } else if (shape.polygon && shape.polygon.length >= 3) {
      tracePath(ctx, shape.polygon, true);
      // A mask is a region, not a wire: a light fill in the class colour
      // shows what it covers at a glance, and the photo still shows through.
      if (kind === "polygon") {
        const alpha = ctx.globalAlpha;
        ctx.globalAlpha = alpha * (selected ? 0.3 : 0.22);
        ctx.fillStyle = colour;
        ctx.fill();
        ctx.globalAlpha = alpha;
      }
      ctx.stroke();
      if (selected) drawVertices(ctx, shape.polygon, colour, unit);
      if (selected && kind === "obb") drawRotationHandle(ctx, shape, colour, unit, view.scale);
    } else {
      const [x1, y1, x2, y2] = shape.bbox;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }

    if (selected && (kind === "box" || kind === "polygon")) drawHandles(ctx, shape, colour, unit);
    ctx.globalAlpha = 1;
  }

  if (options.draft) drawDraft(ctx, options.draft, unit);

  ctx.globalAlpha = 1;
  ctx.setLineDash([]);
  ctx.restore();

  // Labels are drawn in screen space, after restore, so text is always the
  // same size and upright regardless of zoom.
  if (options.showLabels !== false) drawLabels(ctx, scene, view, options);
  if (options.crosshair) drawCrosshair(ctx, rect, options.crosshair);
  if (options.draft && options.draft.kind === "box" && options.crosshair) {
    drawSize(ctx, options.draft.bbox, options.crosshair);
  }
}

// Full-canvas guides through the cursor. A box is only as tight as the
// person can judge where its edge meets the object, and a line running the
// whole width of the image makes that judgement against every pixel on the
// row instead of against the few under a mouse pointer. Two strokes -- dark
// under light -- so the guide reads on snow and on asphalt.
function drawCrosshair(ctx, rect, { x, y }) {
  ctx.save();
  const px = Math.round(x) + 0.5;
  const py = Math.round(y) + 0.5;
  for (const [style, width] of [["rgba(0,0,0,.55)", 3], ["rgba(255,255,255,.8)", 1]]) {
    ctx.strokeStyle = style;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(rect.width, py);
    ctx.moveTo(px, 0);
    ctx.lineTo(px, rect.height);
    ctx.stroke();
  }
  ctx.restore();
}

// The size of the box being drawn, in image pixels, beside the cursor. Tiny
// boxes are the commonest label defect inspect.health finds; seeing "6 × 4"
// while drawing is cheaper than finding it after training.
function drawSize(ctx, bbox, { x, y }) {
  const [x1, y1, x2, y2] = bbox;
  const text = `${Math.round(Math.abs(x2 - x1))} × ${Math.round(Math.abs(y2 - y1))}`;
  ctx.save();
  ctx.font = "11px ui-monospace, monospace";
  ctx.textBaseline = "top";
  const width = ctx.measureText(text).width + 8;
  ctx.fillStyle = "rgba(0,0,0,.7)";
  ctx.fillRect(x + 12, y + 12, width, 16);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(text, x + 16, y + 14);
  ctx.restore();
}

function tracePath(ctx, polygon, closed) {
  ctx.beginPath();
  ctx.moveTo(polygon[0][0], polygon[0][1]);
  for (let i = 1; i < polygon.length; i++) ctx.lineTo(polygon[i][0], polygon[i][1]);
  if (closed) ctx.closePath();
}

// A point is a ring with a dot: the ring reads at any zoom, and the hollow
// between them leaves the pixel it marks visible.
function drawPoint(ctx, [x, y], colour, unit, selected) {
  ctx.setLineDash([]);
  const r = (selected ? 9 : 7) * unit;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.lineWidth = 2 * unit;
  ctx.strokeStyle = "#ffffff";
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, r - 1.5 * unit, 0, Math.PI * 2);
  ctx.strokeStyle = colour;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, 2 * unit, 0, Math.PI * 2);
  ctx.fillStyle = colour;
  ctx.fill();
}

function drawRotationHandle(ctx, shape, colour, unit, scale) {
  const handle = rotationHandle(shape, scale);
  if (!handle) return;
  const [[ax, ay], [bx, by]] = shape.polygon;
  ctx.setLineDash([]);
  ctx.lineWidth = 1.5 * unit;
  ctx.strokeStyle = colour;
  ctx.beginPath();
  ctx.moveTo((ax + bx) / 2, (ay + by) / 2);
  ctx.lineTo(handle[0], handle[1]);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(handle[0], handle[1], HANDLE_SIZE * 1.2 * unit, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.stroke();
}

function drawHandles(ctx, shape, colour, unit) {
  ctx.setLineDash([]);
  ctx.fillStyle = colour;
  const size = HANDLE_SIZE * unit;
  for (const [hx, hy] of handlesOf(shape)) {
    ctx.fillRect(hx - size, hy - size, size * 2, size * 2);
  }
}

function drawVertices(ctx, polygon, colour, unit) {
  ctx.setLineDash([]);
  const size = HANDLE_SIZE * unit;
  for (const [x, y] of polygon) {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(x - size, y - size, size * 2, size * 2);
    ctx.fillStyle = colour;
    ctx.fillRect(x - size * 0.6, y - size * 0.6, size * 1.2, size * 1.2);
  }
}

// The shape currently being dragged out or clicked in. Kept out of the scene
// so an abandoned draw leaves nothing behind to clean up.
function drawDraft(ctx, draft, unit) {
  ctx.setLineDash([6 * unit, 4 * unit]);
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5 * unit;

  if (draft.kind === "box") {
    const [x1, y1, x2, y2] = draft.bbox;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  } else if ((draft.kind === "polygon" || draft.kind === "line") && draft.points.length) {
    ctx.beginPath();
    ctx.moveTo(draft.points[0][0], draft.points[0][1]);
    for (let i = 1; i < draft.points.length; i++) {
      ctx.lineTo(draft.points[i][0], draft.points[i][1]);
    }
    if (draft.cursor) ctx.lineTo(draft.cursor[0], draft.cursor[1]);
    ctx.stroke();

    ctx.setLineDash([]);
    const size = HANDLE_SIZE * unit;
    for (const [x, y] of draft.points) {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(x - size, y - size, size * 2, size * 2);
    }
  }
  ctx.setLineDash([]);
}

function drawLabels(ctx, scene, view, options) {
  ctx.save();
  ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
  ctx.textBaseline = "bottom";

  const visible = options.visible || (() => true);

  for (const id of scene.order) {
    const shape = scene.shapes.get(id);
    if (!shape || !visible(shape)) continue;
    const [x1, y1] = shape.bbox;
    const [sx, sy] = view.toScreen(x1, y1);
    // Off-screen labels are the bulk of the text cost on a busy image.
    if (sx < -200 || sy < -40 || sx > ctx.canvas.width || sy > ctx.canvas.height) continue;

    const colour = options.colourFor ? options.colourFor(shape) : "#4363d8";
    const text =
      shape.confidence == null
        ? shape.label
        : `${shape.label} ${Math.round(shape.confidence * 100)}%`;
    const width = ctx.measureText(text).width;

    ctx.globalAlpha = 0.92;
    ctx.fillStyle = colour;
    ctx.fillRect(sx, sy - 16, width + 10, 16);
    ctx.globalAlpha = 1;
    ctx.fillStyle = readableOn(colour);
    ctx.fillText(text, sx + 5, sy - 3);
  }
  ctx.restore();
}

// Perceived luminance, so a label on pale yellow is not white-on-white.
export function readableOn(hex) {
  const value = hex.replace("#", "");
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return 0.299 * r + 0.587 * g + 0.114 * b > 140 ? "#101010" : "#ffffff";
}

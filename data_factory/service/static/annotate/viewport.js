// The view transform, and the one technique that makes a canvas editor fast.
//
// Phase 0 measured this: drawing a 12-megapixel bitmap scaled down to 1600px
// wide costs 29ms per frame, because the browser resamples all 12 megapixels
// every time. Pre-building the image halved repeatedly and drawing from the
// level nearest the display size takes that to 0.6ms -- a 48x improvement, and
// the difference between unusable and idle.
//
// See spikes/canvas/README.md for the numbers.

export const MIN_SCALE = 0.02;
export const MAX_SCALE = 20;

export class Viewport {
  constructor() {
    this.scale = 1;
    this.tx = 0;
    this.ty = 0;
    this.width = 1;
    this.height = 1;
  }

  setImageSize(width, height) {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
  }

  toImage(px, py) {
    return [(px - this.tx) / this.scale, (py - this.ty) / this.scale];
  }

  toScreen(ix, iy) {
    return [ix * this.scale + this.tx, iy * this.scale + this.ty];
  }

  // Fit the whole image with a small margin. `F` in the editor.
  fit(rect) {
    this.scale =
      Math.min(rect.width / this.width, rect.height / this.height) * 0.95;
    this.centre(rect);
  }

  // 1:1 pixels. `0` in the editor, and the only zoom at which fine polygon
  // work is honest about what it is doing.
  actual(rect) {
    this.scale = 1;
    this.centre(rect);
  }

  centre(rect) {
    this.tx = (rect.width - this.width * this.scale) / 2;
    this.ty = (rect.height - this.height * this.scale) / 2;
  }

  // Zoom about a point, so the pixel under the cursor stays under the cursor.
  // Anchoring anywhere else is the single most disorienting thing an image
  // editor can do.
  zoomAt(px, py, factor) {
    const [ix, iy] = this.toImage(px, py);
    this.scale = clamp(this.scale * factor, MIN_SCALE, MAX_SCALE);
    this.tx = px - ix * this.scale;
    this.ty = py - iy * this.scale;
  }

  panBy(dx, dy) {
    this.tx += dx;
    this.ty += dy;
  }
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

// -- mip levels -------------------------------------------------------------

// Halve repeatedly down to 64px. Costs one pass over the image at load, and
// buys back most of the frame budget at every zoom below 1:1.
export function buildMips(source) {
  const levels = [source];
  let width = source.width;
  let height = source.height;

  while (width > 64 && height > 64) {
    const previous = levels[levels.length - 1];
    width = Math.max(1, Math.floor(width / 2));
    height = Math.max(1, Math.floor(height / 2));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(previous, 0, 0, width, height);
    levels.push(canvas);
  }
  return levels;
}

// The smallest level still at least as wide as we need, so we downsample a
// little rather than upsampling a blurry one.
export function mipFor(levels, displayWidth) {
  for (let i = levels.length - 1; i >= 0; i--) {
    if (levels[i].width >= displayWidth) return levels[i];
  }
  return levels[0];
}

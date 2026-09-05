// What is under the cursor.
//
// A linear scan, deliberately. Phase 0 measured hit-testing at 0.0ms median
// and still 0.4ms at 10,000 shapes on one image, against a realistic load of
// tens. A spatial index here would be complexity bought with no return.
//
// Priority is what makes it feel right: the selected shape's handles first,
// then its vertices, then shape bodies topmost-first. Without that ordering a
// resize handle sitting over another shape becomes unreachable.

//: Grab radius in *screen* pixels, converted at the current zoom, so a handle
//: is the same physical size whether you are at 5% or 800%.
export const TOLERANCE = 10;

export function handlesOf(shape) {
  const [x1, y1, x2, y2] = shape.bbox;
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  return [
    [x1, y1], [x2, y1], [x2, y2], [x1, y2],
    [mx, y1], [mx, y2], [x1, my], [x2, my],
  ];
}

export function hitTest(scene, ix, iy, scale, visible = null) {
  const tol = TOLERANCE / scale;
  // A shape the confidence slider is hiding must not be grabbable. Dragging
  // something you cannot see is indistinguishable from the tool being broken.
  const shown = visible || (() => true);

  for (const id of scene.selection) {
    const shape = scene.shapes.get(id);
    if (!shape || !shown(shape)) continue;

    const handles = handlesOf(shape);
    for (let h = 0; h < handles.length; h++) {
      if (near(ix, iy, handles[h], tol)) return { id, part: `handle-${h}` };
    }

    if (shape.polygon) {
      for (let v = 0; v < shape.polygon.length; v++) {
        if (near(ix, iy, shape.polygon[v], tol)) return { id, part: `vertex-${v}` };
      }
      // Edge midpoints insert a vertex. Checked after real vertices so a
      // three-point polygon, where midpoints crowd the corners, still lets you
      // grab the corner.
      const edge = nearestEdge(shape.polygon, ix, iy, tol);
      if (edge >= 0) return { id, part: `edge-${edge}` };
    }
  }

  for (let i = scene.order.length - 1; i >= 0; i--) {
    const shape = scene.shapes.get(scene.order[i]);
    if (!shape || !shown(shape)) continue;
    const inside = shape.polygon
      ? pointInPolygon(ix, iy, shape.polygon)
      : inBox(ix, iy, shape.bbox);
    if (inside) return { id: shape.id, part: "body" };
  }
  return null;
}

function near(x, y, [px, py], tol) {
  return Math.abs(x - px) <= tol && Math.abs(y - py) <= tol;
}

export function inBox(x, y, [x1, y1, x2, y2]) {
  return x >= x1 && x <= x2 && y >= y1 && y <= y2;
}

// Even-odd ray casting. Handles concave outlines, which matters: a door frame
// traced around its handle is concave, and a convex test would claim the hole
// is inside the shape.
export function pointInPolygon(x, y, points) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i];
    const [xj, yj] = points[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

// Index of the edge whose midpoint is within tolerance, or -1.
function nearestEdge(points, x, y, tol) {
  for (let i = 0; i < points.length; i++) {
    const [ax, ay] = points[i];
    const [bx, by] = points[(i + 1) % points.length];
    if (near(x, y, [(ax + bx) / 2, (ay + by) / 2], tol)) return i;
  }
  return -1;
}

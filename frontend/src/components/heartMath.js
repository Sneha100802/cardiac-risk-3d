/* Procedural heart: no download, no licence, and the artery tubes are GUARANTEED to lie on it.
 *
 * Surface = zero set of the classic implicit "heart" equation, in "formula space":
 *     x = left/right, y = depth (front/back), z = up
 *     F(x,y,z) = (x² + 2.25y² + z² - 1)³ - x²z³ - 0.1125·y²z³ ;  F < 0 inside, F > 0 outside.
 * This is a STYLISED heart (not patient-specific anatomy).
 * Viewer convention: we look at the FRONT (y > 0). Patient's left = viewer's right (+x).
 */

export function F(x, y, z) {
  const a = x * x + 2.25 * y * y + z * z - 1;
  return a * a * a - x * x * z * z * z - 0.1125 * y * y * z * z * z;
}

/** Distance from the origin to the surface along unit direction (dx,dy,dz): first inside->outside crossing. */
export function radiusAlong(dx, dy, dz) {
  const step = 0.02;
  let prev = 0;
  for (let r = step; r <= 2.2; r += step) {
    if (F(dx * r, dy * r, dz * r) > 0) {
      let lo = prev, hi = r;                         // bisection between inside (lo) and outside (hi)
      for (let i = 0; i < 30; i++) {
        const mid = (lo + hi) / 2;
        if (F(dx * mid, dy * mid, dz * mid) <= 0) lo = mid; else hi = mid;
      }
      return lo;
    }
    prev = r;
  }
  return prev;
}

/** x of the left/right silhouette edge at height z (sign = +1 viewer-right, -1 viewer-left). */
export function silhouetteX(z, sign = 1) {
  let lo = 0, hi = 1.6;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (F(mid, 0, z) <= 0) lo = mid; else hi = mid;
  }
  return sign * lo;
}

/** Point on the front (side=+1) or back (side=-1) surface above/below (x,z); pushed `lift` outwards. */
export function surfacePoint(x, z, side = 1, lift = 0.035) {
  const edge = silhouetteX(z, Math.sign(x) || 1);
  if (Math.abs(x) > Math.abs(edge) * 0.98) x = edge * 0.98;      // keep points inside the silhouette
  let lo = 0, hi = 1.4;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (F(x, side * mid, z) <= 0) lo = mid; else hi = mid;
  }
  const y = side * lo;
  const e = 1e-4;                                                  // outward normal = gradient of F
  let nx = F(x + e, y, z) - F(x - e, y, z);
  let ny = F(x, y + e, z) - F(x, y - e, z);
  let nz = F(x, y, z + e) - F(x, y, z - e);
  const n = Math.hypot(nx, ny, nz) || 1;
  return [x + (lift * nx) / n, y + (lift * ny) / n, z + (lift * nz) / n];
}

/** Silhouette point (y = 0) at height z; used where a vessel wraps from front to back. */
function edgePoint(sign, z, lift = 0.035) {
  return [silhouetteX(z, sign) + sign * lift, 0, z];
}

/* Vessel recipes. 'f' = front surface, 'b' = back surface, 'e' = silhouette edge.
 *   ['f', x, z]   ['b', x, z]   ['e', sign, z]
 * Tweak these numbers to move an artery; every point is snapped to the surface automatically. */
const RECIPES = {
  // Left anterior descending: runs down the front groove from the top towards the apex.
  LAD: [["f", 0.22, 0.8], ["f", 0.25, 0.45], ["f", 0.2, 0.1], ["f", 0.13, -0.3], ["f", 0.07, -0.62], ["f", 0.03, -0.88]],
  // Left circumflex: runs round the patient's left side (viewer's right) to the back.
  LCX: [["f", 0.3, 0.85], ["f", 0.65, 0.8], ["f", 0.95, 0.55], ["e", 1, 0.2], ["b", 0.95, -0.1], ["b", 0.75, -0.4], ["b", 0.5, -0.6]],
  // Right coronary: runs down the patient's right side (viewer's left), ending towards the bottom.
  RCA: [["f", -0.25, 0.85], ["f", -0.6, 0.78], ["f", -0.92, 0.5], ["f", -1.0, 0.2], ["f", -0.85, -0.2], ["f", -0.55, -0.55], ["f", -0.25, -0.8]],
};

export function buildVesselPaths() {
  const out = {};
  for (const [name, pts] of Object.entries(RECIPES)) {
    out[name] = pts.map(([kind, a, b]) =>
      kind === "f" ? surfacePoint(a, b, 1) : kind === "b" ? surfacePoint(a, b, -1) : edgePoint(a, b));
  }
  return out;                       // formula-space points [x, y, z]
}

/** Big vessels drawn for context only (not clickable): aorta and pulmonary trunk. */
export const GREAT_VESSELS = {
  aorta: { points: [[0.05, 0.12, 0.7], [0.05, 0.1, 1.15], [-0.15, 0.0, 1.5], [-0.55, -0.1, 1.55]], radius: 0.17, color: "#c9a39e" },
  pulmonary: { points: [[0.3, 0.4, 0.75], [0.4, 0.3, 1.1], [0.62, 0.1, 1.3]], radius: 0.13, color: "#9fb3c8" },
};

/** formula space (x, y=depth, z=up)  ->  three.js space (x, y=up, z=towards viewer) */
export const toThree = ([x, y, z]) => [x, z, y];

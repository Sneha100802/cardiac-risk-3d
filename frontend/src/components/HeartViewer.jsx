import { Suspense, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Html, OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { GREAT_VESSELS, buildVesselPaths, radiusAlong, toThree } from "./heartMath";

/* ------------------------------------------------------------------ *
 * 1. Probability -> colour   (0 = green, 0.5 = yellow, 1 = red)
 * ------------------------------------------------------------------ */
const LOW = new THREE.Color("#2e9e6b");
const MID = new THREE.Color("#e8b923");
const HIGH = new THREE.Color("#d4352f");
const NEUTRAL = new THREE.Color("#9aa3ad"); // shown before any prediction exists

// thr = the vessel's decision threshold (default 0.5). Yellow sits exactly AT the threshold,
// so "orange/red" always means "the model would flag this vessel".
export function riskColor(p, out = new THREE.Color(), thr = 0.5) {
  if (p === null || p === undefined || Number.isNaN(p)) return out.copy(NEUTRAL);
  const t = THREE.MathUtils.clamp(p, 0, 1);
  const th = THREE.MathUtils.clamp(thr, 0.05, 0.95);
  return t < th
    ? out.copy(LOW).lerp(MID, t / th)
    : out.copy(MID).lerp(HIGH, (t - th) / (1 - th));
}

/* ------------------------------------------------------------------ *
 * 2. Vessel paths: computed on the procedural heart surface (see heartMath.js), so they
 *    always sit exactly on it. Pass your own `vessels` prop (three.js space) to override,
 *    e.g. when you load a real .glb. Set DEBUG_POINTS to see the control points.
 * ------------------------------------------------------------------ */
export const DEFAULT_VESSELS = Object.fromEntries(
  Object.entries(buildVesselPaths()).map(([k, pts]) => [k, pts.map(toThree)])
);
const DEBUG_POINTS = false;

/* ------------------------------------------------------------------ *
 * 3. One coronary vessel = one tube mesh with its own material
 * ------------------------------------------------------------------ */
function Vessel({ name, points, prob, thr, selected, onSelect }) {
  const matRef = useRef();
  const [hovered, setHovered] = useState(false);

  // Geometry is rebuilt only if the control points change, never when prob changes.
  const geometry = useMemo(() => {
    const curve = new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)));
    return new THREE.TubeGeometry(curve, 64, 0.045, 12, false);
  }, [points]);

  // Colour we are fading TOWARDS (recomputed when the probability changes).
  const target = useMemo(() => riskColor(prob, new THREE.Color(), thr), [prob, thr]);
  // Colour the material STARTS with (stable object, so React never resets it mid-fade).
  const initial = useRef(riskColor(prob, new THREE.Color(), thr)).current;

  useFrame((_, dt) => {
    const m = matRef.current;
    if (!m) return;
    m.color.lerp(target, 1 - Math.exp(-dt * 8)); // smooth colour transition
    m.emissive.copy(m.color);
    const glow = selected ? 0.65 : hovered ? 0.35 : 0.08;
    m.emissiveIntensity = THREE.MathUtils.lerp(m.emissiveIntensity, glow, 1 - Math.exp(-dt * 10));
  });

  const labelPos = points[Math.floor(points.length / 2)];
  const pct = prob === null || prob === undefined ? "–" : `${Math.round(prob * 100)}%`;

  return (
    <group>
      <mesh
        geometry={geometry}
        onClick={(e) => { e.stopPropagation(); onSelect(name); }}
        onPointerOver={(e) => { e.stopPropagation(); setHovered(true); document.body.style.cursor = "pointer"; }}
        onPointerOut={() => { setHovered(false); document.body.style.cursor = "auto"; }}
      >
        <meshStandardMaterial ref={matRef} color={initial} roughness={0.45} metalness={0.05} />
      </mesh>

      {/* Floating label; pointerEvents none so it never blocks clicks on the vessel */}
      {/* Inline styles on purpose: they survive print-to-PDF and cannot be overridden by page CSS.
          distanceFactor 3.5 = label stays small; the translate keeps it off the vessel itself. */}
      <Html position={labelPos} center distanceFactor={3.5} style={{ pointerEvents: "none" }}>
        <div style={{
          transform: "translate(34px, -18px)", background: "#1d2733", color: "#fff",
          font: "600 13px system-ui, sans-serif", padding: "3px 8px", borderRadius: 4, whiteSpace: "nowrap",
          outline: selected ? "2px solid #fff" : "none", WebkitPrintColorAdjust: "exact", printColorAdjust: "exact",
        }}>
          {name} {pct}
        </div>
      </Html>

      {DEBUG_POINTS && points.map((p, i) => (
        <mesh key={i} position={p}><sphereGeometry args={[0.05, 8, 8]} /><meshBasicMaterial color="#00f" /></mesh>
      ))}
    </group>
  );
}

/* ------------------------------------------------------------------ *
 * 4. Heart shell: translucent .glb if you pass modelUrl, else a simple ellipsoid
 * ------------------------------------------------------------------ */
function GltfShell({ url }) {
  const { scene } = useGLTF(url);
  const clone = useMemo(() => {
    const c = scene.clone(true);
    c.traverse((o) => {
      if (!o.isMesh) return;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      o.material = mats.map((m) => {
        const mm = m.clone();
        mm.transparent = true; mm.opacity = 0.35; mm.depthWrite = false;
        return mm;
      });
      if (o.material.length === 1) o.material = o.material[0];
      o.raycast = () => null; // clicks pass through the shell, so vessels stay selectable
    });
    return c;
  }, [scene]);
  return <primitive object={clone} />;
}

function ProceduralHeart() {
  const geometry = useMemo(() => {
    // Start from a sphere and push every vertex out to the heart surface along its own direction.
    let g = new THREE.SphereGeometry(1, 96, 64);
    g.deleteAttribute("uv"); g.deleteAttribute("normal");
    g = mergeVertices(g);                                 // weld the seam so shading is smooth
    const pos = g.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const dx = pos.getX(i), dy = pos.getY(i), dz = pos.getZ(i);      // unit direction (formula space)
      const r = radiusAlong(dx, dy, dz);
      const [x, y, z] = toThree([dx * r, dy * r, dz * r]);             // formula -> three space
      pos.setXYZ(i, x, y, z);
    }
    // toThree swaps two axes (a mirror), which flips triangle winding. Flip it back.
    const idx = g.index.array;
    for (let i = 0; i < idx.length; i += 3) { const t = idx[i + 1]; idx[i + 1] = idx[i + 2]; idx[i + 2] = t; }
    g.computeVertexNormals();
    return g;
  }, []);
  const tubes = useMemo(
    () => Object.entries(GREAT_VESSELS).map(([k, v]) => ({
      k, color: v.color,
      geo: new THREE.TubeGeometry(new THREE.CatmullRomCurve3(v.points.map((p) => new THREE.Vector3(...toThree(p)))), 40, v.radius, 16, false),
    })), []);
  return (
    <group>
      <mesh geometry={geometry} raycast={() => null}>
        <meshStandardMaterial color="#c4685d" roughness={0.65} metalness={0.02} transparent opacity={0.88} />
      </mesh>
      {tubes.map((t) => (
        <mesh key={t.k} geometry={t.geo} raycast={() => null}>
          <meshStandardMaterial color={t.color} roughness={0.7} />
        </mesh>
      ))}
    </group>
  );
}

function HeartShell({ modelUrl }) {
  return modelUrl ? <GltfShell url={modelUrl} /> : <ProceduralHeart />;
}

/* ------------------------------------------------------------------ *
 * 5. Public component
 *    risk       : { LAD: 0.82, LCX: 0.31, RCA: 0.55 }  (0..1, missing = grey)
 *    thresholds : { LAD: 0.55, LCX: 0.38, RCA: 0.40 }  (colour midpoint; default 0.5)
 *    selected : "LAD" | "LCX" | "RCA" | null
 *    onSelect : (name | null) => void   (click vessel = name, click empty space = null)
 * ------------------------------------------------------------------ */
export default function HeartViewer({
  risk = {},
  thresholds = {},
  selected = null,
  onSelect = () => {},
  modelUrl = null,
  vessels = DEFAULT_VESSELS,
  height = 520,
}) {
  return (
    <div style={{ width: "100%", height }}>
      <Canvas
        dpr={[1, 1.5]}                                   // cap pixel ratio: smooth without a GPU
        camera={{ position: [0, 0.2, 5.2], fov: 40 }}
        gl={{ antialias: true, powerPreference: "low-power" }}
        onPointerMissed={() => onSelect(null)}
      >
        <ambientLight intensity={0.75} />
        <directionalLight position={[3, 4, 5]} intensity={1.1} />

        {/* Tilt the whole heart (shell + vessels together) so the apex points to the patient's left,
            like a real heart in the chest. Vessels are children, so they stay aligned. */}
        <group rotation={[0, 0, modelUrl ? 0 : 0.4]}>
          <Suspense fallback={null}>
            <HeartShell modelUrl={modelUrl} />
          </Suspense>

          {Object.entries(vessels).map(([name, points]) => (
            <Vessel
              key={name}
              name={name}
              points={points}
              prob={risk[name]}
              thr={thresholds[name] ?? 0.5}
              selected={selected === name}
              onSelect={onSelect}
            />
          ))}
        </group>

        {/* rotate = drag, zoom = scroll/pinch; panning off keeps the heart centred */}
        <OrbitControls enablePan={false} minDistance={2.2} maxDistance={7} />
      </Canvas>
    </div>
  );
}

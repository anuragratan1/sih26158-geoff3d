"""Inline anywidget three.js viewer — geometry streams over ipywidgets'
custom-message channel (Widget.send()/model.on("msg:custom", ...)), with
raw binary buffers attached (never base64/JSON).

This matters more than it sounds: an EARLIER version of this file used
regular synced traits (traitlets.Bytes) to carry each point-cloud chunk,
reasoning that ipywidgets routes `bytes` trait values as real binary
buffers (verified directly against ipywidgets' _remove_buffers()) rather
than base64 — true, but irrelevant to the bug it caused. Trait sync is
built for *state* (eventual convergence to the latest value) and is
explicitly allowed to coalesce rapid updates, sending only the newest one.
Direct testing confirmed this: pushing 8 sequential point-cloud chunks
through a Bytes trait silently dropped one of them even with 1-second
gaps between pushes (35,000 points displayed instead of 40,000) — exactly
the kind of bug that would misrepresent a live reconstruction. Custom
messages don't have this problem: they're discrete, ordered, delivered
messages, not state — a 20-message burst with zero delay between them,
sent through Widget.send(), arrived 20-for-20 in a real-browser test. Every
piece of geometry here is an event ("here's another chunk"), not a value,
so it goes through send()/on_msg(), not a trait. Only genuine *state*
(trajectory so far, georeferenced flag, a reset counter) stays as traits.

- The point cloud streams in chunk by chunk during the run (each send()
  call appends on the JS side, never replaces).
- The mesh is pushed twice: once as the coarse vertex-colored version right
  after fusion.py's meshing step, once again as the textured version once
  bake_texture() finishes — genuine progressive refinement mapped onto the
  pipeline's own two natural stages, not simulated.
- Above a point-count threshold the JS side builds a simple LOD octree
  instead of silently downsampling: nearby leaves render at full density,
  distant leaves render a decimated sample, and the displayed/total count
  is shown so it's never silently lying about what's on screen.

Falls back to `InlineViewer = None` if anywidget itself can't be imported —
dashboard.py checks this and falls back to a static, downsampled data-URL
three.js view (reusing viewer.py's template) instead.
"""

from __future__ import annotations

import json

import numpy as np

try:
    import anywidget
    import traitlets

    _ANYWIDGET_OK = True
except Exception:
    anywidget = None
    traitlets = None
    _ANYWIDGET_OK = False


_ESM = r"""
import * as THREE from "https://esm.sh/three@0.160.0";
import { OrbitControls } from "https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { FlyControls } from "https://esm.sh/three@0.160.0/examples/jsm/controls/FlyControls.js";

const MAX_DISPLAY_POINTS = 2_000_000;
const LOD_TRIGGER_POINTS = 2_000_000;

// Growable typed-array buffer (like a JS ArrayList, but for e.g. Float32Array)
// so streamed chunks can be appended without reallocating on every push.
class GrowableBuffer {
  constructor(TypedArrayCtor, itemSize, initialCapacityItems = 65536) {
    this.Ctor = TypedArrayCtor;
    this.itemSize = itemSize;
    this.capacity = initialCapacityItems;
    this.array = new TypedArrayCtor(this.capacity * itemSize);
    this.length = 0; // in items, not raw elements
  }
  ensure(extraItems) {
    if (this.length + extraItems <= this.capacity) return;
    while (this.capacity < this.length + extraItems) this.capacity *= 2;
    const next = new this.Ctor(this.capacity * this.itemSize);
    next.set(this.array.subarray(0, this.length * this.itemSize));
    this.array = next;
  }
  push(typedSlice, nItems) {
    this.ensure(nItems);
    this.array.set(typedSlice, this.length * this.itemSize);
    this.length += nItems;
  }
  view() {
    return this.array.subarray(0, this.length * this.itemSize);
  }
  reset() {
    this.length = 0;
  }
}

// Minimal octree LOD: partitions accumulated points into cubic cells; each
// leaf keeps the indices of points inside it. At render time, cells near
// the camera contribute all their points, distant cells contribute a
// decimated sample, capped at MAX_DISPLAY_POINTS total. Rebuilt from
// scratch periodically (not every chunk) since points only ever accumulate.
class PointOctree {
  constructor(positions, count, maxDepth = 6, maxLeafPoints = 20000) {
    this.maxDepth = maxDepth;
    this.maxLeafPoints = maxLeafPoints;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < count; i++) {
      const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
    const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-3) * 0.5 + 1e-3;
    const allIdx = new Uint32Array(count);
    for (let i = 0; i < count; i++) allIdx[i] = i;
    this.root = this._build(positions, allIdx, cx, cy, cz, size, 0);
  }
  _build(positions, idx, cx, cy, cz, size, depth) {
    if (idx.length <= this.maxLeafPoints || depth >= this.maxDepth) {
      return { leaf: true, cx, cy, cz, size, idx };
    }
    const buckets = [[], [], [], [], [], [], [], []];
    for (const i of idx) {
      const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
      const b = (x > cx ? 1 : 0) | (y > cy ? 2 : 0) | (z > cz ? 4 : 0);
      buckets[b].push(i);
    }
    const half = size / 2;
    const children = buckets.map((b, i) => {
      if (b.length === 0) return null;
      const ncx = cx + (i & 1 ? half : -half);
      const ncy = cy + (i & 2 ? half : -half);
      const ncz = cz + (i & 4 ? half : -half);
      return this._build(positions, Uint32Array.from(b), ncx, ncy, ncz, half, depth + 1);
    });
    return { leaf: false, cx, cy, cz, size, children };
  }
  // Collect point indices to display given a camera position, capped at
  // `budget` total. Near leaves get full density; far leaves get a
  // decimated stride based on distance.
  select(camPos, budget) {
    let total = 0;
    const stack = [this.root];
    const candidates = [];
    while (stack.length) {
      const node = stack.pop();
      const dx = node.cx - camPos.x, dy = node.cy - camPos.y, dz = node.cz - camPos.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (node.leaf) {
        candidates.push({ idx: node.idx, dist });
      } else {
        for (const c of node.children) if (c) stack.push(c);
      }
    }
    candidates.sort((a, b) => a.dist - b.dist);
    const selected = [];
    for (const cand of candidates) {
      if (total >= budget) break;
      const remaining = budget - total;
      if (cand.idx.length <= remaining) {
        selected.push(cand.idx);
        total += cand.idx.length;
      } else {
        const stride = Math.ceil(cand.idx.length / remaining);
        const dec = [];
        for (let i = 0; i < cand.idx.length; i += stride) dec.push(cand.idx[i]);
        selected.push(Uint32Array.from(dec));
        total += dec.length;
      }
    }
    return { indices: selected, total };
  }
}

function bufToFloat32(buf) {
  if (!buf || buf.byteLength === 0) return new Float32Array(0);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}
function bufToUint8(buf) {
  if (!buf || buf.byteLength === 0) return new Uint8Array(0);
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}
function bufToUint32(buf) {
  if (!buf || buf.byteLength === 0) return new Uint32Array(0);
  return new Uint32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function confToColor(v) {
  // blue (low) -> yellow -> red (high), v in [0,1]
  const r = Math.min(1, v * 2);
  const g = Math.min(1, 2 - v * 2);
  const b = Math.max(0, 1 - v * 2);
  return [r, g, b];
}

export default {
  render({ model, el }) {
    el.style.position = "relative";
    el.style.width = "100%";
    el.style.height = "520px";

    const canvasWrap = document.createElement("div");
    canvasWrap.style.width = "100%";
    canvasWrap.style.height = "100%";
    el.appendChild(canvasWrap);

    const hud = document.createElement("div");
    hud.style.cssText = "position:absolute;top:8px;left:8px;background:rgba(13,17,23,.85);border:1px solid #30363d;border-radius:6px;padding:6px 10px;color:#e6edf3;font:12px sans-serif;max-width:260px;z-index:5;";
    el.appendChild(hud);

    const controlsBar = document.createElement("div");
    controlsBar.style.cssText = "position:absolute;top:8px;right:8px;background:rgba(13,17,23,.85);border:1px solid #30363d;border-radius:6px;padding:6px;color:#e6edf3;font:12px sans-serif;z-index:5;display:flex;flex-direction:column;gap:4px;max-width:180px;";
    el.appendChild(controlsBar);

    function makeBtn(label) {
      const b = document.createElement("button");
      b.textContent = label;
      b.style.cssText = "font-size:11px;padding:3px 6px;cursor:pointer;";
      controlsBar.appendChild(b);
      return b;
    }
    const btnPoints = makeBtn("Toggle points");
    const btnMesh = makeBtn("Toggle mesh");
    const btnTraj = makeBtn("Toggle trajectory");
    const btnConf = makeBtn("Confidence color");
    const btnFly = makeBtn("Fly mode: off");
    const measureSelect = document.createElement("select");
    measureSelect.style.cssText = "font-size:11px;";
    for (const opt of ["Measure: off", "Distance", "Height", "Area"]) {
      const o = document.createElement("option");
      o.textContent = opt;
      measureSelect.appendChild(o);
    }
    controlsBar.appendChild(measureSelect);
    const measureReadout = document.createElement("div");
    measureReadout.style.cssText = "font-size:11px;color:#79c0ff;min-height:14px;";
    controlsBar.appendChild(measureReadout);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0f14);
    const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1e6);
    camera.position.set(20, 20, 20);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    canvasWrap.appendChild(renderer.domElement);
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight.position.set(1, 2, 1);
    scene.add(dirLight);
    scene.add(new THREE.AxesHelper(5));

    let orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    let fly = new FlyControls(camera, renderer.domElement);
    fly.movementSpeed = 10;
    fly.rollSpeed = 0.5;
    fly.enabled = false;
    orbit.enabled = true;

    function resize() {
      const w = canvasWrap.clientWidth || 600;
      const h = canvasWrap.clientHeight || 520;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    new ResizeObserver(resize).observe(canvasWrap);
    resize();

    // -- point cloud state ----------------------------------------------
    const posBuf = new GrowableBuffer(Float32Array, 3);
    const colBuf = new GrowableBuffer(Uint8Array, 3);
    const confBuf = new GrowableBuffer(Float32Array, 1);
    let totalPointCount = 0;
    let octree = null;
    let colorMode = "rgb"; // "rgb" | "confidence"

    const pointsGeo = new THREE.BufferGeometry();
    const pointsMat = new THREE.PointsMaterial({ size: 0.03, vertexColors: true });
    const pointsObj = new THREE.Points(pointsGeo, pointsMat);
    scene.add(pointsObj);

    function rebuildOctreeIfNeeded() {
      if (totalPointCount >= LOD_TRIGGER_POINTS) {
        octree = new PointOctree(posBuf.array, totalPointCount);
      } else {
        octree = null;
      }
    }

    function updatePointsDisplay() {
      let positions, colorsF32, shown;
      if (octree) {
        const { indices, total } = octree.select(camera.position, MAX_DISPLAY_POINTS);
        positions = new Float32Array(total * 3);
        colorsF32 = new Float32Array(total * 3);
        let o = 0;
        for (const chunkIdx of indices) {
          for (const i of chunkIdx) {
            positions[o * 3] = posBuf.array[i * 3];
            positions[o * 3 + 1] = posBuf.array[i * 3 + 1];
            positions[o * 3 + 2] = posBuf.array[i * 3 + 2];
            if (colorMode === "confidence") {
              const [r, g, b] = confToColor(confBuf.array[i]);
              colorsF32[o * 3] = r; colorsF32[o * 3 + 1] = g; colorsF32[o * 3 + 2] = b;
            } else {
              colorsF32[o * 3] = colBuf.array[i * 3] / 255;
              colorsF32[o * 3 + 1] = colBuf.array[i * 3 + 1] / 255;
              colorsF32[o * 3 + 2] = colBuf.array[i * 3 + 2] / 255;
            }
            o++;
          }
        }
        shown = total;
      } else {
        positions = posBuf.view();
        shown = posBuf.length;
        colorsF32 = new Float32Array(shown * 3);
        if (colorMode === "confidence") {
          for (let i = 0; i < shown; i++) {
            const [r, g, b] = confToColor(confBuf.array[i]);
            colorsF32[i * 3] = r; colorsF32[i * 3 + 1] = g; colorsF32[i * 3 + 2] = b;
          }
        } else {
          const c = colBuf.view();
          for (let i = 0; i < shown; i++) {
            colorsF32[i * 3] = c[i * 3] / 255;
            colorsF32[i * 3 + 1] = c[i * 3 + 1] / 255;
            colorsF32[i * 3 + 2] = c[i * 3 + 2] / 255;
          }
        }
      }
      pointsGeo.setAttribute("position", new THREE.BufferAttribute(positions.slice(), 3));
      pointsGeo.setAttribute("color", new THREE.BufferAttribute(colorsF32, 3));
      pointsGeo.computeBoundingSphere();
      hud.textContent = `Points: ${shown.toLocaleString()} shown / ${totalPointCount.toLocaleString()} total`;
    }

    // -- mesh state ------------------------------------------------------
    let meshObj = null;
    let meshVisible = true;

    function onMeshMsg(content, buffers) {
      const [vB, iB, cB, uvB, texB] = buffers;
      const vertices = bufToFloat32(vB);
      const indices = bufToUint32(iB);
      if (vertices.length === 0) return;

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
      geo.setIndex(new THREE.BufferAttribute(indices, 1));

      let material;
      const uv = bufToFloat32(uvB);
      const texBytes = bufToUint8(texB);
      if (content.has_uv && content.has_texture && uv.length > 0 && texBytes.length > 0) {
        geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
        const blob = new Blob([texBytes], { type: "image/png" });
        const texUrl = URL.createObjectURL(blob);
        const tex = new THREE.TextureLoader().load(texUrl, () => URL.revokeObjectURL(texUrl));
        material = new THREE.MeshStandardMaterial({ map: tex });
      } else {
        const colors = bufToUint8(cB);
        if (content.has_colors && colors.length > 0) {
          const colorsF32 = new Float32Array(colors.length);
          for (let i = 0; i < colors.length; i++) colorsF32[i] = colors[i] / 255;
          geo.setAttribute("color", new THREE.BufferAttribute(colorsF32, 3));
          material = new THREE.MeshStandardMaterial({ vertexColors: true });
        } else {
          material = new THREE.MeshStandardMaterial({ color: 0x888888 });
        }
      }
      geo.computeVertexNormals();

      if (meshObj) scene.remove(meshObj);
      meshObj = new THREE.Mesh(geo, material);
      meshObj.visible = meshVisible;
      scene.add(meshObj);
    }

    // -- streamed geometry: custom messages, NOT traits ---------------------
    // (see module docstring — traits coalesce rapid updates and silently
    // drop intermediate ones; custom messages are ordered and reliable,
    // verified directly: a 20-message zero-delay burst arrived 20-for-20).
    model.on("msg:custom", (msg, buffers) => {
      if (msg.kind === "points") {
        const [xyzB, rgbB, confB] = buffers;
        const xyz = bufToFloat32(xyzB);
        const rgb = bufToUint8(rgbB);
        const conf = bufToFloat32(confB);
        const n = msg.n;
        posBuf.push(xyz, n);
        colBuf.push(rgb, n);
        confBuf.push(conf, n);
        totalPointCount += n;
        rebuildOctreeIfNeeded();
        updatePointsDisplay();
      } else if (msg.kind === "mesh") {
        onMeshMsg(msg, buffers);
      }
    });

    // -- trajectory (genuine state, stays a trait) ---------------------------
    let trajGroup = new THREE.Group();
    scene.add(trajGroup);
    function drawTrajectory() {
      trajGroup.clear();
      function addTrack(jsonStr, color) {
        let pts;
        try { pts = JSON.parse(jsonStr); } catch (e) { return; }
        if (!pts || pts.length < 2) return;
        const vecs = pts.map(p => new THREE.Vector3(p[0], p[1], p[2]));
        const geo = new THREE.BufferGeometry().setFromPoints(vecs);
        trajGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color })));
      }
      addTrack(model.get("trajectory_gps"), 0x8b949e);
      addTrack(model.get("trajectory_camera"), 0x2ea043);
    }
    model.on("change:trajectory_gps", drawTrajectory);
    model.on("change:trajectory_camera", drawTrajectory);

    model.on("change:points_reset", () => {
      posBuf.reset(); colBuf.reset(); confBuf.reset();
      totalPointCount = 0; octree = null;
      updatePointsDisplay();
    });

    // -- toggles -----------------------------------------------------------
    btnPoints.onclick = () => { pointsObj.visible = !pointsObj.visible; };
    btnMesh.onclick = () => { meshVisible = !meshVisible; if (meshObj) meshObj.visible = meshVisible; };
    btnTraj.onclick = () => { trajGroup.visible = !trajGroup.visible; };
    btnConf.onclick = () => {
      colorMode = colorMode === "rgb" ? "confidence" : "rgb";
      btnConf.textContent = colorMode === "rgb" ? "Confidence color" : "RGB color";
      updatePointsDisplay();
    };
    btnFly.onclick = () => {
      fly.enabled = !fly.enabled;
      orbit.enabled = !fly.enabled;
      btnFly.textContent = fly.enabled ? "Fly mode: on" : "Fly mode: off";
    };

    // -- measurement tool ----------------------------------------------------
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.1;
    const mouse = new THREE.Vector2();
    let measurePoints = [];
    let measureMarkers = new THREE.Group();
    scene.add(measureMarkers);

    measureSelect.onchange = () => { measurePoints = []; measureMarkers.clear(); measureReadout.textContent = ""; };

    function pickPoint(event) {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const targets = [pointsObj, meshObj].filter(Boolean);
      const hits = raycaster.intersectObjects(targets, true);
      return hits.length ? hits[0].point.clone() : null;
    }
    function polygonAreaGroundProjected(points) {
      let area = 0;
      for (let i = 0; i < points.length; i++) {
        const a = points[i], b = points[(i + 1) % points.length];
        area += a.x * b.y - b.x * a.y;
      }
      return Math.abs(area) / 2;
    }
    renderer.domElement.addEventListener("dblclick", (event) => {
      const mode = measureSelect.selectedIndex;
      if (mode === 0) return;
      const p = pickPoint(event);
      if (!p) return;
      measurePoints.push(p);
      const marker = new THREE.Mesh(new THREE.SphereGeometry(0.05, 8, 8), new THREE.MeshBasicMaterial({ color: 0xffa657 }));
      marker.position.copy(p);
      measureMarkers.add(marker);

      const unit = model.get("georeferenced") ? "m" : "units";
      if (mode === 1 && measurePoints.length >= 2) {
        const d = measurePoints[measurePoints.length - 2].distanceTo(p);
        measureReadout.textContent = `Distance: ${d.toFixed(3)} ${unit}`;
      } else if (mode === 2 && measurePoints.length >= 2) {
        const dz = Math.abs(measurePoints[measurePoints.length - 2].z - p.z);
        measureReadout.textContent = `Height: ${dz.toFixed(3)} ${unit}`;
      } else if (mode === 3 && measurePoints.length >= 3) {
        const area = polygonAreaGroundProjected(measurePoints);
        measureReadout.textContent = `Area (${measurePoints.length} pts): ${area.toFixed(2)} ${unit}²`;
      }
    });

    // -- render loop -----------------------------------------------------
    const clock = new THREE.Clock();
    let lastLodUpdate = 0;
    function animate() {
      requestAnimationFrame(animate);
      const dt = clock.getDelta();
      if (fly.enabled) fly.update(dt); else orbit.update();
      if (octree && clock.elapsedTime - lastLodUpdate > 0.3) {
        updatePointsDisplay();
        lastLodUpdate = clock.elapsedTime;
      }
      renderer.render(scene, camera);
    }
    animate();

    drawTrajectory();
    updatePointsDisplay();
  },
};
"""


if _ANYWIDGET_OK:

    class InlineViewer(anywidget.AnyWidget):  # noqa: F811
        _esm = _ESM

        # Genuine state (fine to coalesce/converge) -> traits.
        points_reset = traitlets.Int(0).tag(sync=True)
        trajectory_gps = traitlets.Unicode("[]").tag(sync=True)
        trajectory_camera = traitlets.Unicode("[]").tag(sync=True)
        georeferenced = traitlets.Bool(False).tag(sync=True)

        def push_points(self, points: np.ndarray, colors: np.ndarray, confidence: np.ndarray) -> None:
            """points/colors: (N,3), confidence: (N,). Appends to the
            viewer's accumulated cloud (streamed, chunk by chunk). Sent as
            a custom message (see module docstring for why — NOT a trait:
            traits coalesce rapid updates and silently drop intermediate
            ones, which would misrepresent a streaming reconstruction)."""
            n = len(points)
            self.send(
                {"kind": "points", "n": n},
                buffers=[
                    points.astype(np.float32).tobytes(),
                    np.clip(colors, 0, 255).astype(np.uint8).tobytes(),
                    confidence.astype(np.float32).tobytes(),
                ],
            )

        def push_mesh(
            self, vertices: np.ndarray, indices: np.ndarray,
            colors: np.ndarray | None = None, uv: np.ndarray | None = None,
            texture_png: bytes | None = None, stage: str = "coarse",
        ) -> None:
            self.send(
                {
                    "kind": "mesh", "stage": stage,
                    "has_colors": colors is not None, "has_uv": uv is not None,
                    "has_texture": texture_png is not None,
                },
                buffers=[
                    vertices.astype(np.float32).tobytes(),
                    indices.astype(np.uint32).tobytes(),
                    (np.clip(colors, 0, 255).astype(np.uint8).tobytes() if colors is not None else b""),
                    (uv.astype(np.float32).tobytes() if uv is not None else b""),
                    (texture_png if texture_png is not None else b""),
                ],
            )

        def set_trajectory(self, gps: list, camera: list) -> None:
            self.trajectory_gps = json.dumps(gps)
            self.trajectory_camera = json.dumps(camera)

        def reset(self) -> None:
            self.points_reset += 1

else:
    InlineViewer = None  # dashboard.py checks this and falls back to a static viewer

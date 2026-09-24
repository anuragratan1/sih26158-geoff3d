"""Writes a single self-contained viewer.html: three.js (loaded from a CDN,
the only external dependency — everything else is embedded inline as
base64 so the file works standalone regardless of how it's downloaded off
Kaggle, without relying on relative fetch() of sibling files, which file://
URLs frequently block).

Embeds: the point cloud (positions/colors/confidence as base64-encoded
typed arrays — far more compact than a JSON number array), the textured/
vertex-colored mesh as a base64 GLB blob (parsed via GLTFLoader.parse from
an ArrayBuffer, no network fetch needed), and both trajectory tracks as
small local-ENU-meter coordinate arrays (not the GeoJSON/KML lon-lat
versions, which are for GIS tools, not directly usable in the same 3D
scene as the point cloud/mesh without reprojecting).
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .events import EventBus
from .fusion import FusedPointCloud


@dataclass
class ExportStatus:
    name: str
    path: Path | None
    ok: bool
    skipped_reason: str | None = None
    size_bytes: int = 0
    timing_s: float = 0.0


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def write_viewer_html(
    out_path: Path,
    bus: EventBus,
    *,
    cloud: FusedPointCloud | None,
    mesh_glb_path: Path | None,
    gps_track_enu: list[tuple[float, float, float]] | None,
    camera_track_enu: list[tuple[float, float, float]] | None,
    georeferenced: bool,
    max_points: int = 200_000,
) -> ExportStatus:
    t0 = time.time()

    points_b64 = colors_b64 = conf_b64 = ""
    n_points = 0
    if cloud is not None and len(cloud.points):
        pts, cols, conf = cloud.points, cloud.colors, cloud.confidence
        if len(pts) > max_points:
            idx = np.random.default_rng(0).choice(len(pts), size=max_points, replace=False)
            pts, cols, conf = pts[idx], cols[idx], conf[idx]
        n_points = len(pts)
        points_b64 = _b64(pts.astype(np.float32))
        colors_b64 = _b64(np.clip(cols, 0, 255).astype(np.uint8))
        conf_max = float(conf.max()) if len(conf) else 1.0
        conf_norm = (conf / conf_max) if conf_max > 0 else conf
        conf_b64 = _b64(conf_norm.astype(np.float32))

    mesh_b64 = ""
    if mesh_glb_path is not None and Path(mesh_glb_path).exists():
        mesh_b64 = base64.b64encode(Path(mesh_glb_path).read_bytes()).decode("ascii")

    gps_json = json.dumps([list(p) for p in (gps_track_enu or [])])
    cam_json = json.dumps([list(p) for p in (camera_track_enu or [])])

    html = _TEMPLATE.format(
        n_points=n_points, points_b64=points_b64, colors_b64=colors_b64, conf_b64=conf_b64,
        mesh_b64=mesh_b64, gps_json=gps_json, cam_json=cam_json,
        georeferenced_label="Georeferenced (metric, CRS-aligned)" if georeferenced else "APPROXIMATE SCALE — NOT GEOREFERENCED",
        georeferenced_color="#2ea043" if georeferenced else "#d29922",
    )
    out_path.write_text(html)
    size = out_path.stat().st_size
    bus.log(f"Wrote {out_path.name}: {n_points} points embedded, mesh={'yes' if mesh_b64 else 'no'} ({size / 1e6:.1f} MB)")
    return ExportStatus(name="viewer.html", path=out_path, ok=True, size_bytes=size, timing_s=time.time() - t0)


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SIH26158 3D Viewer</title>
<style>
  html, body {{ margin:0; height:100%; background:#0b0f14; color:#e6edf3; font-family: -apple-system, sans-serif; overflow:hidden; }}
  #canvas-wrap {{ position:absolute; inset:0; }}
  #hud {{ position:absolute; top:10px; left:10px; z-index:10; background:rgba(13,17,23,0.85);
          border:1px solid #30363d; border-radius:8px; padding:10px 14px; max-width:340px; }}
  #hud h1 {{ font-size:14px; margin:0 0 8px 0; }}
  #hud label {{ display:block; font-size:12px; margin:6px 0 2px; }}
  #hud select, #hud button {{ font-size:12px; padding:3px 6px; margin-right:4px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px; background:{georeferenced_color}; font-size:11px; margin-bottom:6px; }}
  #measure-readout {{ font-size:12px; margin-top:6px; min-height:16px; color:#79c0ff; }}
</style>
</head>
<body>
<div id="canvas-wrap"></div>
<div id="hud">
  <h1>SIH26158 — 3D Model Viewer</h1>
  <div class="badge">{georeferenced_label}</div>
  <div>Points embedded: {n_points}</div>
  <label>Point color</label>
  <select id="color-mode">
    <option value="rgb">RGB</option>
    <option value="confidence">Confidence</option>
  </select>
  <label>Overlays</label>
  <button id="toggle-points">Toggle points</button>
  <button id="toggle-mesh">Toggle mesh</button>
  <button id="toggle-trajectory">Toggle trajectory</button>
  <label>Measure</label>
  <select id="measure-mode">
    <option value="none">Off</option>
    <option value="distance">Distance</option>
    <option value="height">Height</option>
    <option value="area">Area (ground-projected)</option>
  </select>
  <button id="measure-reset">Reset</button>
  <div id="measure-readout"></div>
</div>

<script type="module">
// three.js dropped its legacy non-module examples/js/ UMD builds for
// OrbitControls/GLTFLoader some releases back (they 404 on cdnjs/jsdelivr
// now) — only the ES-module examples/jsm/ versions still exist. Loading
// the old paths left THREE.OrbitControls undefined, which threw and
// silently killed the rest of this script before the point cloud was ever
// added to the scene: the HUD panel (plain HTML/CSS) still rendered, but
// the canvas stayed black. type="module" + these imports both fixes that
// and still works from a file:// URL — the restriction on importing local
// files as modules from file:// doesn't apply to importing a remote
// https:// module.
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.module.js";
import {{ OrbitControls }} from "https://cdn.jsdelivr.net/npm/three@0.158.0/examples/jsm/controls/OrbitControls.js";
import {{ GLTFLoader }} from "https://cdn.jsdelivr.net/npm/three@0.158.0/examples/jsm/loaders/GLTFLoader.js";

const POINTS_B64 = "{points_b64}";
const COLORS_B64 = "{colors_b64}";
const CONF_B64 = "{conf_b64}";
const MESH_B64 = "{mesh_b64}";
const GPS_TRACK = {gps_json};
const CAM_TRACK = {cam_json};

function b64ToArrayBuffer(b64) {{
  if (!b64) return new ArrayBuffer(0);
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf.buffer;
}}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f14);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 100000);
camera.position.set(20, 20, 20);
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById("canvas-wrap").appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(1, 2, 1);
scene.add(dirLight);
scene.add(new THREE.AxesHelper(5));

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// -- point cloud --------------------------------------------------------
let pointCloud = null;
let rgbColors = null, confColors = null;
{{
  const posBuf = new Float32Array(b64ToArrayBuffer(POINTS_B64));
  if (posBuf.length > 0) {{
    const colorBytes = new Uint8Array(b64ToArrayBuffer(COLORS_B64));
    const confFloats = new Float32Array(b64ToArrayBuffer(CONF_B64));
    rgbColors = new Float32Array(colorBytes.length);
    for (let i = 0; i < colorBytes.length; i++) rgbColors[i] = colorBytes[i] / 255.0;
    confColors = new Float32Array(confFloats.length * 3);
    for (let i = 0; i < confFloats.length; i++) {{
      const v = confFloats[i];
      // simple blue->yellow->red ramp
      confColors[i*3]   = Math.min(1, v * 2);
      confColors[i*3+1] = Math.min(1, 2 - v * 2);
      confColors[i*3+2] = Math.max(0, 1 - v * 2);
    }}
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(posBuf, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(rgbColors.slice(), 3));
    const mat = new THREE.PointsMaterial({{ size: 0.03, vertexColors: true }});
    pointCloud = new THREE.Points(geo, mat);
    scene.add(pointCloud);

    const box = new THREE.Box3().setFromObject(pointCloud);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    camera.position.copy(center).add(new THREE.Vector3(size * 0.6, size * 0.6, size * 0.6));
    controls.target.copy(center);
    controls.update();
  }}
}}

document.getElementById("color-mode").addEventListener("change", (e) => {{
  if (!pointCloud) return;
  const arr = e.target.value === "confidence" ? confColors : rgbColors;
  pointCloud.geometry.setAttribute("color", new THREE.BufferAttribute(arr.slice(), 3));
}});

// -- mesh ------------------------------------------------------------------
let meshObject = null;
if (MESH_B64) {{
  const loader = new GLTFLoader();
  loader.parse(b64ToArrayBuffer(MESH_B64), "", (gltf) => {{
    meshObject = gltf.scene;
    scene.add(meshObject);
  }}, (err) => console.error("GLB parse failed", err));
}}

// -- trajectory overlay ------------------------------------------------
let trajectoryGroup = new THREE.Group();
function addTrack(points, color) {{
  if (!points || points.length < 2) return;
  const pts = points.map(p => new THREE.Vector3(p[0], p[1], p[2]));
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({{ color, linewidth: 2 }});
  trajectoryGroup.add(new THREE.Line(geo, mat));
}}
addTrack(GPS_TRACK, 0x8b949e);
addTrack(CAM_TRACK, 0x2ea043);
scene.add(trajectoryGroup);

// -- toggles -------------------------------------------------------------
document.getElementById("toggle-points").onclick = () => {{ if (pointCloud) pointCloud.visible = !pointCloud.visible; }};
document.getElementById("toggle-mesh").onclick = () => {{ if (meshObject) meshObject.visible = !meshObject.visible; }};
document.getElementById("toggle-trajectory").onclick = () => {{ trajectoryGroup.visible = !trajectoryGroup.visible; }};

// -- measurement tool ---------------------------------------------------
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.1;
const mouse = new THREE.Vector2();
let measureMode = "none";
let measurePoints = [];
let measureMarkers = new THREE.Group();
scene.add(measureMarkers);

document.getElementById("measure-mode").addEventListener("change", (e) => {{
  measureMode = e.target.value;
  resetMeasure();
}});
document.getElementById("measure-reset").onclick = resetMeasure;

function resetMeasure() {{
  measurePoints = [];
  measureMarkers.clear();
  document.getElementById("measure-readout").textContent = "";
}}

function pickPoint(event) {{
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const targets = [pointCloud, meshObject].filter(Boolean);
  const hits = raycaster.intersectObjects(targets, true);
  return hits.length ? hits[0].point.clone() : null;
}}

function polygonAreaGroundProjected(points) {{
  // Shoelace formula on the XY (ground) projection — the standard
  // convention for aerial/survey area measurement (ground footprint,
  // not slanted 3D surface area).
  let area = 0;
  for (let i = 0; i < points.length; i++) {{
    const a = points[i], b = points[(i + 1) % points.length];
    area += a.x * b.y - b.x * a.y;
  }}
  return Math.abs(area) / 2;
}}

function addMarker(p) {{
  const geo = new THREE.SphereGeometry(0.05, 8, 8);
  const mat = new THREE.MeshBasicMaterial({{ color: 0xffa657 }});
  const marker = new THREE.Mesh(geo, mat);
  marker.position.copy(p);
  measureMarkers.add(marker);
}}

renderer.domElement.addEventListener("dblclick", (event) => {{
  if (measureMode === "none") return;
  const p = pickPoint(event);
  if (!p) return;
  measurePoints.push(p);
  addMarker(p);

  const readout = document.getElementById("measure-readout");
  if (measureMode === "distance" && measurePoints.length >= 2) {{
    const d = measurePoints[measurePoints.length - 2].distanceTo(p);
    readout.textContent = `Distance: ${{d.toFixed(3)}} m`;
  }} else if (measureMode === "height" && measurePoints.length >= 2) {{
    const dz = Math.abs(measurePoints[measurePoints.length - 2].z - p.z);
    readout.textContent = `Height: ${{dz.toFixed(3)}} m`;
  }} else if (measureMode === "area" && measurePoints.length >= 3) {{
    const area = polygonAreaGroundProjected(measurePoints);
    readout.textContent = `Area (${{measurePoints.length}} pts, ground-projected): ${{area.toFixed(2)}} m²`;
  }}
}});

// -- render loop -----------------------------------------------------------
window.addEventListener("resize", () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""

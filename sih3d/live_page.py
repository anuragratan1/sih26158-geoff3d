"""A small FastAPI server, on its own port (exposed via its own cloudflared
tunnel — see fullscreen_server.start_cloudflared_tunnel, shared code), for
a dedicated live-reconstruction page separate from the in-notebook
dashboard: input video on the left, a live three.js 3D view on the right.

Streams over WebSocket as binary frames (JSON header message immediately
followed by a binary payload message — not base64, not JSON-encoded
numbers), pushed PER KEYFRAME rather than per backbone-inference chunk, so
the client can reveal points frame by frame in sync with the video even
though a whole chunk's geometry becomes available at once computationally.

Every message is kept in an in-memory history list and broadcast to
connected clients. A client that connects late, or reconnects after a
hiccup, gets the full history replayed first, then switches to live tail —
so "replay mode" isn't a special server feature, it's just the client
re-walking the same history array it would build up live, which is also
what makes replay work fully even if live streaming hiccupped.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from pathlib import Path

import numpy as np
# Imported at MODULE level deliberately, not inside _build_app(): this file
# has `from __future__ import annotations`, so every type annotation
# (including `websocket: WebSocket` below) is a lazily-evaluated string.
# FastAPI resolves those strings via typing.get_type_hints(), which looks
# the name up in the function's __globals__ — the ws_endpoint closure's
# enclosing MODULE globals, not _build_app()'s local scope. A local import
# of WebSocket here left it out of those globals entirely, so FastAPI
# silently failed to recognize the parameter as an injected WebSocket
# connection and instead treated "websocket" as a required query string
# parameter — every connection closed with code 1008 ("Field required"),
# which uvicorn's websocket layer in turn reports to the client as a
# generic HTTP 403. Confirmed via TestClient (no network involved) after
# an otherwise-identical inline reproduction without `from __future__
# import annotations` worked fine — isolating the cause to exactly this.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from .events import EventBus
from .fullscreen_server import start_cloudflared_tunnel


class LivePageServer:
    def __init__(self, video_path: Path, bus: EventBus):
        self.video_path = video_path
        self.bus = bus
        self.history: list[tuple[dict, bytes]] = []
        self._clients: set = set()
        self._loop: "asyncio.AbstractEventLoop | None" = None
        self.port: int | None = None
        self._app = self._build_app()

    # -- server -------------------------------------------------------------

    def _build_app(self):
        app = FastAPI()

        @app.get("/")
        async def index():
            return HTMLResponse(_HTML_TEMPLATE)

        @app.get("/video")
        async def video():
            # Starlette's FileResponse supports Range requests natively,
            # which is what lets the client scrub the video during replay.
            return FileResponse(str(self.video_path))

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._clients.add(websocket)
            try:
                for header, payload in list(self.history):
                    await websocket.send_json(header)
                    if payload:
                        await websocket.send_bytes(payload)
                while True:
                    await websocket.receive_text()  # keep-alive ping; content ignored
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                self._clients.discard(websocket)

        return app

    def start(self, port: int | None = None) -> int:
        import uvicorn

        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port = s.getsockname()[1]
        self.port = port

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            config = uvicorn.Config(self._app, host="0.0.0.0", port=port, log_level="warning")
            server = uvicorn.Server(config)
            loop.run_until_complete(server.serve())

        threading.Thread(target=_run, daemon=True, name="sih3d-live-page").start()

        deadline = time.time() + 5.0
        while self._loop is None and time.time() < deadline:
            time.sleep(0.02)

        self.bus.log(f"Live reconstruction page server listening on 127.0.0.1:{port}")
        return port

    def _broadcast(self, header: dict, payload: bytes) -> None:
        self.history.append((header, payload))
        if self._loop is None:
            return

        async def _send() -> None:
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send_json(header)
                    if payload:
                        await ws.send_bytes(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

        try:
            asyncio.run_coroutine_threadsafe(_send(), self._loop)
        except Exception:
            pass

    # -- push API, called from pipeline.py -----------------------------------

    def push_points(
        self, points: np.ndarray, colors: np.ndarray, confidence: np.ndarray, masked: np.ndarray,
        frame_index: int, timestamp_s: float,
    ) -> None:
        n = len(points)
        header = {"kind": "points", "n": n, "frame_index": frame_index, "timestamp_s": timestamp_s, "seq": len(self.history)}
        payload = (
            points.astype(np.float32).tobytes()
            + np.clip(colors, 0, 255).astype(np.uint8).tobytes()
            + confidence.astype(np.float32).tobytes()
            + masked.astype(np.uint8).tobytes()
        )
        self._broadcast(header, payload)

    def push_camera_pose(
        self, frame_index: int, timestamp_s: float,
        position: np.ndarray, forward: np.ndarray, up: np.ndarray,
    ) -> None:
        header = {
            "kind": "camera_pose", "frame_index": frame_index, "timestamp_s": timestamp_s,
            "position": [float(v) for v in position], "forward": [float(v) for v in forward], "up": [float(v) for v in up],
        }
        self._broadcast(header, b"")

    def push_mesh(
        self, vertices: np.ndarray, indices: np.ndarray,
        colors: np.ndarray | None = None, uv: np.ndarray | None = None,
        texture_png: bytes | None = None, stage: str = "coarse",
    ) -> None:
        parts = [
            vertices.astype(np.float32).tobytes(),
            indices.astype(np.uint32).tobytes(),
            np.clip(colors, 0, 255).astype(np.uint8).tobytes() if colors is not None else b"",
            uv.astype(np.float32).tobytes() if uv is not None else b"",
            texture_png if texture_png is not None else b"",
        ]
        header = {
            "kind": "mesh", "stage": stage, "lengths": [len(p) for p in parts],
            "has_colors": colors is not None, "has_uv": uv is not None, "has_texture": texture_png is not None,
        }
        self._broadcast(header, b"".join(parts))


def start_live_page(video_path: Path, bus: EventBus) -> tuple[LivePageServer | None, str | None]:
    """Starts the server + its own cloudflared tunnel. Never raises —
    returns (None, None) on any failure, logged; the caller (bootstrap.py)
    just doesn't show the "Open live reconstruction page" link and the
    dashboard's inline 3D tab remains the fallback."""
    try:
        server = LivePageServer(video_path, bus)
        port = server.start()
    except Exception as e:
        bus.log(f"Live reconstruction page server failed to start ({e}) — falling back to the dashboard's inline 3D tab", level="warn")
        return None, None

    url = start_cloudflared_tunnel(port, bus, label="live reconstruction page")
    return server, url


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SIH26158 Live Reconstruction</title>
<style>
  html, body { margin:0; height:100%; background:#0b0f14; color:#e6edf3; font-family:-apple-system,sans-serif; overflow:hidden; }
  #layout { display:flex; height:100vh; }
  #left { width:38%; display:flex; flex-direction:column; border-right:1px solid #30363d; }
  #right { flex:1; position:relative; }
  video { width:100%; background:#000; }
  #controls { padding:8px; font-size:12px; display:flex; flex-direction:column; gap:6px; overflow-y:auto; }
  #controls label { display:flex; align-items:center; gap:6px; }
  #timeline { width:100%; }
  button, select { font-size:11px; padding:3px 6px; }
  #hud { position:absolute; top:8px; left:8px; background:rgba(13,17,23,.85); border:1px solid #30363d;
         border-radius:6px; padding:6px 10px; font-size:12px; z-index:5; }
  #status { position:absolute; bottom:8px; left:8px; font-size:11px; color:#8b949e; z-index:5; }
</style>
</head>
<body>
<div id="layout">
  <div id="left">
    <video id="vid" src="/video" muted playsinline preload="auto"></video>
    <div id="controls">
      <div><b>SIH26158 — Live Reconstruction</b></div>
      <label><input type="checkbox" id="chkAutoFollow" checked> Auto-follow camera</label>
      <label><input type="checkbox" id="chkMask"> Show masked-out objects (red)</label>
      <label><input type="checkbox" id="chkConf"> Confidence coloring</label>
      <label><input type="checkbox" id="chkTraj" checked> Trajectory</label>
      <label><input type="checkbox" id="chkMesh" checked> Textured mesh (once available)</label>
      <div>
        <label><input type="radio" name="mode" value="live" id="modeLive" checked> Live</label>
        <label><input type="radio" name="mode" value="replay" id="modeReplay"> Replay</label>
      </div>
      <input type="range" id="timeline" min="0" max="0" value="0" step="1" disabled>
      <div id="timelineLabel">frame -</div>
    </div>
  </div>
  <div id="right">
    <div id="hud">Points: 0 shown / 0 total</div>
    <div id="status">connecting...</div>
  </div>
</div>

<script type="module">
import * as THREE from "https://esm.sh/three@0.160.0";
import { OrbitControls } from "https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const vid = document.getElementById("vid");
// `preload="auto"` alone isn't reliably honored by every browser in every
// context (confirmed directly: readyState stayed 0/HAVE_NOTHING on first
// load here until an explicit .load() was called, after which it went
// straight to 4/HAVE_ENOUGH_DATA) — force it deterministically.
vid.load();
const hud = document.getElementById("hud");
const statusEl = document.getElementById("status");
const timeline = document.getElementById("timeline");
const timelineLabel = document.getElementById("timelineLabel");
const chkAutoFollow = document.getElementById("chkAutoFollow");
const chkMask = document.getElementById("chkMask");
const chkConf = document.getElementById("chkConf");
const chkTraj = document.getElementById("chkTraj");
const chkMesh = document.getElementById("chkMesh");
const modeLive = document.getElementById("modeLive");
const modeReplay = document.getElementById("modeReplay");

const rightEl = document.getElementById("right");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f14);
const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1e6);
camera.position.set(20, 20, 20);
const renderer = new THREE.WebGLRenderer({ antialias: true });
rightEl.appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(1, 2, 1);
scene.add(dirLight);
scene.add(new THREE.AxesHelper(5));

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;

function resize() {
  const w = rightEl.clientWidth, h = rightEl.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
new ResizeObserver(resize).observe(rightEl);
resize();

// -- point history: every "points" message ever received, in arrival order.
// Both LIVE and REPLAY draw from this same array — replay just re-walks it
// up to a scrubbed index, so it works even if the live WS connection
// hiccupped (the server replays full history to a (re)connecting client).
const pointHistory = []; // {n, frame_index, timestamp_s, xyz, rgb, conf, masked}
let totalPointsSeen = 0;
let revealIndex = 0; // how many history entries are currently shown

const posArr = [];
const colArr = [];
const pointsGeo = new THREE.BufferGeometry();
const pointsMat = new THREE.PointsMaterial({ size: 0.03, vertexColors: true });
const pointsObj = new THREE.Points(pointsGeo, pointsMat);
scene.add(pointsObj);

function rebuildPointsUpTo(idx) {
  let positions = [], colors = [];
  for (let i = 0; i < idx; i++) {
    const e = pointHistory[i];
    for (let j = 0; j < e.n; j++) {
      positions.push(e.xyz[j*3], e.xyz[j*3+1], e.xyz[j*3+2]);
      if (chkMask.checked && e.masked[j]) {
        colors.push(1, 0, 0);
      } else if (chkConf.checked) {
        const v = e.conf[j];
        colors.push(Math.min(1, v*2), Math.min(1, 2-v*2), Math.max(0, 1-v*2));
      } else {
        colors.push(e.rgb[j*3]/255, e.rgb[j*3+1]/255, e.rgb[j*3+2]/255);
      }
    }
  }
  pointsGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  pointsGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  pointsGeo.computeBoundingSphere();
  hud.textContent = `Points: ${(positions.length/3).toLocaleString()} shown / ${totalPointsSeen.toLocaleString()} total`;
}

// -- camera frustum + trajectory ------------------------------------------
const frustum = new THREE.CameraHelper(new THREE.PerspectiveCamera(50, 1.3, 0.1, 2));
scene.add(frustum);
let trajGroup = new THREE.Group();
scene.add(trajGroup);
const trajPoints = [];

function updateFrustum(pos, forward, up) {
  frustum.camera.position.set(pos[0], pos[1], pos[2]);
  const target = [pos[0]+forward[0], pos[1]+forward[1], pos[2]+forward[2]];
  frustum.camera.up.set(up[0], up[1], up[2]);
  frustum.camera.lookAt(target[0], target[1], target[2]);
  frustum.camera.updateProjectionMatrix();
  frustum.update();

  trajPoints.push(new THREE.Vector3(pos[0], pos[1], pos[2]));
  trajGroup.clear();
  if (trajPoints.length >= 2) {
    const geo = new THREE.BufferGeometry().setFromPoints(trajPoints);
    trajGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0x2ea043 })));
  }

  if (chkAutoFollow.checked) {
    orbit.target.set(pos[0], pos[1], pos[2]);
    camera.position.set(pos[0] - forward[0]*10, pos[1] - forward[1]*10, pos[2] - forward[2]*10 + 5);
  }
}

// -- mesh -------------------------------------------------------------------
let meshObj = null;
function applyMesh(header, buf) {
  const [vLen, iLen, cLen, uvLen, texLen] = header.lengths;
  let off = 0;
  const vertices = new Float32Array(buf, off, vLen/4); off += vLen;
  const indices = new Uint32Array(buf, off, iLen/4); off += iLen;
  const colorsBuf = header.has_colors ? new Uint8Array(buf, off, cLen) : null; off += cLen;
  const uvBuf = header.has_uv ? new Float32Array(buf, off, uvLen/4) : null; off += uvLen;
  const texBuf = header.has_texture ? new Uint8Array(buf, off, texLen) : null;

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  let material;
  if (header.has_uv && header.has_texture) {
    geo.setAttribute("uv", new THREE.BufferAttribute(uvBuf, 2));
    const blob = new Blob([texBuf], { type: "image/png" });
    const url = URL.createObjectURL(blob);
    const tex = new THREE.TextureLoader().load(url, () => URL.revokeObjectURL(url));
    material = new THREE.MeshStandardMaterial({ map: tex });
  } else if (header.has_colors) {
    const colorsF32 = new Float32Array(colorsBuf.length);
    for (let i = 0; i < colorsBuf.length; i++) colorsF32[i] = colorsBuf[i] / 255;
    geo.setAttribute("color", new THREE.BufferAttribute(colorsF32, 3));
    material = new THREE.MeshStandardMaterial({ vertexColors: true });
  } else {
    material = new THREE.MeshStandardMaterial({ color: 0x888888 });
  }
  geo.computeVertexNormals();
  if (meshObj) scene.remove(meshObj);
  meshObj = new THREE.Mesh(geo, material);
  meshObj.visible = chkMesh.checked;
  scene.add(meshObj);
}

// -- WebSocket: JSON header message immediately followed by a binary message
let pendingHeader = null;
function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { statusEl.textContent = "connected"; };
  ws.onclose = () => { statusEl.textContent = "disconnected — retrying in 2s"; setTimeout(connect, 2000); };
  ws.onerror = () => { statusEl.textContent = "connection error"; };
  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const header = JSON.parse(event.data);
      if (header.kind === "camera_pose") {
        if (modeLive.checked) updateFrustum(header.position, header.forward, header.up);
        return; // no binary payload for camera_pose
      }
      pendingHeader = header;
    } else {
      const header = pendingHeader;
      pendingHeader = null;
      if (!header) return;
      if (header.kind === "points") {
        const buf = event.data;
        let off = 0;
        const xyz = new Float32Array(buf, off, header.n*3); off += header.n*3*4;
        const rgb = new Uint8Array(buf, off, header.n*3); off += header.n*3;
        const conf = new Float32Array(buf, off, header.n); off += header.n*4;
        const masked = new Uint8Array(buf, off, header.n);
        pointHistory.push({ n: header.n, frame_index: header.frame_index, timestamp_s: header.timestamp_s, xyz, rgb, conf, masked });
        totalPointsSeen += header.n;
        timeline.max = pointHistory.length - 1;
        if (modeLive.checked) {
          revealIndex = pointHistory.length;
          rebuildPointsUpTo(revealIndex);
          if (chkAutoFollow.checked || vid.paused) vid.currentTime = header.timestamp_s;
        }
      } else if (header.kind === "mesh") {
        if (chkMesh.checked || true) applyMesh(header, event.data);
      }
    }
  };
}
connect();

// -- toggles ------------------------------------------------------------
chkMask.onchange = chkConf.onchange = () => rebuildPointsUpTo(revealIndex);
chkTraj.onchange = () => { trajGroup.visible = chkTraj.checked; };
chkMesh.onchange = () => { if (meshObj) meshObj.visible = chkMesh.checked; };

// -- replay mode ----------------------------------------------------------
modeReplay.onchange = () => { timeline.disabled = !modeReplay.checked; vid.pause(); };
modeLive.onchange = () => { timeline.disabled = true; };
timeline.oninput = () => {
  const idx = parseInt(timeline.value, 10);
  revealIndex = idx + 1;
  rebuildPointsUpTo(revealIndex);
  if (pointHistory[idx]) {
    vid.currentTime = pointHistory[idx].timestamp_s;
    timelineLabel.textContent = `frame ${pointHistory[idx].frame_index} (t=${pointHistory[idx].timestamp_s.toFixed(2)}s)`;
  }
};

// -- render loop -----------------------------------------------------------
function animate() {
  requestAnimationFrame(animate);
  orbit.update();
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
"""

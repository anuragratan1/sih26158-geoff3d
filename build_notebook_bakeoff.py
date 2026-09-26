#!/usr/bin/env python3
"""Generates backbone_bakeoff.ipynb — a standalone, throwaway comparison
notebook. Does NOT modify sih26158_geoff3d.ipynb or any sih3d/ pipeline code;
it only *reuses* sih3d.decode/sih3d.keyframes/sih3d.io_detect (unchanged) to
get an identical keyframe set, then runs three backbones (MapAnything,
SLAM3R, VGGT-Omega) each in its own subprocess for a side-by-side comparison.
No meshing/texturing/DSM. Budget: under 10 minutes wall time total on 2xT4.
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
OUT_PATH = ROOT / "backbone_bakeoff.ipynb"


def code(src: str):
    return nbf.v4.new_code_cell(src.strip("\n"))


def md(src: str):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


TITLE_MD = """
# Backbone Bake-off: MapAnything vs SLAM3R vs VGGT-Omega

Throwaway comparison notebook. No meshing/texturing/DSM, does not touch the
main sih26158_geoff3d pipeline. Budget: under 10 minutes wall time total on
2xT4, installs included.
"""

SETUP_CELL = r'''
# ============================== SETUP =========================================
import os, sys, subprocess, time, json
from pathlib import Path

_t_setup0 = time.time()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

CODE_DIR = Path("/kaggle/working/sih26158-geoff3d")
REPO_URL = "https://github.com/anuragratan1/sih26158-geoff3d.git"
if CODE_DIR.exists():
    subprocess.run(["git", "-C", str(CODE_DIR), "pull", "--ff-only"], capture_output=True, text=True, timeout=60)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(CODE_DIR)], check=True, capture_output=True, text=True, timeout=180)
sys.path.insert(0, str(CODE_DIR))

BAKEOFF_DIR = Path("/kaggle/working/bakeoff")
SCRIPTS_DIR = BAKEOFF_DIR / "scripts"
KEYFRAMES_DIR = Path("/kaggle/working/keyframes")
RESULTS_DIR = BAKEOFF_DIR / "results"
for d in (BAKEOFF_DIR, SCRIPTS_DIR, KEYFRAMES_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

install_log = {}

def _pip(args, label, timeout=180):
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + args, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    ok = r.returncode == 0
    print(f"  [{'OK' if ok else 'FAILED'} {elapsed:.0f}s] {label}")
    if not ok:
        print("   ", r.stderr[-400:])
    install_log[label] = {"ok": ok, "elapsed_s": elapsed}
    return ok

print("== Installs ==")
_t_installs0 = time.time()

# MapAnything (Apache) -- plain pip install, matches main pipeline's own choice.
_pip(["git+https://github.com/facebookresearch/map-anything.git"], "mapanything")

# SLAM3R -- --no-deps then only the light deps actually needed for the
# offline recon.py path (skip pycuda/viser/gradio/tensorboard/pyglet: demo/
# viz-only, and per explicit instruction skip xformers + custom RoPE kernel
# compile).
_pip(["--no-deps", "git+https://github.com/PKU-VCL-3DV/SLAM3R.git"], "slam3r (no-deps)")
_pip(["roma", "einops", "opencv-python-headless", "scipy", "huggingface-hub[torch]>=0.22"], "slam3r light deps")

elapsed_installs_so_far = time.time() - _t_installs0
print(f"Installs so far: {elapsed_installs_so_far:.0f}s")

RUN_VGGT_OMEGA = os.environ.get("HF_TOKEN") not in (None, "")
if RUN_VGGT_OMEGA and elapsed_installs_so_far > 150:
    print(f"Install budget (180s) at risk after mapanything+slam3r ({elapsed_installs_so_far:.0f}s) -- dropping VGGT-Omega per budget rule.")
    RUN_VGGT_OMEGA = False
elif not RUN_VGGT_OMEGA:
    print("HF_TOKEN not set -- VGGT-Omega will be SKIPPED (gated checkpoint).")

if RUN_VGGT_OMEGA:
    _pip(["git+https://github.com/facebookresearch/vggt-omega.git"], "vggt-omega")

print(f"\\nTotal install time: {time.time() - _t_installs0:.0f}s")
print(f"Setup cell total: {time.time() - _t_setup0:.0f}s")
'''

KEYFRAMES_CELL = r'''
# ============================== KEYFRAMES (shared by all backbones) =========
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(CODE_DIR))
_t0 = time.time()

from sih3d.decode import FrameDecoder
from sih3d.events import EventBus
from sih3d.io_detect import find_videos
from sih3d.keyframes import select_keyframes

bus = EventBus()
videos = find_videos(Path("/kaggle/input"))
if not videos:
    raise RuntimeError("No video found under /kaggle/input")
video = videos[0]
print(f"Video: {video.path.name} | {video.width}x{video.height} | {video.fps:.1f} fps | {video.duration_s:.1f}s")

decoder = FrameDecoder(video.path, bus, device="cuda:0")

# Full resolution (no scale_width) -- the bake-off explicitly wants
# full-resolution JPGs, unlike the main pipeline's own working-resolution
# decode used only for its own sharpness pre-pass.
QUICK_SECONDS = 90.0
KEYFRAME_SAMPLE_FPS = 4.0
MAX_KEYFRAMES = 60

raw_indices, raw_ts, raw_frames = [], [], []
for idx, t, frame in decoder.iter_frames(start_s=0.0, end_s=min(QUICK_SECONDS, video.duration_s), target_fps=KEYFRAME_SAMPLE_FPS):
    raw_indices.append(idx)
    raw_ts.append(t)
    raw_frames.append(frame)
    if len(raw_frames) >= MAX_KEYFRAMES * 4:
        break

print(f"Decoded {len(raw_frames)} candidate frames in {time.time() - _t0:.1f}s")

selection = select_keyframes(raw_indices, raw_ts, raw_frames, None, bus, device="cuda:0")
accepted = selection.accepted
if len(accepted) > MAX_KEYFRAMES:
    step = len(accepted) / MAX_KEYFRAMES
    accepted = [accepted[int(i * step)] for i in range(MAX_KEYFRAMES)]

manifest = []
for cand in accepted:
    local_i = raw_indices.index(cand.frame_index)
    img = raw_frames[local_i]
    fn = f"frame_{cand.frame_index:06d}.jpg"
    Image.fromarray(img).save(KEYFRAMES_DIR / fn, quality=95)
    manifest.append({"frame_index": cand.frame_index, "timestamp_s": cand.timestamp_s, "file": fn,
                      "width": int(img.shape[1]), "height": int(img.shape[0])})

(KEYFRAMES_DIR / "manifest.json").write_text(json.dumps({
    "video": str(video.path), "video_width": video.width, "video_height": video.height,
    "video_fps": video.fps, "keyframes": manifest,
}, indent=2))

print(f"Saved {len(manifest)} full-resolution keyframes to {KEYFRAMES_DIR} in {time.time() - _t0:.1f}s total")
'''

WRITE_SCRIPTS_CELL = r'''
# ============================== WRITE BACKBONE RUNNER SCRIPTS ================
# Each runner is a standalone subprocess: loads the shared keyframes, runs
# its own model, writes a result JSON/NPZ, and NEVER raises past its own
# top-level try/except -- a bug or OOM becomes a recorded status, not a
# crash that takes down the orchestrator.
from pathlib import Path

MAPANYTHING_RUNNER = r"""
import json, os, sys, time, traceback
import numpy as np
import torch

status = {"backbone": "mapanything", "status": "error", "runtime_s": None, "peak_vram_mb": None, "frames": 0}
out_dir = sys.argv[1]
keyframes_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
try:
    torch.cuda.reset_peak_memory_stats()
    manifest = json.load(open(os.path.join(keyframes_dir, "manifest.json")))
    image_paths = [os.path.join(keyframes_dir, k["file"]) for k in manifest["keyframes"]]

    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    device = "cuda"
    model = MapAnything.from_pretrained("facebook/map-anything-apache").to(device).eval()
    views = load_images(image_paths)  # official preprocessing: 518 long side, aspect preserved (fixed_mapping)
    with torch.no_grad():
        outputs = model.infer(views, memory_efficient_inference=True, minibatch_size=1, use_amp=True, amp_dtype="fp16")

    points = np.stack([o["pts3d"][0].float().cpu().numpy() for o in outputs], axis=0)
    conf = np.stack([o["conf"][0].float().cpu().numpy() for o in outputs], axis=0)
    poses = np.stack([o["camera_poses"][0].float().cpu().numpy() for o in outputs], axis=0)
    intrinsics = None
    if "intrinsics" in outputs[0]:
        intrinsics = np.stack([o["intrinsics"][0].float().cpu().numpy() for o in outputs], axis=0)

    np.savez(os.path.join(out_dir, "result.npz"), points=points, conf=conf, poses=poses,
             intrinsics=intrinsics if intrinsics is not None else np.zeros((len(outputs), 3, 3)))
    status["status"] = "ok"
    status["frames"] = len(outputs)
    status["has_intrinsics"] = intrinsics is not None
except Exception as e:
    status["error"] = f"{type(e).__name__}: {e}"
    status["traceback"] = traceback.format_exc()
finally:
    status["runtime_s"] = time.time() - t0
    try:
        status["peak_vram_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    json.dump(status, open(os.path.join(out_dir, "status.json"), "w"), indent=2)
    print(json.dumps(status, indent=2))
"""

SLAM3R_RUNNER = r"""
import json, os, sys, time, traceback
import numpy as np
import torch

status = {"backbone": "slam3r", "status": "error", "runtime_s": None, "peak_vram_mb": None, "frames": 0,
          "has_cameras": False}
out_dir = sys.argv[1]
keyframes_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
try:
    torch.cuda.reset_peak_memory_stats()
    manifest = json.load(open(os.path.join(keyframes_dir, "manifest.json")))

    from slam3r.models import Image2PointsModel, Local2WorldModel
    from slam3r.datasets.wild_seq import Seq_Data
    from slam3r.pipeline.recon_offline_pipeline import scene_recon_pipeline_offline

    device = "cuda"
    i2p_model = Image2PointsModel.from_pretrained("siyan824/slam3r_i2p").to(device).eval()
    l2w_model = Local2WorldModel.from_pretrained("siyan824/slam3r_l2w").to(device).eval()

    dataset = Seq_Data(img_dir=keyframes_dir, img_size=224, silent=False, sample_freq=1,
                        start_idx=0, num_views=-1, start_freq=1, to_tensor=True)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)

    class _Args:
        keyframe_stride = 3
        win_r = 5
        max_num_register = 10
        num_scene_frame = 10
        initial_winsize = 5
        conf_thres_l2w = 12
        conf_thres_i2p = 1.5
        num_points_save = 1_000_000
        retrieve_freq = 1
        update_buffer_intv = 1
        buffer_size = 100
        buffer_strategy = "reservoir"
        keyframe_adapt_min = 1
        keyframe_adapt_max = 20
        keyframe_adapt_stride = 1
        norm_input = False
        save_frequency = 3
        save_each_frame = False
        save_preds = True
        save_for_eval = False
        save_online = False
        perframe = 1

    scene_recon_pipeline_offline(i2p_model, l2w_model, dataset, _Args(), out_dir)

    preds_dir = os.path.join(out_dir, "preds")
    pcds = np.load(os.path.join(preds_dir, "registered_pcds.npy"))
    confs = np.load(os.path.join(preds_dir, "registered_confs.npy"))
    status["status"] = "ok"
    status["frames"] = int(pcds.shape[0])
    status["has_cameras"] = False  # SLAM3R produces no camera poses/intrinsics
except Exception as e:
    status["error"] = f"{type(e).__name__}: {e}"
    status["traceback"] = traceback.format_exc()
finally:
    status["runtime_s"] = time.time() - t0
    try:
        status["peak_vram_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    json.dump(status, open(os.path.join(out_dir, "status.json"), "w"), indent=2)
    print(json.dumps(status, indent=2))
"""

VGGT_OMEGA_RUNNER = r"""
import json, os, sys, time, traceback
import numpy as np
import torch

status = {"backbone": "vggt_omega", "status": "error", "runtime_s": None, "peak_vram_mb": None, "frames": 0}
out_dir = sys.argv[1]
keyframes_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
try:
    torch.cuda.reset_peak_memory_stats()
    manifest = json.load(open(os.path.join(keyframes_dir, "manifest.json")))
    image_paths = [os.path.join(keyframes_dir, k["file"]) for k in manifest["keyframes"]]

    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id="facebook/VGGT-Omega", filename="vggt_omega_1b_512.pt",
                                 token=os.environ.get("HF_TOKEN"))

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = "cuda"
    model = VGGTOmega().to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.half()  # T4 has no bf16

    images = load_and_preprocess_images(image_paths, image_resolution=512, mode="max_size").to(device).half()

    with torch.inference_mode():
        predictions = model(images)

    extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    depth = predictions["depth"].float().cpu().numpy()
    depth_conf = predictions["depth_conf"].float().cpu().numpy()

    np.savez(os.path.join(out_dir, "result.npz"), depth=depth, depth_conf=depth_conf,
             extrinsics=extrinsics.float().cpu().numpy(), intrinsics=intrinsics.float().cpu().numpy())
    status["status"] = "ok"
    status["frames"] = int(depth.shape[1]) if depth.ndim > 1 else int(depth.shape[0])
except Exception as e:
    status["error"] = f"{type(e).__name__}: {e}"
    status["traceback"] = traceback.format_exc()
finally:
    status["runtime_s"] = time.time() - t0
    try:
        status["peak_vram_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    json.dump(status, open(os.path.join(out_dir, "status.json"), "w"), indent=2)
    print(json.dumps(status, indent=2))
"""

(SCRIPTS_DIR / "mapanything_runner.py").write_text(MAPANYTHING_RUNNER)
(SCRIPTS_DIR / "slam3r_runner.py").write_text(SLAM3R_RUNNER)
(SCRIPTS_DIR / "vggtomega_runner.py").write_text(VGGT_OMEGA_RUNNER)
print("Runner scripts written:", [p.name for p in SCRIPTS_DIR.glob("*.py")])
'''

RUN_BACKBONES_CELL = r'''
# ============================== RUN BACKBONES (subprocess, parallel where possible)
import json
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT_S = 150
_t0 = time.time()

def launch(script_name, out_subdir, gpu_id, env_extra=None):
    out_dir = RESULTS_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(**__import__("os").environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if env_extra:
        env.update(env_extra)
    log_path = out_dir / "log.txt"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPTS_DIR / script_name), str(out_dir), str(KEYFRAMES_DIR)],
        stdout=log_f, stderr=subprocess.STDOUT, env=env,
    )
    return proc, out_dir, log_f

def wait_with_timeout(proc, log_f, timeout_s):
    try:
        proc.wait(timeout=timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        timed_out = True
    log_f.close()
    return timed_out

results = {}

# MapAnything on GPU0, SLAM3R on GPU1, launched together.
print("Launching MapAnything (GPU0) + SLAM3R (GPU1) in parallel...")
p_ma, dir_ma, log_ma = launch("mapanything_runner.py", "mapanything", 0)
p_s3, dir_s3, log_s3 = launch("slam3r_runner.py", "slam3r", 1)

t_ma0 = time.time()
timed_out_ma = wait_with_timeout(p_ma, log_ma, TIMEOUT_S)
elapsed_ma = time.time() - t_ma0
t_s3_remaining = max(1, TIMEOUT_S - (time.time() - t_ma0))
timed_out_s3 = wait_with_timeout(p_s3, log_s3, t_s3_remaining)

for name, out_dir, timed_out, elapsed_wall in [
    ("mapanything", dir_ma, timed_out_ma, elapsed_ma),
    ("slam3r", dir_s3, timed_out_s3, None),
]:
    status_path = out_dir / "status.json"
    if timed_out:
        results[name] = {"backbone": name, "status": "timeout", "runtime_s": TIMEOUT_S}
        print(f"{name}: TIMEOUT after {TIMEOUT_S}s")
    elif status_path.exists():
        results[name] = json.loads(status_path.read_text())
        print(f"{name}: {results[name]['status']} in {results[name].get('runtime_s', 0):.1f}s")
    else:
        log_text = (out_dir / "log.txt").read_text()[-1000:] if (out_dir / "log.txt").exists() else ""
        results[name] = {"backbone": name, "status": "crash_no_status", "log_tail": log_text}
        print(f"{name}: CRASHED with no status.json -- log tail:\\n{log_text}")

print(f"\\nMapAnything + SLAM3R wall time: {time.time() - _t0:.0f}s")

if RUN_VGGT_OMEGA:
    print("\\nLaunching VGGT-Omega (GPU0)...")
    p_vo, dir_vo, log_vo = launch("vggtomega_runner.py", "vggt_omega", 0, env_extra={"HF_TOKEN": __import__("os").environ.get("HF_TOKEN", "")})
    timed_out_vo = wait_with_timeout(p_vo, log_vo, TIMEOUT_S)
    status_path = dir_vo / "status.json"
    if timed_out_vo:
        results["vggt_omega"] = {"backbone": "vggt_omega", "status": "timeout", "runtime_s": TIMEOUT_S}
        print(f"vggt_omega: TIMEOUT after {TIMEOUT_S}s")
    elif status_path.exists():
        results["vggt_omega"] = json.loads(status_path.read_text())
        print(f"vggt_omega: {results['vggt_omega']['status']} in {results['vggt_omega'].get('runtime_s', 0):.1f}s")
    else:
        log_text = (dir_vo / "log.txt").read_text()[-1000:] if (dir_vo / "log.txt").exists() else ""
        results["vggt_omega"] = {"backbone": "vggt_omega", "status": "crash_no_status", "log_tail": log_text}
        print(f"vggt_omega: CRASHED with no status.json -- log tail:\\n{log_text}")
else:
    results["vggt_omega"] = {"backbone": "vggt_omega", "status": "SKIPPED", "reason": "HF_TOKEN not set or install budget exceeded"}
    print("vggt_omega: SKIPPED")

(RESULTS_DIR / "all_status.json").write_text(json.dumps(results, indent=2))
print(f"\\nTotal backbone runtime: {time.time() - _t0:.0f}s")
'''

METRICS_CELL = r'''
# ============================== METRICS + RENDERS ============================
import json
import time
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(CODE_DIR))
from sih3d.align import solve_sim3

_t0 = time.time()
manifest = json.load(open(KEYFRAMES_DIR / "manifest.json"))["keyframes"]
frame_indices = [k["frame_index"] for k in manifest]
all_status = json.loads((RESULTS_DIR / "all_status.json").read_text())

table_rows = []


def self_warp_error(points, intrinsics, poses):
    """Median/P90 pixel offset reprojecting each pixel's own 3D point
    through its own recorded camera -- validates the (points, camera)
    pair are mutually consistent. points: (N,H,W,3) world frame."""
    errs = []
    for i in range(min(3, len(points))):
        pts = points[i]
        h, w = pts.shape[:2]
        valid = pts[..., 2] != 0
        rows, cols = np.nonzero(valid[..., 0] if valid.ndim == 3 else valid)
        if len(rows) == 0:
            continue
        world = pts[rows, cols]
        cam = np.linalg.inv(poses[i]) @ np.concatenate([world, np.ones((len(world), 1))], axis=1).T
        cam = cam.T[:, :3]
        z = cam[:, 2]
        ok = z > 1e-6
        if not ok.any():
            continue
        fx, fy, cx, cy = intrinsics[i][0, 0], intrinsics[i][1, 1], intrinsics[i][0, 2], intrinsics[i][1, 2]
        u = fx * cam[ok, 0] / z[ok] + cx
        v = fy * cam[ok, 1] / z[ok] + cy
        off = np.sqrt((u - cols[ok]) ** 2 + (v - rows[ok]) ** 2)
        errs.append(off)
    if not errs:
        return None, None
    allv = np.concatenate(errs)
    return float(np.median(allv)), float(np.percentile(allv, 90))


def cross_view_consistency(points, intrinsics, poses):
    """Land-masking SKIPPED for the 1-minute metrics time budget -- this is
    ALL valid-depth pixels, not land-only. Reported honestly as such."""
    abs_d, rel_d = [], []
    n = min(len(points), 10)
    for i in range(n - 1):
        a, b = points[i], points[i + 1]
        h, w = a.shape[:2]
        valid_a = a[..., 2] != 0
        rows, cols = np.nonzero(valid_a)
        if len(rows) == 0:
            continue
        world = a[rows, cols]
        cam_b = (np.linalg.inv(poses[i + 1]) @ np.concatenate([world, np.ones((len(world), 1))], axis=1).T).T[:, :3]
        z_warp = cam_b[:, 2]
        ok = z_warp > 1e-6
        if not ok.any():
            continue
        fx, fy, cx, cy = intrinsics[i + 1][0, 0], intrinsics[i + 1][1, 1], intrinsics[i + 1][0, 2], intrinsics[i + 1][1, 2]
        u = (fx * cam_b[ok, 0] / z_warp[ok] + cx).round().astype(int)
        v = (fy * cam_b[ok, 1] / z_warp[ok] + cy).round().astype(int)
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not inb.any():
            continue
        z_own = b[v[inb], u[inb], 2]
        both = z_own != 0
        if not both.any():
            continue
        diff = np.abs(z_warp[ok][inb][both] - z_own[both])
        abs_d.append(diff)
        rel_d.append(diff / np.abs(z_own[both]))
    if not abs_d:
        return None, None
    rel = np.concatenate(rel_d) * 100
    return float(np.median(rel)), float(np.percentile(rel, 90))


def fit_sim3_to_mapanything(pts_model, pts_ref, grid_n=32):
    """Resamples both per-frame point grids to a common grid_n x grid_n
    (normalized image coords, nearest) and fits Sim(3) model->ref via
    solve_sim3. Returns (scale, median_residual_pct_of_depth) or (None, None)."""
    residuals = []
    scales = []
    for i in range(min(3, len(pts_model), len(pts_ref))):
        hm, wm = pts_model[i].shape[:2]
        hr, wr = pts_ref[i].shape[:2]
        ys = np.linspace(0, 1, grid_n, endpoint=False) + 0.5 / grid_n
        xs = np.linspace(0, 1, grid_n, endpoint=False) + 0.5 / grid_n
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        rm = (gy * hm).astype(int).clip(0, hm - 1)
        cm = (gx * wm).astype(int).clip(0, wm - 1)
        rr = (gy * hr).astype(int).clip(0, hr - 1)
        cr = (gx * wr).astype(int).clip(0, wr - 1)
        pm = pts_model[i][rm, cm].reshape(-1, 3)
        pr = pts_ref[i][rr, cr].reshape(-1, 3)
        valid = (pm[:, 2] != 0) & (pr[:, 2] != 0)
        if valid.sum() < 10:
            continue
        try:
            fit_scale, fit_R, fit_t, _fit_rmse = solve_sim3(pm[valid], pr[valid])
        except Exception:
            continue
        pred = fit_scale * (fit_R @ pm[valid].T).T + fit_t
        res = np.linalg.norm(pred - pr[valid], axis=1)
        depth_ref = np.abs(pr[valid][:, 2])
        depth_ref = np.where(depth_ref > 1e-6, depth_ref, np.nan)
        residuals.append(np.nanmedian(res / depth_ref) * 100)
        scales.append(fit_scale)
    if not residuals:
        return None, None
    return float(np.median(scales)), float(np.median(residuals))


# -- MapAnything (reference) --------------------------------------------------
ma_status = all_status.get("mapanything", {})
ma_points = ma_intr = ma_poses = None
if ma_status.get("status") == "ok":
    d = np.load(RESULTS_DIR / "mapanything" / "result.npz")
    ma_points, ma_conf, ma_poses, ma_intr = d["points"], d["conf"], d["poses"], d["intrinsics"]
    sw_med, sw_p90 = self_warp_error(ma_points, ma_intr, ma_poses)
    cv_med, cv_p90 = cross_view_consistency(ma_points, ma_intr, ma_poses)
    table_rows.append({"backbone": "mapanything", "status": "ok", "runtime_s": ma_status["runtime_s"],
                        "peak_vram_mb": ma_status.get("peak_vram_mb"), "frames": ma_status["frames"],
                        "self_warp_median_px": sw_med, "self_warp_p90_px": sw_p90,
                        "consistency_median_pct": cv_med, "consistency_p90_pct": cv_p90,
                        "sim3_scale_vs_ma": 1.0, "sim3_residual_pct_vs_ma": 0.0})
else:
    table_rows.append({"backbone": "mapanything", "status": ma_status.get("status", "missing"),
                        "error": ma_status.get("error")})

# -- SLAM3R --------------------------------------------------------------------
s3_status = all_status.get("slam3r", {})
if s3_status.get("status") == "ok":
    pcds = np.load(RESULTS_DIR / "slam3r" / "preds" / "registered_pcds.npy")
    row = {"backbone": "slam3r", "status": "ok", "runtime_s": s3_status["runtime_s"],
           "peak_vram_mb": s3_status.get("peak_vram_mb"), "frames": s3_status["frames"],
           "self_warp_median_px": "N/A (no cameras)", "self_warp_p90_px": "N/A (no cameras)",
           "consistency_median_pct": "N/A (no cameras)", "consistency_p90_pct": "N/A (no cameras)"}
    if ma_points is not None:
        scale, resid = fit_sim3_to_mapanything(pcds, ma_points[:len(pcds)])
        row["sim3_scale_vs_ma"] = scale
        row["sim3_residual_pct_vs_ma"] = resid
    table_rows.append(row)
else:
    table_rows.append({"backbone": "slam3r", "status": s3_status.get("status", "missing"),
                        "error": s3_status.get("error")})

# -- VGGT-Omega ------------------------------------------------------------
vo_status = all_status.get("vggt_omega", {})
if vo_status.get("status") == "ok":
    d = np.load(RESULTS_DIR / "vggt_omega" / "result.npz")
    depth, extr, intr_vo = d["depth"], d["extrinsics"], d["intrinsics"]
    # depth (1,N,H,W,1) or (N,H,W,1)-ish depending on vggt_omega's exact
    # output layout -- unproject to camera-space points, squeeze leading dims defensively.
    depth = np.squeeze(depth)
    if depth.ndim == 3:
        n_f, h, w = depth.shape
    else:
        n_f, h, w = depth.shape[0], depth.shape[-2], depth.shape[-1]
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    vo_points = np.zeros((n_f, h, w, 3), dtype=np.float32)
    vo_poses = np.zeros((n_f, 4, 4), dtype=np.float32)
    for i in range(n_f):
        fx, fy, cx, cy = intr_vo[i][0, 0], intr_vo[i][1, 1], intr_vo[i][0, 2], intr_vo[i][1, 2]
        z = depth[i]
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        cam_pts = np.stack([x, y, z], axis=-1)
        pose = np.eye(4)
        pose[:3, :4] = extr[i][:3, :4] if extr[i].shape[0] >= 3 else np.eye(4)[:3, :4]
        pose_c2w = np.linalg.inv(pose) if np.linalg.det(pose[:3, :3]) != 0 else np.eye(4)
        vo_poses[i] = pose_c2w
        world = (pose_c2w[:3, :3] @ cam_pts.reshape(-1, 3).T).T + pose_c2w[:3, 3]
        vo_points[i] = world.reshape(h, w, 3)
    sw_med, sw_p90 = self_warp_error(vo_points, intr_vo, vo_poses)
    cv_med, cv_p90 = cross_view_consistency(vo_points, intr_vo, vo_poses)
    row = {"backbone": "vggt_omega", "status": "ok", "runtime_s": vo_status["runtime_s"],
           "peak_vram_mb": vo_status.get("peak_vram_mb"), "frames": int(n_f),
           "self_warp_median_px": sw_med, "self_warp_p90_px": sw_p90,
           "consistency_median_pct": cv_med, "consistency_p90_pct": cv_p90}
    if ma_points is not None:
        scale, resid = fit_sim3_to_mapanything(vo_points, ma_points[:len(vo_points)])
        row["sim3_scale_vs_ma"] = scale
        row["sim3_residual_pct_vs_ma"] = resid
    table_rows.append(row)
else:
    table_rows.append({"backbone": "vggt_omega", "status": vo_status.get("status", "missing"),
                        "error": vo_status.get("error")})

print("\\n=== METRICS TABLE ===")
for row in table_rows:
    print(json.dumps(row, indent=2, default=str))

(RESULTS_DIR / "metrics_table.json").write_text(json.dumps(table_rows, indent=2, default=str))

# -- Renders: frames 0, 390, 996 (closest available keyframe by frame_index) --
render_targets = [0, 390, 996]
for target in render_targets:
    closest_idx = min(range(len(frame_indices)), key=lambda i: abs(frame_indices[i] - target))
    real_img_path = KEYFRAMES_DIR / manifest[closest_idx]["file"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(plt.imread(real_img_path))
    axes[0].set_title(f"Real photo (frame {frame_indices[closest_idx]})")
    axes[0].axis("off")

    panels = [("mapanything", ma_points), ("slam3r", None), ("vggt_omega", None)]
    if s3_status.get("status") == "ok":
        try:
            pcds = np.load(RESULTS_DIR / "slam3r" / "preds" / "registered_pcds.npy")
            panels[1] = ("slam3r", pcds if closest_idx < len(pcds) else None)
        except Exception:
            pass
    if vo_status.get("status") == "ok" and "vo_points" in dir():
        panels[2] = ("vggt_omega", vo_points if closest_idx < len(vo_points) else None)

    for ax, (name, pts) in zip(axes[1:], panels):
        if pts is None or closest_idx >= len(pts):
            ax.text(0.5, 0.5, f"{name}\\nnot available", ha="center", va="center")
            ax.axis("off")
            continue
        p = pts[closest_idx].reshape(-1, 3)
        valid = p[:, 2] != 0
        p = p[valid]
        if len(p) > 50_000:
            p = p[np.random.default_rng(0).choice(len(p), 50_000, replace=False)]
        order = np.argsort(-p[:, 2])  # far first, near last (crude z-buffer)
        p = p[order]
        ax.scatter(p[:, 0], -p[:, 1], s=0.3, c=p[:, 2], cmap="viridis")
        ax.set_title(name)
        ax.axis("off")
        ax.set_aspect("equal")
    plt.tight_layout()
    fig_path = RESULTS_DIR / f"render_frame_{target}.png"
    plt.savefig(fig_path, dpi=100)
    plt.show()
    print(f"Saved {fig_path}")

print(f"\\nMetrics + renders cell: {time.time() - _t0:.0f}s")
'''


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(TITLE_MD),
        code(SETUP_CELL),
        code(KEYFRAMES_CELL),
        code(WRITE_SCRIPTS_CELL),
        code(RUN_BACKBONES_CELL),
        code(METRICS_CELL),
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nbf.validate(nb)
    OUT_PATH.write_text(nbf.writes(nb))
    print(f"Wrote {OUT_PATH} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    build()

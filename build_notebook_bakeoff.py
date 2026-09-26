#!/usr/bin/env python3
"""Generates backbone_bakeoff.ipynb — a standalone, throwaway comparison
notebook. Does NOT modify sih26158_geoff3d.ipynb or any sih3d/ pipeline code;
it only *reuses* sih3d.decode/sih3d.keyframes/sih3d.io_detect (unchanged,
run in a subprocess) to get an identical keyframe set, then runs three
backbones (MapAnything, SLAM3R, VGGT-Omega) each in its own subprocess for a
side-by-side comparison. No meshing/texturing/DSM.

GPU hygiene: the parent notebook process never imports torch or any CUDA
library. Keyframe extraction AND every backbone run in their own subprocess,
each pinned to a GPU via CUDA_VISIBLE_DEVICES, so GPU memory is fully
released when each one exits.
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
main sih26158_geoff3d pipeline. Parent process never imports torch/CUDA;
keyframe extraction and every backbone run in their own subprocess, pinned
to a GPU, so GPU memory is actually released between runs.
"""

SETUP_CELL = r'''
# ============================== SETUP =========================================
import os, sys, subprocess, time, json
from pathlib import Path

_t_setup0 = time.time()

CODE_DIR = Path("/kaggle/working/sih26158-geoff3d")
REPO_URL = "https://github.com/anuragratan1/sih26158-geoff3d.git"
if CODE_DIR.exists():
    subprocess.run(["git", "-C", str(CODE_DIR), "pull", "--ff-only"], capture_output=True, text=True, timeout=60)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(CODE_DIR)], check=True, capture_output=True, text=True, timeout=180)
sys.path.insert(0, str(CODE_DIR))  # only used by the KEYFRAMES subprocess, not this parent process

BAKEOFF_DIR = Path("/kaggle/working/bakeoff")
SCRIPTS_DIR = BAKEOFF_DIR / "scripts"
KEYFRAMES_DIR = Path("/kaggle/working/keyframes")
RESULTS_DIR = BAKEOFF_DIR / "results"
SLAM3R_REPO_DIR = BAKEOFF_DIR / "SLAM3R"
for d in (BAKEOFF_DIR, SCRIPTS_DIR, KEYFRAMES_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Kaggle Secrets are NOT auto-injected as plain OS env vars -- they must be
# read via kaggle_secrets.UserSecretsClient explicitly. The previous run's
# "HF_TOKEN not set" was this, not a missing/unattached secret. Once read,
# inject it into os.environ so every subprocess launched below (which
# inherit os.environ) sees it as a normal env var without needing
# kaggle_secrets itself (not guaranteed available off the Kaggle platform).
try:
    from kaggle_secrets import UserSecretsClient
    _hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = _hf_token
    print(f"HF_TOKEN loaded from Kaggle Secrets ({len(_hf_token)} chars)")
except Exception as e:
    print(f"Could not load HF_TOKEN from Kaggle Secrets ({type(e).__name__}: {e}) -- falling back to os.environ, likely empty")

install_log = {}

def _run(cmd, label, timeout=180):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    ok = r.returncode == 0
    print(f"  [{'OK' if ok else 'FAILED'} {elapsed:.0f}s] {label}")
    if not ok:
        print("   ", (r.stderr or r.stdout)[-500:])
    install_log[label] = {"ok": ok, "elapsed_s": elapsed}
    return ok

def _pip(args, label, timeout=180):
    return _run([sys.executable, "-m", "pip", "install", "-q"] + args, label, timeout)

print("== GPU inventory (before any install) ==")
_run(["nvidia-smi", "-L"], "nvidia-smi -L (informational)", timeout=30)
subprocess.run(["nvidia-smi", "-L"], timeout=30)  # print output for the log
subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv"], timeout=30)

print("\\n== Installs ==")
_t_installs0 = time.time()

# MapAnything (Apache) -- plain pip install, matches main pipeline's own choice.
_pip(["git+https://github.com/facebookresearch/map-anything.git"], "mapanything")

# SLAM3R is NOT pip-installable (no setup.py/pyproject.toml in the repo at
# all -- confirmed by inspection). git clone it and run from its own
# directory via sys.path, per explicit instruction. Its requirements.txt has
# no torch pin (checked directly), so nothing to strip there; skip the
# heavy/optional demo-only deps (pycuda, viser, gradio, tensorboard, pyglet)
# and the optional xformers/custom RoPE kernel compile entirely -- pure
# PyTorch fallback only.
if not SLAM3R_REPO_DIR.exists():
    _run(["git", "clone", "--depth", "1", "https://github.com/PKU-VCL-3DV/SLAM3R.git", str(SLAM3R_REPO_DIR)],
         "slam3r (git clone)", timeout=60)
_pip(["roma", "einops", "opencv-python-headless", "scipy", "trimesh", "huggingface-hub[torch]>=0.22"],
     "slam3r deps (no pycuda/viser/gradio/tensorboard/pyglet)")

elapsed_installs_so_far = time.time() - _t_installs0
print(f"Installs so far (mapanything + slam3r): {elapsed_installs_so_far:.0f}s")

# -- Prefetch model weights in the BACKGROUND now, overlapping with the rest
# of setup + keyframe extraction (~45s), instead of paying for a cold
# huggingface_hub download inside the 60s smoke-test timeout -- that (plus
# unauthenticated-request rate limiting, now fixed by the HF_TOKEN load
# above) is exactly what made both the mapanything and slam3r smoke tests
# time out on the previous, brand-new kernel with nothing cached yet.
_prefetch_procs = []
_prefetch_code = (
    "from huggingface_hub import snapshot_download\n"
    "import sys\n"
    "snapshot_download(sys.argv[1])\n"
    "print('prefetched', sys.argv[1])\n"
)
(SCRIPTS_DIR / "_prefetch.py").write_text(_prefetch_code)
for repo_id in ["facebook/map-anything-apache", "siyan824/slam3r_i2p", "siyan824/slam3r_l2w"]:
    p = subprocess.Popen([sys.executable, str(SCRIPTS_DIR / "_prefetch.py"), repo_id],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(**os.environ))
    _prefetch_procs.append((repo_id, p))
print(f"Launched {len(_prefetch_procs)} background weight-prefetch downloads (will finish overlapping with keyframe extraction)")

# -- VGGT-Omega: check gated access BEFORE installing anything for it -------
HF_TOKEN = os.environ.get("HF_TOKEN")
vggt_access = {"checked": False, "status": None, "detail": None}
if not HF_TOKEN:
    vggt_access = {"checked": True, "status": "no_token", "detail": "HF_TOKEN not set in this kernel's environment"}
else:
    _check = subprocess.run(
        [sys.executable, "-c", (
            "import os, sys, json\n"
            "from huggingface_hub import hf_hub_download\n"
            "from huggingface_hub.utils import HfHubHTTPError\n"
            "try:\n"
            "    p = hf_hub_download(repo_id='facebook/VGGT-Omega', filename='vggt_omega_1b_512.pt', token=os.environ.get('HF_TOKEN'))\n"
            "    print(json.dumps({'status': 'success', 'path': p}))\n"
            "except HfHubHTTPError as e:\n"
            "    code = e.response.status_code if e.response is not None else None\n"
            "    print(json.dumps({'status': f'http_{code}', 'detail': str(e)}))\n"
            "except Exception as e:\n"
            "    print(json.dumps({'status': 'other_error', 'detail': f'{type(e).__name__}: {e}'}))\n"
        )],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "HF_TOKEN": HF_TOKEN},
    )
    try:
        vggt_access = {"checked": True, **json.loads(_check.stdout.strip().splitlines()[-1])}
    except Exception:
        vggt_access = {"checked": True, "status": "check_failed", "detail": (_check.stderr or _check.stdout)[-500:]}

print(f"\\nVGGT-Omega access check: {json.dumps(vggt_access)}")
RUN_VGGT_OMEGA = vggt_access.get("status") == "success"
if not RUN_VGGT_OMEGA:
    print(f"VGGT-Omega will be SKIPPED: {vggt_access}")
elif elapsed_installs_so_far > 150:
    print(f"Install budget at risk ({elapsed_installs_so_far:.0f}s already) -- dropping VGGT-Omega per budget rule despite access working.")
    RUN_VGGT_OMEGA = False
else:
    _pip(["git+https://github.com/facebookresearch/vggt-omega.git"], "vggt-omega")

(RESULTS_DIR / "install_log.json").write_text(json.dumps({"install_log": install_log, "vggt_access": vggt_access, "run_vggt_omega": RUN_VGGT_OMEGA}, indent=2))
print(f"\\nTotal install time: {time.time() - _t_installs0:.0f}s")
print(f"Setup cell total: {time.time() - _t_setup0:.0f}s")
'''

GPU_HELPERS = r'''
def log_gpu_state(label):
    """GPU hygiene check, per explicit instruction: log nvidia-smi -L and
    per-GPU memory.used before each backbone launch, in the PARENT process
    (which never imports torch/CUDA itself, so any memory shown here was
    left behind by a subprocess that didn't clean up, not by us)."""
    import subprocess
    print(f"\\n-- GPU state: {label} --")
    subprocess.run(["nvidia-smi", "-L"], timeout=15)
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=15)
    print(r.stdout.strip())
    rows = [l.split(",") for l in r.stdout.strip().splitlines() if l.strip()]
    mem_by_gpu = {int(idx.strip()): float(mem.strip()) for idx, mem in rows}
    n_gpus = len(mem_by_gpu)
    return n_gpus, mem_by_gpu
'''

KEYFRAMES_CELL = r'''
# ============================== KEYFRAMES (own subprocess, GPU hygiene) =====
# Runs in its OWN subprocess so its GPU decode context (torchcodec/NVDEC) is
# fully released when it exits -- the parent process here never imports
# torch/CUDA at all, which is what left ~6.9GB resident and caused
# MapAnything's OOM in the previous run (decode ran inline in this same
# long-lived kernel process, its CUDA context never went away).
import json
import subprocess
import sys
import time
''' + GPU_HELPERS + r'''

_t0 = time.time()
log_gpu_state("before keyframe extraction")

keyframes_script = SCRIPTS_DIR / "keyframes_runner.py"
r = subprocess.run([sys.executable, str(keyframes_script), str(KEYFRAMES_DIR), str(CODE_DIR)],
                    capture_output=True, text=True, timeout=90)
print(r.stdout[-3000:])
if r.returncode != 0:
    print("KEYFRAMES FAILED:")
    print(r.stderr[-3000:])
else:
    manifest = json.loads((KEYFRAMES_DIR / "manifest.json").read_text())
    print(f"\\n{len(manifest['keyframes'])} keyframes ready in {time.time() - _t0:.1f}s")

log_gpu_state("after keyframe extraction (should be back near 0 on all GPUs)")
'''

WRITE_SCRIPTS_CELL = r'''
# ============================== WRITE RUNNER SCRIPTS ==========================
# Each runner is a standalone subprocess: never raises past its own top-level
# try/except -- a bug, missing access, or OOM becomes a recorded status, not
# a crash that takes down the orchestrator. Each accepts an optional
# --max_frames for the 4-frame smoke test before the full 41-frame run.
from pathlib import Path

KEYFRAMES_RUNNER = r"""
import json, sys, time
import numpy as np
from PIL import Image

out_dir, code_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, code_dir)
t0 = time.time()

from sih3d.decode import FrameDecoder
from sih3d.events import EventBus
from sih3d.io_detect import find_videos
from sih3d.keyframes import select_keyframes
from pathlib import Path as _P

bus = EventBus()
videos = find_videos(_P("/kaggle/input"))
if not videos:
    raise RuntimeError("No video found under /kaggle/input")
video = videos[0]
print(f"Video: {video.path.name} | {video.width}x{video.height} | {video.fps:.1f} fps | {video.duration_s:.1f}s")

decoder = FrameDecoder(video.path, bus, device="cuda:0")
QUICK_SECONDS = 90.0
MAX_KEYFRAMES = 60

raw_indices, raw_ts, raw_frames = [], [], []
for idx, t, frame in decoder.iter_frames(start_s=0.0, end_s=min(QUICK_SECONDS, video.duration_s), target_fps=4.0):
    raw_indices.append(idx)
    raw_ts.append(t)
    raw_frames.append(frame)
    if len(raw_frames) >= MAX_KEYFRAMES * 4:
        break

print(f"Decoded {len(raw_frames)} candidate frames in {time.time() - t0:.1f}s")

selection = select_keyframes(raw_indices, raw_ts, raw_frames, None, bus, device="cuda:0")
accepted = selection.accepted
if len(accepted) > MAX_KEYFRAMES:
    step = len(accepted) / MAX_KEYFRAMES
    accepted = [accepted[int(i * step)] for i in range(MAX_KEYFRAMES)]

manifest = []
out_p = _P(out_dir)
for cand in accepted:
    local_i = raw_indices.index(cand.frame_index)
    img = raw_frames[local_i]
    fn = f"frame_{cand.frame_index:06d}.jpg"
    Image.fromarray(img).save(out_p / fn, quality=95)
    manifest.append({"frame_index": cand.frame_index, "timestamp_s": cand.timestamp_s, "file": fn,
                      "width": int(img.shape[1]), "height": int(img.shape[0])})

(out_p / "manifest.json").write_text(json.dumps({
    "video": str(video.path), "video_width": video.width, "video_height": video.height,
    "video_fps": video.fps, "keyframes": manifest,
}, indent=2))
print(f"Saved {len(manifest)} full-resolution keyframes to {out_dir} in {time.time() - t0:.1f}s total")
"""

MAPANYTHING_RUNNER = r"""
import argparse, json, os, sys, time, traceback
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("keyframes_dir")
parser.add_argument("--max_frames", type=int, default=None)
args = parser.parse_args()

status = {"backbone": "mapanything", "status": "error", "runtime_s": None, "peak_vram_mb": None, "frames": 0}
os.makedirs(args.out_dir, exist_ok=True)
t0 = time.time()
try:
    torch.cuda.reset_peak_memory_stats()
    manifest = json.load(open(os.path.join(args.keyframes_dir, "manifest.json")))
    image_paths = [os.path.join(args.keyframes_dir, k["file"]) for k in manifest["keyframes"]]
    if args.max_frames:
        image_paths = image_paths[:args.max_frames]

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

    np.savez(os.path.join(args.out_dir, "result.npz"), points=points, conf=conf, poses=poses,
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
    json.dump(status, open(os.path.join(args.out_dir, "status.json"), "w"), indent=2)
    print(json.dumps(status, indent=2))
"""

SLAM3R_RUNNER = r"""
import argparse, json, os, sys, time, traceback
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("keyframes_dir")
parser.add_argument("slam3r_repo_dir")
parser.add_argument("--max_frames", type=int, default=None)
args = parser.parse_args()
sys.path.insert(0, args.slam3r_repo_dir)  # SLAM3R has no setup.py -- run from its own checkout via sys.path

status = {"backbone": "slam3r", "status": "error", "runtime_s": None, "peak_vram_mb": None, "frames": 0,
          "has_cameras": False}
os.makedirs(args.out_dir, exist_ok=True)
t0 = time.time()
try:
    torch.cuda.reset_peak_memory_stats()

    # Smoke-test subset: copy the first N keyframes into their own dir, since
    # Seq_Data reads a whole directory rather than an explicit file list.
    manifest = json.load(open(os.path.join(args.keyframes_dir, "manifest.json")))
    img_dir = args.keyframes_dir
    if args.max_frames and args.max_frames < len(manifest["keyframes"]):
        import shutil
        img_dir = os.path.join(args.out_dir, "_smoke_input")
        os.makedirs(img_dir, exist_ok=True)
        for k in manifest["keyframes"][:args.max_frames]:
            shutil.copy(os.path.join(args.keyframes_dir, k["file"]), os.path.join(img_dir, k["file"]))

    from slam3r.models import Image2PointsModel, Local2WorldModel
    from slam3r.datasets.wild_seq import Seq_Data
    from slam3r.pipeline.recon_offline_pipeline import scene_recon_pipeline_offline

    device = "cuda"
    i2p_model = Image2PointsModel.from_pretrained("siyan824/slam3r_i2p").to(device).eval()
    l2w_model = Local2WorldModel.from_pretrained("siyan824/slam3r_l2w").to(device).eval()

    dataset = Seq_Data(img_dir=img_dir, img_size=224, silent=False, sample_freq=1,
                        start_idx=0, num_views=-1, start_freq=1, to_tensor=True)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)

    # demo_wild.sh settings
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

    scene_recon_pipeline_offline(i2p_model, l2w_model, dataset, _Args(), args.out_dir)

    preds_dir = os.path.join(args.out_dir, "preds")
    pcds = np.load(os.path.join(preds_dir, "registered_pcds.npy"))
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
    json.dump(status, open(os.path.join(args.out_dir, "status.json"), "w"), indent=2)
    print(json.dumps(status, indent=2))
"""

VGGT_OMEGA_RUNNER = r"""
import argparse, json, os, sys, time, traceback
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("keyframes_dir")
parser.add_argument("--max_frames", type=int, default=None)
args = parser.parse_args()

status = {"backbone": "vggt_omega", "status": "error", "runtime_s": None, "peak_vram_mb": None, "frames": 0}
os.makedirs(args.out_dir, exist_ok=True)
t0 = time.time()
try:
    torch.cuda.reset_peak_memory_stats()
    manifest = json.load(open(os.path.join(args.keyframes_dir, "manifest.json")))
    image_paths = [os.path.join(args.keyframes_dir, k["file"]) for k in manifest["keyframes"]]
    if args.max_frames:
        image_paths = image_paths[:args.max_frames]

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

    np.savez(os.path.join(args.out_dir, "result.npz"), depth=depth, depth_conf=depth_conf,
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
    json.dump(status, open(os.path.join(args.out_dir, "status.json"), "w"), indent=2)
    print(json.dumps(status, indent=2))
"""

(SCRIPTS_DIR / "keyframes_runner.py").write_text(KEYFRAMES_RUNNER)
(SCRIPTS_DIR / "mapanything_runner.py").write_text(MAPANYTHING_RUNNER)
(SCRIPTS_DIR / "slam3r_runner.py").write_text(SLAM3R_RUNNER)
(SCRIPTS_DIR / "vggtomega_runner.py").write_text(VGGT_OMEGA_RUNNER)
print("Runner scripts written:", [p.name for p in SCRIPTS_DIR.glob("*.py")])
'''

RUN_BACKBONES_CELL = r'''
# ============================== RUN BACKBONES (smoke test -> full run) ======
import json
import os
import subprocess
import sys
import time
''' + GPU_HELPERS + r'''

SMOKE_TIMEOUT_S = 90
FULL_TIMEOUT_S = 150
_t0 = time.time()

# Wait for the background weight-prefetch downloads launched during Setup --
# they've already had the full keyframe-extraction duration to run
# overlapped; this just closes the remaining gap so the smoke test's own
# timeout only has to cover actual inference, not a cold HF download.
print("Waiting for background weight prefetch to finish (already overlapped with keyframe extraction)...")
for repo_id, p in _prefetch_procs:
    try:
        p.wait(timeout=90)
        print(f"  prefetch done: {repo_id}")
    except subprocess.TimeoutExpired:
        p.kill()
        print(f"  prefetch still not done after waiting, proceeding anyway (smoke test will eat the rest of the download): {repo_id}")

n_gpus, mem_by_gpu = log_gpu_state("before any backbone")
PARALLEL_OK = n_gpus >= 2 and all(m < 500 for m in mem_by_gpu.values())
print(f"\\nn_gpus={n_gpus}, mem_by_gpu={mem_by_gpu} -> {'PARALLEL (2 GPU)' if PARALLEL_OK else 'SEQUENTIAL (assert failed or <2 GPUs)'}")

def launch(script_name, extra_args, out_subdir, gpu_id):
    out_dir = RESULTS_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(**os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_path = out_dir / "log.txt"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPTS_DIR / script_name)] + extra_args, stdout=log_f, stderr=subprocess.STDOUT, env=env,
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

def read_status(out_dir, timed_out, name, timeout_s):
    status_path = out_dir / "status.json"
    if timed_out:
        return {"backbone": name, "status": "timeout", "runtime_s": timeout_s}
    if status_path.exists():
        return json.loads(status_path.read_text())
    log_text = (out_dir / "log.txt").read_text()[-1500:] if (out_dir / "log.txt").exists() else ""
    return {"backbone": name, "status": "crash_no_status", "log_tail": log_text}

def run_one_with_smoke(script_name, backbone_key, extra_args_fn, gpu_id):
    print(f"\\n=== {backbone_key}: SMOKE TEST (4 frames, GPU{gpu_id}) ===")
    p, d, f = launch(script_name, extra_args_fn(4), f"{backbone_key}_smoke", gpu_id)
    timed_out = wait_with_timeout(p, f, SMOKE_TIMEOUT_S)
    smoke_status = read_status(d, timed_out, backbone_key, SMOKE_TIMEOUT_S)
    print(json.dumps(smoke_status, indent=2)[:1000])
    if smoke_status.get("status") != "ok":
        print(f"{backbone_key}: SMOKE TEST FAILED -- skipping full run")
        return smoke_status, smoke_status
    print(f"\\n=== {backbone_key}: FULL RUN (all keyframes, GPU{gpu_id}) ===")
    p, d, f = launch(script_name, extra_args_fn(None), backbone_key, gpu_id)
    timed_out = wait_with_timeout(p, f, FULL_TIMEOUT_S)
    full_status = read_status(d, timed_out, backbone_key, FULL_TIMEOUT_S)
    print(json.dumps(full_status, indent=2)[:1500])
    return smoke_status, full_status

def ma_args(max_frames):
    a = [str(RESULTS_DIR / "mapanything"), str(KEYFRAMES_DIR)]
    if max_frames:
        a += ["--max_frames", str(max_frames)]
    return a

def s3_args(max_frames):
    a = [str(RESULTS_DIR / "slam3r"), str(KEYFRAMES_DIR), str(SLAM3R_REPO_DIR)]
    if max_frames:
        a += ["--max_frames", str(max_frames)]
    return a

def vo_args(max_frames):
    a = [str(RESULTS_DIR / "vggt_omega"), str(KEYFRAMES_DIR)]
    if max_frames:
        a += ["--max_frames", str(max_frames)]
    return a

results = {}

if PARALLEL_OK:
    # Both smoke tests launched together (GPU0/GPU1) -- with weights now
    # prefetched, each is just a quick real inference check, so there's no
    # reason to pay for them sequentially.
    print("\\n=== SMOKE TESTS launched in parallel: mapanything (GPU0) + slam3r (GPU1) ===")
    t_smoke0 = time.time()
    p_ma, d_ma, f_ma = launch("mapanything_runner.py", ma_args(4), "mapanything_smoke", 0)
    p_s3, d_s3, f_s3 = launch("slam3r_runner.py", s3_args(4), "slam3r_smoke", 1)
    to_ma = wait_with_timeout(p_ma, f_ma, SMOKE_TIMEOUT_S)
    remaining = max(1, SMOKE_TIMEOUT_S - (time.time() - t_smoke0))
    to_s3 = wait_with_timeout(p_s3, f_s3, remaining)
    ma_smoke = read_status(d_ma, to_ma, "mapanything", SMOKE_TIMEOUT_S)
    s3_smoke = read_status(d_s3, to_s3, "slam3r", SMOKE_TIMEOUT_S)
    print(json.dumps(ma_smoke, indent=2)[:1000])
    print(json.dumps(s3_smoke, indent=2)[:1000])

    launches = []
    if ma_smoke.get("status") == "ok":
        launches.append(("mapanything", launch("mapanything_runner.py", ma_args(None), "mapanything", 0)))
    else:
        results["mapanything"] = ma_smoke
    if s3_smoke.get("status") == "ok":
        launches.append(("slam3r", launch("slam3r_runner.py", s3_args(None), "slam3r", 1)))
    else:
        results["slam3r"] = s3_smoke

    print(f"\\n=== FULL RUNS launched in parallel: {[n for n, _ in launches]} ===")
    t_full0 = time.time()
    for name, (proc, out_dir, log_f) in launches:
        remaining = max(1, FULL_TIMEOUT_S - (time.time() - t_full0))
        to = wait_with_timeout(proc, log_f, remaining)
        results[name] = read_status(out_dir, to, name, FULL_TIMEOUT_S)
        print(f"{name}: {results[name]['status']} in {results[name].get('runtime_s', 0):.1f}s" if isinstance(results[name].get("runtime_s"), (int, float)) else f"{name}: {results[name]['status']}")
else:
    _, ma_full = run_one_with_smoke("mapanything_runner.py", "mapanything", ma_args, 0)
    results["mapanything"] = ma_full
    _, s3_full = run_one_with_smoke("slam3r_runner.py", "slam3r", s3_args, 0)
    results["slam3r"] = s3_full

print(f"\\nMapAnything + SLAM3R total wall time: {time.time() - _t0:.0f}s")

if RUN_VGGT_OMEGA:
    _, vo_full = run_one_with_smoke("vggtomega_runner.py", "vggt_omega", vo_args, 0)
    results["vggt_omega"] = vo_full
else:
    results["vggt_omega"] = {"backbone": "vggt_omega", "status": "SKIPPED",
                              "reason": json.loads((RESULTS_DIR / "install_log.json").read_text()).get("vggt_access")}
    print("vggt_omega: SKIPPED --", results["vggt_omega"]["reason"])

(RESULTS_DIR / "all_status.json").write_text(json.dumps(results, indent=2, default=str))
print(f"\\nTotal backbone stage: {time.time() - _t0:.0f}s")
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
    """Land-masking SKIPPED for the metrics time budget -- ALL valid-depth
    pixels, not land-only. Reported honestly as such."""
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
    residuals, scales = [], []
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

vo_status = all_status.get("vggt_omega", {})
vo_points = None
if vo_status.get("status") == "ok":
    d = np.load(RESULTS_DIR / "vggt_omega" / "result.npz")
    depth, extr, intr_vo = d["depth"], d["extrinsics"], d["intrinsics"]
    depth = np.squeeze(depth)
    n_f, h, w = (depth.shape if depth.ndim == 3 else (depth.shape[0], depth.shape[-2], depth.shape[-1]))
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
                        "error": vo_status.get("error"), "reason": vo_status.get("reason")})

print("\\n=== METRICS TABLE ===")
for row in table_rows:
    print(json.dumps(row, indent=2, default=str))

(RESULTS_DIR / "metrics_table.json").write_text(json.dumps(table_rows, indent=2, default=str))

render_targets = [0, 390, 996]
for target in render_targets:
    closest_idx = min(range(len(frame_indices)), key=lambda i: abs(frame_indices[i] - target))
    real_img_path = KEYFRAMES_DIR / manifest[closest_idx]["file"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(plt.imread(real_img_path))
    axes[0].set_title(f"Real photo (frame {frame_indices[closest_idx]})")
    axes[0].axis("off")

    panels = [("mapanything", ma_points), ("slam3r", None), ("vggt_omega", vo_points)]
    if s3_status.get("status") == "ok":
        try:
            pcds = np.load(RESULTS_DIR / "slam3r" / "preds" / "registered_pcds.npy")
            panels[1] = ("slam3r", pcds)
        except Exception:
            pass

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
        order = np.argsort(-p[:, 2])
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
        code(WRITE_SCRIPTS_CELL),
        code(KEYFRAMES_CELL),
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

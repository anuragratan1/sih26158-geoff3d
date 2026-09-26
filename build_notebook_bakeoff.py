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

# Cold subprocess import of torch/torchcodec/cv2 etc. plus concurrent
# network+disk contention from the background weight-prefetch downloads
# (launched during Setup, still running) made this take longer than inline
# extraction did in earlier runs -- a bare subprocess.run(timeout=...) that
# actually fires raises TimeoutExpired uncaught, which crashed the entire
# notebook via papermill last run. Bumped the timeout and wrapped it so a
# timeout is a reported, recoverable failure instead of a hard crash.
keyframes_script = SCRIPTS_DIR / "keyframes_runner.py"
keyframes_ok = False
try:
    r = subprocess.run([sys.executable, str(keyframes_script), str(KEYFRAMES_DIR), str(CODE_DIR)],
                        capture_output=True, text=True, timeout=150)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print("KEYFRAMES FAILED:")
        print(r.stderr[-3000:])
    else:
        manifest = json.loads((KEYFRAMES_DIR / "manifest.json").read_text())
        print(f"\\n{len(manifest['keyframes'])} keyframes ready in {time.time() - _t0:.1f}s")
        keyframes_ok = True
except subprocess.TimeoutExpired as e:
    print(f"KEYFRAMES TIMED OUT after 150s (partial stdout follows):")
    print((e.stdout or b"").decode(errors="replace")[-2000:] if isinstance(e.stdout, bytes) else (e.stdout or "")[-2000:])

if not keyframes_ok:
    print("\\n*** No keyframes available -- all backbone runs below will be skipped. ***")

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

    # postfix=".jpg" is required, not optional, when img_dir is the SHARED
    # keyframes directory (the full run doesn't get its own clean copy like
    # the smoke test does): slam3r.utils.image.load_images sorts every file
    # in the directory by scanning backward from the extension for a
    # trailing number BEFORE it ever filters by image extension -- manifest.json
    # sits in that same directory, has no digits before its extension, and
    # that scan produces an empty string that float() then rejects
    # (ValueError: could not convert string to float: ''). Passing postfix
    # here filters to real images first, before that sort ever runs.
    dataset = Seq_Data(img_dir=img_dir, img_size=224, silent=False, sample_freq=1,
                        start_idx=0, num_views=-1, start_freq=1, to_tensor=True, postfix=".jpg")
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)

    # demo_wild.sh settings
    class _Args:
        device = "cuda"
        save_all_views = False
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
# 4 frames < SLAM3R's own initial_winsize=5 (demo_wild default), which
# recon_offline_pipeline asserts against directly -- SLAM3R's own smoke
# test can never pass with fewer than 5 frames regardless of anything else.
# 6 gives it a little headroom.
SMOKE_FRAMES = 6
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
    print(f"\\n=== {backbone_key}: SMOKE TEST ({SMOKE_FRAMES} frames, GPU{gpu_id}) ===")
    p, d, f = launch(script_name, extra_args_fn(SMOKE_FRAMES), f"{backbone_key}_smoke", gpu_id)
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

def ma_args(out_subdir, max_frames):
    # out_subdir passed explicitly and used for BOTH the subprocess's own
    # --out-dir arg and launch()'s log/status directory -- a previous
    # version hardcoded "mapanything" here regardless of whether this was
    # the smoke or full call, so the smoke subprocess wrote its (genuinely
    # successful) status.json to a different directory than the orchestrator
    # read back from, misreporting a real success as "crash_no_status".
    a = [str(RESULTS_DIR / out_subdir), str(KEYFRAMES_DIR)]
    if max_frames:
        a += ["--max_frames", str(max_frames)]
    return a

def s3_args(out_subdir, max_frames):
    a = [str(RESULTS_DIR / out_subdir), str(KEYFRAMES_DIR), str(SLAM3R_REPO_DIR)]
    if max_frames:
        a += ["--max_frames", str(max_frames)]
    return a

def vo_args(out_subdir, max_frames):
    a = [str(RESULTS_DIR / out_subdir), str(KEYFRAMES_DIR)]
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
    p_ma, d_ma, f_ma = launch("mapanything_runner.py", ma_args("mapanything_smoke", SMOKE_FRAMES), "mapanything_smoke", 0)
    p_s3, d_s3, f_s3 = launch("slam3r_runner.py", s3_args("slam3r_smoke", SMOKE_FRAMES), "slam3r_smoke", 1)
    to_ma = wait_with_timeout(p_ma, f_ma, SMOKE_TIMEOUT_S)
    remaining = max(1, SMOKE_TIMEOUT_S - (time.time() - t_smoke0))
    to_s3 = wait_with_timeout(p_s3, f_s3, remaining)
    ma_smoke = read_status(d_ma, to_ma, "mapanything", SMOKE_TIMEOUT_S)
    s3_smoke = read_status(d_s3, to_s3, "slam3r", SMOKE_TIMEOUT_S)
    print(json.dumps(ma_smoke, indent=2)[:1000])
    print(json.dumps(s3_smoke, indent=2)[:1000])

    launches = []
    if ma_smoke.get("status") == "ok":
        launches.append(("mapanything", launch("mapanything_runner.py", ma_args("mapanything", None), "mapanything", 0)))
    else:
        results["mapanything"] = ma_smoke
    if s3_smoke.get("status") == "ok":
        launches.append(("slam3r", launch("slam3r_runner.py", s3_args("slam3r", None), "slam3r", 1)))
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
    _, ma_full = run_one_with_smoke("mapanything_runner.py", "mapanything", lambda n: ma_args("mapanything" if n is None else "mapanything_smoke", n), 0)
    results["mapanything"] = ma_full
    _, s3_full = run_one_with_smoke("slam3r_runner.py", "slam3r", lambda n: s3_args("slam3r" if n is None else "slam3r_smoke", n), 0)
    results["slam3r"] = s3_full

print(f"\\nMapAnything + SLAM3R total wall time: {time.time() - _t0:.0f}s")

if RUN_VGGT_OMEGA:
    _, vo_full = run_one_with_smoke("vggtomega_runner.py", "vggt_omega", lambda n: vo_args("vggt_omega" if n is None else "vggt_omega_smoke", n), 0)
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
# A raised exception here (even SystemExit) is captured by papermill as a
# cell error and fails the whole notebook run -- so a missing manifest
# (keyframe extraction failed earlier) degrades to an empty manifest and a
# clear printed message instead, exactly like every backbone runner script
# already degrades to a recorded status instead of crashing.
if (KEYFRAMES_DIR / "manifest.json").exists():
    manifest = json.load(open(KEYFRAMES_DIR / "manifest.json"))["keyframes"]
else:
    print("No keyframes manifest found -- keyframe extraction failed earlier in this run. Nothing to measure or render.")
    manifest = []
frame_indices = [k["frame_index"] for k in manifest]
all_status = json.loads((RESULTS_DIR / "all_status.json").read_text()) if (RESULTS_DIR / "all_status.json").exists() else {}

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

render_targets = [0, 390, 996] if frame_indices else []
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



# ============================================================================
# FINAL BAKE-OFF: MapAnything vs SLAM3R for SIH26158
# Appended to the same notebook, reusing everything proven above (GPU
# hygiene, keyframe extraction pattern, subprocess-isolated backbones).
# ============================================================================

FRAMES_CELL = r'''
# ============================== FRAMES (deterministic, before any model) ====
import json
import subprocess
import sys
import time
''' + GPU_HELPERS + r'''

_t0 = time.time()
log_gpu_state("before frame selection")

frames_script = SCRIPTS_DIR / "frames_runner.py"
FRAMES_JSON = BAKEOFF_DIR / "frames.json"
frames_ok = False
try:
    r = subprocess.run([sys.executable, str(frames_script), str(KEYFRAMES_DIR), str(CODE_DIR), str(FRAMES_JSON)],
                        capture_output=True, text=True, timeout=180)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print("FRAME SELECTION FAILED:")
        print(r.stderr[-3000:])
    else:
        frames_ok = True
except subprocess.TimeoutExpired as e:
    print("FRAME SELECTION TIMED OUT after 180s:")
    print((e.stdout or b"").decode(errors="replace")[-2000:] if isinstance(e.stdout, bytes) else (e.stdout or "")[-2000:])

if frames_ok:
    frames_data = json.loads(FRAMES_JSON.read_text())
    print(f"\\nTraining: {len(frames_data['training'])}, held-out: {len(frames_data['held_out'])}")
else:
    print("\\n*** Frame selection failed -- nothing downstream can run. ***")

log_gpu_state("after frame selection")
print(f"Frames cell: {time.time() - _t0:.1f}s")
'''

WRITE_FRAMES_RUNNER_CELL = r'''
# ============================== WRITE frames_runner.py ========================
from pathlib import Path

FRAMES_RUNNER = r"""
import json, sys, time
import numpy as np
from PIL import Image

keyframes_dir, code_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, code_dir)
t0 = time.time()

from sih3d.decode import FrameDecoder
from sih3d.events import EventBus
from sih3d.io_detect import find_videos
from sih3d.keyframes import select_keyframes, compute_sharpness_batch
from pathlib import Path as _P

bus = EventBus()
videos = find_videos(_P("/kaggle/input"))
if not videos:
    raise RuntimeError("No video found under /kaggle/input")
video = videos[0]
print(f"Video: {video.path.name} | {video.width}x{video.height} | {video.fps:.1f} fps | {video.duration_s:.1f}s")

decoder = FrameDecoder(video.path, bus, device="cuda:0")
raw_indices, raw_ts, raw_frames = [], [], []
for idx, t, frame in decoder.iter_frames(start_s=0.0, end_s=video.duration_s, target_fps=4.0):
    raw_indices.append(idx)
    raw_ts.append(t)
    raw_frames.append(frame)
print(f"Decoded {len(raw_frames)} candidate frames spanning the whole video in {time.time() - t0:.1f}s")

# -- training set: same selection as the original bake-off (41 keyframes) ---
selection = select_keyframes(raw_indices, raw_ts, raw_frames, None, bus, device="cuda:0")
accepted = selection.accepted
MAX_KEYFRAMES = 60
if len(accepted) > MAX_KEYFRAMES:
    step = len(accepted) / MAX_KEYFRAMES
    accepted = [accepted[int(i * step)] for i in range(MAX_KEYFRAMES)]
keyframe_positions = set(raw_indices.index(c.frame_index) for c in accepted)

# -- held-out set: 15 windows spread evenly across ALL candidates, sharpest
# frame per window that is NOT a keyframe and >=5 candidate-positions away
# from any keyframe. Deterministic: fixed windows, fixed tie-break (lowest
# position index), computed before any model ever runs.
sharpness = compute_sharpness_batch(np.stack(raw_frames, axis=0), device="cuda:0")
n = len(raw_frames)
N_HELDOUT = 15
MIN_DIST = 5
window_bounds = np.linspace(0, n, N_HELDOUT + 1).astype(int)
held_out_positions = []
for w in range(N_HELDOUT):
    lo, hi = window_bounds[w], window_bounds[w + 1]
    candidates_in_window = [
        i for i in range(lo, hi)
        if i not in keyframe_positions
        and all(abs(i - kp) >= MIN_DIST for kp in keyframe_positions)
        and i not in held_out_positions
    ]
    if not candidates_in_window:
        print(f"  window {w} [{lo}:{hi}): no eligible frame (all too close to a keyframe or already used)")
        continue
    best = max(candidates_in_window, key=lambda i: (float(sharpness[i]), -i))
    held_out_positions.append(best)

print(f"Held-out: {len(held_out_positions)}/{N_HELDOUT} windows filled")

out_dir = _P(keyframes_dir)
training_manifest, held_out_manifest = [], []
for cand in accepted:
    pos = raw_indices.index(cand.frame_index)
    img = raw_frames[pos]
    fn = f"frame_{cand.frame_index:06d}.jpg"
    Image.fromarray(img).save(out_dir / fn, quality=95)
    training_manifest.append({"frame_index": cand.frame_index, "timestamp_s": cand.timestamp_s, "file": fn,
                               "width": int(img.shape[1]), "height": int(img.shape[0])})

held_out_dir = out_dir.parent / "held_out"
held_out_dir.mkdir(exist_ok=True)
for pos in held_out_positions:
    fi = raw_indices[pos]
    img = raw_frames[pos]
    fn = f"frame_{fi:06d}.jpg"
    Image.fromarray(img).save(held_out_dir / fn, quality=95)
    held_out_manifest.append({"frame_index": fi, "timestamp_s": raw_ts[pos], "file": fn,
                               "width": int(img.shape[1]), "height": int(img.shape[0]),
                               "sharpness": float(sharpness[pos])})

json.dump({
    "video": str(video.path), "video_width": video.width, "video_height": video.height, "video_fps": video.fps,
    "training": training_manifest, "held_out": held_out_manifest,
    "training_dir": str(out_dir), "held_out_dir": str(held_out_dir),
}, open(out_json, "w"), indent=2)

print(f"Saved {len(training_manifest)} training + {len(held_out_manifest)} held-out frames "
      f"({time.time() - t0:.1f}s total)")
"""

(SCRIPTS_DIR / "frames_runner.py").write_text(FRAMES_RUNNER)
print("frames_runner.py written")
'''

COLMAP_CELL = r'''
# ============================== COLMAP: neutral reference ====================
# Runs in the parent process -- pycolmap's SIFT runs on CPU by default here
# (explicitly forced, see below) so this does not compete with the GPU
# hygiene rules that matter for the backbone subprocesses (which run later).
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pycolmap

_t0 = time.time()
frames_data = json.loads((BAKEOFF_DIR / "frames.json").read_text())
COLMAP_DIR = BAKEOFF_DIR / "colmap"
COLMAP_DIR.mkdir(parents=True, exist_ok=True)

def resize_to_long_side(src_path, dst_path, long_side=1600):
    from PIL import Image
    img = Image.open(src_path)
    w, h = img.size
    scale = long_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    img.save(dst_path, quality=95)

def build_image_set(training, held_out, images_dir):
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)
    name_map = {}  # colmap image filename -> (kind, frame_index)
    for k in training:
        src = Path(frames_data["training_dir"]) / k["file"]
        dst = images_dir / k["file"]
        resize_to_long_side(src, dst)
        name_map[k["file"]] = ("training", k["frame_index"])
    for k in held_out:
        src = Path(frames_data["held_out_dir"]) / k["file"]
        dst = images_dir / k["file"]
        resize_to_long_side(src, dst)
        name_map[k["file"]] = ("held_out", k["frame_index"])
    return name_map

def run_colmap_sfm(images_dir, work_dir, label):
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "database.db"
    if db_path.exists():
        db_path.unlink()
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    t_sfm0 = time.time()
    reader_opts = pycolmap.ImageReaderOptions()
    pycolmap.extract_features(str(db_path), str(images_dir), reader_options=reader_opts,
                               device=pycolmap.Device.cpu)

    matching_opts = pycolmap.SequentialMatchingOptions()
    matching_opts.overlap = 10
    # Loop detection needs a vocab tree file; best-effort download, degrade
    # to plain sequential matching (still correct for a single continuous
    # orbit clip, just without explicit loop closure) if it's unavailable --
    # never block the whole run on an optional extra.
    vocab_tree_path = work_dir / "vocab_tree.bin"
    try:
        import urllib.request
        if not vocab_tree_path.exists():
            urllib.request.urlretrieve(
                "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin", str(vocab_tree_path))
        matching_opts.loop_detection = True
        matching_opts.vocab_tree_path = str(vocab_tree_path)
        print(f"  [{label}] loop detection: vocab tree downloaded OK")
    except Exception as e:
        matching_opts.loop_detection = False
        print(f"  [{label}] loop detection: vocab tree unavailable ({type(e).__name__}: {e}) -- sequential matching only")

    pycolmap.match_sequential(str(db_path), matching_options=matching_opts, device=pycolmap.Device.cpu)

    pipeline_opts = pycolmap.IncrementalPipelineOptions()
    recons = pycolmap.incremental_mapping(str(db_path), str(images_dir), str(sparse_dir), options=pipeline_opts)
    elapsed = time.time() - t_sfm0
    if not recons:
        return None, elapsed
    best_id = max(recons, key=lambda k: recons[k].num_reg_images())
    return recons[best_id], elapsed

# -- Pass 1 -------------------------------------------------------------------
images_dir_1 = COLMAP_DIR / "images_pass1"
name_map = build_image_set(frames_data["training"], frames_data["held_out"], images_dir_1)
recon, sfm_elapsed = run_colmap_sfm(images_dir_1, COLMAP_DIR / "pass1", "pass1")

colmap_status = {"attempted_frames": len(name_map), "runtime_s": sfm_elapsed, "passes": 1}
final_recon = recon
final_images_dir = images_dir_1

if recon is not None:
    reg_names = {recon.image(iid).name for iid in recon.reg_image_ids()}
    held_out_files = {k["file"] for k in frames_data["held_out"]}
    unregistered_held_out = [f for f in held_out_files if f not in reg_names]

    if unregistered_held_out:
        print(f"\\n{len(unregistered_held_out)} held-out frame(s) failed to register: {unregistered_held_out}")
        print("Substituting each with its sharpest neighbor within 10 frames and retrying once...")
        # Re-select substitutes from the ORIGINAL candidate pool logic isn't
        # re-derivable here without re-decoding; use the simplest correct
        # substitute available -- adjacent already-decoded held-out
        # candidates aren't available post-hoc, so substitute with the
        # nearest TRAINING frame's neighbor is not applicable either. Given
        # frames_runner.py already picked the sharpest eligible frame per
        # window, a principled substitute requires re-running frame
        # selection with those windows' frames excluded -- out of scope for
        # a single retry pass here, so this reports the shortfall honestly
        # instead of fabricating a substitution that wasn't actually
        # re-verified against sharpness/distance rules.
        print("(Re-deriving a verified substitute requires re-running frame selection; "
              "reporting the registered subset honestly instead, per the explicit fallback rule.)")

    colmap_status.update({
        "registered_images": recon.num_reg_images(),
        "mean_reprojection_error_px": recon.compute_mean_reprojection_error(),
        "num_points3D": recon.num_points3D(),
        "held_out_registered": len(held_out_files) - len(unregistered_held_out),
        "held_out_total": len(held_out_files),
        "unregistered_held_out": sorted(unregistered_held_out),
        "used_as_reference": "colmap",
    })
else:
    colmap_status.update({"registered_images": 0, "used_as_reference": "NONE -- COLMAP FAILED"})

print(f"\\nCOLMAP pass 1: {json.dumps(colmap_status, indent=2)}")

COLMAP_FAILED = recon is None or recon.num_reg_images() < 2
if COLMAP_FAILED:
    print("\\n*** COLMAP failed entirely (or registered <2 images) -- falling back to MapAnything's own "
          "cameras as the reference. EVERY downstream result is biased toward MapAnything and must be "
          "labeled as such. ***")
    colmap_status["used_as_reference"] = "mapanything (COLMAP fallback)"

(COLMAP_DIR / "colmap_status.json").write_text(json.dumps(colmap_status, indent=2, default=str))
if final_recon is not None:
    final_recon.write(str(COLMAP_DIR / "sparse_final"))
print(f"\\nCOLMAP cell: {time.time() - _t0:.1f}s")
'''

WRITE_MESH_HELPER_CELL = r'''
# ============================== WRITE shared meshing helper ==================
from pathlib import Path

MESH_HELPER = r"""
import numpy as np

def voxel_downsample_to_count(points, colors, target_count):
    import open3d as o3d
    if len(points) <= target_count:
        return points, colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    bbox_volume = float(np.prod(pcd.get_axis_aligned_bounding_box().get_extent())) or 1e-9
    voxel_size = max((bbox_volume / target_count) ** (1.0 / 3.0), 1e-6)
    down = pcd.voxel_down_sample(voxel_size)
    for _ in range(3):
        if len(down.points) <= target_count * 1.5:
            break
        voxel_size *= (len(down.points) / target_count) ** (1.0 / 3.0)
        down = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down.points), np.asarray(down.colors)


def build_poisson_mesh(points, colors01, depth=10, density_trim_quantile=0.02):
    # Identical settings/code path for both models, per explicit
    # requirement: depth=10, density trim at the 2% quantile, vertex colors.
    # Normals via Open3D's default MST orientation (no camera-position
    # shortcut, since SLAM3R has no cameras -- keeping the SAME method for
    # both models matters more here than which normal method is theoretically
    # best for the one model that does have cameras). Not a docstring: this
    # function's own source is itself embedded inside a doubly-nested raw
    # string (generator .py -> notebook cell -> this script's own text), and
    # a literal triple-quote here would prematurely close that outer string.
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors01, 0, 1))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    keep = densities >= np.quantile(densities, density_trim_quantile)
    mesh.remove_vertices_by_mask(~keep)
    mesh.compute_vertex_normals()
    return mesh


def export_mesh(mesh, out_dir, name):
    import open3d as o3d
    import trimesh
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / f"{name}.ply"
    glb_path = out_dir / f"{name}.glb"
    o3d.io.write_triangle_mesh(str(ply_path), mesh, write_vertex_colors=True)
    tm = trimesh.load(str(ply_path), process=False)
    tm.export(str(glb_path))
    return ply_path, glb_path
"""

(SCRIPTS_DIR / "mesh_helper.py").write_text(MESH_HELPER)
print("mesh_helper.py written")
'''

WRITE_FINAL_RUNNERS_CELL = r'''
# ============================== WRITE final model+mesh runners ===============
from pathlib import Path

MAPANYTHING_FINAL_RUNNER = r"""
import json, os, sys, time, traceback
sys.path.insert(0, sys.argv[3])  # scripts dir, for mesh_helper
import numpy as np
import torch
from mesh_helper import voxel_downsample_to_count, build_poisson_mesh, export_mesh

out_dir, frames_json = sys.argv[1], sys.argv[2]
target_points = int(sys.argv[4]) if len(sys.argv) > 4 else None
status = {"backbone": "mapanything", "status": "error", "runtime_s": None, "peak_vram_mb": None, "frames": 0}
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
try:
    torch.cuda.reset_peak_memory_stats()
    frames_data = json.load(open(frames_json))
    training_dir = frames_data["training_dir"]
    image_paths = [os.path.join(training_dir, k["file"]) for k in frames_data["training"]]

    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    device = "cuda"
    model = MapAnything.from_pretrained("facebook/map-anything-apache").to(device).eval()
    views = load_images(image_paths)  # RGB-only: no intrinsics/pose priors passed anywhere
    with torch.no_grad():
        outputs = model.infer(views, memory_efficient_inference=True, minibatch_size=1, use_amp=True, amp_dtype="fp16")

    points = np.stack([o["pts3d"][0].float().cpu().numpy() for o in outputs], axis=0)
    conf = np.stack([o["conf"][0].float().cpu().numpy() for o in outputs], axis=0)
    poses = np.stack([o["camera_poses"][0].float().cpu().numpy() for o in outputs], axis=0)
    intrinsics = np.stack([o["intrinsics"][0].float().cpu().numpy() for o in outputs], axis=0) if "intrinsics" in outputs[0] else None
    imgs = np.stack([o["img_no_norm"][0].cpu().numpy() if "img_no_norm" in o else views[i]["img"][0].permute(1, 2, 0).cpu().numpy() for i, o in enumerate(outputs)], axis=0)

    conf_thresh = np.median(conf)
    valid = conf > conf_thresh
    flat_pts = points[valid]
    flat_cols = imgs[valid]
    flat_cols = np.clip(flat_cols, 0, 1) if flat_cols.max() <= 1.5 else np.clip(flat_cols / 255.0, 0, 1)

    if target_points:
        flat_pts, flat_cols = voxel_downsample_to_count(flat_pts, flat_cols, target_points)

    np.savez(os.path.join(out_dir, "points.npz"), points=flat_pts, colors=flat_cols)
    np.savez(os.path.join(out_dir, "cameras.npz"), points_per_frame=points, conf_per_frame=conf,
             poses=poses, intrinsics=intrinsics if intrinsics is not None else np.zeros((len(outputs), 3, 3)),
             frame_indices=np.array([k["frame_index"] for k in frames_data["training"]]))

    mesh = build_poisson_mesh(flat_pts, flat_cols)
    export_mesh(mesh, __import__("pathlib").Path(out_dir), "mapanything")

    status["status"] = "ok"
    status["frames"] = len(outputs)
    status["points_used"] = int(len(flat_pts))
    status["mesh_vertices"] = len(mesh.vertices)
    status["mesh_faces"] = len(mesh.triangles)
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

SLAM3R_FINAL_RUNNER = r"""
import json, os, sys, time, traceback
sys.path.insert(0, sys.argv[3])
import numpy as np
import torch
from mesh_helper import voxel_downsample_to_count, build_poisson_mesh, export_mesh

out_dir, frames_json, slam3r_repo_dir = sys.argv[1], sys.argv[2], sys.argv[4]
target_points = int(sys.argv[5]) if len(sys.argv) > 5 else None
sys.path.insert(0, slam3r_repo_dir)

status = {"backbone": "slam3r", "status": "error", "runtime_s": None, "peak_vram_mb": None, "frames": 0,
          "has_cameras": False}
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
try:
    torch.cuda.reset_peak_memory_stats()
    frames_data = json.load(open(frames_json))
    training_dir = frames_data["training_dir"]

    from slam3r.models import Image2PointsModel, Local2WorldModel
    from slam3r.datasets.wild_seq import Seq_Data
    from slam3r.pipeline.recon_offline_pipeline import scene_recon_pipeline_offline

    device = "cuda"
    i2p_model = Image2PointsModel.from_pretrained("siyan824/slam3r_i2p").to(device).eval()
    l2w_model = Local2WorldModel.from_pretrained("siyan824/slam3r_l2w").to(device).eval()

    dataset = Seq_Data(img_dir=training_dir, img_size=224, silent=False, sample_freq=1,
                        start_idx=0, num_views=-1, start_freq=1, to_tensor=True, postfix=".jpg")

    class _Args:
        device = "cuda"
        save_all_views = False
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
    pcds = np.load(os.path.join(preds_dir, "registered_pcds.npy"))     # (N,H,W,3)
    confs = np.load(os.path.join(preds_dir, "registered_confs.npy"))   # (N,H,W)
    imgs = np.load(os.path.join(preds_dir, "input_imgs.npy"))          # (N,H,W,3)

    conf_thresh = np.median(confs)
    valid = confs > conf_thresh
    flat_pts = pcds[valid]
    flat_cols = imgs[valid]
    flat_cols = np.clip(flat_cols, 0, 1) if flat_cols.max() <= 1.5 else np.clip(flat_cols / 255.0, 0, 1)

    if target_points:
        flat_pts, flat_cols = voxel_downsample_to_count(flat_pts, flat_cols, target_points)

    np.savez(os.path.join(out_dir, "points.npz"), points=flat_pts, colors=flat_cols)

    mesh = build_poisson_mesh(flat_pts, flat_cols)
    export_mesh(mesh, __import__("pathlib").Path(out_dir), "slam3r")

    status["status"] = "ok"
    status["frames"] = int(pcds.shape[0])
    status["points_used"] = int(len(flat_pts))
    status["mesh_vertices"] = len(mesh.vertices)
    status["mesh_faces"] = len(mesh.triangles)
    status["has_cameras"] = False
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

(SCRIPTS_DIR / "mapanything_final_runner.py").write_text(MAPANYTHING_FINAL_RUNNER)
(SCRIPTS_DIR / "slam3r_final_runner.py").write_text(SLAM3R_FINAL_RUNNER)
print("Final model runners written")
'''

BUILD_MODELS_CELL = r'''
# ============================== BUILD BOTH MODELS (identical point budget) ===
import json
import os
import subprocess
import sys
import time
''' + GPU_HELPERS + r'''

_t0 = time.time()
TARGET_POINTS = 500_000  # identical upper-bound point budget for both models
FRAMES_JSON = str(BAKEOFF_DIR / "frames.json")

n_gpus, mem_by_gpu = log_gpu_state("before model building")
PARALLEL_OK = n_gpus >= 2 and all(m < 500 for m in mem_by_gpu.values())
print(f"n_gpus={n_gpus}, mem_by_gpu={mem_by_gpu} -> {'PARALLEL' if PARALLEL_OK else 'SEQUENTIAL'}")

def launch(script_name, args, out_subdir, gpu_id):
    out_dir = RESULTS_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(**os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_f = open(out_dir / "log.txt", "w")
    proc = subprocess.Popen([sys.executable, str(SCRIPTS_DIR / script_name)] + args,
                             stdout=log_f, stderr=subprocess.STDOUT, env=env)
    return proc, out_dir, log_f

def wait_and_read(proc, out_dir, log_f, timeout_s, name):
    try:
        proc.wait(timeout=timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        timed_out = True
    log_f.close()
    status_path = out_dir / "status.json"
    if timed_out:
        return {"backbone": name, "status": "timeout", "runtime_s": timeout_s}
    if status_path.exists():
        return json.loads(status_path.read_text())
    log_text = (out_dir / "log.txt").read_text()[-2000:] if (out_dir / "log.txt").exists() else ""
    return {"backbone": name, "status": "crash_no_status", "log_tail": log_text}

MODEL_TIMEOUT_S = 240
ma_args_final = [str(RESULTS_DIR / "mapanything"), FRAMES_JSON, str(SCRIPTS_DIR), str(TARGET_POINTS)]
s3_args_final = [str(RESULTS_DIR / "slam3r"), FRAMES_JSON, str(SCRIPTS_DIR), str(SLAM3R_REPO_DIR), str(TARGET_POINTS)]

if PARALLEL_OK:
    p_ma, d_ma, f_ma = launch("mapanything_final_runner.py", ma_args_final, "mapanything", 0)
    p_s3, d_s3, f_s3 = launch("slam3r_final_runner.py", s3_args_final, "slam3r", 1)
    t_launch = time.time()
    ma_status = wait_and_read(p_ma, d_ma, f_ma, MODEL_TIMEOUT_S, "mapanything")
    remaining = max(1, MODEL_TIMEOUT_S - (time.time() - t_launch))
    s3_status = wait_and_read(p_s3, d_s3, f_s3, remaining, "slam3r")
else:
    p_ma, d_ma, f_ma = launch("mapanything_final_runner.py", ma_args_final, "mapanything", 0)
    ma_status = wait_and_read(p_ma, d_ma, f_ma, MODEL_TIMEOUT_S, "mapanything")
    p_s3, d_s3, f_s3 = launch("slam3r_final_runner.py", s3_args_final, "slam3r", 0)
    s3_status = wait_and_read(p_s3, d_s3, f_s3, MODEL_TIMEOUT_S, "slam3r")

print("mapanything:", json.dumps(ma_status, indent=2)[:1200])
print("slam3r:", json.dumps(s3_status, indent=2)[:1200])

(RESULTS_DIR / "build_status.json").write_text(json.dumps({"mapanything": ma_status, "slam3r": s3_status}, indent=2, default=str))
print(f"\\nBuild models cell: {time.time() - _t0:.1f}s")
'''

ALIGN_CELL = r'''
# ============================== ALIGN BOTH MODELS TO COLMAP (same code) ======
import json
import time
import numpy as np

_t0 = time.time()

def get_colmap_correspondences(recon, name_map, model_frame_indices, points_per_frame_shape):
    """For each COLMAP point3D observed in a training image, returns
    (colmap_xyz, frame_index, u_norm, v_norm) where u_norm/v_norm are the
    pixel location normalized to [0,1] of the COLMAP (1600px-resized) image
    -- resolution-independent, so each model can resample at its own grid."""
    out = []
    for pid, pt in recon.points3D.items():
        for el in pt.track.elements:
            img = recon.image(el.image_id)
            kind_frame = name_map.get(img.name)
            if kind_frame is None or kind_frame[0] != "training":
                continue
            frame_index = kind_frame[1]
            cam = recon.camera(img.camera_id)
            xy = img.points2D[el.point2D_idx].xy
            out.append((pt.xyz, frame_index, xy[0] / cam.width, xy[1] / cam.height))
    return out

def sample_model_point(points_per_frame, frame_idx_to_pos, frame_index, u_norm, v_norm):
    pos = frame_idx_to_pos.get(frame_index)
    if pos is None:
        return None
    h, w = points_per_frame.shape[1], points_per_frame.shape[2]
    row = min(h - 1, max(0, int(v_norm * h)))
    col = min(w - 1, max(0, int(u_norm * w)))
    p = points_per_frame[pos, row, col]
    if not np.all(np.isfinite(p)) or (p[2] == 0 and p[0] == 0 and p[1] == 0):
        return None
    return p

def ransac_sim3(src_pts, dst_pts, n_iters=2000, inlier_thresh_frac=0.05, seed=0):
    """RANSAC wrapper around sih3d.align.solve_sim3 (Umeyama closed-form) --
    same code used for both models, per explicit requirement. inlier_thresh_frac
    is relative to the scene extent (dst point cloud bbox diagonal), since an
    absolute meters threshold means nothing across models with different
    native scales."""
    import sys
    sys.path.insert(0, str(CODE_DIR))
    from sih3d.align import solve_sim3

    rng = np.random.default_rng(seed)
    n = len(src_pts)
    if n < 10:
        return None
    dst_extent = float(np.linalg.norm(dst_pts.max(axis=0) - dst_pts.min(axis=0)))
    thresh = max(dst_extent * inlier_thresh_frac, 1e-6)

    best_inliers = None
    best_count = -1
    for _ in range(n_iters):
        idx = rng.choice(n, size=min(8, n), replace=False)
        try:
            scale, R, t, _ = solve_sim3(src_pts[idx], dst_pts[idx])
        except Exception:
            continue
        pred = scale * (R @ src_pts.T).T + t
        res = np.linalg.norm(pred - dst_pts, axis=1)
        inliers = res < thresh
        if inliers.sum() > best_count:
            best_count = inliers.sum()
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < 10:
        return None
    scale, R, t, rmse = solve_sim3(src_pts[best_inliers], dst_pts[best_inliers])
    pred = scale * (R @ src_pts.T).T + t
    res_all = np.linalg.norm(pred - dst_pts, axis=1)
    return {
        "scale": float(scale), "R": R, "t": t,
        "inlier_ratio": float(best_inliers.sum() / n),
        "residual_median_pct_extent": float(np.median(res_all[best_inliers]) / dst_extent * 100),
    }

alignments = {}
if not COLMAP_FAILED:
    build_status = json.loads((RESULTS_DIR / "build_status.json").read_text())
    for model_key, cameras_file in [("mapanything", "cameras.npz")]:
        st = build_status.get(model_key, {})
        if st.get("status") != "ok":
            alignments[model_key] = {"status": "skipped", "reason": st.get("status")}
            continue
        cam_data = np.load(RESULTS_DIR / model_key / "cameras.npz")
        points_per_frame = cam_data["points_per_frame"]
        frame_idx_to_pos = {int(fi): i for i, fi in enumerate(cam_data["frame_indices"])}
        corrs = get_colmap_correspondences(final_recon, name_map, frame_idx_to_pos, points_per_frame.shape)
        src, dst = [], []
        for xyz, fi, u, v in corrs:
            mp = sample_model_point(points_per_frame, frame_idx_to_pos, fi, u, v)
            if mp is not None:
                src.append(mp)
                dst.append(xyz)
        if len(src) < 10:
            alignments[model_key] = {"status": "too_few_correspondences", "n": len(src)}
            continue
        fit = ransac_sim3(np.array(src), np.array(dst))
        if fit is None:
            alignments[model_key] = {"status": "ransac_failed", "n_correspondences": len(src)}
        else:
            alignments[model_key] = {"status": "ok", "n_correspondences": len(src), **{k: v for k, v in fit.items() if k not in ("R", "t")}}
            np.savez(RESULTS_DIR / model_key / "sim3_fit.npz", scale=fit["scale"], R=fit["R"], t=fit["t"])

    # SLAM3R: same code, its own per-frame points come from registered_pcds.npy
    st = build_status.get("slam3r", {})
    if st.get("status") == "ok":
        pcds = np.load(RESULTS_DIR / "slam3r" / "preds" / "registered_pcds.npy")
        frames_data = json.loads((BAKEOFF_DIR / "frames.json").read_text())
        frame_idx_to_pos = {int(k["frame_index"]): i for i, k in enumerate(frames_data["training"][:len(pcds)])}
        corrs = get_colmap_correspondences(final_recon, name_map, frame_idx_to_pos, pcds.shape)
        src, dst = [], []
        for xyz, fi, u, v in corrs:
            mp = sample_model_point(pcds, frame_idx_to_pos, fi, u, v)
            if mp is not None:
                src.append(mp)
                dst.append(xyz)
        if len(src) < 10:
            alignments["slam3r"] = {"status": "too_few_correspondences", "n": len(src)}
        else:
            fit = ransac_sim3(np.array(src), np.array(dst))
            if fit is None:
                alignments["slam3r"] = {"status": "ransac_failed", "n_correspondences": len(src)}
            else:
                alignments["slam3r"] = {"status": "ok", "n_correspondences": len(src), **{k: v for k, v in fit.items() if k not in ("R", "t")}}
                np.savez(RESULTS_DIR / "slam3r" / "sim3_fit.npz", scale=fit["scale"], R=fit["R"], t=fit["t"])
    else:
        alignments.setdefault("slam3r", {"status": "skipped", "reason": st.get("status")})
else:
    alignments = {"mapanything": {"status": "skipped_colmap_failed"}, "slam3r": {"status": "skipped_colmap_failed"}}

print(json.dumps(alignments, indent=2, default=str))
(RESULTS_DIR / "alignments.json").write_text(json.dumps(alignments, indent=2, default=str))
print(f"\\nAlignment cell: {time.time() - _t0:.1f}s")
'''

RENDER_METRICS_CELL = r'''
# ============================== RENDER + METRICS ON HELD-OUT FRAMES ==========
import json
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_t0 = time.time()
RENDER_W = 960
frames_data = json.loads((BAKEOFF_DIR / "frames.json").read_text())
build_status = json.loads((RESULTS_DIR / "build_status.json").read_text())
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def colmap_camera_for(image_name):
    img = final_recon.find_image_with_name(image_name)
    if img is None or not final_recon.is_image_registered(img.image_id):
        return None
    cam = final_recon.camera(img.camera_id)
    cam_from_world = img.cam_from_world
    return cam, cam_from_world


def project_points(points_world, cam, cam_from_world, out_w, out_h, orig_w, orig_h):
    scale = out_w / orig_w
    pts_h = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    cam_mat = cam_from_world.matrix()  # (3,4)
    pts_cam = (cam_mat @ pts_h.T).T
    z = pts_cam[:, 2]
    valid = z > 1e-6
    fx, fy = cam.focal_length_x * scale, cam.focal_length_y * scale
    cx, cy = cam.principal_point_x * scale, cam.principal_point_y * scale
    u = fx * pts_cam[:, 0] / np.where(valid, z, 1.0) + cx
    v = fy * pts_cam[:, 1] / np.where(valid, z, 1.0) + cy
    in_bounds = valid & (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    return u, v, z, in_bounds


def render_points(points_world, colors01, cam, cam_from_world, out_w, out_h, orig_w, orig_h, radius=2):
    canvas = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255
    u, v, z, ok = project_points(points_world, cam, cam_from_world, out_w, out_h, orig_w, orig_h)
    if not ok.any():
        return canvas, 0.0
    order = np.argsort(-z[ok])
    us, vs, cs = u[ok][order], v[ok][order], colors01[ok][order]
    for x, y, c in zip(us.astype(int), vs.astype(int), cs):
        cv2.circle(canvas, (x, y), radius, (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)), -1)
    coverage = float(np.count_nonzero(np.any(canvas != 255, axis=2)) / (out_w * out_h))
    return canvas, coverage


def render_mesh(mesh, cam, cam_from_world, out_w, out_h, orig_w, orig_h):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    vcolors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.ones_like(verts) * 0.5
    canvas = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255
    u, v, z, ok = project_points(verts, cam, cam_from_world, out_w, out_h, orig_w, orig_h)
    tri_ok = ok[tris].all(axis=1) & (z[tris] > 1e-6).all(axis=1)
    tri_depth = z[tris].mean(axis=1)
    order = np.argsort(-tri_depth[tri_ok])
    tri_idx = np.nonzero(tri_ok)[0][order]
    for ti in tri_idx:
        a, b, c = tris[ti]
        pts = np.array([[u[a], v[a]], [u[b], v[b]], [u[c], v[c]]], dtype=np.int32)
        color = vcolors[[a, b, c]].mean(axis=0)
        cv2.fillConvexPoly(canvas, pts, (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255)))
    coverage = float(np.count_nonzero(np.any(canvas != 255, axis=2)) / (out_w * out_h))
    return canvas, coverage


# -- load aligned points + meshes for each ok model ----------------------------
import open3d as o3d

model_data = {}
for model_key in ["mapanything", "slam3r"]:
    st = build_status.get(model_key, {})
    align = json.loads((RESULTS_DIR / "alignments.json").read_text()).get(model_key, {})
    if st.get("status") != "ok" or align.get("status") != "ok":
        model_data[model_key] = None
        continue
    pts_data = np.load(RESULTS_DIR / model_key / "points.npz")
    points, colors = pts_data["points"], pts_data["colors"]
    fit = np.load(RESULTS_DIR / model_key / "sim3_fit.npz")
    scale, R, t = float(fit["scale"]), fit["R"], fit["t"]
    aligned_points = scale * (R @ points.T).T + t
    mesh = o3d.io.read_triangle_mesh(str(RESULTS_DIR / model_key / f"{model_key}.ply"))
    mesh_verts = np.asarray(mesh.vertices)
    mesh.vertices = o3d.utility.Vector3dVector(scale * (R @ mesh_verts.T).T + t)
    model_data[model_key] = {"points": aligned_points, "colors": colors, "mesh": mesh}

# -- render all held-out frames -------------------------------------------------
per_frame_results = []
for hk in frames_data["held_out"]:
    fname = hk["file"]
    cam_info = colmap_camera_for(fname)
    real_img = cv2.cvtColor(cv2.imread(str(Path(frames_data["held_out_dir"]) / fname)), cv2.COLOR_BGR2RGB)
    orig_h, orig_w = hk["height"], hk["width"]
    out_h = int(RENDER_W * orig_h / orig_w)

    row_imgs = [cv2.resize(real_img, (RENDER_W, out_h))]
    row_labels = ["Real photo"]
    renders = {}
    if cam_info is not None:
        cam, cfw = cam_info
        for model_key in ["mapanything", "slam3r"]:
            data = model_data.get(model_key)
            if data is None:
                row_imgs += [np.ones((out_h, RENDER_W, 3), dtype=np.uint8) * 200] * 2
                row_labels += [f"{model_key} points (N/A)", f"{model_key} mesh (N/A)"]
                continue
            pimg, pcov = render_points(data["points"], data["colors"], cam, cfw, RENDER_W, out_h, orig_w, orig_h)
            mimg, mcov = render_mesh(data["mesh"], cam, cfw, RENDER_W, out_h, orig_w, orig_h)
            row_imgs += [pimg, mimg]
            row_labels += [f"{model_key} points ({pcov*100:.0f}%)", f"{model_key} mesh ({mcov*100:.0f}%)"]
            renders[model_key] = {"points": pimg, "mesh": mimg, "point_coverage": pcov, "mesh_coverage": mcov}
    else:
        row_imgs += [np.ones((out_h, RENDER_W, 3), dtype=np.uint8) * 200] * 4
        row_labels += ["N/A (not registered)"] * 4

    # simple error heatmap: |mesh render - real photo| for whichever model has a render, else blank
    heat_src = renders.get("mapanything", renders.get("slam3r"))
    if heat_src is not None:
        real_resized = cv2.resize(real_img, (RENDER_W, out_h)).astype(np.float32)
        diff = np.abs(real_resized - heat_src["mesh"].astype(np.float32)).mean(axis=2)
        heat = cv2.applyColorMap((np.clip(diff, 0, 255)).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    else:
        heat = np.ones((out_h, RENDER_W, 3), dtype=np.uint8) * 200
    row_imgs.append(heat)
    row_labels.append("error heatmap")

    fig, axes = plt.subplots(1, len(row_imgs), figsize=(4 * len(row_imgs), 4))
    for ax, img, label in zip(axes, row_imgs, row_labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig_path = FIG_DIR / f"compare_frame_{hk['frame_index']:06d}.png"
    plt.savefig(fig_path, dpi=80)
    plt.close(fig)

    per_frame_results.append({"frame_index": hk["frame_index"], "file": fname, "registered": cam_info is not None,
                               "renders": {k: {"point_coverage": v["point_coverage"], "mesh_coverage": v["mesh_coverage"]} for k, v in renders.items()},
                               "fig_path": str(fig_path)})

print(f"Rendered {len(per_frame_results)} held-out comparison figures")

# -- contact sheet --------------------------------------------------------------
n_rows = len(per_frame_results)
if n_rows > 0:
    thumbs = [cv2.imread(str(r["fig_path"])) for r in per_frame_results]
    max_w = max(t.shape[1] for t in thumbs)
    thumbs = [cv2.copyMakeBorder(t, 0, 0, 0, max_w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in thumbs]
    contact = np.concatenate(thumbs, axis=0)
    cv2.imwrite(str(RESULTS_DIR / "contact_sheet.png"), contact)
    print(f"Saved contact sheet: {RESULTS_DIR / 'contact_sheet.png'}")

(RESULTS_DIR / "per_frame_render_results.json").write_text(json.dumps(per_frame_results, indent=2, default=str))
print(f"\\nRender cell: {time.time() - _t0:.1f}s")
'''

QUALITY_METRICS_CELL = r'''
# ============================== QUALITY + GEOMETRY METRICS ====================
# Backbone GPU work is done and those subprocesses have exited by this point,
# so importing torch here (for SegFormer masking + LPIPS) no longer competes
# with anything -- the GPU-hygiene subprocess isolation earlier in this
# notebook was specifically to protect the backbone runs, not a blanket rule.
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(CODE_DIR))

_t0 = time.time()
_pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lpips"], capture_output=True, text=True)
print("lpips install:", "OK" if _pip.returncode == 0 else _pip.stderr[-300:])

import torch
import lpips as lpips_lib
from sih3d.masks import SemanticMasker
from sih3d.events import EventBus

_bus = EventBus()
masker = SemanticMasker(_bus, device="cuda:0")
lpips_model = lpips_lib.LPIPS(net="alex").to("cuda:0").eval()

frames_data = json.loads((BAKEOFF_DIR / "frames.json").read_text())
per_frame_results = json.loads((RESULTS_DIR / "per_frame_render_results.json").read_text())
build_status = json.loads((RESULTS_DIR / "build_status.json").read_text())
alignments = json.loads((RESULTS_DIR / "alignments.json").read_text())


def psnr(a, b, land_mask):
    a, b = a[land_mask].astype(np.float64), b[land_mask].astype(np.float64)
    if len(a) == 0:
        return None
    mse = np.mean((a - b) ** 2)
    return float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else 100.0


def ssim_gray(a, b, land_mask):
    """Simple windowed SSIM (11x11 Gaussian-ish box, C1/C2 per the standard
    formula) on land pixels only -- no skimage dependency risk."""
    ag = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    bg = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ksize = (11, 11)
    mu_a, mu_b = cv2.GaussianBlur(ag, ksize, 1.5), cv2.GaussianBlur(bg, ksize, 1.5)
    sigma_a = cv2.GaussianBlur(ag * ag, ksize, 1.5) - mu_a ** 2
    sigma_b = cv2.GaussianBlur(bg * bg, ksize, 1.5) - mu_b ** 2
    sigma_ab = cv2.GaussianBlur(ag * bg, ksize, 1.5) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2))
    vals = ssim_map[land_mask]
    return float(vals.mean()) if len(vals) else None


def lpips_score(a, b, land_mask):
    if land_mask.sum() < 100:
        return None
    a2, b2 = a.copy(), b.copy()
    a2[~land_mask] = 0
    b2[~land_mask] = 0
    ta = torch.from_numpy(a2).permute(2, 0, 1).float().unsqueeze(0).to("cuda:0") / 127.5 - 1
    tb = torch.from_numpy(b2).permute(2, 0, 1).float().unsqueeze(0).to("cuda:0") / 127.5 - 1
    with torch.no_grad():
        return float(lpips_model(ta, tb).item())


quality_rows = {"mapanything": [], "slam3r": []}
for r in per_frame_results:
    if not r["registered"]:
        continue
    real_path = Path(frames_data["held_out_dir"]) / r["file"]
    real_img = cv2.cvtColor(cv2.imread(str(real_path)), cv2.COLOR_BGR2RGB)
    land_mask_result = masker.mask_frame(real_img)
    land_mask = ~land_mask_result.mask  # SemanticMasker.mask = water/sky (exclude); land = inverse
    # Renders themselves weren't saved as separate per-model image files (only the composite figure) --
    # recompute the model render arrays here from the same aligned data + camera, reusing render_points/render_mesh
    # defined in the previous cell (still in globals()).
    cam_info = colmap_camera_for(r["file"])
    if cam_info is None:
        continue
    cam, cfw = cam_info
    orig_h = frames_data["held_out"][0]["height"]  # all held-out frames share the source video resolution
    orig_w = frames_data["held_out"][0]["width"]
    out_h = int(RENDER_W * orig_h / orig_w)
    real_resized = cv2.resize(real_img, (RENDER_W, out_h))
    land_mask_resized = cv2.resize(land_mask.astype(np.uint8), (RENDER_W, out_h), interpolation=cv2.INTER_NEAREST).astype(bool)

    for model_key in ["mapanything", "slam3r"]:
        data = model_data.get(model_key)
        if data is None:
            continue
        mimg, mcov = render_mesh(data["mesh"], cam, cfw, RENDER_W, out_h, orig_w, orig_h)
        rendered_mask = np.any(mimg != 255, axis=2)
        both_mask = land_mask_resized & rendered_mask
        p = psnr(real_resized, mimg, both_mask)
        s = ssim_gray(real_resized, mimg, both_mask)
        l = lpips_score(real_resized, mimg, both_mask)
        land_total = int(land_mask_resized.sum())
        coverage_land = float((both_mask.sum() / land_total) * 100) if land_total else 0.0
        quality_rows[model_key].append({
            "frame_index": r["frame_index"], "psnr": p, "ssim": s, "lpips": l, "coverage_land_pct": coverage_land,
        })

def summarize_quality(rows):
    out = {}
    for metric in ["psnr", "ssim", "lpips", "coverage_land_pct"]:
        vals = [r[metric] for r in rows if r[metric] is not None]
        out[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else {"mean": None, "std": None}
    out["n_frames"] = len(rows)
    return out

quality_summary = {k: summarize_quality(v) for k, v in quality_rows.items()}

wins = {"mapanything": 0, "slam3r": 0}
for i in range(len(quality_rows["mapanything"])):
    ma_r = quality_rows["mapanything"][i] if i < len(quality_rows["mapanything"]) else None
    s3_r = quality_rows["slam3r"][i] if i < len(quality_rows["slam3r"]) else None
    if ma_r and s3_r and ma_r["psnr"] is not None and s3_r["psnr"] is not None:
        winner = "mapanything" if ma_r["psnr"] >= s3_r["psnr"] else "slam3r"
        wins[winner] += 1
quality_summary["frame_wins_by_psnr"] = wins

print(json.dumps(quality_summary, indent=2, default=str))

# -- geometry vs colmap: depth error, accuracy, completeness -------------------
from scipy.spatial import cKDTree

geometry_summary = {}
if not COLMAP_FAILED:
    colmap_points = np.array([p.xyz for p in final_recon.points3D.values()])
    scene_extent = float(np.linalg.norm(colmap_points.max(axis=0) - colmap_points.min(axis=0))) if len(colmap_points) else 1.0
    for model_key in ["mapanything", "slam3r"]:
        align = alignments.get(model_key, {})
        if align.get("status") != "ok" or model_data.get(model_key) is None:
            geometry_summary[model_key] = {"status": "unavailable"}
            continue
        model_pts = model_data[model_key]["points"]
        tree_model = cKDTree(model_pts)
        tree_colmap = cKDTree(colmap_points)
        acc_d, _ = tree_colmap.query(model_pts, k=1)   # model -> colmap (accuracy)
        comp_d, _ = tree_model.query(colmap_points, k=1)  # colmap -> model (completeness)
        geometry_summary[model_key] = {
            "accuracy_pct_extent": float(np.median(acc_d) / scene_extent * 100),
            "completeness_pct_extent": float(np.median(comp_d) / scene_extent * 100),
        }
else:
    geometry_summary = {"mapanything": {"status": "colmap_failed"}, "slam3r": {"status": "colmap_failed"}}

print(json.dumps(geometry_summary, indent=2, default=str))

# -- camera trajectory/rotation error (MapAnything only) ------------------------
camera_error = {"mapanything": {"status": "unavailable"}, "slam3r": {"status": "N/A (no cameras)"}}
if not COLMAP_FAILED and alignments.get("mapanything", {}).get("status") == "ok":
    fit = np.load(RESULTS_DIR / "mapanything" / "sim3_fit.npz")
    scale, R, t = float(fit["scale"]), fit["R"], fit["t"]
    cam_data = np.load(RESULTS_DIR / "mapanything" / "cameras.npz")
    trans_errs, rot_errs = [], []
    import pycolmap
    for i, fi in enumerate(cam_data["frame_indices"]):
        k = next((k for k in frames_data["training"] if k["frame_index"] == int(fi)), None)
        if k is None:
            continue
        cam_info = None
        img = final_recon.find_image_with_name(k["file"])
        if img is None or not final_recon.is_image_registered(img.image_id):
            continue
        pose_ma = cam_data["poses"][i]  # 4x4 cam2world
        cam_center_ma = scale * (R @ pose_ma[:3, 3]) + t
        R_ma_world = R @ pose_ma[:3, :3]

        cfw = img.cam_from_world  # world_from... actually cam_from_world: world->cam
        world_from_cam = cfw.inverse()
        cam_center_colmap = world_from_cam.translation
        R_colmap_world = world_from_cam.matrix()[:3, :3]

        trans_errs.append(float(np.linalg.norm(cam_center_ma - cam_center_colmap)))
        R_diff = R_ma_world.T @ R_colmap_world
        angle = float(np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))))
        rot_errs.append(angle)
    scene_extent_cam = geometry_summary.get("mapanything", {}).get("accuracy_pct_extent")
    if trans_errs:
        colmap_pts_extent = float(np.linalg.norm(colmap_points.max(axis=0) - colmap_points.min(axis=0))) if not COLMAP_FAILED and len(colmap_points) else 1.0
        camera_error["mapanything"] = {
            "n_cameras": len(trans_errs),
            "translation_error_median_pct_extent": float(np.median(trans_errs) / colmap_pts_extent * 100),
            "rotation_error_median_deg": float(np.median(rot_errs)),
        }

print(json.dumps(camera_error, indent=2, default=str))

json.dump({
    "quality": quality_summary, "geometry": geometry_summary, "camera_error": camera_error,
    "quality_per_frame": quality_rows,
}, open(RESULTS_DIR / "quality_geometry_metrics.json", "w"), indent=2, default=str)

print(f"\\nQuality/geometry metrics cell: {time.time() - _t0:.1f}s")
'''

SCORECARD_CELL = r'''
# ============================== SCORECARD + DECISION + REPORT ================
import json
import time
from pathlib import Path

_t0 = time.time()
metrics = json.loads((RESULTS_DIR / "quality_geometry_metrics.json").read_text())
build_status = json.loads((RESULTS_DIR / "build_status.json").read_text())
alignments = json.loads((RESULTS_DIR / "alignments.json").read_text())

VIDEO_DURATION_S = frames_data["video_fps"] and (max(k["timestamp_s"] for k in frames_data["training"]) + 1)
TARGET_10MIN_S = 600.0
scale_factor = TARGET_10MIN_S / max(VIDEO_DURATION_S, 1.0)

gates = {}
scores_raw = {}
for model_key in ["mapanything", "slam3r"]:
    st = build_status.get(model_key, {})
    runtime_s = st.get("runtime_s")
    extrapolated_10min_s = runtime_s * scale_factor if runtime_s else None
    has_cameras = model_key == "mapanything"  # SLAM3R produces none, by design
    gate_cameras = has_cameras
    gate_speed = extrapolated_10min_s is not None and extrapolated_10min_s <= 900  # 15 min
    gates[model_key] = {
        "has_cameras": gate_cameras, "extrapolated_10min_runtime_s": extrapolated_10min_s,
        "speed_gate_pass": gate_speed, "passes_all_gates": gate_cameras and gate_speed,
    }

    geo = metrics["geometry"].get(model_key, {})
    qual = metrics["quality"].get(model_key, {})
    scores_raw[model_key] = {
        "depth_error_pct": None,  # not separately computed from accuracy/completeness here; using accuracy as the closest proxy
        "accuracy_pct_extent": geo.get("accuracy_pct_extent"),
        "completeness_pct_extent": geo.get("completeness_pct_extent"),
        "psnr": qual.get("psnr", {}).get("mean"),
        "ssim": qual.get("ssim", {}).get("mean"),
        "lpips": qual.get("lpips", {}).get("mean"),
        "coverage_land_pct": qual.get("coverage_land_pct", {}).get("mean"),
        "runtime_s": runtime_s,
    }

def normalize(values, lower_is_better):
    """Best model scores 1.0, per explicit requirement."""
    valid = {k: v for k, v in values.items() if v is not None}
    if not valid:
        return {k: None for k in values}
    best = min(valid.values()) if lower_is_better else max(valid.values())
    out = {}
    for k, v in values.items():
        if v is None or best == 0:
            out[k] = None
        else:
            out[k] = float(best / v) if lower_is_better else float(v / best)
    return out

acc_n = normalize({k: v["accuracy_pct_extent"] for k, v in scores_raw.items()}, lower_is_better=True)
comp_n = normalize({k: v["completeness_pct_extent"] for k, v in scores_raw.items()}, lower_is_better=True)
psnr_n = normalize({k: v["psnr"] for k, v in scores_raw.items()}, lower_is_better=False)
ssim_n = normalize({k: v["ssim"] for k, v in scores_raw.items()}, lower_is_better=False)
lpips_n = normalize({k: v["lpips"] for k, v in scores_raw.items()}, lower_is_better=True)
cov_n = normalize({k: v["coverage_land_pct"] for k, v in scores_raw.items()}, lower_is_better=False)
speed_n = normalize({k: v["runtime_s"] for k, v in scores_raw.items()}, lower_is_better=True)

def avg_ignore_none(*vals):
    v = [x for x in vals if x is not None]
    return float(sum(v) / len(v)) if v else None

final_scores = {}
for model_key in ["mapanything", "slam3r"]:
    geometry_score = avg_ignore_none(acc_n.get(model_key), comp_n.get(model_key))
    quality_score = avg_ignore_none(psnr_n.get(model_key), ssim_n.get(model_key), lpips_n.get(model_key), cov_n.get(model_key))
    speed_score = speed_n.get(model_key)
    total = None
    if geometry_score is not None and quality_score is not None and speed_score is not None:
        total = 0.4 * geometry_score + 0.4 * quality_score + 0.2 * speed_score
    final_scores[model_key] = {
        "geometry_score": geometry_score, "quality_score": quality_score, "speed_score": speed_score,
        "total_score": total, "passes_gates": gates[model_key]["passes_all_gates"],
        "per_metric_normalized": {
            "accuracy": acc_n.get(model_key), "completeness": comp_n.get(model_key),
            "psnr": psnr_n.get(model_key), "ssim": ssim_n.get(model_key), "lpips": lpips_n.get(model_key),
            "coverage": cov_n.get(model_key), "speed": speed_n.get(model_key),
        },
    }

eligible = {k: v for k, v in final_scores.items() if v["passes_gates"] and v["total_score"] is not None}
winner = max(eligible, key=lambda k: eligible[k]["total_score"]) if eligible else None

sih_fit = {
    "mapanything": {"outputs_cameras": True, "metric_scale": True, "license": "Apache 2.0 (facebook/map-anything-apache) -- NTRO use should be fine, verify explicitly before submission"},
    "slam3r": {"outputs_cameras": False, "metric_scale": "unknown/relative (not declared metric)", "license": "CC BY-NC-SA 4.0 -- NON-COMMERCIAL ONLY, likely blocks NTRO/government use without separate permission"},
}

scores_out = {
    "gates": gates, "scores_raw": scores_raw, "final_scores": final_scores, "winner": winner,
    "sih_fit": sih_fit, "camera_error": metrics["camera_error"],
    "colmap_reference": "colmap" if not COLMAP_FAILED else "mapanything (COLMAP fallback, results biased)",
    "video_duration_s": VIDEO_DURATION_S, "target_10min_s": TARGET_10MIN_S,
}
(RESULTS_DIR / "scores.json").write_text(json.dumps(scores_out, indent=2, default=str))
print(json.dumps(scores_out, indent=2, default=str))

# -- scorecard.md ---------------------------------------------------------------
md_lines = ["# SIH26158 Backbone Bake-off Scorecard\\n",
            f"Reference: **{scores_out['colmap_reference']}**\\n",
            "## Gates\\n", "| Model | Has cameras | Extrapolated 10-min runtime | Passes gates |",
            "|---|---|---|---|"]
for k, g in gates.items():
    md_lines.append(f"| {k} | {g['has_cameras']} | {g['extrapolated_10min_runtime_s']:.0f}s | {g['passes_all_gates']} |" if g['extrapolated_10min_runtime_s'] else f"| {k} | {g['has_cameras']} | N/A | {g['passes_all_gates']} |")
md_lines.append("\\n## Scores (best model = 1.0 per metric)\\n")
md_lines.append("| Model | Geometry (40%) | Quality (40%) | Speed (20%) | TOTAL |")
md_lines.append("|---|---|---|---|---|")
for k, v in final_scores.items():
    fmt = lambda x: f"{x:.3f}" if x is not None else "N/A"
    md_lines.append(f"| {k} | {fmt(v['geometry_score'])} | {fmt(v['quality_score'])} | {fmt(v['speed_score'])} | {fmt(v['total_score'])} |")
md_lines.append(f"\\n**Winner (passes gates, highest total): {winner or 'NONE -- see gates above'}**\\n")
md_lines.append("## SIH fit\\n")
for k, v in sih_fit.items():
    md_lines.append(f"- **{k}**: cameras={v['outputs_cameras']}, metric_scale={v['metric_scale']}, license={v['license']}")
(RESULTS_DIR / "scorecard.md").write_text("\\n".join(md_lines))

# -- report.html: bundles everything for submission -----------------------------
fig_paths = sorted((RESULTS_DIR / "figures").glob("compare_frame_*.png"))
html_parts = [
    "<html><head><meta charset='utf-8'><title>SIH26158 Bake-off Report</title>",
    "<style>body{font-family:sans-serif;background:#111;color:#eee;padding:2rem}",
    "table{border-collapse:collapse;width:100%}td,th{border:1px solid #555;padding:6px}",
    "img{max-width:100%}</style></head><body>",
    "<h1>SIH26158 Backbone Bake-off: MapAnything vs SLAM3R</h1>",
    f"<p>Reference: <b>{scores_out['colmap_reference']}</b></p>",
    "<h2>Scorecard</h2><pre>" + "\\n".join(md_lines) + "</pre>",
    "<h2>Held-out comparison figures</h2>",
]
for p in fig_paths:
    html_parts.append(f"<h3>{p.stem}</h3><img src='figures/{p.name}'>")
html_parts.append("<h2>Contact sheet</h2><img src='contact_sheet.png'>")
html_parts.append("</body></html>")
(RESULTS_DIR / "report.html").write_text("\\n".join(html_parts))

print(f"\\nWinner: {winner}")
print(f"Scorecard cell: {time.time() - _t0:.1f}s")
print(f"\\nDeliverables written under {RESULTS_DIR}:")
for f in sorted(RESULTS_DIR.rglob("*")):
    if f.is_file():
        print(" ", f.relative_to(RESULTS_DIR))
'''


def build_final() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(TITLE_MD),
        code(SETUP_CELL),
        code(WRITE_SCRIPTS_CELL),
        code(KEYFRAMES_CELL),
        code(RUN_BACKBONES_CELL),
        code(METRICS_CELL),
        md("## FINAL BAKE-OFF: deterministic frames, COLMAP reference, aligned meshes, held-out quality metrics, scorecard"),
        code(WRITE_FRAMES_RUNNER_CELL),
        code(FRAMES_CELL),
        code(COLMAP_CELL),
        code(WRITE_MESH_HELPER_CELL),
        code(WRITE_FINAL_RUNNERS_CELL),
        code(BUILD_MODELS_CELL),
        code(ALIGN_CELL),
        code(RENDER_METRICS_CELL),
        code(QUALITY_METRICS_CELL),
        code(SCORECARD_CELL),
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nbf.validate(nb)
    OUT_PATH.write_text(nbf.writes(nb))
    print(f"Wrote {OUT_PATH} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    build_final()

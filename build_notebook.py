#!/usr/bin/env python3
"""Generates sih26158_geoff3d.ipynb.

The sih3d/ package is NOT embedded in the notebook — the Setup cell git
clones/pulls it from https://github.com/anuragratan1/sih26158-geoff3d at
run time, so a code fix is `git push` + re-running the Setup and Launch
cells, never a full notebook re-upload. This script only assembles the
driver cells (config/setup/launch/results); re-run it whenever those
change, not for ordinary sih3d/ code changes (those just need a git push).

Usage:
    python3 build_notebook.py

Validates the result with nbformat before writing it out.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
OUT_PATH = ROOT / "sih26158_geoff3d.ipynb"


def code(src: str):
    return nbf.v4.new_code_cell(src.strip("\n") + "\n")


def md(src: str):
    return nbf.v4.new_markdown_cell(src.strip("\n") + "\n")


# ---------------------------------------------------------------------------
# Cell content
# ---------------------------------------------------------------------------

TITLE_MD = """
# SIH26158 — Single-Pass Drone Video → Georeferenced 3D Model

Converts a single-pass drone video + GPS/flight metadata into a georeferenced,
metric 3D model, with lightweight terminal progress bars and final results.

**Run All** to execute end to end. See `RUN_ON_KAGGLE.md` (in the repo this
notebook was generated from) for upload/dataset/settings instructions.
"""

CONFIG_CELL = '''
# ============================== CONFIG ======================================
# MODE: "QUICK" = first ~90s / ~120 keyframes, target <5 min end-to-end on
#       2x T4 (excluding first-time installs). "FULL" = whole video, target
#       <15 min for a 10-min video.
MODE = "QUICK"

# BACKBONE: "A" (GeoFF3D+SLRF — NOT VIABLE, no published checkpoint exists;
# selecting it raises a clear error rather than pretending to work — see
# PHASE0_NOTES.md), "B" (Pi3X, loaded directly, no SLRF), "C" (MapAnything
# Apache, recommended default — safest install, no torch/CUDA pin).
BACKBONE = "C"

# PRIOR_MODE: "RGB" | "C" (+intrinsics) | "P" (+poses) | "CP" (+both) | "AUTO"
# (resolves to CP when a fine-tuned checkpoint is detected + GPS is present,
# else C; forced to RGB automatically with no GPS regardless of this setting
# — pose/intrinsics priors without real GPS would be fabricated garbage).
PRIOR_MODE = "AUTO"

# Use a fine-tuned UAVFF3D checkpoint (github.com/yanxian-ll/UAVFF3D) if one
# is auto-detected in /kaggle/input (see RUN_ON_KAGGLE.md for how to attach
# it as a dataset). Falls back to the stock pretrained backbone otherwise.
USE_FINETUNED_CHECKPOINT = True

# Reverted back to False: enabling this for the first time with real
# multi-chunk data produced a visibly WORSE point cloud (a dense correctly-
# placed strip surrounded by a huge sparse scattered halo — the signature
# of two chunks merged without a correct relative transform between them).
# That's a real correctness bug in the DUAL_GPU alignment path, not
# something to leave on speculatively — needs actual debugging before
# re-enabling. Off by default: data-parallel backbone across both GPUs
# (even chunks on cuda:0, odd on cuda:1). Ignored with a warning if fewer
# than 2 GPUs are detected.
DUAL_GPU = False

# Off by default per the task spec: a viser server with a public share URL
# for a full-resolution external live-3D view. Never blocks/crashes the
# pipeline if it fails to start (falls back to cloudflared, then just skips).
LIVE_3D_EXTERNAL = False

INPUT_ROOT = "/kaggle/input"
OUTPUT_DIR = "/kaggle/working/outputs"
CACHE_DIR = "/kaggle/working/cache"

print(f"MODE={MODE}  BACKBONE={BACKBONE}  PRIOR_MODE={PRIOR_MODE}  USE_FINETUNED_CHECKPOINT={USE_FINETUNED_CHECKPOINT}  DUAL_GPU={DUAL_GPU}")
'''

SETUP_CELL = '''
# ============================== SETUP: sync code, env check, install, detect inputs ====
# One cell, top to bottom: pull the sih3d/ package from GitHub -> environment
# checks -> install only what's missing -> detect the video/telemetry/
# checkpoints already attached under /kaggle/input -> ready for the Launch
# cell below. Kaggle's image already has torch preinstalled — nothing here
# reinstalls/pins it (GeoFF3D's own pyproject pins torch==2.5.0, exactly the
# kind of forced-reinstall the task spec says to avoid; see PHASE0_NOTES.md
# section 4).
import importlib
import subprocess
import sys
import time
from pathlib import Path

_t0 = time.time()

# -- 0. sync code from GitHub -------------------------------------------
# The sih3d/ package lives in git, not embedded in this notebook — a code
# fix is a `git push` on the source side, then re-running just THIS cell
# (which re-pulls) and the Launch cell below, never a full notebook
# re-upload or a Setup/install rerun. `--ff-only` refuses to silently
# overwrite local edits made directly in CODE_DIR (e.g. via a scratch
# debugging cell) with a fast-forward that would discard them.
REPO_URL = "https://github.com/anuragratan1/sih26158-geoff3d.git"
CODE_DIR = Path("/kaggle/working/sih26158-geoff3d")


def _fresh_clone():
    import shutil

    if CODE_DIR.exists():
        shutil.rmtree(CODE_DIR)
    _c = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(CODE_DIR)], capture_output=True, text=True, timeout=180)
    if _c.returncode != 0:
        raise RuntimeError(f"git clone of {REPO_URL} failed: {_c.stderr.strip()[-500:]}")
    print(f"Code: cloned into {CODE_DIR}")


if CODE_DIR.exists():
    # --ff-only can fail (e.g. local edits, a stale/shallow history that
    # can't fast-forward) — if so, don't silently keep running stale code:
    # blow the checkout away and re-clone fresh. A silent stale checkout
    # is exactly how a pushed fix can appear to "not apply" after a pull.
    _sync = subprocess.run(["git", "-C", str(CODE_DIR), "pull", "--ff-only"], capture_output=True, text=True, timeout=60)
    if _sync.returncode == 0:
        print(f"Code: pulled latest into {CODE_DIR}\\n{_sync.stdout.strip()}")
    else:
        print(f"Code: pull failed ({_sync.stderr.strip()[-300:]}) — re-cloning fresh instead of using a possibly-stale checkout")
        _fresh_clone()
else:
    _fresh_clone()
sys.path.insert(0, str(CODE_DIR))

# -- 1. environment checks ---------------------------------------------------

def _check_internet(timeout_s: float = 5.0) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=timeout_s)
        return True
    except Exception:
        return False

def _check_gpu() -> tuple[bool, str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            return True, out.stdout.strip()
    except Exception as e:
        return False, str(e)
    return False, "nvidia-smi returned no GPUs"

_internet_ok = _check_internet()
_gpu_ok, _gpu_info = _check_gpu()
print(f"Internet: {'ON' if _internet_ok else 'OFF'}")
print(f"GPU(s):\\n{_gpu_info if _gpu_ok else '(none detected)'}")

if not _internet_ok:
    raise RuntimeError(
        "Internet is OFF for this notebook session. Enable it under "
        "Notebook Settings -> Internet -> On, then Run All again. "
        "Model weights (MapAnything) and pip installs need it."
    )
if not _gpu_ok:
    print("WARNING: no GPU detected. The pipeline will still run on CPU as a "
          "last-resort fallback, but this will be VERY slow. Enable a GPU "
          "under Notebook Settings -> Accelerator.")

# -- 2. install ---------------------------------------------------------------

from sih3d.io_detect import find_cache_dir  # noqa: E402

# Everything that downloads anything slow — pip wheels, HuggingFace weights
# (MapAnything), torch.hub weights (dinov2 backbone), ultralytics checkpoints
# — is pointed at CACHE_DIR (attached read-only input dataset if one exists
# from a previous run's publish, else the writable working dir this run
# will populate). Setting these env vars BEFORE any package that reads them
# is imported is required — huggingface_hub/torch both read HF_HOME/
# TORCH_HOME once at import time, not per-download. Without this, the
# previous "publish /kaggle/working/cache/ as a dataset" instructions
# published an empty folder: nothing was ever writing into it.
import os

_cache_ds = find_cache_dir(Path(INPUT_ROOT))
_cache_root = Path(_cache_ds) if _cache_ds is not None else Path(CACHE_DIR)
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_cache_root / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_cache_root / "torch"))
os.environ.setdefault("PIP_CACHE_DIR", str(_cache_root / "pip"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_cache_root / "ultralytics"))
_pip_extra = ["--find-links", str(_cache_ds)] if _cache_ds is not None else []
if _cache_ds is not None:
    print(f"\\nFound a cache dataset at {_cache_ds} — HF/torch/pip/YOLO caches point at it directly, nothing should re-download.")
else:
    print(f"\\nNo cache dataset attached — this run's downloads will land under {CACHE_DIR}. "
          f"Publish that folder as a Kaggle dataset after this run finishes and attach it as an "
          f"input next time to skip re-downloading (see the note below the Setup cell).")

def _try_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False

def _pip_install(spec: str, extra_args: list[str] | None = None, label: str | None = None) -> None:
    label = label or spec
    t0 = time.time()
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + _pip_extra + (extra_args or []) + [spec]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        elapsed = time.time() - t0
        if result.returncode != 0:
            _tqdm.write(f"  [FAILED, {elapsed:.0f}s] {label}: {result.stderr[-500:]}")
        else:
            _tqdm.write(f"  [OK, {elapsed:.0f}s] {label}")
    except Exception as e:
        _tqdm.write(f"  [FAILED, {time.time()-t0:.0f}s] {label}: {e}")

# One named step per package/verification, run under a single tqdm bar so
# the whole install phase shows %, elapsed, ETA, and time-per-step instead
# of just scrolling text with no sense of how far along it is.
from tqdm import tqdm as _tqdm

def _step_simple_deps() -> None:
    # Core numeric/vision/geo stack (no torch dependency in any of these).
    # No live widget/dashboard dependencies: they can monopolize Kaggle
    # vCPUs serializing previews while frame extraction is running.
    for _mod, _pkg in [
        # xatlas removed: mesh.py's texture baking is now vectorized
        # per-vertex color sampling (bake_vertex_colors_from_keyframes),
        # no UV atlas / xatlas dependency needed at all.
        ("scipy", "scipy"), ("PIL", "pillow"),
        ("laspy", "laspy"), ("rasterio", "rasterio"), ("pyproj", "pyproj"),
        ("trimesh", "trimesh"), ("open3d", "open3d"), ("pymavlink", "pymavlink"),
        ("huggingface_hub", "huggingface_hub"), ("safetensors", "safetensors"),
        ("psutil", "psutil"),  # CPU utilization tracking, alongside gpu_monitor's GPU tracking
    ]:
        if not _try_import(_mod):
            _pip_install(_pkg)
        else:
            _tqdm.write(f"  [already present] {_pkg}")

def _step_torchcodec() -> None:
    # GPU video decode. Both backends are verified by decode.py with real
    # frames; successful import alone is never treated as proof of CUDA
    # decoding.
    if _try_import("torchcodec"):
        _tqdm.write("  [already present] torchcodec (will only be trusted if decode.py's real GPU-throughput check passes)")
    else:
        _pip_install("torchcodec", extra_args=["--extra-index-url", "https://download.pytorch.org/whl/cu121"], label="torchcodec CUDA fallback")

def _step_pynvvideocodec() -> None:
    if _try_import("PyNvVideoCodec"):
        _tqdm.write("  [already present] PyNvVideoCodec")
    else:
        # Pinned + --only-binary: PyPI's 0.1.x releases are broken sdists
        # (their own setup.py reports version 0.0.0, which pip discards as
        # inconsistent) that a resolver can fall through to if it decides
        # no compatible wheel exists for some reason. Forcing wheel-only
        # turns that into the loud [FAILED] line above instead of a
        # silent, unexplained ModuleNotFoundError two steps later inside
        # decode.py.
        _pip_install("PyNvVideoCodec>=2.0,<3", extra_args=["--only-binary=:all:"], label="PyNvVideoCodec (sequential NVDEC primary)")
        if not _try_import("PyNvVideoCodec"):
            _tqdm.write("  [WARNING] PyNvVideoCodec is not importable; decode.py will use verified torchcodec CUDA fallback")

def _step_ultralytics() -> None:
    # Dynamic-object masking (best-effort; masks.py degrades to all-static).
    if not _try_import("ultralytics"):
        _pip_install("ultralytics", label="ultralytics (YOLO-seg masking)")
    else:
        _tqdm.write("  [already present] ultralytics")

def _step_transformers() -> None:
    # SegFormer semantic masking (water/sky exclusion before fusion —
    # best-effort; masks.py's SemanticMasker degrades to no exclusion).
    if not _try_import("transformers"):
        _pip_install("transformers", label="transformers (SegFormer water/sky masking)")
    else:
        _tqdm.write("  [already present] transformers")

def _step_pyrender() -> None:
    # GPU-accelerated offscreen mesh/point-cloud rendering for the 3D
    # Showcase cell, via EGL — not Vulkan (Open3D's rendering.OffscreenRenderer
    # needs Vulkan, which errored out on a real Kaggle GPU session with
    # "Failed to load vulkan library!"). EGL is the standard headless-GPU-
    # rendering path used across ML/robotics tooling on cloud GPU boxes
    # (SMPL/Habitat/MuJoCo-adjacent code all use this exact pattern) and
    # needs no display server or Vulkan loader, just the NVIDIA driver's
    # existing EGL library. Best-effort: stage_views.py falls back to a
    # flat-shaded matplotlib render if this isn't importable/working.
    if not _try_import("pyrender"):
        # Two packages need two separate pip specs — "pyrender pyopengl" as
        # ONE string makes pip try to install a package literally named
        # "pyrender pyopengl" (with a space), which doesn't exist and fails
        # in under a second. That's exactly the bug that made every run
        # report "already present: False" and never actually install
        # anything, no matter how many times Setup ran.
        _pip_install("pyrender", label="pyrender (GPU offscreen render via EGL)")
        _pip_install("pyopengl", label="pyopengl (pyrender dependency)")
    else:
        _tqdm.write("  [already present] pyrender")

def _step_assimp() -> None:
    # export.py's mesh.fbx output shells out to the `assimp` CLI
    # (best-effort per the task spec) — not a pip package, needs the
    # system binary. Without this it's silently "skipped" every run with
    # no way to fix it short of the user manually apt-get-ing on Kaggle.
    import shutil as _shutil

    if _shutil.which("assimp") is not None:
        _tqdm.write("  [already present] assimp CLI")
        return
    _r = subprocess.run(
        ["apt-get", "install", "-y", "assimp-utils"], capture_output=True, text=True, timeout=120,
    )
    if _r.returncode == 0 and _shutil.which("assimp") is not None:
        _tqdm.write("  [OK] installed assimp-utils")
    else:
        _tqdm.write(f"  [FAILED] assimp-utils install (mesh.fbx export will be skipped): {_r.stderr.strip()[-300:]}")

def _step_mapanything() -> None:
    # MapAnything (Backbone C, the default) — a PLAIN pip install, no
    # --no-deps and no hand-picked extra-deps list: PHASE0_NOTES.md
    # confirmed its own pyproject has no torch/CUDA pin, so letting pip
    # resolve its declared deps itself (rather than guessing which ones it
    # needs) is both simpler and more correct than maintaining a
    # hand-picked list that can drift from the upstream pyproject.
    if not _try_import("mapanything"):
        _pip_install("git+https://github.com/facebookresearch/map-anything.git", label="mapanything (Backbone C)")
    else:
        _tqdm.write("  [already present] mapanything")

def _step_verify_mapanything() -> None:
    # Verify MapAnything actually works — import AND construct the real
    # model — before the pipeline starts, so a broken install fails here
    # with a clear message instead of deep inside the first chunk's
    # geometric_reconstruction stage. This also front-loads the
    # pretrained-weights download the pipeline needs anyway (the slowest
    # single step here — a multi-GB download — so it isn't wasted work).
    try:
        from mapanything.models import MapAnything as _MapAnythingCheck
        _mapanything_model_check = _MapAnythingCheck.from_pretrained("facebook/map-anything-apache")
        del _mapanything_model_check
        _tqdm.write("  [OK] mapanything imports and facebook/map-anything-apache loads")
    except Exception as e:
        _tqdm.write(f"  [FAILED] MapAnything verification: {type(e).__name__}: {e}")
        _tqdm.write("  The pipeline will hit this same error in geometric_reconstruction; fix the install above and Run All again.")

_install_steps = [
    ("core numeric/vision/geo stack", _step_simple_deps),
    ("torchcodec (GPU decode)", _step_torchcodec),
    ("PyNvVideoCodec (GPU decode)", _step_pynvvideocodec),
    ("ultralytics (masking)", _step_ultralytics),
    ("transformers (semantic masking)", _step_transformers),
    ("pyrender (3D showcase render)", _step_pyrender),
    ("assimp-utils (mesh.fbx export)", _step_assimp),
    ("mapanything (backbone)", _step_mapanything),
    ("verify mapanything + download weights", _step_verify_mapanything),
]
print()
for _label, _fn in _tqdm(_install_steps, desc="Installing dependencies", unit="step"):
    _step_t0 = time.time()
    _fn()
    _tqdm.write(f"  -> {_label}: {time.time() - _step_t0:.1f}s")

print(f"\\nInstalls done in {time.time()-_t0:.1f}s. If anything above FAILED, the "
      f"corresponding pipeline stage will log a fallback rather than crash — "
      f"check report.json after the run.")

# -- 3. detect inputs ---------------------------------------------------------

import sih3d.io_detect as io_detect
import sih3d.telemetry as telemetry_mod
importlib.reload(io_detect)
importlib.reload(telemetry_mod)

detected = io_detect.detect_all(Path(INPUT_ROOT))
if detected.video is None:
    raise RuntimeError(
        f"No video file found under {INPUT_ROOT}. Attach a dataset containing "
        f"a drone video (.mp4/.mov/.mkv/.avi/.m4v/.ts) — see RUN_ON_KAGGLE.md."
    )

print(f"\\nVideo: {detected.video.path.name} "
      f"({detected.video.width}x{detected.video.height}, {detected.video.fps:.1f}fps, "
      f"{detected.video.duration_s:.0f}s)")
print(f"Telemetry: {detected.telemetry_path or '(none found)'} (kind={detected.telemetry_kind})")
print(f"Intrinsics: {detected.intrinsics_path or '(none — will estimate)'}")
print(f"Fine-tuned checkpoints detected: {detected.checkpoints or '(none)'}")
for w in detected.warnings:
    print(f"  note: {w}")

telemetry = None
if detected.telemetry_path is not None:
    telemetry = telemetry_mod.parse_telemetry(detected.telemetry_path, detected.telemetry_kind)
    print(f"Parsed telemetry: {len(telemetry)} samples ({telemetry.kind}); notes: {telemetry.notes}")
elif detected.telemetry_kind == "embedded":
    telemetry = telemetry_mod.extract_embedded_telemetry(detected.video.path)
    print(f"Parsed embedded telemetry: {len(telemetry)} samples; notes: {telemetry.notes}")

if telemetry is None or len(telemetry) == 0:
    print("\\nNO GPS TELEMETRY FOUND. Continuing RGB-only: outputs will be "
          "APPROXIMATE SCALE — NOT GEOREFERENCED. See report.json/viewer.html "
          "for this run's labeling.")

print(f"\\nSetup done in {time.time()-_t0:.1f}s total.")
'''

CACHE_SAVE_NOTE_MD = """
### Publishing a cache dataset (speeds up future runs — do this once)

The Setup cell points `HF_HOME`/`TORCH_HOME`/`PIP_CACHE_DIR`/`YOLO_CONFIG_DIR`
at `/kaggle/working/cache/` for this run, so everything slow — pip wheels,
MapAnything weights (HuggingFace), the dinov2 backbone (torch.hub),
the YOLO-seg checkpoint (ultralytics) — lands there instead of scattered
default cache locations. After your first successful run:

1. In the Kaggle notebook viewer, open the **Data** pane → **Output**.
2. Click **New Dataset** from the `cache/` folder, name it (e.g.
   `sih3d-cache`), and publish it.
3. Add that dataset as an input to this notebook (**Add Input** → search
   your username → select it) — it stays attached across future Run Alls
   of this notebook, no need to redo this step per run.
4. On every future run, the Setup cell's `find_cache_dir()` detects it
   under `/kaggle/input/` and points those same env vars directly at it —
   zero downloads, install phase should drop from minutes to seconds.
"""

LAUNCH_CELL = '''
# ============================== LAUNCH =======================================
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(CODE_DIR))  # CODE_DIR from the Setup cell above

# Reload any sih3d.* submodule already cached in sys.modules BEFORE
# importing sih3d.pipeline below. This has to happen first, not just
# after (as a later pass in this cell also does): if sih3d.pipeline
# previously failed to import (e.g. one of ITS submodules had a bug,
# not pipeline.py itself), Python evicts sih3d.pipeline from sys.modules
# on that failure but leaves the submodule that DID import successfully
# still cached — stale. The `import sih3d.pipeline` line below would
# then re-exec pipeline.py fresh, but pipeline.py's own
# `from .mesh import X`-style lines pull from whatever's ALREADY in
# sys.modules for that submodule, stale or not — so it fails again the
# exact same way even after a successful Setup re-pull, and a
# reload-loop placed only after this import never gets a chance to run
# because the exception fires first. Reloading everything already
# cached, before attempting the import, closes that gap.
for _name in sorted(k for k in sys.modules if k == "sih3d" or k.startswith("sih3d.")):
    try:
        importlib.reload(sys.modules[_name])
    except Exception as _e:
        print(f"  [pre-reload warning] {_name}: {_e}")

import sih3d.events as events_mod
import sih3d.gpu_monitor as gpu_monitor_mod
import sih3d.report as report_mod
import sih3d.progress as progress_mod
import sih3d.pipeline as pipeline_mod

# Reload EVERY already-imported sih3d.* module again, not just the five
# named above — pipeline.py alone pulls in align/export/masks/backbone/
# mesh/decode/fusion/io_detect/keyframes/telemetry/artifacts, and
# reloading only pipeline_mod does NOT refresh any of those: `from .mesh
# import X` inside pipeline.py just rebinds a name from the already-
# cached sys.modules entry, so edits to mesh.py (or any other submodule)
# silently kept running under the OLD code with no error. This is what
# forced a full kernel restart for every one-line fix. Reloading every
# sih3d module already in sys.modules (pipeline_mod last, so its
# rebinding sees the fresh objects) means: after editing a %%writefile
# module-source cell above and re-running it, you only need to re-run
# THIS cell — no restart, no re-running Setup, install state and
# detected inputs are untouched.
_pipeline_mod_ref = sys.modules.get("sih3d.pipeline")
for _name in sorted(k for k in sys.modules if k == "sih3d" or k.startswith("sih3d.")):
    _m = sys.modules.get(_name)
    if _m is None or _m is _pipeline_mod_ref:
        continue
    try:
        importlib.reload(_m)
    except Exception as _e:
        print(f"  [reload warning] {_name}: {_e}")
if _pipeline_mod_ref is not None:
    importlib.reload(_pipeline_mod_ref)

from sih3d.events import EventBus
from sih3d.gpu_monitor import GpuMonitor
from sih3d.progress import ConsoleProgress
from sih3d.report import ReportBuilder
from sih3d.pipeline import Pipeline, PipelineConfig

bus = EventBus()
gpu_monitor = GpuMonitor(bus, interval_s=0.5)
report = ReportBuilder()

cfg = PipelineConfig(
    mode=MODE, backbone_choice=BACKBONE, prior_mode=PRIOR_MODE,
    use_finetuned=USE_FINETUNED_CHECKPOINT, output_dir=Path(OUTPUT_DIR),
    device0="cuda:0" if gpu_monitor.device_count >= 1 else "cpu",
    device1="cuda:1" if gpu_monitor.device_count >= 2 else ("cuda:0" if gpu_monitor.device_count >= 1 else "cpu"),
    dual_gpu=DUAL_GPU,
)

pipeline = Pipeline(cfg, detected, telemetry, bus, gpu_monitor, report)
progress = ConsoleProgress()

print(f"Video: {detected.video.path.name} | {detected.video.width}x{detected.video.height} | "
      f"{detected.video.fps:.1f} fps | mode={MODE} | GPUs={gpu_monitor.device_names}")
pipeline.start()

# This drain loop is the SOLE path from any bus event to `report` (see
# pipeline.py's _emit/_log docstrings) — pipeline.py itself only
# publishes to the bus now, it no longer feeds `report` directly. That
# used to be split: pipeline.py fed its own STAGE_*/KEYFRAME_*/LOG
# events into `report` directly and synchronously, while this loop only
# forwarded GPU_SAMPLE/CPU_SAMPLE (GpuMonitor's own thread has no
# `report` reference to call directly). The gap: anything logged via a
# bare `bus.log(...)` from OUTSIDE pipeline.py — mesh.py, fusion.py,
# masks.py, align.py, validation.py all do this — was never reaching
# `report` either way, so report.json's fallbacks/warnings lists were
# silently blind to most of the pipeline's own real warnings (e.g. "TSDF
# available but produced an empty mesh" never once showed up, no matter
# how many runs actually hit it). Forwarding everything here, from one
# place, fixes that without reopening the double-counting risk a second
# direct feed would create.
while pipeline.is_alive():
    for evt in bus.drain():
        progress.on_event(evt)
        report.on_event(evt)
    time.sleep(0.1)

# Final drain after the pipeline exits, so terminal progress catches final
# stage completions and warnings.
for evt in bus.drain():
    progress.on_event(evt)
    report.on_event(evt)
progress.close()

report.write_json(Path(OUTPUT_DIR) / "report.json")
report.write_html(Path(OUTPUT_DIR) / "report.html")

print(f"\\nPipeline finished: success={pipeline.result.success if pipeline.result else 'UNKNOWN'}")
'''

RESULTS_CELL = '''
# ============================== RESULTS ======================================
# Everything here renders INLINE — nothing needs to be downloaded to check
# the run. All the underlying files are still saved under /kaggle/working/
# outputs/ (table + download links at the end of this cell) for later use.
%matplotlib inline
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(CODE_DIR))  # CODE_DIR from the Setup cell above
import sih3d.stage_views as stage_views_mod
importlib.reload(stage_views_mod)
from sih3d.stage_views import (
    render_frame_extraction, render_poses, render_geometric_reconstruction,
    render_large_scale_alignment, render_dense_point_cloud,
    render_mesh_textured_model, render_final_summary,
)
from IPython.display import display, FileLink, HTML

artifacts = pipeline.artifacts

print("=" * 80); print("KEYFRAMES"); print("=" * 80)
render_frame_extraction(artifacts)

print("=" * 80); print("CAMERA TRAJECTORY"); print("=" * 80)
render_poses(artifacts)

print("=" * 80); print("SAMPLE DEPTH / CONFIDENCE MAPS"); print("=" * 80)
render_geometric_reconstruction(artifacts)

print("=" * 80); print("LARGE-SCALE ALIGNMENT"); print("=" * 80)
render_large_scale_alignment(artifacts)

print("=" * 80); print("DENSE POINT CLOUD"); print("=" * 80)
render_dense_point_cloud(artifacts)

print("=" * 80); print("TEXTURED MESH + DSM/ORTHOMOSAIC"); print("=" * 80)
render_mesh_textured_model(artifacts)

print("=" * 80); print("TIMINGS"); print("=" * 80)
render_final_summary(report)

print("\\n" + "=" * 80); print("OUTPUT FILES (saved under", OUTPUT_DIR, ")"); print("=" * 80)
out_dir = Path(OUTPUT_DIR)
rows = []
for status in getattr(pipeline, "_export_statuses", []):
    size_mb = (status.size_bytes or 0) / 1e6
    state = "OK" if status.ok else f"SKIPPED ({status.skipped_reason})"
    rows.append(f"<tr><td>{status.name}</td><td>{state}</td><td>{size_mb:.2f} MB</td></tr>")
display(HTML("<table><tr><th>File</th><th>Status</th><th>Size</th></tr>" + "".join(rows) + "</table>"))
for status in getattr(pipeline, "_export_statuses", []):
    if status.ok and status.path is not None:
        display(FileLink(str(status.path)))
'''

GROUND_TRUTH_CELL = '''
# ============================== RENDER vs GROUND TRUTH =======================
# The point cloud reprojected into a few real keyframe cameras (their
# actual estimated pose + intrinsics), next to the real photo each one
# captured, with a point-coverage percentage per view — see validation.py
# for exactly what "coverage" does and doesn't measure.
import importlib
import sys

sys.path.insert(0, str(CODE_DIR))  # CODE_DIR from the Setup cell above
import sih3d.stage_views as stage_views_mod
importlib.reload(stage_views_mod)
from sih3d.stage_views import render_ground_truth_comparison

render_ground_truth_comparison(pipeline.artifacts)
'''

DIAG_STEP_A_CELL = '''
# ============================== DIAGNOSTIC: Step A - decode strategies ======
# Measurement only, per explicit direction: no new reconstruction code.
# Compares GOP length / I-frame-only decode / raw NVDEC+scale_cuda against
# the pipeline's own (already-measured) decode time, on the SAME video.
import json
import subprocess
import time
from pathlib import Path

video_path = detected.video.path
results = {}

# -- GOP length (how far apart I-frames actually are) -----------------------
_probe = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
     "frame=pict_type", "-of", "csv=p=0", "-read_intervals", "%+#300", str(video_path)],
    capture_output=True, text=True, timeout=60,
)
pict_types = [l.strip() for l in _probe.stdout.splitlines() if l.strip()]
i_frame_positions = [i for i, t in enumerate(pict_types) if t == "I"]
gop_lengths = [b - a for a, b in zip(i_frame_positions[:-1], i_frame_positions[1:])]
avg_gop = sum(gop_lengths) / len(gop_lengths) if gop_lengths else None
print(f"GOP: sampled {len(pict_types)} frames, {len(i_frame_positions)} I-frames, "
      f"avg GOP length = {avg_gop} frames" if avg_gop else "GOP: could not determine (check ffprobe output)")
results["gop_avg_frames"] = avg_gop

# -- I-frame-only decode (-skip_frame nokey) ---------------------------------
_t0 = time.time()
_r = subprocess.run(
    ["ffmpeg", "-v", "error", "-skip_frame", "nokey", "-i", str(video_path),
     "-vsync", "0", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
    capture_output=True, timeout=120,
)
_elapsed = time.time() - _t0
_n_bytes = len(_r.stdout)
_frame_bytes = detected.video.width * detected.video.height * 3
_n_iframes_decoded = _n_bytes // _frame_bytes if _frame_bytes else 0
print(f"I-frame-only decode (-skip_frame nokey): {_elapsed:.1f}s, {_n_iframes_decoded} I-frames "
      f"({_n_iframes_decoded / _elapsed:.1f} frames/s)" if _elapsed > 0 else "I-frame-only decode: 0s?")
results["iframe_only_decode_s"] = _elapsed
results["iframe_only_count"] = _n_iframes_decoded

# -- raw NVDEC decode + GPU-side scale_cuda, bypassing our own decode.py's
# CPU-scale fallback and PyNvVideoCodec entirely, to isolate whether
# scale_cuda specifically is what's been unavailable ---------------------
_target_w = 1920
_target_h = int(round(detected.video.height * (_target_w / detected.video.width))) // 2 * 2
_t0 = time.time()
_r2 = subprocess.run(
    ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
     "-i", str(video_path), "-vf", f"scale_cuda={_target_w}:{_target_h}",
     "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
    capture_output=True, timeout=120,
)
_elapsed2 = time.time() - _t0
_ok2 = _r2.returncode == 0 and len(_r2.stdout) > 0
_stderr2_tail = _r2.stderr[-500:].decode(errors="replace")
print(f"NVDEC decode + scale_cuda ({_target_w}x{_target_h}): "
      f"{'OK' if _ok2 else 'FAILED'}, {_elapsed2:.1f}s"
      + ("" if _ok2 else f" — stderr tail: {_stderr2_tail}"))
results["nvdec_scale_cuda_ok"] = _ok2
results["nvdec_scale_cuda_s"] = _elapsed2 if _ok2 else None

# -- for comparison: this run's OWN already-measured decode time -----------
_own_decode_s = None
for _s in report.to_dict()["stages"]:
    if _s["name"] == "frame_extraction":
        _own_decode_s = _s["elapsed_s"]
print(f"Pipeline's own frame_extraction stage (this run, includes sharpness+selection, not just decode): {_own_decode_s}s")
results["pipeline_frame_extraction_s"] = _own_decode_s
results["video_duration_s"] = detected.video.duration_s if hasattr(detected.video, "duration_s") else None
results["video_resolution"] = f"{detected.video.width}x{detected.video.height}"

Path(OUTPUT_DIR, "diag_step_a_decode.json").write_text(json.dumps(results, indent=2))
print("\\nSaved diag_step_a_decode.json")
'''

DIAG_STEP_B_CELL = '''
# ============================== DIAGNOSTIC: Step B - MapAnything COLMAP export
# Official facebookresearch/map-anything scripts/demo_colmap.py, run on the
# exact same keyframes this notebook's own run used (saved to disk by
# pipeline.py as a measurement-only side effect, see diag_keyframes/).
# Measurement + glue only, per explicit direction — no new reconstruction code.
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

diag_root = Path(OUTPUT_DIR) / "diag_step_b"
diag_root.mkdir(parents=True, exist_ok=True)
keyframes_dir = Path(OUTPUT_DIR) / "diag_keyframes"
n_keyframes = len(list(keyframes_dir.glob("*.jpg")))
print(f"Keyframe images available for export: {n_keyframes}")

_pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycolmap==3.10.0"], capture_output=True, text=True)
print("pycolmap install:", "OK" if _pip.returncode == 0 else _pip.stderr[-500:])

demo_script_path = diag_root / "demo_colmap.py"
subprocess.run(["curl", "-sL", "-o", str(demo_script_path),
                 "https://raw.githubusercontent.com/facebookresearch/map-anything/main/scripts/demo_colmap.py"], check=True)
print("demo_colmap.py fetched:", demo_script_path.exists(), demo_script_path.stat().st_size if demo_script_path.exists() else 0, "bytes")
_has_use_ba = "--use_ba" in demo_script_path.read_text()
print(f"This script has a --use_ba flag: {_has_use_ba} "
      "(map-anything's own demo_colmap.py has NO bundle-adjustment option, unlike facebookresearch/vggt's "
      "same-named script — only a single MapAnything-inference export pass is possible here)")


class _MemSampler:
    def __init__(self):
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not self._stop.is_set():
                info = pynvml.nvmlDeviceGetMemoryInfo(h)
                self.peak_mb = max(self.peak_mb, info.used / (1024 * 1024))
                time.sleep(0.5)
        except Exception as e:
            print("  [mem sampler warning]", e)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


def run_demo_colmap(output_dir, extra_args=None):
    sampler = _MemSampler()
    sampler.start()
    t0 = time.time()
    cmd = [sys.executable, str(demo_script_path), "--images_dir", str(keyframes_dir),
           "--output_dir", str(output_dir), "--apache"] + (extra_args or [])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    elapsed = time.time() - t0
    sampler.stop()
    print(f"\\n-- demo_colmap.py -> {output_dir.name} --")
    print(f"exit={result.returncode}, elapsed={elapsed:.1f}s, peak_gpu_mem={sampler.peak_mb:.0f}MB")
    if result.returncode != 0:
        print("STDERR (tail):", result.stderr[-2000:])
    else:
        print("STDOUT (tail):", result.stdout[-800:])
    return {"elapsed_s": elapsed, "peak_gpu_mem_mb": sampler.peak_mb, "returncode": result.returncode}

out_dir = diag_root / "colmap_export"
run_result = run_demo_colmap(out_dir)

results = {"has_use_ba_flag": _has_use_ba, "n_keyframes": n_keyframes, "export": run_result}

sparse_dir = out_dir / "sparse"
if sparse_dir.exists():
    try:
        import pycolmap
        recon = pycolmap.Reconstruction(str(sparse_dir))
        cams = list(recon.cameras.values())
        cam0 = cams[0] if cams else None
        print(f"\\nExported COLMAP model: {len(recon.images)} images, {len(recon.points3D)} 3D points, {len(recon.cameras)} camera(s)")
        if cam0 is not None:
            print(f"Camera resolution in export: {cam0.width}x{cam0.height} "
                  f"(full-res keyframes were {detected.video.width}x{detected.video.height} -- "
                  f"{'MATCHES full res' if cam0.width == detected.video.width else 'does NOT match full res, needs intrinsics rescale for full-res texturing'})")
            results["export_camera_resolution"] = f"{cam0.width}x{cam0.height}"

        # "BA reprojection error" - COLMAP's own per-point mean track error,
        # already computed by the export (no separate --use_ba pass exists
        # to run here, see above).
        errors = [p.error for p in recon.points3D.values() if p.error is not None and p.error > 0]
        if errors:
            import numpy as _np
            print(f"Reprojection error across {len(errors)} points3D: "
                  f"mean={_np.mean(errors):.3f}px, median={_np.median(errors):.3f}px, "
                  f"P90={_np.percentile(errors, 90):.3f}px")
            results["reprojection_error_px"] = {
                "mean": float(_np.mean(errors)), "median": float(_np.median(errors)),
                "p90": float(_np.percentile(errors, 90)), "n_points": len(errors),
            }
        else:
            print("No points3D.error values found (points2D may have been skipped, or all points are new)")

        # -- Self-warp / reprojection check using the EXPORTED cameras: for
        # each image, reproject its own observed track points through its
        # own recorded pose+intrinsics and compare to the observed pixel.
        # This is the correct, direct version of the ad-hoc self-warp test
        # from the custom-TSDF debugging phase (frozen per explicit
        # direction) -- target: under 1px.
        import numpy as _np
        all_px_err = []
        for img in recon.images.values():
            cam = recon.cameras[img.camera_id]
            cam_from_world = img.cam_from_world
            for p2d in img.points2D:
                if not p2d.has_point3D():
                    continue
                pt3d = recon.points3D[p2d.point3D_id].xyz
                pt_cam = cam_from_world * pt3d
                if pt_cam[2] <= 0:
                    continue
                uv_reproj = cam.img_from_cam(pt_cam)
                err = float(_np.linalg.norm(uv_reproj - p2d.xy))
                all_px_err.append(err)
        if all_px_err:
            arr = _np.array(all_px_err)
            print(f"\\nDirect self-consistency check (exported cameras, {len(arr):,} observations): "
                  f"median={_np.median(arr):.3f}px, P90={_np.percentile(arr, 90):.3f}px, mean={arr.mean():.3f}px "
                  f"(target: under 1px)")
            results["self_consistency_px"] = {
                "median": float(_np.median(arr)), "p90": float(_np.percentile(arr, 90)), "mean": float(arr.mean()),
            }
        else:
            print("\\nNo points2D-with-track observations found for self-consistency check "
                  "(likely exported with --skip_point2d, or export format differs from expected)")
    except Exception as e:
        import traceback
        print(f"Failed to read exported COLMAP model: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["read_error"] = str(e)
else:
    print(f"No sparse/ output found at {sparse_dir} -- export likely failed, see STDERR above")

Path(OUTPUT_DIR, "diag_step_b_colmap.json").write_text(json.dumps(results, indent=2, default=str))
print("\\nSaved diag_step_b_colmap.json")
'''

DIAG_STEP_C_CELL = '''
# ============================== DIAGNOSTIC: Step C - OpenMVS on Kaggle ======
# Tests the prebuilt Ubuntu x64 release binaries first (cheapest possible
# check) before considering a from-source build. Measurement only.
import json
import subprocess
from pathlib import Path

diag_root = Path(OUTPUT_DIR) / "diag_step_c"
diag_root.mkdir(parents=True, exist_ok=True)
zip_path = diag_root / "OpenMVS_Ubuntu_x64.zip"
bin_dir = diag_root / "bin"

_dl = subprocess.run(
    ["curl", "-sL", "-o", str(zip_path),
     "https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Ubuntu_x64.zip"],
    capture_output=True, text=True, timeout=180,
)
print(f"Download: exit={_dl.returncode}, size={zip_path.stat().st_size if zip_path.exists() else 0} bytes")

bin_dir.mkdir(exist_ok=True)
_unzip = subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(bin_dir)], capture_output=True, text=True)
print(f"Unzip: exit={_unzip.returncode}")
if _unzip.returncode != 0:
    print(_unzip.stderr[-1000:])

all_bins = list(bin_dir.rglob("*"))
exe_candidates = {p.name: p for p in all_bins if p.is_file() and (p.stat().st_mode & 0o111)}
print(f"Executables found: {sorted(exe_candidates.keys())}")

tools = ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh"]
results = {}
for tool in tools:
    path = exe_candidates.get(tool)
    entry = {"found": path is not None}
    if path is None:
        print(f"\\n-- {tool}: NOT FOUND in release archive --")
        results[tool] = entry
        continue
    path.chmod(0o755)
    _ldd = subprocess.run(["ldd", str(path)], capture_output=True, text=True)
    missing = [l.strip() for l in _ldd.stdout.splitlines() if "not found" in l]
    entry["ldd_missing_libs"] = missing
    _help = subprocess.run([str(path), "--help"], capture_output=True, text=True, timeout=30)
    entry["help_returncode"] = _help.returncode
    entry["help_ok"] = _help.returncode == 0 and len(_help.stdout) > 0
    print(f"\\n-- {tool} ({path}) --")
    print(f"ldd missing libs: {missing or 'none'}")
    print(f"--help: exit={_help.returncode}, ok={entry['help_ok']}")
    if not entry["help_ok"]:
        print("  stderr:", (_help.stderr or "")[-500:])
        print("  stdout:", (_help.stdout or "")[-500:])
    results[tool] = entry

n_working = sum(1 for v in results.values() if v.get("help_ok"))
print(f"\\n{n_working}/{len(tools)} tools runnable from the prebuilt release with no missing libs and a working --help.")
if n_working < len(tools):
    print("Building from source was NOT attempted this run (would need a separate, longer Kaggle session -- "
          "reporting the prebuilt-binary result only, as instructed to check that first).")

Path(OUTPUT_DIR, "diag_step_c_openmvs.json").write_text(json.dumps(results, indent=2, default=str))
print("\\nSaved diag_step_c_openmvs.json")
'''


def build() -> None:
    nb = nbf.v4.new_notebook()
    cells = [
        md(TITLE_MD), code(CONFIG_CELL), code(SETUP_CELL), md(CACHE_SAVE_NOTE_MD),
        code(LAUNCH_CELL), code(RESULTS_CELL), code(GROUND_TRUTH_CELL),
        code(DIAG_STEP_A_CELL), code(DIAG_STEP_B_CELL), code(DIAG_STEP_C_CELL),
    ]

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }

    nbf.validate(nb)
    OUT_PATH.write_text(nbf.writes(nb))
    print(f"Wrote {OUT_PATH} ({len(cells)} cells) — sih3d/ is pulled from GitHub by the Setup cell, no module-source cells embedded")


if __name__ == "__main__":
    build()

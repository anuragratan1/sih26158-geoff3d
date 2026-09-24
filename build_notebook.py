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

# Off by default: data-parallel backbone across both GPUs (even chunks on
# cuda:0, odd on cuda:1, each with its own prefetch queue; alignment stays
# single-threaded/ordered afterward, since it carries state between
# consecutive chunks). Ignored with a warning if fewer than 2 GPUs are
# detected. Kept behind this flag rather than auto-enabled on 2 GPUs so a
# single-GPU-backbone baseline run stays available on demand, unmixed with
# the dual-GPU path's own behavior (double model load time, split logs, a
# DUAL_GPU utilization summary at the end of geometric_reconstruction).
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
if CODE_DIR.exists():
    _sync = subprocess.run(["git", "-C", str(CODE_DIR), "pull", "--ff-only"], capture_output=True, text=True, timeout=60)
    if _sync.returncode == 0:
        print(f"Code: pulled latest into {CODE_DIR}\\n{_sync.stdout.strip()}")
    else:
        print(f"Code: pull failed, using existing checkout as-is ({_sync.stderr.strip()[-300:]})")
else:
    _sync = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(CODE_DIR)], capture_output=True, text=True, timeout=180)
    if _sync.returncode != 0:
        raise RuntimeError(f"git clone of {REPO_URL} failed: {_sync.stderr.strip()[-500:]}")
    print(f"Code: cloned into {CODE_DIR}")
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
        ("scipy", "scipy"), ("xatlas", "xatlas"), ("PIL", "pillow"),
        ("laspy", "laspy"), ("rasterio", "rasterio"), ("pyproj", "pyproj"),
        ("trimesh", "trimesh"), ("open3d", "open3d"), ("pymavlink", "pymavlink"),
        ("huggingface_hub", "huggingface_hub"), ("safetensors", "safetensors"),
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
        _pip_install("pyrender pyopengl", label="pyrender (GPU offscreen render via EGL)")
    else:
        _tqdm.write("  [already present] pyrender")

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
    ("pyrender (3D showcase render)", _step_pyrender),
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
import sih3d.events as events_mod
import sih3d.gpu_monitor as gpu_monitor_mod
import sih3d.report as report_mod
import sih3d.progress as progress_mod
import sih3d.pipeline as pipeline_mod

# Reload EVERY already-imported sih3d.* module, not just the five named
# above — pipeline.py alone pulls in align/export/masks/backbone/mesh/
# decode/fusion/io_detect/keyframes/telemetry/artifacts, and reloading only
# pipeline_mod does NOT refresh any of those: `from .mesh import X` inside
# pipeline.py just rebinds a name from the already-cached sys.modules
# entry, so edits to mesh.py (or any other submodule) silently kept running
# under the OLD code with no error. This is what forced a full kernel
# restart for every one-line fix. Reloading every sih3d module already in
# sys.modules (pipeline_mod last, so its rebinding sees the fresh objects)
# means: after editing a %%writefile module-source cell above and
# re-running it, you only need to re-run THIS cell — no restart, no
# re-running Setup, install state and detected inputs are untouched.
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

# The pipeline writes report state synchronously.  This loop only renders
# lightweight text progress; it never serializes frames or updates widgets.
while pipeline.is_alive():
    for evt in bus.drain():
        progress.on_event(evt)
    time.sleep(0.1)

# Final drain after the pipeline exits, so terminal progress catches final
# stage completions and warnings.
for evt in bus.drain():
    progress.on_event(evt)
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

VIEWER_CELL = '''
# ============================== 3D SHOWCASE ==================================
# A few stills + a short orbit video rendered straight from the mesh/point
# cloud, so you can actually SEE the result without leaving the notebook or
# downloading mesh.glb/pointcloud.ply to a separate viewer. Best-effort —
# degrades to a clear message rather than crashing this cell if something's
# unavailable. Separate cell from the ground-truth comparison below: a
# combined cell's output got long enough (stills + video + 3 comparison
# figures) to risk Kaggle truncating it, and each is easier to find on its
# own anyway.
import importlib
import sys

sys.path.insert(0, str(CODE_DIR))  # CODE_DIR from the Setup cell above
import sih3d.stage_views as stage_views_mod
importlib.reload(stage_views_mod)
from sih3d.stage_views import render_showcase

render_showcase(pipeline.artifacts)
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


def build() -> None:
    nb = nbf.v4.new_notebook()
    cells = [
        md(TITLE_MD), code(CONFIG_CELL), code(SETUP_CELL), md(CACHE_SAVE_NOTE_MD),
        code(LAUNCH_CELL), code(RESULTS_CELL), code(VIEWER_CELL), code(GROUND_TRUTH_CELL),
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

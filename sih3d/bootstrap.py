"""Everything that used to be separate notebook cells (environment checks,
dependency installs, cache/input detection) now runs in ONE function, called
from the single RUN cell's own background thread — so the dashboard can
display immediately (before any of this starts) and these steps report
their progress into it instead of raw stdout/pip spam. `bootstrap.run(...)`
blocks until the whole pipeline finishes; that's expected, since it's
already running off the main thread.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .events import EventBus, EventType
from .gpu_monitor import GpuMonitor
from .io_detect import DetectedInputs, detect_all, find_cache_dir
from .pipeline import Pipeline, PipelineConfig
from .report import ReportBuilder
from .telemetry import extract_embedded_telemetry, parse_telemetry


class BootstrapResult:
    def __init__(self):
        self.ok = False
        self.error: str | None = None
        self.pipeline: "Pipeline | None" = None
        self.detected: DetectedInputs | None = None


def _check_internet(timeout_s: float = 5.0) -> bool:
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=timeout_s)
        return True
    except Exception:
        return False


def _check_gpu() -> tuple[int, list[str]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            return [l.strip() for l in out.stdout.strip().splitlines()].__len__(), [
                l.strip() for l in out.stdout.strip().splitlines()
            ]
    except Exception:
        pass
    return 0, []


def _try_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _pip_install_silent(
    spec: str, bus: EventBus, label: str | None = None,
    extra_args: list[str] | None = None, find_links: Path | None = None,
) -> None:
    """Captures pip's stdout/stderr (never prints it raw into the notebook
    output — the dashboard's install panel is the only place progress
    shows) and reports one INSTALL_PROGRESS event per package."""
    label = label or spec
    bus.publish(EventType.INSTALL_PROGRESS, package=label, status="installing", elapsed_s=0.0)
    t0 = time.time()
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if find_links:
        cmd += ["--find-links", str(find_links)]
    cmd += (extra_args or []) + [spec]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        elapsed = time.time() - t0
        if result.returncode != 0:
            bus.publish(EventType.INSTALL_PROGRESS, package=label, status="failed", elapsed_s=elapsed)
            bus.log(f"pip install {label} failed: {result.stderr.strip()[-300:]}", level="warn")
        else:
            bus.publish(EventType.INSTALL_PROGRESS, package=label, status="ok", elapsed_s=elapsed)
    except Exception as e:
        bus.publish(EventType.INSTALL_PROGRESS, package=label, status="failed", elapsed_s=time.time() - t0)
        bus.log(f"pip install {label} raised {type(e).__name__}: {e}", level="warn")


def _mark_present(pkg: str, bus: EventBus) -> None:
    bus.publish(EventType.INSTALL_PROGRESS, package=pkg, status="ok", elapsed_s=0.0)


def run(
    input_root: Path, output_dir: Path, cache_dir: Path,
    mode: str, backbone_choice: str, prior_mode: str, use_finetuned: bool,
    bus: EventBus, report: ReportBuilder, result: BootstrapResult,
) -> None:
    # -- env_check -------------------------------------------------------
    bus.publish(EventType.SETUP_STAGE_START, stage="env_check")
    internet_ok = _check_internet()
    gpu_count, gpu_names = _check_gpu()
    bus.log(f"Internet: {'ON' if internet_ok else 'OFF'}")
    bus.log(f"GPUs detected: {gpu_names or '(none)'}")

    if not internet_ok:
        err = (
            "Internet is OFF for this notebook session. Enable it under "
            "Notebook Settings -> Internet -> On, then Run All again. Model "
            "weights and pip installs need it."
        )
        bus.publish(EventType.SETUP_STAGE_ERROR, stage="env_check", note="Internet is OFF")
        result.error = err
        bus.publish(EventType.PIPELINE_ERROR, error=err, traceback="")
        return

    if gpu_count == 0:
        bus.publish(EventType.SETUP_STAGE_FALLBACK, stage="env_check", note="no GPU detected — will run on CPU (slow)")
    bus.publish(EventType.SETUP_STAGE_DONE, stage="env_check")

    # -- install -----------------------------------------------------------
    bus.publish(EventType.SETUP_STAGE_START, stage="install")
    cache_ds = find_cache_dir(input_root)
    if cache_ds is not None:
        bus.log(f"Found a cache dataset at {cache_ds} — pip will prefer wheels there over downloading")

    simple_deps = [
        ("scipy", "scipy"), ("xatlas", "xatlas"), ("PIL", "pillow"),
        ("laspy", "laspy"), ("rasterio", "rasterio"), ("pyproj", "pyproj"),
        ("trimesh", "trimesh"), ("open3d", "open3d"), ("pymavlink", "pymavlink"),
        ("plotly", "plotly"), ("huggingface_hub", "huggingface_hub"),
        ("safetensors", "safetensors"), ("anywidget", "anywidget"),
    ]
    for mod, pkg in simple_deps:
        if _try_import(mod):
            _mark_present(pkg, bus)
        else:
            _pip_install_silent(pkg, bus, find_links=cache_ds)

    if _try_import("torchcodec"):
        _mark_present("torchcodec", bus)
    else:
        _pip_install_silent("torchcodec", bus, label="torchcodec (optional GPU decode)", find_links=cache_ds)

    if _try_import("ultralytics"):
        _mark_present("ultralytics", bus)
    else:
        _pip_install_silent("ultralytics", bus, label="ultralytics (dynamic-object masking)", find_links=cache_ds)

    if _try_import("mapanything"):
        _mark_present("mapanything", bus)
    else:
        _pip_install_silent(
            "git+https://github.com/facebookresearch/map-anything.git", bus,
            extra_args=["--no-deps"], label="mapanything (Backbone C)", find_links=cache_ds,
        )
        for pkg in ["opencv-python-headless", "hydra-core", "omegaconf", "uniception", "roma"]:
            if not _try_import(pkg.replace("-", "_")):
                _pip_install_silent(pkg, bus, find_links=cache_ds)

    try:
        from .fullscreen_server import ensure_cloudflared

        ensure_cloudflared(bus)
    except Exception as e:
        bus.log(f"cloudflared setup skipped ({e}) — full-screen viewer link will be unavailable", level="warn")

    bus.publish(EventType.SETUP_STAGE_DONE, stage="install")

    # -- detect_inputs -----------------------------------------------------
    bus.publish(EventType.SETUP_STAGE_START, stage="detect_inputs")
    detected = detect_all(input_root)
    result.detected = detected

    if detected.video is None:
        err = f"No video file found under {input_root}. Attach a dataset containing a drone video — see RUN_ON_KAGGLE.md."
        bus.publish(EventType.SETUP_STAGE_ERROR, stage="detect_inputs", note=err)
        result.error = err
        bus.publish(EventType.PIPELINE_ERROR, error=err, traceback="")
        return

    bus.log(
        f"Video: {detected.video.path.name} ({detected.video.width}x{detected.video.height}, "
        f"{detected.video.fps:.1f}fps, {detected.video.duration_s:.0f}s)"
    )
    for w in detected.warnings:
        bus.log(w)

    telemetry = None
    if detected.telemetry_path is not None:
        telemetry = parse_telemetry(detected.telemetry_path, detected.telemetry_kind)
        bus.log(f"Parsed telemetry: {len(telemetry)} samples ({telemetry.kind}); notes: {telemetry.notes}")
    elif detected.telemetry_kind == "embedded":
        telemetry = extract_embedded_telemetry(detected.video.path)
        bus.log(f"Parsed embedded telemetry: {len(telemetry)} samples; notes: {telemetry.notes}")
    if telemetry is None or len(telemetry) == 0:
        bus.log("NO GPS TELEMETRY FOUND. Continuing RGB-only: outputs will be APPROXIMATE SCALE — NOT GEOREFERENCED.", level="warn")

    gpu_monitor = GpuMonitor(bus, interval_s=0.5)

    bus.publish(
        EventType.HEADER_UPDATE,
        video_name=detected.video.path.name,
        resolution=f"{detected.video.width}x{detected.video.height}",
        fps=round(detected.video.fps, 1),
        duration_s=detected.video.duration_s,
        telemetry_type=detected.telemetry_kind or "none",
        telemetry_sample_count=len(telemetry) if telemetry else 0,
        sync_method="timestamp" if (telemetry and telemetry.has_timestamps) else "proportional interpolation",
        mode=mode, gpu_count=gpu_monitor.device_count, gpu_names=gpu_monitor.device_names,
        backbone=backbone_choice, prior_mode=prior_mode,
    )
    bus.publish(EventType.SETUP_STAGE_DONE, stage="detect_inputs")

    # -- full-screen server (best-effort, never blocks the pipeline) -------
    try:
        from .fullscreen_server import start_fullscreen_server

        url = start_fullscreen_server(output_dir, bus)
        bus.publish(EventType.TUNNEL_READY, url=url)
    except Exception as e:
        bus.log(f"Full-screen viewer server unavailable ({e}); inline viewer still works", level="warn")
        bus.publish(EventType.TUNNEL_READY, url=None)

    # -- live reconstruction page (best-effort, never blocks the pipeline) -
    live_page_server = None
    try:
        from .live_page import start_live_page

        live_page_server, live_url = start_live_page(detected.video.path, bus)
        bus.publish(EventType.LIVE_PAGE_READY, url=live_url)
    except Exception as e:
        bus.log(f"Live reconstruction page unavailable ({e}); the dashboard's inline 3D tab is the fallback", level="warn")
        bus.publish(EventType.LIVE_PAGE_READY, url=None)

    # -- pipeline ------------------------------------------------------------
    cfg = PipelineConfig(
        mode=mode, backbone_choice=backbone_choice, prior_mode=prior_mode, use_finetuned=use_finetuned,
        output_dir=output_dir,
        device0="cuda:0" if gpu_monitor.device_count >= 1 else "cpu",
        device1="cuda:1" if gpu_monitor.device_count >= 2 else ("cuda:0" if gpu_monitor.device_count >= 1 else "cpu"),
    )
    pipeline = Pipeline(cfg, detected, telemetry, bus, gpu_monitor, report, live_page=live_page_server)
    result.pipeline = pipeline
    pipeline._run_guarded()  # synchronous — bootstrap.run() already runs off the main thread
    result.ok = bool(pipeline.result and pipeline.result.success)

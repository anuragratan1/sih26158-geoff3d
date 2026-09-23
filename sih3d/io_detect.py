"""Zero-manual-path input detection.

Recursively scans an input root (Kaggle: /kaggle/input/) for:
  - drone video(s): picks the longest by ffprobe duration, lists the rest
  - telemetry matched to the chosen video (basename first, then folder)
  - camera intrinsics (json/yaml)
  - a cache dataset (previously published wheels/checkpoints)

Never raises on missing optional inputs — callers decide what to do with
`None` telemetry (RGB-only relative-scale fallback is handled in align.py).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts"}
TELEMETRY_EXTS = {".srt", ".csv", ".gpx", ".kml", ".json", ".tlog", ".bin"}
INTRINSICS_HINT_EXTS = {".json", ".yaml", ".yml"}
INTRINSICS_KEYS = {"fx", "fy", "cx", "cy", "camera_matrix", "K", "intrinsics", "intrinsic_matrix"}
CACHE_DIR_HINTS = {"cache", "checkpoints", "wheels", "sih3d_cache", "sih3d-cache"}


@dataclass
class VideoInfo:
    path: Path
    duration_s: float
    width: int
    height: int
    fps: float
    codec: str = ""


@dataclass
class DetectedInputs:
    video: VideoInfo | None = None
    other_videos: list[VideoInfo] = field(default_factory=list)
    telemetry_path: Path | None = None
    telemetry_kind: str | None = None  # "srt" | "csv" | "gpx" | "kml" | "json" | "mavlink" | "embedded"
    intrinsics_path: Path | None = None
    intrinsics: dict | None = None
    cache_dir: Path | None = None
    warnings: list[str] = field(default_factory=list)


def _run_ffprobe(path: Path) -> dict | None:
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path),
            ],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return json.loads(out.stdout)
    except Exception:
        return None


def probe_video(path: Path) -> VideoInfo | None:
    info = _run_ffprobe(path)
    if info is None:
        return None
    vstream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if vstream is None:
        return None
    fmt = info.get("format", {})
    duration = float(fmt.get("duration") or vstream.get("duration") or 0.0)
    num, den = (vstream.get("r_frame_rate", "0/1").split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den or 1) else 0.0
    return VideoInfo(
        path=path,
        duration_s=duration,
        width=int(vstream.get("width", 0)),
        height=int(vstream.get("height", 0)),
        fps=fps,
        codec=str(vstream.get("codec_name", "")),
    )


def _iter_files(root: Path):
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def find_videos(root: Path) -> list[VideoInfo]:
    videos = []
    for p in _iter_files(root):
        if p.suffix.lower() in VIDEO_EXTS:
            vi = probe_video(p)
            if vi is not None:
                videos.append(vi)
    videos.sort(key=lambda v: v.duration_s, reverse=True)
    return videos


def _basename_stem(p: Path) -> str:
    return p.stem.lower()


def find_telemetry(root: Path, video: VideoInfo) -> tuple[Path | None, str | None]:
    """Match telemetry to the chosen video: basename first, then folder."""
    candidates = [p for p in _iter_files(root) if p.suffix.lower() in TELEMETRY_EXTS]
    if not candidates:
        return None, None

    video_stem = _basename_stem(video.path)
    same_name = [p for p in candidates if _basename_stem(p) == video_stem]
    if same_name:
        chosen = same_name[0]
        return chosen, _kind_from_suffix(chosen)

    same_folder = [p for p in candidates if p.parent == video.path.parent]
    if same_folder:
        # Prefer SRT (richest per-frame gimbal/GPS) if present in-folder.
        srt = [p for p in same_folder if p.suffix.lower() == ".srt"]
        chosen = srt[0] if srt else same_folder[0]
        return chosen, _kind_from_suffix(chosen)

    return None, None


def _kind_from_suffix(p: Path) -> str:
    suf = p.suffix.lower()
    return {
        ".srt": "srt", ".csv": "csv", ".gpx": "gpx", ".kml": "kml",
        ".json": "json", ".tlog": "mavlink", ".bin": "mavlink",
    }.get(suf, suf.lstrip("."))


def find_intrinsics(root: Path) -> tuple[Path | None, dict | None]:
    for p in _iter_files(root):
        if p.suffix.lower() not in INTRINSICS_HINT_EXTS:
            continue
        try:
            if p.suffix.lower() == ".json":
                data = json.loads(p.read_text())
            else:
                import yaml  # deferred import; only needed if a yaml file is present

                data = yaml.safe_load(p.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        flat_keys = set(_flatten_keys(data))
        if flat_keys & INTRINSICS_KEYS:
            return p, data
    return None, None


def _flatten_keys(d: dict, depth: int = 0):
    if depth > 3:
        return
    for k, v in d.items():
        yield k
        if isinstance(v, dict):
            yield from _flatten_keys(v, depth + 1)


def find_cache_dir(input_root: Path) -> Path | None:
    if not input_root.exists():
        return None
    for child in input_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if any(hint in name for hint in CACHE_DIR_HINTS):
            return child
        # Heuristic: a dataset containing .whl files or a checkpoints/ subdir
        try:
            if any(f.suffix == ".whl" for f in child.rglob("*.whl")):
                return child
            if (child / "checkpoints").exists():
                return child
        except Exception:
            continue
    return None


def detect_all(input_root: Path = Path("/kaggle/input")) -> DetectedInputs:
    result = DetectedInputs()

    videos = find_videos(input_root)
    if not videos:
        result.warnings.append(f"No video files found under {input_root}")
        return result

    result.video = videos[0]
    result.other_videos = videos[1:]
    if result.other_videos:
        names = ", ".join(v.path.name for v in result.other_videos)
        result.warnings.append(f"Multiple videos found; using longest ({result.video.path.name}). Ignored: {names}")

    tel_path, tel_kind = find_telemetry(input_root, result.video)
    if tel_path is None:
        result.warnings.append("No telemetry file found — will check for embedded subtitle/data stream, then fall back to RGB-only")
    result.telemetry_path = tel_path
    result.telemetry_kind = tel_kind

    intr_path, intr_data = find_intrinsics(input_root)
    result.intrinsics_path = intr_path
    result.intrinsics = intr_data
    if intr_path is None:
        result.warnings.append("No intrinsics file found — will use metadata or model-estimated intrinsics")

    result.cache_dir = find_cache_dir(input_root)

    return result

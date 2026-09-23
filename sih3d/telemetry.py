"""Telemetry parsing + GPS/attitude sync to frames + ENU conversion.

Supports: DJI .SRT (both the bracket format and the older GPS(lon,lat,alt)
format), telemetry embedded as a subtitle/data stream inside the video
(extracted via ffmpeg), .csv (auto column matching), .gpx, .kml, .json,
and MAVLink .tlog/.bin (via pymavlink).

Never raises on a missing/unparseable file — returns an empty TelemetryTrack
and lets the caller (io_detect / the pipeline) decide to fall back to
RGB-only reconstruction.
"""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass
class TelemetrySample:
    t: float  # seconds, relative to track start (or absolute unix if known)
    lat: float
    lon: float
    alt: float | None = None  # meters, absolute or relative — see `alt_is_relative`
    alt_is_relative: bool = False
    yaw: float | None = None  # degrees
    pitch: float | None = None
    roll: float | None = None
    gimbal_yaw: float | None = None
    gimbal_pitch: float | None = None
    gimbal_roll: float | None = None
    source_index: int = 0


@dataclass
class TelemetryTrack:
    samples: list[TelemetrySample] = field(default_factory=list)
    kind: str = "none"
    has_timestamps: bool = True
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __bool__(self) -> bool:
        return len(self.samples) > 0


# ---------------------------------------------------------------------------
# DJI SRT
# ---------------------------------------------------------------------------

_SRT_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
# Newer bracket format, e.g.:
# [latitude: 22.5726] [longitude: 88.3639] [rel_alt: 12.3 abs_alt: 45.6]
# optionally with [gb_yaw: 1.2] [gb_pitch: -30.0] [gb_roll: 0.0]
_BRACKET_LAT = re.compile(r"\[latitude\s*:\s*(-?\d+\.?\d*)\]")
_BRACKET_LON = re.compile(r"\[longitude\s*:\s*(-?\d+\.?\d*)\]")
_BRACKET_RELALT = re.compile(r"rel_alt\s*:\s*(-?\d+\.?\d*)")
_BRACKET_ABSALT = re.compile(r"abs_alt\s*:\s*(-?\d+\.?\d*)")
_BRACKET_GB_YAW = re.compile(r"gb_yaw\s*:\s*(-?\d+\.?\d*)")
_BRACKET_GB_PITCH = re.compile(r"gb_pitch\s*:\s*(-?\d+\.?\d*)")
_BRACKET_GB_ROLL = re.compile(r"gb_roll\s*:\s*(-?\d+\.?\d*)")
# Older format: GPS(lon,lat,alt)
_OLD_GPS = re.compile(r"GPS\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)")


def _srt_time_to_seconds(h, m, s, ms) -> float:
    ms = ms.ljust(3, "0")[:3]
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path: Path) -> TelemetryTrack:
    text = path.read_text(errors="ignore")
    blocks = re.split(r"\n\s*\n", text.strip())
    samples: list[TelemetrySample] = []
    fmt_seen = None

    for idx, block in enumerate(blocks):
        tmatch = _SRT_TIME_RE.search(block)
        t = None
        if tmatch:
            h1, m1, s1, ms1, *_ = tmatch.groups()
            t = _srt_time_to_seconds(h1, m1, s1, ms1)

        lat_m = _BRACKET_LAT.search(block)
        lon_m = _BRACKET_LON.search(block)
        if lat_m and lon_m:
            fmt_seen = "bracket"
            lat = float(lat_m.group(1))
            lon = float(lon_m.group(1))
            rel_m = _BRACKET_RELALT.search(block)
            abs_m = _BRACKET_ABSALT.search(block)
            alt = None
            alt_is_relative = False
            if abs_m:
                alt = float(abs_m.group(1))
            elif rel_m:
                alt = float(rel_m.group(1))
                alt_is_relative = True
            gy = _BRACKET_GB_YAW.search(block)
            gp = _BRACKET_GB_PITCH.search(block)
            gr = _BRACKET_GB_ROLL.search(block)
            samples.append(TelemetrySample(
                t=t if t is not None else float(idx),
                lat=lat, lon=lon, alt=alt, alt_is_relative=alt_is_relative,
                gimbal_yaw=float(gy.group(1)) if gy else None,
                gimbal_pitch=float(gp.group(1)) if gp else None,
                gimbal_roll=float(gr.group(1)) if gr else None,
                source_index=idx,
            ))
            continue

        old_m = _OLD_GPS.search(block)
        if old_m:
            fmt_seen = "gps_tuple"
            lon, lat, alt = (float(x) for x in old_m.groups())
            samples.append(TelemetrySample(
                t=t if t is not None else float(idx), lat=lat, lon=lon, alt=alt,
                source_index=idx,
            ))

    track = TelemetryTrack(samples=samples, kind="srt")
    if fmt_seen:
        track.notes.append(f"DJI SRT format detected: {fmt_seen}")
    if not samples:
        track.notes.append("SRT parsed but no GPS entries matched either known format")
    return track


# ---------------------------------------------------------------------------
# Embedded subtitle/data stream extraction (ffmpeg)
# ---------------------------------------------------------------------------

def extract_embedded_telemetry(video_path: Path) -> TelemetryTrack:
    """Probe for a subtitle/data stream inside the video and extract it.

    DJI drones sometimes mux the same SRT-style telemetry as a subtitle
    stream instead of shipping a sidecar .srt file.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        info = json.loads(probe.stdout)
    except Exception:
        return TelemetryTrack(kind="embedded", notes=["ffprobe failed on video for embedded telemetry check"])

    sub_streams = [
        s for s in info.get("streams", [])
        if s.get("codec_type") in ("subtitle", "data")
    ]
    if not sub_streams:
        return TelemetryTrack(kind="embedded", notes=["no subtitle/data stream found in video"])

    stream_index = sub_streams[0]["index"]
    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        res = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-map", f"0:{stream_index}", str(tmp_path)],
            capture_output=True, text=True, timeout=120,
        )
        if res.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
            return TelemetryTrack(kind="embedded", notes=["ffmpeg extraction of embedded stream produced no usable data"])
        track = parse_srt(tmp_path)
        track.kind = "embedded"
        track.notes.append(f"extracted from embedded stream index {stream_index}")
        return track
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CSV (auto column matching)
# ---------------------------------------------------------------------------

_COL_ALIASES = {
    "lat": {"lat", "latitude"},
    "lon": {"lon", "lng", "longitude"},
    "alt": {"alt", "altitude", "rel_alt", "relative_altitude", "abs_alt"},
    "time": {"time", "timestamp", "t", "time_s", "datetime"},
    "yaw": {"yaw", "heading"},
    "pitch": {"pitch"},
    "roll": {"roll"},
    "gimbal_yaw": {"gimbal_yaw", "gb_yaw"},
    "gimbal_pitch": {"gimbal_pitch", "gb_pitch"},
    "gimbal_roll": {"gimbal_roll", "gb_roll"},
}


def _match_columns(fieldnames: list[str]) -> dict[str, str]:
    lower = {f.lower().strip(): f for f in fieldnames}
    matched: dict[str, str] = {}
    for canon, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                matched[canon] = lower[alias]
                break
    return matched


def _parse_time_value(v: str, idx: int) -> float:
    v = v.strip()
    try:
        return float(v)
    except ValueError:
        pass
    for fmt_try in ("iso",):
        try:
            from datetime import datetime

            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
    return float(idx)


def parse_csv(path: Path) -> TelemetryTrack:
    with open(path, newline="", errors="ignore") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return TelemetryTrack(kind="csv", notes=["CSV has no header row"])
        cols = _match_columns(reader.fieldnames)
        if "lat" not in cols or "lon" not in cols:
            return TelemetryTrack(kind="csv", notes=[f"CSV missing lat/lon columns; found headers: {reader.fieldnames}"])

        samples = []
        has_time = "time" in cols
        for idx, row in enumerate(reader):
            try:
                lat = float(row[cols["lat"]])
                lon = float(row[cols["lon"]])
            except (ValueError, KeyError):
                continue
            alt = None
            if "alt" in cols:
                try:
                    alt = float(row[cols["alt"]])
                except (ValueError, KeyError):
                    alt = None
            t = _parse_time_value(row[cols["time"]], idx) if has_time else float(idx)

            def _f(key):
                if key not in cols:
                    return None
                try:
                    return float(row[cols[key]])
                except (ValueError, KeyError):
                    return None

            samples.append(TelemetrySample(
                t=t, lat=lat, lon=lon, alt=alt,
                alt_is_relative="rel_alt" in cols.get("alt", ""),
                yaw=_f("yaw"), pitch=_f("pitch"), roll=_f("roll"),
                gimbal_yaw=_f("gimbal_yaw"), gimbal_pitch=_f("gimbal_pitch"), gimbal_roll=_f("gimbal_roll"),
                source_index=idx,
            ))
        track = TelemetryTrack(samples=samples, kind="csv", has_timestamps=has_time)
        if not has_time:
            track.notes.append("CSV had no time column; using row index, will interpolate proportionally over duration")
        return track


# ---------------------------------------------------------------------------
# GPX
# ---------------------------------------------------------------------------

def parse_gpx(path: Path) -> TelemetryTrack:
    try:
        tree = ET.parse(path)
    except Exception as e:
        return TelemetryTrack(kind="gpx", notes=[f"GPX parse failed: {e}"])
    root = tree.getroot()
    ns_match = re.match(r"\{(.*)\}", root.tag)
    ns = {"g": ns_match.group(1)} if ns_match else {}

    def tag(name):
        return f"g:{name}" if ns else name

    samples = []
    pts = root.findall(f".//{tag('trkpt')}", ns) or root.findall(f".//{tag('wpt')}", ns)
    for idx, pt in enumerate(pts):
        try:
            lat = float(pt.attrib["lat"])
            lon = float(pt.attrib["lon"])
        except (KeyError, ValueError):
            continue
        ele_el = pt.find(tag("ele"), ns)
        alt = float(ele_el.text) if ele_el is not None and ele_el.text else None
        time_el = pt.find(tag("time"), ns)
        t = None
        if time_el is not None and time_el.text:
            try:
                from datetime import datetime

                t = datetime.fromisoformat(time_el.text.replace("Z", "+00:00")).timestamp()
            except Exception:
                t = None
        samples.append(TelemetrySample(t=t if t is not None else float(idx), lat=lat, lon=lon, alt=alt, source_index=idx))

    track = TelemetryTrack(samples=samples, kind="gpx", has_timestamps=all(s.t is not None for s in samples))
    return track


# ---------------------------------------------------------------------------
# KML
# ---------------------------------------------------------------------------

def parse_kml(path: Path) -> TelemetryTrack:
    try:
        tree = ET.parse(path)
    except Exception as e:
        return TelemetryTrack(kind="kml", notes=[f"KML parse failed: {e}"])
    root = tree.getroot()
    ns_match = re.match(r"\{(.*)\}", root.tag)
    ns = {"k": ns_match.group(1)} if ns_match else {}

    def tag(name):
        return f"k:{name}" if ns else name

    samples = []
    idx = 0
    for coord_el in root.findall(f".//{tag('coordinates')}", ns):
        if not coord_el.text:
            continue
        for tup in coord_el.text.strip().split():
            parts = tup.split(",")
            if len(parts) < 2:
                continue
            lon, lat = float(parts[0]), float(parts[1])
            alt = float(parts[2]) if len(parts) > 2 else None
            samples.append(TelemetrySample(t=float(idx), lat=lat, lon=lon, alt=alt, source_index=idx))
            idx += 1

    track = TelemetryTrack(samples=samples, kind="kml", has_timestamps=False)
    if samples:
        track.notes.append("KML has no per-point timestamps; using sequence index, will interpolate proportionally")
    return track


# ---------------------------------------------------------------------------
# JSON (generic list-of-records)
# ---------------------------------------------------------------------------

def parse_json_telemetry(path: Path) -> TelemetryTrack:
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return TelemetryTrack(kind="json", notes=[f"JSON parse failed: {e}"])

    records = data
    if isinstance(data, dict):
        for key in ("samples", "points", "telemetry", "track", "data"):
            if key in data and isinstance(data[key], list):
                records = data[key]
                break
        else:
            return TelemetryTrack(kind="json", notes=["JSON is an object but no list of records found under common keys"])
    if not isinstance(records, list):
        return TelemetryTrack(kind="json", notes=["JSON telemetry is not a list of records"])

    aliases = {k: v for k, v in _COL_ALIASES.items()}
    samples = []
    has_time = True
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        lower_rec = {k.lower(): v for k, v in rec.items()}

        def find(canon):
            for alias in aliases[canon]:
                if alias in lower_rec:
                    return lower_rec[alias]
            return None

        lat, lon = find("lat"), find("lon")
        if lat is None or lon is None:
            continue
        t_raw = find("time")
        if t_raw is None:
            has_time = False
            t = float(idx)
        else:
            t = t_raw if isinstance(t_raw, (int, float)) else _parse_time_value(str(t_raw), idx)
        samples.append(TelemetrySample(
            t=float(t), lat=float(lat), lon=float(lon),
            alt=float(find("alt")) if find("alt") is not None else None,
            yaw=float(find("yaw")) if find("yaw") is not None else None,
            pitch=float(find("pitch")) if find("pitch") is not None else None,
            roll=float(find("roll")) if find("roll") is not None else None,
            source_index=idx,
        ))
    return TelemetryTrack(samples=samples, kind="json", has_timestamps=has_time)


# ---------------------------------------------------------------------------
# MAVLink (.tlog / .bin) via pymavlink
# ---------------------------------------------------------------------------

def parse_mavlink(path: Path) -> TelemetryTrack:
    try:
        from pymavlink import mavutil
    except ImportError:
        return TelemetryTrack(kind="mavlink", notes=["pymavlink not installed — cannot parse MAVLink log"])

    try:
        conn = mavutil.mavlink_connection(str(path))
    except Exception as e:
        return TelemetryTrack(kind="mavlink", notes=[f"failed to open MAVLink log: {e}"])

    samples = []
    idx = 0
    attitude_cache = {"yaw": None, "pitch": None, "roll": None}
    while True:
        try:
            msg = conn.recv_match(blocking=False)
        except Exception:
            break
        if msg is None:
            break
        mtype = msg.get_type()
        if mtype == "ATTITUDE":
            attitude_cache["yaw"] = math.degrees(msg.yaw)
            attitude_cache["pitch"] = math.degrees(msg.pitch)
            attitude_cache["roll"] = math.degrees(msg.roll)
        elif mtype in ("GLOBAL_POSITION_INT", "GPS_RAW_INT"):
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = getattr(msg, "relative_alt", None)
            alt = (alt / 1000.0) if alt is not None else (getattr(msg, "alt", 0) / 1000.0)
            t = getattr(msg, "time_boot_ms", idx * 100) / 1000.0
            samples.append(TelemetrySample(
                t=t, lat=lat, lon=lon, alt=alt, alt_is_relative=hasattr(msg, "relative_alt"),
                yaw=attitude_cache["yaw"], pitch=attitude_cache["pitch"], roll=attitude_cache["roll"],
                source_index=idx,
            ))
            idx += 1

    return TelemetryTrack(samples=samples, kind="mavlink")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_PARSERS = {
    "srt": parse_srt,
    "csv": parse_csv,
    "gpx": parse_gpx,
    "kml": parse_kml,
    "json": parse_json_telemetry,
    "mavlink": parse_mavlink,
}


def parse_telemetry(path: Path, kind: str) -> TelemetryTrack:
    parser = _PARSERS.get(kind)
    if parser is None:
        return TelemetryTrack(kind=kind, notes=[f"no parser for telemetry kind '{kind}'"])
    try:
        return parser(path)
    except Exception as e:
        return TelemetryTrack(kind=kind, notes=[f"parser raised {type(e).__name__}: {e}"])


# ---------------------------------------------------------------------------
# GPS -> local ENU (meters) + UTM zone/EPSG
# ---------------------------------------------------------------------------

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563


def utm_zone_epsg(lat: float, lon: float) -> tuple[int, int]:
    zone = int(math.floor((lon + 180) / 6) + 1)
    epsg = (32600 if lat >= 0 else 32700) + zone
    return zone, epsg


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    e2 = WGS84_F * (2 - WGS84_F)
    n = WGS84_A / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    x = (n + alt_m) * math.cos(lat) * math.cos(lon)
    y = (n + alt_m) * math.cos(lat) * math.sin(lon)
    z = (n * (1 - e2) + alt_m) * math.sin(lat)
    return x, y, z


def ecef_to_enu(x, y, z, lat0_deg, lon0_deg, alt0_m) -> tuple[float, float, float]:
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
    dx, dy, dz = x - x0, y - y0, z - z0
    lat0, lon0 = math.radians(lat0_deg), math.radians(lon0_deg)
    sl, cl = math.sin(lat0), math.cos(lat0)
    so, co = math.sin(lon0), math.cos(lon0)
    e = -so * dx + co * dy
    n = -sl * co * dx - sl * so * dy + cl * dz
    u = cl * co * dx + cl * so * dy + sl * dz
    return e, n, u


def enu_to_ecef(e: float, n: float, u: float, lat0_deg: float, lon0_deg: float, alt0_m: float) -> tuple[float, float, float]:
    """Inverse of ecef_to_enu — used to convert an aligned world-frame
    (post-align.py) point or camera track back to ECEF/geodetic for
    trajectory.geojson/.kml export, which need lon/lat."""
    lat0, lon0 = math.radians(lat0_deg), math.radians(lon0_deg)
    sl, cl = math.sin(lat0), math.cos(lat0)
    so, co = math.sin(lon0), math.cos(lon0)
    # Transpose of the rotation used in ecef_to_enu (it's orthonormal).
    dx = -so * e - sl * co * n + cl * co * u
    dy = co * e - sl * so * n + cl * so * u
    dz = cl * n + sl * u
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
    return x0 + dx, y0 + dy, z0 + dz


def ecef_to_geodetic(x: float, y: float, z: float, n_iters: int = 5) -> tuple[float, float, float]:
    """Iterative ECEF -> geodetic (lat_deg, lon_deg, alt_m). A handful of
    Newton iterations on the WGS84 ellipsoid converges to sub-millimeter
    accuracy, which is more than enough given our GPS input is already
    consumer-grade at best."""
    lon = math.atan2(y, x)
    e2 = WGS84_F * (2 - WGS84_F)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - e2))
    alt = 0.0
    for _ in range(n_iters):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1 - e2 * sin_lat ** 2)
        alt = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1 - e2 * n / (n + alt)))
    return math.degrees(lat), math.degrees(lon), alt


def enu_to_geodetic(e: float, n: float, u: float, lat0_deg: float, lon0_deg: float, alt0_m: float) -> tuple[float, float, float]:
    x, y, z = enu_to_ecef(e, n, u, lat0_deg, lon0_deg, alt0_m)
    return ecef_to_geodetic(x, y, z)


def track_to_enu(track: TelemetryTrack) -> tuple[list[tuple[float, float, float]], tuple[float, float, float], tuple[int, int]]:
    """Returns (enu_points, origin(lat,lon,alt), (utm_zone, epsg))."""
    if not track:
        return [], (0.0, 0.0, 0.0), (0, 0)
    origin = track.samples[0]
    alt0 = origin.alt if origin.alt is not None else 0.0
    origin_tup = (origin.lat, origin.lon, alt0)
    enu = []
    for s in track.samples:
        alt = s.alt if s.alt is not None else 0.0
        x, y, z = geodetic_to_ecef(s.lat, s.lon, alt)
        enu.append(ecef_to_enu(x, y, z, *origin_tup))
    zone_epsg = utm_zone_epsg(origin.lat, origin.lon)
    return enu, origin_tup, zone_epsg


# ---------------------------------------------------------------------------
# Collinearity index (PCA eigenvalue ratio on the ENU ground track)
# ---------------------------------------------------------------------------

def collinearity_index(enu_points: list[tuple[float, float, float]]) -> float:
    """Ratio of the 2nd/1st eigenvalue of the horizontal (E,N) covariance.

    ~0 => perfectly straight line (single-pass flight: cannot constrain
    roll around the flight axis from GPS alone).
    ~1 => isotropic spread (e.g. a grid/orbit survey).
    """
    if len(enu_points) < 3:
        return 0.0
    import numpy as np

    pts = np.array([(e, n) for e, n, _ in enu_points])
    pts = pts - pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    if eigvals[0] <= 1e-9:
        return 0.0
    return float(eigvals[1] / eigvals[0])


# ---------------------------------------------------------------------------
# Time-sync telemetry samples to frame timestamps
# ---------------------------------------------------------------------------

def sync_to_frame_times(track: TelemetryTrack, frame_times_s: list[float]) -> list[TelemetrySample | None]:
    """Interpolate telemetry to each frame timestamp (linear on lat/lon/alt/attitude).

    If the track has no real timestamps, `frame_times_s` values are ignored
    and interpolation is proportional over track duration/frame count
    (caller must have already logged this via track.notes).
    """
    if not track:
        return [None for _ in frame_times_s]

    ts = [s.t for s in track.samples]
    if not track.has_timestamps:
        n = len(track.samples)
        span = max(len(frame_times_s) - 1, 1)
        proportional = [i / span for i in range(len(frame_times_s))]
        idx_float = [p * (n - 1) for p in proportional]
        out = []
        for f in idx_float:
            lo = int(math.floor(f))
            hi = min(lo + 1, n - 1)
            frac = f - lo
            out.append(_lerp_sample(track.samples[lo], track.samples[hi], frac))
        return out

    out = []
    for t in frame_times_s:
        if t <= ts[0]:
            out.append(track.samples[0])
            continue
        if t >= ts[-1]:
            out.append(track.samples[-1])
            continue
        lo = 0
        hi = len(ts) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ts[mid] <= t:
                lo = mid
            else:
                hi = mid
        span = ts[hi] - ts[lo]
        frac = (t - ts[lo]) / span if span > 1e-9 else 0.0
        out.append(_lerp_sample(track.samples[lo], track.samples[hi], frac))
    return out


def _lerp_sample(a: TelemetrySample, b: TelemetrySample, frac: float) -> TelemetrySample:
    def lerp(x, y):
        if x is None or y is None:
            return x if x is not None else y
        return x + (y - x) * frac

    return TelemetrySample(
        t=lerp(a.t, b.t), lat=lerp(a.lat, b.lat), lon=lerp(a.lon, b.lon), alt=lerp(a.alt, b.alt),
        alt_is_relative=a.alt_is_relative,
        yaw=lerp(a.yaw, b.yaw), pitch=lerp(a.pitch, b.pitch), roll=lerp(a.roll, b.roll),
        gimbal_yaw=lerp(a.gimbal_yaw, b.gimbal_yaw), gimbal_pitch=lerp(a.gimbal_pitch, b.gimbal_pitch),
        gimbal_roll=lerp(a.gimbal_roll, b.gimbal_roll),
    )

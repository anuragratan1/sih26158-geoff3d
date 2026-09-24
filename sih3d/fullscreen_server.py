"""Local HTTP server + cloudflared quick tunnel, serving /kaggle/working/outputs/
so the same self-contained viewer.html (see viewer.py — already embeds its
point cloud/mesh/trajectories inline, no sibling-file fetches needed) can be
opened full-screen and shared with teammates during the session, not just
viewed inline in the notebook.

Every failure here is logged and degrades gracefully — never raises into the
caller, never blocks the pipeline. The "Open full-screen viewer" link in the
dashboard header simply doesn't appear if the tunnel couldn't be established;
the inline (anywidget) viewer keeps working regardless.
"""

from __future__ import annotations

import functools
import http.server
import platform
import re
import shutil
import socket
import stat
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from .events import EventBus

# Kaggle is Linux — that's the only platform this auto-downloads a binary
# for. On other platforms (e.g. local dev on macOS) it just logs and skips;
# install cloudflared manually there if you want to test this path locally.
_CLOUDFLARED_LINUX_URLS = {
    "x86_64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    "aarch64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
}


def ensure_cloudflared(bus: EventBus, dest: Path = Path("/usr/local/bin/cloudflared")) -> Path | None:
    """Best-effort: download the cloudflared binary if it's not already on
    PATH (Kaggle images don't ship it). Never raises."""
    existing = shutil.which("cloudflared")
    if existing:
        return Path(existing)

    if platform.system() != "Linux":
        bus.log(
            f"cloudflared auto-download only supported on Linux (this is {platform.system()}) — "
            f"full-screen viewer link unavailable unless cloudflared is installed manually",
            level="warn",
        )
        return None

    url = _CLOUDFLARED_LINUX_URLS.get(platform.machine())
    if url is None:
        bus.log(f"No known cloudflared build for architecture {platform.machine()} — full-screen viewer link unavailable", level="warn")
        return None

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dest))
        dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
        bus.log(f"Downloaded cloudflared to {dest}")
        return dest
    except Exception as e:
        bus.log(f"cloudflared download failed ({e}) — full-screen viewer link unavailable", level="warn")
        return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_http_server(directory: Path, port: int, bus: EventBus) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="sih3d-http-server")
    thread.start()
    bus.log(f"Local HTTP server serving {directory} on 127.0.0.1:{port}")


_TRYCLOUDFLARE_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


def start_cloudflared_tunnel(port: int, bus: EventBus, label: str = "tunnel", timeout_s: float = 20.0) -> str | None:
    """Starts a cloudflared quick tunnel pointing at a local port already
    listening on 127.0.0.1. Shared by start_fullscreen_server() (outputs/)
    and live_page.py (the live reconstruction page) so the cloudflared
    process-management/URL-parsing logic exists exactly once. Returns the
    public https://*.trycloudflare.com URL, or None if anything failed —
    logged, never raised."""
    cloudflared = shutil.which("cloudflared")
    if cloudflared is None and Path("/usr/local/bin/cloudflared").exists():
        cloudflared = "/usr/local/bin/cloudflared"
    if cloudflared is None:
        bus.log(f"cloudflared not available — {label} link unavailable", level="warn")
        return None

    try:
        proc = subprocess.Popen(
            [cloudflared, "tunnel", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except Exception as e:
        bus.log(f"cloudflared failed to start ({e}) — {label} link unavailable", level="warn")
        return None

    url = None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            continue
        m = _TRYCLOUDFLARE_RE.search(line)
        if m:
            url = m.group(0)
            break

    if url is None:
        bus.log(f"cloudflared did not produce a tunnel URL within {timeout_s:.0f}s — {label} link unavailable", level="warn")
        try:
            proc.terminate()
        except Exception:
            pass
        return None

    def _drain_stdout() -> None:
        try:
            for _ in proc.stdout:
                pass  # discard further cloudflared chatter — never spam the notebook
        except Exception:
            pass

    threading.Thread(target=_drain_stdout, daemon=True, name=f"sih3d-cloudflared-drain-{label}").start()

    bus.log(f"{label} tunnel ready: {url}")
    return url


def start_fullscreen_server(output_dir: Path, bus: EventBus, timeout_s: float = 20.0) -> str | None:
    """Starts a local HTTP server over `output_dir` and a cloudflared quick
    tunnel pointing at it. Returns the public https://*.trycloudflare.com
    base URL (append "/viewer.html"), or None if anything failed."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        port = _free_port()
        _start_http_server(output_dir, port, bus)
    except Exception as e:
        bus.log(f"Local HTTP server failed to start ({e}) — full-screen viewer unavailable", level="warn")
        return None

    return start_cloudflared_tunnel(port, bus, label="full-screen viewer", timeout_s=timeout_s)

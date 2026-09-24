"""Thread-safe event bus connecting the pipeline thread to the dashboard.

The pipeline runs in a background thread and calls `bus.publish(...)`.
The dashboard (main thread, ipywidgets) polls `bus.drain()` on a timer.
No locks are held across UI rendering — publish() is O(1) and never blocks
on a full queue (it drops the oldest event instead of blocking the pipeline).
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    STAGE_START = "stage_start"
    STAGE_PROGRESS = "stage_progress"
    STAGE_DONE = "stage_done"
    STAGE_FALLBACK = "stage_fallback"
    STAGE_ERROR = "stage_error"
    FRAME_DECODED = "frame_decoded"
    KEYFRAME_SELECTED = "keyframe_rejected"
    KEYFRAME_ACCEPTED = "keyframe_accepted"
    KEYFRAME_REJECTED = "keyframe_rejected"
    GPU_SAMPLE = "gpu_sample"
    TRAJECTORY_POINT = "trajectory_point"
    GEOMETRY_CHUNK = "geometry_chunk"
    POINTCLOUD_GROWTH = "pointcloud_growth"
    MESH_PREVIEW = "mesh_preview"
    LOG = "log"
    PIPELINE_DONE = "pipeline_done"
    PIPELINE_ERROR = "pipeline_error"
    HEADER_UPDATE = "header_update"       # bootstrap.py: video/telemetry/GPU info becomes known after detection
    SETUP_STAGE_START = "setup_stage_start"    # bootstrap phases: env_check, install, detect_inputs
    SETUP_STAGE_DONE = "setup_stage_done"
    SETUP_STAGE_FALLBACK = "setup_stage_fallback"
    SETUP_STAGE_ERROR = "setup_stage_error"
    INSTALL_PROGRESS = "install_progress"  # one per package: pending/installing/ok/failed + timing
    TUNNEL_READY = "tunnel_ready"           # fullscreen_server.py: cloudflared URL (or failure) known
    RASTERS_READY = "rasters_ready"         # export.py's DSM/orthomosaic/coverage paths, for the Rasters tab


@dataclass
class Event:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class EventBus:
    """Bounded, drop-oldest, thread-safe pub/sub queue.

    Bounded so a stalled UI consumer can never cause the producer (pipeline
    thread) to block or accumulate unbounded memory.
    """

    def __init__(self, maxsize: int = 4096):
        self._q: "queue.Queue[Event]" = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._log_tail: list[str] = []
        self._log_tail_max = 200

    def publish(self, type: EventType, **payload: Any) -> None:
        evt = Event(type=type, payload=payload)
        if type == EventType.LOG:
            with self._lock:
                self._log_tail.append(str(payload.get("message", "")))
                if len(self._log_tail) > self._log_tail_max:
                    self._log_tail.pop(0)
        try:
            self._q.put_nowait(evt)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(evt)
            except queue.Full:
                pass

    def log(self, message: str, level: str = "info") -> None:
        self.publish(EventType.LOG, message=message, level=level)

    def drain(self, max_events: int = 500) -> list[Event]:
        """Non-blocking: pull up to max_events currently queued events."""
        out: list[Event] = []
        for _ in range(max_events):
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out

    def recent_logs(self) -> list[str]:
        with self._lock:
            return list(self._log_tail)

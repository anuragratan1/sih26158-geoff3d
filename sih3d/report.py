"""Builds report.json / report.html from the pipeline's event stream.

Deliberately decoupled from EventBus.drain() itself (a queue.Queue can only
have one destructive consumer) — the orchestration loop drains the bus once
per tick and fans each event out to both `dashboard.on_event()` and
`ReportBuilder.on_event()`. This module only accumulates state from events
plus a handful of explicit setters for values that aren't naturally
event-shaped (point/face counts, alignment RMSE, etc.).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .events import Event, EventType

STAGE_ORDER = [
    "frame_extraction",
    "camera_trajectory",
    "geometric_reconstruction",
    "large_scale_alignment",
    "dense_point_cloud",
    "mesh_textured_model",
]

_FALLBACK_KEYWORDS = ("fallback", "falling back", "unavailable", "skip")


@dataclass
class StageInfo:
    name: str
    status: str = "pending"
    start_ts: float | None = None
    end_ts: float | None = None
    fallback_notes: list[str] = field(default_factory=list)


class ReportBuilder:
    def __init__(self):
        self.stages: dict[str, StageInfo] = {name: StageInfo(name=name) for name in STAGE_ORDER}
        self.warnings: list[str] = []
        self.fallbacks: list[str] = []
        self.backbone_used: str | None = None
        self.prior_mode: str | None = None
        self.checkpoint_source: str | None = None
        self.mode: str | None = None
        self.gpus: list[str] = []
        self.georeferenced: bool | None = None
        self.collinearity_index: float | None = None
        self.alignment_rmse_m: float | None = None
        self.keyframe_count = 0
        self.rejected_keyframe_count = 0
        self.point_count = 0
        self.mesh_faces = 0
        self.outputs: list[dict] = []
        self.started_at = time.time()
        self.finished_at: float | None = None
        self._gpu_samples: list[tuple[float, int, float]] = []

    # -- event-driven updates -------------------------------------------------

    def on_event(self, evt: Event) -> None:
        p = evt.payload
        if evt.type == EventType.STAGE_START:
            st = self.stages.get(p.get("stage"))
            if st:
                st.status, st.start_ts = "running", evt.ts
        elif evt.type == EventType.STAGE_DONE:
            st = self.stages.get(p.get("stage"))
            if st:
                st.status, st.end_ts = "done", evt.ts
        elif evt.type == EventType.STAGE_FALLBACK:
            st = self.stages.get(p.get("stage"))
            note = p.get("note", "")
            if st:
                st.status = "fallback" if st.status == "running" else st.status
                st.fallback_notes.append(note)
            self.fallbacks.append(f"[{p.get('stage')}] {note}" if p.get("stage") else note)
        elif evt.type == EventType.STAGE_ERROR:
            st = self.stages.get(p.get("stage"))
            if st:
                st.status, st.end_ts = "error", evt.ts
        elif evt.type == EventType.GPU_SAMPLE:
            self._gpu_samples.append((p["ts"], p["index"], p["util_pct"]))
        elif evt.type == EventType.KEYFRAME_ACCEPTED:
            self.keyframe_count += 1
        elif evt.type == EventType.KEYFRAME_REJECTED:
            self.rejected_keyframe_count += 1
        elif evt.type == EventType.LOG:
            level, msg = p.get("level", "info"), p.get("message", "")
            if level == "warn":
                self.warnings.append(msg)
                if any(k in msg.lower() for k in _FALLBACK_KEYWORDS):
                    self.fallbacks.append(msg)

    # -- explicit setters for non-event-shaped values --------------------------

    def set_run_config(self, mode: str, backbone: str, prior_mode: str, checkpoint_source: str, gpus: list[str]) -> None:
        self.mode, self.backbone_used, self.prior_mode = mode, backbone, prior_mode
        self.checkpoint_source, self.gpus = checkpoint_source, gpus

    def set_geometry_summary(
        self, georeferenced: bool, collinearity_index: float | None, alignment_rmse_m: float | None,
        point_count: int, mesh_faces: int,
    ) -> None:
        self.georeferenced = georeferenced
        self.collinearity_index = collinearity_index
        self.alignment_rmse_m = alignment_rmse_m
        self.point_count = point_count
        self.mesh_faces = mesh_faces

    def add_output(self, status) -> None:
        """`status`: export.ExportStatus (kept as a plain dict here to avoid
        a circular import — export.py already knows its own shape)."""
        self.outputs.append({
            "name": status.name, "ok": status.ok, "skipped_reason": status.skipped_reason,
            "size_bytes": status.size_bytes, "path": str(status.path) if status.path else None,
        })

    def finish(self) -> None:
        self.finished_at = time.time()

    # -- derived values ---------------------------------------------------------

    def stage_avg_gpu_util(self, name: str) -> dict[int, float]:
        st = self.stages.get(name)
        if st is None or st.start_ts is None:
            return {}
        end = st.end_ts or time.time()
        by_gpu: dict[int, list[float]] = {}
        for ts, idx, util in self._gpu_samples:
            if st.start_ts <= ts <= end:
                by_gpu.setdefault(idx, []).append(util)
        return {idx: sum(v) / len(v) for idx, v in by_gpu.items()}

    def to_dict(self) -> dict:
        stages_out = []
        for name, st in self.stages.items():
            elapsed = (st.end_ts - st.start_ts) if (st.start_ts and st.end_ts) else None
            stages_out.append({
                "name": name, "status": st.status, "elapsed_s": elapsed,
                "avg_gpu_util_pct": self.stage_avg_gpu_util(name),
                "fallback_notes": st.fallback_notes,
            })
        total_elapsed = (self.finished_at - self.started_at) if self.finished_at else None
        return {
            "mode": self.mode, "backbone": self.backbone_used, "prior_mode": self.prior_mode,
            "checkpoint_source": self.checkpoint_source, "gpus": self.gpus,
            "georeferenced": self.georeferenced, "collinearity_index": self.collinearity_index,
            "alignment_rmse_m": self.alignment_rmse_m,
            "keyframe_count": self.keyframe_count, "rejected_keyframe_count": self.rejected_keyframe_count,
            "point_count": self.point_count, "mesh_faces": self.mesh_faces,
            "stages": stages_out, "fallbacks_triggered": self.fallbacks, "warnings": self.warnings,
            "outputs": self.outputs, "started_at": self.started_at, "finished_at": self.finished_at,
            "total_elapsed_s": total_elapsed,
        }

    def low_util_stages(self, threshold: float = 50.0) -> list[tuple[str, float]]:
        out = []
        for name in self.stages:
            util = self.stage_avg_gpu_util(name)
            if util:
                worst = min(util.values())
                if worst < threshold:
                    out.append((name, worst))
        return out

    # -- writers -----------------------------------------------------------

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def write_html(self, path: Path, dashboard_snapshot_html: str = "") -> None:
        d = self.to_dict()

        def fmt_s(v):
            return f"{v:.1f}s" if v is not None else "-"

        rows = "".join(
            f"<tr><td>{s['name']}</td><td>{s['status']}</td><td>{fmt_s(s['elapsed_s'])}</td>"
            f"<td>{', '.join(f'GPU{k}: {v:.0f}%' for k, v in s['avg_gpu_util_pct'].items()) or '-'}</td>"
            f"<td>{'; '.join(s['fallback_notes']) or '-'}</td></tr>"
            for s in d["stages"]
        )
        low_util = self.low_util_stages()
        low_util_html = "".join(f"<li>Stage '{name}' averaged {util:.0f}% GPU utilization (below 50%)</li>" for name, util in low_util)
        outputs_rows = "".join(
            f"<tr><td>{o['name']}</td><td>{'OK' if o['ok'] else 'SKIPPED: ' + str(o.get('skipped_reason', ''))}</td>"
            f"<td>{(o.get('size_bytes', 0) or 0) / 1e6:.2f} MB</td></tr>"
            for o in d["outputs"]
        )
        fallback_items = "".join(f"<li class='warn'>{f}</li>" for f in d["fallbacks_triggered"]) or "<li>none</li>"

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SIH26158 Pipeline Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 2rem; background:#0b0f14; color:#e6edf3; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
th, td {{ border: 1px solid #30363d; padding: 6px 10px; text-align: left; font-size: 0.9rem; }}
th {{ background: #161b22; }}
h1, h2 {{ font-weight: 600; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:4px; background:#1f6feb; margin-right:6px; }}
.badge.warn {{ background:#9e6a03; }}
.warn {{ color: #d29922; }}
</style></head>
<body>
<h1>SIH26158 — Drone Video to Georeferenced 3D Model</h1>
<p>
<span class="badge">Mode: {d['mode']}</span>
<span class="badge">Backbone: {d['backbone']} ({d['prior_mode']})</span>
<span class="badge {'warn' if not d['georeferenced'] else ''}">{'Georeferenced' if d['georeferenced'] else 'NOT GEOREFERENCED'}</span>
<span class="badge">Total time: {fmt_s(d['total_elapsed_s'])}</span>
</p>
<h2>Stages</h2>
<table><tr><th>Stage</th><th>Status</th><th>Elapsed</th><th>Avg GPU Util</th><th>Fallback</th></tr>{rows}</table>
<h2>Geometry</h2>
<ul>
<li>Keyframes: {d['keyframe_count']} accepted, {d['rejected_keyframe_count']} rejected</li>
<li>Points: {d['point_count']:,}</li>
<li>Mesh faces: {d['mesh_faces']:,}</li>
<li>Collinearity index: {d['collinearity_index']}</li>
<li>Alignment RMSE vs GPS: {d['alignment_rmse_m']} m</li>
</ul>
<h2>Outputs</h2>
<table><tr><th>File</th><th>Status</th><th>Size</th></tr>{outputs_rows}</table>
<h2>Fallbacks triggered ({len(d['fallbacks_triggered'])})</h2>
<ul>{fallback_items}</ul>
{"<h2>GPU Utilization Warnings</h2><ul>" + low_util_html + "</ul>" if low_util_html else ""}
<h2>Final Dashboard Snapshot</h2>
{dashboard_snapshot_html}
</body></html>"""
        path.write_text(html)

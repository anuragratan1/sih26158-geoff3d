"""Live ipywidgets dashboard, consuming events drained from the pipeline's
EventBus in the notebook's main thread (the pipeline itself runs in a
background thread — see the "Launch" cell). Refresh is throttled to <=2Hz
via `render_if_due()`; `on_event()` itself is cheap (just accumulates state)
so it never slows down event draining.

ipywidgets objects can be constructed and updated outside a live Jupyter
kernel (they just have no frontend to render to), so the state-accumulation
logic here is unit-testable headlessly; the actual visual layout can only be
verified in a real notebook.
"""

from __future__ import annotations

import io
import time
from collections import deque

import numpy as np

from .events import Event, EventType

STAGE_LABELS = {
    "frame_extraction": "Frame Extraction",
    "camera_trajectory": "Camera Trajectory / Poses",
    "geometric_reconstruction": "Geometric Reconstruction",
    "large_scale_alignment": "Large-Scale Alignment",
    "dense_point_cloud": "Dense 3D Point Cloud",
    "mesh_textured_model": "Mesh / Textured 3D Model",
}
STAGE_ORDER = list(STAGE_LABELS.keys())

_STATUS_COLOR = {
    "pending": "#30363d", "running": "#1f6feb", "done": "#238636",
    "fallback": "#9e6a03", "error": "#da3633",
}


def _encode_png(img_rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.clip(img_rgb, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


class StageCard:
    def __init__(self, W, key: str, label: str, has_progress: bool = True):
        self.key = key
        self.status = "pending"
        self.elapsed_s = 0.0
        self.fallback_note: str | None = None
        self._start_ts: float | None = None

        self.label = W.HTML(f"<b>{label}</b>")
        self.status_dot = W.HTML(self._dot_html("pending"))
        self.progress = W.FloatProgress(value=0.0, min=0.0, max=1.0, layout=W.Layout(width="90%")) if has_progress else None
        self.elapsed_label = W.Label("")
        children = [W.HBox([self.status_dot, self.label])]
        if self.progress is not None:
            children.append(self.progress)
        children.append(self.elapsed_label)
        self.widget = W.VBox(children, layout=W.Layout(border="1px solid #30363d", padding="6px", width="150px"))

    def _dot_html(self, status: str) -> str:
        color = _STATUS_COLOR.get(status, "#30363d")
        return f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin-right:6px;"></span>'

    def start(self, ts: float) -> None:
        self.status, self._start_ts = "running", ts
        self.status_dot.value = self._dot_html("running")

    def done(self, ts: float) -> None:
        self.status = "done"
        self.elapsed_s = ts - self._start_ts if self._start_ts else self.elapsed_s
        self.status_dot.value = self._dot_html("done")
        if self.progress is not None:
            self.progress.value = 1.0
        self.elapsed_label.value = f"{self.elapsed_s:.1f}s"

    def fallback(self, note: str) -> None:
        self.status = "fallback"
        self.fallback_note = note
        self.status_dot.value = self._dot_html("fallback")
        self.label.value = f"<b>{self.label.value.split('</b>')[0].replace('<b>', '')}</b> <span style='color:#9e6a03'>[fallback]</span>"

    def error(self, ts: float) -> None:
        self.status = "error"
        self.status_dot.value = self._dot_html("error")

    def set_progress(self, frac: float) -> None:
        if self.progress is not None:
            self.progress.value = max(0.0, min(1.0, frac))

    def tick_elapsed(self, now: float) -> None:
        if self.status == "running" and self._start_ts is not None:
            self.elapsed_label.value = f"{now - self._start_ts:.1f}s"


class Dashboard:
    def __init__(self, header_info: dict, refresh_hz: float = 2.0):
        import ipywidgets as W

        self._W = W
        self.refresh_interval = 1.0 / max(refresh_hz, 0.1)
        self._last_render = 0.0
        self._dirty = True

        self.start_ts = time.time()
        self.total_frames_expected = header_info.get("total_frames_expected", 0)
        self.frames_processed = 0

        self._build_header(header_info)
        self._build_pipeline_strip()
        self._build_gpu_panel(header_info.get("gpu_count", 0), header_info.get("gpu_names", []))
        self._build_frames_panel()
        self._build_trajectory_panel()
        self._build_geometry_panel()
        self._build_live3d_panel()
        self._build_log_panel()
        self._build_results_card()

        self.root = W.VBox([
            self.header_box,
            self.pipeline_box,
            W.HBox([self.gpu_box, self.frames_box]),
            W.HBox([self.trajectory_box, self.geometry_box]),
            self.live3d_box,
            self.log_box,
            self.results_box,
        ])

    def display(self) -> None:
        from IPython.display import display

        display(self.root)

    # -- construction ---------------------------------------------------------

    def _build_header(self, info: dict) -> None:
        W = self._W
        lines = [
            f"<b>Video:</b> {info.get('video_name', '?')} "
            f"({info.get('resolution', '?')}, {info.get('fps', '?')} fps, {info.get('duration_s', 0):.0f}s)",
            f"<b>Telemetry:</b> {info.get('telemetry_type', 'none')} "
            f"({info.get('telemetry_sample_count', 0)} samples, sync: {info.get('sync_method', '-')})",
            f"<b>Mode:</b> {info.get('mode', '?')} &nbsp; "
            f"<b>GPUs:</b> {', '.join(info.get('gpu_names', [])) or 'none detected'} &nbsp; "
            f"<b>Backbone:</b> {info.get('backbone', '?')} ({info.get('prior_mode', '?')})",
        ]
        self.header_html = W.HTML("<br>".join(lines))
        self.overall_progress = W.FloatProgress(value=0.0, min=0.0, max=1.0, description="Overall:", layout=W.Layout(width="60%"))
        self.elapsed_eta_label = W.Label("elapsed 0s")
        self.header_box = W.VBox([
            self.header_html, W.HBox([self.overall_progress, self.elapsed_eta_label]),
        ], layout=W.Layout(border="1px solid #30363d", padding="8px", margin="0 0 8px 0"))

    def _build_pipeline_strip(self) -> None:
        W = self._W
        self.input_card = W.VBox([
            W.HTML("<b>Drone Video</b>"),
            W.HTML(self._dot_html_static("done")),
        ], layout=W.Layout(border="1px solid #30363d", padding="6px", width="150px"))
        self.stage_cards: dict[str, StageCard] = {key: StageCard(W, key, label) for key, label in STAGE_LABELS.items()}
        self.pipeline_box = W.HBox(
            [self.input_card] + [c.widget for c in self.stage_cards.values()],
            layout=W.Layout(overflow_x="auto", margin="0 0 8px 0"),
        )

    def _dot_html_static(self, status: str) -> str:
        color = _STATUS_COLOR.get(status, "#30363d")
        return f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};"></span>'

    def _build_gpu_panel(self, gpu_count: int, gpu_names: list[str]) -> None:
        W = self._W
        self.gpu_count = gpu_count
        self.gpu_util_history: list[deque] = [deque(maxlen=120) for _ in range(max(gpu_count, 1))]
        self.gpu_mem_history: list[deque] = [deque(maxlen=120) for _ in range(max(gpu_count, 1))]
        self.gpu_out = W.Output()
        self.gpu_box = W.VBox([W.HTML("<b>GPU</b>"), self.gpu_out], layout=W.Layout(border="1px solid #30363d", padding="6px", width="48%"))

    def _build_frames_panel(self) -> None:
        W = self._W
        self.current_frame_img = W.Image(format="png", layout=W.Layout(width="160px"))
        self.current_frame_label = W.Label("")
        self.filmstrip = W.HBox([], layout=W.Layout(overflow_x="auto"))
        self.filmstrip_items: deque = deque(maxlen=12)
        self.frames_box = W.VBox([
            W.HTML("<b>Frames</b>"), self.current_frame_img, self.current_frame_label, self.filmstrip,
        ], layout=W.Layout(border="1px solid #30363d", padding="6px", width="48%"))

    def _build_trajectory_panel(self) -> None:
        W = self._W
        self.gps_track: list[tuple[float, float]] = []
        self.processed_track: list[tuple[float, float]] = []
        self.camera_track: list[tuple[float, float]] = []
        self.trajectory_out = W.Output()
        self.trajectory_box = W.VBox([W.HTML("<b>Trajectory</b>"), self.trajectory_out], layout=W.Layout(border="1px solid #30363d", padding="6px", width="48%"))

    def _build_geometry_panel(self) -> None:
        W = self._W
        self.geometry_out = W.Output()
        self.geometry_box = W.VBox([W.HTML("<b>Geometry (latest chunk)</b>"), self.geometry_out], layout=W.Layout(border="1px solid #30363d", padding="6px", width="48%"))

    def _build_live3d_panel(self) -> None:
        W = self._W
        self.color_mode = "rgb"
        self.points_xyz = np.zeros((0, 3), dtype=np.float32)
        self.points_color = np.zeros((0, 3), dtype=np.uint8)
        self.points_height = np.zeros((0,), dtype=np.float32)
        self.points_conf = np.zeros((0,), dtype=np.float32)
        self.mesh_mode = False

        self.color_toggle = W.Dropdown(options=["rgb", "height", "confidence"], value="rgb", description="Color:")
        self.color_toggle.observe(self._on_color_toggle, names="value")
        self.live3d_out = W.Output()
        self.live3d_box = W.VBox([
            W.HTML("<b>Live 3D</b>"), self.color_toggle, self.live3d_out,
        ], layout=W.Layout(border="1px solid #30363d", padding="6px", margin="0 0 8px 0"))
        self._fig = None

    def _on_color_toggle(self, change) -> None:
        self.color_mode = change["new"]
        self._dirty = True

    def _build_log_panel(self) -> None:
        W = self._W
        self.log_lines: deque = deque(maxlen=30)
        self.log_html = W.HTML("")
        self.log_box = W.VBox([W.HTML("<b>Log</b>"), self.log_html], layout=W.Layout(
            border="1px solid #30363d", padding="6px", max_height="200px", overflow_y="auto", margin="0 0 8px 0",
        ))

    def _build_results_card(self) -> None:
        W = self._W
        self.results_html = W.HTML("<i>Results will appear here once the pipeline finishes.</i>")
        self.results_box = W.VBox([W.HTML("<b>Results</b>"), self.results_html], layout=W.Layout(border="1px solid #30363d", padding="6px"))

    # -- event handling ----------------------------------------------------

    def on_event(self, evt: Event) -> None:
        p = evt.payload
        self._dirty = True

        if evt.type == EventType.STAGE_START:
            card = self.stage_cards.get(p.get("stage"))
            if card:
                card.start(evt.ts)
        elif evt.type == EventType.STAGE_PROGRESS:
            card = self.stage_cards.get(p.get("stage"))
            if card and "frac" in p:
                card.set_progress(p["frac"])
            self._update_overall_progress()
        elif evt.type == EventType.STAGE_DONE:
            card = self.stage_cards.get(p.get("stage"))
            if card:
                card.done(evt.ts)
            self._update_overall_progress()
        elif evt.type == EventType.STAGE_FALLBACK:
            card = self.stage_cards.get(p.get("stage"))
            if card:
                card.fallback(p.get("note", ""))
        elif evt.type == EventType.STAGE_ERROR:
            card = self.stage_cards.get(p.get("stage"))
            if card:
                card.error(evt.ts)

        elif evt.type == EventType.GPU_SAMPLE:
            idx = p["index"]
            if idx < len(self.gpu_util_history):
                self.gpu_util_history[idx].append(p["util_pct"])
                self.gpu_mem_history[idx].append(p["mem_used_mb"])

        elif evt.type == EventType.FRAME_DECODED:
            self.frames_processed += 1
            frame = p.get("frame")
            if frame is not None:
                self.current_frame_img.value = _encode_png(frame)
            self.current_frame_label.value = f"frame {p.get('frame_index', '?')}  sharpness={p.get('sharpness', 0):.0f}"

        elif evt.type in (EventType.KEYFRAME_ACCEPTED, EventType.KEYFRAME_REJECTED):
            thumb = p.get("thumbnail")
            if thumb is not None:
                accepted = evt.type == EventType.KEYFRAME_ACCEPTED
                self.filmstrip_items.append((thumb, accepted))
                self._rebuild_filmstrip()

        elif evt.type == EventType.TRAJECTORY_POINT:
            kind = p.get("kind", "gps")
            xy = (p["x"], p["y"])
            {"gps": self.gps_track, "processed": self.processed_track, "camera": self.camera_track}.get(kind, self.gps_track).append(xy)

        elif evt.type == EventType.GEOMETRY_CHUNK:
            self._latest_depth = p.get("depth")
            self._latest_confidence = p.get("confidence")

        elif evt.type == EventType.POINTCLOUD_GROWTH:
            xyz = p.get("points")
            if xyz is not None and len(xyz) > 0:
                self._append_points(xyz, p.get("colors"), p.get("confidence"))

        elif evt.type == EventType.MESH_PREVIEW:
            self.mesh_mode = True
            self._mesh_preview = p

        elif evt.type == EventType.LOG:
            level = p.get("level", "info")
            prefix = {"warn": "⚠", "error": "✗"}.get(level, "")
            self.log_lines.append(f"{prefix} {p.get('message', '')}".strip())

        elif evt.type == EventType.PIPELINE_DONE:
            self._render_results(p)

    def _update_overall_progress(self) -> None:
        n = len(self.stage_cards)
        done = sum(1 for c in self.stage_cards.values() if c.status == "done")
        running_partial = sum(
            (c.progress.value if c.progress is not None else 0.0)
            for c in self.stage_cards.values() if c.status == "running"
        )
        self.overall_progress.value = min(1.0, (done + running_partial) / n)

    def _rebuild_filmstrip(self) -> None:
        W = self._W
        items = []
        for thumb, accepted in self.filmstrip_items:
            img = W.Image(value=_encode_png(thumb), format="png", layout=W.Layout(width="48px", opacity="1.0" if accepted else "0.35"))
            items.append(img)
        self.filmstrip.children = items

    def _append_points(self, xyz: np.ndarray, colors: np.ndarray | None, confidence: np.ndarray | None, cap: int = 150_000) -> None:
        self.points_xyz = np.concatenate([self.points_xyz, xyz.astype(np.float32)], axis=0)
        if colors is not None:
            self.points_color = np.concatenate([self.points_color, colors.astype(np.uint8)], axis=0)
        self.points_height = self.points_xyz[:, 2]
        if confidence is not None:
            self.points_conf = np.concatenate([self.points_conf, confidence.astype(np.float32)], axis=0)

        if len(self.points_xyz) > cap:
            idx = np.random.default_rng(0).choice(len(self.points_xyz), size=cap, replace=False)
            self.points_xyz = self.points_xyz[idx]
            if len(self.points_color):
                self.points_color = self.points_color[idx]
            self.points_height = self.points_height[idx]
            if len(self.points_conf):
                self.points_conf = self.points_conf[idx]

    # -- rendering -----------------------------------------------------------

    def render_if_due(self, force: bool = False) -> None:
        now = time.time()
        if not force and (not self._dirty or now - self._last_render < self.refresh_interval):
            return
        self._last_render = now
        self._dirty = False

        elapsed = now - self.start_ts
        frac = self.overall_progress.value
        eta = f", ETA {elapsed * (1 - frac) / frac:.0f}s" if frac > 0.02 else ""
        self.elapsed_eta_label.value = f"elapsed {elapsed:.0f}s{eta}"
        for c in self.stage_cards.values():
            c.tick_elapsed(now)

        self._render_gpu_panel()
        self._render_trajectory_panel()
        self._render_geometry_panel()
        self._render_live3d_panel()
        self._render_log_panel()

    def _render_gpu_panel(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        with self.gpu_out:
            self.gpu_out.clear_output(wait=True)
            fig, axes = plt.subplots(1, max(self.gpu_count, 1), figsize=(3 * max(self.gpu_count, 1), 1.5))
            axes = np.atleast_1d(axes)
            for i, ax in enumerate(axes):
                hist = list(self.gpu_util_history[i]) if i < len(self.gpu_util_history) else []
                ax.plot(hist, color="#1f6feb")
                ax.set_ylim(0, 100)
                ax.set_title(f"GPU{i} {hist[-1]:.0f}%" if hist else f"GPU{i}", fontsize=8)
                ax.set_xticks([])
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    def _render_trajectory_panel(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        with self.trajectory_out:
            self.trajectory_out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(4, 3))
            if self.gps_track:
                xs, ys = zip(*self.gps_track)
                ax.plot(xs, ys, "-", color="#8b949e", label="GPS", linewidth=1)
            if self.processed_track:
                xs, ys = zip(*self.processed_track)
                ax.scatter(xs, ys, s=10, color="#1f6feb", label="processed keyframes")
            if self.camera_track:
                xs, ys = zip(*self.camera_track)
                ax.plot(xs, ys, "-", color="#238636", label="estimated camera", linewidth=1)
            ax.set_aspect("equal")
            ax.legend(fontsize=6)
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    def _render_geometry_panel(self) -> None:
        depth = getattr(self, "_latest_depth", None)
        conf = getattr(self, "_latest_confidence", None)
        if depth is None:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        with self.geometry_out:
            self.geometry_out.clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(5, 2.2))
            axes[0].imshow(depth, cmap="viridis")
            axes[0].set_title("depth", fontsize=8)
            axes[0].axis("off")
            if conf is not None:
                axes[1].imshow(conf, cmap="magma", vmin=0, vmax=1)
            axes[1].set_title("confidence", fontsize=8)
            axes[1].axis("off")
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    def _render_live3d_panel(self) -> None:
        if len(self.points_xyz) == 0:
            return
        try:
            import plotly.graph_objects as go
        except Exception:
            return

        if self.color_mode == "rgb" and len(self.points_color):
            colors = [f"rgb({r},{g},{b})" for r, g, b in self.points_color]
        elif self.color_mode == "height":
            colors = self.points_height
        elif self.color_mode == "confidence" and len(self.points_conf):
            colors = self.points_conf
        else:
            colors = "#1f6feb"

        with self.live3d_out:
            self.live3d_out.clear_output(wait=True)
            fig = go.Figure(data=[go.Scatter3d(
                x=self.points_xyz[:, 0], y=self.points_xyz[:, 1], z=self.points_xyz[:, 2],
                mode="markers", marker=dict(size=1.5, color=colors),
            )])
            fig.update_layout(
                width=700, height=400, margin=dict(l=0, r=0, t=0, b=0),
                scene=dict(aspectmode="data"),
            )
            fig.show()

    def _render_log_panel(self) -> None:
        self.log_html.value = "<pre style='font-size:11px;margin:0;'>" + "\n".join(self.log_lines) + "</pre>"

    def _render_results(self, payload: dict) -> None:
        outputs = payload.get("outputs", [])
        rows = "".join(
            f"<tr><td>{o.get('name')}</td><td>{'OK' if o.get('ok') else 'skipped: ' + str(o.get('skipped_reason'))}</td>"
            f"<td>{(o.get('size_bytes', 0) or 0) / 1e6:.2f} MB</td></tr>" for o in outputs
        )
        viewer_note = ""
        viewer_path = payload.get("viewer_html_path")
        if viewer_path:
            try:
                viewer_src = open(viewer_path, "r", encoding="utf-8").read().replace('"', "&quot;")
                viewer_note = f'<iframe srcdoc="{viewer_src}" style="width:100%;height:500px;border:1px solid #30363d;"></iframe>'
            except Exception:
                viewer_note = f"<p>viewer.html at {viewer_path}</p>"
        self.results_html.value = (
            f"<table><tr><th>File</th><th>Status</th><th>Size</th></tr>{rows}</table>{viewer_note}"
        )

"""Live ipywidgets dashboard, consuming events drained from the pipeline's
EventBus in the notebook's main thread (the single RUN cell — see
bootstrap.py). Refresh is throttled to <=2Hz via `render_if_due()`;
`on_event()` itself is cheap (just accumulates state) so it never slows
down event draining.

ipywidgets objects can be constructed and updated outside a live Jupyter
kernel (they just have no frontend to render to), so the state-accumulation
logic here is unit-testable headlessly; the actual visual layout can only be
verified in a real notebook.

The dashboard must render before installs/downloads start (per the task's
"dashboard-first" requirement), so nothing in construction can depend on a
package that isn't preinstalled on Kaggle. ipywidgets/matplotlib/PIL are
safe; anywidget/plotly are NOT guaranteed present yet at construction time
this early — the inline 3D viewer degrades to a lightweight fallback (see
_build_live3d_panel) rather than importing anywidget eagerly and failing.
"""

from __future__ import annotations

import io
import time
from collections import deque
from pathlib import Path

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

SETUP_STAGE_LABELS = {
    "env_check": "Environment Check",
    "install": "Install Dependencies",
    "detect_inputs": "Detect Inputs",
}

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
        self._base_label = label

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
        self.label.value = f"<b>{self._base_label}</b> <span style='color:#9e6a03'>[fallback]</span>"

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
    def __init__(self, header_info: dict, refresh_hz: float = 2.0, output_dir: str | Path | None = None):
        import ipywidgets as W

        self._W = W
        self.refresh_interval = 1.0 / max(refresh_hz, 0.1)
        self._last_render = 0.0
        self._dirty = True
        self.output_dir = Path(output_dir) if output_dir else None

        self.start_ts = time.time()
        self.frames_processed = 0
        self._header_info = dict(header_info)
        self._outputs_tunnel_url: str | None = None
        self._live_page_url: str | None = None

        self._build_header()
        self._build_setup_strip()
        self._build_install_panel()
        self._build_pipeline_strip()
        self._build_gpu_panel(header_info.get("gpu_count", 0))
        self._build_frames_panel()
        self._build_trajectory_panel()
        self._build_geometry_panel()
        self._build_live3d_panel()
        self._build_rasters_panel()
        self._build_log_panel()
        self._build_results_card()

        self.root = W.VBox([
            self.header_box,
            self.setup_box,
            self.pipeline_box,
            W.HBox([self.gpu_box, self.frames_box]),
            W.HBox([self.trajectory_box, self.geometry_box]),
            self.live3d_box,
            self.rasters_box,
            self.log_box,
            self.results_box,
        ])

    def display(self) -> None:
        from IPython.display import display

        display(self.root)

    # -- construction ---------------------------------------------------------

    def _build_header(self) -> None:
        W = self._W
        self.header_html = W.HTML(self._render_header_html())
        self.header_links_html = W.HTML("")
        self.overall_progress = W.FloatProgress(value=0.0, min=0.0, max=1.0, description="Overall:", layout=W.Layout(width="60%"))
        self.elapsed_eta_label = W.Label("elapsed 0s")
        self.header_box = W.VBox([
            self.header_html, self.header_links_html,
            W.HBox([self.overall_progress, self.elapsed_eta_label]),
        ], layout=W.Layout(border="1px solid #30363d", padding="8px", margin="0 0 8px 0"))

    def _render_header_html(self) -> str:
        info = self._header_info
        lines = [
            f"<b>Video:</b> {info.get('video_name', '?')} "
            f"({info.get('resolution', '?')}, {info.get('fps', '?')} fps, {info.get('duration_s', 0):.0f}s)",
            f"<b>Telemetry:</b> {info.get('telemetry_type', 'none')} "
            f"({info.get('telemetry_sample_count', 0)} samples, sync: {info.get('sync_method', '-')})",
            f"<b>Mode:</b> {info.get('mode', '?')} &nbsp; "
            f"<b>GPUs:</b> {', '.join(info.get('gpu_names', [])) or 'none detected'} &nbsp; "
            f"<b>Backbone:</b> {info.get('backbone', '?')} ({info.get('prior_mode', '?')})",
        ]
        return "<br>".join(lines)

    def update_header(self, info: dict) -> None:
        self._header_info.update(info)
        self.header_html.value = self._render_header_html()
        self._dirty = True

    def _update_header_links(self) -> None:
        links = []
        if self._live_page_url:
            links.append(f"<a href='{self._live_page_url}' target='_blank'>Open live reconstruction page</a>")
        if self._outputs_tunnel_url:
            links.append(f"<a href='{self._outputs_tunnel_url}/viewer.html' target='_blank'>Open full-screen viewer</a>")
        self.header_links_html.value = " &nbsp;|&nbsp; ".join(links)

    def _build_setup_strip(self) -> None:
        W = self._W
        self.setup_cards: dict[str, StageCard] = {
            key: StageCard(W, key, label, has_progress=False) for key, label in SETUP_STAGE_LABELS.items()
        }
        self.setup_box = W.HBox(
            [c.widget for c in self.setup_cards.values()],
            layout=W.Layout(overflow_x="auto", margin="0 0 8px 0"),
        )

    def _build_install_panel(self) -> None:
        W = self._W
        self.install_rows: dict[str, dict] = {}
        self.install_html = W.HTML("")
        self.install_box = W.VBox(
            [W.HTML("<b>Installs</b>"), self.install_html],
            layout=W.Layout(border="1px solid #30363d", padding="6px", margin="0 0 8px 0", max_height="140px", overflow_y="auto"),
        )
        self.setup_box.children = list(self.setup_box.children) + [self.install_box]

    def _render_install_panel(self) -> None:
        if not self.install_rows:
            return
        rows = "".join(
            f"<tr><td>{pkg}</td><td>{r['status']}</td><td>{r['elapsed_s']:.1f}s</td></tr>"
            for pkg, r in self.install_rows.items()
        )
        self.install_html.value = f"<table style='font-size:11px'><tr><th>Package</th><th>Status</th><th>Time</th></tr>{rows}</table>"

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

    def _build_gpu_panel(self, gpu_count: int) -> None:
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
        """Primary: the anywidget inline viewer (full-quality, binary-
        streamed — see inline_viewer.py). Fallback (anywidget unavailable
        or fails to import): a periodically-regenerated data-URL three.js
        view reusing viewer.py's own HTML builder, with a downsampled
        point cloud. Never imports anywidget/plotly at Dashboard
        construction time in a way that could delay the dashboard-first
        requirement — this try/except is the only place that happens, and
        failure here just means "use the fallback", not "crash"."""
        W = self._W
        self.inline_viewer = None
        try:
            from .inline_viewer import InlineViewer

            if InlineViewer is not None:
                self.inline_viewer = InlineViewer()
        except Exception:
            self.inline_viewer = None

        self.mesh_mode = False
        self._latest_mesh_payload = None

        if self.inline_viewer is not None:
            self.live3d_box = W.VBox(
                [W.HTML("<b>Live 3D</b>"), self.inline_viewer],
                layout=W.Layout(border="1px solid #30363d", padding="6px", margin="0 0 8px 0"),
            )
        else:
            self._fallback_points_xyz = np.zeros((0, 3), dtype=np.float32)
            self._fallback_points_color = np.zeros((0, 3), dtype=np.uint8)
            self._fallback_points_conf = np.zeros((0,), dtype=np.float32)
            self._last_fallback_render = 0.0
            self.live3d_fallback_html = W.HTML("<i>3D preview will appear here once chunks are processed (fallback mode — anywidget unavailable, using a downsampled static viewer).</i>")
            self.live3d_box = W.VBox(
                [W.HTML("<b>Live 3D (fallback mode)</b>"), self.live3d_fallback_html],
                layout=W.Layout(border="1px solid #30363d", padding="6px", margin="0 0 8px 0"),
            )

    def _build_rasters_panel(self) -> None:
        W = self._W
        self.raster_paths: dict[str, str] = {}
        self.raster_images: dict[str, W.Image] = {}
        self.raster_zoom_sliders: dict[str, W.IntSlider] = {}
        self.rasters_tab = W.Tab()
        self.rasters_box = W.VBox(
            [W.HTML("<b>Rasters</b>"), self.rasters_tab],
            layout=W.Layout(border="1px solid #30363d", padding="6px", margin="0 0 8px 0"),
        )

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
        self.package_button = W.Button(description="Package outputs", icon="archive")
        self.package_status = W.Label("")
        self.package_button.on_click(self._on_package_click)
        self.results_box = W.VBox(
            [W.HTML("<b>Results</b>"), self.results_html, W.HBox([self.package_button, self.package_status])],
            layout=W.Layout(border="1px solid #30363d", padding="6px"),
        )

    def _on_package_click(self, _btn) -> None:
        """Downloading stays a separate, deliberate action — never
        triggered automatically by the pipeline itself."""
        if self.output_dir is None or not self.output_dir.exists():
            self.package_status.value = "no output_dir configured"
            return
        import shutil

        self.package_status.value = "zipping..."
        try:
            zip_base = str(self.output_dir.parent / "outputs_package")
            zip_path = shutil.make_archive(zip_base, "zip", root_dir=self.output_dir)
            from IPython.display import FileLink, display

            display(FileLink(zip_path))
            self.package_status.value = f"done: {Path(zip_path).name}"
        except Exception as e:
            self.package_status.value = f"failed: {e}"

    # -- event handling ----------------------------------------------------

    def on_event(self, evt: Event) -> None:
        p = evt.payload
        self._dirty = True

        if evt.type == EventType.SETUP_STAGE_START:
            card = self.setup_cards.get(p.get("stage"))
            if card:
                card.start(evt.ts)
        elif evt.type == EventType.SETUP_STAGE_DONE:
            card = self.setup_cards.get(p.get("stage"))
            if card:
                card.done(evt.ts)
        elif evt.type == EventType.SETUP_STAGE_FALLBACK:
            card = self.setup_cards.get(p.get("stage"))
            if card:
                card.fallback(p.get("note", ""))
        elif evt.type == EventType.SETUP_STAGE_ERROR:
            card = self.setup_cards.get(p.get("stage"))
            if card:
                card.error(evt.ts)

        elif evt.type == EventType.INSTALL_PROGRESS:
            self.install_rows[p.get("package", "?")] = {"status": p.get("status", "?"), "elapsed_s": p.get("elapsed_s", 0.0)}

        elif evt.type == EventType.HEADER_UPDATE:
            self.update_header(p)

        elif evt.type == EventType.TUNNEL_READY:
            self._outputs_tunnel_url = p.get("url")
            self._update_header_links()
        elif evt.type == EventType.LIVE_PAGE_READY:
            self._live_page_url = p.get("url")
            self._update_header_links()

        elif evt.type == EventType.STAGE_START:
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
                if self.inline_viewer is not None:
                    try:
                        self.inline_viewer.push_points(xyz, p.get("colors"), p.get("confidence"))
                    except Exception:
                        pass
                else:
                    self._append_fallback_points(xyz, p.get("colors"), p.get("confidence"))

        elif evt.type == EventType.MESH_PREVIEW:
            self.mesh_mode = True
            self._latest_mesh_payload = p
            self._push_mesh_preview(p)

        elif evt.type == EventType.RASTERS_READY:
            for name in ("dsm_path", "orthomosaic_path", "coverage_path"):
                path = p.get(name)
                if path:
                    self.raster_paths[name.replace("_path", "")] = path
            self._rebuild_rasters_tab()

        elif evt.type == EventType.LOG:
            level = p.get("level", "info")
            prefix = {"warn": "⚠", "error": "✗"}.get(level, "")
            self.log_lines.append(f"{prefix} {p.get('message', '')}".strip())

        elif evt.type == EventType.PIPELINE_DONE:
            self._render_results(p)

    def _push_mesh_preview(self, payload: dict) -> None:
        mesh_result = payload.get("mesh")
        bake = payload.get("bake")
        stage = payload.get("stage", "coarse")
        if mesh_result is None or mesh_result.mesh is None or self.inline_viewer is None:
            return
        mesh = mesh_result.mesh
        try:
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            indices = np.asarray(mesh.triangles, dtype=np.uint32).reshape(-1)
            colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None
            uv = texture_png = None
            if bake is not None and getattr(bake, "textured", False):
                uv = bake.uv
                texture_png = _encode_png(bake.texture_rgb)
            self.inline_viewer.push_mesh(vertices, indices, colors=colors, uv=uv, texture_png=texture_png, stage=stage)
        except Exception:
            pass

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

    def _append_fallback_points(self, xyz: np.ndarray, colors: np.ndarray | None, confidence: np.ndarray | None, cap: int = 150_000) -> None:
        self._fallback_points_xyz = np.concatenate([self._fallback_points_xyz, xyz.astype(np.float32)], axis=0)
        if colors is not None:
            self._fallback_points_color = np.concatenate([self._fallback_points_color, colors.astype(np.uint8)], axis=0)
        if confidence is not None:
            self._fallback_points_conf = np.concatenate([self._fallback_points_conf, confidence.astype(np.float32)], axis=0)

        if len(self._fallback_points_xyz) > cap:
            idx = np.random.default_rng(0).choice(len(self._fallback_points_xyz), size=cap, replace=False)
            self._fallback_points_xyz = self._fallback_points_xyz[idx]
            if len(self._fallback_points_color):
                self._fallback_points_color = self._fallback_points_color[idx]
            if len(self._fallback_points_conf):
                self._fallback_points_conf = self._fallback_points_conf[idx]

    def _rebuild_rasters_tab(self) -> None:
        W = self._W
        children = []
        titles = []
        for name, path in self.raster_paths.items():
            try:
                png_bytes = _geotiff_to_png(path)
            except Exception:
                continue
            img = W.Image(value=png_bytes, format="png", layout=W.Layout(width="600px"))
            slider = W.IntSlider(value=600, min=200, max=1600, description="zoom (px)")
            slider.observe(lambda change, im=img: setattr(im.layout, "width", f"{change['new']}px"), names="value")
            children.append(W.VBox([slider, img]))
            titles.append(name)
        self.rasters_tab.children = children
        for i, t in enumerate(titles):
            self.rasters_tab.set_title(i, t)

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
        for c in self.setup_cards.values():
            c.tick_elapsed(now)

        self._render_install_panel()
        self._render_gpu_panel()
        self._render_trajectory_panel()
        self._render_geometry_panel()
        self._render_fallback_live3d_panel(now, force)
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

    def _render_fallback_live3d_panel(self, now: float, force: bool) -> None:
        """Only used when anywidget/InlineViewer is unavailable. Much
        coarser cadence than the main 2Hz throttle (rebuilding a three.js
        HTML blob and re-embedding it as an iframe is comparatively heavy)
        — every 5s, or on force (e.g. the final render)."""
        if self.inline_viewer is not None:
            return
        if not force and now - self._last_fallback_render < 5.0:
            return
        if len(self._fallback_points_xyz) == 0:
            return
        self._last_fallback_render = now
        try:
            from .fusion import FusedPointCloud
            from .viewer import write_viewer_html

            n = len(self._fallback_points_xyz)
            cloud = FusedPointCloud(
                points=self._fallback_points_xyz,
                colors=self._fallback_points_color if len(self._fallback_points_color) == n else np.full((n, 3), 128, dtype=np.uint8),
                view_count=np.ones(n, dtype=np.int32),
                confidence=self._fallback_points_conf if len(self._fallback_points_conf) == n else np.ones(n, dtype=np.float32),
            )
            mesh_glb_path = None
            if self._latest_mesh_payload is not None:
                candidate = self.output_dir / "mesh.glb" if self.output_dir else None
                if candidate is not None and candidate.exists():
                    mesh_glb_path = candidate

            from .events import EventBus

            tmp_path = Path("/tmp/sih3d_fallback_viewer.html")
            status = write_viewer_html(
                tmp_path, EventBus(), cloud=cloud, mesh_glb_path=mesh_glb_path,
                gps_track_enu=self.gps_track, camera_track_enu=self.camera_track,
                georeferenced=bool(self._header_info.get("georeferenced", False)),
                max_points=50_000,
            )
            if status.ok:
                html = tmp_path.read_text(encoding="utf-8").replace('"', "&quot;")
                self.live3d_fallback_html.value = f"<iframe srcdoc=\"{html}\" style='width:100%;height:500px;border:none;'></iframe>"
        except Exception:
            pass

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

    def snapshot_html(self) -> str:
        """A static HTML render of the dashboard's current state, for
        report.html — ipywidgets themselves don't survive a static export,
        and Kaggle's "Save Version" runs the notebook headless (no one
        watching the live widgets), so report.html is the only place the
        final dashboard state is ever actually seen for that run. Re-uses
        the same matplotlib panels as the live view, saved as embedded
        base64 PNGs instead of rendered into an Output widget."""
        import base64
        import io as _io

        parts = ["<div style='font-family:sans-serif'>"]

        stage_rows = "".join(
            f"<tr><td>{c.key}</td><td>{c.status}</td><td>{c.elapsed_s:.1f}s</td>"
            f"<td>{c.fallback_note or '-'}</td></tr>"
            for c in self.stage_cards.values()
        )
        parts.append(f"<h3>Pipeline stages</h3><table border='1' style='border-collapse:collapse'>"
                     f"<tr><th>Stage</th><th>Status</th><th>Elapsed</th><th>Fallback</th></tr>{stage_rows}</table>")

        try:
            import matplotlib.pyplot as plt

            if any(len(h) for h in self.gpu_util_history):
                fig, axes = plt.subplots(1, max(self.gpu_count, 1), figsize=(3 * max(self.gpu_count, 1), 1.5))
                axes = np.atleast_1d(axes)
                for i, ax in enumerate(axes):
                    hist = list(self.gpu_util_history[i]) if i < len(self.gpu_util_history) else []
                    ax.plot(hist, color="#1f6feb")
                    ax.set_ylim(0, 100)
                    ax.set_title(f"GPU{i}", fontsize=8)
                buf = _io.BytesIO()
                plt.tight_layout()
                fig.savefig(buf, format="png", dpi=100)
                plt.close(fig)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                parts.append(f"<h3>GPU utilization</h3><img src='data:image/png;base64,{b64}'>")
        except Exception:
            pass

        parts.append(f"<h3>Log (last {len(self.log_lines)} lines)</h3><pre style='font-size:11px'>" + "\n".join(self.log_lines) + "</pre>")
        parts.append("</div>")
        return "".join(parts)


def _geotiff_to_png(path: str, max_dim: int = 1600) -> bytes:
    """Reads a GeoTIFF via rasterio and downsamples for a notebook-friendly
    preview (the actual full-resolution file stays on disk untouched)."""
    import rasterio
    from PIL import Image

    with rasterio.open(path) as src:
        data = src.read()  # (bands, H, W)

    if data.shape[0] >= 3:
        arr = np.transpose(data[:3], (1, 2, 0)).astype(np.float32)
    else:
        band = data[0].astype(np.float32)
        arr = np.stack([band, band, band], axis=-1)

    finite = np.isfinite(arr)
    if finite.any():
        lo, hi = np.percentile(arr[finite], [2, 98])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        arr = np.zeros_like(arr)
    arr = np.nan_to_num(arr)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

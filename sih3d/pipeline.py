"""Pipeline orchestration (v1): correctness and a working QUICK run first,
deep GPU overlap later.

v1 design, deliberately simple:
  - One background pipeline thread does: input scan -> frame extraction +
    keyframe selection -> per-keyframe pose/intrinsics priors -> a chunk loop
    that runs the geometry backbone (GPU0) and immediately gravity-fixed-
    aligns each chunk.
  - When 2 GPUs are detected, a second worker thread (GPU1) consumes raw
    chunk results from a bounded queue and does dynamic-object masking +
    voxel fusion, so GPU0 can start the next chunk's backbone inference
    without waiting on masking/fusion. On one GPU (or CPU), masking+fusion
    just run inline in the same thread right after the backbone call.
  - v2 (not implemented here) would add CUDA-stream overlap for
    decode/copy/compute and let GPU1 also prefetch the *next* chunk's
    frames while GPU0 is still busy — see decode.py's docstring. Adding
    that before v1 produces a single correct end-to-end QUICK run on real
    Kaggle hardware would be optimizing something we haven't confirmed
    works yet.

Every stage is wrapped so a failure inside it degrades (logs a fallback,
returns something the rest of the pipeline can keep going with) rather than
killing the run. Only the one-time setup step (selecting/loading the
backbone) is allowed to be fatal — there's nothing useful to do without a
model — and even then the run still writes whatever partial report/outputs
exist before re-raising, and the exception+traceback are captured on
`PipelineResult` for the notebook's debug cell.

v1 simplifications, called out explicitly (not silently cut corners):
  - Each chunk's point cloud is fused into `VoxelPointFusion` using that
    chunk's own direct alignment (gravity-fixed 4-DoF vs GPS, or Sim(3) vs
    the previous chunk with no GPS) as soon as it's computed. The end-of-run
    pose-graph refinement (align.refine_pose_graph) is used to report a more
    accurate global alignment RMSE and to write a consistent
    trajectory.geojson/.kml, but does NOT retroactively re-fuse
    already-accumulated points with the refined transforms (that would need
    keeping every chunk's raw pre-alignment points in memory, which we
    deliberately don't for a first working run). A v2 could re-fuse.
  - No TSDF integration in v1 (mesh.py's Open3DTsdfFusion path is left
    unused here) — meshing goes straight from the fused point cloud via
    Poisson, which is already a real, working fallback path with its own
    tests. TSDF's tensor-CUDA call site was intentionally left unverified
    against Kaggle's actual Open3D version (see PHASE0_NOTES.md); wiring it
    in before that's confirmed would just be guessing.
  - Full-resolution keyframe images (needed for texture baking) are kept
    in memory rather than re-decoded on demand, bounded by QUICK mode's own
    keyframe cap. FULL mode with many more/larger keyframes may need this
    revisited during FULL-mode performance tuning (a later, explicit step).
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import align, export, masks
from . import backbone as backbone_mod
from . import mesh as mesh_mod
from .artifacts import ChunkAlignmentRecord, ChunkGeometrySample, KeyframeRecord, RunArtifacts
from .backbone import ChunkResult, PriorMode, ViewInput
from .decode import FrameDecoder
from .events import Event, EventBus, EventType
from .fusion import FusedPointCloud, VoxelPointFusion, remove_statistical_outliers
from .gpu_monitor import GpuMonitor
from .io_detect import DetectedInputs
from .keyframes import select_keyframes
from .mesh import KeyframeForBaking
from .report import ReportBuilder
from .telemetry import (
    TelemetryTrack, collinearity_index as compute_collinearity, sync_to_frame_times, track_to_enu,
)


@dataclass
class PipelineConfig:
    mode: str = "QUICK"
    backbone_choice: str = "C"
    prior_mode: str = "AUTO"
    use_finetuned: bool = True
    chunk_size: int = 30
    chunk_overlap: int = 6
    quick_seconds: float = 90.0
    # Kept under chunk_size (30): _make_chunks emits exactly one chunk when
    # len(keyframes) <= chunk_size, which skips large_scale_alignment's
    # cross-chunk blending entirely and runs the backbone exactly once.
    # That's the single biggest lever on QUICK-mode wall time, well above
    # what tuning decode speed alone can buy back.
    quick_max_keyframes: int = 28
    # FULL mode's own keyframe budget — chunk_size(30)/overlap(6) chunking
    # is otherwise unbounded for a long video: whatever fraction of decoded
    # candidates clears the sharpness floor becomes the keyframe count with
    # no ceiling, so wall time (chunk count x backbone pass) and raw-frame
    # host RAM during decode both scale with video length with no cap. 240
    # keyframes is ~8 chunks with DUAL_GPU splitting them across both T4s —
    # sized for the documented "<15 min for a 10-min video" FULL-mode target
    # (see CONFIG_CELL in build_notebook.py).
    full_max_keyframes: int = 240
    backbone_input_size: int = 518
    output_dir: Path = Path("/kaggle/working/outputs")
    device0: str = "cuda:0"
    device1: str = "cuda:1"
    texture_time_budget_s: float = 60.0
    dsm_cell_size_m: float = 0.2
    min_confidence: float = 0.1
    min_view_count: int = 1
    voxel_size_m: float = 0.05
    keyframe_min_gps_spacing_m: float = 2.0
    keyframe_max_no_gps_stride: int = 5
    keyframe_sharpness_percentile_floor: float = 15.0
    keyframe_sample_fps: float = 4.0
    decode_working_width: int = 1920
    watchdog_stall_s: float = 60.0
    dual_gpu: bool = False


@dataclass
class PreparedKeyframe:
    frame_index: int
    timestamp_s: float
    image_full: np.ndarray
    gps_enu: tuple[float, float, float] | None
    intrinsics: np.ndarray
    camera_pose_c2w_prior: np.ndarray | None


@dataclass
class PipelineResult:
    success: bool
    error: str | None = None
    traceback: str | None = None
    outputs: list = field(default_factory=list)


def _default_intrinsics(w: int, h: int) -> np.ndarray:
    """Rough-FOV fallback when no calibration/metadata is available: assumes
    a typical consumer/drone-camera field of view (roughly 70-80 deg
    horizontal). This is explicitly a fallback, logged by the caller — real
    intrinsics from metadata or a detected calibration file always win."""
    f = 0.9 * max(w, h)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def _attitude_to_cam2world(position_enu: np.ndarray, attitude_deg: tuple[float, float, float]) -> np.ndarray:
    """Builds a full 4x4 cam2world pose from a GPS ENU position + gimbal/IMU
    attitude, reusing align.py's world->cam rotation construction (same ZYX
    gimbal convention) since it's already implemented and tested there."""
    from .align import estimate_gravity_imu

    g = estimate_gravity_imu(attitude_deg)
    yaw, pitch, roll = (np.radians(a) for a in attitude_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    R_world_to_cam = Rx @ Ry @ Rz
    R_cam_to_world = R_world_to_cam.T
    pose = np.eye(4)
    pose[:3, :3] = R_cam_to_world
    pose[:3, 3] = position_enu
    return pose


def _resize_for_backbone(img: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    pil = Image.fromarray(img)
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.array(pil)


def _make_thumbnail(img: np.ndarray, max_width: int = 120) -> np.ndarray:
    """Small RGB copy for the notebook's keyframe grid — keeping full-res
    frames around for every keyframe (not just accepted ones, since
    rejected ones are shown greyed out too) would be a real memory cost at
    FULL-mode scale."""
    from PIL import Image

    h, w = img.shape[:2]
    if w <= max_width:
        return img.copy()
    new_h = max(1, int(h * max_width / w))
    return np.array(Image.fromarray(img).resize((max_width, new_h), Image.BILINEAR))


class Pipeline:
    def __init__(
        self, config: PipelineConfig, detected: DetectedInputs, telemetry: TelemetryTrack | None,
        bus: EventBus, gpu_monitor: GpuMonitor, report: ReportBuilder,
        backbone_override: tuple | None = None, live_page=None,
    ):
        """`backbone_override`: (GeometryBackbone, PriorMode), pre-built and
        already loaded. Testing/dependency-injection hook only — the real
        notebook cells never pass this, so production behavior (backbone
        chosen via backbone.select_backbone from BACKBONE/PRIOR_MODE config)
        is unaffected. Used by the local dry-run harness to exercise the
        entire orchestration loop on CPU with a synthetic backbone, without
        downloading real multi-GB models.

        `live_page`: an already-started live_page.LivePageServer, or None if
        the tunnel/server failed to come up (bootstrap.py logs that and the
        pipeline just runs without pushing to it — the dashboard's inline 3D
        tab remains the fallback view). Kept as a plain optional handle
        rather than a required dependency so the local dry-run harness and
        any future non-notebook caller don't need to stand up a real server."""
        self.config = config
        self.detected = detected
        self.telemetry = telemetry
        self.bus = bus
        self.gpu_monitor = gpu_monitor
        self.report = report
        self._backbone_override = backbone_override
        self.live_page = live_page

        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu1_thread: threading.Thread | None = None
        self._chunk_queue: "queue.Queue" = queue.Queue(maxsize=3)
        self._SENTINEL = object()

        # Watchdog: touched by every _emit()/_log() call, polled by a
        # background thread — if a stage sits without producing a single
        # event for watchdog_stall_s, that's exactly the kind of silent
        # stall a first real Kaggle run hit (20+ min stuck in frame
        # extraction with no indication anything was wrong).
        self._last_progress_ts = time.time()
        self._run_t0: float = time.time()
        self._current_stage: str | None = None
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_last_warn_ts: float = 0.0

        self.two_gpu = gpu_monitor.device_count >= 2
        self.fusion_acc = VoxelPointFusion(config.voxel_size_m, device=config.device0)
        self.masker: masks.DynamicObjectMasker | None = None
        self.masker0: masks.DynamicObjectMasker | None = None  # DUAL_GPU only — device0's own masker, alongside self.masker on device1

        # DUAL_GPU idle-wait tracking, keyed by device string — logged and
        # surfaced in the final summary so "is a GPU actually busy" has a
        # real answer instead of just eyeballing the dashboard's SM% graph.
        self._gpu_idle_s: dict[str, float] = {}

        self.chunk_alignments: list[align.ChunkAlignment] = []
        self.overlap_constraints: list[align.OverlapConstraint] = []
        self.gps_factors: list[align.GpsFactor] = []
        self._prev_chunk_cam_local: dict[int, np.ndarray] = {}  # keyframe_index -> local cam center, from the previous chunk only

        self.result: PipelineResult | None = None
        self._export_statuses: list = []
        self.georeferenced: bool = self.detected.telemetry_path is not None or self.detected.telemetry_kind == "embedded"
        self.epsg: int | None = None
        self.artifacts = RunArtifacts()
        self._chunk_alignment_records: dict[int, ChunkAlignmentRecord] = {}

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_guarded, name="sih3d-pipeline", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 60.0) -> None:
        self.stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        if self._gpu1_thread:
            self._gpu1_thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_guarded(self) -> None:
        try:
            self._run()
            self.result = PipelineResult(success=True, outputs=self._export_statuses)
        except Exception as e:
            tb = traceback.format_exc()
            self._log(f"PIPELINE FATAL ERROR: {type(e).__name__}: {e}", level="error")
            self._emit(EventType.PIPELINE_ERROR, error=str(e), traceback=tb)
            self.result = PipelineResult(success=False, error=str(e), traceback=tb, outputs=self._export_statuses)
        finally:
            self.stop_event.set()
            self._watchdog_stop.set()
            try:
                self._report_full_run_utilization()
            except Exception:
                pass
            try:
                self.gpu_monitor.stop()
            except Exception:
                pass
            self.report.finish()
            viewer_path = self.config.output_dir / "viewer.html"
            self._emit(
                EventType.PIPELINE_DONE,
                outputs=[o.__dict__ for o in self._export_statuses],
                viewer_html_path=str(viewer_path) if viewer_path.exists() else None,
            )

    def _emit(self, event_type: EventType, **payload) -> None:
        """Publishes to the bus (for the dashboard/any external consumer)
        AND feeds ReportBuilder directly and synchronously. report-building
        is cheap bookkeeping, not rendering, so there's no reason report.json
        should depend on an external consumer having drained the bus in
        time — that's exactly the race this fixes (caught by the local dry
        run: report.json showed every stage as "pending" because nothing
        was draining the bus into `report` at all yet)."""
        ts = time.time()
        self._last_progress_ts = ts
        if event_type == EventType.STAGE_START:
            self._current_stage = payload.get("stage")
        elif event_type in (EventType.STAGE_DONE, EventType.STAGE_ERROR):
            self._current_stage = None
        self.bus.publish(event_type, **payload)
        self.report.on_event(Event(type=event_type, payload=payload, ts=ts))

    def _log(self, message: str, level: str = "info") -> None:
        self._last_progress_ts = time.time()
        self.bus.log(message, level=level)
        self.report.on_event(Event(type=EventType.LOG, payload={"message": message, "level": level}, ts=time.time()))

    def _watchdog_loop(self) -> None:
        """Polls for silent stalls (see class docstring / decode.py's own
        docstring for the real Kaggle run this is a response to). Only
        warns while a stage is actually in progress (STAGE_START seen,
        STAGE_DONE/ERROR not yet) — a long gap between stages (e.g.
        waiting on a subprocess mesh export) isn't a stall by itself as
        long as SOME stage is marked active; every meaningful unit of work
        inside a stage goes through _emit()/_log(), which is what resets
        the clock this checks."""
        while not self._watchdog_stop.wait(10.0):
            if self._current_stage is None:
                continue
            now = time.time()
            # bus.last_event_ts (not self._last_progress_ts) — heartbeat
            # progress from mesh.py's isolated subprocess helpers publishes
            # straight to self.bus, bypassing Pipeline._emit()/_log()
            # entirely (mesh.py only has the EventBus, not this Pipeline),
            # so watching only _last_progress_ts produced real "no progress"
            # false alarms during steps that WERE actively heartbeating.
            stalled_s = now - self.bus.last_event_ts
            if stalled_s < self.config.watchdog_stall_s:
                continue
            if now - self._watchdog_last_warn_ts < 30.0:
                continue  # already warned recently about this same stall
            self._watchdog_last_warn_ts = now
            self.bus.log(
                f"WATCHDOG: no progress for {stalled_s:.0f}s in stage '{self._current_stage}' — "
                f"this may be a stall, not necessarily an error", level="warn",
            )

    def _fallback(self, stage: str, exc: Exception) -> None:
        note = f"{type(exc).__name__}: {exc}"
        self._log(f"Stage '{stage}': {note} — continuing with degraded output", level="warn")
        self._emit(EventType.STAGE_FALLBACK, stage=stage, note=note)

    def _push_live(self, fn, *args, **kwargs) -> None:
        """Best-effort call into self.live_page. Disables further pushes
        (rather than logging per-keyframe) on the first failure — a broken
        live-page connection would otherwise spam one STAGE_FALLBACK per
        frame; the dashboard's inline 3D tab keeps working regardless."""
        if self.live_page is None:
            return
        try:
            fn(*args, **kwargs)
        except Exception as e:
            self._log(f"Live reconstruction page push failed ({type(e).__name__}: {e}) — disabling further pushes; inline 3D tab still works", level="warn")
            self.live_page = None

    # -- main run ---------------------------------------------------------

    def _run(self) -> None:
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_t0 = time.time()
        self.gpu_monitor.start()
        self._last_progress_ts = time.time()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="sih3d-watchdog")
        self._watchdog_thread.start()

        video = self.detected.video
        if video is None:
            raise RuntimeError("no video detected — nothing to process")

        # georeferenced only depends on what was detected, not on the
        # backbone — resolved here so frame_extraction/camera_trajectory
        # below can use it without the backbone needing to exist yet.
        self.georeferenced = self.detected.telemetry_path is not None or self.detected.telemetry_kind == "embedded"
        if not self.georeferenced:
            self._log("NO TELEMETRY: outputs will be APPROXIMATE SCALE — NOT GEOREFERENCED", level="warn")

        if self.two_gpu:
            self.masker = masks.DynamicObjectMasker(self.bus, device=cfg.device1)
            if cfg.dual_gpu:
                self.masker0 = masks.DynamicObjectMasker(self.bus, device=cfg.device0)

        # -- Stage: frame_extraction -------------------------------------
        # Backbone selection/loading is deliberately NOT done before this
        # stage (it used to be) — a first real Kaggle run OOM'd here: with
        # MapAnything already resident on device0 (~13.7 of a 14.6 GiB T4),
        # frame extraction's own sharpness-scoring batch (all candidate
        # frames as one tensor, also on device0) had nowhere near enough
        # headroom left. Frame extraction and camera_trajectory use
        # neither the backbone nor its device, so loading it only right
        # before the chunk loop that actually needs it — after chunking,
        # below — means it competes for GPU0 memory with nothing but
        # itself.
        self._emit(EventType.STAGE_START, stage="frame_extraction")
        try:
            keyframes = self._extract_keyframes(video)
        except Exception as e:
            self._fallback("frame_extraction", e)
            keyframes = []
        self._emit(EventType.STAGE_DONE, stage="frame_extraction")
        if not keyframes:
            raise RuntimeError("frame extraction produced zero usable keyframes")

        # -- Stage: camera_trajectory --------------------------------------
        self._emit(EventType.STAGE_START, stage="camera_trajectory")
        try:
            self._attach_priors(keyframes)
        except Exception as e:
            self._fallback("camera_trajectory", e)
        for kf in keyframes:
            if kf.gps_enu is not None:
                self._emit(EventType.TRAJECTORY_POINT, kind="gps", x=kf.gps_enu[0], y=kf.gps_enu[1])
        self._emit(EventType.STAGE_DONE, stage="camera_trajectory")

        collinearity = None
        if self.georeferenced:
            enu_pts = [kf.gps_enu for kf in keyframes if kf.gps_enu is not None]
            if len(enu_pts) >= 3:
                collinearity = compute_collinearity(enu_pts)
                self._log(f"Flight-track collinearity index: {collinearity:.3f} (near 0 = straight single pass)")
        self.artifacts.collinearity_index = collinearity
        self.artifacts.georeferenced = self.georeferenced

        # -- Chunking --------------------------------------------------------
        chunks = self._make_chunks(keyframes)
        self._log(f"Chunked {len(keyframes)} keyframes into {len(chunks)} chunk(s) (size={cfg.chunk_size}, overlap={cfg.chunk_overlap})")

        dual_gpu = cfg.dual_gpu and self.two_gpu
        if cfg.dual_gpu and not self.two_gpu:
            self._log("DUAL_GPU requested but fewer than 2 GPUs detected — running single-GPU", level="warn")

        # Backbone selection is the one allowed-fatal setup step, and the
        # only thing here allowed to put multi-GB of weights on device0 —
        # deliberately done only now, right before the chunk loop that's
        # the first thing to actually need it (see the frame_extraction
        # comment above for why: it used to run before frame extraction
        # and OOM'd there on a real 4K Kaggle run). DUAL_GPU loads a second,
        # fully independent instance on device1 too — sequentially, not
        # via two parallel loads: both instances fetch the SAME pretrained
        # weights, and racing two concurrent downloads into a possibly-cold
        # shared HuggingFace cache is a real corruption risk not worth
        # taking in a code path with no real hardware to test that race
        # against here.
        bb1 = None
        if self._backbone_override is not None:
            bb, prior_mode = self._backbone_override
        else:
            bb, prior_mode = backbone_mod.select_backbone(
                cfg.backbone_choice, cfg.prior_mode, self.detected, self.bus,
                device=cfg.device0, use_finetuned=cfg.use_finetuned,
            )
            if dual_gpu:
                bb1, prior_mode1 = backbone_mod.select_backbone(
                    cfg.backbone_choice, cfg.prior_mode, self.detected, self.bus,
                    device=cfg.device1, use_finetuned=cfg.use_finetuned,
                )
                if prior_mode1 != prior_mode:
                    self._log(
                        f"DUAL_GPU: device1 backbone resolved prior_mode={prior_mode1.value}, "
                        f"device0 resolved {prior_mode.value} — using device0's for both", level="warn",
                    )
        self.report.set_run_config(
            mode=cfg.mode, backbone=bb.name, prior_mode=prior_mode.value,
            checkpoint_source=getattr(bb, "checkpoint_source", "pretrained"),
            gpus=self.gpu_monitor.device_names,
        )

        self._emit(EventType.STAGE_START, stage="geometric_reconstruction")
        self._emit(EventType.STAGE_START, stage="large_scale_alignment")
        self._emit(EventType.STAGE_START, stage="dense_point_cloud")
        geom_stage_t0 = time.time()

        if dual_gpu and bb1 is not None:
            self._run_chunks_dual_gpu(chunks, bb, bb1, prior_mode)
        else:
            self._gpu1_thread = None
            if self.two_gpu:
                self._gpu1_thread = threading.Thread(target=self._gpu1_worker, name="sih3d-gpu1-worker", daemon=True)
                self._gpu1_thread.start()

            for chunk_idx, chunk_kfs in enumerate(chunks):
                if self.stop_event.is_set():
                    break
                chunk_result = self._process_chunk(bb, prior_mode, chunk_kfs, chunk_idx)
                frac = (chunk_idx + 1) / len(chunks)
                rate_label = self._chunk_rate_label(chunk_idx + 1, len(chunks), geom_stage_t0)
                self._emit(EventType.STAGE_PROGRESS, stage="geometric_reconstruction", frac=frac, rate_label=rate_label)
                self._emit(EventType.STAGE_PROGRESS, stage="large_scale_alignment", frac=frac, rate_label=rate_label)

                if chunk_result is None:
                    continue

                if self.two_gpu:
                    try:
                        self._chunk_queue.put((chunk_idx, chunk_kfs, chunk_result), timeout=30)
                    except queue.Full:
                        self._fallback("dense_point_cloud", RuntimeError("GPU1 worker queue full/stalled; dropping chunk"))
                else:
                    self._mask_and_fuse_chunk(chunk_idx, chunk_kfs, chunk_result)
                    self._emit(EventType.STAGE_PROGRESS, stage="dense_point_cloud", frac=frac, rate_label=rate_label)

            if self.two_gpu:
                self._chunk_queue.put(self._SENTINEL)
                if self._gpu1_thread:
                    self._gpu1_thread.join(timeout=300)

        self._emit(EventType.STAGE_DONE, stage="geometric_reconstruction")
        self._emit(EventType.STAGE_DONE, stage="large_scale_alignment")
        self._emit(EventType.STAGE_DONE, stage="dense_point_cloud")

        if dual_gpu:
            self._report_dual_gpu_utilization(geom_stage_t0)

        alignment_rmse = None
        try:
            alignment_rmse = self._refine_global_alignment()
        except Exception as e:
            self._fallback("large_scale_alignment", e)

        # -- Stage: mesh_textured_model ---------------------------------
        self._emit(EventType.STAGE_START, stage="mesh_textured_model")
        cloud = self.fusion_acc.extract_points(min_view_count=cfg.min_view_count)
        try:
            cloud = remove_statistical_outliers(cloud, self.bus)
        except Exception as e:
            self._fallback("mesh_textured_model", e)

        self.artifacts.point_count = len(cloud.points)
        if len(cloud.points) > 0:
            preview_n = min(len(cloud.points), 50_000)
            preview_idx = np.random.default_rng(0).choice(len(cloud.points), size=preview_n, replace=False)
            self.artifacts.point_cloud_preview = cloud.points[preview_idx]
            self.artifacts.point_cloud_preview_colors = cloud.colors[preview_idx]

        mesh_result = mesh_mod.MeshResult(mesh=None, method="none")
        bake_result = None
        try:
            mesh_result = mesh_mod.build_vertex_colored_mesh(cloud, tsdf=None, bus=self.bus)
            if mesh_result.mesh is not None:
                self._emit(EventType.MESH_PREVIEW, mesh=mesh_result, bake=None, stage="coarse")
        except Exception as e:
            self._fallback("mesh_textured_model", e)

        posed_keyframes = [kf for kf in keyframes if getattr(kf, "_resolved_world_pose", None) is not None]
        try:
            if mesh_result.mesh is not None:
                kf_for_bake = [
                    KeyframeForBaking(image_rgb=kf.image_full, camera_pose_c2w=self._resolved_pose(kf, chunk_idx=None), intrinsics=kf.intrinsics)
                    for kf in posed_keyframes
                ]
                bake_result = mesh_mod.bake_texture(mesh_result.mesh, kf_for_bake, self.bus, time_budget_s=cfg.texture_time_budget_s)
                if bake_result is not None and bake_result.textured:
                    self._emit(EventType.MESH_PREVIEW, mesh=mesh_result, bake=bake_result, stage="textured")
        except Exception as e:
            self._fallback("mesh_textured_model", e)

        self._emit(EventType.STAGE_DONE, stage="mesh_textured_model")

        self.artifacts.mesh_n_vertices = mesh_result.n_vertices
        self.artifacts.mesh_n_faces = mesh_result.n_faces
        self.artifacts.mesh_method = mesh_result.method

        self.report.set_geometry_summary(
            georeferenced=self.georeferenced, collinearity_index=collinearity, alignment_rmse_m=alignment_rmse,
            point_count=len(cloud.points), mesh_faces=mesh_result.n_faces,
        )

        # -- Export ------------------------------------------------------
        self._export_all(cloud, mesh_result, bake_result, keyframes)

        # -- Render-vs-ground-truth comparison -----------------------------
        # Runs AFTER export, not before: bake_texture() never modifies
        # mesh_result.mesh in place (it returns a separate TextureBakeResult
        # with its own atlas image + UVs) — the geometry and the baked
        # texture only actually get combined into one object when
        # export_mesh_glb() writes mesh.glb, so reading mesh_result.mesh
        # directly here would always show the pre-bake vertex-colored mesh,
        # silently ignoring a successful bake. Reading the exported GLB
        # file (same as stage_views.render_showcase) is what actually shows
        # the final polished mesh a bake produced.
        try:
            from .validation import _RenderKeyframe, render_vs_ground_truth

            render_kfs = [
                _RenderKeyframe(
                    frame_index=kf.frame_index, image_rgb=kf.image_full,
                    camera_pose_c2w=self._resolved_pose(kf, chunk_idx=None), intrinsics=kf.intrinsics,
                )
                for kf in posed_keyframes
            ]
            self.artifacts.comparisons = render_vs_ground_truth(
                cloud.points, cloud.colors, render_kfs, cfg.output_dir, self.bus,
                mesh_glb_path=self.artifacts.output_paths.get("mesh.glb"),
            )
            for comp in self.artifacts.comparisons:
                path = Path(comp.image_path)
                status = export.ExportStatus(
                    name=path.name, path=path, ok=True, size_bytes=path.stat().st_size if path.exists() else 0,
                )
                self._export_statuses.append(status)
                self.artifacts.output_paths[status.name] = str(path)
        except Exception as e:
            self._fallback("mesh_textured_model", e)

    # -- frame extraction / keyframes --------------------------------------

    def _extract_keyframes(self, video) -> list[PreparedKeyframe]:
        cfg = self.config
        decoder = FrameDecoder(video.path, self.bus, device=cfg.device0)

        end_s = cfg.quick_seconds if cfg.mode == "QUICK" else None
        raw_indices, raw_ts, raw_frames = [], [], []
        for idx, t, frame in decoder.iter_frames(
            start_s=0.0, end_s=end_s,
            target_fps=cfg.keyframe_sample_fps, scale_width=cfg.decode_working_width,
        ):
            if self.stop_event.is_set():
                break
            raw_indices.append(idx)
            raw_ts.append(t)
            raw_frames.append(frame)
            # There is intentionally no per-frame UI event here.  Passing a
            # 1920px ndarray through a dashboard queue and PNG encoder can
            # monopolize Kaggle's vCPUs.  The authoritative sharpness pass
            # below processes all candidates efficiently in GPU batches.
            max_keyframes = cfg.quick_max_keyframes if cfg.mode == "QUICK" else cfg.full_max_keyframes
            if len(raw_frames) >= max_keyframes * 4:
                break  # decode a bounded multiple of the target count; selection will thin it out

        if not raw_frames:
            return []

        gps_per_frame = None
        if self.telemetry is not None and self.telemetry.samples:
            synced = sync_to_frame_times(self.telemetry, raw_ts)
            enu_pts, origin, zone_epsg = track_to_enu(self.telemetry)
            self.epsg = zone_epsg[1]
            origin_lat, origin_lon, origin_alt = origin
            gps_per_frame = []
            for s in synced:
                if s is None:
                    gps_per_frame.append(None)
                    continue
                from .telemetry import geodetic_to_ecef, ecef_to_enu

                alt = s.alt if s.alt is not None else 0.0
                x, y, z = geodetic_to_ecef(s.lat, s.lon, alt)
                gps_per_frame.append(ecef_to_enu(x, y, z, origin_lat, origin_lon, origin_alt))
            self._telemetry_synced = synced

        selection = select_keyframes(
            raw_indices, raw_ts, raw_frames, gps_per_frame, self.bus,
            sharpness_percentile_floor=cfg.keyframe_sharpness_percentile_floor,
            min_gps_spacing_m=cfg.keyframe_min_gps_spacing_m,
            max_no_gps_frame_stride=cfg.keyframe_max_no_gps_stride,
            device=cfg.device0 if "cuda" in cfg.device0 else "cpu",
        )
        for c in selection.rejected:
            img = raw_frames[raw_indices.index(c.frame_index)] if c.frame_index in raw_indices else None
            self._emit(EventType.KEYFRAME_REJECTED, thumbnail=img)
            self.artifacts.keyframes.append(KeyframeRecord(
                frame_index=c.frame_index, timestamp_s=c.timestamp_s,
                thumbnail=_make_thumbnail(img) if img is not None else None,
                sharpness=c.sharpness, accepted=False, reject_reason=c.reject_reason,
            ))

        accepted = selection.accepted
        max_keyframes = cfg.quick_max_keyframes if cfg.mode == "QUICK" else cfg.full_max_keyframes
        if len(accepted) > max_keyframes:
            step = len(accepted) / max_keyframes
            accepted = [accepted[int(i * step)] for i in range(max_keyframes)]

        prepared = []
        for cand in accepted:
            local_i = raw_indices.index(cand.frame_index)
            img = raw_frames[local_i]
            h, w = img.shape[:2]
            intrinsics = self._resolve_intrinsics(w, h)
            prepared.append(PreparedKeyframe(
                frame_index=cand.frame_index, timestamp_s=cand.timestamp_s, image_full=img,
                gps_enu=cand.gps_enu, intrinsics=intrinsics, camera_pose_c2w_prior=None,
            ))
            self._emit(EventType.KEYFRAME_ACCEPTED, thumbnail=img)
            self.artifacts.keyframes.append(KeyframeRecord(
                frame_index=cand.frame_index, timestamp_s=cand.timestamp_s,
                thumbnail=_make_thumbnail(img), sharpness=cand.sharpness, accepted=True,
            ))
            if cand.gps_enu is not None:
                self.artifacts.gps_track_enu.append(cand.gps_enu)

        return prepared

    def _resolve_intrinsics(self, w: int, h: int) -> np.ndarray:
        info = self.detected.intrinsics
        if info:
            flat = {}

            def collect(d, depth=0):
                if depth > 3:
                    return
                for k, v in d.items():
                    flat[k] = v
                    if isinstance(v, dict):
                        collect(v, depth + 1)

            collect(info)
            if all(k in flat for k in ("fx", "fy", "cx", "cy")):
                return np.array([[flat["fx"], 0, flat["cx"]], [0, flat["fy"], flat["cy"]], [0, 0, 1.0]])
            for key in ("camera_matrix", "K", "intrinsic_matrix"):
                if key in flat:
                    mat = np.array(flat[key], dtype=float)
                    if mat.shape == (3, 3):
                        return mat
        return _default_intrinsics(w, h)

    def _attach_priors(self, keyframes: list[PreparedKeyframe]) -> None:
        if self.telemetry is None or not self.telemetry.samples:
            return
        synced = getattr(self, "_telemetry_synced", None)
        for kf in keyframes:
            if kf.gps_enu is None:
                continue
            sample_idx = None
            # `synced` is aligned to the same frame timestamps used during
            # extraction; find the matching sample by nearest timestamp
            # (cheap linear scan — keyframe counts are small by design).
            if synced:
                best = min(range(len(synced)), key=lambda i: abs((synced[i].t if synced[i] else 1e18) - kf.timestamp_s), default=None)
                if best is not None and synced[best] is not None:
                    s = synced[best]
                    attitude = (s.gimbal_yaw or s.yaw, s.gimbal_pitch or s.pitch, s.gimbal_roll or s.roll)
                    if all(a is not None for a in attitude):
                        kf.camera_pose_c2w_prior = _attitude_to_cam2world(np.array(kf.gps_enu), attitude)
                        continue
            # No attitude available: translation-only prior (identity rotation).
            pose = np.eye(4)
            pose[:3, 3] = kf.gps_enu
            kf.camera_pose_c2w_prior = pose

    # -- chunking --------------------------------------------------------

    def _make_chunks(self, keyframes: list[PreparedKeyframe]) -> list[list[PreparedKeyframe]]:
        cfg = self.config
        step = max(cfg.chunk_size - cfg.chunk_overlap, 1)
        chunks = []
        i = 0
        n = len(keyframes)
        while i < n:
            chunk = keyframes[i:i + cfg.chunk_size]
            if chunk:
                chunks.append(chunk)
            if i + cfg.chunk_size >= n:
                break
            i += step
        return chunks or [keyframes]

    # -- per-chunk processing ----------------------------------------------

    def _infer_chunk_backbone(
        self, bb, prior_mode: PriorMode, chunk_kfs: list[PreparedKeyframe], chunk_idx: int,
    ) -> tuple[list[PreparedKeyframe], ChunkResult | None]:
        """Backbone inference only — no alignment. Split out from
        `_process_chunk` (which still does inference+alignment together,
        for the single-GPU path, unchanged) so DUAL_GPU's two backbone
        worker threads can call this directly and hand the raw result to
        a single ordered "stitcher" step instead — alignment carries
        state between consecutive chunks (`_prev_chunk_cam_local`), so it
        must stay strictly sequential even when inference itself runs in
        parallel across two GPUs."""
        cfg = self.config
        views = [
            ViewInput(
                image=_resize_for_backbone(kf.image_full, cfg.backbone_input_size),
                intrinsics=self._scale_intrinsics(kf.intrinsics, kf.image_full.shape, cfg.backbone_input_size),
                camera_pose_c2w=kf.camera_pose_c2w_prior,
                frame_index=kf.frame_index, timestamp_s=kf.timestamp_s,
            )
            for kf in chunk_kfs
        ]

        chunk_size = len(views)
        while True:
            try:
                result = bb.infer_chunk(views, prior_mode)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and chunk_size > 2:
                    chunk_size = max(chunk_size // 2, 2)
                    self._log(f"OOM on chunk {chunk_idx} (device {getattr(bb, 'device', '?')}) — halving to {chunk_size} views and retrying", level="warn")
                    self._emit(EventType.STAGE_FALLBACK, stage="geometric_reconstruction", note=f"OOM, halved chunk to {chunk_size}")
                    views = views[:chunk_size]
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                self._fallback("geometric_reconstruction", e)
                return chunk_kfs[:chunk_size], None
            except Exception as e:
                self._fallback("geometric_reconstruction", e)
                return chunk_kfs[:chunk_size], None

        return chunk_kfs[:chunk_size], result

    def _finish_chunk_alignment(self, chunk_kfs: list[PreparedKeyframe], result: ChunkResult, chunk_idx: int) -> None:
        """Alignment + the live geometry preview — the sequential tail end
        of processing one chunk, called from the single-GPU path's loop
        directly (via _process_chunk) or from DUAL_GPU's ordered stitcher."""
        try:
            self._align_chunk(chunk_kfs, result, chunk_idx)
        except Exception as e:
            self._fallback("large_scale_alignment", e)

        depth_preview = result.points_world[0, ..., 2] if len(result.points_world) else None
        conf_preview = result.confidence[0] if len(result.confidence) else None
        self._emit(EventType.GEOMETRY_CHUNK, depth=depth_preview, confidence=conf_preview)

    def _process_chunk(self, bb, prior_mode: PriorMode, chunk_kfs: list[PreparedKeyframe], chunk_idx: int) -> ChunkResult | None:
        chunk_kfs, result = self._infer_chunk_backbone(bb, prior_mode, chunk_kfs, chunk_idx)
        if result is None:
            return None
        self._finish_chunk_alignment(chunk_kfs, result, chunk_idx)
        return result

    def _chunk_rate_label(self, done: int, total: int, stage_t0: float) -> str:
        """Shared by the single-GPU chunk loop and the DUAL_GPU stitcher —
        throughput + ETA for the dashboard's per-stage rate label, same
        purpose as decode.py's fps line: tell a slow-but-working stage
        apart from a genuinely stalled one at a glance."""
        elapsed = time.time() - stage_t0
        rate_per_min = (done / elapsed) * 60.0 if elapsed > 0 else 0.0
        eta = f", ETA {(total - done) * (elapsed / done):.0f}s" if done > 0 and done < total else ""
        return f"{done}/{total} chunks, {rate_per_min:.1f}/min{eta}"

    def _scale_intrinsics(self, K: np.ndarray, orig_shape: tuple, target_size: int) -> np.ndarray:
        h, w = orig_shape[:2]
        sx, sy = target_size / w, target_size / h
        K2 = K.copy()
        K2[0, 0] *= sx
        K2[0, 2] *= sx
        K2[1, 1] *= sy
        K2[1, 2] *= sy
        return K2

    def _align_chunk(self, chunk_kfs: list[PreparedKeyframe], result: ChunkResult, chunk_idx: int) -> None:
        cam_local = result.camera_poses_est[:, :3, 3]
        cam_local_R = result.camera_poses_est[:, :3, :3]

        if self.georeferenced:
            gps_pts = np.array([kf.gps_enu if kf.gps_enu is not None else [np.nan] * 3 for kf in chunk_kfs])
            valid = ~np.isnan(gps_pts).any(axis=1)
            if valid.sum() >= 2:
                attitudes = [
                    getattr(kf, "_attitude", None) for kf in chunk_kfs
                ]
                gravity = align.estimate_gravity_ransac(result.points_world[valid].reshape(-1, 3)) or align.GravityEstimate(
                    up_local=np.array([0.0, 0.0, 1.0]), source="assumed_identity", confidence=0.2,
                )
                if valid.sum() >= 2:
                    alt_trend = gps_pts[valid][:, 2]
                    gravity = align.disambiguate_gravity_sign(gravity, cam_local[valid], alt_trend)
                alignment = align.align_chunk_4dof(cam_local[valid], gps_pts[valid], gravity)
            else:
                alignment = align.first_chunk_identity(georeferenced=True)
        else:
            overlap_keys = [kf.frame_index for kf in chunk_kfs if kf.frame_index in self._prev_chunk_cam_local]
            if chunk_idx == 0 or not overlap_keys:
                alignment = align.first_chunk_identity(georeferenced=False)
            else:
                idx_map = {kf.frame_index: i for i, kf in enumerate(chunk_kfs)}
                src = np.array([cam_local[idx_map[k]] for k in overlap_keys])
                dst = np.array([self._prev_chunk_cam_local[k] for k in overlap_keys])
                if len(src) >= 3:
                    alignment = align.align_chunk_to_previous(src, dst)
                else:
                    alignment = self.chunk_alignments[-1] if self.chunk_alignments else align.first_chunk_identity(False)

        self.chunk_alignments.append(alignment)
        record = ChunkAlignmentRecord(chunk_idx=chunk_idx, mode=alignment.mode, rmse_before_m=alignment.rmse_m)
        self._chunk_alignment_records[chunk_idx] = record
        self.artifacts.chunk_alignments.append(record)

        world_cam = alignment.apply(cam_local)
        for i, (kf, wc) in enumerate(zip(chunk_kfs, world_cam)):
            kf._resolved_world_pose = wc  # noqa: SLF001 — internal bookkeeping between pipeline stages
            # Rotation composes under a similarity transform independent of
            # scale/translation: alignment.apply(p) = scale*(R@p)+t, so a
            # vector/orientation transforms as R_world = alignment.R @
            # R_local. This used to be dropped entirely (_resolved_pose
            # returned identity rotation always), which meant texture
            # baking's "most fronto-parallel view" scoring and any
            # reprojection-based comparison were both working with a camera
            # that was never actually pointed the way it was really pointed.
            kf._resolved_world_rotation = alignment.R @ cam_local_R[i]  # noqa: SLF001
            self._emit(EventType.TRAJECTORY_POINT, kind="camera", x=wc[0], y=wc[1])
            self.artifacts.camera_track_enu.append(tuple(wc))
            if self.georeferenced and kf.gps_enu is not None:
                self.gps_factors.append(align.GpsFactor(chunk=chunk_idx, cam_center_local=cam_local[len(self.gps_factors) % len(cam_local)], gps_enu=np.array(kf.gps_enu)))
            if self.live_page is not None:
                # Forward/up are only used by the client to orient the
                # camera-frustum helper, not for any geometry math here — a
                # coarse estimate (next waypoint direction, world +Z up) is
                # enough for that purpose.
                forward = world_cam[i + 1] - wc if i + 1 < len(world_cam) else (wc - world_cam[i - 1] if i > 0 else np.array([1.0, 0.0, 0.0]))
                norm = np.linalg.norm(forward)
                forward = forward / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
                self._push_live(
                    self.live_page.push_camera_pose,
                    frame_index=kf.frame_index, timestamp_s=kf.timestamp_s,
                    position=wc, forward=forward, up=np.array([0.0, 0.0, 1.0]),
                )

        overlap_keys = [kf.frame_index for kf in chunk_kfs if kf.frame_index in self._prev_chunk_cam_local]
        if overlap_keys:
            idx_map = {kf.frame_index: i for i, kf in enumerate(chunk_kfs)}
            self.overlap_constraints.append(align.OverlapConstraint(
                chunk_a=chunk_idx - 1, chunk_b=chunk_idx,
                cam_centers_a_local=np.array([self._prev_chunk_cam_local[k] for k in overlap_keys]),
                cam_centers_b_local=np.array([cam_local[idx_map[k]] for k in overlap_keys]),
            ))

        overlap_start = max(len(chunk_kfs) - self.config.chunk_overlap, 0)
        self._prev_chunk_cam_local = {kf.frame_index: cam_local[i] for i, kf in enumerate(chunk_kfs[overlap_start:], start=overlap_start)}

        self._last_alignment = alignment

    def _resolved_pose(self, kf: PreparedKeyframe, chunk_idx) -> np.ndarray:
        wc = getattr(kf, "_resolved_world_pose", None)
        wr = getattr(kf, "_resolved_world_rotation", None)
        pose = np.eye(4)
        if wr is not None:
            pose[:3, :3] = wr
        if wc is not None:
            pose[:3, 3] = wc
        return pose

    # -- masking + fusion (inline or GPU1 worker) ---------------------------

    def _mask_and_fuse_chunk(
        self, chunk_idx: int, chunk_kfs: list[PreparedKeyframe], result: ChunkResult,
        masker: "masks.DynamicObjectMasker | None" = None,
    ) -> None:
        """`masker` defaults to `self.masker` (the single-GPU/GPU1-worker
        behavior, unchanged) — DUAL_GPU's stitcher passes `self.masker0`
        or `self.masker` explicitly per chunk instead, so masking runs on
        whichever GPU actually produced that chunk's geometry rather than
        being pinned to one device."""
        if masker is None:
            masker = self.masker
        alignment = self.chunk_alignments[chunk_idx] if chunk_idx < len(self.chunk_alignments) else None
        if alignment is None:
            return

        points_world = alignment.apply(result.points_world)
        colors = np.stack([_resize_for_backbone(kf.image_full, points_world.shape[1]) for kf in chunk_kfs], axis=0)

        dynamic_mask = None
        if masker is not None:
            try:
                masks_list = [masker.mask_frame(colors[i]).mask for i in range(len(chunk_kfs))]
                dynamic_mask = np.stack(masks_list, axis=0)
            except Exception as e:
                self._fallback("dense_point_cloud", e)

        self.artifacts.add_geometry_sample(ChunkGeometrySample(
            chunk_idx=chunk_idx, rgb=colors[0].copy(),
            depth=result.points_world[0, ..., 2].copy(), confidence=result.confidence[0].copy(),
            dynamic_mask=dynamic_mask[0].copy() if dynamic_mask is not None else None,
        ))

        n_new = self.fusion_acc.add_chunk(
            points_world, colors, result.confidence, dynamic_mask=dynamic_mask,
            min_confidence=self.config.min_confidence,
        )
        # Full-resolution chunk, not downsampled — the inline viewer (see
        # dashboard.py's Live 3D panel) needs real data to stream, and it
        # does its own LOD above 2M accumulated points. A too-small preview
        # here would defeat the point of a "full-quality" live view.
        flat_pts = points_world.reshape(-1, 3)
        flat_cols = colors.reshape(-1, 3)
        flat_conf = result.confidence.reshape(-1)
        keep = np.isfinite(flat_pts).all(axis=1)
        if dynamic_mask is not None:
            keep &= ~dynamic_mask.reshape(-1)
        self._emit(EventType.POINTCLOUD_GROWTH, points=flat_pts[keep], colors=flat_cols[keep], confidence=flat_conf[keep])
        self.artifacts.point_count = len(self.fusion_acc)
        _ = n_new

        # Live reconstruction page: pushed PER FRAME (not the whole chunk in
        # one message) so the client can reveal points frame by frame in
        # sync with the left-hand video instead of a single chunk-sized pop.
        # `masked` marks points the dynamic-object masker dropped from the
        # fused cloud above — the client shows those in red when its
        # "show masked-out objects" toggle is on rather than hiding them,
        # so still emit them here (only NaNs are actually dropped).
        if self.live_page is not None:
            for i, kf in enumerate(chunk_kfs):
                pts_i = points_world[i].reshape(-1, 3)
                keep_i = np.isfinite(pts_i).all(axis=1)
                masked_i = (
                    dynamic_mask[i].reshape(-1)[keep_i].astype(np.uint8)
                    if dynamic_mask is not None
                    else np.zeros(int(keep_i.sum()), dtype=np.uint8)
                )
                self._push_live(
                    self.live_page.push_points,
                    pts_i[keep_i], colors[i].reshape(-1, 3)[keep_i],
                    result.confidence[i].reshape(-1)[keep_i], masked_i,
                    frame_index=kf.frame_index, timestamp_s=kf.timestamp_s,
                )

    def _gpu1_worker(self) -> None:
        while True:
            item = self._chunk_queue.get()
            if item is self._SENTINEL:
                break
            chunk_idx, chunk_kfs, result = item
            try:
                self._mask_and_fuse_chunk(chunk_idx, chunk_kfs, result)
            except Exception as e:
                self._fallback("dense_point_cloud", e)
            frac = None
            self._emit(EventType.STAGE_PROGRESS, stage="dense_point_cloud", frac=1.0)

    # -- DUAL_GPU: data-parallel backbone across both GPUs -------------------

    def _run_chunks_dual_gpu(self, chunks: list, bb0, bb1, prior_mode: PriorMode) -> None:
        """Even chunks run backbone inference on device0, odd chunks on
        device1, concurrently — chunk-local geometry is independent of
        other chunks (only alignment carries state between them), so this
        is safe. Alignment/masking/fusion then happen in ONE ordered
        "stitcher" loop (this method's own thread) that waits for each
        chunk_idx in turn: exactly the same single-writer guarantee the
        single-GPU path already relies on for `chunk_alignments`,
        `fusion_acc`, and `_prev_chunk_cam_local`, just fed by two
        producers instead of one.

        An OOM on either GPU is handled entirely within that GPU's own
        worker (via _infer_chunk_backbone's existing halve-and-retry) —
        it never touches the other GPU's queue, so a bad chunk on device1
        never demotes device0 to running solo.

        Known scope limit, stated plainly: "prefetch" here means each
        GPU's queue can hold a few chunks of lookahead (dispatch is
        decoupled from inference, so the next chunk is already queued
        the moment a worker is free) — it does NOT mean pinned-memory,
        non_blocking CPU->GPU tensor transfers inside the backbone's own
        inference call, which would require changes to backbone.py's
        model-loading internals that can't be verified without real GPU
        hardware. The queue-level prefetch is what actually keeps a GPU
        from sitting idle between chunks; the pinned-memory piece is a
        finer-grained optimization on top that's deferred.
        """
        cfg = self.config
        n = len(chunks)
        stage_t0 = time.time()
        q0: "queue.Queue" = queue.Queue(maxsize=3)
        q1: "queue.Queue" = queue.Queue(maxsize=3)
        results: dict[int, tuple[list[PreparedKeyframe], ChunkResult | None]] = {}
        results_cv = threading.Condition()
        stop = self.stop_event

        def worker(bb, q: "queue.Queue", device_label: str) -> None:
            while True:
                wait_t0 = time.time()
                item = q.get()
                waited = time.time() - wait_t0
                self._gpu_idle_s[device_label] = self._gpu_idle_s.get(device_label, 0.0) + waited
                if waited > 5.0:
                    self.bus.log(f"{device_label} waited {waited:.1f}s for its next chunk (other GPU/dispatch is the bottleneck)", level="warn")
                if item is self._SENTINEL:
                    break
                chunk_idx, chunk_kfs = item
                if stop.is_set():
                    with results_cv:
                        results[chunk_idx] = (chunk_kfs, None)
                        results_cv.notify_all()
                    continue
                trimmed_kfs, result = self._infer_chunk_backbone(bb, prior_mode, chunk_kfs, chunk_idx)
                with results_cv:
                    results[chunk_idx] = (trimmed_kfs, result)
                    results_cv.notify_all()

        def dispatch() -> None:
            for chunk_idx, chunk_kfs in enumerate(chunks):
                if stop.is_set():
                    break
                (q0 if chunk_idx % 2 == 0 else q1).put((chunk_idx, chunk_kfs))
            q0.put(self._SENTINEL)
            q1.put(self._SENTINEL)

        dispatch_thread = threading.Thread(target=dispatch, name="sih3d-dual-dispatch", daemon=True)
        worker0_thread = threading.Thread(target=worker, args=(bb0, q0, f"backbone-worker[{cfg.device0}]"), name="sih3d-backbone0", daemon=True)
        worker1_thread = threading.Thread(target=worker, args=(bb1, q1, f"backbone-worker[{cfg.device1}]"), name="sih3d-backbone1", daemon=True)
        dispatch_thread.start()
        worker0_thread.start()
        worker1_thread.start()

        for chunk_idx in range(n):
            if stop.is_set():
                break
            with results_cv:
                while chunk_idx not in results and not stop.is_set():
                    results_cv.wait(timeout=5.0)
                if chunk_idx not in results:
                    break
                chunk_kfs, result = results.pop(chunk_idx)

            frac = (chunk_idx + 1) / n
            rate_label = self._chunk_rate_label(chunk_idx + 1, n, stage_t0)
            self._emit(EventType.STAGE_PROGRESS, stage="geometric_reconstruction", frac=frac, rate_label=rate_label)
            self._emit(EventType.STAGE_PROGRESS, stage="large_scale_alignment", frac=frac, rate_label=rate_label)

            if result is None:
                continue

            self._finish_chunk_alignment(chunk_kfs, result, chunk_idx)
            # Masking runs on whichever GPU produced this chunk's geometry
            # — device0's own masker for even chunks, device1's for odd —
            # rather than pinned to one device, so masking work is spread
            # across both GPUs instead of bottlenecking on one.
            masker = self.masker0 if (chunk_idx % 2 == 0) else self.masker
            self._mask_and_fuse_chunk(chunk_idx, chunk_kfs, result, masker=masker)
            self._emit(EventType.STAGE_PROGRESS, stage="dense_point_cloud", frac=frac, rate_label=rate_label)

        dispatch_thread.join(timeout=60)
        worker0_thread.join(timeout=300)
        worker1_thread.join(timeout=300)

    def _report_dual_gpu_utilization(self, stage_t0: float) -> None:
        """Logs per-GPU average SM utilization + near-idle fraction over
        the just-finished chunk-processing window, flagging anything
        under the >70% avg / >20% idle targets. The same numbers are
        also in report.json per-stage (avg_gpu_util_pct, already computed
        from the same GPU_SAMPLE history) for anyone inspecting the file
        directly rather than the live log."""
        t1 = time.time()
        for idx in range(self.gpu_monitor.device_count):
            avg = self.gpu_monitor.history.average_util(idx, stage_t0, t1)
            idle_frac = self.gpu_monitor.history.idle_fraction(idx, stage_t0, t1, idle_below_pct=5.0)
            if avg is None:
                continue
            note = f"DUAL_GPU: GPU{idx} averaged {avg:.0f}% SM utilization during geometry processing"
            if idle_frac is not None:
                note += f" ({idle_frac * 100:.0f}% of samples near-idle)"
            flagged = avg < 70.0 or (idle_frac or 0.0) > 0.20
            self._log(note, level="warn" if flagged else "info")

    def _report_full_run_utilization(self, idle_below_pct: float = 5.0) -> None:
        """Unconditional (unlike _report_dual_gpu_utilization, which only
        ever ran under DUAL_GPU and only for the geometry-processing
        window) — the actual "how much of the whole run did each GPU/the
        CPU sit idle" answer, for every run regardless of mode, so
        optimizing wall time has real numbers to aim at instead of
        eyeballing a screenshot of a resource widget mid-run. Always runs,
        even on a failed/partial pipeline (called from _run_guarded's
        finally), since a stall's utilization signature is itself useful
        diagnostic information."""
        t0, t1 = self._run_t0, time.time()
        total_s = t1 - t0
        if total_s <= 0:
            return

        lines = [f"===== Resource utilization over the full run ({total_s:.0f}s) ====="]
        for idx in range(self.gpu_monitor.device_count):
            avg = self.gpu_monitor.history.average_util(idx, t0, t1)
            dec_avg = self.gpu_monitor.history.average_util(idx, t0, t1, metric="decoder_util_pct")
            idle_frac = self.gpu_monitor.history.idle_fraction(idx, t0, t1, idle_below_pct=idle_below_pct)
            if avg is None:
                lines.append(f"GPU{idx}: no samples collected")
                continue
            idle_s = (idle_frac or 0.0) * total_s
            lines.append(
                f"GPU{idx}: {avg:.0f}% avg SM util, {dec_avg or 0.0:.0f}% avg NVDEC util — "
                f"idle (<{idle_below_pct:.0f}% SM) for {idle_s:.0f}s ({(idle_frac or 0.0) * 100:.0f}% of the run)"
            )
        cpu_avg = self.gpu_monitor.cpu_history.average(t0, t1)
        if cpu_avg is not None:
            cpu_vals = [s.percent for s in self.gpu_monitor.cpu_history.samples if t0 <= s.ts <= t1]
            cpu_idle_frac = (sum(1 for v in cpu_vals if v < 15.0) / len(cpu_vals)) if cpu_vals else 0.0
            lines.append(f"CPU: {cpu_avg:.0f}% avg (all cores) — idle (<15%) for {cpu_idle_frac * total_s:.0f}s ({cpu_idle_frac * 100:.0f}% of the run)")
        else:
            lines.append("CPU: no samples collected (psutil unavailable)")

        # The single most actionable signal for "which resource is the
        # bottleneck right now": every GPU idle at the same time CPU is
        # busy means the current stage is CPU-bound (meshing/xatlas-style
        # work), not something more GPU compute would speed up — the
        # opposite (GPUs busy, CPU idle) means backbone/decode work, where
        # DUAL_GPU or decode tuning are the actual levers.
        lines.append("(low GPU% + high CPU% together = a CPU-bound stage, e.g. meshing — more/faster GPUs won't help there;")
        lines.append(" high GPU% + low CPU% = a GPU-bound stage, e.g. backbone inference/decode — DUAL_GPU or decode tuning help there.)")
        self._log("\n".join(lines))
        for device_label, idle_s in self._gpu_idle_s.items():
            if idle_s > 1.0:
                self._log(f"DUAL_GPU: {device_label} spent {idle_s:.1f}s total waiting for its input queue", level="warn" if idle_s > 10 else "info")

    # -- global alignment refinement -----------------------------------------

    def _refine_global_alignment(self) -> float | None:
        if not self.chunk_alignments:
            return None
        if not self.overlap_constraints and not self.gps_factors:
            rmses = [a.rmse_m for a in self.chunk_alignments if a.rmse_m is not None]
            return float(np.mean(rmses)) if rmses else None

        refined = align.refine_pose_graph(self.chunk_alignments, self.overlap_constraints, self.gps_factors, self.bus)
        self.chunk_alignments = refined
        for i, ca in enumerate(refined):
            record = self._chunk_alignment_records.get(i)
            if record is not None:
                record.rmse_after_m = ca.rmse_m
        rmses = [a.rmse_m for a in refined if a.rmse_m is not None]
        return float(np.mean(rmses)) if rmses else None

    # -- export --------------------------------------------------------------

    def _export_all(self, cloud: FusedPointCloud, mesh_result, bake_result, keyframes: list[PreparedKeyframe]) -> None:
        cfg = self.config
        out = cfg.output_dir

        def rec(status):
            self._export_statuses.append(status)
            self.report.add_output(status)
            if status.ok and status.path is not None:
                self.artifacts.output_paths[status.name] = str(status.path)

        rec(export.export_pointcloud_ply(cloud, out / "pointcloud.ply", self.bus))
        rec(export.export_pointcloud_las(cloud, out / "pointcloud.las", self.bus, self.epsg, self.georeferenced))

        obj_status = export.export_mesh_obj(mesh_result, bake_result, out, self.bus)
        rec(obj_status)
        glb_status = export.export_mesh_glb(mesh_result, bake_result, out / "mesh.glb", self.bus)
        rec(glb_status)
        rec(export.export_mesh_ply(mesh_result, bake_result, out / "mesh.ply", self.bus))
        rec(export.export_mesh_fbx(obj_status.path, out / "mesh.fbx", self.bus))

        dsm_status, ortho_status = export.export_dsm_orthomosaic(cloud, out, self.bus, self.epsg, self.georeferenced, cfg.dsm_cell_size_m)
        rec(dsm_status)
        rec(ortho_status)
        coverage_status = export.export_coverage(cloud, out, self.bus, self.epsg, self.georeferenced, cfg.dsm_cell_size_m)
        rec(coverage_status)
        self._emit(
            EventType.RASTERS_READY,
            dsm_path=str(dsm_status.path) if dsm_status.ok else None,
            orthomosaic_path=str(ortho_status.path) if ortho_status.ok else None,
            coverage_path=str(coverage_status.path) if coverage_status.ok else None,
        )

        gps_track = [kf.gps_enu for kf in keyframes if kf.gps_enu is not None]
        cam_track = [getattr(kf, "_resolved_world_pose", None) for kf in keyframes]
        cam_track = [c for c in cam_track if c is not None]
        if gps_track and self.telemetry is not None:
            origin = track_to_enu(self.telemetry)[1]
            geo_status, kml_status = export.export_trajectories(gps_track, cam_track, origin, out, self.bus)
            rec(geo_status)
            rec(kml_status)

        try:
            from .viewer import write_viewer_html

            viewer_status = write_viewer_html(
                out / "viewer.html", self.bus,
                cloud=cloud, mesh_glb_path=glb_status.path,
                gps_track_enu=gps_track, camera_track_enu=cam_track,
                georeferenced=self.georeferenced,
            )
            rec(viewer_status)
        except Exception as e:
            self._fallback("mesh_textured_model", e)

        self.report.write_json(out / "report.json")
        self.report.write_html(out / "report.html")

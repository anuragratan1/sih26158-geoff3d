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
    quick_max_keyframes: int = 120
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


class Pipeline:
    def __init__(
        self, config: PipelineConfig, detected: DetectedInputs, telemetry: TelemetryTrack | None,
        bus: EventBus, gpu_monitor: GpuMonitor, report: ReportBuilder,
        backbone_override: tuple | None = None,
    ):
        """`backbone_override`: (GeometryBackbone, PriorMode), pre-built and
        already loaded. Testing/dependency-injection hook only — the real
        notebook cells never pass this, so production behavior (backbone
        chosen via backbone.select_backbone from BACKBONE/PRIOR_MODE config)
        is unaffected. Used by the local dry-run harness to exercise the
        entire orchestration loop on CPU with a synthetic backbone, without
        downloading real multi-GB models."""
        self.config = config
        self.detected = detected
        self.telemetry = telemetry
        self.bus = bus
        self.gpu_monitor = gpu_monitor
        self.report = report
        self._backbone_override = backbone_override

        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu1_thread: threading.Thread | None = None
        self._chunk_queue: "queue.Queue" = queue.Queue(maxsize=3)
        self._SENTINEL = object()

        self.two_gpu = gpu_monitor.device_count >= 2
        self.fusion_acc = VoxelPointFusion(config.voxel_size_m, device=config.device0)
        self.masker: masks.DynamicObjectMasker | None = None

        self.chunk_alignments: list[align.ChunkAlignment] = []
        self.overlap_constraints: list[align.OverlapConstraint] = []
        self.gps_factors: list[align.GpsFactor] = []
        self._prev_chunk_cam_local: dict[int, np.ndarray] = {}  # keyframe_index -> local cam center, from the previous chunk only

        self.result: PipelineResult | None = None
        self._export_statuses: list = []
        self.georeferenced: bool = self.detected.telemetry_path is not None or self.detected.telemetry_kind == "embedded"
        self.epsg: int | None = None

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
        self.bus.publish(event_type, **payload)
        self.report.on_event(Event(type=event_type, payload=payload, ts=ts))

    def _log(self, message: str, level: str = "info") -> None:
        self.bus.log(message, level=level)
        self.report.on_event(Event(type=EventType.LOG, payload={"message": message, "level": level}, ts=time.time()))

    def _fallback(self, stage: str, exc: Exception) -> None:
        note = f"{type(exc).__name__}: {exc}"
        self._log(f"Stage '{stage}': {note} — continuing with degraded output", level="warn")
        self._emit(EventType.STAGE_FALLBACK, stage=stage, note=note)

    # -- main run ---------------------------------------------------------

    def _run(self) -> None:
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_monitor.start()

        video = self.detected.video
        if video is None:
            raise RuntimeError("no video detected — nothing to process")

        # Backbone selection is the one allowed-fatal setup step.
        if self._backbone_override is not None:
            bb, prior_mode = self._backbone_override
        else:
            bb, prior_mode = backbone_mod.select_backbone(
                cfg.backbone_choice, cfg.prior_mode, self.detected, self.bus,
                device=cfg.device0, use_finetuned=cfg.use_finetuned,
            )
        self.georeferenced = self.georeferenced and bb.name != "?" and prior_mode is not None
        # select_backbone already forces C/RGB when there's no telemetry;
        # re-derive georeferenced from what it actually decided, not our
        # own earlier guess.
        self.georeferenced = self.detected.telemetry_path is not None or self.detected.telemetry_kind == "embedded"
        if bb.name == "C" and prior_mode == PriorMode.RGB and not self.georeferenced:
            pass  # expected no-GPS path
        self.report.set_run_config(
            mode=cfg.mode, backbone=bb.name, prior_mode=prior_mode.value,
            checkpoint_source=getattr(bb, "checkpoint_source", "pretrained"),
            gpus=self.gpu_monitor.device_names,
        )
        if not self.georeferenced:
            self._log("NO TELEMETRY: outputs will be APPROXIMATE SCALE — NOT GEOREFERENCED", level="warn")

        if self.two_gpu:
            self.masker = masks.DynamicObjectMasker(self.bus, device=cfg.device1)

        # -- Stage: frame_extraction -------------------------------------
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

        # -- Chunking --------------------------------------------------------
        chunks = self._make_chunks(keyframes)
        self._log(f"Chunked {len(keyframes)} keyframes into {len(chunks)} chunk(s) (size={cfg.chunk_size}, overlap={cfg.chunk_overlap})")

        self._gpu1_thread = None
        if self.two_gpu:
            self._gpu1_thread = threading.Thread(target=self._gpu1_worker, name="sih3d-gpu1-worker", daemon=True)
            self._gpu1_thread.start()

        self._emit(EventType.STAGE_START, stage="geometric_reconstruction")
        self._emit(EventType.STAGE_START, stage="large_scale_alignment")
        self._emit(EventType.STAGE_START, stage="dense_point_cloud")

        for chunk_idx, chunk_kfs in enumerate(chunks):
            if self.stop_event.is_set():
                break
            chunk_result = self._process_chunk(bb, prior_mode, chunk_kfs, chunk_idx)
            frac = (chunk_idx + 1) / len(chunks)
            self._emit(EventType.STAGE_PROGRESS, stage="geometric_reconstruction", frac=frac)
            self._emit(EventType.STAGE_PROGRESS, stage="large_scale_alignment", frac=frac)

            if chunk_result is None:
                continue

            if self.two_gpu:
                try:
                    self._chunk_queue.put((chunk_idx, chunk_kfs, chunk_result), timeout=30)
                except queue.Full:
                    self._fallback("dense_point_cloud", RuntimeError("GPU1 worker queue full/stalled; dropping chunk"))
            else:
                self._mask_and_fuse_chunk(chunk_idx, chunk_kfs, chunk_result)
                self._emit(EventType.STAGE_PROGRESS, stage="dense_point_cloud", frac=frac)

        if self.two_gpu:
            self._chunk_queue.put(self._SENTINEL)
            if self._gpu1_thread:
                self._gpu1_thread.join(timeout=300)

        self._emit(EventType.STAGE_DONE, stage="geometric_reconstruction")
        self._emit(EventType.STAGE_DONE, stage="large_scale_alignment")
        self._emit(EventType.STAGE_DONE, stage="dense_point_cloud")

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

        mesh_result = mesh_mod.MeshResult(mesh=None, method="none")
        bake_result = None
        try:
            mesh_result = mesh_mod.build_vertex_colored_mesh(cloud, tsdf=None, bus=self.bus)
        except Exception as e:
            self._fallback("mesh_textured_model", e)

        try:
            if mesh_result.mesh is not None:
                kf_for_bake = [
                    KeyframeForBaking(image_rgb=kf.image_full, camera_pose_c2w=self._resolved_pose(kf, chunk_idx=None), intrinsics=kf.intrinsics)
                    for kf in keyframes if getattr(kf, "_resolved_world_pose", None) is not None
                ]
                bake_result = mesh_mod.bake_texture(mesh_result.mesh, kf_for_bake, self.bus, time_budget_s=cfg.texture_time_budget_s)
        except Exception as e:
            self._fallback("mesh_textured_model", e)
        self._emit(EventType.MESH_PREVIEW, mesh=mesh_result)
        self._emit(EventType.STAGE_DONE, stage="mesh_textured_model")

        self.report.set_geometry_summary(
            georeferenced=self.georeferenced, collinearity_index=collinearity, alignment_rmse_m=alignment_rmse,
            point_count=len(cloud.points), mesh_faces=mesh_result.n_faces,
        )

        # -- Export ------------------------------------------------------
        self._export_all(cloud, mesh_result, bake_result, keyframes)

    # -- frame extraction / keyframes --------------------------------------

    def _extract_keyframes(self, video) -> list[PreparedKeyframe]:
        cfg = self.config
        decoder = FrameDecoder(video.path, self.bus, device=cfg.device0)

        end_s = cfg.quick_seconds if cfg.mode == "QUICK" else None
        raw_indices, raw_ts, raw_frames = [], [], []
        for idx, t, frame in decoder.iter_frames(start_s=0.0, end_s=end_s):
            if self.stop_event.is_set():
                break
            raw_indices.append(idx)
            raw_ts.append(t)
            raw_frames.append(frame)
            self._emit(EventType.FRAME_DECODED, frame=frame, frame_index=idx, sharpness=0.0)
            if cfg.mode == "QUICK" and len(raw_frames) >= cfg.quick_max_keyframes * 4:
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
            self._emit(EventType.KEYFRAME_REJECTED, thumbnail=raw_frames[raw_indices.index(c.frame_index)] if c.frame_index in raw_indices else None)

        accepted = selection.accepted
        if cfg.mode == "QUICK" and len(accepted) > cfg.quick_max_keyframes:
            step = len(accepted) / cfg.quick_max_keyframes
            accepted = [accepted[int(i * step)] for i in range(cfg.quick_max_keyframes)]

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

    def _process_chunk(self, bb, prior_mode: PriorMode, chunk_kfs: list[PreparedKeyframe], chunk_idx: int) -> ChunkResult | None:
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
                    self._log(f"OOM on chunk {chunk_idx} — halving to {chunk_size} views and retrying", level="warn")
                    self._emit(EventType.STAGE_FALLBACK, stage="geometric_reconstruction", note=f"OOM, halved chunk to {chunk_size}")
                    views = views[:chunk_size]
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                self._fallback("geometric_reconstruction", e)
                return None
            except Exception as e:
                self._fallback("geometric_reconstruction", e)
                return None

        chunk_kfs = chunk_kfs[:chunk_size]
        try:
            self._align_chunk(chunk_kfs, result, chunk_idx)
        except Exception as e:
            self._fallback("large_scale_alignment", e)

        depth_preview = result.points_world[0, ..., 2] if len(result.points_world) else None
        conf_preview = result.confidence[0] if len(result.confidence) else None
        self._emit(EventType.GEOMETRY_CHUNK, depth=depth_preview, confidence=conf_preview)

        return result

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
        world_cam = alignment.apply(cam_local)
        for kf, wc in zip(chunk_kfs, world_cam):
            kf._resolved_world_pose = wc  # noqa: SLF001 — internal bookkeeping between pipeline stages
            self._emit(EventType.TRAJECTORY_POINT, kind="camera", x=wc[0], y=wc[1])
            if self.georeferenced and kf.gps_enu is not None:
                self.gps_factors.append(align.GpsFactor(chunk=chunk_idx, cam_center_local=cam_local[len(self.gps_factors) % len(cam_local)], gps_enu=np.array(kf.gps_enu)))

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
        pose = np.eye(4)
        if wc is not None:
            pose[:3, 3] = wc
        return pose

    # -- masking + fusion (inline or GPU1 worker) ---------------------------

    def _mask_and_fuse_chunk(self, chunk_idx: int, chunk_kfs: list[PreparedKeyframe], result: ChunkResult) -> None:
        alignment = self.chunk_alignments[chunk_idx] if chunk_idx < len(self.chunk_alignments) else None
        if alignment is None:
            return

        points_world = alignment.apply(result.points_world)
        colors = np.stack([_resize_for_backbone(kf.image_full, points_world.shape[1]) for kf in chunk_kfs], axis=0)

        dynamic_mask = None
        if self.masker is not None:
            try:
                masks_list = [self.masker.mask_frame(colors[i]).mask for i in range(len(chunk_kfs))]
                dynamic_mask = np.stack(masks_list, axis=0)
            except Exception as e:
                self._fallback("dense_point_cloud", e)

        n_new = self.fusion_acc.add_chunk(
            points_world, colors, result.confidence, dynamic_mask=dynamic_mask,
            min_confidence=self.config.min_confidence,
        )
        sample = min(len(points_world.reshape(-1, 3)), 5000)
        flat_pts = points_world.reshape(-1, 3)[:sample]
        flat_cols = colors.reshape(-1, 3)[:sample]
        flat_conf = result.confidence.reshape(-1)[:sample]
        self._emit(EventType.POINTCLOUD_GROWTH, points=flat_pts, colors=flat_cols, confidence=flat_conf)
        _ = n_new

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

    # -- global alignment refinement -----------------------------------------

    def _refine_global_alignment(self) -> float | None:
        if not self.chunk_alignments:
            return None
        if not self.overlap_constraints and not self.gps_factors:
            rmses = [a.rmse_m for a in self.chunk_alignments if a.rmse_m is not None]
            return float(np.mean(rmses)) if rmses else None

        refined = align.refine_pose_graph(self.chunk_alignments, self.overlap_constraints, self.gps_factors, self.bus)
        self.chunk_alignments = refined
        rmses = [a.rmse_m for a in refined if a.rmse_m is not None]
        return float(np.mean(rmses)) if rmses else None

    # -- export --------------------------------------------------------------

    def _export_all(self, cloud: FusedPointCloud, mesh_result, bake_result, keyframes: list[PreparedKeyframe]) -> None:
        cfg = self.config
        out = cfg.output_dir

        def rec(status):
            self._export_statuses.append(status)
            self.report.add_output(status)

        rec(export.export_pointcloud_ply(cloud, out / "pointcloud.ply", self.bus))
        rec(export.export_pointcloud_las(cloud, out / "pointcloud.las", self.bus, self.epsg, self.georeferenced))

        obj_status = export.export_mesh_obj(mesh_result, bake_result, out, self.bus)
        rec(obj_status)
        glb_status = export.export_mesh_glb(mesh_result, bake_result, out / "mesh.glb", self.bus)
        rec(glb_status)
        rec(export.export_mesh_fbx(obj_status.path, out / "mesh.fbx", self.bus))

        dsm_status, ortho_status = export.export_dsm_orthomosaic(cloud, out, self.bus, self.epsg, self.georeferenced, cfg.dsm_cell_size_m)
        rec(dsm_status)
        rec(ortho_status)
        rec(export.export_coverage(cloud, out, self.bus, self.epsg, self.georeferenced, cfg.dsm_cell_size_m))

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

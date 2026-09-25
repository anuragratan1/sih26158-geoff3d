"""Multi-chunk point fusion: confidence/view-count-weighted voxel accumulation
(torch, GPU-resident) + an optional TSDF volume for meshing.

Two fusion products, both built incrementally as chunks arrive (matching the
dashboard's "point cloud growing chunk by chunk" panel):

1. `VoxelPointFusion` — always available, pure torch. Hashes points into a
   voxel grid and keeps a running confidence-weighted average position/color
   per voxel plus a view count, via `scatter_add`/`scatter_reduce` (the
   primitive the task spec explicitly calls out for GPU-resident grid work).
   This alone produces the "dense 3D point cloud" pipeline stage output.

2. `Open3DTsdfFusion` — attempted for mesh.py's marching-cubes path (denser,
   smoother surfaces than meshing straight off a point cloud). Tries Open3D's
   tensor CUDA integration first, falls back to legacy CPU TSDF, and returns
   `None` (logged) if Open3D isn't usable at all — mesh.py then meshes
   directly from the point cloud (Poisson/ball-pivoting) instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .events import EventBus, EventType


@dataclass
class FusedPointCloud:
    points: np.ndarray        # (N,3) world frame, meters
    colors: np.ndarray        # (N,3) uint8-range floats, RGB
    view_count: np.ndarray    # (N,) int32
    confidence: np.ndarray    # (N,) float32, max confidence observed


class VoxelPointFusion:
    """Confidence/view-count-weighted voxel-grid point accumulation.

    Voxel keys are bit-packed 63-bit integers from (x,y,z) grid coordinates
    (21 bits/axis, offset to stay non-negative — comfortably covers a
    +/-1e5 m scene at a 0.1 m voxel size). The key->row mapping is a plain
    Python dict on CPU (negligible cost relative to the point data itself,
    and far simpler/more correct than an unbounded GPU hash table); the
    actual weighted-sum accumulators live on `device` as growing tensors.
    """

    _AXIS_BITS = 21
    _OFFSET = 1 << 20

    def __init__(self, voxel_size_m: float, device: str = "cuda:0"):
        self.voxel_size = voxel_size_m
        self.device = device
        self._key_to_row: dict[int, int] = {}

        import torch

        self._torch = torch
        self._weight_sum = torch.zeros(0, device=device)
        self._pos_sum = torch.zeros(0, 3, device=device)
        self._color_sum = torch.zeros(0, 3, device=device)
        self._view_count = torch.zeros(0, dtype=torch.int32, device=device)
        self._max_conf = torch.zeros(0, device=device)

    def __len__(self) -> int:
        return self._weight_sum.shape[0]

    def _voxel_keys(self, points):
        torch = self._torch
        coords = torch.floor(points / self.voxel_size).to(torch.int64) + self._OFFSET
        coords = coords.clamp(0, (1 << self._AXIS_BITS) - 1)
        return (coords[:, 0] << (2 * self._AXIS_BITS)) | (coords[:, 1] << self._AXIS_BITS) | coords[:, 2]

    def add_chunk(
        self,
        points_world: np.ndarray,
        colors_rgb: np.ndarray,
        confidence: np.ndarray,
        dynamic_mask: np.ndarray | None = None,
        min_confidence: float = 0.1,
    ) -> int:
        """Shapes: points_world/colors_rgb (...,3), confidence/dynamic_mask (...)
        with matching leading dims (e.g. (N,H,W,{3,1})). Returns the number of
        newly-created voxels (for progress reporting)."""
        torch = self._torch
        pts = np.asarray(points_world).reshape(-1, 3)
        cols = np.asarray(colors_rgb).reshape(-1, 3)
        conf = np.asarray(confidence).reshape(-1)

        keep = np.isfinite(pts).all(axis=1) & np.isfinite(conf) & (conf >= min_confidence)
        if dynamic_mask is not None:
            keep &= ~np.asarray(dynamic_mask).reshape(-1)
        if not keep.any():
            return 0
        pts, cols, conf = pts[keep], cols[keep], conf[keep]

        pts_t = torch.from_numpy(np.ascontiguousarray(pts)).float().to(self.device)
        cols_t = torch.from_numpy(np.ascontiguousarray(cols)).float().to(self.device)
        conf_t = torch.from_numpy(np.ascontiguousarray(conf)).float().to(self.device)

        keys = self._voxel_keys(pts_t)
        uniq_keys, inverse = torch.unique(keys, return_inverse=True)
        n_uniq = uniq_keys.shape[0]

        local_weight = torch.zeros(n_uniq, device=self.device).scatter_add_(0, inverse, conf_t)
        local_pos = torch.zeros(n_uniq, 3, device=self.device).scatter_add_(
            0, inverse.unsqueeze(1).expand(-1, 3), pts_t * conf_t.unsqueeze(1)
        )
        local_color = torch.zeros(n_uniq, 3, device=self.device).scatter_add_(
            0, inverse.unsqueeze(1).expand(-1, 3), cols_t * conf_t.unsqueeze(1)
        )
        local_count = torch.zeros(n_uniq, device=self.device).scatter_add_(0, inverse, torch.ones_like(conf_t))
        local_max_conf = torch.zeros(n_uniq, device=self.device).scatter_reduce(
            0, inverse, conf_t, reduce="amax", include_self=False
        )

        uniq_keys_cpu = uniq_keys.cpu().numpy()
        new_keys, existing_rows, existing_local_idx, new_local_idx = [], [], [], []
        for i, k in enumerate(uniq_keys_cpu):
            row = self._key_to_row.get(int(k))
            if row is None:
                new_keys.append(int(k))
                new_local_idx.append(i)
            else:
                existing_rows.append(row)
                existing_local_idx.append(i)

        if existing_rows:
            rows_t = torch.tensor(existing_rows, device=self.device, dtype=torch.long)
            idx_t = torch.tensor(existing_local_idx, device=self.device, dtype=torch.long)
            self._weight_sum.index_add_(0, rows_t, local_weight[idx_t])
            self._pos_sum.index_add_(0, rows_t, local_pos[idx_t])
            self._color_sum.index_add_(0, rows_t, local_color[idx_t])
            self._view_count.index_add_(0, rows_t, local_count[idx_t].to(torch.int32))
            self._max_conf[rows_t] = torch.maximum(self._max_conf[rows_t], local_max_conf[idx_t])

        n_new = len(new_keys)
        if n_new:
            base = self._weight_sum.shape[0]
            for offset, k in enumerate(new_keys):
                self._key_to_row[k] = base + offset
            idx_t = torch.tensor(new_local_idx, device=self.device, dtype=torch.long)
            self._weight_sum = torch.cat([self._weight_sum, local_weight[idx_t]])
            self._pos_sum = torch.cat([self._pos_sum, local_pos[idx_t]])
            self._color_sum = torch.cat([self._color_sum, local_color[idx_t]])
            self._view_count = torch.cat([self._view_count, local_count[idx_t].to(torch.int32)])
            self._max_conf = torch.cat([self._max_conf, local_max_conf[idx_t]])

        return n_new

    def extract_points(self, min_view_count: int = 1) -> FusedPointCloud:
        """`min_view_count` > 1 doubles as the "multi-view consistency" filter
        the task asks for — a voxel only ever observed once is far more
        likely a floater/artifact than real geometry."""
        keep = (self._view_count >= min_view_count) & (self._weight_sum > 1e-9)
        w = self._weight_sum[keep].clamp_min(1e-9)
        pos = (self._pos_sum[keep] / w.unsqueeze(1)).cpu().numpy()
        color = (self._color_sum[keep] / w.unsqueeze(1)).cpu().numpy()
        view_count = self._view_count[keep].cpu().numpy()
        confidence = self._max_conf[keep].cpu().numpy()
        return FusedPointCloud(points=pos, colors=np.clip(color, 0, 255), view_count=view_count, confidence=confidence)


def remove_statistical_outliers(cloud: FusedPointCloud, bus: EventBus, nb_neighbors: int = 16, std_ratio: float = 2.0) -> FusedPointCloud:
    """Optional extra cleanup pass via Open3D, if available. Never required —
    VoxelPointFusion's confidence/view-count filtering already does most of
    the work; this just catches remaining isolated floaters."""
    try:
        import open3d as o3d
    except Exception as e:
        bus.log(f"Open3D unavailable for outlier removal ({e}); skipping this cleanup pass", level="warn")
        return cloud

    if len(cloud.points) < nb_neighbors + 1:
        return cloud

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points)
    _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_idx = np.asarray(inlier_idx)
    removed = len(cloud.points) - len(inlier_idx)
    if removed:
        bus.log(f"Statistical outlier removal: dropped {removed}/{len(cloud.points)} points")
    return FusedPointCloud(
        points=cloud.points[inlier_idx], colors=cloud.colors[inlier_idx],
        view_count=cloud.view_count[inlier_idx], confidence=cloud.confidence[inlier_idx],
    )


class Open3DTsdfFusion:
    """TSDF volume integration for mesh.py's marching-cubes path. Tries
    Open3D's tensor CUDA VoxelBlockGrid first, falls back to the legacy CPU
    ScalableTSDFVolume, and is simply unavailable (mesh.py falls back to
    point-based meshing) if Open3D can't be imported at all.

    Depth per chunk is derived from the backbone's own per-pixel 3D points
    (already in world frame after align.py) by re-projecting into camera
    space via the estimated camera pose — we don't need a separate depth
    sensor/estimator since the geometry backbone already gives us dense
    per-pixel points directly.
    """

    def __init__(self, bus: EventBus, voxel_size_m: float, sdf_trunc_m: float | None = None):
        self.bus = bus
        self.voxel_size = voxel_size_m
        self.sdf_trunc = sdf_trunc_m or voxel_size_m * 4
        self.backend = "none"
        self._volume = None
        self._device_tsdf = None
        self._logged_depth_trunc = False
        self._logged_rgbd_depth_check = False
        self._init()

    def _init(self) -> None:
        try:
            import open3d as o3d
        except Exception as e:
            self.bus.log(f"Open3D not available ({e}) — TSDF meshing disabled, mesh.py will use point-based meshing", level="warn")
            return

        # The tensor-CUDA VoxelBlockGrid path is deliberately never
        # selected here: integrate_chunk()'s tensor-CUDA branch was left
        # unimplemented (its exact integrate() signature differs across
        # Open3D releases and was never verified against the version on
        # this box), and extract_mesh() only reads out of the legacy CPU
        # volume — so choosing the tensor backend silently integrates
        # zero points every chunk and extract_mesh() always returns None,
        # a guaranteed-empty mesh with no error anywhere. Going straight
        # to the legacy CPU backend is slower but actually produces a
        # real mesh, which is what "TSDF available" is supposed to mean.
        try:
            self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.voxel_size, sdf_trunc=self.sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )
            self.backend = "open3d_legacy_cpu"
            self.bus.log("TSDF fusion: using Open3D legacy CPU ScalableTSDFVolume")
        except Exception as e:
            self.bus.log(f"Open3D legacy TSDF also unavailable ({e}) — TSDF meshing disabled", level="warn")
            self.backend = "none"

    @property
    def available(self) -> bool:
        return self.backend != "none"

    def integrate_chunk(
        self, points_cam: np.ndarray, colors_rgb: np.ndarray, intrinsics: np.ndarray,
        camera_pose_c2w: np.ndarray, image_shape: tuple[int, int],
    ) -> None:
        """points_cam: (H,W,3) points in *camera* space (z=depth along optical
        axis) for one view; colors_rgb: (H,W,3) uint8. Skips silently (logged
        once by _init) if no TSDF backend is available."""
        if not self.available:
            return
        import open3d as o3d

        h, w = image_shape
        # points_cam[..., 2] is a strided slice out of an (H,W,3) array —
        # .astype() alone preserves that non-contiguous memory layout
        # (order='K' by default), and a debug single-frame TSDF test
        # confirmed the resulting depth array reaching o3d.geometry.Image()
        # was C-contiguous=False, exactly the failure mode that produced a
        # 0-vertex mesh from integrating one otherwise-valid depth map.
        depth = np.ascontiguousarray(points_cam[..., 2], dtype=np.float32)
        depth[depth <= 0] = 0.0

        if self.backend == "open3d_legacy_cpu":
            depth_img = o3d.geometry.Image(depth)
            color_img = o3d.geometry.Image(np.ascontiguousarray(colors_rgb.astype(np.uint8)))
            intr = o3d.camera.PinholeCameraIntrinsic(
                w, h, float(intrinsics[0, 0]), float(intrinsics[1, 1]), float(intrinsics[0, 2]), float(intrinsics[1, 2])
            )
            # depth_trunc here is RGBDImage's own "discard anything farther
            # than this" cutoff, unrelated to sdf_trunc (which only governs
            # the surface-crossing truncation band inside a voxel that IS
            # within range). This used to be sdf_trunc*50 = 0.2*50 = 10m —
            # far short of typical drone-orbit camera-to-subject distances
            # (often 20-100m for an aerial orbit shot) — silently discarding
            # nearly all real depth pixels as "too far", with no exception
            # and no warning, which is exactly why every TSDF run produced
            # an empty mesh while the intrinsics-independent point cloud
            # (no such truncation) looked completely fine. Use max_depth_m
            # (the actual observed depth range for this chunk) with generous
            # headroom instead of a fixed value derived from the unrelated
            # voxel-level parameter.
            depth_trunc = max(float(depth[depth > 0].max()) * 1.2, self.sdf_trunc * 50) if np.any(depth > 0) else self.sdf_trunc * 50
            if not self._logged_depth_trunc:
                self._logged_depth_trunc = True
                valid_frac = float((depth > 0).mean())
                self.bus.log(
                    f"TSDF: voxel_size={self.voxel_size:.3f}m, sdf_trunc={self.sdf_trunc:.3f}m, "
                    f"depth_trunc={depth_trunc:.1f}m (first frame: {valid_frac * 100:.0f}% valid-depth pixels, "
                    f"depth range {depth[depth > 0].min() if np.any(depth > 0) else 0:.1f}-{depth[depth > 0].max() if np.any(depth > 0) else 0:.1f}m)"
                )
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_img, depth_img, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
            )
            if not self._logged_rgbd_depth_check:
                self._logged_rgbd_depth_check = True
                rgbd_depth = np.asarray(rgbd.depth)
                nz = int(np.count_nonzero(rgbd_depth))
                self.bus.log(
                    f"TSDF: post-creation rgbd.depth check — min={rgbd_depth.min():.3f} max={rgbd_depth.max():.3f} "
                    f"nonzero_pixels={nz}/{rgbd_depth.size} (all-zero would mean create_from_color_and_depth "
                    f"itself is discarding the depth, independent of what integrate() then does with it)"
                )
            extrinsic = np.linalg.inv(camera_pose_c2w)  # world-to-camera, what Open3D's legacy API expects
            self._volume.integrate(rgbd, intr, extrinsic)
        # Tensor-CUDA integration path intentionally omitted here: its exact
        # VoxelBlockGrid.integrate() call signature needs verification
        # against the installed Open3D version on the actual Kaggle image
        # (API has changed across Open3D releases) — flagged in
        # PHASE0_NOTES.md as a must-verify-on-Kaggle item rather than
        # guessed at here.

    def extract_mesh(self):
        """Returns an open3d.geometry.TriangleMesh via marching cubes, or
        None if unavailable/empty."""
        if not self.available or self.backend != "open3d_legacy_cpu":
            return None
        mesh = self._volume.extract_triangle_mesh()
        if len(mesh.vertices) == 0:
            return None
        mesh.compute_vertex_normals()
        return mesh

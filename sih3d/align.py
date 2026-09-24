"""Chunk-to-world alignment: gravity-fixed 4-DoF (GPS) + Sim(3)-between-chunks (no-GPS).

Task requirement (critical): a single-pass flight is a near-straight line, so plain
Sim(3) registration to GPS cannot constrain roll around the flight axis — the GPS
track alone gives no leverage on that rotation. So when GPS is available we NEVER fit
a full Sim(3) transform straight to GPS. Instead:

  1. Fix gravity first (IMU/gimbal attitude when available, else RANSAC on the
     chunk's own points for the dominant ground-plane normal, cross-checked against
     the GPS altitude trend).
  2. With roll/pitch now fixed by gravity, solve only the remaining 4 degrees of
     freedom: scale + yaw (rotation about the now-known vertical axis) + translation.
     This mirrors GeoFF3D's `scale_yaw_translation` alignment mode conceptually, but
     is our own implementation (SLRF's version is architecturally tied to the GeoFF3D
     model we aren't running — see PHASE0_NOTES.md §3).
  3. A lightweight pose-graph refinement then ties adjacent chunks together via their
     overlapping cameras, blended with robust (Huber) GPS unary factors, weighting
     RTK-sourced GPS much higher than consumer GPS when the caller says so.

No-GPS mode: when no telemetry exists at all (backbone auto-selected to C/RGB in
backbone.py), 4-DoF-to-GPS alignment is impossible by definition. We fall back to a
full Sim(3) fit between each chunk and the previous one, using their overlapping
camera centers as point correspondences (our chunking always overlaps consecutive
chunks). This produces a single continuous, metrically-plausible-but-unanchored
model. Every caller of this fallback path MUST propagate `georeferenced=False`
downstream (report.py/export.py/viewer.html all check `ChunkAlignment.georeferenced`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import EventBus


@dataclass
class GravityEstimate:
    up_local: np.ndarray  # (3,) unit vector: chunk-local "up", in the backbone's own gauge
    source: str            # "imu" | "gimbal" | "ransac_ground_plane"
    confidence: float = 1.0


@dataclass
class ChunkAlignment:
    scale: float
    R: np.ndarray               # (3,3) rotation, chunk-local -> target frame
    t: np.ndarray                # (3,) translation, chunk-local -> target frame
    mode: str                    # "4dof_gps" | "sim3_chunk_overlap"
    georeferenced: bool
    rmse_m: float | None = None  # residual RMSE of the fit, meters
    gravity_source: str | None = None
    notes: list[str] = field(default_factory=list)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """points: (...,3) chunk-local -> target frame."""
        shape = points.shape
        flat = points.reshape(-1, 3)
        out = (self.scale * (self.R @ flat.T)).T + self.t
        return out.reshape(shape)


# ---------------------------------------------------------------------------
# Gravity estimation
# ---------------------------------------------------------------------------

def estimate_gravity_imu(attitude_deg: tuple[float, float, float] | None) -> GravityEstimate | None:
    """attitude_deg = (yaw, pitch, roll) of the gimbal/IMU for chunk's reference
    camera (camera 0), degrees. Returns world-up expressed in that camera's local
    frame by inverting the known world->camera attitude rotation (ZYX gimbal
    convention: yaw about world z, then pitch about the new y, then roll about the
    new x)."""
    if attitude_deg is None or any(a is None for a in attitude_deg):
        return None
    yaw, pitch, roll = (np.radians(a) for a in attitude_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    R_world_to_cam = Rx @ Ry @ Rz
    up_in_cam = R_world_to_cam @ np.array([0.0, 0.0, 1.0])
    norm = np.linalg.norm(up_in_cam)
    if norm < 1e-9:
        return None
    return GravityEstimate(up_local=up_in_cam / norm, source="imu")


def estimate_gravity_ransac(
    points: np.ndarray,
    n_iters: int = 300,
    dist_thresh: float = 0.05,
    min_inlier_frac: float = 0.08,
    rng: np.random.Generator | None = None,
) -> GravityEstimate | None:
    """RANSAC-fit the dominant plane in `points` (N,3), treat its normal as
    chunk-local "up". No sign disambiguation is attempted here (see
    `disambiguate_gravity_sign` — needs the GPS altitude trend, not available
    to this pure-geometry function)."""
    pts = np.asarray(points).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    n = len(pts)
    if n < 50:
        return None
    rng = rng or np.random.default_rng(0)
    best_inliers = -1
    best_normal = None
    for _ in range(n_iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -np.dot(normal, p0)
        dist = np.abs(pts @ normal + d)
        inliers = int((dist < dist_thresh).sum())
        if inliers > best_inliers:
            best_inliers, best_normal = inliers, normal
    if best_normal is None or best_inliers < min_inlier_frac * n:
        return None
    return GravityEstimate(up_local=best_normal, source="ransac_ground_plane", confidence=best_inliers / n)


def disambiguate_gravity_sign(
    gravity: GravityEstimate, cam_centers_local: np.ndarray, gps_altitude_trend: np.ndarray,
) -> GravityEstimate:
    """Flip `up_local` if needed so it agrees with the GPS altitude trend: cameras
    with a higher projection onto `up_local` must correspond to higher GPS altitude.
    If the correlation is weak/negative even after the best sign choice, halve
    confidence and note it (caller should consider falling back to RANSAC-only /
    no gravity fix at all for this chunk)."""
    if len(cam_centers_local) < 2 or len(gps_altitude_trend) != len(cam_centers_local):
        return gravity
    proj = cam_centers_local @ gravity.up_local
    corr = np.corrcoef(proj, gps_altitude_trend)[0, 1]
    if np.isnan(corr):
        return gravity
    if corr < 0:
        gravity = GravityEstimate(up_local=-gravity.up_local, source=gravity.source, confidence=gravity.confidence)
        corr = -corr
    if corr < 0.3:
        gravity.confidence *= 0.5
    return gravity


def gravity_rotation_matrix(up_local: np.ndarray, world_up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Rotation R such that R @ up_local == world_up (Rodrigues' formula)."""
    a = up_local / np.linalg.norm(up_local)
    b = world_up / np.linalg.norm(world_up)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-9:
        return np.eye(3) if c > 0 else _rotation_180_about_any_perpendicular(a)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def _rotation_180_about_any_perpendicular(a: np.ndarray) -> np.ndarray:
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis = np.cross(a, helper)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    return np.array([
        [2 * x * x - 1, 2 * x * y, 2 * x * z],
        [2 * x * y, 2 * y * y - 1, 2 * y * z],
        [2 * x * z, 2 * y * z, 2 * z * z - 1],
    ])


# ---------------------------------------------------------------------------
# 4-DoF (scale + yaw + translation) fit, gravity already fixed
# ---------------------------------------------------------------------------

def solve_4dof(
    src_grav: np.ndarray,   # (M,3) source points, already gravity-rotated (z ~ up)
    dst: np.ndarray,        # (M,3) target points (GPS ENU)
    weights: np.ndarray | None = None,
    huber_delta: float = 2.0,
    n_irls_iters: int = 5,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Solve scale s, yaw-only rotation R (about z), translation t minimizing
    weighted robust residuals ||s*R@src_grav[i] + t - dst[i]||. Rotation about z
    doesn't touch the z-component, so we jointly solve the xy similarity (Umeyama
    closed-form, scale + rotation) and z as a separate 1D scale+offset that must
    share the same scale — done via IRLS reweighting the xy Umeyama fit toward
    points whose z also agrees well, then closing z with the shared scale.

    Returns (scale, R (3x3), t (3,), rmse_m).
    """
    src_grav = np.asarray(src_grav, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    m = len(src_grav)
    if m < 2:
        raise ValueError("solve_4dof needs at least 2 correspondences")
    w = np.ones(m) if weights is None else np.asarray(weights, dtype=np.float64).copy()
    w = w / w.sum()

    R = np.eye(3)
    scale = 1.0
    t = np.zeros(3)

    for _ in range(n_irls_iters):
        src_c = (src_grav * w[:, None]).sum(axis=0)
        dst_c = (dst * w[:, None]).sum(axis=0)
        src_xy = src_grav[:, :2] - src_c[:2]
        dst_xy = dst[:, :2] - dst_c[:2]

        H = (src_xy * w[:, None]).T @ dst_xy
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, d])
        R2 = Vt.T @ D @ U.T  # 2x2 rotation, src_xy -> dst_xy

        var_src_xy = (w * (src_xy ** 2).sum(axis=1)).sum()
        scale = float((S[0] + d * S[1]) / var_src_xy) if var_src_xy > 1e-12 else 1.0
        scale = max(scale, 1e-6)

        R = np.eye(3)
        R[:2, :2] = R2

        t_xy = dst_c[:2] - scale * (R2 @ src_c[:2])
        t_z = dst_c[2] - scale * src_c[2]  # R about z leaves z unchanged
        t = np.array([t_xy[0], t_xy[1], t_z])

        pred = scale * (R @ src_grav.T).T + t
        resid = np.linalg.norm(pred - dst, axis=1)
        w = np.where(resid <= huber_delta, 1.0, huber_delta / np.maximum(resid, 1e-9))
        w = w / w.sum()

    pred = scale * (R @ src_grav.T).T + t
    rmse = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    return scale, R, t, rmse


def align_chunk_4dof(
    cam_centers_local: np.ndarray,
    cam_centers_gps_enu: np.ndarray,
    gravity: GravityEstimate,
    gps_weights: np.ndarray | None = None,
) -> ChunkAlignment:
    R_grav = gravity_rotation_matrix(gravity.up_local)
    src_grav = (R_grav @ np.asarray(cam_centers_local).T).T
    scale, R_yaw, t, rmse = solve_4dof(src_grav, np.asarray(cam_centers_gps_enu), weights=gps_weights)
    R_total = R_yaw @ R_grav
    return ChunkAlignment(
        scale=scale, R=R_total, t=t, mode="4dof_gps", georeferenced=True,
        rmse_m=rmse, gravity_source=gravity.source,
        notes=[f"gravity confidence={gravity.confidence:.2f}"] if gravity.confidence < 0.9 else [],
    )


# ---------------------------------------------------------------------------
# No-GPS fallback: Sim(3) between overlapping chunks
# ---------------------------------------------------------------------------

def solve_sim3(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Full Umeyama Sim(3): scale, 3x3 rotation, translation minimizing
    weighted ||s*R@src[i] + t - dst[i]||^2. Used only when GPS anchoring is
    unavailable (no-GPS mode) — chunk k is registered to chunk k-1 via their
    shared overlapping camera centers, not to any absolute frame."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    m = len(src)
    if m < 3:
        raise ValueError("solve_sim3 needs at least 3 correspondences")
    w = np.ones(m) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    src_c = (src * w[:, None]).sum(axis=0)
    dst_c = (dst * w[:, None]).sum(axis=0)
    src0 = src - src_c
    dst0 = dst - dst_c

    H = (src0 * w[:, None]).T @ dst0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    var_src = (w * (src0 ** 2).sum(axis=1)).sum()
    scale = float((S * np.array([1.0, 1.0, d])).sum() / var_src) if var_src > 1e-12 else 1.0
    scale = max(scale, 1e-6)

    t = dst_c - scale * (R @ src_c)
    pred = scale * (R @ src.T).T + t
    rmse = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    return scale, R, t, rmse


def align_chunk_to_previous(
    overlap_cam_centers_local: np.ndarray,      # this chunk's local coords, for cameras shared with the previous chunk
    overlap_cam_centers_prev_world: np.ndarray,  # those same cameras' already-resolved world coords
) -> ChunkAlignment:
    scale, R, t, rmse = solve_sim3(overlap_cam_centers_local, overlap_cam_centers_prev_world)
    return ChunkAlignment(
        scale=scale, R=R, t=t, mode="sim3_chunk_overlap", georeferenced=False, rmse_m=rmse,
        notes=["no GPS: aligned to previous chunk's overlap only, not to an absolute frame"],
    )


def first_chunk_identity(georeferenced: bool) -> ChunkAlignment:
    """The very first chunk (no previous chunk to register against, and in
    no-GPS mode no GPS to register against either) defines the world frame
    as its own local frame."""
    return ChunkAlignment(
        scale=1.0, R=np.eye(3), t=np.zeros(3), mode="identity_seed",
        georeferenced=georeferenced, rmse_m=0.0,
        notes=[] if georeferenced else ["no GPS: world frame is this chunk's own local frame, not georeferenced"],
    )


# ---------------------------------------------------------------------------
# Lightweight global pose-graph refinement across chunks
# ---------------------------------------------------------------------------

@dataclass
class OverlapConstraint:
    chunk_a: int
    chunk_b: int
    cam_centers_a_local: np.ndarray  # (K,3), chunk_a-local coords of the shared cameras
    cam_centers_b_local: np.ndarray  # (K,3), chunk_b-local coords of the same cameras


@dataclass
class GpsFactor:
    chunk: int
    cam_center_local: np.ndarray  # (3,)
    gps_enu: np.ndarray            # (3,)
    weight: float = 1.0            # higher for RTK/PPK


def refine_pose_graph(
    initial: list[ChunkAlignment],
    overlaps: list[OverlapConstraint],
    gps_factors: list[GpsFactor],
    bus: EventBus,
    huber_delta: float = 2.0,
    max_nfev: int = 200,
) -> list[ChunkAlignment]:
    """Jointly refine each chunk's (scale, yaw, translation) to minimize robust
    residuals from both overlap constraints (chunk-to-chunk agreement on shared
    cameras) and GPS unary factors (chunk-to-world agreement), via
    scipy.optimize.least_squares with a Huber loss. This is a lightweight
    substitute for a full SE3 pose-graph optimizer (g2o/GTSAM are not installed
    and are a real Kaggle install risk we're avoiding) — parameterized as
    (log_scale, yaw, tx, ty, tz) per chunk, i.e. rotation is yaw-only. Chunks
    whose original alignment mode was "sim3_chunk_overlap" (no-GPS) keep a
    yaw-only reparameterization too for simplicity; their absolute rotation is
    already free (no GPS to anchor it), so this only smooths relative
    consistency between chunks, which is the only thing that matters when
    nothing is georeferenced anyway.
    """
    if not initial:
        return initial
    if len(overlaps) == 0 and len(gps_factors) == 0:
        return initial

    from scipy.optimize import least_squares

    n = len(initial)

    def r_of(R: np.ndarray) -> float:
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def R_of(yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    x0 = np.zeros(5 * n)
    for i, ca in enumerate(initial):
        x0[5 * i + 0] = np.log(max(ca.scale, 1e-6))
        x0[5 * i + 1] = r_of(ca.R)
        x0[5 * i + 2:5 * i + 5] = ca.t

    def unpack(x, i):
        s = np.exp(x[5 * i + 0])
        yaw = x[5 * i + 1]
        t = x[5 * i + 2:5 * i + 5]
        return s, R_of(yaw), t

    def residuals(x):
        res = []
        for oc in overlaps:
            sa, Ra, ta = unpack(x, oc.chunk_a)
            sb, Rb, tb = unpack(x, oc.chunk_b)
            pa = sa * (Ra @ oc.cam_centers_a_local.T).T + ta
            pb = sb * (Rb @ oc.cam_centers_b_local.T).T + tb
            res.append((pa - pb).ravel())
        for gf in gps_factors:
            s, R, t = unpack(x, gf.chunk)
            pred = s * (R @ gf.cam_center_local) + t
            res.append(np.sqrt(gf.weight) * (pred - gf.gps_enu))
        return np.concatenate(res) if res else np.zeros(1)

    result = least_squares(residuals, x0, loss="huber", f_scale=huber_delta, max_nfev=max_nfev)
    bus.log(
        f"Pose graph refinement: {len(overlaps)} overlap constraints, {len(gps_factors)} GPS factors, "
        f"cost {result.cost:.3f}, {'converged' if result.success else 'did not fully converge'}"
    )

    # Recompute each chunk's own RMSE against its GPS factors using the
    # REFINED transform — the pre-refinement rmse_m on `initial` is not a
    # valid "after" value (it was carried straight through here in an
    # earlier version of this function, which is wrong: the whole point of
    # refinement is that per-chunk residuals should change).
    gps_by_chunk: dict[int, list[GpsFactor]] = {}
    for gf in gps_factors:
        gps_by_chunk.setdefault(gf.chunk, []).append(gf)

    refined = []
    for i, ca in enumerate(initial):
        s, R, t = unpack(result.x, i)
        chunk_gps = gps_by_chunk.get(i, [])
        if chunk_gps:
            preds = np.array([s * (R @ gf.cam_center_local) + t for gf in chunk_gps])
            targets = np.array([gf.gps_enu for gf in chunk_gps])
            rmse_after = float(np.sqrt(np.mean(np.sum((preds - targets) ** 2, axis=1))))
        else:
            rmse_after = ca.rmse_m  # no GPS factors for this chunk — nothing to recompute against
        refined.append(ChunkAlignment(
            scale=s, R=R, t=t, mode=ca.mode, georeferenced=ca.georeferenced,
            rmse_m=rmse_after, gravity_source=ca.gravity_source, notes=ca.notes,
        ))
    return refined

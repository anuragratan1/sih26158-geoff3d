"""Mesh generation: vertex-colored mesh first (always), then best-effort
texture baking from full-resolution keyframes.

Vertex-colored mesh: from the TSDF volume via marching cubes if
`fusion.Open3DTsdfFusion` produced one (denser/smoother), else directly from
the fused point cloud via Open3D Poisson surface reconstruction. If Open3D
itself isn't available at all, meshing is skipped with a clearly logged
reason — the pipeline still ships the point cloud (LAS/PLY), matching the
robustness rule "log clearly if unavailable; never crash".

Texture baking (bake_vertex_colors_from_keyframes): for each mesh vertex,
picks the "best view" among the full-resolution keyframes (most
fronto-parallel, i.e. maximizing -dot(vertex_normal, view_direction),
among cameras where the vertex actually projects inside the image) and
sets that vertex's color to the real photo pixel sampled there. Fully
vectorized per keyframe (batched projection + scoring + color gather over
every vertex at once), not per vertex or per face — no UV atlas, no
xatlas dependency, no per-element Python loop. An earlier UV-atlas-based
version (xatlas.parametrize + a nested per-face-times-per-keyframe Python
loop) was replaced entirely: both of its slow parts were confirmed
classical-CPU bottlenecks (xatlas's own documented serial chart
segmentation, and unvectorized scalar Python math), not an inherent cost
of photo texturing. Skipped (logged) if no keyframe sees any vertex.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .events import EventBus, EventType
from .fusion import FusedPointCloud, Open3DTsdfFusion


@dataclass
class MeshResult:
    mesh: "object | None"       # open3d.geometry.TriangleMesh, vertex-colored
    method: str                 # "tsdf_marching_cubes" | "poisson" | "none"
    n_vertices: int = 0
    n_faces: int = 0


@dataclass
class TextureBakeResult:
    textured: bool
    texture_rgb: np.ndarray | None = None   # (H,W,3) uint8
    uv: np.ndarray | None = None             # (n_faces*3, 2) float32, per-face-corner UVs
    skipped_reason: str | None = None
    timing_s: float = 0.0


def build_vertex_colored_mesh(
    cloud: FusedPointCloud, tsdf: Open3DTsdfFusion | None, bus: EventBus,
    poisson_depth: int = 10,
) -> MeshResult:
    if tsdf is not None and tsdf.available:
        mesh = tsdf.extract_mesh()
        if mesh is not None:
            bus.log(f"Mesh: extracted via TSDF marching cubes ({len(mesh.vertices)} vertices)")
            return MeshResult(mesh=mesh, method="tsdf_marching_cubes", n_vertices=len(mesh.vertices), n_faces=len(mesh.triangles))
        bus.log("TSDF available but produced an empty mesh; falling back to point-based Poisson meshing", level="warn")

    try:
        import open3d as o3d
    except Exception as e:
        bus.log(f"Open3D not available ({e}) — mesh generation skipped, point cloud outputs are unaffected", level="warn")
        return MeshResult(mesh=None, method="none")

    if len(cloud.points) < 10:
        bus.log("Too few fused points for meshing — mesh generation skipped", level="warn")
        return MeshResult(mesh=None, method="none")

    points = cloud.points
    colors01 = np.clip(cloud.colors, 0, 255) / 255.0

    # Cap the point count BEFORE normal estimation — cost scales with point
    # count, so a bigger timeout alone just means a longer guaranteed wait,
    # not a more reliable one. Was 200k, which real output revealed was
    # throwing away ~95% of the fused cloud (a 4.4M-point pointcloud.ply
    # from one real run, vs. only 200k of it ever reaching Poisson) — that
    # gap is real and worth spending more time on now that quality is the
    # priority. 800k is a 4x increase, paired with a higher Poisson depth
    # below so the extra density actually translates into a higher-
    # resolution mesh instead of being wasted on an unchanged octree.
    _MESH_POINT_CAP = 800_000
    if len(points) > _MESH_POINT_CAP:
        idx = np.random.default_rng(0).choice(len(points), size=_MESH_POINT_CAP, replace=False)
        points, colors01 = points[idx], colors01[idx]
        bus.log(f"Meshing: downsampled {len(cloud.points):,} -> {_MESH_POINT_CAP:,} points for normal estimation/reconstruction speed")

    # orient_normals_consistent_tangent_plane is a minimum-spanning-tree
    # propagation over the point cloud's KNN graph — real-world observed
    # cost on a several-hundred-thousand-point cloud: minutes, with zero
    # progress reporting since it's one opaque native call. Isolated + timed
    # out like every other step here; on timeout, falls back to plain
    # per-point estimate_normals with NO consistent orientation (faster, no
    # MST) — which is ALSO isolated+timed-out, not called inline: a real run
    # showed the naive fallback itself stall the exact same way (large
    # clouds make even plain KDTree normal estimation slow enough to trip
    # the watchdog). If both attempts fail, normals stays None and meshing
    # skips straight to the no-normals-needed Delaunay fallback below rather
    # than feeding Poisson/ball-pivoting normals that were never computed.
    # Timeouts raised alongside the point cap above (4x points -> allow
    # meaningfully more time, not the same budget for 4x the work).
    stage = "mesh_textured_model"
    normals = _run_isolated_meshing(
        _normals_worker, (points, colors01), bus, "normal estimation",
        timeout_s=100.0, stage=stage, progress_range=(0.0, 0.15),
    )
    if normals is None:
        bus.log("Falling back to fast normal estimation without consistent orientation", level="warn")
        normals = _run_isolated_meshing(
            _normals_fast_worker, (points, colors01), bus, "normal estimation (fast)",
            timeout_s=50.0, stage=stage, progress_range=(0.15, 0.25),
        )

    # Open3D's native Poisson/ball-pivoting solvers can hard-abort the whole
    # process on degenerate/pathological point distributions (observed
    # directly: a synthetic test point cloud crashed the interpreter with
    # "libc++abi: terminating" from deep inside PoissonRecon's C++ — no
    # Python try/except can catch a native abort). Both are run in an
    # isolated subprocess so a crash there kills only that subprocess; the
    # pipeline sees it as an ordinary failure and falls back normally.
    method = "poisson"
    vertices = triangles = vcolors = None
    if normals is not None:
        poisson_out = _run_isolated_meshing(
            _poisson_worker, (points, colors01, normals, poisson_depth), bus, "Poisson",
            timeout_s=90.0, stage=stage, progress_range=(0.25, 0.5),  # was the 45s default — not enough for 4x the points + depth 9->10
        )
        if poisson_out is not None:
            vertices, triangles, vcolors = poisson_out
        else:
            bus.log("Poisson reconstruction failed/crashed; trying ball-pivoting as a second fallback", level="warn")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = float(np.mean(distances)) if len(distances) else 0.05
            radii = [avg_dist * r for r in (1.5, 2.0, 3.0)]
            bp_out = _run_isolated_meshing(
                _ball_pivot_worker, (points, colors01, normals, radii), bus, "ball-pivoting",
                stage=stage, progress_range=(0.25, 0.5),
            )
            if bp_out is not None:
                vertices, triangles, vcolors = bp_out
                method = "ball_pivoting"

    if vertices is None:
        bus.log(
            "Trying 2.5D Delaunay triangulation (a ground-projected heightfield mesh, needs no normals — "
            "often the more robust choice for open, nadir-view aerial point clouds anyway, since Poisson/"
            "ball-pivoting assume more closed/volumetric input)",
            level="warn",
        )
        dt_out = _run_isolated_meshing(
            _delaunay_2p5d_worker, (points, colors01), bus, "Delaunay 2.5D",
            timeout_s=30.0, stage=stage, progress_range=(0.5, 0.65),
        )
        if dt_out is None:
            bus.log("2.5D Delaunay also failed — mesh generation skipped", level="warn")
            return MeshResult(mesh=None, method="none")
        vertices, triangles, vcolors = dt_out
        method = "delaunay_2p5d"

    if len(vertices) == 0:
        bus.log("Meshing produced zero vertices — mesh generation skipped", level="warn")
        return MeshResult(mesh=None, method="none")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    if vcolors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(vcolors)
    mesh.compute_vertex_normals()
    bus.log(f"Mesh: built via {method} from point cloud ({len(mesh.vertices)} vertices, {len(mesh.triangles)} faces)")
    return MeshResult(mesh=mesh, method=method, n_vertices=len(mesh.vertices), n_faces=len(mesh.triangles))


def _normals_worker(points, colors01, result_queue) -> None:
    try:
        import open3d as _o3d

        pcd = _o3d.geometry.PointCloud()
        pcd.points = _o3d.utility.Vector3dVector(points)
        pcd.colors = _o3d.utility.Vector3dVector(colors01)
        pcd.estimate_normals(search_param=_o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(30)
        result_queue.put(np.asarray(pcd.normals))
    except Exception as e:
        result_queue.put(e)


def _normals_fast_worker(points, colors01, result_queue) -> None:
    """No orient_normals_consistent_tangent_plane (the slow MST step) —
    just per-point PCA normals. Still isolated+timed-out like every other
    step here: on a large enough cloud even plain estimate_normals' KDTree
    build/query can run long enough to trip the watchdog, and this was
    previously called inline with zero timeout at all — the exact bug
    being fixed (a real run showed a second "normal estimation timed out
    after 60s" from this fallback itself)."""
    try:
        import open3d as _o3d

        pcd = _o3d.geometry.PointCloud()
        pcd.points = _o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=_o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        result_queue.put(np.asarray(pcd.normals))
    except Exception as e:
        result_queue.put(e)


def _poisson_worker(points, colors, normals, depth, result_queue) -> None:
    import numpy as _np
    import open3d as _o3d

    try:
        pcd = _o3d.geometry.PointCloud()
        pcd.points = _o3d.utility.Vector3dVector(points)
        pcd.colors = _o3d.utility.Vector3dVector(colors)
        pcd.normals = _o3d.utility.Vector3dVector(normals)
        mesh, densities = _o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        densities = _np.asarray(densities)
        # Trim low-density (extrapolated/hallucinated) vertices at the
        # reconstruction's outer fringe. This is a BLUNT, global cut — it
        # removes the bottom X% by density everywhere, with no way to tell
        # "isolated hallucinated fringe" (spikes we want gone) apart from
        # "legitimate but locally-sparser real surface" (still real
        # geometry, just observed from fewer/more oblique viewpoints).
        # Raised to 12% earlier to fight spiky artifacts over water — that
        # was an overcorrection, confirmed by a real run: it punched holes
        # through large areas of otherwise-good surface ("swiss cheese"),
        # not just the outer fringe, because 12% of ANY real mesh's
        # vertices are locally-sparser-but-real, not hallucinated. Back
        # down to 2% (close to Open3D's own reference-tutorial value) —
        # the connected-component pass below is the actual right tool for
        # spike/hair artifacts specifically (removes genuinely isolated
        # small fragments, not a density percentile blind to whether a
        # vertex is isolated debris or part of the main surface).
        keep = densities >= _np.quantile(densities, 0.02)
        mesh.remove_vertices_by_mask(~keep)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

        # Small disconnected fragments left over from the density trim
        # above (isolated spike/hair clusters, not part of the main
        # surface) — keep only clusters big enough to plausibly be real
        # structure, not reconstruction noise.
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = _np.asarray(triangle_clusters)
        cluster_n_triangles = _np.asarray(cluster_n_triangles)
        if len(cluster_n_triangles) > 1:
            min_cluster_triangles = max(50, int(0.005 * len(mesh.triangles)))
            remove_mask = cluster_n_triangles[triangle_clusters] < min_cluster_triangles
            mesh.remove_triangles_by_mask(remove_mask)
            mesh.remove_unreferenced_vertices()
        result_queue.put((
            _np.asarray(mesh.vertices), _np.asarray(mesh.triangles),
            _np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None,
        ))
    except Exception as e:
        result_queue.put(RuntimeError(f"{type(e).__name__}: {e}"))


def _ball_pivot_worker(points, colors, normals, radii, result_queue) -> None:
    import numpy as _np
    import open3d as _o3d

    try:
        pcd = _o3d.geometry.PointCloud()
        pcd.points = _o3d.utility.Vector3dVector(points)
        pcd.colors = _o3d.utility.Vector3dVector(colors)
        pcd.normals = _o3d.utility.Vector3dVector(normals)
        mesh = _o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, _o3d.utility.DoubleVector(radii))
        result_queue.put((
            _np.asarray(mesh.vertices), _np.asarray(mesh.triangles),
            _np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None,
        ))
    except Exception as e:
        result_queue.put(RuntimeError(f"{type(e).__name__}: {e}"))


def _delaunay_2p5d_worker(points, colors, result_queue) -> None:
    """Ground-projected 2.5D triangulation: Delaunay on the XY projection,
    Z carried through as height. A much more mature, robust algorithm than
    Poisson/ball-pivoting for this specific shape of input (scipy's Qhull
    binding essentially never crashes or hangs on a well-formed 2D point
    set), and arguably the more *appropriate* one for open, single-pass
    nadir aerial capture in the first place — it's the same footprint
    concept as the DSM raster (export.py's export_dsm_orthomosaic), just
    kept as an actual triangle mesh instead of a rasterized grid."""
    import numpy as _np
    from scipy.spatial import Delaunay as _Delaunay

    try:
        xy = points[:, :2]
        tri = _Delaunay(xy)
        simplices = tri.simplices

        p0, p1, p2 = points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]]
        edge_lens = _np.stack([
            _np.linalg.norm(p0[:, :2] - p1[:, :2], axis=1),
            _np.linalg.norm(p1[:, :2] - p2[:, :2], axis=1),
            _np.linalg.norm(p2[:, :2] - p0[:, :2], axis=1),
        ], axis=1)
        max_edge = edge_lens.max(axis=1)
        # Delaunay triangulates the full convex hull, including spurious
        # triangles that bridge across real gaps in an open point cloud
        # (unobserved areas). Drop the longest outlier edges.
        thresh = max(float(_np.percentile(max_edge, 95)) * 2.0, 1e-6)
        simplices = simplices[max_edge < thresh]

        result_queue.put((points, simplices, colors))
    except Exception as e:
        result_queue.put(RuntimeError(f"{type(e).__name__}: {e}"))


def _run_isolated_meshing(
    worker_fn, args: tuple, bus: EventBus, label: str, timeout_s: float = 45.0,
    stage: str | None = None, progress_range: tuple[float, float] = (0.0, 1.0),
):
    """Runs `worker_fn(*args, result_queue)` in a subprocess. Returns
    (vertices, triangles, colors) on success, None on any failure —
    including a native crash (nonzero/None exit code) or a timeout, both
    logged the same way as an ordinary caught exception.

    Drains the queue BEFORE calling proc.join() — calling join() first is a
    classic multiprocessing deadlock (documented in Python's own docs, and
    hit directly here): a mesh result is a few MB, comfortably over the OS
    pipe buffer, so Queue.put() in the child blocks on its background
    feeder thread until the parent reads, and the child can't fully exit
    (so join() never returns) until put() finishes. Poisson's occasional
    fast "crashed" report earlier masked this — it usually aborted before
    ever reaching put() — but ball-pivoting and even a plain scipy Delaunay
    call (independently confirmed to run in ~0.2s standalone) both hung
    until forcibly killed, because their results *did* reach put().

    The worker itself has no way to report partial progress (it's one
    opaque call into Open3D/scipy/xatlas), so while `stage` is given this
    emits a time-based heartbeat (elapsed/timeout mapped into
    `progress_range`) every ~2s instead of leaving the stage bar frozen at
    its starting value for the whole timeout window with no way to tell a
    slow-but-working run from a genuinely stuck one.
    """
    import multiprocessing as mp

    # "fork" copies the already-fully-loaded parent process (torch/open3d/
    # scipy already imported) via copy-on-write, essentially instant.
    # "spawn" re-imports everything from scratch in the child — observed
    # locally to cost 15-20s+ per attempt just on startup. Kaggle (Linux)
    # supports fork; only platforms without it (Windows) fall back to spawn.
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
    result_queue: "mp.Queue" = ctx.Queue()
    proc = ctx.Process(target=worker_fn, args=(*args, result_queue))
    proc.start()

    start = time.time()
    deadline = start + timeout_s
    range_lo, range_hi = progress_range
    last_heartbeat = 0.0
    result = _SENTINEL_NO_RESULT = object()
    while time.time() < deadline:
        try:
            result = result_queue.get(timeout=0.1)
            break
        except Exception:
            pass
        if not proc.is_alive():
            break
        now = time.time()
        if stage is not None and now - last_heartbeat >= 2.0:
            elapsed = now - start
            frac = range_lo + (range_hi - range_lo) * min(0.95, elapsed / timeout_s)
            bus.publish(
                EventType.STAGE_PROGRESS, stage=stage, frac=frac,
                rate_label=f"{label}: {elapsed:.0f}s (working, not stuck — times out at {timeout_s:.0f}s)",
            )
            last_heartbeat = now

    if result is _SENTINEL_NO_RESULT:
        try:
            result = result_queue.get_nowait()
        except Exception:
            result = _SENTINEL_NO_RESULT

    if proc.is_alive():
        proc.terminate()
    proc.join(5)

    if result is not _SENTINEL_NO_RESULT:
        if isinstance(result, Exception):
            bus.log(f"{label} failed: {result}", level="warn")
            return None
        return result

    if proc.exitcode not in (0, None) and proc.exitcode != -15:  # -15 = our own terminate()
        bus.log(f"{label} subprocess crashed (exit code {proc.exitcode}, likely a native library abort) — isolated, pipeline continues", level="warn")
    else:
        bus.log(f"{label} timed out after {timeout_s:.0f}s — treating as failed", level="warn")
    return None


def _project_points(points_world: np.ndarray, camera_pose_c2w: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pixel_xy (N,2), depth (N,)) — depth <= 0 means behind the camera."""
    w2c = np.linalg.inv(camera_pose_c2w)
    pts_h = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    depth = pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pix = (intrinsics @ pts_cam.T).T
        pix_xy = pix[:, :2] / pix[:, 2:3]
    return pix_xy, depth


def bake_vertex_colors_from_keyframes(mesh, keyframes: list[KeyframeForBaking], bus: EventBus) -> bool:
    """Direct per-vertex color baking from real photos — no UV atlas, no
    xatlas UV-unwrap, no per-face Python loop. Replaces the old bake_texture()
    as the default texturing path: profiling and direct measurement on a
    real run both pointed at the SAME kind of bug, not a fundamental "fast
    vs. good" tradeoff — xatlas's own documented bottleneck is serial
    per-face chart segmentation, and the old per-face loop here was ALSO a
    nested Python for-loop (faces x keyframes) doing scalar math one pair
    at a time. Neither of those slow things was the neural network or an
    inherent property of "photo-realistic texturing" — they were just
    classical CPU code that didn't scale, bolted on after an already-fast
    feed-forward reconstruction pass.

    This does the exact same "best fronto-parallel view wins" selection as
    before, but vectorized per KEYFRAME (a handful, ~20-60) instead of per
    VERTEX (hundreds of thousands): each keyframe does one batched
    projection of every vertex via _project_points (the same vectorized
    math the render-vs-ground-truth comparison already uses efficiently on
    hundreds of thousands of points), then one batched score comparison
    and one batched color gather — no per-element Python overhead at all.
    Runs on the FULL mesh, not a decimated approximation, since there's no
    chart-segmentation cost to control for anymore.

    Mutates `mesh.vertex_colors` in place and returns whether anything was
    baked; callers don't need a separate TextureBakeResult — the existing
    vertex-colored export path (export_mesh_glb/obj/ply's `else` branch)
    already picks up whatever's in vertex_colors, baked or not.
    """
    t0 = time.time()
    if mesh is None or len(mesh.vertices) == 0 or not keyframes:
        return False

    import open3d as o3d

    vertices = np.asarray(mesh.vertices)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    n = len(vertices)

    best_score = np.full(n, -1.0, dtype=np.float64)
    best_color = np.zeros((n, 3), dtype=np.float64)
    hit = np.zeros(n, dtype=bool)

    for kf in keyframes:
        h, w = kf.image_rgb.shape[:2]
        pix_xy, depth = _project_points(vertices, kf.camera_pose_c2w, kf.intrinsics)
        x, y = pix_xy[:, 0], pix_xy[:, 1]
        visible = (depth > 0) & (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
        if not visible.any():
            continue

        view_dir = vertices - kf.camera_pose_c2w[:3, 3]
        dist = np.linalg.norm(view_dir, axis=1)
        view_dir_n = view_dir / np.where(dist > 1e-6, dist, 1.0)[:, None]
        score = -np.sum(normals * view_dir_n, axis=1)  # most fronto-parallel wins, same criterion as the old per-face bake

        better = visible & (score > best_score)
        if not better.any():
            continue

        xi = np.clip(x[better].astype(np.int32), 0, w - 1)
        yi = np.clip(y[better].astype(np.int32), 0, h - 1)
        best_color[better] = kf.image_rgb[yi, xi].astype(np.float64)
        best_score[better] = score[better]
        hit |= better

    if not hit.any():
        bus.log("Vertex color baking: no vertices were visible in any keyframe — keeping the pre-bake mesh colors", level="warn")
        return False

    # Vertices no camera ever saw keep whatever color Poisson's own
    # point-interpolation already gave them, rather than going black.
    existing = np.asarray(mesh.vertex_colors) * 255.0 if mesh.has_vertex_colors() else np.full((n, 3), 128.0)
    final_colors = np.where(hit[:, None], best_color, existing)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(final_colors / 255.0, 0.0, 1.0))

    bus.log(f"Vertex color baking: {int(hit.sum()):,}/{n:,} vertices colored from real photos in {time.time()-t0:.1f}s (no UV atlas needed)")
    return True


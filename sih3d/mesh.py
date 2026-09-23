"""Mesh generation: vertex-colored mesh first (always), then best-effort
texture baking from full-resolution keyframes.

Vertex-colored mesh: from the TSDF volume via marching cubes if
`fusion.Open3DTsdfFusion` produced one (denser/smoother), else directly from
the fused point cloud via Open3D Poisson surface reconstruction. If Open3D
itself isn't available at all, meshing is skipped with a clearly logged
reason — the pipeline still ships the point cloud (LAS/PLY), matching the
robustness rule "log clearly if unavailable; never crash".

Texture baking: xatlas UV-unwraps the mesh, then for each face we pick the
"best view" among the full-resolution keyframes (most fronto-parallel,
i.e. maximizing -dot(face_normal, view_direction), among cameras where the
face's centroid actually projects inside the image) and fill that face's
triangle in the atlas with the color sampled at its centroid's projection.
This is a flat-per-face bake (constant color per triangle, not full per-texel
projective sampling) — deliberately simpler than a full rasterizer, and
still literally matches the task's own phrasing ("best-view per face").
Skipped (logged) if it would blow the time budget or if xatlas isn't
installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .events import EventBus
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
    poisson_depth: int = 9,
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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cloud.colors, 0, 255) / 255.0)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
        densities = np.asarray(densities)
        # Trim low-density (extrapolated/hallucinated) vertices at the
        # reconstruction's outer fringe — Poisson fills holes by design,
        # which without trimming produces a bloated blob past the actual
        # observed surface.
        keep = densities >= np.quantile(densities, 0.02)
        mesh.remove_vertices_by_mask(~keep)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    except Exception as e:
        bus.log(f"Poisson reconstruction failed ({e}); trying ball-pivoting as a second fallback", level="warn")
        try:
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = float(np.mean(distances))
            radii = [avg_dist * r for r in (1.5, 2.0, 3.0)]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        except Exception as e2:
            bus.log(f"Ball-pivoting also failed ({e2}) — mesh generation skipped", level="warn")
            return MeshResult(mesh=None, method="none")

    if len(mesh.vertices) == 0:
        bus.log("Meshing produced zero vertices — mesh generation skipped", level="warn")
        return MeshResult(mesh=None, method="none")

    mesh.compute_vertex_normals()
    bus.log(f"Mesh: built via Poisson/ball-pivoting from point cloud ({len(mesh.vertices)} vertices, {len(mesh.triangles)} faces)")
    return MeshResult(mesh=mesh, method="poisson", n_vertices=len(mesh.vertices), n_faces=len(mesh.triangles))


@dataclass
class KeyframeForBaking:
    image_rgb: np.ndarray        # (H,W,3) uint8, full resolution
    camera_pose_c2w: np.ndarray  # (4,4)
    intrinsics: np.ndarray       # (3,3)


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


def bake_texture(
    mesh, keyframes: list[KeyframeForBaking], bus: EventBus,
    time_budget_s: float = 60.0, atlas_resolution: int = 2048,
) -> TextureBakeResult:
    t0 = time.time()
    try:
        import xatlas
    except Exception as e:
        return TextureBakeResult(textured=False, skipped_reason=f"xatlas not installed ({e})")

    if mesh is None or len(mesh.triangles) == 0:
        return TextureBakeResult(textured=False, skipped_reason="no mesh to texture")
    if not keyframes:
        return TextureBakeResult(textured=False, skipped_reason="no full-resolution keyframes available for baking")

    try:
        from PIL import Image, ImageDraw
    except Exception as e:
        return TextureBakeResult(textured=False, skipped_reason=f"PIL not available ({e})")

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.triangle_normals) if mesh.has_triangle_normals() else None
    if normals is None:
        mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals)

    bus.log(f"Texture baking: UV-unwrapping {len(faces)} faces via xatlas...")
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    # xatlas may duplicate/reorder vertices at UV seams; `vmapping` maps each
    # new (post-unwrap) vertex back to its original vertex index.
    unwrapped_positions = vertices[vmapping]
    face_centroids_world = unwrapped_positions[indices].mean(axis=1)  # (n_faces, 3), matches `indices`' face order
    face_normals = normals  # original per-triangle normals, same face order as `faces`/`indices` (xatlas preserves face order)

    atlas = Image.new("RGB", (atlas_resolution, atlas_resolution), (128, 128, 128))
    draw = ImageDraw.Draw(atlas)

    n_faces = len(indices)
    baked = 0
    skipped_time_budget = False

    for fi in range(n_faces):
        if time.time() - t0 > time_budget_s:
            skipped_time_budget = True
            break

        centroid = face_centroids_world[fi]
        normal = face_normals[fi] if fi < len(face_normals) else np.array([0.0, 0.0, 1.0])

        best_score = -1.0
        best_color = None
        for kf in keyframes:
            view_dir = centroid - kf.camera_pose_c2w[:3, 3]
            dist = np.linalg.norm(view_dir)
            if dist < 1e-6:
                continue
            view_dir = view_dir / dist
            facing = -float(np.dot(normal, view_dir))
            if facing <= 0.05:  # back-facing or near-grazing: unusable
                continue

            pix, depth = _project_points(centroid[None, :], kf.camera_pose_c2w, kf.intrinsics)
            if depth[0] <= 0:
                continue
            x, y = pix[0]
            h, w = kf.image_rgb.shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                continue

            score = facing / max(dist, 1e-3)  # prefer fronto-parallel and close
            if score > best_score:
                best_score = score
                best_color = kf.image_rgb[int(y), int(x)]

        if best_color is None:
            continue

        uv_tri = uvs[indices[fi]] * atlas_resolution
        poly = [(float(uv_tri[k, 0]), float(atlas_resolution - uv_tri[k, 1])) for k in range(3)]
        draw.polygon(poly, fill=tuple(int(c) for c in best_color))
        baked += 1

    elapsed = time.time() - t0
    if skipped_time_budget:
        bus.log(f"Texture baking hit the time budget ({time_budget_s:.0f}s) after {baked}/{n_faces} faces — using partial bake", level="warn")
    else:
        bus.log(f"Texture baking: {baked}/{n_faces} faces colored in {elapsed:.1f}s")

    if baked == 0:
        return TextureBakeResult(textured=False, skipped_reason="no face could be matched to any keyframe view", timing_s=elapsed)

    return TextureBakeResult(
        textured=True, texture_rgb=np.array(atlas), uv=uvs[indices].reshape(-1, 2).astype(np.float32),
        timing_s=elapsed,
    )

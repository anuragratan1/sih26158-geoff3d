"""Render-vs-ground-truth comparison: a 3-panel view per keyframe — the real
photo, the raw fused point cloud reprojected into that exact camera, and the
final polished mesh rendered through that exact camera — so you can see
where each stage of the pipeline stands relative to reality and relative to
each other (point cloud always available; the mesh panel says so plainly
when meshing didn't produce one for this run).

Both reprojections use the keyframe's own real estimated intrinsics + pose,
not a generic orbit view — this is a check against what the camera actually
saw, not an approximation.

Mesh panel: rendered via pyrender (GPU, EGL — the standard headless-GPU-
rendering path used across ML/robotics tooling on cloud GPU boxes; Open3D's
rendering.OffscreenRenderer needs Vulkan instead, which errored out on a
real Kaggle session) using the keyframe's real intrinsics as a
pyrender.IntrinsicsCamera and its real pose
converted from this pipeline's OpenCV camera convention (+Z forward, +Y
down) to OpenGL's (-Z forward, +Y up). Loads from the actually-exported
mesh.glb (not the pre-bake in-memory mesh) so a successful texture bake is
what's shown, not the vertex-colored mesh that precedes it — bake_texture()
never modifies the mesh object in place, the geometry and the baked texture
only combine into one object at export time.

Point-cloud panel: reprojects points into the same camera model. Coverage
there is coarse-bin (fraction of the photo's area a reprojected point
landed in) — deliberately not per-pixel, since a sparse point cloud will
never cover every pixel even for a perfect reconstruction.

Neither panel attempts a photometric/SSIM score against the photo: even a
real mesh render here is unlit-flat/vertex-colored, not shaded to match the
photo's actual lighting, so a pixel-for-pixel intensity match isn't
something either method can honestly claim — coverage (did the geometry
actually reach this part of frame at all) is what both can back up.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import ComparisonRecord
from .events import EventBus


@dataclass
class _RenderKeyframe:
    frame_index: int
    image_rgb: np.ndarray
    camera_pose_c2w: np.ndarray
    intrinsics: np.ndarray


def _project_points(points_world: np.ndarray, camera_pose_c2w: np.ndarray, intrinsics: np.ndarray):
    """Same pinhole projection math as mesh.py's texture baker — reused
    (not imported) since it's an 8-line pure function and this module has
    no other reason to depend on mesh.py."""
    w2c = np.linalg.inv(camera_pose_c2w)
    pts_h = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    depth = pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pix = (intrinsics @ pts_cam.T).T
        pix_xy = pix[:, :2] / pix[:, 2:3]
    return pix_xy, depth


def _load_mesh_for_render(mesh_glb_path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Loads the actually-exported mesh.glb, baking any real UV texture
    into per-vertex colors (to_color()) so a successful bake is what
    renders here, not silently-ignored default vertex colors. Returns
    (vertices, faces, vertex_colors_or_None), or None if unavailable."""
    if not mesh_glb_path:
        return None
    try:
        import trimesh

        m = trimesh.load(str(mesh_glb_path), process=False)
        if hasattr(m, "geometry"):  # a Scene (GLB) — take the first mesh
            m = next(iter(m.geometry.values()))
        if hasattr(m.visual, "to_color"):
            try:
                m.visual = m.visual.to_color()
            except Exception:
                pass
        vertex_colors = None
        if hasattr(m.visual, "vertex_colors") and m.visual.vertex_colors is not None:
            vertex_colors = np.asarray(m.visual.vertex_colors)[:, :3] / 255.0
        return np.asarray(m.vertices), np.asarray(m.faces), vertex_colors
    except Exception:
        return None


def _render_mesh_exact_camera(
    mesh_vertices: np.ndarray, mesh_faces: np.ndarray, mesh_vertex_colors: np.ndarray | None,
    camera_pose_c2w: np.ndarray, intrinsics: np.ndarray, width: int, height: int,
) -> tuple[np.ndarray, float] | None:
    """Renders the real mesh through the keyframe's own real camera model.
    Returns (rgb_frame, coverage_pct) or None on any failure (missing
    pyrender/EGL, a context that fails on first render, ...) — the caller
    shows a "mesh not available" panel instead, this is never fatal."""
    try:
        import os
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender
        import trimesh as _trimesh

        tm = _trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces, process=False)
        if mesh_vertex_colors is not None:
            rgba = np.concatenate([np.clip(mesh_vertex_colors, 0, 1), np.ones((len(mesh_vertex_colors), 1))], axis=1)
            tm.visual.vertex_colors = (rgba * 255).astype(np.uint8)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.6, 0.6, 0.6])
        scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))

        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=10000.0)
        # This pipeline's camera_pose_c2w is OpenCV convention (+X right,
        # +Y down, +Z forward). pyrender/OpenGL/glTF expects +Y up, +Z
        # backward (camera looks down its own -Z). Flipping the Y and Z
        # basis columns converts between them without changing the
        # camera's actual position or where it's looking.
        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
        pose_gl = camera_pose_c2w @ cv_to_gl
        scene.add(camera, pose=pose_gl)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=pose_gl)

        renderer = pyrender.OffscreenRenderer(width, height)
        try:
            color, _depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        finally:
            renderer.delete()

        alpha = color[:, :, 3]
        coverage_pct = 100.0 * float((alpha > 0).sum()) / alpha.size
        return color[:, :, :3], coverage_pct
    except Exception:
        return None


def _reproject_points(points_world, colors01, camera_pose_c2w, intrinsics, w, h):
    """Returns (vis_x, vis_y, vis_colors, coverage_pct) or None if nothing
    reprojected into frame."""
    pix_xy, depth = _project_points(points_world, camera_pose_c2w, intrinsics)
    in_front = depth > 0
    x, y = pix_xy[:, 0], pix_xy[:, 1]
    visible = in_front & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not visible.any():
        return None

    vis_x, vis_y, vis_colors = x[visible], y[visible], colors01[visible]

    # Coverage: how much of the photo's area a reprojected point landed in,
    # at a coarse (64px) bin resolution — deliberately not per-pixel, since
    # a sparse point cloud will never cover every single pixel even for a
    # geometrically perfect reconstruction; coarse bins measure "did
    # geometry reach this region of frame," which is what a point splat can
    # honestly say.
    bin_px = 64
    # Ceiling division, not floor: floor(w // bin_px) undercounts the true
    # number of reachable bins whenever w/h isn't an exact multiple of
    # bin_px (e.g. w=2560 gives bins_x=40 exactly, but h=1440 gives
    # floor(1440/64)=22 while a point can still land in bin index 22 — the
    # partial last bin — pushing hit_bins past the undercounted denominator
    # and coverage over 100%, which is a nonsense value for a percentage.
    # This was a real observed bug: a run reported "106% coverage".
    bins_x, bins_y = max(1, -(-w // bin_px)), max(1, -(-h // bin_px))
    hit_bins = set(zip((vis_x // bin_px).astype(int), (vis_y // bin_px).astype(int)))
    coverage_pct = 100.0 * len(hit_bins) / (bins_x * bins_y)
    return vis_x, vis_y, vis_colors, coverage_pct


def render_vs_ground_truth(
    points_world: np.ndarray, colors: np.ndarray, render_keyframes: list[_RenderKeyframe],
    out_dir: Path, bus: EventBus, n_views: int = 3, max_points: int = 300_000,
    mesh_glb_path=None,
) -> list[ComparisonRecord]:
    if len(points_world) == 0 or not render_keyframes:
        return []

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        bus.log(f"Render-vs-ground-truth comparison skipped: matplotlib unavailable ({e})", level="warn")
        return []

    mesh_data = _load_mesh_for_render(mesh_glb_path)
    if mesh_glb_path and mesh_data is None:
        bus.log(f"Render-vs-ground-truth: could not load {mesh_glb_path} for rendering; mesh panel will show 'not available'", level="warn")

    if len(points_world) > max_points:
        idx = np.random.default_rng(0).choice(len(points_world), size=max_points, replace=False)
        points_world, colors = points_world[idx], colors[idx]
    colors01 = np.clip(colors, 0, 255) / 255.0

    # Evenly spread picks across the sequence rather than the first N —
    # comparing only early frames would hide any drift that accumulates
    # later in the run.
    step = max(1, len(render_keyframes) // n_views)
    picks = render_keyframes[::step][:n_views]

    results: list[ComparisonRecord] = []
    for kf in picks:
        try:
            h, w = kf.image_rgb.shape[:2]

            point_result = _reproject_points(points_world, colors01, kf.camera_pose_c2w, kf.intrinsics, w, h)
            mesh_result = None
            if mesh_data is not None:
                mesh_result = _render_mesh_exact_camera(*mesh_data, kf.camera_pose_c2w, kf.intrinsics, w, h)

            if point_result is None and mesh_result is None:
                bus.log(f"Render-vs-ground-truth: nothing reprojected/rendered into frame {kf.frame_index} (out of view)", level="warn")
                continue

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            axes[0].imshow(kf.image_rgb)
            axes[0].set_title(f"Real photo — frame {kf.frame_index}", fontsize=9)
            axes[0].axis("off")

            point_coverage = None
            axes[1].set_facecolor((0.05, 0.06, 0.08))
            if point_result is not None:
                vis_x, vis_y, vis_colors, point_coverage = point_result
                # Coverage above is computed from the full visible set;
                # matplotlib's scatter() is pure CPU rasterization (no GPU
                # path exists for this, regardless of what's installed) and
                # visibly slow at the >100k points/panel this can reach at
                # full image resolution — capping what's actually PLOTTED
                # (display only, not the coverage math) to 20k is
                # indistinguishable by eye at this figure size and much
                # faster to render.
                plot_x, plot_y, plot_c = vis_x, vis_y, vis_colors
                if len(plot_x) > 20_000:
                    plot_idx = np.random.default_rng(0).choice(len(plot_x), size=20_000, replace=False)
                    plot_x, plot_y, plot_c = plot_x[plot_idx], plot_y[plot_idx], plot_c[plot_idx]
                axes[1].scatter(plot_x, plot_y, c=plot_c, s=1.5, marker=".")
                axes[1].set_xlim(0, w)
                axes[1].set_ylim(h, 0)  # row 0 = top, matching imshow's convention on the left panel
                axes[1].set_aspect("equal")
                axes[1].set_title(f"Point cloud (raw) — {point_coverage:.0f}% coverage", fontsize=9)
            else:
                axes[1].text(0.5, 0.5, "no points\nreprojected", ha="center", va="center", color="#8b949e", transform=axes[1].transAxes)
                axes[1].set_title("Point cloud (raw)", fontsize=9)
            axes[1].axis("off")

            mesh_coverage = None
            axes[2].set_facecolor((0.05, 0.06, 0.08))
            if mesh_result is not None:
                render_rgb, mesh_coverage = mesh_result
                axes[2].imshow(render_rgb)
                axes[2].set_title(f"Mesh (final output) — {mesh_coverage:.0f}% coverage", fontsize=9)
            else:
                reason = "mesh render unavailable" if mesh_data is not None else "no mesh for this run"
                axes[2].text(0.5, 0.5, reason, ha="center", va="center", color="#8b949e", transform=axes[2].transAxes)
                axes[2].set_title("Mesh (final output) — not available", fontsize=9)
            axes[2].axis("off")

            plt.tight_layout()
            out_path = out_dir / f"comparison_{kf.frame_index}.png"
            fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
            plt.close(fig)

            results.append(ComparisonRecord(
                frame_index=kf.frame_index, point_coverage_pct=point_coverage,
                mesh_coverage_pct=mesh_coverage, image_path=str(out_path),
            ))
            bus.log(
                f"Render-vs-ground-truth: frame {kf.frame_index} — "
                f"point coverage {f'{point_coverage:.0f}%' if point_coverage is not None else 'n/a'}, "
                f"mesh coverage {f'{mesh_coverage:.0f}%' if mesh_coverage is not None else 'n/a'} ({out_path.name})"
            )
        except Exception as e:
            bus.log(f"Render-vs-ground-truth comparison failed for frame {kf.frame_index} ({type(e).__name__}: {e})", level="warn")

    return results

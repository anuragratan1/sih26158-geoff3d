"""Pre-rendered (matplotlib, static) result views for the notebook's
per-stage cells. Every function here renders directly into the calling
cell's output via plt.show() — Jupyter captures that as a static image, so
the notebook file itself stays light (no live widgets re-embedded per
cell, no growing JSON state). Each function reads straight from
RunArtifacts (already populated by pipeline.py as it runs) and, where
useful, ReportBuilder for timing/GPU data.

Every function degrades gracefully (prints a plain-text note) if the data
it needs isn't there yet — a stage cell run against a run that hit an
early fallback shouldn't raise, it should say what's missing.
"""

from __future__ import annotations

import numpy as np

from .artifacts import RunArtifacts


def _geotiff_to_png(path: str, max_dim: int = 1600) -> bytes:
    """Read a GeoTIFF into a bounded PNG for the final static results cell."""
    import io
    from PIL import Image
    import rasterio

    with rasterio.open(path) as src:
        data = src.read()
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
    image = Image.fromarray((np.nan_to_num(arr) * 255).astype(np.uint8))
    if max(image.size) > max_dim:
        scale = max_dim / max(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
from .report import ReportBuilder


def _require_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def render_frame_extraction(artifacts: RunArtifacts) -> None:
    plt = _require_matplotlib()
    if not artifacts.keyframes:
        print("No keyframe data captured.")
        return

    accepted = [k for k in artifacts.keyframes if k.accepted]
    rejected = [k for k in artifacts.keyframes if not k.accepted]
    print(f"Keyframes: {len(accepted)} accepted, {len(rejected)} rejected (of {len(artifacts.keyframes)} candidates)")

    ordered = sorted(artifacts.keyframes, key=lambda k: k.frame_index)
    n = len(ordered)
    cols = min(12, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.1))
    axes = np.atleast_2d(axes)
    for i, kf in enumerate(ordered):
        ax = axes[i // cols, i % cols]
        if kf.thumbnail is not None:
            ax.imshow(kf.thumbnail, alpha=1.0 if kf.accepted else 0.35)
        ax.set_title(f"{kf.sharpness:.0f}", fontsize=6, color="#2ea043" if kf.accepted else "#d29922")
        ax.axis("off")
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].axis("off")
    fig.suptitle("Keyframe grid (rejected greyed out, title = sharpness score)", fontsize=9)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 2.5))
    xs_a = [k.frame_index for k in accepted]
    ys_a = [k.sharpness for k in accepted]
    xs_r = [k.frame_index for k in rejected]
    ys_r = [k.sharpness for k in rejected]
    ax2.scatter(xs_a, ys_a, s=10, color="#2ea043", label="accepted")
    ax2.scatter(xs_r, ys_r, s=10, color="#d29922", label="rejected")
    ax2.set_xlabel("frame index")
    ax2.set_ylabel("sharpness")
    ax2.legend(fontsize=8)
    ax2.set_title("Sharpness by frame")
    plt.tight_layout()
    plt.show()
    plt.close(fig2)


def render_poses(artifacts: RunArtifacts) -> None:
    plt = _require_matplotlib()
    if not artifacts.gps_track_enu and not artifacts.camera_track_enu:
        print("No trajectory data captured (no GPS and no resolved camera poses).")
        return

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    if artifacts.gps_track_enu:
        g = np.array(artifacts.gps_track_enu)
        ax.plot(g[:, 0], g[:, 1], g[:, 2], color="#8b949e", label="GPS", linewidth=1.5)
    if artifacts.camera_track_enu:
        c = np.array(artifacts.camera_track_enu)
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color="#2ea043", label="estimated camera", linewidth=1.5)
    ax.set_xlabel("E (m)")
    ax.set_ylabel("N (m)")
    ax.set_zlabel("U (m)")
    ax.legend(fontsize=8)
    title = "Camera trajectory: GPS vs. estimated"
    if artifacts.collinearity_index is not None:
        title += f"  |  collinearity index: {artifacts.collinearity_index:.4f} (near 0 = straight single pass)"
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def render_geometric_reconstruction(artifacts: RunArtifacts) -> None:
    plt = _require_matplotlib()
    if not artifacts.geometry_samples:
        print("No geometry samples captured.")
        return

    n = len(artifacts.geometry_samples)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), squeeze=False)
    for row, sample in enumerate(artifacts.geometry_samples):
        ax_rgb, ax_depth, ax_conf, ax_mask = axes[row]
        if sample.rgb is not None:
            ax_rgb.imshow(sample.rgb)
        ax_rgb.set_title(f"chunk {sample.chunk_idx}: RGB", fontsize=8)
        ax_rgb.axis("off")

        if sample.depth is not None:
            ax_depth.imshow(sample.depth, cmap="viridis")
        ax_depth.set_title("depth", fontsize=8)
        ax_depth.axis("off")

        if sample.confidence is not None:
            ax_conf.imshow(sample.confidence, cmap="magma", vmin=0, vmax=1)
        ax_conf.set_title("confidence", fontsize=8)
        ax_conf.axis("off")

        if sample.rgb is not None:
            overlay = sample.rgb.copy().astype(np.float32)
            if sample.dynamic_mask is not None and sample.dynamic_mask.any():
                red = np.zeros_like(overlay)
                red[..., 0] = 255
                m = sample.dynamic_mask[..., None].astype(np.float32)
                overlay = overlay * (1 - 0.5 * m) + red * (0.5 * m)
            ax_mask.imshow(overlay.astype(np.uint8))
        mask_note = "dynamic mask (red)" if (sample.dynamic_mask is not None and sample.dynamic_mask.any()) else "no dynamic objects masked"
        ax_mask.set_title(mask_note, fontsize=8)
        ax_mask.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def render_large_scale_alignment(artifacts: RunArtifacts) -> None:
    plt = _require_matplotlib()
    if not artifacts.chunk_alignments:
        print("No per-chunk alignment data captured (no-GPS mode, or alignment fell back before any chunk completed).")
        return

    records = sorted(artifacts.chunk_alignments, key=lambda r: r.chunk_idx)
    idx = [r.chunk_idx for r in records]
    before = [r.rmse_before_m if r.rmse_before_m is not None else np.nan for r in records]
    after = [r.rmse_after_m if r.rmse_after_m is not None else np.nan for r in records]

    fig, ax = plt.subplots(figsize=(8, 3))
    width = 0.35
    x = np.arange(len(idx))
    ax.bar(x - width / 2, before, width, label="before refinement", color="#9e6a03")
    ax.bar(x + width / 2, after, width, label="after refinement", color="#2ea043")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in idx])
    ax.set_xlabel("chunk index")
    ax.set_ylabel("RMSE vs GPS (m)")
    ax.set_title(f"Per-chunk alignment residual, before/after pose-graph refinement ({records[0].mode})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def render_dense_point_cloud(artifacts: RunArtifacts) -> None:
    plt = _require_matplotlib()
    print(f"Point count: {artifacts.point_count:,}")
    if artifacts.point_cloud_preview is None or len(artifacts.point_cloud_preview) == 0:
        print("No point cloud preview captured.")
        return

    pts = artifacts.point_cloud_preview
    colors = artifacts.point_cloud_preview_colors
    colors01 = np.clip(colors, 0, 255) / 255.0 if colors is not None else None

    fig = plt.figure(figsize=(11, 5))
    ax_top = fig.add_subplot(121, projection="3d")
    ax_top.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors01, s=0.5)
    ax_top.view_init(elev=89, azim=-90)
    ax_top.set_title("Top-down", fontsize=9)
    ax_top.set_axis_off()

    ax_obl = fig.add_subplot(122, projection="3d")
    ax_obl.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors01, s=0.5)
    ax_obl.view_init(elev=25, azim=-60)
    ax_obl.set_title("Oblique", fontsize=9)
    ax_obl.set_axis_off()

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    coverage_path = artifacts.output_paths.get("coverage.tif")
    if coverage_path:
        try:
            from PIL import Image
            import io

            png = _geotiff_to_png(coverage_path)
            img = Image.open(io.BytesIO(png))
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            ax2.imshow(img)
            ax2.set_title("Coverage map (view count / confidence)", fontsize=9)
            ax2.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close(fig2)
        except Exception as e:
            print(f"Coverage map preview unavailable: {e}")
    else:
        print("coverage.tif not yet written.")


def render_mesh_textured_model(artifacts: RunArtifacts) -> None:
    plt = _require_matplotlib()
    print(f"Mesh: {artifacts.mesh_n_vertices:,} vertices, {artifacts.mesh_n_faces:,} faces (method: {artifacts.mesh_method})")

    mesh_path = artifacts.output_paths.get("mesh.glb") or artifacts.output_paths.get("mesh.obj")
    rendered = False
    if mesh_path:
        try:
            rendered = _render_mesh_offscreen(mesh_path, plt)
        except Exception as e:
            print(f"Offscreen mesh render unavailable ({e}); showing a flat wireframe fallback instead.")
    if not rendered and mesh_path:
        try:
            _render_mesh_wireframe(mesh_path, plt)
        except Exception as e:
            print(f"Mesh preview unavailable: {e}")
    elif not mesh_path:
        print("No mesh file available yet.")

    for name, title in [("dsm.tif", "DSM"), ("orthomosaic.tif", "Orthomosaic")]:
        path = artifacts.output_paths.get(name)
        if not path:
            print(f"{title}: not yet written.")
            continue
        try:
            from PIL import Image
            import io

            png = _geotiff_to_png(path, max_dim=500)
            img = Image.open(io.BytesIO(png))
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"{title} preview unavailable: {e}")


def _render_mesh_offscreen(mesh_path: str, plt) -> bool:
    """Tries a real textured/shaded render via Open3D's offscreen
    renderer, top-down + oblique (matching the point-cloud preview's two
    views). Needs a working (EGL/OSMesa) headless GL context, which isn't
    guaranteed on Kaggle — returns False (never raises past this function)
    so the caller falls back to a flat wireframe."""
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
    if len(mesh.vertices) == 0:
        return False
    mesh.compute_vertex_normals()

    center = mesh.get_center()
    extent = np.asarray(mesh.get_max_bound()) - np.asarray(mesh.get_min_bound())
    radius = float(np.linalg.norm(extent)) or 10.0

    renderer = rendering.OffscreenRenderer(640, 480)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", mesh, mat)
    renderer.scene.set_background([0.05, 0.06, 0.08, 1.0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    views = [("Top-down", center + [0, 0, radius], [0, 1, 0]), ("Oblique", center + [radius, radius, radius], [0, 0, 1])]
    for ax, (title, eye, up) in zip(axes, views):
        renderer.scene.camera.look_at(center, eye, up)
        img = renderer.render_to_image()
        ax.imshow(np.asarray(img))
        ax.set_title(f"Textured mesh — {title}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return True


def _render_mesh_wireframe(mesh_path: str, plt) -> None:
    """Flat-shaded matplotlib fallback, top-down + oblique: no texture,
    but shows real geometry/vertex colors, and never needs a GL context."""
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    m = trimesh.load(mesh_path, process=False)
    if hasattr(m, "geometry"):  # a Scene (e.g. GLB), take the first mesh
        m = next(iter(m.geometry.values()))
    verts = np.asarray(m.vertices)
    faces = np.asarray(m.faces)
    if len(faces) > 20000:
        idx = np.random.default_rng(0).choice(len(faces), size=20000, replace=False)
        faces = faces[idx]

    face_colors = None
    if hasattr(m.visual, "vertex_colors") and m.visual.vertex_colors is not None:
        vc = np.asarray(m.visual.vertex_colors)[:, :3] / 255.0
        face_colors = vc[faces].mean(axis=1)

    fig = plt.figure(figsize=(11, 5))
    views = [("Top-down", 89, -90), ("Oblique", 25, -60)]
    for i, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        poly = Poly3DCollection(verts[faces], facecolor=face_colors if face_colors is not None else "#888888", linewidths=0)
        ax.add_collection3d(poly)
        ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
        ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
        ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Mesh (flat-shaded fallback) — {title}", fontsize=9)
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def render_final_summary(report: ReportBuilder) -> None:
    plt = _require_matplotlib()
    d = report.to_dict()

    def _row(s: dict) -> str:
        elapsed = f"{s['elapsed_s']:.1f}s" if s["elapsed_s"] is not None else "-"
        gpu = ", ".join(f"GPU{k}: {v:.0f}%" for k, v in s["avg_gpu_util_pct"].items()) or "-"
        return f"<tr><td>{s['name']}</td><td>{s['status']}</td><td>{elapsed}</td><td>{gpu}</td></tr>"

    rows = "".join(_row(s) for s in d["stages"])
    from IPython.display import HTML, display

    display(HTML(f"<table><tr><th>Stage</th><th>Status</th><th>Elapsed</th><th>Avg GPU Util</th></tr>{rows}</table>"))
    print(f"\nTotal elapsed: {d['total_elapsed_s']:.1f}s" if d["total_elapsed_s"] else "")
    print(f"Georeferenced: {d['georeferenced']}")
    print(f"Alignment RMSE vs GPS: {d['alignment_rmse_m']} m")
    print(f"Collinearity index: {d['collinearity_index']}")
    if d["fallbacks_triggered"]:
        print(f"\n{len(d['fallbacks_triggered'])} fallback(s) triggered:")
        for f in d["fallbacks_triggered"]:
            print(" -", f)


def render_showcase(artifacts: RunArtifacts, n_photos: int = 3, video_frames: int = 36, video_fps: int = 12) -> None:
    """Best-effort: a few stills + a short orbit video rendered straight
    from the reconstructed mesh (falls back to the point cloud if no mesh
    was produced), displayed inline so there's something to actually look
    at without leaving the notebook or downloading anything.

    Uses matplotlib, not Open3D's offscreen renderer — Open3D's high-level
    `rendering.OffscreenRenderer` needs a working Vulkan loader (Filament),
    which a real Kaggle GPU session does not have installed by default
    ("Failed to load vulkan library!"), and installing system Vulkan
    packages isn't something to gamble a notebook run on. matplotlib is
    already a hard dependency of this file and needs no GPU rendering
    backend at all — the tradeoff is a flat-shaded look, not a lit/
    textured one, but it reliably renders on any box.

    No separate "camera up" vector to get wrong here either: mplot3d's
    `view_init(elev, azim)` is inherently relative to Z being vertical,
    which already matches this pipeline's local-ENU (Z-up) point
    convention (see viewer.py/telemetry.py) — the geometry is plotted
    as-is, in its own (x, y, z) axes, with no separate up-vector to
    misconfigure the way a real camera-based renderer has.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    from IPython.display import Video, display
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plt = _require_matplotlib()

    mesh_path = artifacts.output_paths.get("mesh.glb") or artifacts.output_paths.get("mesh.obj")
    cloud_path = artifacts.output_paths.get("pointcloud.ply")
    if not mesh_path and not cloud_path:
        print("Nothing to render yet: no mesh or point cloud output found.")
        return

    verts = faces = colors = points = point_colors = None
    if mesh_path:
        try:
            import trimesh

            m = trimesh.load(mesh_path, process=False)
            if hasattr(m, "geometry"):  # a Scene (e.g. GLB) — take the first mesh
                m = next(iter(m.geometry.values()))
            verts, faces = np.asarray(m.vertices), np.asarray(m.faces)
            if len(faces) > 20000:
                idx = np.random.default_rng(0).choice(len(faces), size=20000, replace=False)
                faces = faces[idx]
            if hasattr(m.visual, "vertex_colors") and m.visual.vertex_colors is not None:
                vc = np.asarray(m.visual.vertex_colors)[:, :3] / 255.0
                colors = vc[faces].mean(axis=1)
        except Exception as e:
            print(f"Could not load {mesh_path} for showcase render ({e}); trying point cloud instead.")
            verts = None
    if verts is None and cloud_path:
        try:
            # trimesh, not Open3D: a plain PLY point cloud (no faces) loads
            # as a trimesh.PointCloud with no GPU/rendering backend touched
            # at all — Open3D's import alone was enough to trip the exact
            # "Failed to load vulkan library!" this function exists to
            # avoid (its Jupyter-environment auto-detection eagerly
            # initializes the Filament/Vulkan-based renderer on import,
            # regardless of whether anything then actually asks it to
            # render), so it has no business being imported in this
            # function at all, not even for pure point-cloud I/O.
            import trimesh

            pc = trimesh.load(cloud_path, process=False)
            points = np.asarray(pc.vertices)
            point_colors = None
            if hasattr(pc, "colors") and pc.colors is not None and len(pc.colors):
                point_colors = np.asarray(pc.colors)[:, :3] / 255.0
            if len(points) > 100_000:
                idx = np.random.default_rng(0).choice(len(points), size=100_000, replace=False)
                points = points[idx]
                if point_colors is not None:
                    point_colors = point_colors[idx]
        except Exception as e:
            print(f"Could not load {cloud_path} for showcase render ({e}).")
    if verts is None and points is None:
        print("Showcase render skipped: nothing loadable.")
        return

    try:
        all_pts = verts if verts is not None else points
        bounds = list(zip((all_pts[:, i].min() for i in range(3)), (all_pts[:, i].max() for i in range(3))))

        fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor((0.05, 0.06, 0.08))
        fig.patch.set_facecolor((0.05, 0.06, 0.08))
        if verts is not None:
            ax.add_collection3d(Poly3DCollection(
                verts[faces], facecolor=colors if colors is not None else "#888888", linewidths=0,
            ))
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_colors, s=0.5, marker=".")
        ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1]); ax.set_zlim(*bounds[2])
        ax.set_box_aspect((bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0], bounds[2][1] - bounds[2][0]))
        ax.axis("off")

        def _snapshot(azim: float) -> np.ndarray:
            ax.view_init(elev=25, azim=azim)
            fig.canvas.draw()
            return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

        print(f"Rendering {n_photos} stills...")
        for angle in np.linspace(0, 360, n_photos, endpoint=False):
            _snapshot(float(angle))
            plt.show()

        print(f"Rendering a {video_frames}-frame orbit video...")
        from PIL import Image as PILImage

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for i in range(video_frames):
                frame = _snapshot(360.0 * i / video_frames)
                PILImage.fromarray(frame).save(tmp_path / f"frame_{i:04d}.png")

            video_path = tmp_path / "orbit.mp4"
            result = subprocess.run(
                ["ffmpeg", "-y", "-framerate", str(video_fps), "-i", str(tmp_path / "frame_%04d.png"),
                 "-pix_fmt", "yuv420p", str(video_path)],
                capture_output=True, text=True, timeout=60,
            )
            plt.close(fig)
            if result.returncode != 0 or not video_path.exists():
                print(f"Orbit video assembly failed ({result.stderr.strip()[-300:]}); stills above are still available.")
                return
            display(Video(str(video_path), embed=True, html_attributes="controls loop"))
    except Exception as e:
        print(f"Showcase render failed ({type(e).__name__}: {e}); the downloadable mesh.glb/pointcloud.ply above are still valid.")


def render_ground_truth_comparison(artifacts: RunArtifacts) -> None:
    """Displays the [real photo | reprojected point cloud] comparisons
    validation.py already rendered to PNG during the run — this just shows
    them inline and prints the coverage numbers next to each. See
    validation.py's module docstring for exactly what "coverage" does and
    doesn't claim to measure."""
    if not artifacts.comparisons:
        print("No render-vs-ground-truth comparisons available for this run "
              "(needs at least one keyframe with a resolved camera pose).")
        return

    from IPython.display import Image as IPyImage, display

    for comp in artifacts.comparisons:
        print(f"Frame {comp.frame_index}: {comp.coverage_pct:.0f}% point coverage "
              f"(fraction of the photo's area a reprojected point landed in — not a pixel-accuracy score)")
        try:
            display(IPyImage(filename=comp.image_path))
        except Exception as e:
            print(f"  (could not display {comp.image_path}: {e})")

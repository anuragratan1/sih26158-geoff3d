"""Render-vs-ground-truth comparison: reprojects the fused point cloud into
a few real keyframe cameras (their actual estimated intrinsics + pose) and
places that reprojection next to the real photo the camera actually
captured — using the exact camera model the pipeline itself estimated, not
a generic orbit view, so this is a direct check of "does the reconstructed
geometry line up with what the camera saw here."

Reports point-coverage percentage per view (fraction of the photo's pixels
that had at least one reprojected point land on them) as the quantitative
signal. Deliberately does NOT attempt a photometric/SSIM score against the
photo: this is a sparse point splat, not a rasterized/shaded triangle
render, so a pixel-for-pixel match isn't something a sparse reprojection
can honestly claim to measure — coverage (did the geometry actually reach
this part of frame at all) is the metric this method can back up.
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


def render_vs_ground_truth(
    points_world: np.ndarray, colors: np.ndarray, render_keyframes: list[_RenderKeyframe],
    out_dir: Path, bus: EventBus, n_views: int = 3, max_points: int = 300_000,
) -> list[ComparisonRecord]:
    if len(points_world) == 0 or not render_keyframes:
        return []

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        bus.log(f"Render-vs-ground-truth comparison skipped: matplotlib unavailable ({e})", level="warn")
        return []

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
            pix_xy, depth = _project_points(points_world, kf.camera_pose_c2w, kf.intrinsics)
            in_front = depth > 0
            x, y = pix_xy[:, 0], pix_xy[:, 1]
            visible = in_front & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            if not visible.any():
                bus.log(f"Render-vs-ground-truth: no points reprojected into frame {kf.frame_index} (out of view)", level="warn")
                continue

            vis_x, vis_y, vis_colors = x[visible], y[visible], colors01[visible]

            # Coverage: how much of the photo's area a reprojected point
            # landed in, at a coarse (64px) bin resolution — deliberately
            # not per-pixel, since a sparse point cloud will never cover
            # every single pixel even for a geometrically perfect
            # reconstruction; coarse bins measure "did geometry reach this
            # region of frame," which is what a point splat can honestly say.
            bin_px = 64
            bins_x, bins_y = max(1, w // bin_px), max(1, h // bin_px)
            hit_bins = set(zip((vis_x // bin_px).astype(int), (vis_y // bin_px).astype(int)))
            coverage_pct = 100.0 * len(hit_bins) / (bins_x * bins_y)

            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            axes[0].imshow(kf.image_rgb)
            axes[0].set_title(f"Real photo — frame {kf.frame_index}", fontsize=9)
            axes[0].axis("off")

            axes[1].set_facecolor((0.05, 0.06, 0.08))
            axes[1].scatter(vis_x, vis_y, c=vis_colors, s=1.5, marker=".")
            axes[1].set_xlim(0, w)
            axes[1].set_ylim(h, 0)  # row 0 = top, matching imshow's convention on the left panel
            axes[1].set_aspect("equal")
            axes[1].set_title(f"Reprojected point cloud — {coverage_pct:.0f}% coverage", fontsize=9)
            axes[1].axis("off")

            plt.tight_layout()
            out_path = out_dir / f"comparison_{kf.frame_index}.png"
            fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
            plt.close(fig)

            results.append(ComparisonRecord(frame_index=kf.frame_index, coverage_pct=coverage_pct, image_path=str(out_path)))
            bus.log(f"Render-vs-ground-truth: frame {kf.frame_index} — {coverage_pct:.0f}% point coverage ({out_path.name})")
        except Exception as e:
            bus.log(f"Render-vs-ground-truth comparison failed for frame {kf.frame_index} ({type(e).__name__}: {e})", level="warn")

    return results

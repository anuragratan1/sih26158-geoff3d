"""Keyframe selection: sharpness (Laplacian variance, torch conv, GPU) + GPS-distance
spacing for target overlap. Entirely in torch so it can run on the same GPU stream as
decoding without a CPU round-trip per frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import EventBus

_LAPLACIAN_KERNEL = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]


@dataclass
class KeyframeCandidate:
    frame_index: int
    timestamp_s: float
    sharpness: float
    accepted: bool
    reject_reason: str | None = None
    gps_enu: tuple[float, float, float] | None = None


@dataclass
class KeyframeSelection:
    accepted: list[KeyframeCandidate] = field(default_factory=list)
    rejected: list[KeyframeCandidate] = field(default_factory=list)


def _laplacian_variance_batch(frames: "torch.Tensor", downscale: int = 4) -> "torch.Tensor":
    """frames: (N,H,W,3) uint8 or float on GPU/CPU. Returns (N,) sharpness scores.

    Downscales first (sharpness selection doesn't need full resolution and this
    keeps the conv cheap), converts to grayscale, and runs a fixed 3x3 Laplacian
    kernel via torch conv2d. Score = variance of the Laplacian response, the
    standard "blur detection" metric.
    """
    import torch
    import torch.nn.functional as F

    if frames.dtype == torch.uint8:
        frames = frames.float()
    n, h, w, _ = frames.shape
    gray = (0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2])  # (N,H,W)
    gray = gray.unsqueeze(1)  # (N,1,H,W)
    if downscale > 1:
        gray = F.avg_pool2d(gray, kernel_size=downscale, stride=downscale)
    kernel = torch.tensor(_LAPLACIAN_KERNEL, dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    # Replicate-pad (not conv2d's default zero-pad) so a real image doesn't
    # get a fake high-frequency edge at the border from an artificial jump
    # to 0 — that would otherwise dominate the variance on a small
    # downsampled grid and inflate scores for flat/dark-bordered frames.
    gray_padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    lap = F.conv2d(gray_padded, kernel, padding=0)
    return lap.var(dim=(1, 2, 3))


def compute_sharpness(frame_hwc: np.ndarray, device: str = "cpu") -> float:
    """Single-frame convenience wrapper (used by the dashboard's "current frame"
    panel, which shows one frame's score as it's decoded)."""
    import torch

    t = torch.from_numpy(frame_hwc).unsqueeze(0).to(device)
    return float(_laplacian_variance_batch(t).item())


def compute_sharpness_batch(frames_nhwc: np.ndarray, device: str = "cpu") -> np.ndarray:
    import torch

    t = torch.from_numpy(frames_nhwc).to(device)
    return _laplacian_variance_batch(t).cpu().numpy()


def _haversine_like_enu_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def select_keyframes(
    frame_indices: list[int],
    frame_timestamps: list[float],
    frames_hwc: list[np.ndarray],
    gps_enu_per_frame: list[tuple[float, float, float]] | None,
    bus: EventBus,
    sharpness_percentile_floor: float = 15.0,
    min_gps_spacing_m: float = 2.0,
    max_no_gps_frame_stride: int = 5,
    device: str = "cpu",
) -> KeyframeSelection:
    """Two-stage selection:

    1. Sharpness gate: reject frames scoring below `sharpness_percentile_floor`
       of the batch's own sharpness distribution (adapts to the video's overall
       focus quality instead of a fixed threshold).
    2. Spacing gate: among sharpness-surviving frames, greedily keep a frame
       only if it's at least `min_gps_spacing_m` from the last *kept* frame
       (by GPS ENU distance, for target view overlap), or — with no GPS —
       every `max_no_gps_frame_stride`'th surviving frame.

    Every rejection is recorded with a reason so the dashboard's filmstrip can
    grey out rejected frames instead of silently dropping them.
    """
    n = len(frame_indices)
    if n == 0:
        return KeyframeSelection()

    sharpness = compute_sharpness_batch(np.stack(frames_hwc, axis=0), device=device)
    floor = float(np.percentile(sharpness, sharpness_percentile_floor))

    selection = KeyframeSelection()
    last_kept_gps = None
    kept_since_gps_none = 0

    for i in range(n):
        gps = gps_enu_per_frame[i] if gps_enu_per_frame is not None else None
        cand = KeyframeCandidate(
            frame_index=frame_indices[i], timestamp_s=frame_timestamps[i],
            sharpness=float(sharpness[i]), accepted=False, gps_enu=gps,
        )

        if sharpness[i] < floor:
            cand.reject_reason = f"sharpness {sharpness[i]:.1f} below floor {floor:.1f}"
            selection.rejected.append(cand)
            continue

        if gps is not None:
            if last_kept_gps is not None and _haversine_like_enu_distance(gps, last_kept_gps) < min_gps_spacing_m:
                cand.reject_reason = f"GPS spacing < {min_gps_spacing_m}m from last keyframe"
                selection.rejected.append(cand)
                continue
            last_kept_gps = gps
        else:
            kept_since_gps_none += 1
            if kept_since_gps_none % max_no_gps_frame_stride != 1 and n > max_no_gps_frame_stride:
                cand.reject_reason = "frame-stride spacing (no GPS available)"
                selection.rejected.append(cand)
                continue

        cand.accepted = True
        selection.accepted.append(cand)

    bus.log(
        f"Keyframe selection: {len(selection.accepted)} accepted, {len(selection.rejected)} rejected "
        f"(sharpness floor={floor:.1f}, from {n} candidates)"
    )
    return selection

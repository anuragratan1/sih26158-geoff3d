"""In-memory artifacts the pipeline accumulates as it runs, beyond what it
publishes as transient events — the per-stage notebook result cells and the
live reconstruction page both read this directly (not by replaying events,
not by re-parsing report.json) since it's already sitting in the same
process. Bounded/capped where it could otherwise grow unboundedly over a
FULL-mode run (e.g. only a few representative chunks' depth/confidence/mask
samples are kept, not every chunk's).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KeyframeRecord:
    frame_index: int
    timestamp_s: float
    thumbnail: np.ndarray | None  # small RGB, for the keyframe grid
    sharpness: float
    accepted: bool
    reject_reason: str | None = None


@dataclass
class ChunkAlignmentRecord:
    chunk_idx: int
    mode: str
    rmse_before_m: float | None       # this chunk's own direct-fit residual
    rmse_after_m: float | None = None  # after global pose-graph refinement


@dataclass
class ChunkGeometrySample:
    chunk_idx: int
    rgb: np.ndarray | None            # (H,W,3) uint8, one representative view
    depth: np.ndarray | None          # (H,W) float, camera-space z
    confidence: np.ndarray | None     # (H,W) float in [0,1]
    dynamic_mask: np.ndarray | None   # (H,W) bool, True = excluded (masked)


@dataclass
class RunArtifacts:
    keyframes: list[KeyframeRecord] = field(default_factory=list)
    gps_track_enu: list[tuple[float, float, float]] = field(default_factory=list)
    camera_track_enu: list[tuple[float, float, float]] = field(default_factory=list)
    collinearity_index: float | None = None
    chunk_alignments: list[ChunkAlignmentRecord] = field(default_factory=list)
    geometry_samples: list[ChunkGeometrySample] = field(default_factory=list)  # capped, see pipeline.py
    point_count: int = 0
    point_cloud_preview: np.ndarray | None = None   # (<=50k, 3) downsampled, for quick matplotlib renders
    point_cloud_preview_colors: np.ndarray | None = None
    mesh_n_vertices: int = 0
    mesh_n_faces: int = 0
    mesh_method: str = "none"
    output_paths: dict[str, str] = field(default_factory=dict)  # name -> absolute path, once export.py finishes
    georeferenced: bool = False

    MAX_GEOMETRY_SAMPLES = 6

    def add_geometry_sample(self, sample: ChunkGeometrySample) -> None:
        if len(self.geometry_samples) < self.MAX_GEOMETRY_SAMPLES:
            self.geometry_samples.append(sample)

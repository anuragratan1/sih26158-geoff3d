"""Geometry backbone config switch (Options A/B/C) — see PHASE0_NOTES.md.

Verified against actual source (GeoFF3D, MapAnything, UAVFF3D), not guessed:

- **A (GeoFF3D+SLRF)**: not viable. No trained GeoFF3D checkpoint is
  downloadable anywhere; training one is out of budget. `BackboneA.load()`
  raises a clear error rather than pretending to work.
- **B (Pi3X)**: loaded directly via `Pi3X.from_pretrained("yyfz233/Pi3X")`,
  vendored from GeoFF3D (`sih3d/vendor/pi3/`, Apache-2.0) with import paths
  rewritten. We do NOT go through GeoFF3D's SLRF orchestration — SLRF forces
  `load_pretrained_weights=false` and only loads weights via a raw
  `state_dict` load keyed to `Pi3XWrapper`'s `model.*`-prefixed submodule,
  a real footgun (see PHASE0_NOTES.md §6). We do our own chunking, same as C.
- **C (MapAnything Apache)**: `MapAnything.from_pretrained("facebook/map-anything-apache")`
  (pip-installed `mapanything` package, not vendored — Phase 0 confirmed a
  clean install with no torch/CUDA pin). Our own chunking.

Both B and C support loading UAVFF3D fine-tuned checkpoints
(github.com/yanxian-ll/UAVFF3D) auto-detected from /kaggle/input by
io_detect.find_checkpoints(), verified in mapanything/scripts/convert_hf_to_benchmark_checkpoint.py
to be saved as `{"model": <raw state_dict>}` — plain keys, no wrapper
prefix. `strict_load_checkpoint()` never silently drops mismatched keys
(unlike the reference benchmark's own loader, which calls
`load_state_dict(..., strict=False)`); it fails loudly with the exact
missing/unexpected key lists.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .events import EventBus

if TYPE_CHECKING:
    from .io_detect import DetectedInputs


class BackboneChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class PriorMode(str, Enum):
    """Matches UAVFF3D's own evaluation-protocol naming (RGB/C/P/CP)."""

    RGB = "RGB"                 # images only
    INTRINSICS = "C"            # + camera intrinsics
    POSES = "P"                 # + camera poses
    INTRINSICS_POSES = "CP"     # + both


@dataclass
class ViewInput:
    image: np.ndarray                       # HWC, uint8 or float32 [0,1]
    intrinsics: np.ndarray | None = None    # 3x3, pixel coords
    camera_pose_c2w: np.ndarray | None = None  # 4x4 OpenCV cam2world, from GPS translation + attitude prior
    frame_index: int = 0
    timestamp_s: float = 0.0


@dataclass
class ChunkResult:
    points_world: np.ndarray        # (N,H,W,3)
    confidence: np.ndarray          # (N,H,W)
    camera_poses_est: np.ndarray    # (N,4,4) cam2world, as predicted by the backbone
    metric_scaling_factor: float | None = None
    backbone_used: str = ""
    prior_mode: str = ""
    checkpoint_source: str = "pretrained"   # "pretrained" | "finetuned:<filename>"
    timing_s: float = 0.0


class BackboneNotAvailableError(RuntimeError):
    """Raised when a config-selected backbone cannot actually run."""


def _to_chw_float(img: np.ndarray):
    import torch

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    t = torch.from_numpy(np.ascontiguousarray(img)).float()
    if t.ndim == 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1)
    return t


def strict_load_checkpoint(model, ckpt_path: Path, bus: EventBus, label: str) -> str:
    """Load a checkpoint's state dict with strict key matching.

    Never silently drops mismatched keys the way the reference SLRF/UAVFF3D
    benchmark loaders do (they call `load_state_dict(..., strict=False)` and
    only print the result). Tries the state dict as-is first; if that fails,
    tries mechanically stripping or adding a "model." prefix — the one
    key-naming variant actually observed across GeoFF3D/UAVFF3D checkpoints
    (see PHASE0_NOTES.md §6) — and logs which one worked. Raises
    BackboneNotAvailableError with the missing/unexpected key lists if none
    of these match cleanly.
    """
    import torch

    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    if not isinstance(state, dict):
        raise BackboneNotAvailableError(f"{label}: checkpoint at {ckpt_path} has no usable state dict")

    own_keys = set(model.state_dict().keys())

    def _matches(candidate: dict) -> bool:
        missing = own_keys - set(candidate.keys())
        unexpected = set(candidate.keys()) - own_keys
        return not missing and not unexpected

    if _matches(state):
        model.load_state_dict(state, strict=True)
        bus.log(f"{label}: checkpoint keys matched exactly ({ckpt_path.name})")
        return "exact"

    stripped = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    if stripped and _matches(stripped):
        model.load_state_dict(stripped, strict=True)
        bus.log(f"{label}: checkpoint keys matched after stripping 'model.' prefix ({ckpt_path.name})")
        return "stripped_prefix"

    prefixed = {f"model.{k}": v for k, v in state.items()}
    if _matches(prefixed):
        model.load_state_dict(prefixed, strict=True)
        bus.log(f"{label}: checkpoint keys matched after adding 'model.' prefix ({ckpt_path.name})")
        return "added_prefix"

    ckpt_keys = set(state.keys())
    missing = sorted(own_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - own_keys)
    raise BackboneNotAvailableError(
        f"{label}: checkpoint at {ckpt_path} does not match the model architecture "
        f"(tried as-is and with 'model.' prefix stripped/added). "
        f"{len(missing)} missing keys (e.g. {missing[:5]}), "
        f"{len(unexpected)} unexpected keys (e.g. {unexpected[:5]}). "
        f"Refusing to silently load with strict=False."
    )


class GeometryBackbone:
    name: str = "?"

    def load(self) -> None:
        raise NotImplementedError

    def infer_chunk(self, views: list[ViewInput], prior_mode: PriorMode) -> ChunkResult:
        raise NotImplementedError


class BackboneA(GeometryBackbone):
    """GeoFF3D + SLRF. Not viable — see module docstring and PHASE0_NOTES.md §1/§6."""

    name = "A"

    def load(self) -> None:
        raise BackboneNotAvailableError(
            "Backbone A (GeoFF3D+SLRF) has no publicly downloadable trained checkpoint. "
            "Verified by reading the GeoFF3D source (bash_scripts/run_slrf/geoff3d.sh defaults "
            "to a checkpoint path only produced by running the two-stage training pipeline "
            "yourself — see PHASE0_NOTES.md §1). Training one from scratch is out of budget. "
            "Select BACKBONE='B' or BACKBONE='C' instead."
        )

    def infer_chunk(self, views: list[ViewInput], prior_mode: PriorMode) -> ChunkResult:
        raise BackboneNotAvailableError("Backbone A was never loaded (see load()).")


class BackboneB(GeometryBackbone):
    """Pi3X, loaded directly (no SLRF), our own chunking."""

    name = "B"

    def __init__(self, bus: EventBus, device: str = "cuda:0", checkpoint_path: Path | None = None):
        self.bus = bus
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.dtype = None
        self.checkpoint_source = "pretrained"

    def load(self) -> None:
        import torch

        from .vendor.pi3.models.pi3x import Pi3X

        self.bus.log("Backbone B: loading Pi3X (yyfz233/Pi3X) directly, no SLRF")
        t0 = time.time()
        self.model = Pi3X.from_pretrained("yyfz233/Pi3X")
        if self.checkpoint_path is not None:
            strict_load_checkpoint(self.model, self.checkpoint_path, self.bus, "Backbone B (Pi3X)")
            self.checkpoint_source = f"finetuned:{self.checkpoint_path.name}"
        self.model.to(self.device)
        self.model.eval()
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(self.device)
            self.dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        self.bus.log(
            f"Backbone B: Pi3X loaded in {time.time() - t0:.1f}s "
            f"(checkpoint={self.checkpoint_source}, dtype={self.dtype})"
        )

    def infer_chunk(self, views: list[ViewInput], prior_mode: PriorMode) -> ChunkResult:
        import torch

        assert self.model is not None, "call load() first"
        t0 = time.time()
        device = self.device

        imgs = torch.stack([_to_chw_float(v.image) for v in views], dim=0).unsqueeze(0).to(device)

        use_intrinsics = prior_mode in (PriorMode.INTRINSICS, PriorMode.INTRINSICS_POSES) and all(
            v.intrinsics is not None for v in views
        )
        use_poses = prior_mode in (PriorMode.POSES, PriorMode.INTRINSICS_POSES) and all(
            v.camera_pose_c2w is not None for v in views
        )

        intrinsics = None
        if use_intrinsics:
            intrinsics = torch.stack(
                [torch.from_numpy(v.intrinsics).float() for v in views], dim=0
            ).unsqueeze(0).to(device)

        poses = None
        pose_mask = None
        if use_poses:
            poses = torch.stack(
                [torch.from_numpy(v.camera_pose_c2w).float() for v in views], dim=0
            ).unsqueeze(0).to(device)
            pose_mask = torch.ones(1, len(views), dtype=torch.bool, device=device)

        with torch.autocast(
            device_type="cuda" if str(device).startswith("cuda") else "cpu",
            dtype=self.dtype, enabled=str(device).startswith("cuda"),
        ):
            with torch.no_grad():
                out = self.model(
                    imgs=imgs, intrinsics=intrinsics, poses=poses, pose_mask=pose_mask,
                    with_prior=True, overall_prob=1.0,
                    ray_dirs_prob=1.0 if use_intrinsics else 0.0,
                    cam_prob=1.0 if use_poses else 0.0,
                )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

        points = out["points"][0].float().cpu().numpy()
        conf = out["conf"][0, ..., 0].float().cpu().numpy()
        cam_poses = out["camera_poses"][0].float().cpu().numpy()
        metric = float(out["metric"][0].item())

        return ChunkResult(
            points_world=points, confidence=conf, camera_poses_est=cam_poses,
            metric_scaling_factor=metric, backbone_used="B", prior_mode=prior_mode.value,
            checkpoint_source=self.checkpoint_source, timing_s=time.time() - t0,
        )


class BackboneC(GeometryBackbone):
    """MapAnything (Apache), our own chunking."""

    name = "C"

    def __init__(
        self, bus: EventBus, device: str = "cuda:0", checkpoint_path: Path | None = None,
        hf_repo: str = "facebook/map-anything-apache",
    ):
        self.bus = bus
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.hf_repo = hf_repo
        self.model = None
        self.checkpoint_source = "pretrained"

    def load(self) -> None:
        from mapanything.models import MapAnything

        self.bus.log(f"Backbone C: loading MapAnything from {self.hf_repo}")
        t0 = time.time()
        self.model = MapAnything.from_pretrained(self.hf_repo)
        if self.checkpoint_path is not None:
            strict_load_checkpoint(self.model, self.checkpoint_path, self.bus, "Backbone C (MapAnything)")
            self.checkpoint_source = f"finetuned:{self.checkpoint_path.name}"
        self.model.to(self.device)
        self.model.eval()
        self.bus.log(
            f"Backbone C: MapAnything loaded in {time.time() - t0:.1f}s "
            f"(checkpoint={self.checkpoint_source})"
        )

    def infer_chunk(self, views: list[ViewInput], prior_mode: PriorMode) -> ChunkResult:
        import torch

        assert self.model is not None, "call load() first"
        t0 = time.time()
        device = self.device

        use_intrinsics = prior_mode in (PriorMode.INTRINSICS, PriorMode.INTRINSICS_POSES)
        use_poses = prior_mode in (PriorMode.POSES, PriorMode.INTRINSICS_POSES)

        mv_views: list[dict[str, Any]] = []
        for v in views:
            img = _to_chw_float(v.image).unsqueeze(0).to(device)
            entry: dict[str, Any] = {"img": img, "data_norm_type": ["dinov2"]}
            if use_intrinsics and v.intrinsics is not None:
                entry["intrinsics"] = torch.from_numpy(v.intrinsics).float().unsqueeze(0).to(device)
            if use_poses and v.camera_pose_c2w is not None:
                entry["camera_poses"] = torch.from_numpy(v.camera_pose_c2w).float().unsqueeze(0).to(device)
                entry["is_metric_scale"] = True
            mv_views.append(entry)

        with torch.no_grad():
            preds = self.model.infer(
                mv_views, memory_efficient_inference=True, use_amp=True, amp_dtype="fp16",
            )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

        points = np.stack([p["pts3d"][0].float().cpu().numpy() for p in preds], axis=0)
        conf = np.stack([p["conf"][0].float().cpu().numpy() for p in preds], axis=0)
        cam_poses = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in preds], axis=0)
        metric = (
            float(preds[0]["metric_scaling_factor"][0].item())
            if "metric_scaling_factor" in preds[0] else None
        )

        return ChunkResult(
            points_world=points, confidence=conf, camera_poses_est=cam_poses,
            metric_scaling_factor=metric, backbone_used="C", prior_mode=prior_mode.value,
            checkpoint_source=self.checkpoint_source, timing_s=time.time() - t0,
        )


_CHECKPOINT_KEY_BY_BACKBONE = {"B": "pi3x", "C": "mapanything"}


def select_backbone(
    choice: str,
    prior_mode: str,
    detected: "DetectedInputs",
    bus: EventBus,
    device: str = "cuda:0",
    use_finetuned: bool = True,
) -> tuple[GeometryBackbone, PriorMode]:
    """Resolve BACKBONE/PRIOR_MODE config, apply no-GPS auto-select, load, return.

    No-GPS mode (task spec): if no telemetry was found (sidecar file or
    embedded stream), force Backbone C with RGB-only priors — MapAnything
    predicts metric scale from images alone; feeding fabricated pose/
    intrinsics priors without real GPS would be worse than none. Chunks are
    aligned to each other only via align.py's Sim(3)-between-chunks
    fallback, and every output gets labeled "APPROXIMATE SCALE — NOT
    GEOREFERENCED" upstream in export.py/report.py/viewer.html.

    prior_mode="AUTO" resolves to CP (intrinsics+poses) when a fine-tuned
    checkpoint is present (per the UAVFF3D authors' own eval protocol, fine-
    tuning targets the CP setting) and telemetry exists, else C
    (intrinsics-only, still useful without pose priors that may hurt dense
    geometry on UAV data per the task's own note — compare via an explicit
    PRIOR_MODE instead of AUTO when benchmarking).
    """
    has_telemetry = detected.telemetry_path is not None or detected.telemetry_kind == "embedded"

    if not has_telemetry:
        if choice != "C" or prior_mode not in ("RGB", "AUTO"):
            bus.log(
                f"No telemetry detected: forcing BACKBONE=C, PRIOR_MODE=RGB (no-GPS mode). "
                f"Outputs will be APPROXIMATE SCALE — NOT GEOREFERENCED.",
                level="warn",
            )
        resolved_choice = "C"
        resolved_prior = PriorMode.RGB
        checkpoint_path = None
    else:
        resolved_choice = choice
        checkpoint_key = _CHECKPOINT_KEY_BY_BACKBONE.get(resolved_choice)
        checkpoint_path = detected.checkpoints.get(checkpoint_key) if (use_finetuned and checkpoint_key) else None
        if prior_mode == "AUTO":
            resolved_prior = PriorMode.INTRINSICS_POSES if checkpoint_path is not None else PriorMode.INTRINSICS
        else:
            resolved_prior = PriorMode(prior_mode)

    if resolved_choice == "A":
        backbone: GeometryBackbone = BackboneA()
    elif resolved_choice == "B":
        backbone = BackboneB(bus, device=device, checkpoint_path=checkpoint_path)
    elif resolved_choice == "C":
        backbone = BackboneC(bus, device=device, checkpoint_path=checkpoint_path)
    else:
        raise ValueError(f"Unknown backbone choice: {resolved_choice!r} (expected A, B, or C)")

    backbone.load()
    return backbone, resolved_prior

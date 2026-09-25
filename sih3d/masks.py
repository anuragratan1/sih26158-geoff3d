"""Dynamic-object masking: a small fp16 segmentation model (YOLO-seg) flags
vehicles/people/animals so fusion.py can exclude those pixels from the point
cloud (moving objects corrupt multi-view fusion — they aren't in the same
place across views the way static scene geometry is).

Designed to run on GPU1 while GPU0 runs the geometry backbone on the next
chunk (see the pipeline's two-GPU producer/consumer wiring) — this module
only does the segmentation itself; the threading/queueing lives in the
pipeline orchestration, not here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .events import EventBus

# COCO class indices YOLO-seg models are typically trained on, restricted to
# things that actually move in drone footage.
_DYNAMIC_CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train", "boat",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
}


@dataclass
class MaskResult:
    mask: np.ndarray            # (H,W) bool, True = dynamic/exclude-from-fusion
    detections: int
    backend: str
    timing_s: float


class DynamicObjectMasker:
    """Lazily loads a YOLO-seg model on first use (so importing this module
    never requires ultralytics/torch until masking is actually enabled).
    Falls back to an all-static (empty) mask, logged once, if the model
    can't be loaded — dynamic-object exclusion is a quality improvement,
    not a hard requirement, so this must never crash the pipeline."""

    def __init__(self, bus: EventBus, device: str = "cuda:1", model_name: str = "yolo11n-seg.pt", confidence: float = 0.35):
        self.bus = bus
        self.device = device
        self.model_name = model_name
        self.confidence = confidence
        self.model = None
        self.backend = "none"
        self._load_attempted = False

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            import logging

            from ultralytics import YOLO
            from ultralytics.utils import LOGGER as _ultralytics_logger

            # Ultralytics logs a "'half' is deprecated" warning through its
            # own LOGGER on every single predict() call (once per frame,
            # not once per process) — with masking running per-frame across
            # every chunk, that's hundreds of identical lines flooding the
            # notebook and making a live, progressing run look frozen. This
            # is purely the deprecation notice, not an error; ERROR level
            # still surfaces anything that actually matters.
            _ultralytics_logger.setLevel(logging.ERROR)

            t0 = time.time()
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            self.backend = "yolo-seg"
            self.bus.log(f"Dynamic-object masker: loaded {self.model_name} on {self.device} in {time.time() - t0:.1f}s")
        except Exception as e:
            self.backend = "none"
            self.bus.log(
                f"Dynamic-object masker unavailable ({type(e).__name__}: {e}) — "
                f"proceeding with no dynamic-object exclusion (all-static mask)",
                level="warn",
            )

    def mask_frame(self, frame_hwc: np.ndarray) -> MaskResult:
        self._ensure_loaded()
        t0 = time.time()
        h, w = frame_hwc.shape[:2]

        if self.model is None:
            return MaskResult(mask=np.zeros((h, w), dtype=bool), detections=0, backend="none", timing_s=time.time() - t0)

        try:
            results = self.model.predict(frame_hwc, verbose=False, conf=self.confidence, half=True, device=self.device)
        except Exception as e:
            self.bus.log(f"Dynamic-object mask inference failed on one frame ({e}); treating as all-static", level="warn")
            return MaskResult(mask=np.zeros((h, w), dtype=bool), detections=0, backend="error_fallback", timing_s=time.time() - t0)

        mask = np.zeros((h, w), dtype=bool)
        n_det = 0
        for r in results:
            if r.masks is None:
                continue
            names = r.names
            for seg_mask, cls_idx in zip(r.masks.data, r.boxes.cls):
                cls_name = names.get(int(cls_idx), "") if isinstance(names, dict) else str(names[int(cls_idx)])
                if cls_name not in _DYNAMIC_CLASS_NAMES:
                    continue
                m = seg_mask.detach().float().cpu().numpy()
                if m.shape != (h, w):
                    import cv2

                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask |= m > 0.5
                n_det += 1

        return MaskResult(mask=mask, detections=n_det, backend=self.backend, timing_s=time.time() - t0)

    def mask_batch(self, frames_hwc: list[np.ndarray]) -> list[MaskResult]:
        return [self.mask_frame(f) for f in frames_hwc]


# ADE20K class names (the training set for the SegFormer checkpoint below)
# that are never part of the physical structure being reconstructed and are
# exactly the classes that corrupt multi-view fusion the same way moving
# objects do: water/sky have no stable, view-consistent depth (water is
# reflective/textureless — multi-view matching either fails outright or
# returns noisy, inconsistent depth; sky is at infinity and shouldn't
# produce any 3D points at all). This is what was actually producing the
# spiky/hallucinated Poisson artifacts over water in a real run — masking
# these pixels out BEFORE they ever enter the fused point cloud fixes the
# problem at its source, instead of trying to clean up Poisson's output
# after the fact (density trimming can't tell "isolated noise" apart from
# "real but sparser surface" — see mesh.py's build_vertex_colored_mesh).
_IRRELEVANT_SEMANTIC_LABELS = {"water", "sea", "river", "lake", "swimming pool", "sky"}


class SemanticMasker:
    """SegFormer semantic segmentation (ADE20K-trained) to mask out
    water/sky before fusion, alongside DynamicObjectMasker's moving-object
    masking — both feed the same `dynamic_mask` exclusion mechanism in
    fusion.py, just for different reasons (moving vs. never-should-be-
    reconstructed). Lazily loaded, same resilience contract as
    DynamicObjectMasker: never crashes the pipeline, degrades to an
    all-clear mask (logged once) if transformers/the checkpoint aren't
    available."""

    def __init__(self, bus: EventBus, device: str = "cuda:1", model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
        self.bus = bus
        self.device = device
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.irrelevant_ids: set[int] = set()
        self.backend = "none"
        self._load_attempted = False

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

            t0 = time.time()
            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name).to(self.device).eval()
            self.irrelevant_ids = {
                int(i) for i, name in self.model.config.id2label.items()
                if name.lower() in _IRRELEVANT_SEMANTIC_LABELS
            }
            self.backend = "segformer"
            self.bus.log(
                f"Semantic masker: loaded {self.model_name} on {self.device} in {time.time() - t0:.1f}s "
                f"(masking classes: {sorted(self.model.config.id2label[i] for i in self.irrelevant_ids)})"
            )
        except Exception as e:
            self.backend = "none"
            self.bus.log(
                f"Semantic masker unavailable ({type(e).__name__}: {e}) — "
                f"proceeding with no water/sky exclusion",
                level="warn",
            )

    def mask_frame(self, frame_hwc: np.ndarray) -> MaskResult:
        self._ensure_loaded()
        t0 = time.time()
        h, w = frame_hwc.shape[:2]

        if self.model is None:
            return MaskResult(mask=np.zeros((h, w), dtype=bool), detections=0, backend="none", timing_s=time.time() - t0)

        try:
            import torch
            import torch.nn.functional as F

            inputs = self.processor(images=frame_hwc, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits  # (1, num_classes, h/4, w/4)
            upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            pred = upsampled.argmax(dim=1)[0].cpu().numpy()
        except Exception as e:
            self.bus.log(f"Semantic mask inference failed on one frame ({e}); treating as no exclusion", level="warn")
            return MaskResult(mask=np.zeros((h, w), dtype=bool), detections=0, backend="error_fallback", timing_s=time.time() - t0)

        mask = np.isin(pred, list(self.irrelevant_ids)) if self.irrelevant_ids else np.zeros((h, w), dtype=bool)
        return MaskResult(mask=mask, detections=int(mask.any()), backend=self.backend, timing_s=time.time() - t0)

    def mask_batch(self, frames_hwc: list[np.ndarray]) -> list[MaskResult]:
        return [self.mask_frame(f) for f in frames_hwc]

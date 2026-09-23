"""GPU-first video decode with a CPU fallback chain.

Priority: torchcodec CUDA decode -> PyNvVideoCodec -> ffmpeg -hwaccel cuda
-> plain CPU decode (OpenCV/PyAV). Every fallback is logged. Decoding is
lazy/on-demand: callers ask for specific frame indices or a time range
(the keyframe selector decides which frames actually get pulled), so we
never decode the whole video into memory up front.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .events import EventBus, EventType


@dataclass
class DecoderInfo:
    backend: str  # "torchcodec" | "pynvvideocodec" | "ffmpeg_cuda" | "cpu"
    device: str


class FrameDecoder:
    """Yields (frame_index, timestamp_s, HWC uint8 RGB ndarray) tuples.

    Construction probes backends in priority order and logs which one is
    active. `iter_frames(start_s, end_s, stride)` streams frames rather
    than materializing the whole range.
    """

    def __init__(self, video_path: Path, bus: EventBus, device: str = "cuda:0"):
        self.video_path = Path(video_path)
        self.bus = bus
        self.device = device
        self.info = self._select_backend()
        self.bus.log(f"Video decoder backend: {self.info.backend} on {self.info.device}")

    def _select_backend(self) -> DecoderInfo:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA device")
            import torchcodec  # noqa: F401
            return DecoderInfo(backend="torchcodec", device=self.device)
        except Exception as e:
            self.bus.log(f"torchcodec CUDA decode unavailable ({e}); trying PyNvVideoCodec", level="warn")

        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA device")
            import PyNvVideoCodec  # noqa: F401
            return DecoderInfo(backend="pynvvideocodec", device=self.device)
        except Exception as e:
            self.bus.log(f"PyNvVideoCodec unavailable ({e}); trying ffmpeg -hwaccel cuda", level="warn")

        if self._ffmpeg_has_cuda_hwaccel():
            return DecoderInfo(backend="ffmpeg_cuda", device=self.device)

        self.bus.log("No GPU decode path available; falling back to CPU decode", level="warn")
        return DecoderInfo(backend="cpu", device="cpu")

    def _ffmpeg_has_cuda_hwaccel(self) -> bool:
        import subprocess
        try:
            out = subprocess.run(["ffmpeg", "-hwaccels"], capture_output=True, text=True, timeout=10)
            return "cuda" in out.stdout.lower()
        except Exception:
            return False

    def iter_frames(self, start_s: float = 0.0, end_s: float | None = None, stride: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
        if self.info.backend == "torchcodec":
            yield from self._iter_torchcodec(start_s, end_s, stride)
        elif self.info.backend == "pynvvideocodec":
            yield from self._iter_pynvvideocodec(start_s, end_s, stride)
        elif self.info.backend == "ffmpeg_cuda":
            yield from self._iter_ffmpeg(start_s, end_s, stride, hwaccel=True)
        else:
            yield from self._iter_ffmpeg(start_s, end_s, stride, hwaccel=False)

    def _iter_torchcodec(self, start_s, end_s, stride):
        from torchcodec.decoders import VideoDecoder

        dec = VideoDecoder(str(self.video_path), device=self.device)
        n = len(dec)
        meta = dec.metadata
        fps = meta.average_fps or 30.0
        start_idx = int(start_s * fps)
        end_idx = int(end_s * fps) if end_s is not None else n
        for i in range(start_idx, min(end_idx, n), stride):
            frame = dec[i]
            arr = frame.data.permute(1, 2, 0).clamp(0, 255).to("cpu", dtype=__import__("torch").uint8).numpy()
            yield i, i / fps, arr

    def _iter_pynvvideocodec(self, start_s, end_s, stride):
        # PyNvVideoCodec's API varies by version; fall back to ffmpeg CUDA
        # if the demuxer/decoder objects aren't importable/usable at runtime.
        try:
            yield from self._iter_ffmpeg(start_s, end_s, stride, hwaccel=True)
        except Exception as e:
            self.bus.log(f"PyNvVideoCodec path failed at runtime ({e}); falling back to CPU decode", level="warn")
            yield from self._iter_ffmpeg(start_s, end_s, stride, hwaccel=False)

    def _iter_ffmpeg(self, start_s, end_s, stride, hwaccel: bool):
        import subprocess

        probe = self._probe_dims()
        w, h, fps = probe
        cmd = ["ffmpeg", "-v", "error"]
        if hwaccel:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        cmd += ["-ss", str(start_s), "-i", str(self.video_path)]
        if end_s is not None:
            cmd += ["-t", str(max(0.0, end_s - start_s))]
        vf = f"select='not(mod(n\\,{stride}))'" if stride > 1 else None
        if hwaccel:
            filt = "hwdownload,format=nv12" + (f",{vf}" if vf else "")
        else:
            filt = vf
        if filt:
            cmd += ["-vf", filt]
        cmd += ["-pix_fmt", "rgb24", "-f", "rawvideo", "-"]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_bytes = w * h * 3
        idx = int(start_s * fps)
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
                yield idx, idx / fps, arr
                idx += stride
        finally:
            proc.stdout.close()
            proc.wait(timeout=10)
            if proc.returncode not in (0, None) and proc.returncode != 0:
                err = proc.stderr.read().decode("utf-8", "ignore")[-2000:]
                if hwaccel:
                    self.bus.log(f"ffmpeg CUDA decode failed ({err.strip()[-300:]}); retrying with CPU decode", level="warn")
                    yield from self._iter_ffmpeg(start_s, end_s, stride, hwaccel=False)

    def _probe_dims(self) -> tuple[int, int, float]:
        from .io_detect import probe_video

        vi = probe_video(self.video_path)
        if vi is None:
            raise RuntimeError(f"ffprobe could not read {self.video_path}")
        return vi.width, vi.height, vi.fps or 30.0

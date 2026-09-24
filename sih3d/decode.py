"""GPU-first video decode with a CPU fallback chain.

Priority: PyNvVideoCodec sequential NVDEC -> torchcodec CUDA decode -> ffmpeg -hwaccel cuda
-> plain CPU decode (OpenCV/PyAV). Every fallback is logged. Decoding is
lazy/on-demand: callers provide a time range and the decoder streams it in
small sequential batches, sampling on the GPU before any selected frame is
copied back to host memory.  It never materializes a whole video up front.

Three real Kaggle runs (2x T4, 4K source) narrowed this down progressively:
(1) torchcodec claimed cuda:0 but the PyPI wheel silently decoded on CPU —
fixed by actually decoding a probe and checking the tensor's device+fps,
not just that the import succeeded. (2) `ffmpeg -hwaccels` listing "cuda"
only means ffmpeg was compiled with CUDA support, not that NVDEC engages at
runtime — same fix, verify by decoding a real probe. (3) Even once ffmpeg
decode itself was confirmed, the working-resolution downscale used
`scale_cuda`/npp in the filter graph, which isn't reliably available in
every ffmpeg build (and wasn't the thing actually decoding — see below);
removed that dependency entirely in favor of PyNvVideoCodec (NVIDIA's
direct NVDEC binding, decodes straight to a GPU tensor, no ffmpeg
subprocess/pipe boundary at all) as the primary GPU path, with the
downscale done via torch on the GPU right after decode, and ffmpeg
`-hwaccel cuda` demoted to decode-only fallback (plain software `scale=`
after hwaccel's automatic frame download, no cuda-specific filters).
PyNvVideoCodec's exact API has shifted across versions and isn't
verifiable against real NVIDIA hardware in this dev environment — it's
implemented best-effort and, like every other backend here, gated behind
an actual decode-and-measure check before being trusted, so a wrong guess
just falls through to the next backend rather than crashing.
"""

from __future__ import annotations

import threading
import time
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
    active. `iter_frames(start_s, end_s, target_fps, scale_width)` streams
    frames rather than materializing the whole range.
    """

    def __init__(self, video_path: Path, bus: EventBus, device: str = "cuda:0"):
        self.video_path = Path(video_path)
        self.bus = bus
        self.device = device
        self.info = self._select_backend()
        self.bus.log(f"Video decoder backend: {self.info.backend} on {self.info.device}")

    def _select_backend(self) -> DecoderInfo:
        if self._verify_pynvvideocodec_gpu():
            return DecoderInfo(backend="pynvvideocodec", device=self.device)

        torchcodec_info = self._verify_torchcodec_gpu()
        if torchcodec_info is not None:
            return torchcodec_info

        if self._ffmpeg_has_cuda_hwaccel() and self._verify_ffmpeg_cuda():
            return DecoderInfo(backend="ffmpeg_cuda", device=self.device)

        self.bus.log("No GPU decode path available; falling back to CPU decode", level="warn")
        return DecoderInfo(backend="cpu", device="cpu")

    # A 4K RGB frame is about 24 MB.  Keep this modest: it is large enough
    # to amortize Python/NVDEC calls, but does not make a transient multi-GB
    # stack while the reconstruction model is resident on a 16 GB T4.
    _NVDEC_BATCH_4K = 8
    _NVDEC_BATCH_HD = 24

    def _make_pynvvideocodec_decoder(self, nvc, dev_id: int, *, planar: bool = True):
        """Create a device-memory decoder across the two published APIs.

        RGBP is deliberate: it imports as a CHW PyTorch tensor through
        DLPack, avoiding both a colour conversion and a layout copy before
        batched sampling/resizing below.
        """
        color = getattr(nvc.OutputColorType, "RGBP", nvc.OutputColorType.RGB) if planar else nvc.OutputColorType.RGB
        try:
            return nvc.SimpleDecoder(
                str(self.video_path), gpu_id=dev_id, use_device_memory=True,
                output_color_type=color,
            )
        except TypeError:
            # Older releases used cuda_device_id, but still support device
            # memory and the same sequential-batch API when available.
            return nvc.SimpleDecoder(
                str(self.video_path), cuda_device_id=dev_id,
                use_device_memory=True, output_color_type=color,
            )

    def _pynvvideocodec_frame_to_hwc(self, frame, device: str):
        """Best-effort conversion of whatever PyNvVideoCodec hands back
        into an HWC uint8 torch tensor on `device`. Tries the access
        patterns documented across recent PyNvVideoCodec versions in
        order; the first one that doesn't raise wins. Unverified against
        real hardware — if none of these match the installed version,
        this raises and the caller (verification or iteration, both
        already wrapped) treats it as "this backend doesn't work here"."""
        import torch

        t = None
        # PyNvVideoCodec implements DLPack.  This is the only conversion
        # path which is explicitly zero-copy between NVDEC and PyTorch.
        if hasattr(frame, "__dlpack__"):
            t = torch.from_dlpack(frame)
        elif hasattr(frame, "cuda"):
            t = frame.cuda()  # some versions expose a torch-tensor-like accessor
        elif hasattr(frame, "torch"):
            t = frame.torch()
        elif hasattr(frame, "__cuda_array_interface__") or hasattr(frame, "__array_interface__"):
            t = torch.as_tensor(frame, device=device)
        else:
            t = torch.as_tensor(frame, device=device)
        t = t.to(device)
        if t.ndim == 3 and t.shape[0] in (3, 4):  # CHW -> HWC
            t = t[:3].permute(1, 2, 0)
        elif t.ndim == 3 and t.shape[-1] in (3, 4):
            t = t[..., :3]
        else:
            raise RuntimeError(f"unexpected PyNvVideoCodec frame shape {tuple(t.shape)}")
        return t

    def _pynvvideocodec_batch_to_nchw(self, frames, device: str):
        """Zero-copy-import NVDEC surfaces, then make one GPU batch.

        Individual surfaces own their decoder memory, so `stack` is needed
        to form an independent tensor.  It is still entirely GPU-to-GPU and
        lets the rest of the work happen once per batch rather than once per
        frame.
        """
        import torch

        tensors = [self._pynvvideocodec_frame_to_hwc(frame, device) for frame in frames]
        if not tensors:
            return None
        return torch.stack([t.permute(2, 0, 1) for t in tensors], dim=0)

    @staticmethod
    def _nvdec_batch_size(width: int, height: int) -> int:
        return FrameDecoder._NVDEC_BATCH_4K if max(width, height) >= 3000 else FrameDecoder._NVDEC_BATCH_HD

    def _verify_pynvvideocodec_gpu(self, n_probe: int = 30, min_fps_4k: float = 100.0, min_fps_other: float = 40.0) -> bool:
        """NVIDIA's official NVDEC binding — decodes straight into GPU
        memory with no ffmpeg subprocess/pipe boundary at all, so it
        should comfortably clear a higher fps floor than the ffmpeg paths
        (targeting >100fps at 4K on a T4's dedicated NVDEC engine, well
        past what a 4-vCPU Kaggle instance can do in software). See the
        module docstring for why this is best-effort/unverified."""
        try:
            import PyNvVideoCodec as nvc

            w, h, _ = self._probe_dims()
            min_fps = min_fps_4k if max(w, h) >= 3000 else min_fps_other
            dev_id = int(self.device.split(":")[-1]) if ":" in self.device else 0

            dec = self._make_pynvvideocodec_decoder(nvc, dev_id)
            n = min(n_probe, len(dec))
            if n == 0:
                return False

            t0 = time.time()
            decoded = 0
            while decoded < n:
                # get_batch_frames is PyNvVideoCodec's sequential fast path.
                # Calling dec[i] repeatedly is random access and can seek/
                # re-prime decode for every sampled frame.
                frames = dec.get_batch_frames(min(self._nvdec_batch_size(w, h), n - decoded))
                if not frames:
                    break
                batch = self._pynvvideocodec_batch_to_nchw(frames, self.device)
                decoded += len(frames)
                del batch
            import torch
            torch.cuda.synchronize(self.device)
            elapsed = time.time() - t0
            fps = decoded / elapsed if elapsed > 0 else 0.0

            if decoded == 0 or fps < min_fps:
                self.bus.log(
                    f"PyNvVideoCodec measured only {fps:.1f} fps over {decoded} sequential frames at {w}x{h} "
                    f"(below the {min_fps:.0f} fps floor) — trying ffmpeg -hwaccel cuda", level="warn",
                )
                return False

            self.bus.log(f"PyNvVideoCodec verified: {fps:.1f} fps over {decoded} sequential frames at {w}x{h}, on {self.device}")
            return True
        except Exception as e:
            import traceback
            err_msg = f"PyNvVideoCodec unavailable/failed ({type(e).__name__}: {e})\n{traceback.format_exc()}"
            with open("gpu_debug.log", "a") as f: f.write(f"[DEBUG] {err_msg}\n")
            self.bus.log(f"PyNvVideoCodec unavailable/failed ({type(e).__name__}: {e}) — trying ffmpeg -hwaccel cuda", level="warn")
            return False

    def _verify_ffmpeg_cuda(self, n_probe: int = 30, min_fps_4k: float = 60.0, min_fps_other: float = 24.0) -> bool:
        """`ffmpeg -hwaccels` listing "cuda" only means ffmpeg was compiled
        with CUDA hwaccel support — not that NVDEC actually engages at
        runtime for this codec/driver/container. ffmpeg can silently fall
        back to software decode on a hwaccel init failure rather than
        erroring, which looks identical to a slow-but-working GPU path
        from the caller's side (this is exactly what a real run showed:
        CPU pegged, GPU idle, frame extraction crawling, with the log
        confidently claiming "ffmpeg_cuda"). Same principle as
        _verify_torchcodec_gpu above: actually decode and measure, don't
        trust the capability check alone."""
        try:
            w, h, native_fps = self._probe_dims()
            min_fps = min_fps_4k if max(w, h) >= 3000 else min_fps_other
            probe_end_s = n_probe / max(native_fps, 1.0)

            t0 = time.time()
            count = 0
            for _ in self._iter_ffmpeg(0.0, probe_end_s, stride=1, hwaccel=True, allow_fallback=False):
                count += 1
                if count >= n_probe:
                    break
            elapsed = time.time() - t0
            fps = count / elapsed if elapsed > 0 else 0.0

            if count == 0:
                self.bus.log("ffmpeg -hwaccel cuda produced no frames during verification — using CPU decode instead", level="warn")
                return False
            if fps < min_fps:
                self.bus.log(
                    f"ffmpeg -hwaccel cuda measured only {fps:.1f} fps over {count} frames at {w}x{h} "
                    f"(below the {min_fps:.0f} fps floor — hwaccel likely silently fell back to software) — "
                    f"using CPU decode instead", level="warn",
                )
                return False

            self.bus.log(f"ffmpeg -hwaccel cuda verified: {fps:.1f} fps over {count} real frames at {w}x{h}")
            return True
        except Exception as e:
            import traceback
            err_msg = f"ffmpeg -hwaccel cuda verification failed ({e})\n{traceback.format_exc()}"
            with open("gpu_debug.log", "a") as f: f.write(f"[DEBUG] {err_msg}\n")
            self.bus.log(f"ffmpeg -hwaccel cuda verification failed ({e}) — using CPU decode instead", level="warn")
            return False

    def _make_torchcodec_decoder(self, VideoDecoder):
        """Prefer approximate seek: keyframe selection tolerates codec-level
        seek approximation and it avoids an expensive full index scan."""
        try:
            return VideoDecoder(str(self.video_path), device=self.device, seek_mode="approximate")
        except TypeError:
            return VideoDecoder(str(self.video_path), device=self.device)

    def _verify_torchcodec_gpu(self, n_probe: int = 30, min_fps_4k: float = 30.0, min_fps_other: float = 24.0) -> DecoderInfo | None:
        """Decodes a handful of real frames and checks BOTH that they
        actually land on the GPU and that throughput clears a resolution-
        scaled floor — never trust `import torchcodec` succeeding alone
        (see module docstring)."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            from torchcodec.decoders import VideoDecoder

            w, h, _ = self._probe_dims()
            min_fps = min_fps_4k if max(w, h) >= 3000 else min_fps_other

            dec = self._make_torchcodec_decoder(VideoDecoder)
            n = min(n_probe, len(dec))
            if n == 0:
                return None

            t0 = time.time()
            on_gpu = True
            for i in range(n):
                frame = dec[i]
                if not str(frame.data.device).startswith("cuda"):
                    on_gpu = False
            elapsed = time.time() - t0
            fps = n / elapsed if elapsed > 0 else 0.0

            if not on_gpu:
                self.bus.log(
                    f"torchcodec requested device={self.device} but decoded frames landed on CPU "
                    f"(the installed wheel is likely a CPU-only build) — trying PyNvVideoCodec", level="warn",
                )
                return None
            if fps < min_fps:
                self.bus.log(
                    f"torchcodec CUDA decode measured {fps:.1f} fps over {n} frames at {w}x{h} "
                    f"(below the {min_fps:.0f} fps floor) — trying ffmpeg -hwaccel cuda", level="warn",
                )
                return None

            self.bus.log(f"torchcodec CUDA decode verified: {fps:.1f} fps over {n} real frames at {w}x{h}, on {self.device}")
            return DecoderInfo(backend="torchcodec", device=self.device)
        except Exception as e:
            import traceback
            err_msg = f"torchcodec CUDA decode unavailable ({e})\n{traceback.format_exc()}"
            with open("gpu_debug.log", "a") as f: f.write(f"[DEBUG] {err_msg}\n")
            self.bus.log(f"torchcodec CUDA decode unavailable ({e}); trying ffmpeg -hwaccel cuda", level="warn")
            return None

    def _ffmpeg_has_cuda_hwaccel(self) -> bool:
        import subprocess
        try:
            out = subprocess.run(["ffmpeg", "-hwaccels"], capture_output=True, text=True, timeout=10)
            return "cuda" in out.stdout.lower()
        except Exception:
            return False

    def iter_frames(
        self, start_s: float = 0.0, end_s: float | None = None, stride: int = 1,
        target_fps: float | None = None, scale_width: int | None = None,
    ) -> Iterator[tuple[int, float, np.ndarray]]:
        """`target_fps`, when set, samples candidates at roughly that rate
        (time-based, via ffmpeg's `fps` filter or an equivalent index step)
        instead of every raw frame — this is what actually bounds decode
        work on a long/high-fps source; `stride` is a plain frame-count
        step, used only when `target_fps` is not given. `scale_width`, when
        set and smaller than the source width, scales frames down early to
        a working resolution before downstream sharpness scoring — on the
        GPU via torch, right after decode, on the PyNvVideoCodec path; via
        a plain software ffmpeg filter otherwise."""
        gen = self._iter_backend(start_s, end_s, stride, target_fps, scale_width)
        yield from self._with_progress_logging(gen, start_s, end_s, target_fps)

    def _iter_backend(self, start_s, end_s, stride, target_fps, scale_width):
        if self.info.backend == "torchcodec":
            yield from self._iter_torchcodec(start_s, end_s, stride, target_fps, scale_width)
        elif self.info.backend == "pynvvideocodec":
            yield from self._iter_pynvvideocodec(start_s, end_s, stride, target_fps, scale_width)
        elif self.info.backend == "ffmpeg_cuda":
            yield from self._iter_ffmpeg(start_s, end_s, stride, hwaccel=True, target_fps=target_fps, scale_width=scale_width)
        else:
            yield from self._iter_ffmpeg(start_s, end_s, stride, hwaccel=False, target_fps=target_fps, scale_width=scale_width)

    def _with_progress_logging(self, gen, start_s, end_s, target_fps, log_every_s: float = 5.0, ui_every_s: float = 1.0):
        t0 = time.time()
        last_log = t0
        last_ui = t0
        count = 0
        est_total = None
        if end_s is not None and target_fps:
            est_total = max(1, int((end_s - start_s) * target_fps))
        for item in gen:
            count += 1
            now = time.time()
            elapsed = now - t0
            fps = count / elapsed if elapsed > 0 else 0.0
            if now - last_log >= log_every_s:
                eta = f", ETA {(est_total - count) / fps:.0f}s" if (est_total and fps > 0 and count < est_total) else ""
                total_note = f"/{est_total}" if est_total else ""
                self.bus.log(f"Decode progress: {count}{total_note} frames, {fps:.1f} fps{eta}")
                last_log = now
            if now - last_ui >= ui_every_s:
                # A tighter cadence than the log line above, specifically
                # so the frame_extraction progress bar/rate label visibly
                # moves — the thing actually being watched to tell "is
                # this frozen or just slow" (see decode.py's module
                # docstring for the run where the absence of exactly this
                # signal made a real stall indistinguishable from normal
                # progress until it had already been stuck for minutes).
                self.bus.publish(
                    EventType.STAGE_PROGRESS, stage="frame_extraction",
                    frac=(count / est_total) if est_total else None,
                    rate_label=f"{fps:.1f} fps" + (f" ({count}/{est_total})" if est_total else f" ({count} frames)"),
                )
                last_ui = now
            yield item

    def _iter_torchcodec(self, start_s, end_s, stride, target_fps, scale_width):
        from torchcodec.decoders import VideoDecoder
        import torch
        import torch.nn.functional as F

        dec = self._make_torchcodec_decoder(VideoDecoder)
        n = len(dec)
        meta = dec.metadata
        fps = meta.average_fps or 30.0
        start_idx = int(start_s * fps)
        end_idx = int(end_s * fps) if end_s is not None else n
        step = max(1, round(fps / target_fps)) if target_fps else stride
        indices = list(range(start_idx, min(end_idx, n), step))
        w, h, _ = self._probe_dims()
        out_wh = None
        if scale_width and scale_width < w:
            out_w = max(2, int(scale_width) // 2 * 2)
            out_h = max(2, int(round(h * (out_w / w))) // 2 * 2)
            out_wh = (out_w, out_h)

        # TorchCodec's batch API preserves CUDA tensors.  Decoding the
        # selected indices and resizing them as a batch removes the old
        # per-frame Python call and 4K GPU-to-CPU transfer.
        batch_size = self._nvdec_batch_size(w, h)
        use_range = hasattr(dec, "get_frames_in_range")
        for offset in range(0, len(indices), batch_size):
            chunk_indices = indices[offset:offset + batch_size]
            if use_range:
                # get_frames_in_range(start, stop, step) is a true sequential
                # scan with a stride done inside torchcodec's C++ core — no
                # per-index Python seek. get_frames_at(indices=...) below is
                # index-based random access instead: on a real Kaggle run it
                # measured ~7-8 fps yielded against a 34.7 fps sequential
                # probe on the same backend/video, so it is not actually
                # exploiting sequential decode despite the indices being
                # evenly spaced. Fall back only if this API is missing.
                try:
                    frames = dec.get_frames_in_range(
                        start=chunk_indices[0], stop=chunk_indices[-1] + 1, step=step,
                    )
                except Exception:
                    use_range = False
                    frames = dec.get_frames_at(indices=chunk_indices)
            else:
                frames = dec.get_frames_at(indices=chunk_indices)
            batch = frames.data if hasattr(frames, "data") else frames
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)
            if out_wh is not None:
                batch = F.interpolate(batch.float(), size=(out_wh[1], out_wh[0]), mode="bilinear", align_corners=False)
            host = batch.clamp(0, 255).to("cpu", dtype=torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
            for i, arr in zip(chunk_indices, host):
                yield i, i / fps, arr

    def _iter_pynvvideocodec(self, start_s, end_s, stride, target_fps, scale_width):
        """Decodes straight to a GPU tensor and downscales there too (via
        torch, right after decode) — the whole point of this backend over
        the ffmpeg paths is that nothing ever needs to cross back to CPU
        memory until after the (small, already-scaled) frame is ready.
        Only reached after `_verify_pynvvideocodec_gpu` already exercised
        this same conversion path successfully, but still wrapped: a
        failure partway through (e.g. on a frame this specific video
        triggers differently) falls through to ffmpeg -hwaccel cuda rather
        than losing the whole run."""
        try:
            import PyNvVideoCodec as nvc
            import torch
            import torch.nn.functional as F

            dev_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
            dec = self._make_pynvvideocodec_decoder(nvc, dev_id)
            n = len(dec)
            w, h, native_fps = self._probe_dims()
            start_idx = int(start_s * native_fps)
            end_idx = int(end_s * native_fps) if end_s is not None else n
            step = max(1, round(native_fps / target_fps)) if target_fps else stride

            out_wh = None
            if scale_width and scale_width < w:
                out_w = max(2, int(scale_width) // 2 * 2)
                out_h = max(2, int(round(h * (out_w / w))) // 2 * 2)
                out_wh = (out_w, out_h)

            # Seek exactly once, then decode the entire window sequentially.
            # The sampling mask and resize both execute on the GPU.  Only the
            # already-small selected batch crosses PCIe to the existing NumPy
            # pipeline contract.
            if start_idx:
                dec.seek_to_index(start_idx)
            cursor = start_idx
            batch_size = self._nvdec_batch_size(w, h)
            while cursor < min(end_idx, n):
                frames = dec.get_batch_frames(min(batch_size, min(end_idx, n) - cursor))
                if not frames:
                    break
                batch = self._pynvvideocodec_batch_to_nchw(frames, self.device)
                actual = len(frames)
                positions = torch.arange(actual, device=batch.device)
                selected_positions = positions[(positions + cursor - start_idx).remainder(step) == 0]
                if selected_positions.numel():
                    sampled = batch.index_select(0, selected_positions)
                    if out_wh is not None:
                        sampled = F.interpolate(
                            sampled.float(), size=(out_wh[1], out_wh[0]),
                            mode="bilinear", align_corners=False,
                        ).clamp_(0, 255).to(torch.uint8)
                    # One batched transfer, rather than one synchronization
                    # and device-to-host copy per selected frame.
                    host = sampled.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                    # The first index in a later batch may not line up with
                    # its local zero, so derive the compact host indices from
                    # the same sampling phase as the GPU mask.
                    first = cursor + ((start_idx - cursor) % step)
                    selected_indices = list(range(first, cursor + actual, step))
                    for i, arr in zip(selected_indices, host):
                        yield i, i / native_fps, arr
                cursor += actual
        except Exception as e:
            self.bus.log(f"PyNvVideoCodec decode failed at runtime ({type(e).__name__}: {e}); falling back to ffmpeg -hwaccel cuda", level="warn")
            yield from self._iter_ffmpeg(start_s, end_s, stride, hwaccel=True, target_fps=target_fps, scale_width=scale_width)

    def _iter_ffmpeg(self, start_s, end_s, stride, hwaccel: bool, target_fps=None, scale_width=None, allow_fallback: bool = True):
        import subprocess

        w, h, native_fps = self._probe_dims()
        cmd = ["ffmpeg", "-v", "error"]
        if hwaccel:
            # `-hwaccel cuda` alone (no `-hwaccel_output_format cuda`): this
            # is decode-only GPU acceleration — NVDEC does the actual
            # decode, and ffmpeg auto-downloads each frame to system memory
            # right after. Deliberately NOT using scale_cuda/npp for the
            # downscale below: that filter isn't reliably available in
            # every ffmpeg build (confirmed a real issue — the "GPU decode
            # is on but everything's still slow" case turned out to be this
            # filter silently failing, not NVDEC itself), and PyNvVideoCodec
            # is now the primary path for a true no-CPU-roundtrip GPU
            # decode+scale anyway. This path is decode-acceleration only;
            # scaling happens the same (software, CPU) way as the plain-CPU
            # path below.
            cmd += ["-hwaccel", "cuda"]
        # `-ss` before `-i`: a single fast seek to the QUICK window's start,
        # never a per-frame seek — everything after this is one sequential
        # decode read straight off the pipe.
        cmd += ["-ss", str(start_s), "-i", str(self.video_path)]
        if end_s is not None:
            cmd += ["-t", str(max(0.0, end_s - start_s))]

        out_w, out_h = w, h
        if scale_width and scale_width < w:
            out_w = max(2, int(scale_width) // 2 * 2)
            out_h = max(2, int(round(h * (out_w / w))) // 2 * 2)

        filters = []
        if target_fps:
            filters.append(f"fps={target_fps}")
        elif stride > 1:
            filters.append(f"select='not(mod(n\\,{stride}))'")
        if (out_w, out_h) != (w, h):
            filters.append(f"scale={out_w}:{out_h}")
        if filters:
            cmd += ["-vf", ",".join(filters)]
        cmd += ["-pix_fmt", "rgb24", "-f", "rawvideo", "-"]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_bytes = out_w * out_h * 3
        idx = int(start_s * native_fps)
        idx_step = max(1, round(native_fps / target_fps)) if target_fps else stride

        # Drain stderr continuously in a background thread. This is not
        # optional: a real run froze hard mid-decode (hundreds of frames
        # in fine, then a sudden total stall, GPU going idle) with no
        # exception, no timeout, nothing in the logs — a classic Python
        # subprocess deadlock. stdout and stderr are separate OS pipes
        # with a limited buffer each (commonly 64KB on Linux); this loop
        # only ever read stdout, so once ffmpeg had written enough to
        # stderr (container/timestamp warnings etc. — `-v error` reduces
        # but doesn't guarantee zero) to fill that pipe, ffmpeg's own
        # write() to stderr blocked, which blocks ffmpeg entirely,
        # including producing any more stdout — so our stdout.read() here
        # then blocks forever waiting for frames that will never come.
        # Draining stderr on its own thread the whole time removes the
        # only way that pipe could ever fill.
        stderr_chunks: list[bytes] = []

        def _drain_stderr() -> None:
            try:
                for chunk in iter(lambda: proc.stderr.read(4096), b""):
                    stderr_chunks.append(chunk)
            except Exception:
                pass

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True, name="sih3d-ffmpeg-stderr")
        stderr_thread.start()

        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(out_h, out_w, 3)
                yield idx, idx / native_fps, arr
                idx += idx_step
        finally:
            proc.stdout.close()
            proc.wait(timeout=10)
            stderr_thread.join(timeout=5)
            if proc.returncode not in (0, None) and proc.returncode != 0:
                err = b"".join(stderr_chunks).decode("utf-8", "ignore")[-2000:]
                if hwaccel and allow_fallback:
                    self.bus.log(f"ffmpeg CUDA decode failed ({err.strip()[-300:]}); retrying with CPU decode", level="warn")
                    yield from self._iter_ffmpeg(start_s, end_s, stride, hwaccel=False, target_fps=target_fps, scale_width=scale_width)
                elif hwaccel:
                    # allow_fallback=False: used only by _verify_ffmpeg_cuda's
                    # probe, which needs to know hwaccel itself failed rather
                    # than silently receiving CPU-decoded frames it would
                    # mistake for fast GPU decode (confirmed by testing: the
                    # auto-fallback above measured 350+ fps on a failed probe
                    # because those frames were actually from the CPU retry).
                    self.bus.log(f"ffmpeg CUDA decode failed during verification ({err.strip()[-300:]})", level="warn")

    def _probe_dims(self) -> tuple[int, int, float]:
        from .io_detect import probe_video

        vi = probe_video(self.video_path)
        if vi is None:
            raise RuntimeError(f"ffprobe could not read {self.video_path}")
        return vi.width, vi.height, vi.fps or 30.0

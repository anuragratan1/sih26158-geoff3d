"""Background GPU + CPU utilization/memory sampler (pynvml + psutil), 0.5s
cadence.

Publishes GPU_SAMPLE events to the shared bus and keeps an in-memory
per-GPU (and overall-CPU) history so report.py can compute per-stage
average utilization (stage boundaries are correlated by timestamp against
STAGE_START/DONE events already on the bus's log, tracked separately in
report.py) and pipeline.py can report full-run idle time — the actual
"how much of the run did each GPU/the CPU sit idle" answer, not just an
eyeballed screenshot of a resource widget.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .events import EventBus, EventType

try:
    import pynvml

    _NVML_OK = True
except Exception:
    pynvml = None
    _NVML_OK = False

try:
    import psutil

    _PSUTIL_OK = True
except Exception:
    psutil = None
    _PSUTIL_OK = False


@dataclass
class CpuSample:
    ts: float
    percent: float  # 0-100, averaged across all cores (100 = every core fully busy)


@dataclass
class CpuHistory:
    samples: list[CpuSample] = field(default_factory=list)

    def append(self, s: CpuSample, cap: int = 20000) -> None:
        self.samples.append(s)
        if len(self.samples) > cap:
            del self.samples[: len(self.samples) - cap]

    def average(self, t0: float, t1: float) -> float | None:
        vals = [s.percent for s in self.samples if t0 <= s.ts <= t1]
        return sum(vals) / len(vals) if vals else None


@dataclass
class GpuSample:
    ts: float
    index: int
    name: str
    util_pct: float
    mem_used_mb: float
    mem_total_mb: float
    decoder_util_pct: float = 0.0


@dataclass
class GpuHistory:
    samples: list[GpuSample] = field(default_factory=list)

    def append(self, s: GpuSample, cap: int = 20000) -> None:
        self.samples.append(s)
        if len(self.samples) > cap:
            del self.samples[: len(self.samples) - cap]

    def average_util(self, index: int, t0: float, t1: float, metric: str = "util_pct") -> float | None:
        vals = [getattr(s, metric) for s in self.samples if s.index == index and t0 <= s.ts <= t1]
        return sum(vals) / len(vals) if vals else None

    def idle_fraction(self, index: int, t0: float, t1: float, idle_below_pct: float = 5.0) -> float | None:
        """Fraction of samples in [t0, t1] where SM utilization sat below
        `idle_below_pct` — used to flag "GPU sat idle" stretches during a
        stage rather than just reporting a possibly-misleading average."""
        vals = [s.util_pct for s in self.samples if s.index == index and t0 <= s.ts <= t1]
        return (sum(1 for v in vals if v < idle_below_pct) / len(vals)) if vals else None


class GpuMonitor:
    """Runs a daemon thread sampling every `interval_s` seconds.

    Falls back to a no-GPU / no-pynvml stub that still publishes zeroed
    samples at the same cadence (logged once), so the dashboard's GPU
    panel never has to special-case "no GPU" — it just shows flat lines.
    """

    def __init__(self, bus: EventBus, interval_s: float = 0.5):
        self.bus = bus
        self.interval_s = interval_s
        self.history = GpuHistory()
        self.cpu_history = CpuHistory()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._device_count = 0
        self._handles: list = []
        self._names: list[str] = []
        self._init_nvml()
        if _PSUTIL_OK:
            psutil.cpu_percent(interval=None)  # first call always returns 0.0/garbage; primes the internal baseline
        else:
            self.bus.log("psutil not available — CPU utilization tracking will show no data", level="warn")

    def _init_nvml(self) -> None:
        if not _NVML_OK:
            self.bus.log("pynvml not available — GPU panel will show no data", level="warn")
            return
        try:
            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            for i in range(self._device_count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                self._handles.append(h)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "ignore")
                self._names.append(name)
        except Exception as e:
            self.bus.log(f"pynvml init failed ({e}) — GPU panel will show no data", level="warn")
            self._device_count = 0

    @property
    def device_count(self) -> int:
        return self._device_count

    @property
    def device_names(self) -> list[str]:
        return list(self._names)

    def start(self) -> None:
        # Always starts, even with zero GPUs: CPU sampling (psutil) is
        # independent of GPU presence, and a CPU-only fallback run
        # shouldn't lose CPU utilization tracking just because there's no
        # GPU to also watch. The per-GPU loop below is simply a no-op when
        # self._handles is empty.
        if self._device_count == 0 and not _PSUTIL_OK:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            for i, h in enumerate(self._handles):
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    # Separate try/except: NVDEC load does NOT show up in
                    # `util.gpu` (that's SM/compute utilization only) — a
                    # real Kaggle run showed a near-empty GPU graph during
                    # frame extraction even once GPU decode was genuinely
                    # engaged, because this dashboard was blind to the
                    # decode engine's own utilization counter. Some driver/
                    # GPU combinations don't expose this call, so it's
                    # allowed to fail independently of the main sample.
                    try:
                        dec = pynvml.nvmlDeviceGetDecoderUtilization(h)
                        decoder_pct = float(dec[0]) if isinstance(dec, (tuple, list)) else float(getattr(dec, "utilization", 0))
                    except Exception:
                        decoder_pct = 0.0
                    sample = GpuSample(
                        ts=t0,
                        index=i,
                        name=self._names[i],
                        util_pct=float(util.gpu),
                        mem_used_mb=mem.used / (1024 * 1024),
                        mem_total_mb=mem.total / (1024 * 1024),
                        decoder_util_pct=decoder_pct,
                    )
                except Exception:
                    sample = GpuSample(ts=t0, index=i, name=self._names[i], util_pct=0.0, mem_used_mb=0.0, mem_total_mb=0.0)
                self.history.append(sample)
                self.bus.publish(
                    EventType.GPU_SAMPLE,
                    index=sample.index,
                    name=sample.name,
                    util_pct=sample.util_pct,
                    mem_used_mb=sample.mem_used_mb,
                    mem_total_mb=sample.mem_total_mb,
                    decoder_util_pct=sample.decoder_util_pct,
                    ts=sample.ts,
                )
            if _PSUTIL_OK:
                try:
                    # interval=None: non-blocking, returns the average over
                    # the time since the last call (primed once in
                    # __init__) — NOT a blocking 1s measurement, which
                    # would double this thread's own sampling interval.
                    cpu_pct = float(psutil.cpu_percent(interval=None))
                except Exception:
                    cpu_pct = 0.0
                self.cpu_history.append(CpuSample(ts=t0, percent=cpu_pct))
                self.bus.publish(EventType.CPU_SAMPLE, ts=t0, percent=cpu_pct)
            elapsed = time.time() - t0
            self._stop.wait(max(0.0, self.interval_s - elapsed))

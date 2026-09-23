"""Background GPU utilization/memory sampler (pynvml), 0.5s cadence.

Publishes GPU_SAMPLE events to the shared bus and keeps an in-memory
per-GPU history so report.py can compute per-stage average utilization
(stage boundaries are correlated by timestamp against STAGE_START/DONE
events already on the bus's log, tracked separately in report.py).
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


@dataclass
class GpuSample:
    ts: float
    index: int
    name: str
    util_pct: float
    mem_used_mb: float
    mem_total_mb: float


@dataclass
class GpuHistory:
    samples: list[GpuSample] = field(default_factory=list)

    def append(self, s: GpuSample, cap: int = 20000) -> None:
        self.samples.append(s)
        if len(self.samples) > cap:
            del self.samples[: len(self.samples) - cap]

    def average_util(self, index: int, t0: float, t1: float) -> float | None:
        vals = [s.util_pct for s in self.samples if s.index == index and t0 <= s.ts <= t1]
        return sum(vals) / len(vals) if vals else None


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
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._device_count = 0
        self._handles: list = []
        self._names: list[str] = []
        self._init_nvml()

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
        if self._device_count == 0:
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
                    sample = GpuSample(
                        ts=t0,
                        index=i,
                        name=self._names[i],
                        util_pct=float(util.gpu),
                        mem_used_mb=mem.used / (1024 * 1024),
                        mem_total_mb=mem.total / (1024 * 1024),
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
                    ts=sample.ts,
                )
            elapsed = time.time() - t0
            self._stop.wait(max(0.0, self.interval_s - elapsed))

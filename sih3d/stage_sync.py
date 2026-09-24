"""Per-stage completion signals for the notebook's per-stage result cells.

Fed from the SAME bus-drain loop that feeds dashboard.on_event()/
report.on_event() — a bus.drain() queue can only have one destructive
consumer (see events.py), so this is a third fan-out target in that same
loop, not an independent reader. Each result cell calls wait_for(stage)
before rendering, which blocks (showing a live "waiting" line, the cell's
own responsibility) until that stage's STAGE_DONE/STAGE_ERROR fires, or
the whole pipeline ends early (e.g. a fatal setup error before that stage
was ever reached) — either way the wait returns rather than hanging
forever.
"""

from __future__ import annotations

import threading

from .events import Event, EventType
from .report import STAGE_ORDER


class StageSync:
    def __init__(self, stage_names: list[str] = STAGE_ORDER):
        self.done_events = {name: threading.Event() for name in stage_names}
        self.error_events = {name: threading.Event() for name in stage_names}
        self.pipeline_done = threading.Event()
        self.pipeline_error: str | None = None

    def on_event(self, evt: Event) -> None:
        if evt.type == EventType.STAGE_DONE:
            ev = self.done_events.get(evt.payload.get("stage"))
            if ev:
                ev.set()
        elif evt.type == EventType.STAGE_ERROR:
            name = evt.payload.get("stage")
            if name in self.error_events:
                self.error_events[name].set()
                self.done_events[name].set()
        elif evt.type in (EventType.PIPELINE_DONE, EventType.PIPELINE_ERROR):
            if evt.type == EventType.PIPELINE_ERROR:
                self.pipeline_error = evt.payload.get("error")
            self.pipeline_done.set()
            # Unblock any stage that never even started (a fatal setup
            # error before it was reached) so its result cell doesn't hang.
            for ev in self.done_events.values():
                ev.set()

    def wait_for(self, stage: str, timeout: float | None = None) -> bool:
        ev = self.done_events.get(stage)
        if ev is None:
            return True
        return ev.wait(timeout)

    def failed(self, stage: str) -> bool:
        ev = self.error_events.get(stage)
        return bool(ev and ev.is_set())

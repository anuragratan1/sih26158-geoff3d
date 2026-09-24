"""Lightweight notebook progress reporting without widget or image rendering."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .events import Event, EventType

# A bare "{desc}: {percentage}%|{bar}| {elapsed}<{remaining}" — no raw
# iteration-rate unit (tqdm's default shows e.g. "70.78permille/s", which is
# both jargon and, for a stage that has no natural "items" to count, a
# meaningless number). Real throughput (fps, chunk counts, ...) is passed
# explicitly per stage via the `rate_label` postfix instead, so it only
# ever shows something an actual person asked for.
_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}{postfix}"


@dataclass
class _StageBar:
    bar: object
    progress: int = 0
    t0: float = 0.0


class ConsoleProgress:
    """Consumes events with plain tqdm bars and terse, useful log lines.

    It deliberately never serializes images, plots GPU graphs, or creates
    ipywidgets.  This keeps the notebook's main thread out of the frame
    extraction hot path while retaining stage, FPS, ETA, and warning output.
    Every stage prints one plain-English "done in Xs" line on completion —
    the bar itself disappears once a stage finishes (tqdm's `leave=False`),
    so the scrollback reads as a clean list of finished steps with times,
    not a wall of stale 100% bars.
    """

    def __init__(self) -> None:
        from tqdm import tqdm

        self._tqdm = tqdm
        self._bars: dict[str, _StageBar] = {}
        self._last_log: str | None = None
        self._run_t0 = time.time()

    def _get_or_make_bar(self, stage: str) -> _StageBar:
        state = self._bars.get(stage)
        if state is None:
            bar = self._tqdm(total=100, desc=stage.replace("_", " "), bar_format=_BAR_FORMAT, leave=False)
            state = _StageBar(bar=bar, t0=time.time())
            self._bars[stage] = state
        return state

    def on_event(self, evt: Event) -> None:
        payload = evt.payload
        if evt.type in (EventType.STAGE_START, EventType.SETUP_STAGE_START):
            self._get_or_make_bar(payload.get("stage", "working"))
        elif evt.type == EventType.STAGE_PROGRESS:
            state = self._get_or_make_bar(payload.get("stage", "working"))
            frac = payload.get("frac")
            if frac is not None:
                target = min(100, max(state.progress, round(float(frac) * 100)))
                state.bar.update(target - state.progress)
                state.progress = target
            rate = payload.get("rate_label")
            if rate:
                state.bar.set_postfix_str(rate, refresh=False)
        elif evt.type in (EventType.STAGE_DONE, EventType.SETUP_STAGE_DONE):
            stage = payload.get("stage", "working")
            state = self._bars.get(stage)
            if state is not None:
                state.bar.update(100 - state.progress)
                state.progress = 100
                state.bar.close()
                elapsed = time.time() - state.t0
                total = time.time() - self._run_t0
                self._tqdm.write(f"[done] {stage.replace('_', ' ')} — {elapsed:.0f}s (total elapsed {total:.0f}s)")
        elif evt.type == EventType.LOG:
            message = str(payload.get("message", ""))
            # Progress logs already carry rate/ETA through the bar; preserve
            # warnings and state transitions once, without notebook spam.
            if payload.get("level") == "warn" or message.startswith(("Video decoder backend:", "PyNvVideoCodec verified:", "torchcodec CUDA decode verified:")):
                if message != self._last_log:
                    self._tqdm.write(("WARNING: " if payload.get("level") == "warn" else "") + message)
                    self._last_log = message

    def close(self) -> None:
        for state in self._bars.values():
            if state.progress < 100:
                state.bar.close()
        self._tqdm.write(f"[done] total elapsed {time.time() - self._run_t0:.0f}s")

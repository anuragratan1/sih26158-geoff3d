"""Lightweight notebook progress reporting without widget or image rendering."""

from __future__ import annotations

import time

from .events import Event, EventType

# A bare "{desc}: {percentage}%|{bar}| {elapsed}<{remaining}" — no raw
# iteration-rate unit (tqdm's default shows e.g. "70.78permille/s", which is
# both jargon and, for a stage that has no natural "items" to count, a
# meaningless number). Real throughput (fps, chunk counts, ...) is passed
# explicitly per stage via the `rate_label` postfix instead, so it only
# ever shows something an actual person asked for.
_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}{postfix}"

# One dynamic bar spans the WHOLE run instead of one bar per pipeline stage
# stacking up in the output. The six underlying stage names pipeline.py
# emits are collapsed into four human-meaningful phases (geometric_
# reconstruction/large_scale_alignment/dense_point_cloud all START and
# FINISH together for a single QUICK-mode chunk, so showing them as three
# separate simultaneous bars was pure clutter, not three separate things
# actually happening in sequence). Weights are a rough split of typical
# wall time, not a promise — they only affect how far the single bar moves
# per phase, never whether a phase is "done" (that's driven by real
# STAGE_DONE events, same as before).
_PHASES = [
    ("frame_extraction", "Decoding video", ("frame_extraction",), 15),
    ("camera_trajectory", "Estimating camera poses", ("camera_trajectory",), 5),
    ("reconstruction", "3D reconstruction", ("geometric_reconstruction", "large_scale_alignment", "dense_point_cloud"), 40),
    ("mesh_textured_model", "Building mesh + texture", ("mesh_textured_model",), 40),
]

_STAGE_TO_PHASE: dict[str, str] = {}
_PHASE_RANGE: dict[str, tuple[float, float]] = {}
_PHASE_LABEL: dict[str, str] = {}
_PHASE_MEMBERS: dict[str, tuple[str, ...]] = {}
_cursor = 0.0
for _key, _label, _members, _weight in _PHASES:
    _PHASE_RANGE[_key] = (_cursor, _cursor + _weight)
    _PHASE_LABEL[_key] = _label
    _PHASE_MEMBERS[_key] = _members
    for _m in _members:
        _STAGE_TO_PHASE[_m] = _key
    _cursor += _weight


class ConsoleProgress:
    """Consumes events with ONE plain tqdm bar spanning the whole run, plus
    terse, useful log lines.

    Deliberately never serializes images, plots GPU graphs, or creates
    ipywidgets — keeps the notebook's main thread out of the frame
    extraction hot path while retaining stage, FPS, ETA, and warning
    output. Prints one plain-English "done in Xs" line per phase on
    completion; the bar itself never multiplies — there is exactly one,
    always visible, moving left to right across the whole run.
    """

    def __init__(self) -> None:
        from tqdm import tqdm

        self._tqdm = tqdm
        self._bar = None
        self._progress = 0.0  # 0-100, monotonic
        self._current_phase: str | None = None
        self._phase_t0: dict[str, float] = {}
        self._done_stages: set[str] = set()
        self._reported_phase_done: set[str] = set()
        self._last_log: str | None = None
        self._run_t0 = time.time()

    def _ensure_bar(self) -> None:
        if self._bar is None:
            self._bar = self._tqdm(total=100, desc="Starting", bar_format=_BAR_FORMAT, leave=True)

    def _set_progress(self, target: float) -> None:
        self._ensure_bar()
        target = min(100.0, max(self._progress, target))
        self._bar.update(target - self._progress)
        self._progress = target

    def _enter_phase(self, phase: str) -> None:
        self._ensure_bar()
        if phase not in self._phase_t0:
            self._phase_t0[phase] = time.time()
        if self._current_phase != phase:
            self._current_phase = phase
            self._bar.set_description(_PHASE_LABEL.get(phase, phase.replace("_", " ")), refresh=False)
            lo, _ = _PHASE_RANGE.get(phase, (self._progress, 100.0))
            self._set_progress(max(self._progress, lo))

    def _maybe_report_phase_done(self, phase: str) -> None:
        members = _PHASE_MEMBERS.get(phase, (phase,))
        if phase in self._reported_phase_done or not all(m in self._done_stages for m in members):
            return
        self._reported_phase_done.add(phase)
        _, hi = _PHASE_RANGE.get(phase, (self._progress, self._progress))
        self._set_progress(hi)
        elapsed = time.time() - self._phase_t0.get(phase, time.time())
        total = time.time() - self._run_t0
        self._tqdm.write(f"[done] {_PHASE_LABEL.get(phase, phase)} — {elapsed:.0f}s (total elapsed {total:.0f}s)")

    def on_event(self, evt: Event) -> None:
        payload = evt.payload
        if evt.type == EventType.STAGE_START:
            stage = payload.get("stage", "working")
            self._enter_phase(_STAGE_TO_PHASE.get(stage, stage))
        elif evt.type == EventType.STAGE_PROGRESS:
            stage = payload.get("stage", "working")
            phase = _STAGE_TO_PHASE.get(stage, stage)
            self._enter_phase(phase)
            frac = payload.get("frac")
            if frac is not None:
                lo, hi = _PHASE_RANGE.get(phase, (self._progress, 100.0))
                self._set_progress(lo + (hi - lo) * min(1.0, max(0.0, float(frac))))
            rate = payload.get("rate_label")
            if rate:
                self._ensure_bar()
                self._bar.set_postfix_str(rate, refresh=False)
        elif evt.type == EventType.STAGE_DONE:
            stage = payload.get("stage", "working")
            self._done_stages.add(stage)
            self._maybe_report_phase_done(_STAGE_TO_PHASE.get(stage, stage))
        elif evt.type == EventType.LOG:
            message = str(payload.get("message", ""))
            # Progress logs already carry rate/ETA through the bar; preserve
            # warnings and state transitions once, without notebook spam.
            if payload.get("level") == "warn" or message.startswith(("Video decoder backend:", "PyNvVideoCodec verified:", "torchcodec CUDA decode verified:")):
                if message != self._last_log:
                    self._tqdm.write(("WARNING: " if payload.get("level") == "warn" else "") + message)
                    self._last_log = message

    def close(self) -> None:
        if self._bar is not None:
            self._set_progress(100.0)
            self._bar.close()
        self._tqdm.write(f"[done] total elapsed {time.time() - self._run_t0:.0f}s")

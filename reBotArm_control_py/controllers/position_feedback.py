"""Rate-limited position acquisition; no synchronous reads in the servo loop."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
from ..actuator.dm_feedback import DmFeedbackFault


@dataclass(frozen=True)
class PositionSample:
    q: np.ndarray
    sampled_at: float
    source: str
    sequence: int = 0
    read_span: float = 0.0
    read_duration: float = 0.0


class PositionFeedback:
    def __init__(self, group, rate: float = 25.0, timeout_ms: int = 20):
        if not np.isfinite(rate) or rate <= 0 or timeout_ms <= 0:
            raise ValueError("Invalid feedback rate/timeout")
        self._group = group
        self._period = 1.0 / rate
        self._timeout_ms = timeout_ms
        self._lock = threading.Lock()
        self._acquisition_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._sample = None
        self.error_count = 0
        self.last_error = None
        self.fatal_error = None

    def refresh(self):
        # A startup/manual refresh and the worker must not overlap requests or
        # overwrite a newer batch with an older batch that finished later.
        with self._acquisition_lock:
            return self._refresh()

    def _refresh(self):
        try:
            q, stamp, source = self._group.read_position_sample(self._timeout_ms)
        except DmFeedbackFault as error:
            self.fatal_error = str(error)
            raise
        with self._lock:
            sequence = 1 if self._sample is None else self._sample.sequence + 1
            self._sample = PositionSample(q.copy(), stamp, source, sequence,
                getattr(self._group, "last_position_read_span", 0.0),
                getattr(self._group, "last_position_read_duration", 0.0))
            self.fatal_error = None
        return self.latest()

    def latest(self):
        with self._lock:
            if self._sample is None:
                raise RuntimeError("Position feedback has not been initialized")
            s = self._sample
            return PositionSample(s.q.copy(), s.sampled_at, s.source,
                                  s.sequence, s.read_span, s.read_duration)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Position feedback is already running")
        self.refresh()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rebotarm-position-feedback", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            began = time.monotonic()
            try:
                self.refresh()
            except Exception as error:
                self.error_count += 1
                self.last_error = repr(error)
                # Keep the old timestamp: failures must not make stale data fresh.
            self._stop.wait(max(0.0, self._period - (time.monotonic() - began)))

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._group.num_joints * self._timeout_ms / 1000 + 0.5))
            if self._thread.is_alive():
                raise RuntimeError("Position feedback thread did not stop")

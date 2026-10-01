"""Acquisition on the Qt side: one controller that owns the current SignalSource, a ring buffer for the
plots, a recorder for START/STOP recording, and turns the source's background-thread callbacks into
Qt signals delivered in the GUI thread."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal, Slot

from app.sources import SignalSource, SourceError, SourceKind, SourceState
from network import config as C


class RingBuffer:
    """Fixed-capacity sample buffer [capacity, n_channels] float32; `latest(n)` returns the newest n samples in order."""

    def __init__(self, seconds: float = 30.0, fs: float = C.FS, n_channels: int = C.N_CHANNELS):
        self.capacity = int(seconds * fs)
        self.n_channels = n_channels
        self._buf = np.zeros((self.capacity, n_channels), dtype=np.float32)
        self._pos = 0            # next write index
        self.count = 0           # valid samples (<= capacity)
        self.total = 0           # samples ever appended

    def clear(self) -> None:
        self._pos = self.count = self.total = 0

    def append(self, chunk: np.ndarray) -> None:
        n = len(chunk)
        if n == 0:
            return
        if n >= self.capacity:                            # only the newest `capacity` samples matter
            self._buf[:] = chunk[-self.capacity:]
            self._pos, self.count = 0, self.capacity
        else:
            end = self._pos + n
            if end <= self.capacity:
                self._buf[self._pos:end] = chunk
            else:
                first = self.capacity - self._pos
                self._buf[self._pos:] = chunk[:first]
                self._buf[:n - first] = chunk[first:]
            self._pos = end % self.capacity
            self.count = min(self.capacity, self.count + n)
        self.total += n

    def latest(self, n: int) -> np.ndarray:
        n = min(int(n), self.count)
        if n <= 0:
            return np.zeros((0, self.n_channels), dtype=np.float32)
        start = (self._pos - n) % self.capacity
        if start + n <= self.capacity:
            return self._buf[start:start + n].copy()
        return np.concatenate([self._buf[start:], self._buf[:(start + n) % self.capacity]])


class Recorder:
    """Collects chunks between start() and stop()."""

    def __init__(self):
        self._chunks: list[np.ndarray] = []
        self.active = False
        self.n_samples = 0

    def start(self) -> None:
        self._chunks, self.n_samples, self.active = [], 0, True

    def append(self, chunk: np.ndarray) -> None:
        if self.active:
            self._chunks.append(np.asarray(chunk, dtype=np.float32))
            self.n_samples += len(chunk)

    def stop(self) -> None:
        self.active = False

    def data(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros((0, C.N_CHANNELS), dtype=np.float32)
        return np.concatenate(self._chunks)

    def clear(self) -> None:
        self._chunks, self.n_samples, self.active = [], 0, False


class AcquisitionController(QObject):
    """Owns the active source. All public signals are emitted in the GUI thread."""

    chunk_ready = Signal(object)              # np.ndarray [n_new, 4] float32 (mV)
    state_changed = Signal(object, str)       # SourceState, message
    progress = Signal(int)                    # samples delivered by a replay / file source
    source_changed = Signal(object)           # SignalSource or None
    recording_changed = Signal(bool, int)     # recording active?, samples recorded
    error = Signal(str)

    _samples_from_thread = Signal(object)
    _state_from_thread = Signal(object, str)
    _progress_from_thread = Signal(int)

    def __init__(self, ring_seconds: float = 30.0, parent=None):
        super().__init__(parent)
        self.source: SignalSource | None = None
        self.ring = RingBuffer(ring_seconds)
        self.recorder = Recorder()
        self.total_samples = 0
        self.inference_enabled = True
        # queued connections: the source calls back from its own thread, these slots run in the GUI thread
        self._samples_from_thread.connect(self._ingest, Qt.QueuedConnection)
        self._state_from_thread.connect(self._on_state, Qt.QueuedConnection)
        self._progress_from_thread.connect(self.progress, Qt.QueuedConnection)

    # ------------------------------------------------------------------ source management
    @property
    def fs(self) -> float:
        return self.source.fs if self.source else float(C.FS)

    @property
    def kind(self) -> SourceKind | None:
        return self.source.kind if self.source else None

    @property
    def state(self) -> SourceState:
        return self.source.state if self.source else SourceState.DISCONNECTED

    def set_source(self, source: SignalSource | None) -> None:
        if self.source is not None:
            try:
                self.source.disconnect()
            except Exception:                                   # a dying source must not take the app down
                pass
            self.source.on_samples = self.source.on_state = self.source.on_progress = None
        self.source = source
        self.reset()
        if source is not None:
            source.on_samples = self._samples_from_thread.emit
            source.on_state = self._state_from_thread.emit
            source.on_progress = self._progress_from_thread.emit
        self.source_changed.emit(source)

    def reset(self) -> None:
        self.ring.clear()
        self.recorder.clear()
        self.total_samples = 0
        self.recording_changed.emit(False, 0)

    def connect_source(self) -> bool:
        return self._guard(lambda: self.source.connect(), "connect")

    def disconnect_source(self) -> bool:
        ok = self._guard(lambda: self.source.disconnect(), "disconnect")
        return ok

    def start_stream(self) -> bool:
        self.reset()
        return self._guard(lambda: self.source.start(), "start")

    def stop_stream(self) -> bool:
        return self._guard(lambda: self.source.stop(), "stop")

    def _guard(self, action, what: str) -> bool:
        if self.source is None:
            self.error.emit(f"cannot {what}: no signal source selected")
            return False
        try:
            action()
            return True
        except SourceError as e:
            self.error.emit(str(e))
        except Exception as e:                                  # unexpected: report, never crash the GUI
            self.error.emit(f"{what} failed: {e.__class__.__name__}: {e}")
        return False

    # ------------------------------------------------------------------ recording
    @property
    def is_recording(self) -> bool:
        return self.recorder.active

    def start_recording(self) -> None:
        self.recorder.start()
        self.recording_changed.emit(True, 0)

    def stop_recording(self) -> np.ndarray:
        self.recorder.stop()
        self.recording_changed.emit(False, self.recorder.n_samples)
        return self.recorder.data()

    def discard_recording(self) -> None:
        self.recorder.clear()
        self.recording_changed.emit(False, 0)

    # ------------------------------------------------------------------ slots (GUI thread)
    @Slot(object)
    def _ingest(self, chunk) -> None:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != C.N_CHANNELS:
            self.error.emit(f"source delivered samples with shape {chunk.shape}, expected [n, {C.N_CHANNELS}]")
            return
        self.ring.append(chunk)
        self.total_samples += len(chunk)
        if self.recorder.active:
            self.recorder.append(chunk)
            self.recording_changed.emit(True, self.recorder.n_samples)
        if self.inference_enabled:
            self.chunk_ready.emit(chunk)

    @Slot(object, str)
    def _on_state(self, state, message) -> None:
        self.state_changed.emit(state, message)
        if state is SourceState.ERROR and message:
            self.error.emit(message)

    def shutdown(self) -> None:
        self.set_source(None)

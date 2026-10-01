"""Signal sources: where the EMG comes from. Plain Python, no Qt, so they are testable on their own.

    SignalSource (abstract)
      ├── FileSignalSource      a recording file, delivered at once (offline mode)
      ├── ReplaySignalSource    a loaded recording streamed in real time (development stand-in for hardware)
      └── DelsysTrignoSource    the Delsys Trigno Avanti via the Trigno Control Utility (app/delsys.py)

A source reports samples through `on_samples(chunk)` with chunk = float32 [n_new, n_channels] in mV,
and state changes through `on_state(state, message)`. Callbacks may run on a background thread; the
Qt side (app/acquisition.py) forwards them to the GUI thread with signals.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np

from network import config as C
from network.recording import Recording, RecordingFormatError, load_recording_file


class SourceState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"          # ready, not streaming
    STREAMING = "streaming"
    ERROR = "error"


class SourceKind(Enum):
    HARDWARE = "LIVE HARDWARE"
    REPLAY = "SIMULATED / REPLAY"
    FILE = "FILE"


class SourceError(Exception):
    """Connection / streaming problems that the UI should show as a message."""


class SignalSource(ABC):
    name: str = "source"
    kind: SourceKind = SourceKind.FILE

    def __init__(self):
        self.on_samples: Callable[[np.ndarray], None] | None = None
        self.on_state: Callable[[SourceState, str], None] | None = None
        self.on_progress: Callable[[int], None] | None = None     # samples delivered so far (replay / file)
        self.state: SourceState = SourceState.DISCONNECTED
        self.fs: float = float(C.FS)
        self.n_channels: int = C.N_CHANNELS
        self.channel_names: list[str] = [f"CH{i + 1}" for i in range(C.N_CHANNELS)]
        self.units: str = "mV"
        self.last_message: str = ""

    # -- lifecycle -----------------------------------------------------------
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    # -- helpers -------------------------------------------------------------
    def _set_state(self, state: SourceState, message: str = "") -> None:
        self.state, self.last_message = state, message
        if self.on_state:
            self.on_state(state, message)

    def _emit(self, chunk: np.ndarray) -> None:
        if self.on_samples and len(chunk):
            self.on_samples(np.ascontiguousarray(chunk, dtype=np.float32))

    @property
    def is_streaming(self) -> bool:
        return self.state is SourceState.STREAMING

    def describe(self) -> str:
        return f"{self.name}: {self.n_channels} channels @ {self.fs:g} Hz ({self.kind.value})"


class FileSignalSource(SignalSource):
    """A recording file (.mat / .npz / .csv). connect() loads and validates it; start() delivers it at once."""
    kind = SourceKind.FILE

    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.name = self.path.name
        self.recording: Recording | None = None

    def connect(self) -> None:
        self._set_state(SourceState.CONNECTING, f"loading {self.path.name}")
        try:
            rec = load_recording_file(self.path)
        except RecordingFormatError as e:
            self._set_state(SourceState.ERROR, str(e))
            raise SourceError(str(e)) from e
        self.recording, self.fs, self.n_channels, self.channel_names = rec, rec.fs, rec.n_channels, list(rec.channel_names)
        self._set_state(SourceState.CONNECTED, rec.describe() + ("; " + "; ".join(rec.warnings) if rec.warnings else ""))

    def start(self) -> None:
        if self.recording is None:
            raise SourceError("no file loaded")
        self._set_state(SourceState.STREAMING, "delivering file")
        self._emit(self.recording.data)
        if self.on_progress:
            self.on_progress(self.recording.n_samples)
        self._set_state(SourceState.CONNECTED, "file delivered")

    def stop(self) -> None:
        if self.state is SourceState.STREAMING:
            self._set_state(SourceState.CONNECTED, "stopped")

    def disconnect(self) -> None:
        self.recording = None
        self._set_state(SourceState.DISCONNECTED, "")


class ReplaySignalSource(SignalSource):
    """Streams a loaded recording in real time (chunks every `chunk_s`, paced by the recording's fs).

    `speed` > 1 replays faster than real time (tests use large speeds). With `loop=True` the recording
    restarts when it ends; otherwise the source goes back to CONNECTED with the message "replay finished".
    """
    kind = SourceKind.REPLAY

    def __init__(self, recording: Recording, chunk_s: float = 0.05, speed: float = 1.0, loop: bool = False):
        super().__init__()
        self.recording = recording
        self.name = f"replay of {recording.path.name if recording.path else 'recording'}"
        self.fs, self.n_channels, self.channel_names = recording.fs, recording.n_channels, list(recording.channel_names)
        self.chunk_s, self.speed, self.loop = float(chunk_s), float(speed), bool(loop)
        self.position = 0                              # samples delivered in the current pass
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        self._set_state(SourceState.CONNECTED, self.recording.describe())

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.position = 0
        self._set_state(SourceState.STREAMING, f"replaying at {self.speed:g}x")
        self._thread = threading.Thread(target=self._run, name="replay-source", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        data, n = self.recording.data, self.recording.n_samples
        t0 = time.perf_counter()
        while not self._stop.is_set():
            due = min(n, int((time.perf_counter() - t0) * self.fs * self.speed))
            if due > self.position:
                self._emit(data[self.position:due])
                self.position = due
                if self.on_progress:
                    self.on_progress(self.position)
            if self.position >= n:
                if not self.loop:
                    self._set_state(SourceState.CONNECTED, "replay finished")
                    return
                self.position, t0 = 0, time.perf_counter()
            self._stop.wait(self.chunk_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self.state is SourceState.STREAMING:
            self._set_state(SourceState.CONNECTED, "stopped")

    def disconnect(self) -> None:
        self.stop()
        self._set_state(SourceState.DISCONNECTED, "")

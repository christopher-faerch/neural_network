"""Delsys Trigno Avanti via the Trigno Control Utility (TCU) TCP SDK.

STATUS: written against the documented SDK protocol, NOT tested against real hardware (no Delsys
software was available while developing). Everything protocol-related is configuration in
`TrignoConfig`; check these on the lab PC before trusting the stream:

* The TCU must be running on `host`; it opens an ASCII command port (default 50040) and binary data
  ports. Commands are terminated by "\\r\\n\\r\\n"; the TCU answers with a text reply containing "OK".
* EMG data: on the data port (50043 for Avanti sensors, 50041 for legacy Trigno) the TCU streams frames
  of `frame_channels` (16) little-endian float32 values, one per sensor slot, in VOLTS. Unpaired sensor
  slots stream zeros. Data only flows after the START command.
* "BACKWARDS COMPATIBILITY ON" + "UPSAMPLE ON" make the EMG stream 2000 Hz for every sensor type,
  which is the sampling rate the network was trained at. Avanti sensors natively run at 1925.93 Hz.
* `sensors` are the 1-based Trigno sensor numbers that carry CH1..CH4 of the model (the four forearm
  muscles of the dataset, in the same order). `scale_to_mv` converts volts to mV (x1000) because the
  network was trained on mV.

Nothing here is specific to the rest of the app: it is a SignalSource like the file and replay sources.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

from app.sources import SignalSource, SourceError, SourceKind, SourceState
from network import config as C


@dataclass
class TrignoConfig:
    host: str = "localhost"
    command_port: int = 50040
    emg_port: int = 50043                       # 50041 for legacy (non-Avanti) EMG
    sensors: tuple[int, ...] = (1, 2, 3, 4)     # Trigno sensor numbers for CH1..CH4
    frame_channels: int = 16                    # float32 values per frame on the EMG port
    scale_to_mv: float = 1000.0                 # volts -> mV
    fs: float = 2000.0
    setup_commands: tuple[str, ...] = ("BACKWARDS COMPATIBILITY ON", "UPSAMPLE ON")
    connect_timeout_s: float = 2.0
    stall_timeout_s: float = 5.0                # no data for this long while streaming -> connection lost
    read_frames: int = 100                      # frames per recv (100 frames = 50 ms at 2000 Hz)

    def validate(self) -> None:
        if len(self.sensors) != C.N_CHANNELS:
            raise SourceError(f"exactly {C.N_CHANNELS} sensor numbers are needed, got {self.sensors}")
        if any(s < 1 or s > self.frame_channels for s in self.sensors):
            raise SourceError(f"sensor numbers must be between 1 and {self.frame_channels}")
        if len(set(self.sensors)) != len(self.sensors):
            raise SourceError("sensor numbers must be distinct")

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class DelsysTrignoSource(SignalSource):
    kind = SourceKind.HARDWARE
    name = "Delsys Trigno (TCU)"

    def __init__(self, config: TrignoConfig | None = None):
        super().__init__()
        self.config = config or TrignoConfig()
        self.fs = float(self.config.fs)
        self.channel_names = [f"Sensor {s}" for s in self.config.sensors]
        self._cmd: socket.socket | None = None
        self._data: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames_received = 0

    # ------------------------------------------------------------------ protocol helpers
    def _recv_text(self, sock: socket.socket, timeout: float) -> str:
        sock.settimeout(timeout)
        chunks = []
        try:
            while True:
                part = sock.recv(4096)
                if not part:
                    break
                chunks.append(part)
                if b"\r\n\r\n" in part:
                    break
        except socket.timeout:
            pass
        return b"".join(chunks).decode("ascii", errors="replace")

    def _command(self, text: str, expect_ok: bool = True) -> str:
        if self._cmd is None:
            raise SourceError("not connected")
        try:
            self._cmd.sendall(text.encode("ascii") + b"\r\n\r\n")
            reply = self._recv_text(self._cmd, self.config.connect_timeout_s)
        except OSError as e:
            raise SourceError(f"command '{text}' failed: {e}") from e
        if expect_ok and "OK" not in reply.upper():
            raise SourceError(f"Trigno Control Utility rejected '{text}': {reply.strip() or 'no reply'}")
        return reply

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        cfg = self.config
        cfg.validate()
        self._set_state(SourceState.CONNECTING, f"connecting to Trigno Control Utility at {cfg.host}:{cfg.command_port}")
        try:
            self._cmd = socket.create_connection((cfg.host, cfg.command_port), timeout=cfg.connect_timeout_s)
            banner = self._recv_text(self._cmd, 1.0)                         # the TCU greets with a banner
            for command in cfg.setup_commands:
                self._command(command)
            self._data = socket.create_connection((cfg.host, cfg.emg_port), timeout=cfg.connect_timeout_s)
        except (OSError, SourceError) as e:
            self._close_sockets()
            message = (f"Trigno Control Utility not reachable at {cfg.host}:{cfg.command_port}/{cfg.emg_port} "
                       f"({e}). Start the TCU on that machine and check the ports.")
            self._set_state(SourceState.ERROR, message)
            raise SourceError(message) from e
        self._set_state(SourceState.CONNECTED, f"connected to TCU ({banner.strip().splitlines()[0] if banner.strip() else 'no banner'})")

    def start(self) -> None:
        if self._cmd is None or self._data is None:
            raise SourceError("connect() first")
        if self._thread and self._thread.is_alive():
            return
        self._command("START")
        self._stop.clear()
        self.frames_received = 0
        self._set_state(SourceState.STREAMING, "streaming")
        self._thread = threading.Thread(target=self._reader, name="trigno-reader", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        cfg = self.config
        frame_bytes = 4 * cfg.frame_channels
        columns = [s - 1 for s in cfg.sensors]
        buf = bytearray()
        sock = self._data
        sock.settimeout(cfg.stall_timeout_s)
        last_data = time.perf_counter()
        while not self._stop.is_set():
            try:
                part = sock.recv(frame_bytes * cfg.read_frames)
            except socket.timeout:
                self._set_state(SourceState.ERROR, f"no EMG data for {cfg.stall_timeout_s:g} s: connection lost or sensors not streaming")
                return
            except OSError as e:
                if not self._stop.is_set():
                    self._set_state(SourceState.ERROR, f"connection lost: {e}")
                return
            if not part:
                self._set_state(SourceState.ERROR, "Trigno Control Utility closed the data connection")
                return
            buf += part
            n_frames = len(buf) // frame_bytes
            if n_frames:
                frames = np.frombuffer(bytes(buf[:n_frames * frame_bytes]), dtype="<f4").reshape(n_frames, cfg.frame_channels)
                del buf[:n_frames * frame_bytes]
                self.frames_received += n_frames
                self._emit(frames[:, columns] * np.float32(cfg.scale_to_mv))
                last_data = time.perf_counter()

    def stop(self) -> None:
        self._stop.set()
        if self._cmd is not None:
            try:
                self._command("STOP", expect_ok=False)
            except SourceError:
                pass
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self.state is SourceState.STREAMING:
            self._set_state(SourceState.CONNECTED, "stopped")

    def disconnect(self) -> None:
        self.stop()
        if self._cmd is not None:
            try:
                self._command("QUIT", expect_ok=False)
            except SourceError:
                pass
        self._close_sockets()
        self._set_state(SourceState.DISCONNECTED, "")

    def _close_sockets(self) -> None:
        for s in (self._data, self._cmd):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        self._cmd = self._data = None


def assumptions_text() -> str:
    """Shown in the UI next to the Delsys controls."""
    return ("Untested adapter: uses the Trigno Control Utility TCP SDK (commands on port 50040, EMG frames of 16 "
            "float32 values in volts on port 50043). Confirm ports, sensor numbers and the 2000 Hz upsampling in "
            "the TCU before trusting the data. The network was trained on BIOPAC recordings in mV.")

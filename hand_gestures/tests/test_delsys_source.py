"""The Trigno adapter against a fake Trigno Control Utility (TCP server speaking the documented protocol).

This proves the adapter's framing, sensor mapping, scaling and error handling. It does NOT prove
compatibility with real Delsys hardware, which was not available.
"""
import socket
import struct
import threading
import time

import numpy as np
import pytest

from app.delsys import DelsysTrignoSource, TrignoConfig
from app.sources import SourceError, SourceState

FRAME_CHANNELS = 16


class FakeTCU:
    """Command port: banner, 'OK' replies, records commands. Data port: streams deterministic frames after START."""

    def __init__(self, n_frames=600, send_data=True):
        self.n_frames, self.send_data = n_frames, send_data
        self.commands: list[str] = []
        self.cmd_srv = socket.socket(); self.cmd_srv.bind(("127.0.0.1", 0)); self.cmd_srv.listen(1)
        self.data_srv = socket.socket(); self.data_srv.bind(("127.0.0.1", 0)); self.data_srv.listen(1)
        self.cmd_port, self.data_port = self.cmd_srv.getsockname()[1], self.data_srv.getsockname()[1]
        self._streaming = threading.Event()
        self._closed = threading.Event()
        self.threads = [threading.Thread(target=self._cmd_loop, daemon=True), threading.Thread(target=self._data_loop, daemon=True)]
        for t in self.threads:
            t.start()

    def _cmd_loop(self):
        self.cmd_srv.settimeout(10)
        try:
            conn, _ = self.cmd_srv.accept()
        except socket.timeout:
            return
        conn.sendall(b"Delsys Trigno System Digital Protocol Version 3.6\r\n\r\n")
        buf = b""
        conn.settimeout(10)
        while not self._closed.is_set():
            try:
                part = conn.recv(1024)
            except (socket.timeout, OSError):
                break
            if not part:
                break
            buf += part
            while b"\r\n\r\n" in buf:
                cmd, buf = buf.split(b"\r\n\r\n", 1)
                text = cmd.decode().strip()
                self.commands.append(text)
                if text == "START":
                    self._streaming.set()
                elif text == "STOP":
                    self._streaming.clear()
                conn.sendall(b"OK\r\n\r\n")
                if text == "QUIT":
                    self._closed.set()
        conn.close()

    def _data_loop(self):
        self.data_srv.settimeout(10)
        try:
            conn, _ = self.data_srv.accept()
        except socket.timeout:
            return
        if not self._streaming.wait(timeout=10) or not self.send_data:
            self._closed.wait(timeout=10)
            conn.close()
            return
        frame = 0
        while frame < self.n_frames and not self._closed.is_set():
            block = b"".join(struct.pack("<16f", *[(f * FRAME_CHANNELS + ch) * 1e-3 for ch in range(FRAME_CHANNELS)])
                             for f in range(frame, min(frame + 50, self.n_frames)))
            try:
                conn.sendall(block)
            except OSError:
                break
            frame += 50
            time.sleep(0.005)
        self._closed.wait(timeout=10)
        conn.close()

    def close(self):
        self._closed.set()
        for s in (self.cmd_srv, self.data_srv):
            s.close()


@pytest.fixture
def fake_tcu():
    server = FakeTCU()
    yield server
    server.close()


def _config(server, **kw):
    return TrignoConfig(host="127.0.0.1", command_port=server.cmd_port, emg_port=server.data_port,
                        connect_timeout_s=2.0, stall_timeout_s=1.0, **kw)


def test_full_session_against_fake_tcu(fake_tcu):
    cfg = _config(fake_tcu, sensors=(2, 5, 9, 16))
    src = DelsysTrignoSource(cfg)
    chunks, states = [], []
    src.on_samples = chunks.append
    src.on_state = lambda s, m: states.append(s)
    src.connect()
    assert src.state is SourceState.CONNECTED
    assert fake_tcu.commands[:2] == ["BACKWARDS COMPATIBILITY ON", "UPSAMPLE ON"]
    src.start()
    t0 = time.perf_counter()
    while src.frames_received < fake_tcu.n_frames and time.perf_counter() - t0 < 10:
        time.sleep(0.01)
    src.stop()
    src.disconnect()
    assert "START" in fake_tcu.commands and "STOP" in fake_tcu.commands and fake_tcu.commands[-1] == "QUIT"
    data = np.concatenate(chunks)
    assert data.shape == (fake_tcu.n_frames, 4) and data.dtype == np.float32
    frames = np.arange(fake_tcu.n_frames)[:, None]
    expected = (frames * FRAME_CHANNELS + (np.array(cfg.sensors) - 1)[None, :]) * 1e-3 * cfg.scale_to_mv
    np.testing.assert_allclose(data, expected, rtol=1e-5, atol=1e-4)    # right sensor columns, volts -> mV
    assert src.state is SourceState.DISCONNECTED and SourceState.STREAMING in states


def test_unreachable_tcu_fails_fast_with_a_message():
    probe = socket.socket(); probe.bind(("127.0.0.1", 0)); port = probe.getsockname()[1]; probe.close()
    src = DelsysTrignoSource(TrignoConfig(host="127.0.0.1", command_port=port, emg_port=port + 1, connect_timeout_s=1.0))
    t0 = time.perf_counter()
    with pytest.raises(SourceError, match="not reachable"):
        src.connect()
    assert time.perf_counter() - t0 < 5 and src.state is SourceState.ERROR


def test_stalled_stream_is_reported_as_connection_lost():
    server = FakeTCU(send_data=False)
    try:
        src = DelsysTrignoSource(_config(server))
        states = []
        src.on_state = lambda s, m: states.append((s, m))
        src.connect()
        src.start()
        t0 = time.perf_counter()
        while src.state is SourceState.STREAMING and time.perf_counter() - t0 < 5:
            time.sleep(0.02)
        assert src.state is SourceState.ERROR and "no EMG data" in states[-1][1]
        src.disconnect()
    finally:
        server.close()


def test_config_validation():
    with pytest.raises(SourceError):
        TrignoConfig(sensors=(1, 2, 3)).validate()
    with pytest.raises(SourceError):
        TrignoConfig(sensors=(1, 2, 3, 17)).validate()
    with pytest.raises(SourceError):
        TrignoConfig(sensors=(1, 1, 2, 3)).validate()
    TrignoConfig().validate()

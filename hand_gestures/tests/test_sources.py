"""File and replay signal sources (Qt-free)."""
import time

import numpy as np
import pytest

from app.sources import FileSignalSource, ReplaySignalSource, SourceError, SourceKind, SourceState
from network import config as C


def test_file_source_delivers_the_recording_at_once(synthetic_recording_path, synthetic_data):
    src = FileSignalSource(synthetic_recording_path)
    states, chunks, progress = [], [], []
    src.on_state = lambda s, m: states.append(s)
    src.on_samples = chunks.append
    src.on_progress = progress.append
    assert src.kind is SourceKind.FILE and src.state is SourceState.DISCONNECTED
    src.connect()
    assert src.state is SourceState.CONNECTED and src.fs == C.FS and src.n_channels == 4
    assert src.channel_names == ["CH1", "CH2", "CH3", "CH4"]
    src.start()
    assert len(chunks) == 1 and chunks[0].shape == synthetic_data.shape and chunks[0].dtype == np.float32
    np.testing.assert_allclose(chunks[0], synthetic_data, atol=1e-6)
    assert progress == [len(synthetic_data)]
    assert states == [SourceState.CONNECTING, SourceState.CONNECTED, SourceState.STREAMING, SourceState.CONNECTED]
    src.disconnect()
    assert src.state is SourceState.DISCONNECTED and src.recording is None


def test_file_source_reports_bad_files(tmp_path):
    bad = tmp_path / "bad.mat"
    bad.write_text("nope")
    src = FileSignalSource(bad)
    with pytest.raises(SourceError):
        src.connect()
    assert src.state is SourceState.ERROR and "MATLAB" in src.last_message
    with pytest.raises(SourceError):
        FileSignalSource(tmp_path / "missing.mat").connect()


def _collect_replay(src, timeout=10.0):
    chunks, states = [], []
    src.on_samples = chunks.append
    src.on_state = lambda s, m: states.append((s, m))
    src.connect()
    src.start()
    t0 = time.perf_counter()
    while src.state is SourceState.STREAMING and time.perf_counter() - t0 < timeout:
        time.sleep(0.01)
    return chunks, states


def test_replay_source_streams_in_order_and_finishes(synthetic_recording, synthetic_data):
    src = ReplaySignalSource(synthetic_recording, chunk_s=0.01, speed=50.0)
    assert src.kind is SourceKind.REPLAY
    chunks, states = _collect_replay(src)
    assert src.state is SourceState.CONNECTED and states[-1][1] == "replay finished"
    assert all(c.ndim == 2 and c.shape[1] == 4 and c.dtype == np.float32 for c in chunks)
    assert len(chunks) > 3                                          # delivered in several chunks, not at once
    np.testing.assert_allclose(np.concatenate(chunks), synthetic_data, atol=1e-6)
    src.stop()                                                      # idempotent after finishing
    src.disconnect()
    assert src.state is SourceState.DISCONNECTED


def test_replay_source_loops_and_stops(synthetic_recording):
    src = ReplaySignalSource(synthetic_recording, chunk_s=0.005, speed=200.0, loop=True)
    total = []
    src.on_samples = lambda c: total.append(len(c))
    src.connect()
    src.start()
    t0 = time.perf_counter()
    while sum(total) < 1.5 * synthetic_recording.n_samples and time.perf_counter() - t0 < 10:
        time.sleep(0.01)
    src.stop()
    assert sum(total) > synthetic_recording.n_samples                # it wrapped around
    assert src.state is SourceState.CONNECTED and src._thread is None

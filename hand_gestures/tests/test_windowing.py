"""Offline and live windowing produce exactly the same windows."""
import numpy as np
import pytest

from network import config as C
from network.streaming import StreamWindower, window_starts


def test_window_starts_basic():
    assert window_starts(5000).tolist() == [0, 1000, 2000, 3000]
    assert window_starts(2000).tolist() == [0]
    assert window_starts(1999).tolist() == []
    assert window_starts(1_280_000).shape[0] == (1_280_000 - C.WINDOW_SAMPLES) // C.HOP_SAMPLES + 1


@pytest.mark.parametrize("chunk_sizes", [[37, 1, 2500, 100, 999, 1, 1362], [100] * 70, [7000], [2000, 1000, 1000, 3000]])
def test_stream_windower_matches_offline_slicing(chunk_sizes):
    rng = np.random.default_rng(1)
    stream = rng.standard_normal((sum(chunk_sizes), C.N_CHANNELS)).astype(np.float32)
    w = StreamWindower()
    got = []
    pos = 0
    for n in chunk_sizes:
        got.extend(w.push(stream[pos:pos + n]))
        pos += n
    starts = [s for s, _ in got]
    assert starts == window_starts(len(stream)).tolist()
    for s, window in got:
        assert window.shape == (C.WINDOW_SAMPLES, C.N_CHANNELS)
        assert window.dtype == np.float32
        np.testing.assert_array_equal(window, stream[s:s + C.WINDOW_SAMPLES])
    assert w.total_samples == len(stream)


def test_stream_windower_bounded_memory_and_reset():
    w = StreamWindower()
    for _ in range(50):
        w.push(np.zeros((500, C.N_CHANNELS), dtype=np.float32))
    assert w._buf.shape[0] <= C.WINDOW_SAMPLES + 500
    w.reset()
    assert w.total_samples == 0 and w.push(np.zeros((10, C.N_CHANNELS))) == []


def test_stream_windower_rejects_wrong_channel_count():
    with pytest.raises(ValueError):
        StreamWindower().push(np.zeros((10, 3)))

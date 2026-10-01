"""Turning a sample stream into model windows (Qt-free).

Both the offline path (a whole recording) and the live path (chunks arriving from a device) must
produce exactly the same windows: ``WINDOW_SAMPLES`` samples starting every ``HOP_SAMPLES`` samples,
counted from the first sample of the stream. ``window_starts`` gives the start indices for a complete
array; ``StreamWindower`` produces the same windows incrementally from chunks of any size.
"""
from __future__ import annotations

import numpy as np

from network import config as C


def window_starts(n_samples: int, window: int = C.WINDOW_SAMPLES, hop: int = C.HOP_SAMPLES) -> np.ndarray:
    """Start indices of all complete windows in a stream of `n_samples` samples (empty if too short)."""
    if n_samples < window:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, n_samples - window + 1, hop, dtype=np.int64)


class StreamWindower:
    """Incremental windowing: feed sample chunks, get back complete windows.

    >>> w = StreamWindower()
    >>> windows = w.push(chunk)          # chunk: [n_new, n_channels]
    >>> for start_sample, window in windows:  # window: float32 [WINDOW_SAMPLES, n_channels]
    ...     ...
    The windows are identical to ``stream[s:s + window]`` for ``s`` in ``window_starts(len(stream))``.
    Memory use is bounded: only the samples still needed for future windows are kept.
    """

    def __init__(self, window: int = C.WINDOW_SAMPLES, hop: int = C.HOP_SAMPLES, n_channels: int = C.N_CHANNELS):
        self.window, self.hop, self.n_channels = int(window), int(hop), int(n_channels)
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros((0, self.n_channels), dtype=np.float32)
        self._buf_start = 0            # absolute index of self._buf[0]
        self._next_start = 0           # absolute index of the next window to emit
        self.total_samples = 0

    def push(self, chunk) -> list[tuple[int, np.ndarray]]:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        if chunk.ndim != 2 or chunk.shape[1] != self.n_channels:
            raise ValueError(f"chunk must be [n_samples, {self.n_channels}], got shape {chunk.shape}")
        if len(chunk) == 0:
            return []
        self._buf = np.concatenate([self._buf, chunk])
        self.total_samples += len(chunk)
        out = []
        while self._next_start + self.window <= self.total_samples:
            i = self._next_start - self._buf_start
            out.append((self._next_start, self._buf[i:i + self.window].copy()))
            self._next_start += self.hop
        drop = self._next_start - self._buf_start                 # samples no future window needs
        if drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start = self._next_start
        return out

    @property
    def samples_until_next_window(self) -> int:
        return max(0, self._next_start + self.window - self.total_samples)

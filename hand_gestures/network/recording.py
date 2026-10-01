"""Reading, validating and writing EMG recordings (Qt-free).

File format: MATLAB v5 ``.mat`` with the same variables as the Özdemir dataset files, so a recording
saved by the desktop app opens with ``scipy.io.loadmat``, in MATLAB, and in this project's loaders:

    data          [n_samples, n_channels] float, mV          fs            sampling rate in Hz
    iD            participant id (0 if unknown)              isi           1000 / fs  (ms)
    isi_units     'ms'                                       labels        channel names
    start_sample  0                                          units         ['mV', ...]

plus a few extra variables ignored by the dataset tooling: ``format_version``, ``timestamp`` (ISO-8601),
``source`` ('delsys_trigno' | 'replay' | 'file' | ...), ``source_config`` (JSON), ``n_samples``,
``duration_s``, ``app``. ``.npz`` (keys data, fs, optional channel_names) and header-less ``.csv`` files
(the dataset's csv layout) can be read as well; ``.mat`` is the only write format.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio

from network import config as C

FORMAT_VERSION = 1


class RecordingFormatError(ValueError):
    """The file cannot be used as an EMG recording for this project (wrong shape, type, content)."""


@dataclass
class Recording:
    data: np.ndarray                      # float32 [n_samples, n_channels], mV
    fs: float
    channel_names: list[str]
    units: list[str]
    meta: dict = field(default_factory=dict)
    path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    def describe(self) -> str:
        name = self.path.name if self.path else "in-memory recording"
        return f"{name}: {self.n_channels} channels, {self.fs:g} Hz, {self.n_samples:,} samples ({self.duration_s:.1f} s)"


def default_channel_names(n_channels: int = C.N_CHANNELS) -> list[str]:
    return [f"CH{i + 1}" for i in range(n_channels)]


def _strings(value) -> list[str]:
    """MAT char arrays / cell arrays / numpy string arrays -> list of python str."""
    return [str(s).strip() for s in np.atleast_1d(value).ravel() if str(s).strip()]


def _scalar(value, default=None):
    arr = np.ravel(value)
    return default if arr.size == 0 else arr[0].item() if hasattr(arr[0], "item") else arr[0]


def _finalize(data, fs, channel_names, units, meta, path, n_channels, min_samples) -> Recording:
    """Shared validation for every input format. Raises RecordingFormatError on hard errors."""
    warnings = []
    data = np.asarray(data)
    if data.dtype.kind not in "fiu":
        raise RecordingFormatError(f"'data' must be numeric, got dtype {data.dtype}")
    data = np.squeeze(data)
    if data.ndim != 2:
        raise RecordingFormatError(f"'data' must be a 2-D array [samples, channels], got shape {data.shape}")
    if data.shape[1] != n_channels and data.shape[0] == n_channels:
        data = data.T
        warnings.append("data was stored as [channels, samples] and has been transposed")
    if data.shape[1] != n_channels:
        raise RecordingFormatError(f"expected {n_channels} EMG channels, the file has {data.shape[1]} (shape {data.shape})")
    if data.shape[0] < min_samples:
        raise RecordingFormatError(f"recording too short: {data.shape[0]} samples, at least {min_samples} "
                                   f"(one {C.WINDOW_S:g} s window at {C.FS} Hz) are needed")
    if not np.isfinite(data).all():
        raise RecordingFormatError("recording contains NaN or infinite values")
    if fs is None:
        fs = float(C.FS)
        warnings.append(f"no sampling frequency stored in the file; assuming {C.FS} Hz")
    else:
        fs = float(fs)
        if not np.isfinite(fs) or fs <= 0:
            raise RecordingFormatError(f"invalid sampling frequency {fs!r}")
        if abs(fs - C.FS) > 1e-6:
            warnings.append(f"sampling frequency is {fs:g} Hz but the model was trained at {C.FS} Hz; "
                            f"windows are cut by sample count, so their duration will be {C.WINDOW_SAMPLES / fs:.2f} s")
    channel_names = list(channel_names) if channel_names and len(channel_names) == n_channels else default_channel_names(n_channels)
    units = list(units) if units and len(units) == n_channels else ["mV"] * n_channels
    return Recording(data=np.ascontiguousarray(data, dtype=np.float32), fs=fs, channel_names=channel_names,
                     units=units, meta=dict(meta or {}), path=Path(path) if path else None, warnings=warnings)


def load_recording_file(path, n_channels: int = C.N_CHANNELS, min_samples: int = C.WINDOW_SAMPLES) -> Recording:
    """Load a .mat / .npz / .csv recording and validate it. Raises RecordingFormatError with a plain message."""
    path = Path(path)
    if not path.exists():
        raise RecordingFormatError(f"file not found: {path}")
    suffix = path.suffix.lower()
    fs, names, units, meta = None, None, None, {}
    if suffix == ".mat":
        try:
            mat = sio.loadmat(path)
        except Exception as e:                                    # scipy raises several types for non-MAT content
            raise RecordingFormatError(f"not a readable MATLAB file ({e.__class__.__name__}: {e})") from e
        keys = [k for k in mat if not k.startswith("__")]
        if "data" not in mat:
            raise RecordingFormatError(f"no 'data' variable in the MAT file (variables: {keys})")
        data = mat["data"]
        if "fs" in mat:
            fs = _scalar(mat["fs"])
        if "labels" in mat:
            names = _strings(mat["labels"])
        if "units" in mat:
            units = _strings(mat["units"])
        for key in ("iD", "format_version", "n_samples", "duration_s"):
            if key in mat:
                meta[key] = _scalar(mat[key])
        for key in ("timestamp", "source", "app", "isi_units"):
            if key in mat:
                meta[key] = " ".join(_strings(mat[key]))
        if "source_config" in mat:
            try:
                meta["source_config"] = json.loads(" ".join(_strings(mat["source_config"])) or "{}")
            except json.JSONDecodeError:
                meta["source_config"] = " ".join(_strings(mat["source_config"]))
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            if "data" not in z:
                raise RecordingFormatError(f"no 'data' array in the npz file (arrays: {list(z.keys())})")
            data = z["data"]
            fs = float(z["fs"]) if "fs" in z else None
            names = _strings(z["channel_names"]) if "channel_names" in z else None
    elif suffix == ".csv":
        try:
            data = np.loadtxt(path, delimiter=",")
        except ValueError as e:
            raise RecordingFormatError(f"csv could not be parsed as numbers: {e}") from e
    else:
        raise RecordingFormatError(f"unsupported file type '{suffix}': use .mat, .npz or .csv")
    return _finalize(data, fs, names, units, meta, path, n_channels, min_samples)


def validate_recording(rec: Recording, n_channels: int = C.N_CHANNELS, min_samples: int = C.WINDOW_SAMPLES) -> list[str]:
    """Re-check an in-memory Recording (e.g. before saving). Returns the warnings, raises on hard errors."""
    checked = _finalize(rec.data, rec.fs, rec.channel_names, rec.units, rec.meta, rec.path, n_channels, min_samples)
    return checked.warnings


def default_recording_path(source: str, folder: Path | None = None) -> Path:
    folder = Path(folder) if folder else C.PROJECT_ROOT / "recordings"
    return folder / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{source}.mat"


def save_recording(path, data, fs: float = C.FS, channel_names=None, units=None, participant_id: int = 0,
                   source: str = "app", source_config: dict | None = None, meta: dict | None = None) -> Path:
    """Write a recording in the dataset-compatible MAT format described in the module docstring."""
    path = Path(path)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != C.N_CHANNELS:
        raise RecordingFormatError(f"data must be [n_samples, {C.N_CHANNELS}], got shape {data.shape}")
    if data.shape[0] == 0:
        raise RecordingFormatError("nothing to save: the recording is empty")
    names = list(channel_names) if channel_names else default_channel_names(data.shape[1])
    units = list(units) if units else ["mV"] * data.shape[1]
    mat = {
        "data": data, "fs": float(fs), "iD": int(participant_id), "isi": 1000.0 / float(fs), "isi_units": "ms",
        "labels": np.array(names), "start_sample": 0, "units": np.array(units),
        "format_version": FORMAT_VERSION, "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source, "source_config": json.dumps(source_config or {}, default=str),
        "n_samples": int(data.shape[0]), "duration_s": data.shape[0] / float(fs), "app": "hand_gestures app",
    }
    for key, value in (meta or {}).items():
        if key not in mat:
            mat[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(path, mat)
    return path

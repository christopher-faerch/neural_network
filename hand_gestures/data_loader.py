"""Loading, labelling, windowing, splitting and normalising the sEMG dataset.

Pipeline (identical for raw and filtered signals):

    40 MAT files (1,280,000 x 4, 2 kHz)
        -> label 6 s gesture segments from the paper's fixed timeline
        -> cut 1 s windows with 50 % overlap inside each segment
        -> subject-independent train / val / test split (fixed seed)
        -> per-window mean removal + per-channel scaling (train statistics only)
        -> PyTorch tensors  [n_windows, 4 channels, 2000 samples]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, TensorDataset

import config as C


# --------------------------------------------------------------------------- 1. loading
def load_recording(subject: int, signal_type: str) -> np.ndarray:
    """Return the 640 s recording of one participant as float32 array [n_samples, 4]."""
    assert signal_type in C.SIGNAL_TYPES, signal_type
    path = C.DATA_ROOT / signal_type / "mat" / f"{subject}_{signal_type}.mat"
    mat = sio.loadmat(path)
    data = mat["data"].astype(np.float32)
    assert data.shape == (C.RECORDING_SAMPLES, C.N_CHANNELS), f"{path}: unexpected shape {data.shape}"
    assert int(mat["fs"].ravel()[0]) == C.FS, f"{path}: unexpected sampling rate"
    assert int(mat["iD"].ravel()[0]) == subject, f"{path}: iD does not match file name"
    return data


# --------------------------------------------------------------------------- 2. labelling + windowing
def segment_start_sample(cycle: int, gesture: int) -> int:
    """First sample of the 6 s segment in which gesture `gesture` is performed during cycle `cycle`."""
    start_s = cycle * C.CYCLE_PERIOD_S + C.FIRST_GESTURE_OFFSET_S + gesture * C.GESTURE_PERIOD_S
    return start_s * C.FS


def window_start_sample(cycle: int, gesture: int, position: int) -> int:
    """First sample of window `position` (0-based) inside the segment of `gesture` in `cycle`."""
    return segment_start_sample(cycle, gesture) + int(C.SEGMENT_START_OFFSET_S * C.FS) + position * C.HOP_SAMPLES


def make_windows(recording: np.ndarray, subject: int):
    """Cut fixed-length windows from the 50 labelled gesture segments of one recording.

    Windows start SEGMENT_START_OFFSET_S after the cue and are cut with hop HOP_SAMPLES up to the
    end of the 6 s segment, so no window crosses a segment border.

    Returns
        X        float32 [n_windows, channels, WINDOW_SAMPLES]
        y        int64   [n_windows]            gesture class 0..9
        meta     int64   [n_windows, 4]         columns: subject, gesture, cycle, position-in-segment
    """
    X, meta = [], []
    for cycle in range(C.N_CYCLES):
        for gesture in range(C.N_CLASSES):
            start = window_start_sample(cycle, gesture, 0)
            end = segment_start_sample(cycle, gesture) + C.GESTURE_DURATION_S * C.FS - int(C.SEGMENT_END_TRIM_S * C.FS)
            for position, w0 in enumerate(range(start, end - C.WINDOW_SAMPLES + 1, C.HOP_SAMPLES)):
                X.append(recording[w0:w0 + C.WINDOW_SAMPLES].T)      # -> [channels, time]
                meta.append((subject, gesture, cycle, position))
    X = np.stack(X).astype(np.float32)
    meta = np.asarray(meta, dtype=np.int64)
    return X, meta[:, 1].copy(), meta


# gestures preceded by a plain 4 s rest (7 and 9 follow the abduction / supination "hold" instead)
RESTED_GESTURES = (0, 1, 2, 3, 4, 5, 6, 8)


def rest_rms(recording: np.ndarray) -> np.ndarray:
    """Per-channel RMS of the participant's rest posture: the middle 2 s of every 4 s rest before a
    gesture (8 per cycle, 40 in total = 80 s). The mean of each 1 s chunk is removed first, exactly as
    for the windows. Used by the 'rest_rms' normalisation (calibration setting). Shape [channels]."""
    chunks = []
    for cycle in range(C.N_CYCLES):
        for gesture in RESTED_GESTURES:
            a = segment_start_sample(cycle, gesture) - 3 * C.FS          # 3 s .. 1 s before the cue
            seg = recording[a:a + 2 * C.FS].reshape(2, C.FS, C.N_CHANNELS)
            chunks.append(seg - seg.mean(axis=1, keepdims=True))
    chunks = np.concatenate(chunks)                                       # [80, FS, channels]
    return np.sqrt((chunks ** 2).mean(axis=(0, 1))).astype(np.float32)


def build_windows(signal_type: str, subjects=None, verbose: bool = True):
    """Window the recordings of all (or the given) participants.

    Returns X, y, meta and rest_scale: a float32 array [n_windows, channels] holding, for every window,
    the rest-posture RMS of the participant it came from (only used when NORMALIZATION == 'rest_rms').
    """
    subjects = list(range(1, C.N_SUBJECTS + 1)) if subjects is None else list(subjects)
    Xs, ys, metas, rests = [], [], [], []
    for s in subjects:
        rec = load_recording(s, signal_type)
        X, y, meta = make_windows(rec, s)
        Xs.append(X); ys.append(y); metas.append(meta)
        rests.append(np.repeat(rest_rms(rec)[None, :], len(X), axis=0))
    X = np.concatenate(Xs); y = np.concatenate(ys); meta = np.concatenate(metas); rest_scale = np.concatenate(rests)
    if verbose:
        print(f"[{signal_type}] windows: X {X.shape} ({X.nbytes / 1e6:.0f} MB), "
              f"{len(subjects)} subjects, {C.windows_per_segment()} windows per 6 s segment "
              f"(windows from {C.SEGMENT_START_OFFSET_S:.1f} s to {C.GESTURE_DURATION_S - C.SEGMENT_END_TRIM_S:.1f} s after the cue), "
              f"{np.bincount(y).tolist()} windows per class")
    return X, y, meta, rest_scale


# --------------------------------------------------------------------------- 3. subject-independent split
@dataclass
class SubjectSplit:
    train: list
    val: list
    test: list

    def as_dict(self):
        return asdict(self)


def split_subjects(seed: int = C.SPLIT_SEED) -> SubjectSplit:
    """Shuffle participant IDs with a fixed seed and assign whole participants to one set each."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(1, C.N_SUBJECTS + 1))
    n_tr, n_va = C.N_TRAIN_SUBJECTS, C.N_VAL_SUBJECTS
    split = SubjectSplit(train=sorted(ids[:n_tr].tolist()),
                         val=sorted(ids[n_tr:n_tr + n_va].tolist()),
                         test=sorted(ids[n_tr + n_va:].tolist()))
    assert not (set(split.train) & set(split.val)) and not (set(split.train) & set(split.test)) \
        and not (set(split.val) & set(split.test)), "subject overlap between splits"
    return split


def cv_folds(n_folds: int = C.CV_N_FOLDS, seed: int = C.CV_SEED) -> list:
    """Leave-participants-out folds. Fold k tests on group k, validates on group k+1, trains on the rest."""
    rng = np.random.default_rng(seed)
    groups = np.array_split(rng.permutation(np.arange(1, C.N_SUBJECTS + 1)), n_folds)
    folds = []
    for k in range(n_folds):
        test, val = groups[k], groups[(k + 1) % n_folds]
        train = np.concatenate([g for i, g in enumerate(groups) if i not in (k, (k + 1) % n_folds)])
        folds.append(SubjectSplit(train=sorted(train.tolist()), val=sorted(val.tolist()), test=sorted(test.tolist())))
    assert sorted(sum((f.test for f in folds), [])) == list(range(1, C.N_SUBJECTS + 1)), "every participant tests once"
    return folds


def print_split(split: SubjectSplit):
    print(f"Subject-independent split (seed {C.SPLIT_SEED}):")
    print(f"  train ({len(split.train)}): {split.train}")
    print(f"  val   ({len(split.val)}): {split.val}")
    print(f"  test  ({len(split.test)}): {split.test}")


def select(X, y, meta, subjects, *extra):
    mask = np.isin(meta[:, 0], subjects)
    return (X[mask], y[mask], meta[mask]) + tuple(e[mask] for e in extra)


# --------------------------------------------------------------------------- 4. normalisation (train statistics only)
def remove_window_mean(X: np.ndarray) -> np.ndarray:
    """Subtract each window's per-channel mean (removes DC offset / electrode drift)."""
    return X - X.mean(axis=2, keepdims=True)


def compute_channel_std(X_train: np.ndarray) -> np.ndarray:
    """Per-channel standard deviation over all TRAINING windows after mean removal. Shape [channels]."""
    Xc = remove_window_mean(X_train)
    return Xc.std(axis=(0, 2)).astype(np.float32)


def normalize(X: np.ndarray, channel_std: np.ndarray) -> np.ndarray:
    """x_norm[c, t] = (x[c, t] - mean_t x[c, .]) / channel_std[c]   (channel_std from training set)."""
    return remove_window_mean(X) / channel_std[None, :, None]


# --------------------------------------------------------------------------- 5. torch datasets
def to_loader(X, y, batch_size=C.BATCH_SIZE, shuffle=False, seed=C.SEED):
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(y))
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen,
                      num_workers=C.NUM_WORKERS, pin_memory=torch.cuda.is_available())


def prepare_split_arrays(X, y, meta, split: SubjectSplit, rest_scale=None, normalization=None) -> dict:
    """Select the windows of each set and normalise them.

    normalization "train_std": per-window mean removal, then one scale per channel from the TRAINING
    windows (strict). "rest_rms": per-window mean removal, then each participant's own rest-posture RMS
    per channel (calibration setting; `rest_scale` from build_windows is required).
    Returns {"split", "channel_std", "normalization", "train": (X, y, meta), "val": (...), "test": (...)}.
    """
    normalization = normalization or C.NORMALIZATION
    assert normalization in ("train_std", "rest_rms"), normalization
    if rest_scale is None:
        rest_scale = np.ones((len(X), C.N_CHANNELS), dtype=np.float32)
    parts = {name: select(X, y, meta, getattr(split, name), rest_scale) for name in ("train", "val", "test")}
    channel_std = compute_channel_std(parts["train"][0])
    out = {"split": split, "channel_std": channel_std, "normalization": normalization}
    for name, (Xp, yp, mp, rp) in parts.items():
        if normalization == "train_std":
            out[name] = (normalize(Xp, channel_std), yp, mp)
        else:
            out[name] = (remove_window_mean(Xp) / rp[:, :, None], yp, mp)
    return out


def prepare_data(signal_type: str, split: SubjectSplit | None = None, verbose: bool = True, normalization=None):
    """Full pipeline. Returns a dict with normalised arrays per split, meta, channel_std and split."""
    split = split or split_subjects()
    X, y, meta, rest_scale = build_windows(signal_type, verbose=verbose)
    out = prepare_split_arrays(X, y, meta, split, rest_scale, normalization)
    channel_std = out["channel_std"]
    if verbose:
        print_split(split)
        print(f"  normalisation: {out['normalization']}; channel std from training windows (mV): "
              f"{[round(float(v), 4) for v in channel_std]}")
        for name in ("train", "val", "test"):
            print(f"  {name:5s}: X {out[name][0].shape}, class counts {np.bincount(out[name][1]).tolist()}")
    return out


def save_split(split: SubjectSplit, channel_std: np.ndarray, path, normalization=None):
    with open(path, "w") as f:
        json.dump({"split": split.as_dict(), "channel_std": channel_std.tolist(),
                   "normalization": normalization or C.NORMALIZATION, "seed": C.SPLIT_SEED}, f, indent=2)


if __name__ == "__main__":
    # quick manual check:  python data_loader.py
    data = prepare_data("filtered")

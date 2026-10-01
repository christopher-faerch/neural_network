"""The inference interface used by the desktop app (and by predict.py).

    clf = GestureClassifier()                    # picks CUDA if available
    info = clf.load_model("raw")                 # results/raw/best_model_seed*.pt -> 5-seed ensemble
    result = clf.predict(window)                 # window: [2000, 4] (or [4, 2000]) in mV
    result == {"class_index": 3, "class_name": "ulnar_dev", "confidence": 0.97, "probabilities": array([...10])}

Preprocessing is the training pipeline itself (``data_loader.normalize``: per-window mean removal, then
division by the training-set channel std stored in the checkpoint). Nothing here imports Qt.

Two settings: *strict* (the loaded weights and statistics as trained) and *calibrated* (``calibrate()``
re-estimates the BatchNorm statistics on unlabelled windows of the current user; both model sets are kept,
``predict(..., calibrated=True)`` selects the calibrated one).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from network import config as C
from network.data_loader import normalize
from network.model import adapt_batchnorm, load_ensemble, predict_proba as _ensemble_proba


class InferenceError(Exception):
    """Base class for everything the app should show as a message instead of crashing."""


class ModelNotFoundError(InferenceError):
    """No checkpoint folder / files, or predict() called before load_model()."""


class CheckpointIncompatibleError(InferenceError):
    """The checkpoint does not match the current network / preprocessing configuration."""


class WindowShapeError(InferenceError):
    """The input is not a [WINDOW_SAMPLES, N_CHANNELS] window (or a batch of them)."""


@dataclass(frozen=True)
class ModelInfo:
    name: str
    signal_type: str
    checkpoint_dir: Path
    checkpoint_files: tuple[str, ...]
    seeds: tuple[int, ...]
    epochs: tuple[int, ...]
    val_acc: tuple[float, ...]
    channel_std_mv: tuple[float, ...]
    normalization: str
    device: str
    n_channels: int = C.N_CHANNELS
    window_samples: int = C.WINDOW_SAMPLES
    fs: int = C.FS
    class_names: tuple[str, ...] = tuple(C.GESTURES)

    @property
    def ensemble_size(self) -> int:
        return len(self.checkpoint_files)


def make_prediction(probabilities: np.ndarray) -> dict:
    """Probability vector -> the dict handed to the GUI."""
    probs = np.asarray(probabilities, dtype=np.float32).ravel()
    idx = int(probs.argmax())
    return {"class_index": idx, "class_name": C.GESTURES[idx], "confidence": float(probs[idx]), "probabilities": probs}


def causal_smooth(probs: np.ndarray, n: int) -> np.ndarray:
    """Average each row with the previous n-1 rows (consecutive windows of one stream)."""
    probs = np.asarray(probs)
    out = probs.copy()
    for j in range(len(probs)):
        out[j] = probs[max(0, j - n + 1):j + 1].mean(axis=0)
    return out


class PredictionSmoother:
    """Optional smoothing over the last n predictions. mode 'mean': average the probability vectors;
    mode 'vote': majority vote over the predicted classes (ties -> most recent), confidence = mean
    probability of the winning class. The raw per-window prediction stays available to the caller."""

    def __init__(self, n: int = C.SMOOTH_WINDOWS, mode: str = "mean"):
        if mode not in ("mean", "vote"):
            raise ValueError("mode must be 'mean' or 'vote'")
        self.n, self.mode = max(1, int(n)), mode
        self._history: deque[np.ndarray] = deque(maxlen=self.n)

    def reset(self) -> None:
        self._history.clear()

    def update(self, probabilities: np.ndarray) -> dict:
        probs = np.asarray(probabilities, dtype=np.float32).ravel()
        self._history.append(probs)
        stack = np.stack(self._history)
        if self.mode == "mean":
            return make_prediction(stack.mean(axis=0))
        votes = stack.argmax(axis=1)
        counts = np.bincount(votes, minlength=C.N_CLASSES)
        best = np.flatnonzero(counts == counts.max())
        winner = int(best[-1]) if len(best) == 1 else int(votes[-1]) if votes[-1] in best else int(best[-1])
        return {"class_index": winner, "class_name": C.GESTURES[winner],
                "confidence": float(stack[:, winner].mean()), "probabilities": stack.mean(axis=0)}


class GestureClassifier:
    """Loads the trained ensemble and classifies EMG windows. The only model API the app sees."""

    def __init__(self, device=None):
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: list = []
        self._calibrated: list = []
        self._channel_std: np.ndarray | None = None
        self._info: ModelInfo | None = None

    # ------------------------------------------------------------------ loading
    @staticmethod
    def available_models(results_root=None) -> list[str]:
        """Signal types (sub-folders of results/) that contain trained checkpoints, 'raw' first."""
        root = Path(results_root) if results_root else C.RESULTS_DIR
        if not root.exists():
            return []
        found = [d.name for d in root.iterdir() if d.is_dir() and any(d.glob("best_model_seed*.pt"))]
        return sorted(found, key=lambda s: (s != "raw", s))

    def load_model(self, signal_type: str = "raw", results_root=None) -> ModelInfo:
        root = Path(results_root) if results_root else C.RESULTS_DIR
        folder = root / signal_type
        files = sorted(folder.glob("best_model_seed*.pt"), key=lambda p: int(p.stem.split("seed")[-1])) if folder.exists() else []
        if not files:
            raise ModelNotFoundError(f"no checkpoints (best_model_seed*.pt) found in {folder}")
        try:
            models, ckpts = load_ensemble(folder, self.device)
        except (RuntimeError, KeyError, ValueError, TypeError, EOFError) as e:
            raise CheckpointIncompatibleError(f"checkpoint in {folder} cannot be loaded into the current network: {e}") from e
        self._check_compatible(ckpts[0], folder)
        self._models, self._calibrated = models, []
        self._channel_std = np.asarray(ckpts[0]["channel_std"], dtype=np.float32)
        signal = str(ckpts[0].get("signal_type", signal_type))
        self._info = ModelInfo(
            name=f"EMGCNN {signal} ensemble ({len(models)} seeds)", signal_type=signal, checkpoint_dir=folder,
            checkpoint_files=tuple(p.name for p in files), seeds=tuple(int(c.get("seed", -1)) for c in ckpts),
            epochs=tuple(int(c.get("epoch", -1)) for c in ckpts), val_acc=tuple(float(c.get("val_acc", float("nan"))) for c in ckpts),
            channel_std_mv=tuple(float(v) for v in self._channel_std), normalization=str(ckpts[0].get("normalization", "train_std")),
            device=str(self.device))
        return self._info

    @staticmethod
    def _check_compatible(ckpt: dict, folder: Path) -> None:
        cfg = ckpt.get("config", {}) or {}
        problems = []
        if cfg.get("N_CHANNELS", C.N_CHANNELS) != C.N_CHANNELS:
            problems.append(f"channels {cfg.get('N_CHANNELS')} != {C.N_CHANNELS}")
        if cfg.get("WINDOW_SAMPLES", C.WINDOW_SAMPLES) != C.WINDOW_SAMPLES:
            problems.append(f"window {cfg.get('WINDOW_SAMPLES')} samples != {C.WINDOW_SAMPLES}")
        if list(cfg.get("GESTURES", C.GESTURES)) != list(C.GESTURES):
            problems.append("gesture list differs from config.GESTURES")
        if str(ckpt.get("normalization", "train_std")) != "train_std":
            problems.append(f"normalisation '{ckpt.get('normalization')}' is not supported for live inference")
        std = np.asarray(ckpt.get("channel_std", []), dtype=np.float32).ravel()
        if std.shape != (C.N_CHANNELS,) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            problems.append("channel_std missing or invalid")
        if problems:
            raise CheckpointIncompatibleError(f"checkpoint in {folder} is incompatible: " + "; ".join(problems))

    @property
    def is_loaded(self) -> bool:
        return bool(self._models)

    @property
    def info(self) -> ModelInfo | None:
        return self._info

    @property
    def class_names(self) -> list[str]:
        return list(C.GESTURES)

    @property
    def has_calibration(self) -> bool:
        return bool(self._calibrated)

    # ------------------------------------------------------------------ preprocessing (the training pipeline)
    def preprocess(self, window) -> np.ndarray:
        """mV window(s) -> normalised float32 [n, N_CHANNELS, WINDOW_SAMPLES], exactly as in training.

        Accepts one window as [WINDOW_SAMPLES, N_CHANNELS] or [N_CHANNELS, WINDOW_SAMPLES], or a batch
        with a leading dimension.
        """
        if self._channel_std is None:
            raise ModelNotFoundError("no model loaded: call load_model() first")
        x = np.asarray(window, dtype=np.float32)
        T, Cn = C.WINDOW_SAMPLES, C.N_CHANNELS
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3:
            raise WindowShapeError(f"expected a window [{T}, {Cn}] or a batch of windows, got shape {x.shape}")
        if x.shape[1:] == (T, Cn):
            x = np.ascontiguousarray(x.transpose(0, 2, 1))        # -> [n, channels, time] like recording[w0:w0+T].T
        elif x.shape[1:] != (Cn, T):
            raise WindowShapeError(f"expected windows of {T} samples x {Cn} channels, got shape {tuple(x.shape[1:])}")
        if not np.isfinite(x).all():
            raise InferenceError("window contains NaN or infinite values")
        return normalize(x, self._channel_std)

    # ------------------------------------------------------------------ prediction
    def predict_proba(self, windows, calibrated: bool = False) -> np.ndarray:
        """-> [n, N_CLASSES] class probabilities (softmax averaged over the ensemble)."""
        if not self.is_loaded:
            raise ModelNotFoundError("no model loaded: call load_model() first")
        models = self._calibrated if (calibrated and self._calibrated) else self._models
        return _ensemble_proba(models, self.preprocess(windows), self.device)

    def predict(self, window, calibrated: bool = False) -> dict:
        """One window -> {"class_index", "class_name", "confidence", "probabilities"}."""
        return make_prediction(self.predict_proba(window, calibrated=calibrated)[0])

    # ------------------------------------------------------------------ calibration (AdaBN)
    def calibrate(self, windows_mv, min_windows: int = 20) -> int:
        """Re-estimate the BatchNorm statistics of every ensemble member on unlabelled windows of the
        current user (AdaBN). Returns the number of windows used. The strict models are kept untouched."""
        if not self.is_loaded:
            raise ModelNotFoundError("no model loaded: call load_model() first")
        X = self.preprocess(windows_mv)
        if len(X) < min_windows:
            raise InferenceError(f"calibration needs at least {min_windows} windows "
                                 f"({min_windows * C.HOP_S:.0f} s of signal), got {len(X)}")
        self._calibrated = [adapt_batchnorm(m, X, self.device) for m in self._models]
        return int(len(X))

    def clear_calibration(self) -> None:
        self._calibrated = []

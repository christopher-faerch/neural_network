"""The inference worker: the only object that touches the GestureClassifier, living in its own QThread.

Streaming:  on_chunk(chunk) -> StreamWindower -> newest complete window -> classifier.predict -> prediction_ready
Batch:      run_batch(data) -> all windows of a recording -> batch_done (used by offline mode)
Loading and calibration also run here so torch/CUDA start-up never blocks the GUI.
"""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from network import config as C
from network.inference import GestureClassifier, InferenceError, PredictionSmoother, causal_smooth
from network.streaming import StreamWindower, window_starts


class InferenceWorker(QObject):
    model_loaded = Signal(object)          # ModelInfo
    model_failed = Signal(str)
    prediction_ready = Signal(object)      # dict: raw, smoothed|None, start_sample, end_sample, latency_ms, calibrated
    batch_progress = Signal(int, int)      # done, total
    batch_done = Signal(object)            # dict: starts, probs, probs_smoothed, calibrated
    batch_failed = Signal(str)
    calibration_done = Signal(int)         # windows used
    calibration_failed = Signal(str)
    windows_dropped = Signal(int)          # total dropped so far
    status = Signal(str)                   # "idle" | "running" | "batch 12 %" | "error" ...
    error = Signal(str)

    def __init__(self, device=None):
        super().__init__()
        self.classifier = GestureClassifier(device=device)
        self.windower = StreamWindower()
        self.smoother = PredictionSmoother(C.SMOOTH_WINDOWS, "mean")
        self.smoothing_enabled = False
        self.use_calibrated = False
        self.dropped = 0
        self._cancel = False
        self._busy = False

    # ------------------------------------------------------------------ model
    @Slot(str, object)
    def load_model(self, signal_type: str, results_root=None) -> None:
        self.status.emit(f"loading {signal_type} model")
        try:
            info = self.classifier.load_model(signal_type, results_root)
        except InferenceError as e:
            self.model_failed.emit(str(e))
            self.status.emit("no model")
            return
        except Exception as e:                        # torch / file-system surprises
            self.model_failed.emit(f"{e.__class__.__name__}: {e}")
            self.status.emit("no model")
            return
        self.use_calibrated = False
        self.reset_stream()
        self.model_loaded.emit(info)
        self.status.emit("idle")

    @Slot(bool, int)
    def set_smoothing(self, enabled: bool, n: int) -> None:
        self.smoothing_enabled = bool(enabled)
        self.smoother = PredictionSmoother(max(1, int(n)), "mean")

    @Slot(bool)
    def set_calibrated(self, enabled: bool) -> None:
        self.use_calibrated = bool(enabled) and self.classifier.has_calibration

    @Slot()
    def reset_stream(self) -> None:
        self.windower.reset()
        self.smoother.reset()
        self.dropped = 0

    @Slot()
    def cancel(self) -> None:
        self._cancel = True

    # ------------------------------------------------------------------ streaming
    @Slot(object)
    def on_chunk(self, chunk) -> None:
        try:
            windows = self.windower.push(chunk)
        except ValueError as e:
            self.error.emit(str(e))
            return
        if not windows or not self.classifier.is_loaded:
            return
        if len(windows) > 1:                           # we are behind: predict the newest window only
            self.dropped += len(windows) - 1
            self.windows_dropped.emit(self.dropped)
        start, window = windows[-1]
        t0 = time.perf_counter()
        try:
            raw = self.classifier.predict(window, calibrated=self.use_calibrated)
        except InferenceError as e:
            self.status.emit("error")
            self.error.emit(f"inference failed: {e}")
            return
        except Exception as e:
            self.status.emit("error")
            self.error.emit(f"inference failed: {e.__class__.__name__}: {e}")
            return
        smoothed = self.smoother.update(raw["probabilities"]) if self.smoothing_enabled else None
        self.status.emit("running")
        self.prediction_ready.emit({"raw": raw, "smoothed": smoothed, "start_sample": int(start),
                                    "end_sample": int(start + C.WINDOW_SAMPLES),
                                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                                    "calibrated": bool(self.use_calibrated)})

    # ------------------------------------------------------------------ batch (offline)
    @Slot(object)
    def run_batch(self, data) -> None:
        if not self.classifier.is_loaded:
            self.batch_failed.emit("no model loaded")
            return
        data = np.asarray(data, dtype=np.float32)
        starts = window_starts(len(data))
        if len(starts) == 0:
            self.batch_failed.emit(f"recording shorter than one window ({C.WINDOW_SAMPLES} samples)")
            return
        self._cancel, self._busy = False, True
        probs = np.zeros((len(starts), C.N_CLASSES))
        step = 256
        try:
            for i in range(0, len(starts), step):
                if self._cancel:
                    self.status.emit("idle")
                    self.batch_failed.emit("cancelled")
                    return
                idx = starts[i:i + step]
                windows = np.stack([data[s:s + C.WINDOW_SAMPLES] for s in idx])
                probs[i:i + len(idx)] = self.classifier.predict_proba(windows, calibrated=self.use_calibrated)
                done = min(i + step, len(starts))
                self.batch_progress.emit(done, len(starts))
                self.status.emit(f"batch {100 * done // len(starts)} %")
        except Exception as e:
            self.status.emit("error")
            self.batch_failed.emit(f"inference failed: {e.__class__.__name__}: {e}")
            return
        finally:
            self._busy = False
        self.status.emit("idle")
        self.batch_done.emit({"starts": starts, "probs": probs,
                              "probs_smoothed": causal_smooth(probs, self.smoother.n), "calibrated": bool(self.use_calibrated)})

    # ------------------------------------------------------------------ calibration
    @Slot(object)
    def calibrate(self, windows) -> None:
        self.status.emit("calibrating")
        try:
            n = self.classifier.calibrate(windows)
        except InferenceError as e:
            self.status.emit("idle")
            self.calibration_failed.emit(str(e))
            return
        except Exception as e:
            self.status.emit("idle")
            self.calibration_failed.emit(f"{e.__class__.__name__}: {e}")
            return
        self.use_calibrated = True
        self.status.emit("idle")
        self.calibration_done.emit(n)

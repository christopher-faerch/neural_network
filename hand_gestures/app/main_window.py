"""MainWindow: wires the control panel, the two views, the acquisition controller and the inference worker."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget)

from app import theme
from app.acquisition import AcquisitionController
from app.control_panel import ControlPanel
from app.delsys import DelsysTrignoSource
from app.gesture_view import GestureView
from app.inference_worker import InferenceWorker
from app.signal_view import SignalView
from app.sources import FileSignalSource, ReplaySignalSource, SourceError, SourceKind, SourceState
from network import config as C
from network.inference import GestureClassifier, make_prediction
from network.recording import Recording, RecordingFormatError, default_recording_path, load_recording_file, save_recording
from network.streaming import window_starts

FILE_FILTER = "EMG recordings (*.mat *.npz *.csv);;All files (*)"


class MainWindow(QMainWindow):
    # requests to the worker thread (queued connections)
    request_load_model = Signal(str, object)
    request_batch = Signal(object)
    request_calibration = Signal(object)
    request_smoothing = Signal(bool, int)
    request_calibrated = Signal(bool)
    request_reset_stream = Signal()

    def __init__(self, results_root: Path | None = None, model: str | None = None, device=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EMG Hand-Gesture Recognition")
        self.resize(1480, 900)
        self.results_root = Path(results_root) if results_root else C.RESULTS_DIR
        self.recording: Recording | None = None            # offline recording
        self.batch: dict | None = None                     # offline inference results
        self.window_index = 0
        self.unsaved: np.ndarray | None = None             # stopped live recording not yet saved
        self.replay_path: Path | None = None
        self.model_info = None

        # ---------------------------------------------------------------- widgets
        self.panel = ControlPanel()
        self.signal_view = SignalView()
        self.gesture_view = GestureView()
        self.views = QStackedWidget()
        self.views.addWidget(self.signal_view)
        self.views.addWidget(self.gesture_view)

        top = QHBoxLayout()
        title = QLabel("EMG Hand-Gesture Recognition")
        title.setProperty("role", "heading")
        self.badge = QLabel("NO SOURCE")
        self.badge.setProperty("role", "badge-off")
        self.view_signal_btn = QPushButton("EMG signal")
        self.view_gesture_btn = QPushButton("Gestures")
        for b in (self.view_signal_btn, self.view_gesture_btn):
            b.setCheckable(True)
        self.view_signal_btn.setChecked(True)
        self.view_signal_btn.clicked.connect(lambda: self.show_view("signal"))
        self.view_gesture_btn.clicked.connect(lambda: self.show_view("gesture"))
        top.addWidget(title)
        top.addSpacing(12)
        top.addWidget(self.badge)
        top.addStretch(1)
        top.addWidget(self.view_signal_btn)
        top.addWidget(self.view_gesture_btn)

        self.status_label = QLabel("ready")
        self.status_label.setProperty("role", "dim")
        self.inference_label = QLabel("")
        self.inference_label.setProperty("role", "dim")
        status = QFrame()
        status.setProperty("role", "statusbar")
        srow = QHBoxLayout(status)
        srow.setContentsMargins(12, 6, 12, 6)
        srow.addWidget(self.status_label, 1)
        srow.addWidget(self.inference_label)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        right.addLayout(top)
        right.addWidget(self.views, 1)
        body = QHBoxLayout()
        body.setContentsMargins(14, 14, 14, 8)
        body.setSpacing(14)
        body.addWidget(self.panel)
        body.addLayout(right, 1)
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addLayout(body, 1)
        outer.addWidget(status)
        self.setCentralWidget(central)

        # ---------------------------------------------------------------- acquisition + inference
        self.controller = AcquisitionController(parent=self)
        self.controller.state_changed.connect(self._on_source_state)
        self.controller.error.connect(self._show_error)
        self.controller.recording_changed.connect(self._on_recording_changed)
        self.controller.progress.connect(self._on_progress)

        self.worker = InferenceWorker(device=device)
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.request_load_model.connect(self.worker.load_model)
        self.request_batch.connect(self.worker.run_batch)
        self.request_calibration.connect(self.worker.calibrate)
        self.request_smoothing.connect(self.worker.set_smoothing)
        self.request_calibrated.connect(self.worker.set_calibrated)
        self.request_reset_stream.connect(self.worker.reset_stream)
        self.controller.chunk_ready.connect(self.worker.on_chunk)
        self.worker.model_loaded.connect(self._on_model_loaded)
        self.worker.model_failed.connect(self._on_model_failed)
        self.worker.prediction_ready.connect(self._on_prediction)
        self.worker.batch_progress.connect(self.panel.set_batch_progress)
        self.worker.batch_done.connect(self._on_batch_done)
        self.worker.batch_failed.connect(self._on_batch_failed)
        self.worker.calibration_done.connect(self._on_calibration_done)
        self.worker.calibration_failed.connect(self._show_error)
        self.worker.windows_dropped.connect(lambda n: self.inference_label.setText(f"inference: running  |  dropped {n} windows"))
        self.worker.status.connect(self._on_inference_status)
        self.worker.error.connect(self._show_error)
        self.thread.start()

        # ---------------------------------------------------------------- panel wiring
        p = self.panel
        p.mode_changed.connect(self._on_mode)
        p.select_file_clicked.connect(self.select_file)
        p.run_clicked.connect(self.run_offline_inference)
        p.play_toggled.connect(self._on_play)
        p.window_index_changed.connect(self.show_offline_window)
        p.replay_as_live_clicked.connect(self.replay_current_file_as_live)
        p.live_source_changed.connect(lambda _k: self._update_badge())
        p.choose_replay_file.connect(self.choose_replay_file)
        p.connect_clicked.connect(self.connect_live)
        p.disconnect_clicked.connect(self.disconnect_live)
        p.start_recording_clicked.connect(self.start_recording)
        p.stop_recording_clicked.connect(self.stop_recording)
        p.save_recording_clicked.connect(self.save_recording)
        p.discard_recording_clicked.connect(self.discard_recording)
        p.calibrate_clicked.connect(self.calibrate)
        p.inference_while_recording_toggled.connect(self._set_inference_enabled)
        p.model_changed.connect(self._on_model_choice)
        p.smoothing_changed.connect(self._on_smoothing)
        p.calibrated_toggled.connect(self.request_calibrated)
        self.signal_view.position_changed.connect(self._on_signal_scrolled)

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(int(C.HOP_S * 1000))
        self.play_timer.timeout.connect(self._play_step)

        # ---------------------------------------------------------------- initial state
        available = GestureClassifier.available_models(self.results_root)
        self.panel.set_models(available, model if model in available else (available[0] if available else None))
        if available:
            self.request_load_model.emit(self.panel.model_box.currentText(), self.results_root)
        else:
            self._on_model_failed(f"no trained model found in {self.results_root} (expected results/<signal>/best_model_seed*.pt); "
                                  f"signal viewing and recording work, gesture prediction is disabled")
        self._on_mode("offline")
        self._update_badge()

    # ================================================================== views / mode
    def show_view(self, name: str) -> None:
        self.views.setCurrentIndex(0 if name == "signal" else 1)
        self.view_signal_btn.setChecked(name == "signal")
        self.view_gesture_btn.setChecked(name == "gesture")

    def _on_mode(self, mode: str) -> None:
        self.mode = mode
        self.play_timer.stop()
        self.panel.set_playing(False)
        if mode == "offline":
            if self.controller.source is not None and self.controller.kind is not SourceKind.FILE:
                self.controller.set_source(None)
            self.gesture_view.clear()
            if self.recording is not None:
                self.signal_view.set_recording(self.recording)
                self.show_offline_window(self.window_index)
            else:
                self.signal_view.clear()
            self._update_signal_info()
        else:
            self.gesture_view.clear()
            if self.controller.source is None or self.controller.kind is SourceKind.FILE:
                self.signal_view.clear()
                self.panel.set_connection("disconnected", "dim", connected=False, streaming=False)
            self.signal_view.set_live(self.controller)
            self._update_signal_info()
        self._update_badge()

    def _update_badge(self) -> None:
        kind = self.controller.kind
        if self.mode == "offline":
            text, role = ("FILE", "badge-file") if self.recording is not None else ("NO SOURCE", "badge-off")
        elif kind is SourceKind.HARDWARE:
            text, role = "LIVE HARDWARE", "badge-live"
        elif kind is SourceKind.REPLAY:
            text, role = "SIMULATED / REPLAY", "badge-replay"
        else:
            pending = self.panel.live_source()
            text, role = ("LIVE HARDWARE (not connected)", "badge-off") if pending == "delsys" else ("SIMULATED / REPLAY (not connected)", "badge-off")
        self.badge.setText(text)
        self.badge.setProperty("role", role)
        theme.repolish(self.badge)

    def _update_signal_info(self) -> None:
        if self.mode == "offline":
            if self.recording is None:
                self.panel.set_signal_info(None, None, None, None)
            else:
                self.panel.set_signal_info(self.recording.n_channels, self.recording.fs, self.recording.n_samples,
                                           len(window_starts(self.recording.n_samples)))
        else:
            src = self.controller.source
            if src is None:
                self.panel.set_signal_info(None, None, None, None)
            else:
                total = self.controller.total_samples
                self.panel.set_signal_info(src.n_channels, src.fs, total, len(window_starts(total)))

    # ================================================================== status / errors
    def _show_error(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_label.setProperty("role", "error")
        theme.repolish(self.status_label)

    def _show_status(self, message: str, role: str = "dim") -> None:
        self.status_label.setText(message)
        self.status_label.setProperty("role", role)
        theme.repolish(self.status_label)

    def _on_inference_status(self, text: str) -> None:
        self.panel.set_inference_state(text)
        self.inference_label.setText(f"inference: {text}")

    # ================================================================== model
    def _on_model_choice(self, signal_type: str) -> None:
        self.batch = None
        self.panel.set_windows(0)
        self.gesture_view.clear()
        self.request_load_model.emit(signal_type, self.results_root)

    def _on_model_loaded(self, info) -> None:
        self.model_info = info
        note = ""
        if info.signal_type == "filtered":
            note = ("Trained on the dataset's BIOPAC-filtered signals (5-500 Hz band-pass + 50 Hz notch). "
                    "Hardware streams are unfiltered: prefer the raw model for live use.")
        self.panel.set_model_status(info.name, f"✓ loaded ({info.device}, epochs {list(info.epochs)})", "ok", note)
        self.panel.set_calibration_available(False)
        self._show_status(f"model loaded: {info.name} from {info.checkpoint_dir}")
        self.panel.set_offline_ready(self.recording is not None)
        if self.mode == "live" and self.controller.kind is SourceKind.HARDWARE and info.signal_type == "filtered":
            self._show_status("warning: filtered model selected with a hardware source; the Trigno stream is unfiltered", "warn")

    def _on_model_failed(self, message: str) -> None:
        self.model_info = None
        self.panel.set_model_status("-", "✗ not loaded", "error", message)
        self.panel.set_offline_ready(False)
        self.panel.set_calibration_available(False)
        self._show_error(message)

    def _on_smoothing(self, enabled: bool, n: int) -> None:
        self.request_smoothing.emit(enabled, n)
        if self.batch is not None:                        # offline results are re-smoothed on display
            self.show_offline_window(self.window_index)

    # ================================================================== offline mode
    def select_file(self) -> None:
        start_dir = str(self.recording.path.parent) if self.recording and self.recording.path else str(C.PROJECT_ROOT)
        path, _ = QFileDialog.getOpenFileName(self, "Select an EMG recording", start_dir, FILE_FILTER)
        if path:
            self.load_file(Path(path))

    def load_file(self, path: Path) -> bool:
        source = FileSignalSource(path)
        try:
            source.connect()
        except SourceError as e:
            self.recording = None
            self.panel.set_file(None)
            self.panel.set_offline_ready(False)
            self.signal_view.clear()
            self._update_signal_info()
            self._update_badge()
            QMessageBox.warning(self, "Cannot load recording", f"{path.name}\n\n{e}")
            self._show_error(f"cannot load {path.name}: {e}")
            return False
        self.recording = source.recording
        self.batch = None
        self.window_index = 0
        self.panel.set_mode("offline")
        self.panel.set_file(path, self.recording.warnings)
        self.panel.set_windows(0)
        self.panel.set_batch_progress(None, None)
        self.panel.set_offline_ready(self.model_info is not None)
        self.signal_view.set_recording(self.recording)
        self.gesture_view.clear()
        self._update_signal_info()
        self._update_badge()
        self._show_status(self.recording.describe() + ("  |  " + "; ".join(self.recording.warnings) if self.recording.warnings else ""),
                          "warn" if self.recording.warnings else "dim")
        # pick the matching model for dataset files
        wanted = "filtered" if path.stem.endswith("_filtered") else "raw"
        if wanted in [self.panel.model_box.itemText(i) for i in range(self.panel.model_box.count())] \
                and self.panel.model_box.currentText() != wanted:
            self.panel.model_box.setCurrentText(wanted)        # triggers _on_model_choice
        return True

    def run_offline_inference(self) -> None:
        if self.recording is None or self.model_info is None:
            return
        self.panel.set_offline_ready(False)
        self.panel.set_batch_progress(0, 1)
        self._show_status("running inference over the recording…")
        self.request_batch.emit(self.recording.data)

    def _on_batch_done(self, result: dict) -> None:
        self.batch = result
        n = len(result["starts"])
        self.panel.set_batch_progress(None, None)
        self.panel.set_offline_ready(True)
        self.panel.set_windows(n)
        self.window_index = 0
        self.show_offline_window(0)
        self._show_status(f"inference done: {n} windows" + ("  (calibrated model)" if result.get("calibrated") else ""))
        self.show_view("gesture")

    def _on_batch_failed(self, message: str) -> None:
        self.panel.set_batch_progress(None, None)
        self.panel.set_offline_ready(self.recording is not None and self.model_info is not None)
        self._show_error(message)

    def show_offline_window(self, index: int) -> None:
        if self.batch is None or self.recording is None:
            return
        n = len(self.batch["starts"])
        index = max(0, min(index, n - 1))
        self.window_index = index
        start = int(self.batch["starts"][index])
        self.panel.set_window_index(index, n)
        self.signal_view.set_window_marker(start)
        smoothing_on, smooth_n = self.panel.smoothing
        raw = make_prediction(self.batch["probs"][index])
        smoothed = None
        if smoothing_on:
            lo = max(0, index - smooth_n + 1)
            smoothed = make_prediction(self.batch["probs"][lo:index + 1].mean(axis=0))
        self.gesture_view.show_prediction({"raw": raw, "smoothed": smoothed, "smooth_n": smooth_n,
                                           "start_sample": start, "end_sample": start + C.WINDOW_SAMPLES,
                                           "fs": self.recording.fs, "calibrated": self.batch.get("calibrated", False)})

    def _on_signal_scrolled(self, first_sample: int) -> None:
        """Scrolling the signal view in offline mode moves to the first window inside the visible range."""
        if self.batch is None or self.play_timer.isActive():
            return
        starts = self.batch["starts"]
        idx = int(np.searchsorted(starts, first_sample))
        if 0 <= idx < len(starts) and idx != self.window_index:
            self.window_index = idx
            self.panel.set_window_index(idx, len(starts))
            self.signal_view.set_window_marker(int(starts[idx]), follow=False)
            self.show_offline_window(idx)

    def _on_play(self, playing: bool) -> None:
        if playing and self.batch is not None:
            if self.window_index >= len(self.batch["starts"]) - 1:
                self.window_index = 0
            self.play_timer.start()
        else:
            self.play_timer.stop()

    def _play_step(self) -> None:
        if self.batch is None:
            self.play_timer.stop()
            return
        if self.window_index >= len(self.batch["starts"]) - 1:
            self.play_timer.stop()
            self.panel.set_playing(False)
            return
        self.show_offline_window(self.window_index + 1)

    # ================================================================== live mode
    def choose_replay_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a recording to replay", str(C.PROJECT_ROOT), FILE_FILTER)
        if path:
            self.replay_path = Path(path)
            self.panel.set_replay_file(self.replay_path)

    def replay_current_file_as_live(self) -> None:
        if self.recording is None or self.recording.path is None:
            return
        self.replay_path = self.recording.path
        self.panel.set_replay_file(self.replay_path)
        self.panel.set_live_source("replay")
        self.panel.set_mode("live")
        self.connect_live()

    def connect_live(self) -> None:
        if self.controller.source is not None and self.controller.state in (SourceState.CONNECTED, SourceState.STREAMING):
            return
        key = self.panel.live_source()
        try:
            if key == "delsys":
                source = DelsysTrignoSource(self.panel.delsys_config())
            else:
                if self.replay_path is None:
                    self.choose_replay_file()
                    if self.replay_path is None:
                        return
                rec = load_recording_file(self.replay_path)
                source = ReplaySignalSource(rec, loop=self.panel.replay_loop.isChecked())
        except (SourceError, RecordingFormatError) as e:
            self._show_error(str(e))
            QMessageBox.warning(self, "Cannot create source", str(e))
            return
        self.controller.set_source(source)
        self.signal_view.set_channel_names(source.channel_names)
        self.signal_view.set_live(self.controller)
        self.request_reset_stream.emit()
        self._show_status(f"connecting to {source.name}…")
        if not self.controller.connect_source():
            self.controller.set_source(None)
            self.panel.set_connection(self.status_label.text(), "error", connected=False, streaming=False)
        self._update_badge()
        self._update_signal_info()
        if self.model_info is not None and self.model_info.signal_type == "filtered" and source.kind is SourceKind.HARDWARE:
            self._show_status("warning: filtered model selected with a hardware source; the Trigno stream is unfiltered", "warn")

    def disconnect_live(self) -> None:
        if self.controller.is_recording:
            self.stop_recording()
        self.controller.set_source(None)
        self.panel.set_connection("disconnected", "dim", connected=False, streaming=False)
        self.gesture_view.clear()
        self.signal_view.clear()
        self.signal_view.set_live(self.controller)
        self._update_badge()
        self._update_signal_info()

    def start_recording(self) -> None:
        if self.unsaved is not None and not self._confirm_discard():
            return
        self.unsaved = None
        self.request_reset_stream.emit()
        self.gesture_view.clear()
        self.controller.inference_enabled = self.panel.inference_check.isChecked()
        if self.controller.start_stream():
            self.controller.start_recording()
            self._show_status(f"recording from {self.controller.source.name}")

    def stop_recording(self) -> None:
        self.controller.stop_stream()
        data = self.controller.stop_recording()
        self.unsaved = data if len(data) else None
        self._show_status(f"recording stopped: {len(data) / self.controller.fs:.1f} s" if len(data) else "recording stopped: no samples received")

    def save_recording(self) -> None:
        if self.unsaved is None:
            return
        src = self.controller.source
        source_name = "delsys_trigno" if (src and src.kind is SourceKind.HARDWARE) else "replay" if src else "live"
        default = default_recording_path(source_name)
        path, _ = QFileDialog.getSaveFileName(self, "Save recording", str(default), "MATLAB recording (*.mat)")
        if not path:
            return
        try:
            cfg = None
            if src is not None and src.kind is SourceKind.HARDWARE:
                cfg = __import__("json").loads(src.config.to_json())
            elif src is not None and src.kind is SourceKind.REPLAY and self.replay_path is not None:
                cfg = {"replayed_file": str(self.replay_path)}
            saved = save_recording(path, self.unsaved, fs=self.controller.fs,
                                   channel_names=src.channel_names if src else None, source=source_name, source_config=cfg)
        except Exception as e:
            self._show_error(f"saving failed: {e}")
            QMessageBox.warning(self, "Saving failed", str(e))
            return
        self.unsaved = None
        self.controller.discard_recording()
        self._show_status(f"saved {saved}")

    def discard_recording(self) -> None:
        if self.unsaved is not None and self._confirm_discard():
            self.unsaved = None
            self.controller.discard_recording()
            self._show_status("recording discarded")

    def _confirm_discard(self) -> bool:
        answer = QMessageBox.question(self, "Discard recording?", "The last recording has not been saved. Discard it?",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    def calibrate(self) -> None:
        if self.model_info is None:
            self._show_error("no model loaded")
            return
        needed = (20 - 1) * C.HOP_SAMPLES + C.WINDOW_SAMPLES                  # 20 windows = 11.5 s
        data = self.controller.ring.latest(self.controller.ring.count)
        starts = window_starts(len(data))
        if len(starts) < 20:
            self._show_error(f"calibration needs at least {needed / self.controller.fs:.1f} s of signal in the buffer "
                             f"({len(data) / self.controller.fs:.1f} s available)")
            return
        windows = np.stack([data[s:s + C.WINDOW_SAMPLES] for s in starts])
        self._show_status(f"calibrating on {len(windows)} windows…")
        self.request_calibration.emit(windows)

    def _on_calibration_done(self, n: int) -> None:
        self.panel.set_calibration_available(True, checked=True)
        self._show_status(f"calibrated on {n} windows; the calibrated model is now used (uncheck to go back to strict)", "ok")

    def _set_inference_enabled(self, enabled: bool) -> None:
        self.controller.inference_enabled = enabled
        if not enabled:
            self.gesture_view.clear()

    # ================================================================== controller events
    def _on_source_state(self, state, message: str) -> None:
        connected = state in (SourceState.CONNECTED, SourceState.STREAMING)
        streaming = state is SourceState.STREAMING
        role = {"error": "error", "streaming": "ok", "connected": "ok"}.get(state.value, "dim")
        text = f"{state.value}" + (f": {message}" if message else "")
        self.panel.set_connection(text, role, connected=connected, streaming=streaming)
        if state is SourceState.ERROR:
            if self.controller.is_recording:
                self.stop_recording()
        elif state is SourceState.CONNECTED and message == "replay finished" and self.controller.is_recording:
            self.stop_recording()
            self._show_status("replay finished; recording stopped")
        if message and state is not SourceState.ERROR:
            self._show_status(message)
        self._update_badge()

    def _on_recording_changed(self, active: bool, n_samples: int) -> None:
        self.panel.set_recording(active, n_samples, self.controller.fs, has_data=self.unsaved is not None or (active and n_samples > 0))
        if self.mode == "live":
            self._update_signal_info()

    def _on_progress(self, _samples: int) -> None:
        if self.mode == "live":
            self._update_signal_info()

    def _on_prediction(self, payload: dict) -> None:
        payload = dict(payload)
        payload["fs"] = self.controller.fs
        payload["smooth_n"] = self.panel.smoothing[1]
        self.gesture_view.show_prediction(payload)

    # ================================================================== shutdown
    def closeEvent(self, event) -> None:
        if self.controller.is_recording:
            self.stop_recording()
        if self.unsaved is not None:
            answer = QMessageBox.question(self, "Unsaved recording", "Save the last recording before closing?",
                                          QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if answer == QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.Save:
                self.save_recording()
        self.play_timer.stop()
        self.signal_view.timer.stop()
        self.controller.shutdown()
        self.worker.cancel()
        self.thread.quit()
        if not self.thread.wait(3000):
            self.thread.terminate()
            self.thread.wait(1000)
        event.accept()

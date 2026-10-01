"""The left control panel: MODE, INPUT (live or offline controls), MODEL, SIGNAL INFORMATION.

The panel only emits signals and exposes setters; all decisions live in MainWindow.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QSpinBox, QVBoxLayout, QWidget)

from app import theme
from app.delsys import TrignoConfig, assumptions_text
from app.sources import SourceError
from network import config as C


def _title(text: str) -> QLabel:
    lab = QLabel(text.upper())
    lab.setProperty("role", "title")
    return lab


def _panel() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setProperty("role", "panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(8)
    return frame, layout


def _value_label(text: str = "-") -> QLabel:
    lab = QLabel(text)
    lab.setProperty("role", "value")
    _wrap(lab)
    return lab


def _wrap(label: QLabel) -> QLabel:
    """Word-wrapped label that never forces the panel wider than its viewport."""
    label.setWordWrap(True)
    label.setMinimumWidth(1)
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    return label


class ControlPanel(QWidget):
    # mode / views
    mode_changed = Signal(str)                 # "live" | "offline"
    # live
    live_source_changed = Signal(str)          # "delsys" | "replay"
    choose_replay_file = Signal()
    connect_clicked = Signal()
    disconnect_clicked = Signal()
    start_recording_clicked = Signal()
    stop_recording_clicked = Signal()
    save_recording_clicked = Signal()
    discard_recording_clicked = Signal()
    calibrate_clicked = Signal()
    inference_while_recording_toggled = Signal(bool)
    # offline
    select_file_clicked = Signal()
    run_clicked = Signal()
    play_toggled = Signal(bool)
    window_index_changed = Signal(int)
    replay_as_live_clicked = Signal()
    # model
    model_changed = Signal(str)
    smoothing_changed = Signal(bool, int)
    calibrated_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(350)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(10)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # ---------------------------------------------------------------- MODE
        frame, lay = _panel()
        lay.addWidget(_title("Mode"))
        row = QHBoxLayout()
        self.live_btn = QPushButton("Live")
        self.offline_btn = QPushButton("Offline")
        for b in (self.live_btn, self.offline_btn):
            b.setCheckable(True)
            row.addWidget(b)
        self.offline_btn.setChecked(True)
        self.live_btn.clicked.connect(lambda: self._set_mode("live"))
        self.offline_btn.clicked.connect(lambda: self._set_mode("offline"))
        lay.addLayout(row)
        layout.addWidget(frame)

        # ---------------------------------------------------------------- INPUT
        frame, lay = _panel()
        lay.addWidget(_title("Input"))
        # two plain pages shown / hidden (a QStackedWidget would keep the tallest page's minimum height)
        self.offline_page = self._build_offline_controls()
        self.live_page = self._build_live_controls()
        lay.addWidget(self.offline_page)
        lay.addWidget(self.live_page)
        self.live_page.hide()
        layout.addWidget(frame)

        # ---------------------------------------------------------------- MODEL
        frame, lay = _panel()
        lay.addWidget(_title("Model"))
        self.model_box = QComboBox()
        self.model_box.currentTextChanged.connect(self._on_model_box)
        self.model_name = _value_label("-")
        self.model_status = QLabel("not loaded")
        self.model_status.setProperty("role", "warn")
        _wrap(self.model_status)
        self.inference_state = _value_label("idle")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setHorizontalSpacing(12)
        form.addRow(QLabel("Signal type"), self.model_box)
        form.addRow(QLabel("Network"), self.model_name)
        form.addRow(QLabel("Status"), self.model_status)
        form.addRow(QLabel("Inference"), self.inference_state)
        lay.addLayout(form)
        smooth_row = QHBoxLayout()
        self.smooth_check = QCheckBox("Smooth over")
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(2, 10)
        self.smooth_spin.setValue(C.SMOOTH_WINDOWS)
        self.smooth_spin.setSuffix(" windows")
        self.smooth_check.toggled.connect(self._emit_smoothing)
        self.smooth_spin.valueChanged.connect(self._emit_smoothing)
        smooth_row.addWidget(self.smooth_check)
        smooth_row.addWidget(self.smooth_spin)
        smooth_row.addStretch(1)
        lay.addLayout(smooth_row)
        self.calibrated_check = QCheckBox("Use calibrated model (AdaBN)")
        self.calibrated_check.setEnabled(False)
        self.calibrated_check.toggled.connect(self.calibrated_toggled)
        lay.addWidget(self.calibrated_check)
        self.model_note = QLabel("")
        self.model_note.setProperty("role", "dim")
        _wrap(self.model_note)
        lay.addWidget(self.model_note)
        layout.addWidget(frame)

        # ---------------------------------------------------------------- SIGNAL INFORMATION
        frame, lay = _panel()
        lay.addWidget(_title("Signal information"))
        self.info_channels = _value_label("-")
        self.info_fs = _value_label("-")
        self.info_samples = _value_label("-")
        self.info_windows = _value_label("-")
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.addRow(QLabel("Channels"), self.info_channels)
        form.addRow(QLabel("Sampling rate"), self.info_fs)
        form.addRow(QLabel("Samples"), self.info_samples)
        form.addRow(QLabel("Windows"), self.info_windows)
        lay.addLayout(form)
        layout.addWidget(frame)
        layout.addStretch(1)

        self.mode = "offline"
        self._set_mode("offline")

    # ------------------------------------------------------------------ builders
    def _build_offline_controls(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.select_file_btn = QPushButton("Select file…")
        self.select_file_btn.clicked.connect(self.select_file_clicked)
        self.file_label = QLabel("no file selected")
        self.file_label.setProperty("role", "dim")
        _wrap(self.file_label)
        self.file_warning = QLabel("")
        self.file_warning.setProperty("role", "warn")
        _wrap(self.file_warning)
        self.file_warning.hide()
        self.run_btn = QPushButton("Run inference")
        self.run_btn.setProperty("role", "primary")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_clicked)
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        self.batch_progress.hide()
        self.window_slider = QSlider(Qt.Horizontal)
        self.window_slider.setEnabled(False)
        self.window_slider.valueChanged.connect(self.window_index_changed)
        self.window_label = QLabel("window - / -")
        self.window_label.setProperty("role", "dim")
        play_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.setEnabled(False)
        self.play_btn.toggled.connect(self._on_play)
        self.replay_live_btn = QPushButton("Replay as live…")
        self.replay_live_btn.setEnabled(False)
        self.replay_live_btn.setToolTip("Stream this file through the live pipeline (SIMULATED / REPLAY)")
        self.replay_live_btn.clicked.connect(self.replay_as_live_clicked)
        play_row.addWidget(self.play_btn)
        play_row.addWidget(self.replay_live_btn)
        for widget in (self.select_file_btn, self.file_label, self.file_warning, self.run_btn, self.batch_progress,
                       self.window_label, self.window_slider):
            lay.addWidget(widget)
        lay.addLayout(play_row)
        return w

    def _build_live_controls(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.live_source_box = QComboBox()
        self.live_source_box.addItem("Delsys Trigno Avanti (hardware)", "delsys")
        self.live_source_box.addItem("Replay a recording (simulated)", "replay")
        self.live_source_box.currentIndexChanged.connect(
            lambda i: self.live_source_changed.emit(self.live_source_box.itemData(i)))
        lay.addWidget(self.live_source_box)

        # Delsys settings and replay settings: shown / hidden depending on the chosen source
        delsys = QWidget()
        form = QFormLayout(delsys)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(10)
        self.host_edit = QLineEdit("localhost")
        self.cmd_port = QSpinBox()
        self.cmd_port.setRange(1, 65535)
        self.cmd_port.setValue(50040)
        self.emg_port = QSpinBox()
        self.emg_port.setRange(1, 65535)
        self.emg_port.setValue(50043)
        self.sensors_edit = QLineEdit("1, 2, 3, 4")
        self.sensors_edit.setToolTip("Trigno sensor numbers for model channels CH1..CH4")
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.001, 100000)
        self.scale_spin.setDecimals(3)
        self.scale_spin.setValue(1000.0)
        self.scale_spin.setToolTip("multiply the stream to get mV (Trigno streams volts: x1000)")
        form.addRow("Host", self.host_edit)
        form.addRow("Command port", self.cmd_port)
        form.addRow("EMG port", self.emg_port)
        form.addRow("Sensors", self.sensors_edit)
        form.addRow("Scale to mV", self.scale_spin)
        note = QLabel(assumptions_text())
        note.setProperty("role", "dim")
        _wrap(note)
        form.addRow(note)
        self.delsys_page = delsys
        # replay settings
        replay = QWidget()
        rlay = QVBoxLayout(replay)
        rlay.setContentsMargins(0, 0, 0, 0)
        self.replay_file_btn = QPushButton("Choose recording…")
        self.replay_file_btn.clicked.connect(self.choose_replay_file)
        self.replay_file_label = QLabel("no recording chosen")
        self.replay_file_label.setProperty("role", "dim")
        _wrap(self.replay_file_label)
        self.replay_loop = QCheckBox("Loop")
        rlay.addWidget(self.replay_file_btn)
        rlay.addWidget(self.replay_file_label)
        rlay.addWidget(self.replay_loop)
        self.replay_page = replay
        self.replay_page.hide()
        lay.addWidget(self.delsys_page)
        lay.addWidget(self.replay_page)
        self.live_source_box.currentIndexChanged.connect(self._on_live_source_index)

        self.connection_label = QLabel("disconnected")
        self.connection_label.setProperty("role", "dim")
        _wrap(self.connection_label)
        lay.addWidget(self.connection_label)
        row = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_clicked)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.clicked.connect(self.disconnect_clicked)
        row.addWidget(self.connect_btn)
        row.addWidget(self.disconnect_btn)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.start_btn = QPushButton("Start recording")
        self.start_btn.setProperty("role", "primary")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_recording_clicked)
        self.stop_btn = QPushButton("Stop recording")
        self.stop_btn.setProperty("role", "danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_recording_clicked)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        lay.addLayout(row)
        self.inference_check = QCheckBox("Run inference while recording")
        self.inference_check.setChecked(True)
        self.inference_check.toggled.connect(self.inference_while_recording_toggled)
        lay.addWidget(self.inference_check)
        self.recording_label = QLabel("")
        self.recording_label.setProperty("role", "dim")
        lay.addWidget(self.recording_label)
        row = QHBoxLayout()
        self.save_btn = QPushButton("Save recording…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_recording_clicked)
        self.discard_btn = QPushButton("Discard")
        self.discard_btn.setEnabled(False)
        self.discard_btn.clicked.connect(self.discard_recording_clicked)
        row.addWidget(self.save_btn)
        row.addWidget(self.discard_btn)
        lay.addLayout(row)
        self.calibrate_btn = QPushButton("Calibrate on recent signal (AdaBN)")
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.setToolTip("Re-estimate the network's BatchNorm statistics on the last seconds of this participant's signal")
        self.calibrate_btn.clicked.connect(self.calibrate_clicked)
        lay.addWidget(self.calibrate_btn)
        return w

    # ------------------------------------------------------------------ mode
    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.live_btn.setChecked(mode == "live")
        self.offline_btn.setChecked(mode == "offline")
        self.offline_page.setVisible(mode == "offline")
        self.live_page.setVisible(mode == "live")
        self.mode_changed.emit(mode)

    def set_mode(self, mode: str) -> None:
        if mode != self.mode:
            self._set_mode(mode)

    def _on_live_source_index(self, index: int) -> None:
        key = self.live_source_box.itemData(index)
        self.delsys_page.setVisible(key == "delsys")
        self.replay_page.setVisible(key == "replay")

    def live_source(self) -> str:
        return self.live_source_box.currentData()

    def set_live_source(self, key: str) -> None:
        idx = self.live_source_box.findData(key)
        if idx >= 0:
            self.live_source_box.setCurrentIndex(idx)

    def delsys_config(self) -> TrignoConfig:
        try:
            sensors = tuple(int(s) for s in self.sensors_edit.text().replace(";", ",").split(",") if s.strip())
        except ValueError as e:
            raise SourceError(f"sensor list must be four integers, e.g. '1, 2, 3, 4' ({e})") from e
        cfg = TrignoConfig(host=self.host_edit.text().strip() or "localhost", command_port=self.cmd_port.value(),
                           emg_port=self.emg_port.value(), sensors=sensors, scale_to_mv=self.scale_spin.value())
        cfg.validate()
        return cfg

    # ------------------------------------------------------------------ setters used by MainWindow
    def set_models(self, names: list[str], current: str | None) -> None:
        self.model_box.blockSignals(True)
        self.model_box.clear()
        self.model_box.addItems(names)
        if current in names:
            self.model_box.setCurrentText(current)
        self.model_box.blockSignals(False)
        self.model_box.setEnabled(bool(names))

    def _on_model_box(self, text: str) -> None:
        if text:
            self.model_changed.emit(text)

    def set_model_status(self, name: str, status: str, role: str, note: str = "") -> None:
        self.model_name.setText(name)
        self.model_status.setText(status)
        self.model_status.setProperty("role", role)
        theme.repolish(self.model_status)
        self.model_note.setText(note)

    def set_inference_state(self, text: str) -> None:
        self.inference_state.setText(text)

    def set_calibration_available(self, available: bool, checked: bool | None = None) -> None:
        self.calibrated_check.setEnabled(available)
        if checked is not None:
            self.calibrated_check.blockSignals(True)
            self.calibrated_check.setChecked(checked)
            self.calibrated_check.blockSignals(False)
        if not available:
            self.calibrated_check.blockSignals(True)
            self.calibrated_check.setChecked(False)
            self.calibrated_check.blockSignals(False)

    def set_signal_info(self, channels: int | None, fs: float | None, samples: int | None, windows: int | None) -> None:
        self.info_channels.setText("-" if channels is None else str(channels))
        self.info_fs.setText("-" if fs is None else f"{fs:g} Hz")
        self.info_samples.setText("-" if samples is None else f"{samples:,}")
        self.info_windows.setText("-" if windows is None else f"{windows:,}")

    def set_file(self, path: Path | None, warnings: list[str] | None = None) -> None:
        self.file_label.setText(str(path) if path else "no file selected")
        self.file_label.setProperty("role", "value" if path else "dim")
        theme.repolish(self.file_label)
        if warnings:
            self.file_warning.setText("\n".join(warnings))
            self.file_warning.show()
        else:
            self.file_warning.hide()
        self.replay_live_btn.setEnabled(path is not None)

    def set_offline_ready(self, can_run: bool) -> None:
        self.run_btn.setEnabled(can_run)

    def set_batch_progress(self, done: int | None, total: int | None) -> None:
        if done is None:
            self.batch_progress.hide()
            return
        self.batch_progress.show()
        self.batch_progress.setValue(int(100 * done / max(1, total)))

    def set_windows(self, n_windows: int) -> None:
        self.window_slider.blockSignals(True)
        self.window_slider.setRange(0, max(0, n_windows - 1))
        self.window_slider.setValue(0)
        self.window_slider.blockSignals(False)
        self.window_slider.setEnabled(n_windows > 0)
        self.play_btn.setEnabled(n_windows > 0)
        self.set_window_index(0, n_windows)

    def set_window_index(self, index: int, n_windows: int) -> None:
        self.window_label.setText(f"window {index + 1} / {n_windows}" if n_windows else "window - / -")
        if self.window_slider.value() != index:
            self.window_slider.blockSignals(True)
            self.window_slider.setValue(index)
            self.window_slider.blockSignals(False)

    def _on_play(self, checked: bool) -> None:
        self.play_btn.setText("Stop" if checked else "Play")
        self.play_toggled.emit(checked)

    def set_playing(self, playing: bool) -> None:
        if self.play_btn.isChecked() != playing:
            self.play_btn.setChecked(playing)

    def set_replay_file(self, path: Path | None) -> None:
        self.replay_file_label.setText(str(path) if path else "no recording chosen")

    def set_connection(self, text: str, role: str, connected: bool, streaming: bool) -> None:
        self.connection_label.setText(text)
        self.connection_label.setProperty("role", role)
        theme.repolish(self.connection_label)
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected and not streaming)
        self.start_btn.setEnabled(connected and not streaming)
        self.stop_btn.setEnabled(streaming)
        self.live_source_box.setEnabled(not connected)
        self.calibrate_btn.setEnabled(connected)

    def set_recording(self, active: bool, n_samples: int, fs: float, has_data: bool) -> None:
        if active:
            self.recording_label.setText(f"● recording  {n_samples / fs:.1f} s  ({n_samples:,} samples)")
            self.recording_label.setProperty("role", "error")
        elif has_data:
            self.recording_label.setText(f"recorded {n_samples / fs:.1f} s ({n_samples:,} samples), not saved")
            self.recording_label.setProperty("role", "warn")
        else:
            self.recording_label.setText("")
            self.recording_label.setProperty("role", "dim")
        theme.repolish(self.recording_label)
        self.save_btn.setEnabled(has_data and not active)
        self.discard_btn.setEnabled(has_data and not active)

    def _emit_smoothing(self, *_):
        self.smoothing_changed.emit(self.smooth_check.isChecked(), self.smooth_spin.value())

    @property
    def smoothing(self) -> tuple[bool, int]:
        return self.smooth_check.isChecked(), self.smooth_spin.value()

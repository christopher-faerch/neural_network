"""EMG signal view: four stacked channel plots (pyqtgraph).

Live mode: a rolling window read from the controller's ring buffer on a 30 Hz timer.
Offline mode: scroll through a loaded recording; the current inference window is marked.
Controls: pause / resume, autoscale, visible seconds.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QScrollBar, QVBoxLayout,
                               QWidget)

from app import theme
from network import config as C

VISIBLE_CHOICES_S = (2, 5, 10, 20)


class SignalView(QWidget):
    position_changed = Signal(int)           # offline: first visible sample after scrolling

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fs = float(C.FS)
        self.channel_names = [f"CH{i + 1}" for i in range(C.N_CHANNELS)]
        self.controller = None
        self.recording = None
        self.visible_s = 5.0
        self.paused = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._on_pause)
        self.autoscale = QCheckBox("Autoscale")
        self.autoscale.setChecked(True)
        self.autoscale.toggled.connect(self._apply_autoscale)
        self.window_box = QComboBox()
        for s in VISIBLE_CHOICES_S:
            self.window_box.addItem(f"{s} s", s)
        self.window_box.setCurrentIndex(VISIBLE_CHOICES_S.index(5))
        self.window_box.currentIndexChanged.connect(self._on_visible_changed)
        self.info = QLabel("")
        self.info.setProperty("role", "dim")
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.autoscale)
        controls.addWidget(QLabel("Visible:"))
        controls.addWidget(self.window_box)
        controls.addStretch(1)
        controls.addWidget(self.info)
        layout.addLayout(controls)

        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics, 1)
        self.plots, self.curves, self.markers = [], [], []
        for ch in range(C.N_CHANNELS):
            plot = self.graphics.addPlot(row=ch, col=0)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setLabel("left", self.channel_names[ch], units="mV")
            plot.getAxis("left").setWidth(64)
            plot.setDownsampling(auto=True, mode="peak")
            plot.setClipToView(True)
            if ch < C.N_CHANNELS - 1:
                plot.hideAxis("bottom")
            else:
                plot.setLabel("bottom", "time", units="s")
            if ch > 0:
                plot.setXLink(self.plots[0])
            curve = plot.plot(pen=pg.mkPen(theme.CHANNEL_COLORS[ch], width=1))
            marker = pg.LinearRegionItem(values=(0, 1), movable=False,
                                         brush=pg.mkBrush(46, 196, 182, 45), pen=pg.mkPen(theme.ACCENT, width=1))
            marker.setZValue(-10)
            marker.hide()
            plot.addItem(marker)
            self.plots.append(plot)
            self.curves.append(curve)
            self.markers.append(marker)

        self.scroll = QScrollBar(Qt.Horizontal)
        self.scroll.valueChanged.connect(self._on_scroll)
        self.scroll.hide()
        layout.addWidget(self.scroll)

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._refresh_live)

    # ------------------------------------------------------------------ configuration
    def set_channel_names(self, names) -> None:
        self.channel_names = list(names)
        for plot, name in zip(self.plots, self.channel_names):
            plot.setLabel("left", name, units="mV")

    def set_live(self, controller) -> None:
        """Rolling display of the controller's ring buffer."""
        self.recording = None
        self.controller = controller
        self.fs = controller.fs
        self.scroll.hide()
        for m in self.markers:
            m.hide()
        self.info.setText("live")
        self.timer.start()

    def set_recording(self, recording) -> None:
        """Static display of a loaded recording with a scroll bar."""
        self.timer.stop()
        self.controller = None
        self.recording = recording
        self.fs = recording.fs
        self.set_channel_names(recording.channel_names)
        self.scroll.show()
        self._update_scroll_range()
        self.scroll.setValue(0)
        self._draw_offline(0)

    def clear(self) -> None:
        self.timer.stop()
        self.controller = self.recording = None
        self.scroll.hide()
        for curve, marker in zip(self.curves, self.markers):
            curve.setData([], [])
            marker.hide()
        self.info.setText("")

    def set_window_marker(self, start_sample: int | None, follow: bool = True) -> None:
        """Offline: mark [start, start + WINDOW_SAMPLES] and scroll so that it is visible."""
        if start_sample is None:
            for m in self.markers:
                m.hide()
            return
        t0, t1 = start_sample / self.fs, (start_sample + C.WINDOW_SAMPLES) / self.fs
        for m in self.markers:
            m.setRegion((t0, t1))
            m.show()
        if follow and self.recording is not None:
            visible = int(self.visible_s * self.fs)
            first = self.scroll.value()
            if start_sample < first or start_sample + C.WINDOW_SAMPLES > first + visible:
                self.scroll.setValue(max(0, min(self.scroll.maximum(), start_sample - visible // 4)))

    # ------------------------------------------------------------------ drawing
    def _refresh_live(self) -> None:
        if self.paused or self.controller is None:
            return
        n = int(self.visible_s * self.fs)
        data = self.controller.ring.latest(n)
        if len(data) == 0:
            return
        total = self.controller.total_samples
        t = (total - len(data) + np.arange(len(data))) / self.fs
        self._set_curves(t, data)
        self.plots[0].setXRange((total - n) / self.fs, total / self.fs, padding=0)
        self.info.setText(f"live  |  {total / self.fs:8.1f} s  |  {total:,} samples")

    def _draw_offline(self, first_sample: int) -> None:
        if self.recording is None:
            return
        n = int(self.visible_s * self.fs)
        first = max(0, min(first_sample, max(0, self.recording.n_samples - n)))
        data = self.recording.data[first:first + n]
        t = (first + np.arange(len(data))) / self.fs
        self._set_curves(t, data)
        self.plots[0].setXRange(first / self.fs, (first + n) / self.fs, padding=0)
        self.info.setText(f"{first / self.fs:.1f} - {(first + len(data)) / self.fs:.1f} s of {self.recording.duration_s:.1f} s")

    def _set_curves(self, t, data) -> None:
        for ch, curve in enumerate(self.curves):
            curve.setData(t, data[:, ch])
        if self.autoscale.isChecked():
            for ch, plot in enumerate(self.plots):
                y = data[:, ch]
                if len(y):
                    span = float(np.abs(y).max()) or 1e-3
                    plot.setYRange(-1.1 * span, 1.1 * span, padding=0)

    # ------------------------------------------------------------------ controls
    def _on_pause(self, checked: bool) -> None:
        self.paused = checked
        self.pause_btn.setText("Resume" if checked else "Pause")

    def _apply_autoscale(self, checked: bool) -> None:
        if not checked:
            for plot in self.plots:
                plot.enableAutoRange(axis="y", enable=False)

    def _on_visible_changed(self, index: int) -> None:
        self.visible_s = float(self.window_box.itemData(index))
        if self.recording is not None:
            self._update_scroll_range()
            self._draw_offline(self.scroll.value())

    def _update_scroll_range(self) -> None:
        n = int(self.visible_s * self.fs)
        self.scroll.setRange(0, max(0, self.recording.n_samples - n))
        self.scroll.setPageStep(n)
        self.scroll.setSingleStep(max(1, n // 10))

    def _on_scroll(self, value: int) -> None:
        if self.recording is not None:
            self._draw_offline(value)
            self.position_changed.emit(value)

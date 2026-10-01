"""GUI tests (offscreen): the app starts without hardware or model, offline and replay paths work,
view switching does not interrupt acquisition, and both paths feed inference identical arrays."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QFileDialog, QMessageBox  # noqa: E402

from app import main as app_main  # noqa: E402
from app.main_window import MainWindow  # noqa: E402
from app.sources import ReplaySignalSource, SourceState  # noqa: E402
from network import config as C  # noqa: E402
from network.streaming import window_starts  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))


@pytest.fixture
def window(qtbot, tmp_path):
    """A window WITHOUT a model (empty results root) - prediction disabled, everything else works."""
    w = MainWindow(results_root=tmp_path / "no_results", device="cpu")
    qtbot.addWidget(w)
    yield w
    w.close()


@pytest.fixture
def model_window(qtbot):
    if not any((C.RESULTS_DIR / "raw").glob("best_model_seed*.pt")):
        pytest.skip("no trained checkpoints in results/raw")
    w = MainWindow(device="cpu")
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.worker.model_loaded, timeout=120_000):
        pass
    qtbot.waitUntil(lambda: w.model_info is not None, timeout=10_000)
    yield w
    w.close()


def test_app_starts_without_model_or_hardware(window):
    assert "not loaded" in window.panel.model_status.text()
    assert not window.panel.run_btn.isEnabled()
    window.show_view("gesture")
    window.show_view("signal")
    window.panel.set_mode("live")
    assert "not connected" in window.badge.text()
    # a Delsys connection attempt with no TCU running fails cleanly
    window.panel.set_live_source("delsys")
    window.panel.host_edit.setText("127.0.0.1")
    window.panel.cmd_port.setValue(1)                 # nothing listens here
    window.connect_live()
    assert window.controller.source is None and window.status_label.property("role") == "error"
    assert "not reachable" in window.status_label.text()


def test_offline_load_errors_are_clean(window, tmp_path):
    bogus = tmp_path / "bogus.mat"
    bogus.write_text("not a mat file")
    assert window.load_file(bogus) is False
    assert window.recording is None and window.status_label.property("role") == "error"
    import scipy.io as sio
    three = tmp_path / "three.mat"
    sio.savemat(three, {"data": np.zeros((3000, 3)), "fs": 2000})
    assert window.load_file(three) is False and "channels" in window.status_label.text()


def test_offline_load_without_model_shows_info(window, synthetic_recording_path):
    assert window.load_file(synthetic_recording_path) is True
    assert window.panel.info_channels.text() == "4" and window.panel.info_fs.text() == "2000 Hz"
    assert window.panel.info_windows.text() == str(len(window_starts(12 * C.FS)))
    assert window.badge.text() == "FILE" and not window.panel.run_btn.isEnabled()   # no model -> no inference


def test_offline_inference_and_playback(model_window, qtbot, synthetic_recording_path):
    w = model_window
    assert w.load_file(synthetic_recording_path) and w.panel.run_btn.isEnabled()
    with qtbot.waitSignal(w.worker.batch_done, timeout=60_000):
        w.run_offline_inference()
    qtbot.waitUntil(lambda: w.batch is not None, timeout=10_000)       # the window's queued slot may run after waitSignal
    n = len(window_starts(12 * C.FS))
    assert w.batch is not None and w.batch["probs"].shape == (n, C.N_CLASSES)
    assert np.allclose(w.batch["probs"].sum(axis=1), 1.0, atol=1e-5)
    assert w.views.currentWidget() is w.gesture_view and w.gesture_view.current in C.GESTURES
    assert w.gesture_view.cards[w.gesture_view.current].property("state") == "predicted"
    w.show_offline_window(5)
    assert w.panel.window_label.text() == f"window 6 / {n}"
    w.panel.smooth_check.setChecked(True)
    w.show_offline_window(7)
    assert "smoothed" in w.gesture_view.detail.text()
    w.panel.play_btn.setChecked(True)
    qtbot.waitUntil(lambda: w.window_index > 7, timeout=5000)
    w.panel.play_btn.setChecked(False)


def test_replay_live_path_view_switching_and_saving(model_window, qtbot, synthetic_recording, synthetic_recording_path, tmp_path, monkeypatch):
    w = model_window
    clf = w.worker.classifier
    captured = []
    original = clf.preprocess

    def capturing_preprocess(x):
        out = original(x)
        captured.append(out.copy())
        return out
    monkeypatch.setattr(clf, "preprocess", capturing_preprocess)

    # ---- replay as a live source (fast), recording + inference running
    src = ReplaySignalSource(synthetic_recording, chunk_s=0.01, speed=10.0)
    w.panel.set_mode("live")
    w.replay_path = synthetic_recording_path
    w.controller.set_source(src)
    w.signal_view.set_live(w.controller)
    assert w.controller.connect_source()
    w._update_badge()
    assert w.badge.text() == "SIMULATED / REPLAY"
    predictions = []
    w.worker.prediction_ready.connect(lambda p: predictions.append(p))
    w.start_recording()
    assert w.controller.is_recording
    # switching views while streaming must not interrupt acquisition
    counts = []
    for i in range(10):
        w.show_view("gesture" if i % 2 else "signal")
        qtbot.wait(60)
        counts.append(w.controller.total_samples)
    assert counts[-1] > counts[0] > 0 and all(b >= a for a, b in zip(counts, counts[1:]))
    qtbot.waitUntil(lambda: src.state is not SourceState.STREAMING, timeout=20_000)
    qtbot.waitUntil(lambda: not w.controller.is_recording, timeout=5000)      # "replay finished" stops the recording
    qtbot.wait(300)                                                            # let queued predictions arrive
    assert w.unsaved is not None and len(w.unsaved) == synthetic_recording.n_samples
    assert predictions and all(p["raw"]["class_name"] in C.GESTURES for p in predictions)

    # ---- same model input as the offline path for the same windows
    live_inputs = np.concatenate(captured)
    starts = [p["start_sample"] for p in predictions]
    offline = original(np.stack([synthetic_recording.data[s:s + C.WINDOW_SAMPLES] for s in starts]))
    assert live_inputs.shape == offline.shape == (len(starts), 4, 2000) and live_inputs.dtype == np.float32
    np.testing.assert_array_equal(live_inputs, offline)

    # ---- save the recording and reopen it in offline mode
    out = tmp_path / "saved.mat"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    w.save_recording()
    assert out.exists() and w.unsaved is None
    assert w.load_file(out) and w.recording.n_samples == synthetic_recording.n_samples
    assert w.recording.meta["source"] == "replay"


def test_cli_parser():
    args = app_main.parse_args(["--cpu", "--model", "raw"])
    assert args.cpu and args.model == "raw" and args.file is None

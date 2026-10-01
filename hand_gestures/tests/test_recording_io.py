"""Recording files: dataset-compatible writing, validated reading, clean errors."""
import numpy as np
import pytest
import scipy.io as sio

from network import config as C
from network.recording import RecordingFormatError, load_recording_file, save_recording, validate_recording


def test_round_trip_keeps_data_and_metadata(synthetic_recording_path, synthetic_data):
    mat = sio.loadmat(synthetic_recording_path)
    for key in ("data", "fs", "iD", "isi", "isi_units", "labels", "start_sample", "units",
                "format_version", "timestamp", "source", "source_config", "n_samples", "duration_s"):
        assert key in mat, key
    rec = load_recording_file(synthetic_recording_path)
    assert rec.data.dtype == np.float32 and rec.data.shape == synthetic_data.shape
    np.testing.assert_allclose(rec.data, synthetic_data, rtol=0, atol=1e-6)
    assert rec.fs == C.FS and rec.channel_names == ["CH1", "CH2", "CH3", "CH4"] and rec.units == ["mV"] * 4
    assert rec.meta["source"] == "test" and rec.meta["source_config"] == {"note": "pytest fixture"}
    assert "T" in rec.meta["timestamp"] and rec.warnings == []
    assert rec.duration_s == pytest.approx(12.0)


@pytest.mark.parametrize("bad, message", [
    (np.zeros((3000, 3)), "channels"),
    (np.zeros((1999, 4)), "too short"),
    (np.full((3000, 4), np.nan), "NaN"),
])
def test_invalid_contents_are_rejected(tmp_path, bad, message):
    path = tmp_path / "bad.mat"
    sio.savemat(path, {"data": bad, "fs": C.FS})
    with pytest.raises(RecordingFormatError, match=message):
        load_recording_file(path)


def test_transposed_data_is_fixed_with_a_warning(tmp_path):
    path = tmp_path / "transposed.mat"
    sio.savemat(path, {"data": np.random.default_rng(0).standard_normal((4, 3000)), "fs": C.FS})
    rec = load_recording_file(path)
    assert rec.data.shape == (3000, 4) and any("transposed" in w for w in rec.warnings)


def test_missing_fs_assumed_with_warning_and_npz_and_csv(tmp_path):
    data = np.random.default_rng(0).standard_normal((2500, 4))
    sio.savemat(tmp_path / "nofs.mat", {"data": data})
    rec = load_recording_file(tmp_path / "nofs.mat")
    assert rec.fs == C.FS and any("assuming" in w for w in rec.warnings)
    np.savez(tmp_path / "rec.npz", data=data, fs=2000)
    assert load_recording_file(tmp_path / "rec.npz").data.shape == (2500, 4)
    np.savetxt(tmp_path / "rec.csv", data, delimiter=",")
    assert load_recording_file(tmp_path / "rec.csv").data.shape == (2500, 4)


def test_unreadable_files(tmp_path):
    (tmp_path / "text.mat").write_text("this is not a mat file")
    with pytest.raises(RecordingFormatError):
        load_recording_file(tmp_path / "text.mat")
    (tmp_path / "x.wav").write_bytes(b"\0" * 10)
    with pytest.raises(RecordingFormatError, match="unsupported"):
        load_recording_file(tmp_path / "x.wav")
    with pytest.raises(RecordingFormatError, match="not found"):
        load_recording_file(tmp_path / "missing.mat")
    sio.savemat(tmp_path / "nodata.mat", {"signal": np.zeros((3000, 4))})
    with pytest.raises(RecordingFormatError, match="'data'"):
        load_recording_file(tmp_path / "nodata.mat")


def test_save_rejects_empty_or_wrong_shape(tmp_path):
    with pytest.raises(RecordingFormatError):
        save_recording(tmp_path / "a.mat", np.zeros((0, 4)))
    with pytest.raises(RecordingFormatError):
        save_recording(tmp_path / "b.mat", np.zeros((100, 3)))


def test_validate_recording_returns_warnings(synthetic_recording):
    assert validate_recording(synthetic_recording) == []
    synthetic_recording.fs = 1000.0
    assert any("1000" in w for w in validate_recording(synthetic_recording))


@pytest.mark.dataset
def test_dataset_file_loads_and_data_loader_still_works(dataset_file):
    from network.data_loader import load_recording
    rec = load_recording_file(dataset_file)
    assert rec.data.shape == (C.RECORDING_SAMPLES, C.N_CHANNELS) and rec.fs == C.FS
    assert rec.channel_names[0].lower().startswith("emg") and rec.meta["iD"] == 1
    data = load_recording(1, "raw")
    np.testing.assert_array_equal(data, rec.data)

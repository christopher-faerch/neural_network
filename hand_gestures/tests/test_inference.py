"""GestureClassifier: loading, preprocessing shape, valid probabilities, calibration, smoothing, errors."""
import numpy as np
import pytest
import torch

from network import config as C
from network.inference import (CheckpointIncompatibleError, GestureClassifier, InferenceError, ModelNotFoundError,
                               PredictionSmoother, WindowShapeError, causal_smooth, make_prediction)
from network.streaming import StreamWindower, window_starts


@pytest.mark.model
def test_model_info(classifier):
    info = classifier.info
    assert classifier.is_loaded and info.signal_type == "raw" and info.ensemble_size == 5
    assert info.n_channels == 4 and info.window_samples == 2000 and info.class_names == tuple(C.GESTURES)
    assert len(info.channel_std_mv) == 4 and all(v > 0 for v in info.channel_std_mv)
    assert "raw" in GestureClassifier.available_models()


@pytest.mark.model
def test_preprocess_shapes_and_normalisation(classifier, synthetic_data):
    window = synthetic_data[:C.WINDOW_SAMPLES] + 5.0                    # add a DC offset: must be removed
    x = classifier.preprocess(window)
    assert x.shape == (1, 4, 2000) and x.dtype == np.float32
    np.testing.assert_allclose(x.mean(axis=2), 0.0, atol=1e-3)
    # [4, 2000] orientation accepted too (float32 summation order differs between layouts -> loose tolerance)
    np.testing.assert_allclose(classifier.preprocess(window.T), x, atol=1e-3)
    batch = np.stack([synthetic_data[s:s + C.WINDOW_SAMPLES] for s in (0, 1000, 2000)])
    assert classifier.preprocess(batch).shape == (3, 4, 2000)
    # the training normalisation: (x - window mean) / channel_std
    expected = (window.astype(np.float64) - window.astype(np.float64).mean(axis=0)) / np.array(classifier.info.channel_std_mv)
    np.testing.assert_allclose(x[0], expected.T, atol=1e-3)


@pytest.mark.model
def test_predict_returns_valid_probabilities(classifier, synthetic_data):
    result = classifier.predict(synthetic_data[:C.WINDOW_SAMPLES])
    assert set(result) == {"class_index", "class_name", "confidence", "probabilities"}
    probs = result["probabilities"]
    assert probs.shape == (C.N_CLASSES,) and np.all(probs >= 0) and np.all(probs <= 1)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    assert result["class_name"] == C.GESTURES[result["class_index"]] == C.GESTURES[int(probs.argmax())]
    assert result["confidence"] == pytest.approx(float(probs.max()))
    batch = classifier.predict_proba(np.stack([synthetic_data[s:s + C.WINDOW_SAMPLES] for s in (0, 1000, 2000)]))
    assert batch.shape == (3, C.N_CLASSES) and np.allclose(batch.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.model
def test_predict_is_deterministic_and_errors_are_clean(classifier, synthetic_data):
    w = synthetic_data[:C.WINDOW_SAMPLES]
    np.testing.assert_allclose(classifier.predict_proba(w), classifier.predict_proba(w))
    with pytest.raises(WindowShapeError):
        classifier.predict(synthetic_data[:1500])
    with pytest.raises(WindowShapeError):
        classifier.predict(np.zeros((2000, 3)))
    with pytest.raises(InferenceError):
        classifier.predict(np.full((2000, 4), np.nan))


@pytest.mark.model
def test_calibration_keeps_strict_models(classifier, synthetic_data):
    windows = np.stack([synthetic_data[s:s + C.WINDOW_SAMPLES] for s in window_starts(len(synthetic_data))])
    strict = classifier.predict_proba(windows[:3])
    with pytest.raises(InferenceError):
        classifier.calibrate(windows[:5], min_windows=20)
    assert classifier.calibrate(windows, min_windows=20) == len(windows) and classifier.has_calibration
    calibrated = classifier.predict_proba(windows[:3], calibrated=True)
    assert calibrated.shape == (3, C.N_CLASSES) and np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(classifier.predict_proba(windows[:3]), strict)       # strict path unchanged
    classifier.clear_calibration()
    assert not classifier.has_calibration


@pytest.mark.model
def test_offline_and_streamed_windows_give_identical_model_input(classifier, synthetic_data):
    """Requirement: offline (whole file) and live (chunks) paths feed inference with the same arrays."""
    offline = np.stack([synthetic_data[s:s + C.WINDOW_SAMPLES] for s in window_starts(len(synthetic_data))])
    w = StreamWindower()
    streamed = []
    for start in range(0, len(synthetic_data), 97):                        # odd chunk size on purpose
        streamed.extend(win for _, win in w.push(synthetic_data[start:start + 97]))
    streamed = np.stack(streamed)
    assert streamed.shape == offline.shape
    np.testing.assert_array_equal(classifier.preprocess(streamed), classifier.preprocess(offline))


def test_make_prediction_and_causal_smooth():
    p = make_prediction([0.1, 0.7, 0.2] + [0.0] * 7)
    assert p["class_index"] == 1 and p["class_name"] == "extension" and p["confidence"] == pytest.approx(0.7)
    probs = np.eye(C.N_CLASSES)[[0, 0, 1, 1, 1]]
    smoothed = causal_smooth(probs, 3)
    assert smoothed.argmax(axis=1).tolist() == [0, 0, 0, 1, 1]              # third window: 2 votes neutral vs 1 extension


def test_prediction_smoother_mean_and_vote():
    one_hot = np.eye(C.N_CLASSES)
    mean = PredictionSmoother(n=3, mode="mean")
    assert mean.update(one_hot[2])["class_index"] == 2
    assert mean.update(one_hot[5])["class_index"] == 2                    # tie -> argmax picks the first class (2)
    assert mean.update(one_hot[5])["class_index"] == 5
    vote = PredictionSmoother(n=3, mode="vote")
    vote.update(one_hot[1]); vote.update(one_hot[1])
    result = vote.update(one_hot[4])
    assert result["class_index"] == 1 and result["confidence"] == pytest.approx(2 / 3)
    vote.reset()
    assert vote.update(one_hot[4])["class_index"] == 4
    with pytest.raises(ValueError):
        PredictionSmoother(mode="median")


def test_missing_and_incompatible_checkpoints(tmp_path):
    from network.model import EMGCNN
    clf = GestureClassifier(device="cpu")
    with pytest.raises(ModelNotFoundError):
        clf.load_model("raw", results_root=tmp_path)
    with pytest.raises(ModelNotFoundError):
        clf.predict(np.zeros((2000, 4)))
    bad_dir = tmp_path / "raw"
    bad_dir.mkdir()
    ckpt = {"model_state": EMGCNN().state_dict(), "epoch": 1, "val_loss": 1.0, "val_acc": 0.1, "seed": 42,
            "augment": False, "channel_std": np.ones(4, dtype=np.float32), "split": {}, "signal_type": "raw",
            "normalization": "train_std", "config": {"N_CHANNELS": 4, "WINDOW_SAMPLES": 1000, "GESTURES": list(C.GESTURES)}}
    torch.save(ckpt, bad_dir / "best_model_seed42.pt")
    with pytest.raises(CheckpointIncompatibleError, match="window"):
        clf.load_model("raw", results_root=tmp_path)
    ckpt["config"]["WINDOW_SAMPLES"] = 2000
    ckpt["model_state"] = {k: v for k, v in ckpt["model_state"].items() if "classifier" not in k}   # missing layer
    torch.save(ckpt, bad_dir / "best_model_seed42.pt")
    with pytest.raises(CheckpointIncompatibleError):
        clf.load_model("raw", results_root=tmp_path)
    assert GestureClassifier.available_models(tmp_path) == ["raw"]
    assert GestureClassifier.available_models(tmp_path / "nothing") == []

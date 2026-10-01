"""Shared pytest fixtures. Qt runs offscreen; heavy fixtures skip when dataset or checkpoints are absent."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                      # noqa: E402
import pytest                           # noqa: E402

from network import config as C         # noqa: E402
from network.recording import load_recording_file, save_recording   # noqa: E402

SYNTHETIC_SECONDS = 12


@pytest.fixture
def synthetic_data() -> np.ndarray:
    """12 s of noise-like EMG with a distinct amplitude per channel, float32 [24000, 4] in mV."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((SYNTHETIC_SECONDS * C.FS, C.N_CHANNELS)) * np.array([0.08, 0.10, 0.03, 0.02])
    return data.astype(np.float32)


@pytest.fixture
def synthetic_recording_path(tmp_path, synthetic_data):
    return save_recording(tmp_path / "synthetic.mat", synthetic_data, fs=C.FS, source="test",
                          source_config={"note": "pytest fixture"})


@pytest.fixture
def synthetic_recording(synthetic_recording_path):
    return load_recording_file(synthetic_recording_path)


def _has_checkpoints(signal_type: str) -> bool:
    return any((C.RESULTS_DIR / signal_type).glob("best_model_seed*.pt"))


@pytest.fixture(scope="session")
def classifier():
    """The trained raw ensemble on CPU (skipped when results/raw has no checkpoints)."""
    if not _has_checkpoints("raw"):
        pytest.skip("no trained checkpoints in results/raw")
    from network.inference import GestureClassifier
    clf = GestureClassifier(device="cpu")
    clf.load_model("raw")
    return clf


@pytest.fixture
def dataset_file():
    path = C.DATA_ROOT / "raw" / "mat" / "1_raw.mat"
    if not path.exists():
        pytest.skip("dataset file not present")
    return path

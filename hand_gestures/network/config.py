"""Central configuration for the sEMG hand-gesture project.

Every script imports its constants from here so that the raw and the filtered
experiment share exactly the same windowing, split, model and training settings.
"""
from pathlib import Path

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]      # the project folder (network/ is one level down)
DATA_ROOT = PROJECT_ROOT / "sEMG-dataset"          # contains raw/ and filtered/
RESULTS_DIR = PROJECT_ROOT / "results"

# --------------------------------------------------------------------------- dataset facts (verified from the files, see README)
FS = 2000                       # sampling rate in Hz (MAT field "fs")
N_CHANNELS = 4                  # EMG channels CH1..CH4
N_SUBJECTS = 40                 # participant files 1..40
RECORDING_SAMPLES = 1_280_000   # 640 s per participant
SIGNAL_TYPES = ("raw", "filtered")

GESTURES = [                    # class index -> gesture name, in protocol order
    "neutral",      # 0  rest / neutral posture
    "extension",    # 1  wrist extension
    "flexion",      # 2  wrist flexion
    "ulnar_dev",    # 3  ulnar deviation of the wrist
    "radial_dev",   # 4  radial deviation of the wrist
    "grip",         # 5  hand grip
    "abduction",    # 6  abduction of all fingers
    "adduction",    # 7  adduction of all fingers
    "supination",   # 8  forearm supination
    "pronation",    # 9  forearm pronation
]
N_CLASSES = len(GESTURES)

# Recording timeline from Ozdemir et al. (2022), verified against the signals:
# 5 cycles of 134 s (104 s of gestures + 30 s rest). Inside a cycle, gesture k
# (k = 0..9) is performed from  4 + 10*k  s  to  10 + 10*k  s  (6 s each).
N_CYCLES = 5
CYCLE_PERIOD_S = 134
FIRST_GESTURE_OFFSET_S = 4
GESTURE_PERIOD_S = 10
GESTURE_DURATION_S = 6

# --------------------------------------------------------------------------- windowing
WINDOW_S = 1.0                              # window length in seconds
HOP_S = 0.5                                 # hop between windows (50 % overlap)
WINDOW_SAMPLES = int(WINDOW_S * FS)         # 2000 samples
HOP_SAMPLES = int(HOP_S * FS)               # 1000 samples
# Windows start this many seconds after the on-screen cue. 0.0 uses the paper's full 6 s segment
# (11 windows). The failure analysis of the first version showed that the window starting at the cue
# is wrong 62-67 % of the time because the movement begins ~0.5 s later; cross-validation showed that
# skipping the first second improves every fold (results/cv/cv_summary.csv), so 1.0 is the default.
SEGMENT_START_OFFSET_S = 1.0
# Windows must end this many seconds before the end of the 6 s segment (0.0 = use the segment to
# its end). Tested with cross_validate.py because the error rate rises again in the last second.
SEGMENT_END_TRIM_S = 0.0


def windows_per_segment(offset_s: float | None = None, end_trim_s: float | None = None) -> int:
    offset_s = SEGMENT_START_OFFSET_S if offset_s is None else offset_s
    end_trim_s = SEGMENT_END_TRIM_S if end_trim_s is None else end_trim_s
    usable = GESTURE_DURATION_S * FS - int(offset_s * FS) - int(end_trim_s * FS)
    return (usable - WINDOW_SAMPLES) // HOP_SAMPLES + 1


WINDOWS_PER_SEGMENT = windows_per_segment()                 # 11 for offset 0, 9 for offset 1, 7 for offset 1 + trim 1

# --------------------------------------------------------------------------- normalisation
# "train_std": divide by the per-channel std of the TRAINING windows (strict, no test-participant data)
# "rest_rms":  divide by each participant's own rest-posture RMS (calibration setting: uses the
#              participant's unlabelled rest periods, the middle 2 s of the 4 s rests before gestures)
NORMALIZATION = "train_std"

# --------------------------------------------------------------------------- subject-independent split
SPLIT_SEED = 42
N_TRAIN_SUBJECTS = 28
N_VAL_SUBJECTS = 6
N_TEST_SUBJECTS = 6
assert N_TRAIN_SUBJECTS + N_VAL_SUBJECTS + N_TEST_SUBJECTS == N_SUBJECTS

# leave-participants-out cross-validation (cross_validate.py): 5 folds of 8 participants;
# in fold k, fold k is the test set, fold k+1 the validation set, the other 24 participants train
CV_N_FOLDS = 5
CV_SEED = 42

# --------------------------------------------------------------------------- model
CONV_CHANNELS = (32, 64, 128)
KERNEL_SIZES = (15, 9, 5)
POOL_SIZES = (4, 4, 1)          # 1 = no pooling after that block
DROPOUT = 0.3

# --------------------------------------------------------------------------- training
SEED = 42
ENSEMBLE_SIZE = 5               # identical CNNs trained with seeds SEED, SEED+1, ...; their softmax outputs are averaged
ENSEMBLE_SEEDS = tuple(SEED + i for i in range(ENSEMBLE_SIZE))
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8
NUM_WORKERS = 0                 # keep 0 for Windows / Colab simplicity

# training-time augmentation against participant differences (applied to training batches only):
# every channel of every window is multiplied by a random gain drawn log-uniformly from
# AUG_GAIN_RANGE, and Gaussian noise with std AUG_NOISE_STD (in normalised units) is added.
AUGMENT = False
AUG_GAIN_RANGE = (0.7, 1.4)
AUG_NOISE_STD = 0.05

# --------------------------------------------------------------------------- evaluation extras
SMOOTH_WINDOWS = 3              # causal probability averaging over the last N windows (predict.py / evaluate.py)
# merged-class view: gesture pairs the 4 wrist-muscle channels keep confusing
MERGED_CLASS_GROUPS = {"abduction": "fingers abd/add", "adduction": "fingers abd/add",
                       "supination": "forearm rotation", "pronation": "forearm rotation"}


def results_dir(signal_type: str) -> Path:
    """results/<signal_type>/, created on demand."""
    d = RESULTS_DIR / signal_type
    d.mkdir(parents=True, exist_ok=True)
    return d

# Hand Gesture Classification from Surface EMG Using a 1D CNN

University Deep Neural Networks project. A small 1D convolutional network classifies 10 hand and
wrist gestures from 1 s windows of 4-channel surface EMG, evaluated on participants the network has
never seen. The same pipeline is run twice, once on the dataset's **raw** signals and once on its
**filtered** signals, to answer:

> Can a small 1D CNN classify hand gestures from multi-channel sEMG, and how does classification
> performance differ between raw and filtered EMG?

Everything here was derived from the files themselves and from the dataset paper
(`docs/multichannel_emg_signals_for_han.pdf`); nothing about the data was assumed.

---

## 1. Dataset

**Özdemir, Kisa, Guren, Akan (2022), "Dataset for multi-channel surface electromyography (sEMG)
signals of hand gestures", Data in Brief 41, 107921.** Mendeley Data 10.17632/ckwc76xr2z.2.

| Property | Value (verified from the files) |
|---|---|
| Participants | 40 (files `1_raw.mat` … `40_raw.mat`, same for `filtered`); the MAT `iD` field equals the file number |
| Channels | 4 sEMG channels (CH1..CH4) from 4 forearm muscles, unit mV |
| Sampling rate | 2000 Hz (`fs` = 2000, `isi` = 0.5 ms in every file) |
| Recording | 1,280,000 samples = 640 s per participant, identical for all 80 files, no NaN/Inf |
| Formats | `csv/` (4 columns, no header) and `mat/` (`data`, `fs`, `iD`, `isi`, `labels`, `units`, …). The two hold identical numbers; the code loads the MAT files (10 s for all 80) |
| Labels | **Not stored in the files.** They follow from the paper's fixed recording timeline (below) |
| Raw vs filtered | Filtered = BIOPAC online 6th-order Butterworth band-pass 5–500 Hz + 50 Hz notch. Raw = unfiltered ADC output, ±10 mV range (16-bit, step 0.000305 mV) |

### Recording timeline (gives the labels)

5 cycles of 134 s (104 s of gestures, then 30 s rest). Inside a cycle, gesture `k` (k = 0..9) is
performed from **4 + 10·k s to 10 + 10·k s** (6 s), separated by 4 s rests or holds. So gesture `k`
of cycle `c` starts at sample `(134·c + 4 + 10·k) · 2000`. I verified this on every participant: when the
segment grid is shifted by −4..+4 s, shift 0 gives the highest muscle activity for all 40 participants.
Each participant therefore has 50 labelled 6 s segments (5 per gesture), perfectly class balanced.

### Gestures (class index = protocol order)

| idx | name | idx | name |
|---|---|---|---|
| 0 | neutral (rest posture) | 5 | grip |
| 1 | wrist extension | 6 | abduction of all fingers |
| 2 | wrist flexion | 7 | adduction of all fingers |
| 3 | ulnar deviation | 8 | supination |
| 4 | radial deviation | 9 | pronation |

### Data-quality observations (from Phase 1 inspection)

- Raw channels 1–3 have large participant-specific DC offsets (channel means from −8.6 to +8.1 mV) and
  slow drift of several mV; the EMG itself is ~0.03–0.1 mV std. Raw channel 4 has no offset and is
  almost identical to its filtered version (r ≈ 0.92), so it appears to be high-pass coupled already.
- Raw signals hit the ±10 mV ADC limit in 9 participants, substantially in participants 3, 14 and 17
  (≈118 s, 58 s and 17 s). Where the raw signal is saturated the filtered signal becomes almost flat.
- Filtered signals have a ~1.4 s filter start-up transient; it lies before the first labelled segment.
- Mean activity per gesture differs a lot: extension is by far the strongest, neutral the weakest, and
  grip, supination and pronation are only slightly above neutral.
- After the on-screen cue, muscle activity needs ~0.5 s to reach half of its plateau (peak at ~1 s).

---

## 2. Preprocessing and input shape

```
40 recordings [1,280,000 samples x 4 channels]
   -> 50 labelled 6 s segments per participant (timeline above)
   -> 1.0 s windows (2000 samples), hop 0.5 s (50 % overlap), cut INSIDE each segment,
      the first window starting 1.0 s after the on-screen cue (SEGMENT_START_OFFSET_S)
      -> 9 windows per segment, 450 per participant, 18,000 in total (1800 per class)
   -> per-window mean removal, then divide by per-channel std of the TRAINING windows
   -> tensor [n_windows, 4, 2000]  (model input [batch, 4, 2000])
```

**Why skip the first second?** The first version of the pipeline used the paper's full 6 s segments
(offset 0 s, 11 windows per segment). Its failure analysis showed that the window starting at the cue
was wrong 62 to 67 % of the time: the muscles only reach half of their activity ~0.5 s after the cue,
so that window is mostly rest labelled as a gesture. Skipping 1 s removes those windows from training
and testing. Cross-validation over all 40 participants (section 7) confirmed the gain in every fold.

**Normalisation.** `x_norm[c, t] = (x[c, t] − mean_t x[c, ·]) / std_c(train)`.
Step 1 (per-window mean removal) is required because the raw offsets of ±8 mV are 100× the EMG
amplitude and participant specific; without it the raw experiment would be about offsets, not EMG.
Step 2 uses **one scale per channel computed from training participants only**, so it keeps the absolute
amplitude differences between gestures and channels (the main EMG cue) and never touches validation
or test data. Full per-window standardisation was rejected because it would erase the amplitude cue;
per-participant statistics were rejected because they would use test-participant data. After step 1 raw
windows still contain sub-5 Hz drift and 50 Hz line noise; that remaining difference is exactly what
the raw-vs-filtered experiment measures. No resampling is done (resampling raw would itself be a filter).

Each window carries metadata (participant, gesture, cycle, position inside the segment) which is used
in the failure analysis.

---

## 3. Subject-independent split

Participant IDs 1..40 are shuffled once with `numpy.random.default_rng(42)` and whole participants are
assigned to one set each (`data_loader.split_subjects`):

| set | participants |
|---|---|
| train (28) | 4, 5, 6, 7, 8, 10, 11, 12, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 37, 38, 40 |
| validation (6) | 1, 13, 15, 16, 18, 35 |
| test (6) | 2, 3, 9, 14, 34, 39 |

**Why subject-independent?** Windows from one person are strongly correlated: same electrode
placement, skin impedance, muscle anatomy, DC offsets and gesture style, and neighbouring windows
even overlap by 50 %. A random window split would put near-copies of test windows into training and
report an unrealistically high accuracy that says nothing about a new user. Keeping every participant
in exactly one set measures what matters for a gesture interface: performance on an unseen person.
Normalisation statistics, model selection (checkpoint at lowest validation loss) and early stopping
use the training and validation participants only; the test participants are evaluated once, by
`evaluate.py`, after the model is fixed.

---

## 4. CNN architecture (`model.py`, 63,274 trainable parameters)

| layer | output shape | params |
|---|---|---|
| input | [B, 4, 2000] | |
| Conv1d 4→32, k=15, pad 7 · BatchNorm · ReLU · MaxPool 4 | [B, 32, 500] | 2,016 |
| Conv1d 32→64, k=9, pad 4 · BatchNorm · ReLU · MaxPool 4 | [B, 64, 125] | 18,624 |
| Conv1d 64→128, k=5, pad 2 · BatchNorm · ReLU | [B, 128, 125] | 41,344 |
| global average pooling over time | [B, 128] | |
| Dropout · Linear 128→10 | [B, 10] logits | 1,290 |

Receptive field of one output sample: 15 samples = 7.5 ms after block 1, 50 samples = 25 ms after
block 2, 126 samples = 63 ms after block 3 (at 2 kHz). The global average pool then averages
the learned feature maps over the whole second, which is conceptually close to the RMS/MAV features of
the baseline but learned. Cross-entropy loss on the logits. `python model.py` prints this table.

---

## 5. Training procedure (`train.py`)

AdamW, cross-entropy, mini-batches with shuffling, validation after every epoch, checkpoint of the
epoch with the lowest validation loss, early stopping when the validation loss has not improved for
`EARLY_STOPPING_PATIENCE` epochs, fixed seeds (Python, NumPy, PyTorch, cuDNN deterministic), device
chosen automatically (CUDA if available, else CPU). All hyperparameters live in `config.py` and are
identical for the raw and the filtered run. The exact values and the small **validation-only**
hyperparameter check that chose them are documented in section 7 and `results/tuning.csv`.

Per epoch the script prints: device, epoch, train loss, validation loss, train accuracy, validation
accuracy, and stores them in `results/<signal>/history_seed<k>.csv`.

**Ensemble of seeds.** `train.py` trains `ENSEMBLE_SIZE = 5` identical networks with seeds 42 to 46 and
saves each one's best-validation-loss checkpoint (`best_model_seed<k>.pt`). `evaluate.py` and
`predict.py` average the five softmax outputs. Reason: a single run's validation loss jumps by 0.3
between epochs and two single runs disagree on the weak gestures for individual participants (section
8); averaging removes most of that run-to-run variance at a cost of 5 × ~15 s of training.

A training-time augmentation (random per-channel gain, optionally additive noise; `train.augment_batch`)
is implemented and can be switched on with `AUGMENT = True`. It is **off by default** because
cross-validation showed no reliable gain (section 7).

### Two evaluation settings: strict and calibrated

- **Strict (main result):** nothing from a test participant is used before prediction. Normalisation
  uses training-set statistics only (`NORMALIZATION = "train_std"`).
- **Calibrated (reported separately):** the test participant's *unlabelled* signal is used once, as a
  real device would do in a short calibration phase. Two variants are implemented:
  *AdaBN* (`model.adapt_batchnorm`): the trained weights stay fixed and only the BatchNorm running
  mean/variance are re-estimated on that participant's windows before predicting them.
  *Rest-RMS scaling* (`NORMALIZATION = "rest_rms"`): each participant's channels are divided by the RMS
  of their own rest posture (the middle 2 s of the 4 s rests before gestures, 80 s per participant),
  for training and test participants alike. Neither variant sees a gesture label of the test participant,
  but both touch test-participant data, so they are never mixed into the strict numbers.

### Leave-participants-out cross-validation (`cross_validate.py`)

A single 6-participant test set cannot resolve differences of a few points: accuracy varied from 0.52
to 0.77 between the six test participants of the first version. `cross_validate.py` therefore splits
the 40 participants into 5 groups of 8 (fixed seed). In fold k, group k is the test set, group k+1 the
validation set (checkpoint and early stopping) and the remaining 24 participants train. Every
participant is tested exactly once, and the mean and standard deviation over folds say how large a
difference must be before it means anything. Per fold an ensemble of 5 seeds is trained, the single
members are scored too, the per-class recall is recorded, and the fold is evaluated both strictly and
with AdaBN calibration. Nothing is tuned on the test folds. Pipeline variants are selected on the command
line (`--offset`, `--end-trim`, `--normalization`, `--augment`, `--ensemble`).

---

## 6. How to run

```bash
# 1. environment (Python >= 3.10; a project-local venv was used)
python -m venv .venv
.venv/Scripts/activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cu130   # or just: pip install torch  (CPU)
pip install -r requirements.txt

# 2. sanity checks (shapes, labels, one minibatch, loss, backprop, overfitting 80 windows)
python sanity_checks.py --signal filtered
python sanity_checks.py --signal raw

# 3. dataset figures (example recording, example windows)
python plot_examples.py

# 4. train both experiments: 5 seeds each, identical settings (~1 s per epoch on an RTX 4080, minutes on CPU)
python train.py --signal raw
python train.py --signal filtered

# 5. evaluate the ensembles on the test participants (metrics, confusion matrices, failure cases, curves,
#    plus the smoothed, merged-class and AdaBN-calibrated views)
python evaluate.py --signal raw
python evaluate.py --signal filtered

# 6. feature baseline and the raw-vs-filtered comparison
python baseline.py
python compare.py

# optional: the validation-only hyperparameter check (test set never used)
python tune.py

# leave-participants-out cross-validation of a pipeline version (both signals, 5 seeds per fold, ~15 min on the GPU)
python cross_validate.py --label v2_timing_ens                             # current config.py settings
python cross_validate.py --label v1_baseline_ens --offset 0.0              # windows from the cue
python cross_validate.py --label v4_endtrim_ens --end-trim 1.0             # windows 1-5 s instead of 1-6 s
python cross_validate.py --label v5_rest_ens --normalization rest_rms      # per-participant rest-RMS scaling
python cross_validate.py --label v3_aug --augment [--aug-noise 0]          # training-time augmentation

# ask the trained ensemble what it sees in any piece of a recording
python predict.py --signal filtered --subject 2 --cycle 1 --gesture 3 --position 4   # one labelled window
python predict.py --signal filtered --subject 2 --start 14.5                          # 1 s from 14.5 s
python predict.py --signal filtered --subject 2 --start 0 --end 104 --plot            # sliding window + figure
python predict.py --signal filtered --subject 2 --start 0 --end 104 --smooth 3 --plot # ... averaging the last 3 windows
```

`predict.py` prints, per window, the true label (from the timeline, or "rest" between segments), the
predicted gesture, its probability and the top-3 classes; `--smooth N` averages the probabilities of the
last N windows before deciding (more evidence, more latency); `--plot` saves the window with its
probability bars, or a prediction timeline for a sliding window.

The dataset folder `sEMG-dataset/{raw,filtered}/mat/` must sit next to the scripts (see `config.py`).

---

## 7. Results

Final settings (`config.py`), identical for both signals: AdamW, learning rate 1e-3, weight decay 1e-4,
batch 64, dropout 0.3, max 40 epochs, early-stopping patience 8, windows 1 to 6 s after the cue
(9 per segment), training-set channel scaling, no augmentation, **ensemble of 5 seeds (42 to 46)**.
A small **validation-only** hyperparameter check (`tune.py`, run on the first pipeline version, 8
configurations × 2 signals) had given mean validation accuracies between 0.585 and 0.610, all within the
noise of 6 validation participants, so the optimiser settings were never changed (`results/tuning.csv`).

Two settings are reported throughout:
**strict** (nothing from a test participant is used before prediction) and
**calibrated** (per-participant AdaBN: the BatchNorm statistics are re-estimated on the test participant's
own unlabelled windows, weights unchanged; see section 5).

### 7.1 Pipeline versions, cross-validated over all 40 participants

`cross_validate.py`, 5 folds × 8 test participants, 24 training and 8 validation participants per fold,
mean ± std over folds (`results/cv/cv_summary.csv`, `cv_results.png`, `cv_per_class.png`). The first
block used single networks, the second block 5-seed ensembles and also reports the calibrated setting.

| version | windows | normalisation | raw strict | filtered strict | raw calibrated | filtered calibrated |
|---|---|---|---|---|---|---|
| v1 baseline, single net | from the cue, 11/segment | train std | 0.597 ± 0.042 | 0.585 ± 0.029 | | |
| v2 timing, single net | 1–6 s, 9/segment | train std | 0.650 ± 0.039 | 0.622 ± 0.033 | | |
| v3 timing + gain & noise augmentation | 1–6 s | train std | 0.619 ± 0.036 | 0.627 ± 0.025 | | |
| v3b timing + gain-only augmentation | 1–6 s | train std | 0.638 ± 0.051 | 0.625 ± 0.040 | | |
| v1 baseline, **ensemble of 5** | from the cue | train std | 0.621 ± 0.047 | 0.607 ± 0.032 | 0.737 ± 0.012 | 0.717 ± 0.021 |
| **v2 timing, ensemble of 5 (final)** | 1–6 s | train std | **0.659 ± 0.039** | **0.634 ± 0.024** | **0.796 ± 0.022** | **0.773 ± 0.028** |
| v4 timing + end trim, ensemble | 1–5 s, 7/segment | train std | 0.669 ± 0.035 | 0.642 ± 0.039 | 0.806 ± 0.029 | 0.780 ± 0.031 |
| v5 timing + rest-RMS scaling, ensemble | 1–6 s | participant's rest RMS (calibration-type) | 0.695 ± 0.016 | 0.694 ± 0.016 | 0.781 ± 0.025 | 0.768 ± 0.028 |

Single networks inside the ensembles averaged 0.599 / 0.587 (v1), 0.638 / 0.616 (v2), 0.649 / 0.618 (v4)
and 0.677 / 0.672 (v5) for raw / filtered.

What the cross-validation shows:

- **Skipping the first second after the cue** (v1 → v2) is the largest strict gain, +5 points for raw and
  +3 to +4 for filtered, improving every fold. It is the one data change in the final pipeline.
- **The 5-seed ensemble** adds about +2 points over its own single networks in every version, and the
  per-class results stop depending on the seed.
- **Calibration is the largest lever overall**: AdaBN adds +14 points (raw 0.659 → 0.796, filtered
  0.634 → 0.773), with the fold spread unchanged. Rest-RMS scaling, the other calibration-type variant,
  gives +4 to +6 points strict-style (0.695 / 0.694) and halves the fold spread, but adding AdaBN on top of
  it does not reach AdaBN alone, so the two mostly correct the same thing. Both use the test participant's
  unlabelled signal and are therefore reported separately.
- **Trimming the last second** (v4) gives +1 point for both signals, inside the fold spread; it was not
  adopted. **Augmentation** (v3, v3b) gave no gain and stays off.
- **Raw is not worse than filtered.** Raw scores 1 to 3 points higher in every strict version; with rest-RMS
  scaling or calibration the two are within a point of each other. The first single-split run had
  suggested filtered was better, which is exactly the kind of difference a 6-participant test set cannot
  resolve.
- **Per class** (`cv_per_class.png`, strict ensembles): from v1 to v2 recall rose for supination (raw
  0.58 → 0.76), abduction, radial deviation and grip; adduction fell slightly (raw 0.40 → 0.35) and
  pronation stayed at 0.27 to 0.41. Adduction and pronation are the weakest classes in every version;
  rest-RMS scaling lifts neutral to 0.88 to 0.91 and the finger pair by 8 to 12 points.

### 7.2 Main experiment: fixed split, final pipeline

Test participants 2, 3, 9, 14, 34, 39; 2700 windows, 270 per class. Selected epochs per seed: raw
[6, 3, 9, 3, 14], filtered [15, 5, 9, 8, 10]. `results/comparison.md`, `raw_vs_filtered.png`.

| metric | raw EMG | filtered EMG |
|---|---|---|
| validation accuracy (ensemble, used for nothing but reporting) | 0.649 | 0.651 |
| test accuracy, mean of the 5 single networks | 0.656 | 0.600 |
| **test accuracy, ensemble (strict)** | **0.674** | **0.624** |
| test macro precision / recall (ensemble) | 0.700 / 0.674 | 0.668 / 0.624 |
| **test macro F1, ensemble (strict)** | **0.674** | **0.632** |
| test accuracy, probabilities averaged over the last 3 windows | 0.687 | 0.627 |
| test accuracy, majority vote over a whole 6 s segment | 0.693 | 0.643 |
| test accuracy, merged classes (finger pair, rotation pair; 8 groups) | 0.727 | 0.683 |
| **test accuracy / macro F1, calibrated (AdaBN)** | **0.851 / 0.849** | **0.845 / 0.846** |
| baseline: logistic regression on 12 amplitude features, test accuracy / macro F1 | 0.638 / 0.651 | 0.598 / 0.619 |
| baseline: random forest on the same features, test accuracy / macro F1 | 0.598 / 0.606 | 0.598 / 0.606 |
| chance level | 0.100 | 0.100 |
| *first version for reference (single net, windows from the cue): accuracy / macro F1* | *0.598 / 0.592* | *0.615 / 0.627* |

The fixed-split calibrated numbers (0.85) are higher than the cross-validated ones (0.77 to 0.80): these
six participants happen to calibrate well. Quote the cross-validated values.

### Per-class test F1 (fixed split, strict ensemble)

| gesture | raw | filtered | | gesture | raw | filtered |
|---|---|---|---|---|---|---|
| flexion | 0.991 | 0.993 | | ulnar deviation | 0.716 | 0.645 |
| extension | 0.819 | 0.814 | | abduction | 0.605 | 0.565 |
| grip | 0.804 | 0.728 | | neutral | 0.507 | 0.457 |
| supination | 0.772 | 0.657 | | pronation | 0.491 | 0.358 |
| radial deviation | 0.749 | 0.724 | | adduction | 0.285 | 0.379 |

Most accurately classified gestures: **flexion** and **extension** in both experiments. Most commonly
confused pairs (true → predicted, share of that class): raw: adduction → neutral (46 %), pronation →
neutral (32 %), extension → abduction (20 %), ulnar deviation → abduction (18 %); filtered: pronation →
neutral (37 %), adduction → neutral (34 %), neutral → pronation (22 %), grip → neutral (22 %).
Strict test accuracy per participant ranges from 0.55 to 0.78 (raw) and 0.48 to 0.78 (filtered);
calibrated from 0.78 to 0.90 (raw) and 0.75 to 0.92 (filtered).

### What the experiment says (and what it does not)

- A 63k-parameter 1D CNN (five copies averaged) trained on 24 to 28 participants classifies 10 gestures
  of unseen participants at 0.63 to 0.66 window accuracy (strict, 6× chance) from 1 s of 4-channel EMG.
  Letting the network see a new participant's unlabelled signal once (calibration) lifts this to about
  0.77 to 0.80, and 0.85 on the fixed split.
- With per-window mean removal already in place, the dataset's band-pass and notch filtering gave **no
  advantage**: raw scored equal or slightly higher in every setting. The CNN copes with the remaining
  drift and line noise by itself. This holds for this dataset, model and normalisation; it is not a
  general statement about EMG filtering.
- The limiting factor is participant-to-participant variability, not model capacity: the feature
  baselines are within 1 to 4 points of the strict CNN, training accuracy keeps rising while validation
  loss turns up after 5 to 15 epochs, and a 20-line statistics re-estimation per participant is worth
  more than every architectural or training change tried.
- The gains came from the data side: removing the mislabeled first second (+4 to +5), averaging seeds
  (+2), and adapting to the participant (+14). Augmentation, hyperparameters and trimming the segment end
  did not move the result beyond the fold noise.

### Figures

| figure | file |
|---|---|
| example recording, all channels, first cycle | `results/figures/example_recording_{raw,filtered}.png` |
| example windows per gesture | `results/figures/example_windows_{raw,filtered}.png` |
| training vs validation loss / accuracy, all 5 seeds | `results/{raw,filtered}/curve_loss.png`, `curve_accuracy.png` |
| confusion matrix (counts, normalised) | `results/{raw,filtered}/confusion_matrix{,_normalized}.png` |
| raw vs filtered comparison (fixed split) | `results/raw_vs_filtered.png`, `results/comparison.md` |
| cross-validation of the pipeline versions, strict and calibrated | `results/cv/cv_results.png`, `cv_summary.csv` |
| per-class recall per pipeline version | `results/cv/cv_per_class.png` |
| failure cases | `results/{raw,filtered}/failure_cases.png` + `failure_cases.csv` |
| error rate vs position in the segment, accuracy per participant | `results/error_by_position.png`, `results/accuracy_per_subject.png` |
| prediction timelines for one participant (strict, smoothed, calibrated) | `results/{raw,filtered}/predict_P2_0-104s*.png` |

---

## 8. Failure cases ("where and why the model failed")

`evaluate.py` picks, for the most frequent confusions, the most confident mistake and shows it next to a
correctly classified window of the same gesture (same participant where one exists), with the softmax
probabilities, participant ID, cycle and window position, and it checks whether the raw recording was
saturated during that window. Everything below is read from `results/*/metrics.json`,
`failure_cases.csv` and the figures of the final pipeline; first-version numbers are quoted where they
motivated a change.

**Directly observable in the data**

1. *Where in time (first version):* with windows starting at the cue, the first window was wrong 62 %
   (filtered) / 67 % (raw) of the time versus 31 to 40 % for windows starting 1.5 s or later. Phase 1 had
   shown that muscle activity reaches half of its plateau only ~0.5 s after the cue. This motivated the
   1 s offset.
2. *Where in time (final pipeline):* the error is U-shaped over the segment (`error_by_position.png`):
   0.39 / 0.43 (raw / filtered) for the window at 1 to 2 s, lowest at 2.5 to 3.5 s (0.25 / 0.32), and
   0.40 / 0.42 in the last second. Trimming that last second gained only one point in cross-validation.
3. *Which gestures:* errors concentrate on the gestures with the lowest measured activity (adduction,
   pronation, grip); the main confusion is with **neutral**, which has high recall but low precision and
   acts as the sink class. Pronation and supination are confused with each other; extension and ulnar
   deviation are sometimes predicted as abduction. Flexion (F1 0.99) and extension are rarely wrong.
4. *Which participants:* errors are participant specific. For participant 34 none of the 45 adduction
   windows is correct with either signal (strict); for participant 14 none of the 45 neutral windows is
   correct with the filtered ensemble (predicted as pronation, p = 0.85). Participant 34's adduction
   windows have about half the RMS of participant 3's (0.037 / 0.028 / 0.013 mV vs 0.063 / 0.063 /
   0.035 mV on CH1–CH3), whose adduction is classified correctly.
5. *What calibration fixes:* with AdaBN the same ensembles reach 0.75 to 0.92 for every test participant;
   participant 34 goes from 0.55 to 0.81 (raw) and participant 14 from 0.71 to 0.90 (filtered). For
   participant 2, first cycle, grip, pronation and supination go from 1 to 2 of 12 windows to 9 to 12 of 12
   (`predict_P2_0-104s_calibrated.png`). What stays wrong after calibration is this participant's neutral
   (filtered, 0 of 12) and pronation (raw, 4 of 12).
6. *Saturated channels:* in the pronation → neutral failure of participant 3 (filtered, p = 0.89) raw
   channel 2 sits at the +10 mV ADC limit for 96 % of the window and appears as a flat line with isolated
   spikes; in the ulnar deviation → abduction failure of participant 3 (raw, p = 0.81) it is saturated for
   75 % of the window. Where the raw signal is clipped, the filtered signal is flat too.
7. *Confidence:* the plotted failures are confident (p = 0.72 to 0.95 for the wrong class), so they are not
   borderline decisions, and averaging five seeds did not make them less confident.

**Plausible interpretations (not verified)**

- The remaining first- and last-window errors look like label timing: the label covers the full 6 s,
  but the participant starts late and may stop early. Labels derived from the signal itself (movement
  onset/offset detection) would test this; the fixed 1 s trim at the end was not worth its data loss.
- Gestures that create little activity in the four recorded forearm muscles are hard to separate from
  the neutral posture with amplitude-like features, and the CNN's global average pooling makes it
  largely an amplitude/shape-statistics model.
- The size of the calibration gain says that most of the strict error is a mismatch of signal statistics
  between people (electrode placement, skin, muscle size), not a missing pattern: the same weights
  classify a new participant well once the normalisation layers see that person's data. Random gain
  augmentation did not reproduce this, so the mismatch is not a simple overall gain but differs per
  channel and feature. Per-participant calibration is therefore the realistic path for a device.
- Where the raw signal is clipped at the ADC limit, the online filter outputs an almost flat line, so the
  gesture information is lost in both signal versions; this is a recording problem, not a model problem.

---

## 9. Code Autopsy candidates (short snippets for the exam slide)

**Window creation** (`data_loader.py`, `make_windows`): the whole labelling + windowing logic.

```python
for cycle in range(N_CYCLES):
    for gesture in range(N_CLASSES):
        start = segment_start_sample(cycle, gesture)          # (134*cycle + 4 + 10*gesture) * FS
        end = start + GESTURE_DURATION_S * FS                 # 6 s segment
        for position, w0 in enumerate(range(start, end - WINDOW_SAMPLES + 1, HOP_SAMPLES)):
            X.append(recording[w0:w0 + WINDOW_SAMPLES].T)    # -> [channels, time]
            meta.append((subject, gesture, cycle, position))
```

**Normalisation** (`data_loader.py`): train statistics only.

```python
def normalize(X, channel_std):          # channel_std computed from TRAINING windows
    return (X - X.mean(axis=2, keepdims=True)) / channel_std[None, :, None]
```

**CNN forward pass** (`model.py`):

```python
def forward(self, x):                      # x: [batch, channels, time]
    x = self.blocks(x)                     # conv-BN-ReLU-pool blocks -> [batch, 128, 125]
    x = self.global_pool(x).squeeze(-1)    # average over time        -> [batch, 128]
    x = self.dropout(x)
    return self.classifier(x)              # logits                   -> [batch, 10]
```

**Calibration** (`model.py`, `adapt_batchnorm`): the whole calibration setting in six lines.

```python
adapted = copy.deepcopy(model)                 # trained weights stay untouched
for module in adapted.modules():
    if isinstance(module, nn.BatchNorm1d):
        module.reset_running_stats(); module.momentum = None   # cumulative average
adapted.train()
with torch.no_grad():
    for xb, _ in to_loader(X_participant, dummy_labels): adapted(xb.to(device))   # re-estimate mean / var
adapted.eval()
```

**Training update** (`train.py`, `run_epoch`):

```python
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    logits = model(xb)                     # forward pass
    loss = criterion(logits, yb)           # cross-entropy on logits
    if training:
        optimizer.zero_grad()
        loss.backward()                    # backpropagation
        optimizer.step()                   # AdamW update
```

---

## 10. Project structure

```
config.py          all constants: dataset facts, timeline, windowing, split, model, training
data_loader.py     load MAT -> label segments -> windows -> subject split -> normalisation -> tensors
model.py           EMGCNN + layer-by-layer summary + inference helpers (ensemble loading, AdaBN calibration)
train.py           training loop (fit), checkpointing, early stopping, CLI
evaluate.py        test-set metrics, confusion matrices, failure cases, curves
cross_validate.py  leave-participants-out cross-validation (5 folds x 8 participants), pipeline comparison
predict.py         predictions + probabilities for any window or time range of a recording
baseline.py        RMS / MAV / WL features + logistic regression and random forest
tune.py            small validation-only hyperparameter check
compare.py         raw-vs-filtered table and figures
plots.py           all figures
plot_examples.py   dataset figures (example recording, example windows)
sanity_checks.py   pre-training checks incl. overfitting a tiny subset
results/           raw/, filtered/ (best_model_seed*.pt, history_seed*.csv, metrics, figures), figures/, cv/,
                   baseline_results.json, comparison.md, tuning.csv
docs/              the dataset paper
sEMG-dataset/      the data (not included)
```

## Reproducibility

Seeds are fixed (`SEED = 42`, `SPLIT_SEED = 42`), cuDNN runs in deterministic mode and the
DataLoader shuffles with a seeded generator. GPU training can still differ in the last digits between
hardware/driver versions; the numbers in section 7 were produced on an NVIDIA RTX 4080 SUPER with
PyTorch 2.14.1 + CUDA 13.0, Python 3.14.

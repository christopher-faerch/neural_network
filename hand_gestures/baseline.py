"""Simple non-deep-learning baseline: hand-crafted EMG features + classical classifiers.

    python baseline.py

Per channel and window: RMS, mean absolute value (MAV) and waveform length (WL) -> 12 features.
Uses exactly the same windows, normalisation and subject split as the CNN, for raw and filtered.
Writes results/baseline_results.json. This is only a reference point; the CNN is the main method.
"""
import json
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from data_loader import prepare_data

FEATURE_NAMES = [f"{f}_CH{c + 1}" for f in ("RMS", "MAV", "WL") for c in range(C.N_CHANNELS)]


def extract_features(X):
    """X: [n_windows, channels, time] -> [n_windows, 3 * channels]."""
    rms = np.sqrt((X ** 2).mean(axis=2))              # root mean square
    mav = np.abs(X).mean(axis=2)                      # mean absolute value
    wl = np.abs(np.diff(X, axis=2)).sum(axis=2)       # waveform length
    return np.concatenate([rms, mav, wl], axis=1)


def make_models():
    return {
        "logistic_regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=5000, random_state=C.SEED)),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=C.SEED, n_jobs=-1),
    }


def main():
    results = {}
    for signal_type in C.SIGNAL_TYPES:
        t0 = time.time()
        data = prepare_data(signal_type, verbose=False)
        feats = {k: extract_features(data[k][0]) for k in ("train", "val", "test")}
        labels = {k: data[k][1] for k in ("train", "val", "test")}
        print(f"[{signal_type}] features {feats['train'].shape[1]} per window: {FEATURE_NAMES}")
        results[signal_type] = {}
        for name, clf in make_models().items():
            clf.fit(feats["train"], labels["train"])
            res = {}
            for k in ("val", "test"):
                p = clf.predict(feats[k])
                res[f"{k}_acc"] = round(float(accuracy_score(labels[k], p)), 4)
                res[f"{k}_macro_f1"] = round(float(f1_score(labels[k], p, average="macro")), 4)
            results[signal_type][name] = res
            print(f"  {name:20s} val acc {res['val_acc']:.4f}  test acc {res['test_acc']:.4f}  test macro F1 {res['test_macro_f1']:.4f}")
        print(f"  ({time.time() - t0:.0f} s)")
    results["features"] = FEATURE_NAMES
    results["split"] = data["split"].as_dict()
    out = C.RESULTS_DIR / "baseline_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

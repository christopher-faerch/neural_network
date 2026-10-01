"""How many EMG channels are needed? Train and test the CNN on every subset of the four channels.

    python -m network.channel_ablation --signal raw --seeds 3

Uses the fixed participant split and the default pipeline (windows 1-6 s after the cue, training-set
scaling). For each channel subset an ensemble of `--seeds` networks is trained; the test participants
are scored strictly and with per-participant AdaBN calibration. Nothing is tuned on the test set: the
subsets are compared, not selected, and the result is reported as is.

Writes results/channel_ablation_<signal>.csv and results/channel_ablation_<signal>.png.
"""
import argparse
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from network import config as C
from network import plots
from network.data_loader import prepare_data
from network.model import predict_proba, predict_proba_adabn
from network.train import fit, get_device, set_seed

CHANNEL_NAMES = {1: "CH1", 2: "CH2", 3: "CH3", 4: "CH4"}


def subset_data(data, channels):
    """Keep only the given 1-based channels in every split (normalisation is per channel, so it stays valid)."""
    idx = [c - 1 for c in channels]
    out = {"split": data["split"], "channel_std": data["channel_std"][idx], "normalization": data["normalization"]}
    for name in ("train", "val", "test"):
        X, y, meta = data[name]
        out[name] = (np.ascontiguousarray(X[:, idx, :]), y, meta)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", default="raw", choices=C.SIGNAL_TYPES)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    args = ap.parse_args()
    device = get_device()
    data = prepare_data(args.signal, verbose=False)
    X_te, y_te, m_te = data["test"]
    subsets = [combo for k in range(1, C.N_CHANNELS + 1) for combo in itertools.combinations(range(1, C.N_CHANNELS + 1), k)]
    print(f"device: {device}; {args.signal} signal; {len(subsets)} channel subsets x {args.seeds} seeds; "
          f"test participants {data['split'].test}\n")
    rows = []
    for channels in subsets:
        t0 = time.time()
        sub = subset_data(data, channels)
        models = []
        for seed in [C.SEED + i for i in range(args.seeds)]:
            set_seed(seed)
            _, best, model = fit(sub, device, seed=seed, max_epochs=args.epochs, verbose=False)
            model.load_state_dict(best["model_state"])
            models.append(model)
        Xs = sub["test"][0]
        pred = predict_proba(models, Xs, device).argmax(1)
        pred_cal = predict_proba_adabn(models, Xs, m_te, device).argmax(1)
        row = dict(n_channels=len(channels), channels="+".join(CHANNEL_NAMES[c] for c in channels),
                   test_acc=round(float(accuracy_score(y_te, pred)), 4),
                   test_macro_f1=round(float(f1_score(y_te, pred, average="macro")), 4),
                   test_acc_calibrated=round(float(accuracy_score(y_te, pred_cal)), 4),
                   test_macro_f1_calibrated=round(float(f1_score(y_te, pred_cal, average="macro")), 4),
                   per_class_recall=" ".join(f"{r:.2f}" for r in
                                             [float((pred[y_te == k] == k).mean()) for k in range(C.N_CLASSES)]),
                   seconds=round(time.time() - t0, 1))
        rows.append(row)
        print(f"{row['channels']:<18s} strict acc {row['test_acc']:.3f}  F1 {row['test_macro_f1']:.3f}   |  "
              f"calibrated acc {row['test_acc_calibrated']:.3f}  F1 {row['test_macro_f1_calibrated']:.3f}   ({row['seconds']:.0f}s)")
    df = pd.DataFrame(rows)
    out_csv = C.RESULTS_DIR / f"channel_ablation_{args.signal}.csv"
    df.to_csv(out_csv, index=False)
    print("\nbest subset per channel count (strict / calibrated accuracy):")
    for n, g in df.groupby("n_channels"):
        b = g.sort_values("test_acc", ascending=False).iloc[0]
        bc = g.sort_values("test_acc_calibrated", ascending=False).iloc[0]
        print(f"  {n} channel(s): {b['channels']:<18s} {b['test_acc']:.3f}   |   {bc['channels']:<18s} {bc['test_acc_calibrated']:.3f}")
    plots.plot_channel_ablation(df, C.RESULTS_DIR / f"channel_ablation_{args.signal}.png", args.signal)
    print(f"saved {out_csv} and channel_ablation_{args.signal}.png")


if __name__ == "__main__":
    main()

"""Leave-participants-out cross-validation: a reliable estimate of test accuracy and its spread.

    python cross_validate.py --label v2_timing                                   # settings from config.py, both signals
    python cross_validate.py --label v1_baseline --offset 0                      # override the window start offset
    python cross_validate.py --label v4_endtrim --end-trim 1.0                   # windows end 1 s before the segment end
    python cross_validate.py --label v5_rest --normalization rest_rms            # per-participant rest-RMS scaling
    python cross_validate.py --label v3_aug --augment [--aug-noise 0]            # training-time augmentation
    python cross_validate.py --label smoke --signal filtered --epochs 1 --ensemble 2    # quick smoke test

5 folds of 8 participants (fixed seed). In fold k, fold k is the test set, fold k+1 is the validation
set (checkpoint + early stopping) and the remaining 24 participants train. Every participant is a
test participant exactly once. Per fold an ensemble of --ensemble networks (seeds SEED, SEED+1, ...)
is trained and its averaged softmax is evaluated; single-member accuracies are stored too. Each fold
is additionally evaluated in the CALIBRATION setting (per-participant AdaBN: BatchNorm statistics
re-estimated on the test participant's unlabelled windows). Nothing is tuned on the test folds.

Writes results/cv/<label>_<signal>_folds.csv, <label>_<signal>_confusion.csv (summed over folds),
updates results/cv/cv_summary.csv, cv_results.png and cv_per_class.png.
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import config as C
import plots
from data_loader import build_windows, cv_folds, prepare_split_arrays
from model import EMGCNN, predict_proba, predict_proba_adabn
from train import fit, get_device, set_seed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="name of this pipeline version in the summary table")
    ap.add_argument("--signal", default="both", choices=("raw", "filtered", "both"))
    ap.add_argument("--offset", type=float, default=None, help="window start offset after the cue [s] (default: config)")
    ap.add_argument("--end-trim", type=float, default=None, help="seconds cut from the end of each segment (default: config)")
    ap.add_argument("--normalization", choices=("train_std", "rest_rms"), default=None, help="default: config")
    ap.add_argument("--augment", action="store_true", help="switch training-time augmentation on (default: config)")
    ap.add_argument("--aug-gain", type=float, nargs=2, metavar=("LO", "HI"), default=None, help="gain range, e.g. 0.7 1.4")
    ap.add_argument("--aug-noise", type=float, default=None, help="noise std in normalised units, 0 disables the noise")
    ap.add_argument("--ensemble", type=int, default=C.ENSEMBLE_SIZE, help="networks per fold (seeds SEED, SEED+1, ...)")
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--plot-only", action="store_true", help="only redraw the figures from results/cv/cv_summary.csv")
    args = ap.parse_args()
    if args.plot_only:
        return write_summary_and_figures(C.RESULTS_DIR / "cv", [], None)

    if args.offset is not None:                         # override the pipeline settings for this run
        C.SEGMENT_START_OFFSET_S = args.offset
    if args.end_trim is not None:
        C.SEGMENT_END_TRIM_S = args.end_trim
    C.WINDOWS_PER_SEGMENT = C.windows_per_segment()
    if args.aug_gain is not None:
        C.AUG_GAIN_RANGE = tuple(args.aug_gain)
    if args.aug_noise is not None:
        C.AUG_NOISE_STD = args.aug_noise
    normalization = args.normalization or C.NORMALIZATION
    augment = True if args.augment else C.AUGMENT
    seeds = [C.SEED + i for i in range(args.ensemble)]
    signals = list(C.SIGNAL_TYPES) if args.signal == "both" else [args.signal]
    device = get_device()
    out_dir = C.RESULTS_DIR / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = cv_folds()
    aug_desc = f"on (gain {C.AUG_GAIN_RANGE}, noise std {C.AUG_NOISE_STD})" if augment else "off"
    print(f"device: {device}; label '{args.label}'; windows {C.SEGMENT_START_OFFSET_S:.1f}-"
          f"{C.GESTURE_DURATION_S - C.SEGMENT_END_TRIM_S:.1f} s after the cue ({C.windows_per_segment()} per segment); "
          f"normalisation {normalization}; augmentation {aug_desc}; ensemble {len(seeds)}; {len(folds)} folds; signals {signals}")
    for k, f in enumerate(folds):
        print(f"  fold {k}: test {f.test}  val {f.val}")

    summary_rows = []
    for signal_type in signals:
        X, y, meta, rest_scale = build_windows(signal_type, verbose=False)
        rows, cm_total = [], np.zeros((C.N_CLASSES, C.N_CLASSES), dtype=int)
        for k, split in enumerate(folds):
            t0 = time.time()
            data = prepare_split_arrays(X, y, meta, split, rest_scale, normalization)
            X_te, y_te, m_te = data["test"]
            models, best_epochs, val_accs = [], [], []
            for seed in seeds:
                _, best, model = fit(data, device, seed=seed, max_epochs=args.epochs, augment=augment, verbose=False)
                model.load_state_dict(best["model_state"])            # weights of the best validation epoch
                models.append(model)
                best_epochs.append(best["epoch"])
                val_accs.append(best["val_acc"])
            member_acc = [float(accuracy_score(y_te, predict_proba(m, X_te, device).argmax(1))) for m in models]
            probs = predict_proba(models, X_te, device)
            pred = probs.argmax(1)
            acc, f1 = accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")
            cm = confusion_matrix(y_te, pred, labels=range(C.N_CLASSES))
            cm_total += cm
            recall = cm.diagonal() / cm.sum(axis=1)
            pred_adabn = predict_proba_adabn(models, X_te, m_te, device).argmax(1)
            acc_adabn, f1_adabn = accuracy_score(y_te, pred_adabn), f1_score(y_te, pred_adabn, average="macro")
            rows.append(dict(label=args.label, signal=signal_type, fold=k, test_subjects=" ".join(map(str, split.test)),
                             best_epochs=" ".join(map(str, best_epochs)), mean_val_acc=round(float(np.mean(val_accs)), 4),
                             member_test_acc=" ".join(f"{a:.4f}" for a in member_acc), mean_member_test_acc=round(float(np.mean(member_acc)), 4),
                             test_acc=round(float(acc), 4), test_macro_f1=round(float(f1), 4),
                             test_acc_adabn=round(float(acc_adabn), 4), test_macro_f1_adabn=round(float(f1_adabn), 4),
                             **{f"recall_{g}": round(float(r), 4) for g, r in zip(C.GESTURES, recall)},
                             seconds=round(time.time() - t0, 1)))
            print(f"[{signal_type:8s}] fold {k}: members {member_acc[0]:.3f}..{member_acc[-1]:.3f} (mean {np.mean(member_acc):.4f})  "
                  f"ensemble acc {acc:.4f}  F1 {f1:.4f}  |  AdaBN acc {acc_adabn:.4f}   ({time.time() - t0:.0f}s)")
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{args.label}_{signal_type}_folds.csv", index=False)
        pd.DataFrame(cm_total, index=C.GESTURES, columns=C.GESTURES).to_csv(out_dir / f"{args.label}_{signal_type}_confusion.csv")
        summary_rows.append(dict(label=args.label, signal=signal_type, offset_s=C.SEGMENT_START_OFFSET_S,
                                 end_trim_s=C.SEGMENT_END_TRIM_S, normalization=normalization, augment=augment,
                                 aug_gain=str(C.AUG_GAIN_RANGE) if augment else "", aug_noise=C.AUG_NOISE_STD if augment else "",
                                 windows_per_segment=C.windows_per_segment(), ensemble=len(seeds),
                                 mean_member_acc=round(df.mean_member_test_acc.mean(), 4),
                                 mean_test_acc=round(df.test_acc.mean(), 4), std_test_acc=round(df.test_acc.std(ddof=0), 4),
                                 mean_test_macro_f1=round(df.test_macro_f1.mean(), 4), std_test_macro_f1=round(df.test_macro_f1.std(ddof=0), 4),
                                 mean_test_acc_adabn=round(df.test_acc_adabn.mean(), 4), std_test_acc_adabn=round(df.test_acc_adabn.std(ddof=0), 4),
                                 mean_test_macro_f1_adabn=round(df.test_macro_f1_adabn.mean(), 4),
                                 mean_val_acc=round(df.mean_val_acc.mean(), 4), fold_accs=";".join(f"{a:.4f}" for a in df.test_acc),
                                 **{f"recall_{g}": round(df[f"recall_{g}"].mean(), 4) for g in C.GESTURES}))
        print(f"[{signal_type:8s}] single members {df.mean_member_test_acc.mean():.4f} | ensemble test acc {df.test_acc.mean():.4f} ± "
              f"{df.test_acc.std(ddof=0):.4f}, macro F1 {df.test_macro_f1.mean():.4f} | AdaBN acc {df.test_acc_adabn.mean():.4f} ± "
              f"{df.test_acc_adabn.std(ddof=0):.4f}, macro F1 {df.test_macro_f1_adabn.mean():.4f}")
        del X, y, meta, rest_scale

    write_summary_and_figures(out_dir, summary_rows, args.label)


def write_summary_and_figures(out_dir, summary_rows, label):
    """Update results/cv/cv_summary.csv (one row per label x signal) and redraw the two CV figures."""
    summary_path = out_dir / "cv_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if summary_rows:
        new = pd.DataFrame(summary_rows)
        if len(summary):
            summary = summary[~((summary.label == label) & (summary.signal.isin(new.signal)))]
        summary = pd.concat([summary, new], ignore_index=True)
        summary.to_csv(summary_path, index=False)
    cols = ["label", "signal", "offset_s", "end_trim_s", "normalization", "augment", "ensemble", "mean_member_acc",
            "mean_test_acc", "std_test_acc", "mean_test_macro_f1", "mean_test_acc_adabn"]
    print("\nsummary over all cross-validated pipeline versions:")
    print(summary[[c for c in cols if c in summary.columns]].sort_values(["label", "signal"]).to_string(index=False))
    recall_cols = [f"recall_{g}" for g in C.GESTURES]
    if all(c in summary.columns for c in recall_cols):
        print("\nper-class recall (mean over folds):")
        tab = summary.dropna(subset=recall_cols)[["label", "signal"] + recall_cols].copy()
        tab.columns = ["label", "signal"] + C.GESTURES
        print(tab.sort_values(["label", "signal"]).round(3).to_string(index=False))
        plots.plot_cv_per_class(tab, out_dir / "cv_per_class.png")
    rows = [dict(label=r.label, signal=r.signal, fold_accs=[float(a) for a in str(r.fold_accs).split(";")],
                 mean_acc=r.mean_test_acc, adabn_acc=getattr(r, "mean_test_acc_adabn", None))
            for r in summary.itertuples()]
    plots.plot_cv_results(rows, out_dir / "cv_results.png")
    print(f"saved {summary_path}, cv_results.png and cv_per_class.png")


if __name__ == "__main__":
    main()

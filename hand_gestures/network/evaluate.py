"""Evaluate the trained ensemble on the held-out TEST participants.

    python evaluate.py --signal raw
    python evaluate.py --signal filtered

Loads every best_model_seed*.pt of the experiment (each = the epoch with the lowest validation loss
of one seed, chosen in train.py without looking at the test set) and averages their softmax outputs.
This script is the only place where test data is used.

Writes to results/<signal>/:
    metrics.json                overall + per-class metrics, confused pairs, per-participant and per-position errors,
                                single-member accuracies, calibration (AdaBN), smoothing and merged-class views
    classification_report.txt   scikit-learn report
    confusion_matrix.csv        counts, rows = true gesture, columns = predicted gesture
    predictions.csv             one row per test window with all 10 class probabilities
    failure_cases.csv           the plotted failure cases
    curve_loss.png, curve_accuracy.png, confusion_matrix.png, confusion_matrix_normalized.png, failure_cases.png
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from network import config as C
from network import plots
from network.data_loader import SubjectSplit, load_recording, prepare_data, window_start_sample
from network.model import load_ensemble, predict_proba, predict_proba_adabn
from network.train import get_device, set_seed


def smooth_probs(probs, meta, n=C.SMOOTH_WINDOWS):
    """Causal moving average: each window's probabilities are averaged with those of the previous
    n-1 windows of the same segment (same participant, cycle and gesture)."""
    out = probs.copy()
    df = pd.DataFrame({"subject": meta[:, 0], "gesture": meta[:, 1], "cycle": meta[:, 2],
                       "position": meta[:, 3], "i": np.arange(len(meta))})
    for _, g in df.groupby(["subject", "cycle", "gesture"]):
        idx = g.sort_values("position")["i"].to_numpy()
        for j in range(len(idx)):
            out[idx[j]] = probs[idx[max(0, j - n + 1):j + 1]].mean(axis=0)
    return out


def merged_class_map():
    """gesture index -> merged group index (pairs the sensors keep confusing become one class)."""
    groups = list(dict.fromkeys(C.MERGED_CLASS_GROUPS.get(g, g) for g in C.GESTURES))
    return np.array([groups.index(C.MERGED_CLASS_GROUPS.get(g, g)) for g in C.GESTURES]), groups


def raw_clipped_fraction(subject, gesture, cycle, position):
    """Per channel: fraction of samples of this window at which the RAW recording sits at the +-10 mV ADC limit."""
    start = window_start_sample(cycle, gesture, position)
    raw = load_recording(subject, "raw")[start:start + C.WINDOW_SAMPLES]
    return (np.abs(raw) >= 9.999).mean(axis=0)


def select_failure_cases(y, pred, probs, meta, X_mv, n_cases):
    """For the most frequent confusions pick the most confident mistake and a correct window to compare with.

    The comparison window has the same true gesture and, if one exists, the same participant
    (preferring the same cycle and position inside the segment); otherwise the most confident
    correct window of that gesture from another test participant.
    """
    cm = confusion_matrix(y, pred, labels=range(C.N_CLASSES))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], i, j) for i in range(C.N_CLASSES) for j in range(C.N_CLASSES) if i != j), reverse=True)
    cases = []
    for count, t, p in pairs[:n_cases]:
        if count == 0:
            break
        idx = np.where((y == t) & (pred == p))[0]
        i = idx[np.argmax(probs[idx, p])]                       # the most confident mistake of this kind
        subject, _, cycle, position = meta[i]
        n_same = int(((y == t) & (meta[:, 0] == subject)).sum())
        n_same_correct = int(((y == t) & (pred == t) & (meta[:, 0] == subject)).sum())
        case = dict(x_wrong=X_mv[i], true=int(t), pred=int(p), probs_wrong=probs[i], subject=int(subject),
                    cycle=int(cycle), position=int(position), index=int(i),
                    subject_correct_of_this_gesture=f"{n_same_correct}/{n_same}",
                    raw_clipped_fraction=raw_clipped_fraction(int(subject), int(t), int(cycle), int(position)),
                    x_right=None, probs_right=None, subject_right=None, cycle_right=None, position_right=None, index_right=None)
        correct = np.where((y == t) & (pred == t) & (meta[:, 0] == subject))[0]
        if len(correct):                                        # same participant: prefer same cycle, then same position
            j = correct[np.argmin((meta[correct, 2] != cycle) * 100 + np.abs(meta[correct, 3] - position))]
        else:                                                   # fall back to another test participant
            correct = np.where((y == t) & (pred == t))[0]
            j = correct[np.argmax(probs[correct, t])] if len(correct) else None
        if j is not None:
            case.update(x_right=X_mv[j], probs_right=probs[j], subject_right=int(meta[j, 0]), cycle_right=int(meta[j, 2]),
                        position_right=int(meta[j, 3]), index_right=int(j))
        cases.append(case)
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", required=True, choices=C.SIGNAL_TYPES)
    ap.add_argument("--n-failure-cases", type=int, default=4)
    args = ap.parse_args()
    set_seed(C.SEED)
    device = get_device()
    out_dir = C.results_dir(args.signal)

    models, ckpts = load_ensemble(out_dir, device)
    ckpt = ckpts[0]
    split = SubjectSplit(**ckpt["split"])
    normalization = ckpt.get("normalization", "train_std")
    data = prepare_data(args.signal, split=split, normalization=normalization)
    assert np.allclose(data["channel_std"], ckpt["channel_std"]), "normalisation statistics differ from training"
    seeds = [c["seed"] for c in ckpts]
    print(f"\ndevice: {device}; ensemble of {len(models)} (seeds {seeds}), checkpoint epochs "
          f"{[c['epoch'] for c in ckpts]}, member val acc {[round(c['val_acc'], 4) for c in ckpts]}; normalisation {normalization}")

    # ---- predictions
    X_val, y_val, _ = data["val"]
    val_acc = accuracy_score(y_val, predict_proba(models, X_val, device).argmax(1))
    X_te, y_te, m_te = data["test"]
    probs = predict_proba(models, X_te, device)
    pred = probs.argmax(axis=1)
    member_acc = [float(accuracy_score(y_te, predict_proba(m, X_te, device).argmax(1))) for m in models]

    # ---- overall and per-class metrics
    acc = accuracy_score(y_te, pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    p_cls, r_cls, f1_cls, support = precision_recall_fscore_support(y_te, pred, labels=range(C.N_CLASSES), zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=range(C.N_CLASSES))
    report = classification_report(y_te, pred, target_names=C.GESTURES, digits=3, zero_division=0)
    print(f"\nTEST participants {split.test}: ensemble accuracy {acc:.4f}, macro precision {p_macro:.4f}, "
          f"macro recall {r_macro:.4f}, macro F1 {f1_macro:.4f}   (ensemble validation accuracy {val_acc:.4f})")
    print(f"single members: test accuracy {[round(a, 4) for a in member_acc]} (mean {np.mean(member_acc):.4f})\n")
    print(report)

    per_class = pd.DataFrame({"gesture": C.GESTURES, "precision": p_cls, "recall": r_cls, "f1": f1_cls,
                              "support": support}).sort_values("f1", ascending=False)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], i, j) for i in range(C.N_CLASSES) for j in range(C.N_CLASSES) if i != j), reverse=True)
    confused = [{"true": C.GESTURES[i], "predicted": C.GESTURES[j], "count": int(n),
                 "pct_of_true_class": round(float(n / cm[i].sum() * 100), 1)} for n, i, j in pairs[:8] if n > 0]
    print("per-class F1, best to worst:")
    for _, r in per_class.iterrows():
        print(f"  {r['gesture']:11s} precision {r['precision']:.3f}  recall {r['recall']:.3f}  F1 {r['f1']:.3f}")
    print("most confused pairs (true -> predicted):")
    for c in confused:
        print(f"  {c['true']:11s} -> {c['predicted']:11s} {c['count']:4d} windows ({c['pct_of_true_class']:.1f} % of that class)")

    # ---- where does it fail: per participant, per position in segment, per cycle, segment-level vote
    acc_by_subject = {int(s): round(float(accuracy_score(y_te[m_te[:, 0] == s], pred[m_te[:, 0] == s])), 4) for s in split.test}
    n_pos = int(m_te[:, 3].max()) + 1
    err_by_position = [round(float((pred[m_te[:, 3] == p] != y_te[m_te[:, 3] == p]).mean()), 4) for p in range(n_pos)]
    err_by_cycle = [round(float((pred[m_te[:, 2] == c] != y_te[m_te[:, 2] == c]).mean()), 4) for c in range(C.N_CYCLES)]
    seg = pd.DataFrame({"subject": m_te[:, 0], "cycle": m_te[:, 2], "true": y_te, "pred": pred})
    seg_vote = seg.groupby(["subject", "cycle", "true"])["pred"].agg(lambda s: s.value_counts().idxmax()).reset_index()
    seg_acc = float((seg_vote["pred"] == seg_vote["true"]).mean())
    print(f"\naccuracy per test participant: {acc_by_subject}")
    print(f"error rate by window position in segment (position 0 starts {C.SEGMENT_START_OFFSET_S:.1f} s after the cue): {err_by_position}")
    print(f"error rate by cycle (1..5): {err_by_cycle}")
    print(f"segment-level accuracy (majority vote over the {n_pos} windows of a 6 s segment): {seg_acc:.4f}")

    # ---- extra views: temporal smoothing, merged classes, test-time calibration (AdaBN)
    pred_smooth = smooth_probs(probs, m_te).argmax(1)
    acc_smooth = float(accuracy_score(y_te, pred_smooth))
    gmap, groups = merged_class_map()
    acc_merged = float((gmap[y_te] == gmap[pred]).mean())
    probs_adabn = predict_proba_adabn(models, X_te, m_te, device)
    pred_adabn = probs_adabn.argmax(1)
    acc_adabn = float(accuracy_score(y_te, pred_adabn))
    f1_adabn = float(f1_score(y_te, pred_adabn, average="macro"))
    acc_adabn_by_subject = {int(s): round(float(accuracy_score(y_te[m_te[:, 0] == s], pred_adabn[m_te[:, 0] == s])), 4) for s in split.test}
    print(f"accuracy with causal probability averaging over {C.SMOOTH_WINDOWS} windows: {acc_smooth:.4f}")
    print(f"accuracy with merged classes ({len(groups)} groups: {groups}): {acc_merged:.4f}")
    print(f"CALIBRATION SETTING - per-participant AdaBN (BatchNorm statistics re-estimated on each test participant's "
          f"unlabelled windows): accuracy {acc_adabn:.4f}, macro F1 {f1_adabn:.4f}, per participant {acc_adabn_by_subject}")

    metrics = {
        "signal_type": args.signal, "normalization": normalization, "ensemble_size": len(models), "seeds": seeds,
        "checkpoint_epochs": [int(c["epoch"]) for c in ckpts], "member_val_acc": [float(c["val_acc"]) for c in ckpts],
        "val_acc": float(val_acc), "member_test_acc": member_acc, "mean_member_test_acc": float(np.mean(member_acc)),
        "test_acc": float(acc), "test_macro_precision": float(p_macro), "test_macro_recall": float(r_macro),
        "test_macro_f1": float(f1_macro), "test_segment_vote_acc": seg_acc,
        "test_acc_smoothed": acc_smooth, "smooth_windows": C.SMOOTH_WINDOWS,
        "test_acc_merged_classes": acc_merged, "merged_groups": groups,
        "test_acc_adabn": acc_adabn, "test_macro_f1_adabn": f1_adabn, "acc_adabn_by_test_subject": acc_adabn_by_subject,
        "test_subjects": split.test, "n_test_windows": int(len(y_te)),
        "per_class": per_class.round(4).to_dict(orient="records"),
        "most_confused_pairs": confused,
        "acc_by_test_subject": acc_by_subject,
        "error_by_window_position": err_by_position,
        "error_by_cycle": err_by_cycle,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "classification_report.txt").write_text(report)
    pd.DataFrame(cm, index=C.GESTURES, columns=C.GESTURES).to_csv(out_dir / "confusion_matrix.csv")

    pred_df = pd.DataFrame({"subject": m_te[:, 0], "cycle": m_te[:, 2], "position": m_te[:, 3],
                            "true": y_te, "pred": pred, "true_name": [C.GESTURES[k] for k in y_te],
                            "pred_name": [C.GESTURES[k] for k in pred], "correct": pred == y_te,
                            "pred_adabn": pred_adabn, "pred_smoothed": pred_smooth,
                            "p_pred": probs[np.arange(len(pred)), pred], "p_true": probs[np.arange(len(pred)), y_te]})
    for k, g in enumerate(C.GESTURES):
        pred_df[f"p_{g}"] = probs[:, k]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    # ---- figures
    histories = [pd.read_csv(out_dir / f"history_seed{s}.csv") for s in seeds]
    plots.plot_training_curves(histories, args.signal, out_dir / "curve_loss.png", out_dir / "curve_accuracy.png",
                               best_epochs=[int(c["epoch"]) for c in ckpts])
    plots.plot_confusion_matrix(cm, out_dir / "confusion_matrix.png",
                                title=f"Confusion matrix, {args.signal} EMG, test participants (accuracy {acc:.3f})")
    plots.plot_confusion_matrix(cm, out_dir / "confusion_matrix_normalized.png", normalize=True,
                                title=f"Normalised confusion matrix, {args.signal} EMG, test participants")

    # ---- failure cases (signals shown in mV: undo the scaling; the window mean stays removed)
    if normalization == "train_std":
        X_mv = X_te * data["channel_std"][None, :, None]
    else:                                                   # rest_rms: scale back with each window's own factor
        _, _, _, rest = build_rest_scale_for(args.signal, m_te)
        X_mv = X_te * rest[:, :, None]
    cases = select_failure_cases(y_te, pred, probs, m_te, X_mv, args.n_failure_cases)
    plots.plot_failure_cases(cases, args.signal, out_dir / "failure_cases.png")
    rows = []
    for c in cases:
        rows.append({"true": C.GESTURES[c["true"]], "predicted": C.GESTURES[c["pred"]], "subject": c["subject"],
                     "cycle": c["cycle"] + 1, "position": c["position"] + 1, "p_predicted": round(float(c["probs_wrong"][c["pred"]]), 3),
                     "p_true": round(float(c["probs_wrong"][c["true"]]), 3), "test_window_index": c["index"],
                     "windows_of_this_gesture_correct_for_this_subject": c["subject_correct_of_this_gesture"],
                     **{f"raw_clipped_fraction_CH{k + 1}": round(float(v), 3) for k, v in enumerate(c["raw_clipped_fraction"])},
                     "compared_with_subject": c["subject_right"], "compared_with_correct_window_index": c["index_right"],
                     **{f"p_{g}": round(float(c["probs_wrong"][k]), 3) for k, g in enumerate(C.GESTURES)}})
    pd.DataFrame(rows).to_csv(out_dir / "failure_cases.csv", index=False)
    print("\nfailure cases plotted:")
    for r in rows:
        clipped = ", ".join(f"CH{k + 1} {r[f'raw_clipped_fraction_CH{k + 1}']:.0%}" for k in range(C.N_CHANNELS)
                            if r[f"raw_clipped_fraction_CH{k + 1}"] > 0) or "none"
        print(f"  true {r['true']:11s} predicted {r['predicted']:11s} (p={r['p_predicted']:.2f}, p_true={r['p_true']:.2f})  "
              f"participant {r['subject']}, cycle {r['cycle']}, window {r['position']};  this participant's "
              f"{r['true']} windows correct: {r['windows_of_this_gesture_correct_for_this_subject']};  "
              f"raw ADC saturation in this window: {clipped}")
    print(f"\nsaved metrics, report, confusion matrices, predictions, failure cases and figures to {out_dir}")


def build_rest_scale_for(signal_type, meta):
    """Rest-RMS scale factors for the windows in meta (only needed to plot rest_rms-normalised data in mV)."""
    from network.data_loader import rest_rms
    scale = np.zeros((len(meta), C.N_CHANNELS), dtype=np.float32)
    for s in np.unique(meta[:, 0]):
        scale[meta[:, 0] == s] = rest_rms(load_recording(int(s), signal_type))
    return None, None, None, scale


if __name__ == "__main__":
    main()

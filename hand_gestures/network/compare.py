"""Collect the raw and filtered results into one table and the comparison figures.

    python compare.py        (run after train.py + evaluate.py for both signals; baseline.py is optional)

Writes results/comparison.md, results/raw_vs_filtered.png, results/error_by_position.png,
results/accuracy_per_subject.png
"""
import json

from network import config as C
from network import plots


def main():
    metrics = {sig: json.loads((C.RESULTS_DIR / sig / "metrics.json").read_text()) for sig in C.SIGNAL_TYPES}
    baseline_path = C.RESULTS_DIR / "baseline_results.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None

    rows = [("validation accuracy (ensemble)", "val_acc"),
            ("test accuracy, mean of the single networks", "mean_member_test_acc"),
            ("test accuracy (ensemble)", "test_acc"),
            ("test macro precision (ensemble)", "test_macro_precision"), ("test macro recall (ensemble)", "test_macro_recall"),
            ("test macro F1 (ensemble)", "test_macro_f1"),
            ("test accuracy, probabilities averaged over the last 3 windows", "test_acc_smoothed"),
            ("test accuracy, majority vote over a whole 6 s segment", "test_segment_vote_acc"),
            ("test accuracy, merged classes (finger pair, rotation pair)", "test_acc_merged_classes"),
            ("CALIBRATION: test accuracy with per-participant AdaBN", "test_acc_adabn"),
            ("CALIBRATION: test macro F1 with per-participant AdaBN", "test_macro_f1_adabn")]
    lines = ["| metric | raw EMG | filtered EMG |", "|---|---|---|"]
    for label, key in rows:
        if key in metrics["raw"] and key in metrics["filtered"]:
            lines.append(f"| {label} | {metrics['raw'][key]:.3f} | {metrics['filtered'][key]:.3f} |")
    if baseline:
        for name in ("logistic_regression", "random_forest"):
            for k, lab in (("test_acc", "test accuracy"), ("test_macro_f1", "test macro F1")):
                lines.append(f"| {lab} ({name.replace('_', ' ')} baseline) | "
                             f"{baseline['raw'][name][k]:.3f} | {baseline['filtered'][name][k]:.3f} |")
    lines += ["", f"ensemble of {metrics['raw'].get('ensemble_size', 1)}; checkpoint epochs: raw {metrics['raw'].get('checkpoint_epochs')}, "
              f"filtered {metrics['filtered'].get('checkpoint_epochs')}",
              f"test participants: {metrics['raw']['test_subjects']}", ""]
    for sig in C.SIGNAL_TYPES:
        lines.append(f"most confused pairs ({sig}): " + "; ".join(
            f"{c['true']} -> {c['predicted']} ({c['count']})" for c in metrics[sig]["most_confused_pairs"][:4]))
    text = "\n".join(lines)
    print(text)
    (C.RESULTS_DIR / "comparison.md").write_text(text)

    summary = {sig: {k: metrics[sig][k] for k in ("val_acc", "test_acc", "test_macro_f1")} for sig in C.SIGNAL_TYPES}
    base = None
    if baseline:
        base = {sig: {"val_acc": baseline[sig]["logistic_regression"]["val_acc"],
                      "test_acc": baseline[sig]["logistic_regression"]["test_acc"],
                      "test_macro_f1": baseline[sig]["logistic_regression"]["test_macro_f1"]} for sig in C.SIGNAL_TYPES}
    plots.plot_raw_vs_filtered(summary, C.RESULTS_DIR / "raw_vs_filtered.png", baseline=base)
    plots.plot_error_by_position({sig: metrics[sig]["error_by_window_position"] for sig in C.SIGNAL_TYPES},
                                 C.RESULTS_DIR / "error_by_position.png")
    plots.plot_per_subject_accuracy({sig: {int(k): v for k, v in metrics[sig]["acc_by_test_subject"].items()}
                                     for sig in C.SIGNAL_TYPES}, C.RESULTS_DIR / "accuracy_per_subject.png")
    print(f"\nsaved comparison.md, raw_vs_filtered.png, error_by_position.png, accuracy_per_subject.png in {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()

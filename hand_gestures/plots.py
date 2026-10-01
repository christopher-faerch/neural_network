"""Presentation-ready plots. Every function saves a PNG.

Colour roles (fixed, colour-vision-safe order, identity never by colour alone):
    EMG channels CH1..CH4 -> blue, orange, aqua, yellow (always labelled on the axis)
    raw -> orange, filtered -> blue, baseline -> grey
    train -> blue, validation -> orange
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C

CHANNEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SIGNAL_COLORS = {"raw": "#eb6834", "filtered": "#2a78d6"}
BASELINE_COLOR = "#8a8984"
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"
WRONG_COLOR, RIGHT_COLOR = "#d03b3b", "#0ca30c"
CM_CMAP = "Blues"
DPI = 150


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "lines.linewidth": 1.0,
    })


setup_style()


def pretty(name: str) -> str:
    return name.replace("_dev", " deviation").replace("_", " ")


def _save(fig, out):
    fig.savefig(out)
    plt.close(fig)
    return out


def _stacked_channels(ax, x_mv, spread=None):
    """Plot the 4 channels of one window stacked vertically; returns the offset used."""
    t = np.arange(x_mv.shape[1]) / C.FS
    if spread is None:
        spread = max(np.percentile(np.abs(x_mv), 99.5) * 2.2, 1e-6)
    for ch in range(C.N_CHANNELS):
        ax.plot(t, x_mv[ch] + (C.N_CHANNELS - 1 - ch) * spread, color=CHANNEL_COLORS[ch], linewidth=0.6)
    ax.set_yticks([(C.N_CHANNELS - 1 - ch) * spread for ch in range(C.N_CHANNELS)])
    ax.set_yticklabels([f"CH{ch + 1}" for ch in range(C.N_CHANNELS)])
    ax.grid(False)
    return spread


# --------------------------------------------------------------------------- 1. example recording
def plot_example_recording(recording_mv, subject, signal_type, out, t_start_s=0, t_end_s=C.CYCLE_PERIOD_S):
    """All 4 channels of one participant with the labelled gesture segments shaded (default: first cycle)."""
    a, b = int(t_start_s * C.FS), int(t_end_s * C.FS)
    t = np.arange(a, b) / C.FS
    fig, axes = plt.subplots(C.N_CHANNELS, 1, figsize=(14, 8), sharex=True)
    for ch, ax in enumerate(axes):
        ax.plot(t, recording_mv[a:b, ch], color=CHANNEL_COLORS[ch], linewidth=0.5)
        ax.set_ylabel(f"CH{ch + 1}\n[mV]")
        ymax = ax.get_ylim()[1]
        for cycle in range(C.N_CYCLES):
            for g in range(C.N_CLASSES):
                s0 = cycle * C.CYCLE_PERIOD_S + C.FIRST_GESTURE_OFFSET_S + g * C.GESTURE_PERIOD_S
                if s0 + C.GESTURE_DURATION_S <= t_start_s or s0 >= t_end_s:
                    continue
                ax.axvspan(s0, s0 + C.GESTURE_DURATION_S, color="#000000", alpha=0.06, linewidth=0)
                if ch == 0:
                    ax.text(s0 + C.GESTURE_DURATION_S / 2, ymax, pretty(C.GESTURES[g]),
                            ha="center", va="bottom", fontsize=8, rotation=35)
    axes[-1].set_xlabel("time [s]")
    axes[-1].set_xlim(t_start_s, t_end_s)
    fig.suptitle(f"{signal_type} sEMG, participant {subject}, {t_start_s:.0f}-{t_end_s:.0f} s "
                 f"(shaded = labelled 6 s gesture segments)", y=1.02)
    return _save(fig, out)


# --------------------------------------------------------------------------- 2. example windows
def plot_example_windows(windows_mv, labels, subject, signal_type, out):
    """One panel per gesture: a 1 s window with the 4 channels stacked (same y-scale everywhere)."""
    spread = max(np.percentile(np.abs(windows_mv), 99.5) * 2.2, 1e-6)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), sharex=True, sharey=True)
    for ax, x, g in zip(axes.ravel(), windows_mv, labels):
        _stacked_channels(ax, x, spread)
        ax.set_title(pretty(C.GESTURES[g]))
    for ax in axes[1]:
        ax.set_xlabel("time [s]")
    fig.suptitle(f"Example {C.WINDOW_S:.0f} s windows per gesture ({signal_type} EMG, participant {subject}; "
                 f"channel offset {spread:.2f} mV, window mean removed)", y=1.01)
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------- 3./4. training curves
def plot_training_curves(histories, signal_type, out_loss, out_acc, best_epochs=None):
    """Two figures: train vs validation loss, and train vs validation accuracy.

    `histories` is one DataFrame or a list of them (one per ensemble member / seed). The first member is
    drawn bold, the others thin, so the run-to-run variation is visible. `best_epochs` marks the
    selected epoch (lowest validation loss) of every member on its validation curve.
    """
    if not isinstance(histories, (list, tuple)):
        histories = [histories]
    best_epochs = list(best_epochs) if best_epochs is not None else [None] * len(histories)
    for key, ylabel, out in (("loss", "cross-entropy loss", out_loss), ("acc", "accuracy", out_acc)):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for i, (h, be) in enumerate(zip(histories, best_epochs)):
            ep = h["epoch"].to_numpy()
            lw, alpha = (2.2, 1.0) if i == 0 else (1.0, 0.45)
            ax.plot(ep, h[f"train_{key}"], color=TRAIN_COLOR, linewidth=lw, alpha=alpha,
                    label="train" + (f" ({len(histories)} seeds)" if len(histories) > 1 else "") if i == 0 else None)
            ax.plot(ep, h[f"val_{key}"], color=VAL_COLOR, linewidth=lw, alpha=alpha,
                    label="validation" + (f" ({len(histories)} seeds)" if len(histories) > 1 else "") if i == 0 else None)
            if be is not None:
                ax.scatter([be], [h[f"val_{key}"].iloc[be - 1]], s=36, color=VAL_COLOR, edgecolor="white", zorder=4,
                           label="selected epoch (lowest val loss)" if i == 0 else None)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        if key == "acc":
            ax.set_ylim(0, 1.02)
            ax.axhline(1 / C.N_CLASSES, color="#c3c2b7", linestyle=":", linewidth=1)
            ax.text(ax.get_xlim()[1], 1 / C.N_CLASSES, "chance", ha="right", va="bottom", fontsize=9, color="#8a8984")
        ax.set_title(f"{ylabel} per epoch ({signal_type} EMG" + (f", bold = seed {C.SEED})" if len(histories) > 1 else ")"))
        ax.legend(frameon=False)
        _save(fig, out)
    return out_loss, out_acc


def plot_cv_per_class(table, out):
    """Per-class recall (mean over folds) per pipeline version. `table` columns: label, signal, <gesture names>."""
    labels = sorted(dict.fromkeys(table["label"]))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    x = np.arange(C.N_CLASSES)
    w = 0.8 / max(1, len(labels))
    shades = np.linspace(0.45, 1.0, len(labels))
    for ax, sig in zip(axes, ("raw", "filtered")):
        base = np.array(matplotlib.colors.to_rgb(SIGNAL_COLORS[sig]))
        for i, lab in enumerate(labels):
            row = table[(table["label"] == lab) & (table["signal"] == sig)]
            if row.empty:
                continue
            vals = row[C.GESTURES].to_numpy().ravel()
            color = 1 - (1 - base) * shades[i]                      # lighter shades for older versions
            ax.bar(x + (i - (len(labels) - 1) / 2) * w, vals, width=w * 0.92, color=color, label=lab)
        ax.set_ylabel(f"{sig} EMG\nrecall (mean over folds)")
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, ncol=min(4, len(labels)), fontsize=9, loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([pretty(g) for g in C.GESTURES], rotation=30, ha="right")
    fig.suptitle("Per-class recall across the cross-validation folds, by pipeline version", y=0.995)
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------- 5. confusion matrix
def plot_confusion_matrix(cm, out, title="", normalize=False):
    """Counts (normalize=False) or row-normalised percentages (normalize=True)."""
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True) * 100 if normalize else cm
    vmax = 100 if normalize else shown.max()
    names = [pretty(g) for g in C.GESTURES]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(shown, cmap=CM_CMAP, vmin=0, vmax=vmax)
    ax.set_xticks(range(C.N_CLASSES))
    ax.set_yticks(range(C.N_CLASSES))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted gesture")
    ax.set_ylabel("true gesture")
    ax.grid(False)
    for i in range(C.N_CLASSES):
        for j in range(C.N_CLASSES):
            v = shown[i, j]
            ax.text(j, i, f"{v:.0f}" if normalize else f"{int(v)}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.55 * vmax else "#0b0b0b")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("% of true class" if normalize else "number of test windows")
    ax.set_title(title)
    return _save(fig, out)


# --------------------------------------------------------------------------- 6. raw vs filtered
def plot_raw_vs_filtered(summary, out, baseline=None):
    """Grouped bars: validation accuracy, test accuracy and test macro F1 for raw vs filtered.

    summary  = {"raw": {"val_acc", "test_acc", "test_macro_f1"}, "filtered": {...}}
    baseline = optional {"raw": {"val_acc", "test_acc", "test_macro_f1"}, "filtered": {...}} drawn as hollow markers
    """
    metrics = [("val_acc", "validation\naccuracy"), ("test_acc", "test\naccuracy"), ("test_macro_f1", "test\nmacro F1")]
    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, sig in enumerate(("raw", "filtered")):
        vals = [summary[sig][m] for m, _ in metrics]
        bvals = [baseline[sig].get(m, np.nan) for m, _ in metrics] if baseline is not None and sig in baseline else [np.nan] * len(metrics)
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w - 0.04, color=SIGNAL_COLORS[sig], label=f"CNN, {sig} EMG")
        for b, v, bv in zip(bars, vals, bvals):
            label_y = v + 0.012
            if np.isfinite(bv) and abs(bv - v) < 0.06:           # keep the value label clear of the baseline marker
                label_y = max(v, bv) + 0.03
            ax.text(b.get_x() + b.get_width() / 2, label_y, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        if np.isfinite(bvals).any():
            ax.scatter(x + (i - 0.5) * w, bvals, marker="D", s=46, facecolor="white",
                       edgecolor=BASELINE_COLOR, linewidth=1.6, zorder=3,
                       label="feature baseline (logistic regression)" if i == 0 else None)
    ax.axhline(1 / C.N_CLASSES, color="#c3c2b7", linestyle=":", linewidth=1)
    ax.text(x[-1] + w, 1 / C.N_CLASSES, "chance", ha="right", va="bottom", fontsize=9, color="#8a8984")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Raw vs filtered EMG: same model, split and training procedure")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, out)


# --------------------------------------------------------------------------- 7. failure cases
def _plot_probs(ax, probs, true_label, pred_label):
    colors = ["#c3c2b7"] * C.N_CLASSES
    colors[true_label] = RIGHT_COLOR
    if pred_label != true_label:
        colors[pred_label] = WRONG_COLOR
    ax.barh(range(C.N_CLASSES), probs, color=colors)
    ax.set_yticks(range(C.N_CLASSES))
    ax.set_yticklabels([pretty(g) for g in C.GESTURES], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("softmax probability")
    ax.grid(axis="y", visible=False)


def plot_failure_cases(cases, signal_type, out):
    """One row per case: misclassified window | correct window of the same gesture and participant | probabilities.

    Each case is a dict with keys: x_wrong, x_right (mV arrays [4, T]; x_right may be None), true, pred,
    probs_wrong, probs_right, subject, cycle, position, cycle_right, position_right.
    """
    n = len(cases)
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.5 * n), gridspec_kw={"width_ratios": [3, 3, 1.6]})
    axes = np.atleast_2d(axes)
    for row, c in zip(axes, cases):
        spread = _stacked_channels(row[0], c["x_wrong"])
        row[0].set_title(f"MISCLASSIFIED   true: {pretty(C.GESTURES[c['true']])}  ->  predicted: "
                         f"{pretty(C.GESTURES[c['pred']])} (p = {c['probs_wrong'][c['pred']]:.2f})\n"
                         f"participant {c['subject']}, cycle {c['cycle'] + 1}, window {c['position'] + 1}/{C.WINDOWS_PER_SEGMENT}",
                         color=WRONG_COLOR, fontsize=10, loc="left")
        row[0].set_xlabel("time [s]")
        if c.get("x_right") is not None:
            same_participant = c.get("subject_right") in (None, c["subject"])
            if same_participant:                                  # same offset and y-scale as the failure next to it
                _stacked_channels(row[1], c["x_right"], spread)
                row[1].set_ylim(row[0].get_ylim())
                where = f"participant {c['subject']}, cycle {c['cycle_right'] + 1}, window {c['position_right'] + 1}/{C.WINDOWS_PER_SEGMENT}"
            else:                                                 # another participant: own scale, say so
                _stacked_channels(row[1], c["x_right"])
                where = (f"participant {c['subject_right']}, cycle {c['cycle_right'] + 1}, window {c['position_right'] + 1}/{C.WINDOWS_PER_SEGMENT} "
                         f"(own scale; P{c['subject']} had no correct {pretty(C.GESTURES[c['true']])} window)")
            row[1].set_title(f"CORRECT   true: {pretty(C.GESTURES[c['true']])} (p = {c['probs_right'][c['true']]:.2f})\n{where}",
                             color=RIGHT_COLOR, fontsize=10, loc="left")
            row[1].set_xlabel("time [s]")
        else:
            row[1].axis("off")
            row[1].set_title("no correctly classified window of this gesture in the test set", fontsize=9, loc="left")
        _plot_probs(row[2], c["probs_wrong"], c["true"], c["pred"])
        row[2].set_title("probabilities of the misclassified window", fontsize=9, loc="left")
    fig.suptitle(f"Failure cases on the {signal_type} EMG test set "
                 f"(mV, window mean removed; left and middle panels share the y-scale when the participant is the same)", y=1.0)
    fig.tight_layout()
    return _save(fig, out)


def plot_error_by_position(err_by_pos, out):
    """Error rate vs window position inside the 6 s segment; err_by_pos = {"raw": [...], "filtered": [...]}."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    pos = np.arange(len(next(iter(err_by_pos.values()))))
    off = C.SEGMENT_START_OFFSET_S
    for sig, err in err_by_pos.items():
        ax.plot(pos, err, marker="o", markersize=5, linewidth=2, color=SIGNAL_COLORS[sig], label=f"{sig} EMG")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{off + p * C.HOP_S:.1f}-{off + p * C.HOP_S + C.WINDOW_S:.1f}" for p in pos], rotation=45, fontsize=8)
    ax.set_xlabel("window position inside the 6 s gesture segment [s after the cue]")
    ax.set_ylabel("test error rate")
    ax.set_ylim(0, max(0.05, 1.15 * max(max(e) for e in err_by_pos.values())))
    ax.set_title("Where inside a gesture segment does the model fail?")
    ax.legend(frameon=False)
    return _save(fig, out)


def plot_cv_results(rows, out):
    """Cross-validation summary. rows: list of dicts with label, signal, fold_accs (list), mean_acc and
    optionally adabn_acc (mean accuracy in the calibration setting).

    One group per pipeline label (sorted by name); per signal the fold accuracies as small markers and
    the strict mean with a +-1 std bar; the calibrated mean, where available, as a hollow marker.
    """
    labels = sorted(dict.fromkeys(r["label"] for r in rows))
    x = np.arange(len(labels))
    w = 0.32
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(labels) + 2), 5.2))
    for i, sig in enumerate(("raw", "filtered")):
        xs = x + (i - 0.5) * w
        for xi, lab in zip(xs, labels):
            r = next((r for r in rows if r["label"] == lab and r["signal"] == sig), None)
            if r is None:
                continue
            accs = np.asarray(r["fold_accs"], dtype=float)
            jitter = np.linspace(-0.07, 0.07, len(accs))
            ax.scatter(xi + jitter, accs, s=22, color=SIGNAL_COLORS[sig], alpha=0.55, zorder=2)
            ax.errorbar(xi, accs.mean(), yerr=accs.std(), fmt="D", markersize=7, color=SIGNAL_COLORS[sig],
                        markeredgecolor="white", capsize=4, linewidth=1.6, zorder=3,
                        label=f"{sig} EMG, strict (mean ± std over folds)" if lab == labels[0] else None)
            ax.text(xi, accs.min() - 0.025, f"{accs.mean():.3f}", ha="center", va="top", fontsize=8.5)
            adabn = r.get("adabn_acc")
            if adabn is not None and np.isfinite(adabn):
                ax.scatter([xi], [adabn], marker="D", s=52, facecolor="white", edgecolor=SIGNAL_COLORS[sig],
                           linewidth=1.8, zorder=4,
                           label=f"{sig} EMG, calibrated (AdaBN), mean" if lab == next(l for l in labels if any(
                               rr["label"] == l and rr["signal"] == sig and rr.get("adabn_acc") is not None
                               and np.isfinite(rr.get("adabn_acc")) for rr in rows)) else None)
                ax.text(xi, adabn + 0.02, f"{adabn:.3f}", ha="center", va="bottom", fontsize=8.5, color="#52514e")
    ax.axhline(1 / C.N_CLASSES, color="#c3c2b7", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("test accuracy per fold")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Leave-participants-out cross-validation ({C.CV_N_FOLDS} folds x 8 test participants); "
                 f"'_ens' = 5-seed ensembles")
    ax.legend(frameon=False, loc="lower left", fontsize=9, ncol=2)
    return _save(fig, out)


def plot_per_subject_accuracy(acc_by_subject, out):
    """acc_by_subject = {"raw": {subject: acc}, "filtered": {subject: acc}} for the test participants."""
    subjects = sorted(next(iter(acc_by_subject.values())).keys())
    x = np.arange(len(subjects))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (sig, d) in enumerate(acc_by_subject.items()):
        vals = [d[s] for s in subjects]
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w - 0.04, color=SIGNAL_COLORS[sig], label=f"{sig} EMG")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{s}" for s in subjects])
    ax.set_xlabel("test participant")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("Test accuracy per participant")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    return _save(fig, out)

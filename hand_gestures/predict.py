"""Ask the trained ensemble which gesture it sees in a piece of EMG.

    # one labelled window: participant 2, cycle 1 (0-4), gesture 3 (ulnar deviation), window position 4
    python predict.py --signal filtered --subject 2 --cycle 1 --gesture 3 --position 4

    # one 1 s window starting at 14.5 s of participant 2's recording
    python predict.py --signal filtered --subject 2 --start 14.5

    # slide the 1 s window over 0-104 s (first cycle) in 0.5 s steps and print a prediction per step;
    # --smooth 3 averages the probabilities of the last 3 windows before deciding (2 s of evidence)
    python predict.py --signal filtered --subject 2 --start 0 --end 104 --smooth 3 --plot

    # add --plot to save a figure of the window(s) with the probability bars (results/<signal>/predict_*.png)

The window is normalised exactly as in training (window mean removed, divided by the training-set
channel std stored in the checkpoint), passed through every network of the ensemble, and their
softmax outputs are averaged. The predicted gesture is the class with the highest probability.
"""
import argparse

import numpy as np

import config as C
import plots
from data_loader import load_recording, make_windows, normalize, segment_start_sample, window_start_sample
from model import adapt_batchnorm, load_ensemble, predict_proba
from train import get_device


def label_at(sample: int):
    """Gesture index if `sample` lies inside a labelled 6 s segment, else None (rest / unlabelled time)."""
    for cycle in range(C.N_CYCLES):
        for gesture in range(C.N_CLASSES):
            start = segment_start_sample(cycle, gesture)
            if start <= sample < start + C.GESTURE_DURATION_S * C.FS:
                return gesture
    return None


def predict_windows(models, channel_std, recording, starts, device):
    """recording: [n_samples, 4] in mV. Returns ensemble softmax probabilities [n_windows, 10]."""
    X = np.stack([recording[s:s + C.WINDOW_SAMPLES].T for s in starts]).astype(np.float32)   # [n, 4, 2000]
    return predict_proba(models, normalize(X, channel_std), device)


def causal_smooth(probs, n):
    """Average each row with the previous n-1 rows (consecutive windows of one recording)."""
    out = probs.copy()
    for j in range(len(probs)):
        out[j] = probs[max(0, j - n + 1):j + 1].mean(axis=0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", default="filtered", choices=C.SIGNAL_TYPES)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--cycle", type=int, help="0-4, with --gesture and --position selects one labelled window")
    ap.add_argument("--gesture", type=int, help="0-9 (see config.GESTURES)")
    ap.add_argument("--position", type=int, default=0, help="window position inside the 6 s segment")
    ap.add_argument("--start", type=float, help="window start in seconds")
    ap.add_argument("--end", type=float, help="with --start: slide the window from start to end in 0.5 s steps")
    ap.add_argument("--smooth", type=int, default=1, help="average the probabilities of the last N windows (1 = off)")
    ap.add_argument("--calibrate", action="store_true",
                    help="CALIBRATION SETTING: re-estimate the BatchNorm statistics on this participant's own "
                         "(unlabelled) gesture-segment windows before predicting (AdaBN)")
    ap.add_argument("--plot", action="store_true", help="save a figure with the window and the probability bars")
    args = ap.parse_args()

    device = get_device()
    models, ckpts = load_ensemble(C.results_dir(args.signal), device)
    assert ckpts[0].get("normalization", "train_std") == "train_std", "predict.py supports the train_std normalisation"
    channel_std = np.asarray(ckpts[0]["channel_std"], dtype=np.float32)
    recording = load_recording(args.subject, args.signal)
    split = ckpts[0]["split"]
    role = next((name for name, ids in split.items() if args.subject in ids), "unknown")
    print(f"model: {args.signal} ensemble of {len(models)} (checkpoint epochs {[c['epoch'] for c in ckpts]});  "
          f"participant {args.subject} is in the {role.upper()} set")
    if args.calibrate:
        X_cal, _, _ = make_windows(recording, args.subject)                  # this participant's windows, labels unused
        models = [adapt_batchnorm(m, normalize(X_cal, channel_std), device) for m in models]
        print(f"calibrated: BatchNorm statistics re-estimated on {len(X_cal)} unlabelled windows of participant {args.subject}")

    if args.cycle is not None and args.gesture is not None:
        starts = [window_start_sample(args.cycle, args.gesture, args.position)]
    elif args.start is not None:
        s0 = int(round(args.start * C.FS))
        if args.end is None:
            starts = [s0]
        else:
            s1 = int(round(args.end * C.FS))
            starts = list(range(s0, s1 - C.WINDOW_SAMPLES + 1, C.HOP_SAMPLES))
    else:
        ap.error("give either --cycle and --gesture (and --position), or --start (and optionally --end)")
    starts = [s for s in starts if 0 <= s <= C.RECORDING_SAMPLES - C.WINDOW_SAMPLES]
    assert starts, "window outside the recording"

    probs = predict_windows(models, channel_std, recording, starts, device)
    if args.smooth > 1:
        probs = causal_smooth(probs, args.smooth)
        print(f"probabilities averaged over the last {args.smooth} windows")
    names = [plots.pretty(g) for g in C.GESTURES]
    print(f"\n{'start [s]':>9s}  {'true label':<16s} {'predicted':<16s} {'p':>5s}   top-3")
    correct, labelled = 0, 0
    for s, p in zip(starts, probs):
        true = label_at(s)
        pred = int(p.argmax())
        order = np.argsort(p)[::-1][:3]
        top3 = ", ".join(f"{names[k]} {p[k]:.2f}" for k in order)
        mark = ""
        if true is not None:
            labelled += 1
            correct += pred == true
            mark = "  ok" if pred == true else "  WRONG"
        print(f"{s / C.FS:9.1f}  {(names[true] if true is not None else '- (rest)'):<16s} {names[pred]:<16s} {p[pred]:5.2f}   {top3}{mark}")
    if labelled:
        print(f"\n{correct}/{labelled} labelled windows correct")

    if args.plot:
        import matplotlib.pyplot as plt
        tag = (f"_smooth{args.smooth}" if args.smooth > 1 else "") + ("_calibrated" if args.calibrate else "")
        if len(starts) == 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [3, 1.6]})
            x_mv = recording[starts[0]:starts[0] + C.WINDOW_SAMPLES].T
            plots._stacked_channels(ax1, x_mv - x_mv.mean(axis=1, keepdims=True))
            true = label_at(starts[0])
            pred = int(probs[0].argmax())
            ax1.set_title(f"participant {args.subject}, {args.signal}, {starts[0] / C.FS:.1f}-{starts[0] / C.FS + C.WINDOW_S:.1f} s   "
                          f"true: {names[true] if true is not None else 'rest'}   predicted: {names[pred]} (p = {probs[0][pred]:.2f})",
                          fontsize=10, loc="left")
            ax1.set_xlabel("time [s]")
            plots._plot_probs(ax2, probs[0], true if true is not None else pred, pred)
            out = C.results_dir(args.signal) / f"predict_P{args.subject}_{starts[0] / C.FS:.1f}s.png"
        else:                                                   # timeline of predictions
            fig, ax = plt.subplots(figsize=(14, 4))
            t = np.array(starts) / C.FS + C.WINDOW_S / 2
            ax.scatter(t, probs.argmax(1), s=14, color=plots.SIGNAL_COLORS[args.signal], label="predicted", zorder=3)
            truth = [label_at(s) for s in starts]
            ax.plot(t, [g if g is not None else np.nan for g in truth], color="#0b0b0b", linewidth=2, alpha=0.5, label="true label")
            ax.set_yticks(range(C.N_CLASSES))
            ax.set_yticklabels(names)
            ax.set_xlabel("time [s]")
            ax.set_title(f"participant {args.subject}, {args.signal}: predicted gesture per 1 s window"
                         + (f", probabilities averaged over {args.smooth} windows" if args.smooth > 1 else "")
                         + (", CALIBRATED (AdaBN on this participant)" if args.calibrate else ""))
            ax.legend(frameon=False, loc="upper right")
            out = C.results_dir(args.signal) / f"predict_P{args.subject}_{args.start:.0f}-{args.end:.0f}s{tag}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()

"""Dataset figures that do not depend on a trained model.

    python plot_examples.py [--subject 2]

Writes results/figures/example_recording_<signal>.png (all channels, first cycle, segments shaded)
and results/figures/example_windows_<signal>.png (one 1 s window per gesture) for raw and filtered.
"""
import argparse

import numpy as np

from network import config as C
from network import plots
from network.data_loader import load_recording, make_windows, remove_window_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=2)
    ap.add_argument("--cycle", type=int, default=1, help="cycle (0-4) from which the example windows are taken")
    ap.add_argument("--position", type=int, default=4, help="window position inside the segment (0-10)")
    args = ap.parse_args()
    out_dir = C.RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for signal_type in C.SIGNAL_TYPES:
        rec = load_recording(args.subject, signal_type)
        plots.plot_example_recording(rec, args.subject, signal_type, out_dir / f"example_recording_{signal_type}.png")
        X, y, meta = make_windows(rec, args.subject)
        sel = np.where((meta[:, 2] == args.cycle) & (meta[:, 3] == args.position))[0]       # one window per gesture
        plots.plot_example_windows(remove_window_mean(X[sel]), y[sel], args.subject, signal_type,
                                   out_dir / f"example_windows_{signal_type}.png")
        print(f"[{signal_type}] saved example_recording_{signal_type}.png and example_windows_{signal_type}.png")


if __name__ == "__main__":
    main()

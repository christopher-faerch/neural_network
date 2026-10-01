"""Entry point of the desktop application.

    python -m app.main                       # offline mode, raw model if available
    python -m app.main --file recording.mat  # open a recording immediately
    python -m app.main --replay recording.mat --model filtered
    python -m app.main --results-root path/to/results --cpu

Run from the project folder (the one containing app/ and network/).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# torch first: both torch and Qt ship their own DLLs on Windows and torch is the fussier one
import network.inference  # noqa: F401  (imports torch)

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from app import theme  # noqa: E402
from app.main_window import MainWindow  # noqa: E402


def _install_excepthook() -> None:
    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(text)
        app = QApplication.instance()
        if app is not None:
            box = QMessageBox(QMessageBox.Critical, "Unexpected error", f"{exc_type.__name__}: {exc}")
            box.setDetailedText(text)
            box.exec()
    sys.excepthook = hook


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, help="open this recording in offline mode")
    ap.add_argument("--replay", type=Path, help="stream this recording as a simulated live source")
    ap.add_argument("--model", choices=("raw", "filtered"), help="signal type of the model to load first")
    ap.add_argument("--results-root", type=Path, help="folder with <signal>/best_model_seed*.pt (default: results/)")
    ap.add_argument("--cpu", action="store_true", help="run inference on the CPU even if CUDA is available")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("EMG Hand-Gesture Recognition")
    theme.apply_theme(app)
    _install_excepthook()
    window = MainWindow(results_root=args.results_root, model=args.model, device="cpu" if args.cpu else None)
    window.show()
    if args.file:
        window.load_file(args.file)
    if args.replay:
        window.replay_path = args.replay
        window.panel.set_replay_file(args.replay)
        window.panel.set_live_source("replay")
        window.panel.set_mode("live")
        window.connect_live()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

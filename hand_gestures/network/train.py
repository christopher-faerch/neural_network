"""Train the small 1D CNN on raw or filtered sEMG windows.

    python train.py --signal raw
    python train.py --signal filtered

Trains ENSEMBLE_SIZE identical networks with seeds SEED, SEED+1, ... (an ensemble: evaluate.py and
predict.py average their softmax outputs). Both signals use the same subjects, split, windows, model,
optimiser, learning rate, batch size, epochs and seeds. Only the input signal differs. The test set is
never touched here; see evaluate.py.
"""
import argparse
import copy
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from network import config as C
from network.data_loader import prepare_data, to_loader, save_split
from network.model import EMGCNN, summarize


def set_seed(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_snapshot() -> dict:
    """All upper-case scalar settings from config.py, stored next to the results."""
    return {k: v for k, v in vars(C).items()
            if k.isupper() and isinstance(v, (int, float, str, tuple, list, dict))}


def augment_batch(xb, gain_range=None, noise_std=None):
    """Training-time augmentation against participant differences.

    Each channel of each window gets a random gain (log-uniform in gain_range), and Gaussian noise
    is added. Both mimic what changes between people: electrode gain and background noise.
    Defaults are read from config at call time so that experiments can override them.
    """
    gain_range = C.AUG_GAIN_RANGE if gain_range is None else gain_range
    noise_std = C.AUG_NOISE_STD if noise_std is None else noise_std
    lo, hi = math.log(gain_range[0]), math.log(gain_range[1])
    gain = torch.exp(torch.empty(xb.shape[0], xb.shape[1], 1, device=xb.device).uniform_(lo, hi))
    if noise_std > 0:
        xb = xb + noise_std * torch.randn_like(xb)
    return xb * gain


def run_epoch(model, loader, criterion, device, optimizer=None, augment=False):
    """One pass over `loader`. Trains when an optimizer is given, otherwise only evaluates.

    Returns (mean loss, accuracy) over the whole loader.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            if training and augment:
                xb = augment_batch(xb)             # random gain + noise, training batches only
            logits = model(xb)                     # forward pass        [batch, 10]
            loss = criterion(logits, yb)           # cross-entropy on logits
            if training:
                optimizer.zero_grad()
                loss.backward()                    # backpropagation
                optimizer.step()                   # AdamW update
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def fit(data, device, lr=C.LEARNING_RATE, batch_size=C.BATCH_SIZE, weight_decay=C.WEIGHT_DECAY,
        dropout=C.DROPOUT, max_epochs=C.MAX_EPOCHS, patience=C.EARLY_STOPPING_PATIENCE,
        seed=C.SEED, augment=None, checkpoint_path=None, extra_checkpoint_info=None, verbose=True):
    """Train one model on data["train"], validate on data["val"] after every epoch.

    Keeps the weights of the epoch with the lowest validation loss (in best["model_state"], and
    saved to `checkpoint_path` if given) and stops early when the validation loss has not improved
    for `patience` epochs. `augment=None` uses config.AUGMENT.
    Returns (history: list of per-epoch dicts, best: dict, model with the LAST epoch's weights).
    """
    augment = C.AUGMENT if augment is None else augment
    set_seed(seed)
    X_train, y_train, _ = data["train"]
    X_val, y_val, _ = data["val"]
    train_loader = to_loader(X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = to_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    model = EMGCNN(n_channels=X_train.shape[1], dropout=dropout).to(device)   # channel count follows the data
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history, best, epochs_without_improvement = [], {"val_loss": float("inf"), "val_acc": 0.0, "epoch": 0}, 0
    if verbose:
        print(f"\n{'epoch':>5s} {'train_loss':>10s} {'val_loss':>10s} {'train_acc':>10s} {'val_acc':>10s} {'time':>7s}")
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, augment=augment)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        seconds = time.time() - t0
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            train_acc=train_acc, val_acc=val_acc, seconds=seconds))
        improved = val_loss < best["val_loss"]
        if verbose:
            print(f"{epoch:5d} {train_loss:10.4f} {val_loss:10.4f} {train_acc:10.4f} {val_acc:10.4f} {seconds:6.1f}s"
                  + ("  * new best" if improved else ""))
        if improved:
            best = {"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch,
                    "model_state": copy.deepcopy(model.state_dict())}
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                torch.save({"model_state": best["model_state"], "epoch": epoch, "val_loss": val_loss,
                            "val_acc": val_acc, "seed": seed, "augment": augment, "config": config_snapshot(),
                            **(extra_checkpoint_info or {})}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"early stopping: validation loss did not improve for {patience} epochs")
                break
    return history, best, model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", required=True, choices=C.SIGNAL_TYPES)
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--ensemble", type=int, default=C.ENSEMBLE_SIZE, help="number of seeds / ensemble members")
    args = ap.parse_args()
    seeds = [C.SEED + i for i in range(args.ensemble)]

    device = get_device()
    out_dir = C.results_dir(args.signal)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"signal: {args.signal}   results -> {out_dir}")
    print(f"hyperparameters: lr {C.LEARNING_RATE}, batch {C.BATCH_SIZE}, weight decay {C.WEIGHT_DECAY}, "
          f"dropout {C.DROPOUT}, max epochs {args.epochs}, patience {C.EARLY_STOPPING_PATIENCE}, seeds {seeds}")
    print(f"windows from {C.SEGMENT_START_OFFSET_S:.1f} s to {C.GESTURE_DURATION_S - C.SEGMENT_END_TRIM_S:.1f} s after the cue "
          f"({C.windows_per_segment()} per segment); normalisation {C.NORMALIZATION}; "
          f"augmentation {'ON (gain ' + str(C.AUG_GAIN_RANGE) + ', noise std ' + str(C.AUG_NOISE_STD) + ')' if C.AUGMENT else 'off'}")

    # ---- data (the test split is created but never used in this script)
    data = prepare_data(args.signal)
    save_split(data["split"], data["channel_std"], out_dir / "split.json", data["normalization"])

    # ---- model summary
    (out_dir / "model_summary.txt").write_text(summarize(EMGCNN()))

    # ---- train one network per seed
    members = []
    for k, seed in enumerate(seeds):
        print(f"\n===== ensemble member {k + 1}/{len(seeds)}, seed {seed} =====")
        t0 = time.time()
        history, best, _ = fit(data, device, seed=seed, max_epochs=args.epochs,
                               checkpoint_path=out_dir / f"best_model_seed{seed}.pt",
                               extra_checkpoint_info={"channel_std": data["channel_std"], "split": data["split"].as_dict(),
                                                      "signal_type": args.signal, "normalization": data["normalization"]})
        pd.DataFrame(history).to_csv(out_dir / f"history_seed{seed}.csv", index=False)
        members.append({"seed": seed, "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
                        "best_val_acc": best["val_acc"], "epochs_run": len(history), "seconds": round(time.time() - t0, 1)})
        print(f"seed {seed}: best epoch {best['epoch']}, val_loss {best['val_loss']:.4f}, val_acc {best['val_acc']:.4f}")

    # ---- remove checkpoints / histories of earlier runs with other seeds
    for p in list(out_dir.glob("best_model*.pt")) + list(out_dir.glob("history*.csv")):
        keep = {f"best_model_seed{s}.pt" for s in seeds} | {f"history_seed{s}.csv" for s in seeds}
        if p.name not in keep:
            p.unlink()

    summary = {"signal_type": args.signal, "device": str(device), "seeds": seeds, "members": members,
               "mean_best_val_acc": float(np.mean([m["best_val_acc"] for m in members])),
               "total_train_seconds": float(sum(m["seconds"] for m in members)),
               "hyperparameters": config_snapshot()}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nensemble of {len(seeds)}: best epochs {[m['best_epoch'] for m in members]}, "
          f"val acc per member {[round(m['best_val_acc'], 4) for m in members]} (mean {summary['mean_best_val_acc']:.4f})")
    print(f"saved: best_model_seed*.pt, history_seed*.csv, train_summary.json, model_summary.txt, split.json in {out_dir}")


if __name__ == "__main__":
    main()

"""Small hyperparameter check on the VALIDATION participants only. The test set is never used.

    python tune.py

Every configuration is trained once for raw and once for filtered EMG (same seed, same split).
The table reports the best validation loss / accuracy per run; the configuration with the best
mean validation accuracy over the two signals is suggested. One configuration is then used for
BOTH final experiments so that the raw-vs-filtered comparison stays fair.
Writes results/tuning.csv
"""
import time

import pandas as pd

import config as C
from data_loader import prepare_data
from train import fit, get_device, set_seed

CONFIGS = [
    dict(lr=1e-3, batch_size=64, weight_decay=1e-4, dropout=0.3),     # config.py defaults at the time of writing
    dict(lr=3e-4, batch_size=64, weight_decay=1e-4, dropout=0.3),
    dict(lr=1e-3, batch_size=128, weight_decay=1e-4, dropout=0.3),
    dict(lr=3e-4, batch_size=128, weight_decay=1e-4, dropout=0.3),
    dict(lr=1e-3, batch_size=64, weight_decay=1e-2, dropout=0.3),
    dict(lr=3e-4, batch_size=64, weight_decay=1e-2, dropout=0.3),
    dict(lr=1e-3, batch_size=64, weight_decay=1e-4, dropout=0.5),
    dict(lr=3e-4, batch_size=64, weight_decay=1e-4, dropout=0.5),
]


def main():
    set_seed(C.SEED)
    device = get_device()
    print(f"device: {device}; {len(CONFIGS)} configurations x {len(C.SIGNAL_TYPES)} signals, "
          f"max {C.MAX_EPOCHS} epochs, patience {C.EARLY_STOPPING_PATIENCE}\n")
    rows = []
    for signal_type in C.SIGNAL_TYPES:
        data = prepare_data(signal_type, verbose=False)
        for i, cfg in enumerate(CONFIGS):
            t0 = time.time()
            history, best, _ = fit(data, device, verbose=False, **cfg)
            rows.append(dict(config=i, signal=signal_type, **cfg, best_epoch=best["epoch"], epochs_run=len(history),
                             best_val_loss=round(best["val_loss"], 4), val_acc_at_best=round(best["val_acc"], 4),
                             max_val_acc=round(max(h["val_acc"] for h in history), 4),
                             train_acc_at_best=round(history[best["epoch"] - 1]["train_acc"], 4)))
            r = rows[-1]
            print(f"[{signal_type:8s}] cfg {i}: lr {cfg['lr']:.0e} bs {cfg['batch_size']:3d} wd {cfg['weight_decay']:.0e} "
                  f"do {cfg['dropout']} -> best epoch {r['best_epoch']:2d}/{r['epochs_run']:2d}, val loss {r['best_val_loss']:.4f}, "
                  f"val acc {r['val_acc_at_best']:.4f} (train {r['train_acc_at_best']:.4f})   {time.time() - t0:.0f}s")
        del data
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "tuning.csv", index=False)

    agg = df.groupby("config").agg(lr=("lr", "first"), batch_size=("batch_size", "first"),
                                   weight_decay=("weight_decay", "first"), dropout=("dropout", "first"),
                                   mean_val_loss=("best_val_loss", "mean"), mean_val_acc=("val_acc_at_best", "mean"),
                                   mean_best_epoch=("best_epoch", "mean")).sort_values("mean_val_acc", ascending=False)
    print("\nmean over raw and filtered (validation only), best first:")
    print(agg.round(4).to_string())
    best_cfg = agg.index[0]
    print(f"\nsuggested configuration: {CONFIGS[best_cfg]}  -> put these values into config.py and re-run train.py for both signals")
    print(f"saved {C.RESULTS_DIR / 'tuning.csv'}")


if __name__ == "__main__":
    main()

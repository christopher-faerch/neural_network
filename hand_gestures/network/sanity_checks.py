"""Sanity checks to run BEFORE any long training run.

    python sanity_checks.py --signal filtered
    python sanity_checks.py --signal raw

1. tensor shapes          2. labels map to gestures       3. one minibatch through the model
4. loss can be computed   5. backpropagation works        6. the CNN can overfit a tiny subset
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from network import config as C
from network.data_loader import prepare_data, to_loader
from network.model import EMGCNN, summarize
from network.train import set_seed, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="filtered", choices=C.SIGNAL_TYPES)
    ap.add_argument("--n-tiny", type=int, default=80, help="windows in the overfitting test (multiple of 10)")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    set_seed(C.SEED)
    device = get_device()
    print(f"device: {device}\n")

    # ------------------------------------------------------------------ 1. shapes
    print("=== 1. tensor shapes ===")
    data = prepare_data(args.signal)
    X_tr, y_tr, m_tr = data["train"]
    split = data["split"]
    assert X_tr.ndim == 3 and X_tr.shape[1:] == (C.N_CHANNELS, C.WINDOW_SAMPLES), X_tr.shape
    assert X_tr.dtype == np.float32 and y_tr.dtype == np.int64
    assert np.isfinite(X_tr).all(), "non-finite values in training windows"
    expected = C.WINDOWS_PER_SEGMENT * C.N_CYCLES * C.N_CLASSES
    for name in ("train", "val", "test"):
        _, _, m = data[name]
        for s in getattr(split, name):
            assert (m[:, 0] == s).sum() == expected, f"subject {s} has {(m[:, 0] == s).sum()} windows"
    assert not (set(split.train) & set(split.val)) and not (set(split.train) & set(split.test)) \
        and not (set(split.val) & set(split.test))
    print(f"  one window = [{C.N_CHANNELS} channels, {C.WINDOW_SAMPLES} samples] = {C.WINDOW_S} s at {C.FS} Hz")
    print(f"  every subject contributes {expected} windows; no subject appears in two splits          OK")
    print(f"  normalisation check on train: max |per-window mean| = {np.abs(X_tr.mean(axis=2)).max():.2e}, "
          f"per-channel std = {np.round(X_tr.std(axis=(0, 2)), 3).tolist()} (should be ~1)\n")

    # ------------------------------------------------------------------ 2. labels
    print("=== 2. labels map to gestures ===")
    assert (m_tr[:, 1] == y_tr).all() and y_tr.min() == 0 and y_tr.max() == C.N_CLASSES - 1
    rms = np.sqrt((X_tr ** 2).mean(axis=2)).sum(axis=1)      # activity per window (normalised units)
    class_rms = np.array([rms[y_tr == k].mean() for k in range(C.N_CLASSES)])
    for k in range(C.N_CLASSES):
        print(f"  label {k} -> {C.GESTURES[k]:11s}  n = {(y_tr == k).sum():5d}   mean activity = {class_rms[k]:.3f}")
    print(f"  least active class: {C.GESTURES[class_rms.argmin()]}   most active class: {C.GESTURES[class_rms.argmax()]}")
    assert class_rms.argmin() == 0, "the neutral class should be the least active one -- check the timeline"
    print("  neutral is the least active class, as the protocol implies                                 OK\n")

    # ------------------------------------------------------------------ 3. one minibatch
    print("=== 3. one minibatch through the model ===")
    model = EMGCNN().to(device)
    summarize(model)
    xb, yb = next(iter(to_loader(X_tr, y_tr, shuffle=True)))
    xb, yb = xb.to(device), yb.to(device)
    model.train()
    logits = model(xb)
    assert logits.shape == (xb.shape[0], C.N_CLASSES)
    print(f"  batch in {tuple(xb.shape)} -> logits out {tuple(logits.shape)}                                OK\n")

    # ------------------------------------------------------------------ 4. loss
    print("=== 4. loss ===")
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, yb)
    assert torch.isfinite(loss)
    print(f"  cross-entropy of the untrained model: {loss.item():.4f}   (chance level ln(10) = {np.log(10):.4f})   OK\n")

    # ------------------------------------------------------------------ 5. backprop
    print("=== 5. backpropagation ===")
    loss.backward()
    grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters()}
    assert all(g > 0 for g in grad_norms.values()), "a parameter received zero gradient"
    print(f"  all {len(grad_norms)} parameter tensors received non-zero gradients, e.g. "
          f"first conv weight |grad| = {grad_norms['blocks.0.0.weight']:.3e}                      OK\n")

    # ------------------------------------------------------------------ 6. overfit tiny subset
    print(f"=== 6. overfit a tiny balanced subset of {args.n_tiny} training windows ===")
    rng = np.random.default_rng(C.SEED)
    idx = np.concatenate([rng.choice(np.where(y_tr == k)[0], args.n_tiny // C.N_CLASSES, replace=False)
                          for k in range(C.N_CLASSES)])
    x_tiny = torch.from_numpy(X_tr[idx]).to(device)
    y_tiny = torch.from_numpy(y_tr[idx]).to(device)
    set_seed(C.SEED)
    model = EMGCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x_tiny)
        loss = criterion(out, y_tiny)
        loss.backward()
        optimizer.step()
        if step in (1, 10, 25, 50, 100, 200, args.steps):
            print(f"  step {step:4d}: loss {loss.item():.4f}   train acc {(out.argmax(1) == y_tiny).float().mean().item():.3f}")
    model.eval()
    with torch.no_grad():
        acc = (model(x_tiny).argmax(1) == y_tiny).float().mean().item()
    print(f"  accuracy on the tiny subset after training (eval mode): {acc:.3f}")
    assert acc >= 0.95, "the network could not overfit a tiny subset -- investigate the pipeline before training"
    print("\nALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()

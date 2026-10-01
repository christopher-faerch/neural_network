"""A small 1D convolutional network for multi-channel sEMG windows.

    input  [batch, 4 channels, 2000 samples]
      Conv1d(4->32, k=15) + BatchNorm + ReLU + MaxPool(4)     -> [batch, 32, 500]
      Conv1d(32->64, k=9) + BatchNorm + ReLU + MaxPool(4)     -> [batch, 64, 125]
      Conv1d(64->128, k=5) + BatchNorm + ReLU                 -> [batch, 128, 125]
      global average pooling over time                       -> [batch, 128]
      Dropout + Linear(128 -> 10)                             -> [batch, 10] logits
"""
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config as C
from data_loader import to_loader


def conv_block(in_ch, out_ch, kernel_size, pool):
    layers = [nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
              nn.BatchNorm1d(out_ch),
              nn.ReLU()]
    if pool > 1:
        layers.append(nn.MaxPool1d(pool))
    return nn.Sequential(*layers)


class EMGCNN(nn.Module):
    def __init__(self, n_channels=C.N_CHANNELS, n_classes=C.N_CLASSES,
                 conv_channels=C.CONV_CHANNELS, kernel_sizes=C.KERNEL_SIZES,
                 pool_sizes=C.POOL_SIZES, dropout=C.DROPOUT):
        super().__init__()
        blocks, in_ch = [], n_channels
        for out_ch, k, p in zip(conv_channels, kernel_sizes, pool_sizes):
            blocks.append(conv_block(in_ch, out_ch, k, p))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_ch, n_classes)

    def forward(self, x):                      # x: [batch, channels, time]
        x = self.blocks(x)                     # [batch, 128, time / 16]
        x = self.global_pool(x).squeeze(-1)    # [batch, 128]  (average over time)
        x = self.dropout(x)
        return self.classifier(x)              # logits [batch, n_classes]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def summarize(model: nn.Module, input_shape=(1, C.N_CHANNELS, C.WINDOW_SAMPLES)) -> str:
    """Print input shape, the output shape of every layer, the output shape and the parameter count."""
    lines, hooks = [], []
    lines.append(f"{'layer':<45s}{'output shape':<28s}{'params':>10s}")
    lines.append("-" * 83)
    lines.append(f"{'input':<45s}{str(list(input_shape)):<28s}{'':>10s}")

    def hook(name):
        def fn(module, inp, out):
            n = sum(p.numel() for p in module.parameters(recurse=False))
            lines.append(f"{name:<45s}{str(list(out.shape)):<28s}{n:>10,d}")
        return fn

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:            # leaf layers only
            hooks.append(module.register_forward_hook(hook(name + ": " + module.__class__.__name__)))
    device = next(model.parameters()).device          # probe tensor on the model's device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(*input_shape, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    lines.append("-" * 83)
    lines.append(f"{'output (logits)':<45s}{str(list(out.shape)):<28s}")
    lines.append(f"total trainable parameters: {count_parameters(model):,d}")
    text = "\n".join(lines)
    print(text)
    return text


# --------------------------------------------------------------------------- inference helpers
def load_ensemble(results_dir, device):
    """Load every best_model_seed*.pt of one experiment. Returns (models in eval mode, checkpoints)."""
    paths = sorted(Path(results_dir).glob("best_model_seed*.pt"), key=lambda p: int(p.stem.split("seed")[-1]))
    assert paths, f"no best_model_seed*.pt in {results_dir}: run train.py first"
    models, ckpts = [], []
    for p in paths:
        ckpt = torch.load(p, map_location=device, weights_only=False)
        model = EMGCNN().to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models.append(model)
        ckpts.append(ckpt)
    return models, ckpts


def predict_proba(models, X, device, batch_size=256):
    """Softmax probabilities averaged over the ensemble members. X: [n, channels, time] -> [n, n_classes]."""
    if isinstance(models, nn.Module):
        models = [models]
    probs = np.zeros((len(X), C.N_CLASSES))
    dummy = np.zeros(len(X), dtype=np.int64)
    with torch.no_grad():
        for model in models:
            model.eval()
            i = 0
            for xb, _ in to_loader(X, dummy, batch_size=batch_size):
                p = torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
                probs[i:i + len(p)] += p
                i += len(p)
    return probs / len(models)


def adapt_batchnorm(model, X, device, batch_size=256):
    """Test-time calibration (AdaBN): copy the model and re-estimate every BatchNorm layer's running
    mean and variance on the UNLABELLED windows X of one participant. The trained weights are untouched."""
    adapted = copy.deepcopy(model)
    for module in adapted.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
            module.momentum = None                       # cumulative average over all batches seen
    adapted.train()
    with torch.no_grad():
        for xb, _ in to_loader(X, np.zeros(len(X), dtype=np.int64), batch_size=batch_size):
            adapted(xb.to(device))
    adapted.eval()
    return adapted


def predict_proba_adabn(models, X, meta, device):
    """Per-participant AdaBN: adapt the BatchNorm statistics of every member on each participant's own
    windows (meta[:, 0] = participant), then predict that participant's windows with the adapted copies."""
    if isinstance(models, nn.Module):
        models = [models]
    probs = np.zeros((len(X), C.N_CLASSES))
    for subject in np.unique(meta[:, 0]):
        idx = np.where(meta[:, 0] == subject)[0]
        adapted = [adapt_batchnorm(m, X[idx], device) for m in models]
        probs[idx] = predict_proba(adapted, X[idx], device)
    return probs


if __name__ == "__main__":
    summarize(EMGCNN())

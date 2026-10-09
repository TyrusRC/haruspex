"""Train Haruspex: SFT (NLL + Brier) on decision rows, then fit temperature on a
held-out split, then optional isotonic recalibration of boolean P(yes).

Calibration is deliberately post-hoc and uses NO test labels (OpenJev's recipe):
the SFT head learns to rank; temperature + isotonic make the probabilities honest.

Example:
  python -m haruspex.train --train data/seed.jsonl --out runs/haruspex-v0 --epochs 3
"""

from __future__ import annotations

import argparse
import random

import torch

from .config import pick_device
from .data import Example, load_jsonl
from .model import HaruspexModel

_EPS = 1e-7


def _example_loss(model: HaruspexModel, ex: Example, brier_w: float) -> torch.Tensor:
    """SFT loss at T=1: NLL + Brier (choice/score) or BCE + Brier (boolean)."""
    s = model.score_texts(ex.texts)                 # grads flow (no_grad not set here)
    tgt = torch.tensor(ex.target, device=s.device, dtype=s.dtype)
    if ex.type == "boolean":
        p = torch.sigmoid(s[0])
        y = tgt[0]
        bce = -(y * torch.log(p + _EPS) + (1 - y) * torch.log(1 - p + _EPS))
        brier = (p - y) ** 2
        return bce + brier_w * brier
    p = torch.softmax(s, dim=0)
    nll = -(tgt * torch.log(p + _EPS)).sum()
    brier = ((p - tgt) ** 2).sum()
    return nll + brier_w * brier


def train_sft(model, examples, *, epochs, lr, batch, brier_w, device):
    # log_temp is calibrated AFTER sft — freeze it here (T=1 during sft).
    model.log_temp.requires_grad_(False)
    params = [p for n, p in model.named_parameters() if n != "log_temp" and p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    for ep in range(epochs):
        random.shuffle(examples)
        total, opt_steps = 0.0, 0
        opt.zero_grad()
        for i, ex in enumerate(examples, 1):
            loss = _example_loss(model, ex, brier_w) / batch
            loss.backward()
            total += loss.item() * batch
            if i % batch == 0:
                opt.step()
                opt.zero_grad()
                opt_steps += 1
        opt.step()
        opt.zero_grad()
        print(f"[sft] epoch {ep + 1}/{epochs}  mean_loss={total / max(1, len(examples)):.4f}")


@torch.no_grad()
def _cache_scores(model, examples):
    """Raw scores per example (detached) so calibration re-fits without the backbone."""
    model.eval()
    return [(ex, model.score_texts(ex.texts).detach().cpu()) for ex in examples]


def fit_temperature(model, cached, steps=300, lr=0.05):
    """Optimize a single temperature on held-out scores to minimize NLL/BCE."""
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.zeros(())
        for ex, s in cached:
            t = log_t.exp()
            tgt = torch.tensor(ex.target, dtype=s.dtype)
            if ex.type == "boolean":
                p = torch.sigmoid(s[0] / t)
                loss = loss - (tgt[0] * torch.log(p + _EPS) + (1 - tgt[0]) * torch.log(1 - p + _EPS))
            else:
                p = torch.softmax(s / t, dim=0)
                loss = loss - (tgt * torch.log(p + _EPS)).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        model.log_temp.copy_(log_t.detach())
    print(f"[calibrate] temperature T={float(model.temperature.item()):.3f}")


def fit_isotonic(model, cached):
    """Isotonic recalibration of boolean P(yes) on held-out (no test labels)."""
    try:
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        print("[calibrate] scikit-learn/numpy missing — skipping isotonic")
        return
    t = float(model.temperature.item())
    xs, ys = [], []
    for ex, s in cached:
        if ex.type == "boolean":
            xs.append(float(torch.sigmoid(s[0] / t)))
            ys.append(float(ex.target[0]))
    if len(xs) < 10:
        print(f"[calibrate] only {len(xs)} boolean rows — skipping isotonic")
        return
    iso = IsotonicRegression(out_of_bounds="clip").fit(np.array(xs), np.array(ys))
    model.iso_x = [float(v) for v in iso.X_thresholds_]
    model.iso_y = [float(v) for v in iso.y_thresholds_]
    print(f"[calibrate] isotonic fit on {len(xs)} boolean rows")


def main():
    ap = argparse.ArgumentParser(description="Train Haruspex decision model")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default="", help="held-out rows for calibration (else split 15%% of train)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8, help="examples per optimizer step")
    ap.add_argument("--brier-weight", type=float, default=1.0)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[init] device={device}")
    train = load_jsonl(args.train)
    if args.val:
        val = load_jsonl(args.val)
    else:
        random.shuffle(train)
        cut = max(1, int(len(train) * 0.15))
        val, train = train[:cut], train[cut:]
    print(f"[init] train={len(train)}  val={len(val)}")

    model = HaruspexModel().to(device)
    train_sft(model, train, epochs=args.epochs, lr=args.lr, batch=args.batch,
              brier_w=args.brier_weight, device=device)
    cached = _cache_scores(model, val)
    fit_temperature(model, cached)
    fit_isotonic(model, cached)
    model.save(args.out)
    print(f"[done] saved -> {args.out}")


if __name__ == "__main__":
    main()

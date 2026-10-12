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


def _loss_from_scores(ex: Example, s: torch.Tensor, brier_w: float) -> torch.Tensor:
    """SFT loss at T=1 from precomputed candidate scores: NLL + Brier (choice/
    score) or BCE + Brier (boolean)."""
    tgt = torch.tensor(ex.target, device=s.device, dtype=s.dtype)
    if ex.type == "boolean":
        p = torch.sigmoid(s[0])
        y = tgt[0]
        bce = -(y * torch.log(p + _EPS) + (1 - y) * torch.log(1 - p + _EPS))
        return bce + brier_w * (p - y) ** 2
    p = torch.softmax(s, dim=0)
    nll = -(tgt * torch.log(p + _EPS)).sum()
    brier = ((p - tgt) ** 2).sum()
    return nll + brier_w * brier


def _collate(chunk: list[Example]):
    """Flatten a mini-batch of examples' candidate texts + per-example spans so
    one forward scores the whole batch (state-per-candidate, but batched across
    examples — far fewer, larger forwards than one example at a time)."""
    texts: list[str] = []
    spans: list[tuple[int, int, Example]] = []
    for ex in chunk:
        spans.append((len(texts), len(texts) + len(ex.texts), ex))
        texts.extend(ex.texts)
    return texts, spans


def _enable_memory_savers(backbone):
    """Gradient checkpointing + no KV-cache — the big activation-memory savers that
    let a 4B train on a 12 GB card. Best-effort (skip quietly if unsupported)."""
    try:
        backbone.gradient_checkpointing_enable()
    except Exception:  # noqa: BLE001 - best-effort hardware opt
        pass
    try:
        backbone.enable_input_require_grads()   # peft: let checkpointing reach a frozen base
    except Exception:  # noqa: BLE001
        pass
    cfg = getattr(backbone, "config", None)
    if cfg is not None:
        try:
            cfg.use_cache = False
        except Exception:  # noqa: BLE001
            pass


def train_sft(model, examples, *, epochs, lr, batch, brier_w, device, micro=2, checkpoint=True):
    # log_temp is calibrated AFTER sft — freeze it here (T=1 during sft).
    model.log_temp.requires_grad_(False)
    if checkpoint:   # needed for a 4B on 12 GB; a small model fits without it (faster)
        _enable_memory_savers(model.backbone)
    params = [p for n, p in model.named_parameters() if n != "log_temp" and p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    for ep in range(epochs):
        random.shuffle(examples)
        total, n, since_step = 0.0, 0, 0
        opt.zero_grad()
        # Forward a MICRO-batch of examples at a time (bounds peak memory), and
        # accumulate gradients until `batch` examples, then step.
        for i in range(0, len(examples), micro):
            mb = examples[i:i + micro]
            texts, spans = _collate(mb)
            scores = model.score_texts(texts, batch_size=len(texts))
            loss = sum(_loss_from_scores(ex, scores[a:b], brier_w) for a, b, ex in spans) / batch
            loss.backward()
            total += float(loss.item()) * batch        # undo the /batch for logging
            n += len(mb)
            since_step += len(mb)
            if since_step >= batch:
                opt.step()
                opt.zero_grad()
                since_step = 0
        if since_step:
            opt.step()
            opt.zero_grad()
        print(f"[sft] epoch {ep + 1}/{epochs}  mean_loss={total / max(1, n):.4f}")


@torch.no_grad()
def _cache_scores(model, examples, chunk_texts=16):
    """Per-example scores (detached), computed in one batched pass over all texts."""
    model.eval()
    texts, spans = _collate(examples)
    allscores = model.score_texts(texts, batch_size=chunk_texts).detach().cpu()
    return [(ex, allscores[a:b]) for a, b, ex in spans]


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
    ap.add_argument("--batch", type=int, default=8, help="examples per optimizer step (grad accum)")
    ap.add_argument("--micro", type=int, default=2, help="examples per forward (bounds GPU memory)")
    ap.add_argument("--no-checkpoint", dest="checkpoint", action="store_false",
                    help="disable gradient checkpointing (faster; use when the model fits)")
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
              brier_w=args.brier_weight, device=device, micro=args.micro,
              checkpoint=args.checkpoint)
    cached = _cache_scores(model, val)
    fit_temperature(model, cached)
    fit_isotonic(model, cached)
    model.save(args.out)
    print(f"[done] saved -> {args.out}")


if __name__ == "__main__":
    main()

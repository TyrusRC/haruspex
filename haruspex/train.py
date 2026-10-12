"""Specialize GLiNER2.5-Decide on your decision rows.

LoRA by default (measured peak ~3.8 GB on a 12 GB GPU); `--mode full` fine-tunes
all parameters (~8.3 GB). The adapter is merged into the base and saved as a
standalone directory, so a plain `Haruspex.load(<dir>)` / `AutoExtractor
.from_pretrained(<dir>)` loads it with no extra steps.

  python -m haruspex.train --train data/security_seed.jsonl --out runs/sec --epochs 3

Training reuses the published base model — Haruspex does not pretrain one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_rows, to_input_example


def train(base: str, rows, out: str, *, mode="lora", epochs=3, max_steps=-1,
          batch_size=4, grad_accum=1, lora_r=16, lora_alpha=32, logging_steps=5):
    import torch
    from gliner2 import GLiNER2
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    examples = [to_input_example(r) for r in rows]
    cuda = torch.cuda.is_available()
    cfg = TrainingConfig(
        output_dir=out, num_epochs=epochs, max_steps=max_steps,
        batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        use_lora=(mode == "lora"), lora_r=lora_r, lora_alpha=lora_alpha,
        gradient_checkpointing=(mode == "full"),
        bf16=cuda, fp16=False, eval_strategy="no", save_best=False,
        save_total_limit=1, num_workers=0, pin_memory=False,
        validate_data=False, logging_steps=logging_steps, warmup_ratio=0.1,
    )
    model = GLiNER2.from_pretrained(base)
    trainer = GLiNER2Trainer(model=model, config=cfg)
    trainer.train(train_data=examples)
    if mode != "lora":
        # NOTE (ceiling): a full fine-tune trains fine, but merge/save writes an
        # incompatible span-head checkpoint in gliner2 2.1 that will not reload.
        # LoRA is the supported, loadable artifact (and 30 MB vs 2 GB). Upgrade
        # path: a standalone save once the gliner2 save/merge bug is fixed.
        raise NotImplementedError(
            "full-mode reload is blocked by a gliner2 2.1 save bug — use --mode lora")
    # Artifact = the small LoRA adapter the trainer saved under <out>/final, plus
    # a manifest. Haruspex.load() reattaches it to the reused base at inference.
    (Path(out) / "haruspex.json").write_text(
        json.dumps({"base": base, "adapter": "final", "mode": "lora"}, indent=2),
        encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description="Specialize GLiNER2.5-Decide on decision rows")
    ap.add_argument("--train", required=True, help="labeled JSONL")
    ap.add_argument("--out", required=True, help="output dir (standalone, merged model)")
    ap.add_argument("--base", default="", help="base model id/dir (default fastino/GLiNER2.5-Decide)")
    ap.add_argument("--mode", choices=["lora", "full"], default="lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    args = ap.parse_args()

    from .config import resolve_model
    base = resolve_model(args.base)
    rows = load_rows(args.train)
    print(f"[train] base={base} rows={len(rows)} mode={args.mode}")
    out = train(base, rows, args.out, mode=args.mode, epochs=args.epochs,
                max_steps=args.max_steps, batch_size=args.batch, grad_accum=args.grad_accum)
    print(f"[train] saved standalone model -> {out}")
    print("[train] fit a calibrator next: python -m haruspex.calibrate "
          f"--model {out} --data <val.jsonl>")


if __name__ == "__main__":
    main()

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
          batch_size=4, grad_accum=1, lora_r=16, lora_alpha=32, logging_steps=5,
          grad_checkpoint=None, max_len=None, seed=0, sanity_gate=True,
          gate_domain="web", save_every=0, resume=False):
    import torch
    from gliner2 import GLiNER2
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    # Reproducible training: seed torch/numpy/python up front so a good run can be
    # reproduced and a bad run retried deterministically with a different seed.
    try:
        from transformers import set_seed as _set_seed
        _set_seed(seed)
    except Exception:  # noqa: BLE001 - fall back to torch-only seeding
        torch.manual_seed(seed)

    examples = [to_input_example(r) for r in rows]
    cuda = torch.cuda.is_available()
    # Full fine-tune always checkpoints; LoRA checkpoints on request (long,
    # many-option states — real decision data — can OOM a modest GPU otherwise).
    ckpt = (mode == "full") if grad_checkpoint is None else grad_checkpoint
    # save_every: save an intermediate checkpoint every N steps (0 = final only).
    # Intermediate checkpoints land under <out>/checkpoint-<step>/ and the last
    # save_total_limit are kept. On resume, the trainer picks up from the latest.
    save_limit = 2 if save_every > 0 else 1
    cfg = TrainingConfig(
        output_dir=out, num_epochs=epochs, max_steps=max_steps,
        batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        use_lora=(mode == "lora"), lora_r=lora_r, lora_alpha=lora_alpha,
        gradient_checkpointing=ckpt, max_len=max_len,
        bf16=cuda, fp16=False, eval_strategy="no", save_best=False,
        save_total_limit=save_limit, num_workers=0, pin_memory=False,
        validate_data=False, logging_steps=logging_steps, warmup_ratio=0.1,
    )
    model = GLiNER2.from_pretrained(base)
    trainer = GLiNER2Trainer(model=model, config=cfg)
    # Resume from the latest checkpoint if one exists and --resume is set.
    train_kwargs: dict = {"train_data": examples}
    if resume:
        ckpt_dirs = sorted(Path(out).glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
        if ckpt_dirs:
            resume_path = str(ckpt_dirs[-1])
            print(f"[train] resuming from {resume_path}", flush=True)
            train_kwargs["resume_from_checkpoint"] = resume_path
    trainer.train(**train_kwargs)
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
    if sanity_gate:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ok, report = sanity_check(out, device=dev, domain=gate_domain)
        print(f"[sanity-gate] {report}", flush=True)
        if not ok:
            raise RuntimeError(
                "[sanity-gate] the trained model failed the discrimination probe "
                f"(degenerate / collapsed to ~0.5): {report}. This is NOT a usable "
                "checkpoint — retry with a different --seed, add/clean data, or adjust "
                "the recipe. Pass sanity_gate=False / --no-sanity-gate to override.")
    return out


# Fixed discrimination probes, per domain. A healthy triage model scores a confirmed
# finding ~1.0 and clear noise/safe ~0.0; the collapse failure mode (training drives every
# output to ~0.5) fails the margin check, so the gate refuses a degenerate checkpoint.
# `domain="all"` requires the model to discriminate EVERY domain at once — the gate for an
# all-in-one model, which is exactly where the collapse shows up.
_SANITY_Q = {"type": "boolean", "question": "Is this a real, reportable security finding?",
             "criteria": {"true": "a real, exploitable vulnerability", "false": "noise / not reportable alone"}}
_DOMAIN_PROBES = {
    "web": {
        "real": ["Unauthenticated remote code execution confirmed: os.popen('id') returned uid=0(root)",
                 "SQL injection confirmed: a UNION SELECT returned the database version and bypassed authentication"],
        "neg": ["the response is missing the X-Frame-Options header on a page with no sensitive action",
                "the query uses parameterized bindings: cursor.execute(sql, (user_id,))"],
    },
    "mobile": {
        "real": ["Android app ships an AWS access key AKIA... embedded in res/values/strings.xml inside the APK",
                 "Release manifest sets android:debuggable=true",
                 "An Android activity has android:exported=true with no permission; any installed app can launch it"],
        "neg": ["Android symmetric key generated and held in the AndroidKeystore, never leaves the device",
                "Content provider sets android:exported=false with no intent filter",
                "The app sets minSdkVersion 24"],
    },
    "cloud": {
        "real": ["Terraform aws_s3_bucket_public_access_block sets block_public_acls=false and the bucket policy grants Principal \"*\" s3:GetObject",
                 "IAM policy document allows Action \"*\" on Resource \"*\"",
                 "Security group ingress allows 0.0.0.0/0 to port 22"],
        "neg": ["aws_s3_bucket_public_access_block sets all four public-access settings to true",
                "IAM policy scoped to s3:GetObject on one bucket ARN",
                "Security group ingress allows TCP 443 from a single corporate CIDR"],
    },
    "network": {
        "real": ["netexec smb shows signing:False on the domain controller, so NTLM relay is possible",
                 "A service account has an SPN and Kerberoasts with an RC4 hash that cracks offline",
                 "LLMNR and NBT-NS are enabled on the subnet, allowing NTLMv2 capture via responder"],
        "neg": ["netexec smb shows signing:True required on all hosts",
                "Kerberos pre-authentication is required and the service account uses a long AES-only password",
                "LDAP channel binding and signing are enforced"],
    },
    "llm": {
        "real": ["A prompt injection made the model ignore its system prompt and print its hidden instructions",
                 "The model output is passed unsanitized into eval(), so attacker input runs as code",
                 "The LLM endpoint accepted user-controlled tool calls and executed a server-side HTTP request to an internal metadata endpoint"],
        "neg": ["The model refused the jailbreak and stayed on task",
                "The model echoed a benign greeting with no security impact",
                "A tool call is gated behind an explicit user confirmation"],
    },
}


def _probe_set(domain: str) -> dict:
    if domain == "all":
        real, neg = [], []
        for d in _DOMAIN_PROBES.values():
            real += d["real"]; neg += d["neg"]
        return {"real": real, "neg": neg}
    if domain not in _DOMAIN_PROBES:
        raise ValueError(f"unknown gate domain {domain!r}; known: {list(_DOMAIN_PROBES)} + 'all'")
    return _DOMAIN_PROBES[domain]


def sanity_check(model_dir: str, device: str = "cpu", domain: str = "web", probes: dict | None = None):
    """Probe a freshly trained model for discrimination on a domain. Returns (ok, report).

    `domain` selects a built-in probe set ('web', 'mobile', or 'all' = every domain at
    once); `probes` ({"real":[...], "neg":[...]}) overrides it with a caller set. Guards the
    collapse-to-0.5 failure mode: REAL must score high, NOISE/SAFE low, with a clear margin."""
    from .model import Haruspex
    ps = probes or _probe_set(domain)
    m = Haruspex.load(model_dir, device=device)

    def p_yes(state: str) -> float:
        v = m.decide({**_SANITY_Q, "state": state})
        return float(v.get("probability", 0.0) or 0.0)

    real = [p_yes(s) for s in ps["real"]]
    neg = [p_yes(s) for s in ps["neg"]]
    min_real, max_neg = min(real), max(neg)
    margin = (sum(real) / len(real)) - (sum(neg) / len(neg))
    ok = (min_real >= 0.65) and (max_neg <= 0.35) and (margin >= 0.4)
    return ok, {"ok": ok, "domain": "custom" if probes else domain,
                "min_real": round(min_real, 3), "max_neg": round(max_neg, 3),
                "margin": round(margin, 3),
                "real": [round(x, 3) for x in real], "neg": [round(x, 3) for x in neg]}


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
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="trade compute for memory (needed for long many-option states)")
    ap.add_argument("--max-len", type=int, default=0, help="cap sequence length (0 = model default)")
    ap.add_argument("--seed", type=int, default=0, help="training seed (reproducible; retry a collapsed run with a new seed)")
    ap.add_argument("--no-sanity-gate", action="store_true",
                    help="skip the post-train discrimination probe (not recommended)")
    ap.add_argument("--gate-domain", default="web", help="sanity-gate probe set: web | mobile | all")
    ap.add_argument("--save-every", type=int, default=0,
                    help="save intermediate checkpoint every N steps (0 = final only). "
                         "Allows --resume if training is interrupted.")
    ap.add_argument("--resume", action="store_true",
                    help="resume from latest checkpoint in --out dir (requires prior --save-every run)")
    args = ap.parse_args()

    from .config import resolve_model
    base = resolve_model(args.base)
    rows = load_rows(args.train)
    print(f"[train] base={base} rows={len(rows)} mode={args.mode} seed={args.seed}"
          f"{' resume' if args.resume else ''}")
    out = train(base, rows, args.out, mode=args.mode, epochs=args.epochs,
                max_steps=args.max_steps, batch_size=args.batch, grad_accum=args.grad_accum,
                grad_checkpoint=args.grad_checkpoint or None, max_len=args.max_len or None,
                seed=args.seed, sanity_gate=not args.no_sanity_gate, gate_domain=args.gate_domain,
                save_every=args.save_every, resume=args.resume)
    print(f"[train] saved standalone model -> {out}")
    print("[train] fit a calibrator next: python -m haruspex.calibrate "
          f"--model {out} --data <val.jsonl>")


if __name__ == "__main__":
    main()

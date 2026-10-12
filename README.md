# Haruspex

> Train and run a small, local **decision model** for your own decisions — calibrated
> typed verdicts, not text.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Haruspex is an open-source toolkit for a small decision model you **train on your own
decisions and run locally**. Given a situation and bounded options it returns a typed,
**calibrated** answer, not free text:

- **boolean** — yes/no + `P(yes)`
- **choice** — pick one option (+ a probability per option)
- **score** — rate on an ordered rubric (+ an expected level)

It trains on a modest GPU (or CPU), runs on **CPU / CUDA / Apple-MPS**, serves the Jev
/ TypeSafe HTTP contract, and has **no vendor and no dependency on any other project**.

## When to use this (and when not)

Use Haruspex when you want to **own and specialize** a small decision model for a real
workflow — triage, routing, rubric scoring, verifying/gating an agent step, security-ops
decisions — and run it locally/offline. You train it on your data; the model is yours,
small, and specialized to the job.

It is **not** a pretrained, benchmark-tuned checkpoint, and it is not trying to win a
leaderboard. If you just want a ready-made *general* decision model, use an existing one
(TypeSafe **Jev**, or the open **OpenJev / Von / Kev / NanoJev**). Haruspex deliberately
uses the **same standard recipe** as those — its point is a small model that works for
*your* task on *your* hardware, specialized by *your* data, not a new architecture.

This repo ships an **example domain pack for security operations** (see
[Security pack](#example-domain-pack-security-operations)) as one worked specialization;
nothing in the model is security-specific — point it at your own decisions.

## Architecture

```
[STATE] [QUESTION] [CANDIDATE]  ->  backbone  ->  pooled vector per candidate
                                ->  OptionInteractor head (self-attention over the
                                    candidates + scalar readout)  ->  one score each
                                ->  temperature-scale  ->  normalize per question
```
- choice → `softmax(scores / T)`
- boolean → `sigmoid(score / T)` (+ optional **isotonic** recalibration)
- score → `softmax(level scores / T)`, value `= Σ i · pᵢ`

The candidate count is free at runtime — no fixed classifier. Calibration is **post-hoc**
and uses no test labels: NLL+Brier training → temperature on a held-out split → isotonic.
Evaluation reports Accuracy, Brier, and **ECE**.

This recipe is standard for small decision models (see Acknowledgements); Haruspex claims
no architectural novelty — the value is the local, train-your-own workflow below.

**Small and local by default.** The backbone is a ModernBERT encoder (~150M) —
non-autoregressive, single-forward, CPU/edge-friendly. It is backbone-agnostic:
swap in any Hugging Face encoder (or an opt-in decoder) via env, so the model stays
general without shipping a large preset.

```sh
HARUSPEX_BACKBONE=answerdotai/ModernBERT-large     # a bigger encoder, still small
HARUSPEX_BACKBONE=<small-decoder> HARUSPEX_TYPE=decoder HARUSPEX_LORA=1   # opt-in decoder
```

## Install & quickstart

```sh
pip install -e .                      # torch + transformers + scikit-learn
python -m haruspex.train --train data/seed.jsonl --out runs/dev --epochs 3
python -m haruspex.eval  --model runs/dev --data data/seed.jsonl      # Accuracy / Brier / ECE / latency
pip install -e ".[serve]" && python -m haruspex.serve --model runs/dev --port 3000
```
Device auto-selects (CUDA → Apple-MPS → CPU); override with `--device`.

## Training modes

`--mode` trades speed for accuracy:

- **`head`** (default) — freeze the backbone, run it **once** to cache features, then
  train only the head + calibrate. The backbone is never re-encoded across epochs, so
  it is fast and light. Fine-tunes the head without retraining the whole model.
- **`lora`** — train a LoRA adapter + the head (merged on save).
- **`full`** — train the whole backbone.

Memory knobs for large backbones: `--micro` (examples per forward, grad-accumulated to
`--batch`), gradient checkpointing (on by default, `--no-checkpoint` to disable),
`HARUSPEX_MAX_LEN`.

### Hardware

Haruspex is small-native, not a quantized large model. **`--mode head` trains on a
modest GPU** — measured peak ~0.9 GB on a 12 GB RTX 3060 (the backbone runs once for
inference, then only the head trains), and it runs on CPU too. **`--mode full`/`lora`
fine-tune the backbone** and need more memory, especially on long inputs — lower
`--micro` / `HARUSPEX_MAX_LEN` or use a larger GPU. For local use, head mode is the
intended path.

### Accuracy

The head is an **OptionInteractor**: a light self-attention over a question's candidate
vectors (so each option's score accounts for the others — competition/overlap) plus a
scalar readout. It raises accuracy without a larger backbone, and because it operates
on the small candidate vectors it trains fast in head mode.

## Serving (Jev / TypeSafe HTTP contract)

Self-hosted and **keyless** (the key only authenticates a remote gateway, never your
own model). Endpoints:

- `POST /v1/evaluate` — native Jev shape.
- `POST /v1/systemone` — TypeSafe shape (accepts `noul` as boolean).
- `GET /v1/models` — model list.

```
POST /v1/evaluate  (or /v1/systemone)
{ "model":"haruspex", "state": <str|obj>,
  "questions": { "<name>": { "type":"boolean"|"noul"|"choice"|"score",
                             "instructions":"...",
                             "criteria": {true,false} | {opt:desc} | [low,high] } } }
```

## Data

JSONL decision rows — format in [`data/schema.md`](data/schema.md).
`scripts/convert_mmdm.py` pulls a public decision dataset (HF Hub) and converts it;
`scripts/fetch_data.py` is a generic converter. Bootstrap labels from a heuristic, real
outcomes, or one-shot distillation from a larger model.

## Example domain pack: security operations

`data/security_seed.jsonl` is a worked example — labeled decisions for both sides of
security work. It is optional and illustrative; the model itself is general.

- **Offensive:** template/probe relevance, "is this scanner hit a real finding?",
  finding severity, verdict discipline (vulnerable / not-vulnerable / inconclusive).
- **Defensive:** alert triage (true vs false positive), alert severity, detection-rule
  applicability, incident routing, CVE applicability + remediation urgency.

Mix it with general data so the model keeps broad competence:
```sh
python scripts/convert_mmdm.py --out-dir data
cat data/security_seed.jsonl >> data/train.jsonl
python -m haruspex.train --train data/train.jsonl --val data/val.jsonl --out runs/mixed
```

## Publish to Hugging Face

A trained folder is a standard HF model dir (backbone + tokenizer) plus the decision
head (`head.pt`, `meta.json`). Publish it with a generated model card:
```sh
pip install -e ".[data]" && huggingface-cli login     # or export HF_TOKEN=...
python scripts/publish_hf.py --model runs/mixed --repo <you>/haruspex
```
Load it back with `HaruspexModel.load("<you>/haruspex")` or `haruspex-serve`. Loading
needs the `haruspex` package (the head sits beside the backbone) — a plain `AutoModel`
load gives only the encoder.

## Benchmarking (optional)

Haruspex is built to **do work locally**, not to win a leaderboard — the goal is a
small model with strong calibration that you can run anywhere, not a large one tuned
for a score. It does speak the Jev / TypeSafe contract, so you *can* evaluate it:
`haruspex.eval` reports Accuracy / Brier / ECE / latency and has a `--compare-url`
flag to run head-to-head against any `/v1/evaluate` or `/v1/systemone` server. If you
benchmark it, judge it on **capability-per-cost** (quality relative to size, latency,
and memory), which is where a small model is meant to win.

## Status

Early. The toolkit (train → calibrate → eval → serve) runs end-to-end and trains on a
modest GPU in head mode. By design **no pretrained general checkpoint is shipped** —
you train your own on your data. Contributions and independent evaluations welcome.

## Contributing

Issues and PRs welcome — new backbones, calibration methods, datasets/domain packs,
and eval results especially. Keep changes small and claims verifiable (report the
metric and the setup). Run `python -m unittest discover tests` before a PR.

## License

[Apache-2.0](LICENSE).

## Acknowledgements

The idea and the recipe come from this line of work, not from Haruspex: TypeSafe's
**Jev** (the "System One" decision-model idea) and the open reproductions —
**OpenJev**, **Von**, **Kev**, **NanoJev**, **Laya**, **SemIf** — whose per-candidate
scoring and set-interaction Haruspex follows. Haruspex's only aim is to make training
and running a small, specialized one **local and easy**; the architecture is theirs.

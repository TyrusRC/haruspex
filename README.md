# Haruspex

> A small, open-weight **decision model** — reads the signs, returns calibrated verdicts.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Haruspex is an open-source, open-weight **decision model**. Given a situation and a
bounded set of options, it returns a *typed* answer with a **calibrated probability**,
not free text:

- **boolean** — yes/no + `P(yes)`
- **choice** — pick one option (+ a probability per option)
- **score** — rate on an ordered rubric (+ an expected level)

It is a **general** decision model — usable for any decision task — runs on **CPU,
CUDA, or Apple-MPS**, serves the Jev / TypeSafe HTTP contract, and has **no dependency
on any other project**. It is inspired by TypeSafe's Jev and the open OpenJev
reproductions.

## What it's for

Decision models are a fit for classification, routing, rubric scoring, and
verification — anywhere you want a calibrated typed answer instead of a paragraph.
Examples: triage/route a support ticket, grade an answer against a rubric, judge a
model output, gate an agent step. This repo also ships an **example domain pack for
security operations** (see [Security pack](#example-domain-pack-security-operations)),
but nothing in the model is security-specific — point it at your own decision data.

## Architecture

```
[STATE] [QUESTION] [CANDIDATE]  ->  backbone  ->  pooled vector
                                ->  shared Linear(h, 1) head  ->  one scalar per candidate
                                ->  temperature-scale  ->  normalize per question
```
- choice → `softmax(scores / T)`
- boolean → `sigmoid(score / T)` (+ optional **isotonic** recalibration)
- score → `softmax(level scores / T)`, value `= Σ i · pᵢ`

The 1-D head means the candidate count is free at runtime — no fixed classifier.
Calibration is **post-hoc** and uses no test labels: NLL+Brier training → temperature
on a held-out split → isotonic. Evaluation reports Accuracy, Brier, and **ECE**.

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

Early. The pipeline (train → calibrate → eval → serve) is runnable and validated
end-to-end; shipped checkpoints are **not** published yet. Contributions and
independent evaluations are welcome.

## Contributing

Issues and PRs welcome — new backbones, calibration methods, datasets/domain packs,
and eval results especially. Keep changes small and claims verifiable (report the
metric and the setup). Run `python -m unittest discover tests` before a PR.

## License

[Apache-2.0](LICENSE).

## Acknowledgements

Inspired by TypeSafe's **Jev** (the "System One" decision-model idea) and the open
**OpenJev** reproductions, whose per-candidate scoring recipe Haruspex follows.

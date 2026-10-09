# Haruspex

> A small, open-weight **decision model** — reads the signs, returns calibrated verdicts.

Haruspex answers *typed* decisions about a situation — **boolean** (yes/no with a
probability), **choice** (pick one option), **score** (rate on a rubric) — and
returns **calibrated probabilities**, not free text. It is built to run on a
laptop or MacBook from 8 GB RAM, on **CPU, CUDA, or Apple-MPS**.

## Why it exists

Hosted decision models (TypeSafe Jev) and the open OpenJev reproductions are
2B–27B general decoders — accurate, but large and cloud-bound. Haruspex takes
OpenJev's elegant **per-candidate scoring** recipe and shrinks it onto a small
**encoder** backbone, then **specializes** on in-domain (security) decisions. The
goal is not to beat a 27B on the general benchmark — it is to **Pareto-win** on
the axes that matter for a tool you run yourself:

- **Footprint + latency** — ~150M params (ModernBERT-base), CPU/MPS/CUDA, sub-100 ms.
- **In-domain accuracy** — a specialist beats a big generalist on *its* tasks.
- **Calibration** — Brier **and ECE** reported (OpenJev reports neither ECE).
- **No order bug** — order-randomized training removes candidate-order sensitivity.

## Architecture

`[STATE] [QUESTION] [CANDIDATE]` → ModernBERT encoder → mean-pool → a shared
**`Linear(h, 1)`** head → one scalar per candidate → **temperature-scale** →
normalize per question:

- choice → `softmax(scores / T)`
- boolean → `sigmoid(score / T)` (+ optional **isotonic** recalibration)
- score → `softmax(level scores / T)`, value `= Σ i · pᵢ`

The 1-D head means the candidate count is free at runtime — no fixed classifier.
Calibration is **post-hoc** and uses no test labels (NLL+Brier SFT → temperature
on a held-out split → isotonic).

## Quickstart

```sh
pip install -e .            # torch + transformers + sklearn
# train on the seed set (smoke run; swap in a real labeled set to train for keeps)
python -m haruspex.train --train data/seed.jsonl --out runs/haruspex-v0 --epochs 3
# evaluate: Accuracy / Brier / ECE / P50 latency
python -m haruspex.eval --model runs/haruspex-v0 --data data/seed.jsonl
# head-to-head vs an OpenJev /v1/evaluate server
python -m haruspex.eval --model runs/haruspex-v0 --data bench.jsonl --compare-url http://127.0.0.1:3001

# serve the Jev-compatible API
pip install -e ".[serve]"
python -m haruspex.serve --model runs/haruspex-v0 --port 3000
```

Device is auto-selected (CUDA → Apple-MPS → CPU); override with `--device`.
Swap the backbone with `HARUSPEX_BACKBONE=microsoft/deberta-v3-small`.

## Serving (the `/v1/evaluate` contract)

Haruspex speaks the same typed-decision API as hosted Jev, so it drops into any
client that calls it. It is **self-hosted and keyless** — the key only
authenticates a remote gateway, never your own model.

```
POST /v1/evaluate
{ "model":"haruspex", "state": <str|obj>,
  "questions": { "<name>": { "type":"boolean"|"choice"|"score",
                             "instructions":"...",
                             "criteria": {true,false} | {opt:desc} | [low,high] } } }
```

## Data

JSONL decision rows — see [`data/schema.md`](data/schema.md). `data/seed.jsonl`
is a tiny security-domain starter. Bootstrap labels from a heuristic, real
pentest outcomes, or one-shot distillation from a big model.

## Status

v0 scaffold: architecture, training (SFT + calibration), eval (Acc/Brier/ECE/latency),
and the `/v1/evaluate` server. Next: a real labeled security-decision set, a held-out
benchmark, and the head-to-head numbers vs OpenJev-4B. Apache-2.0.

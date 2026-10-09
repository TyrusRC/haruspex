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

## Two variants (one codebase)

Pick with `HARUSPEX_PROFILE`:

| Profile | Backbone | For |
|---|---|---|
| `edge` (default) | ModernBERT-base (~150M, encoder) | runs anywhere (CPU/MPS/8 GB); the keyless local engine for Praetor/assay |
| `4b` | Qwen3.5-4B + LoRA (decoder) | the **JevBench leaderboard** contender (the eligible small class) |

Both share the `Linear(h,1)` candidate scorer + temperature/isotonic calibration.
The `4b` variant trains with LoRA and **merges it into the base on save**, so the
published artifact loads like any HF model.

## JevBench leaderboard plan

Ranked on **Capability = average(Intelligence, Calibration)**; the #10 bar is ~72.2
and the whole eligible small class is Qwen3.5-4B. Strategy: a well-trained 4B gets
Intelligence into the ~55–60 band, and our calibration pipeline pushes Calibration
to ~90 — Capability ~72–75, top-10 range. Train on **general** decision data (not
security-only), since JevBench is general.

```sh
pip install -e ".[train4b,data]"
python scripts/fetch_data.py --dataset shenjunhao/mmdm --split train --out data/train.jsonl
HARUSPEX_PROFILE=4b python -m haruspex.train --train data/train.jsonl --out runs/haruspex-4b --epochs 2
HARUSPEX_PROFILE=4b python -m haruspex.eval --model runs/haruspex-4b --data data/val.jsonl \
    --compare-url http://127.0.0.1:3001   # vs an OpenJev-4B server
```
Fits your RTX 3060 (12 GB) and an 8 GB Mac at 4-bit; passes the "Jev-class" ≤2×
cost/latency eligibility gate.

## Serving (the `/v1/evaluate` + `/v1/systemone` contract)

Haruspex speaks the typed-decision API, self-hosted and **keyless** (the key only
authenticates a remote gateway, never your own model). It exposes:

- `POST /v1/evaluate` — native Jev shape (Praetor / assay).
- `POST /v1/systemone` — TypeSafe shape, accepts `noul` as boolean. **This is what
  JevBench's open-weights adapter calls** — submit the public URL of this server, or
  the HF repo.
- `GET /v1/models` — model list (adapter probe).

```
POST /v1/evaluate  (or /v1/systemone)
{ "model":"haruspex", "state": <str|obj>,
  "questions": { "<name>": { "type":"boolean"|"noul"|"choice"|"score",
                             "instructions":"...",
                             "criteria": {true,false} | {opt:desc} | [low,high] } } }
```

## Data

JSONL decision rows — see [`data/schema.md`](data/schema.md). `data/seed.jsonl`
is a tiny security-domain starter. Bootstrap labels from a heuristic, real
pentest outcomes, or one-shot distillation from a big model.

## Status

Pipeline is runnable and validated end-to-end on GPU for the `edge` variant
(train → calibrate → serve → `/v1/evaluate` + `/v1/systemone` + `noul`). The `4b`
decoder + LoRA path is code-complete but needs a GPU training run to produce real
weights. Next: pull general decision data, train `4b`, fit calibration hard, and
post the JevBench head-to-head vs OpenJev-4B. Apache-2.0.

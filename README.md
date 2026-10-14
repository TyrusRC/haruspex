# Haruspex

> A small, local, **calibrated decision layer** for security operations — reuses a
> published decision encoder, adds typed verdicts and the ability to **abstain**.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Haruspex reads a situation and bounded options and returns a typed, **calibrated**
answer — and says "I'm not sure" instead of guessing when it should:

- **boolean** — yes/no + `P(yes)`
- **choice** — pick one option
- **score** — rate on an ordered rubric

It runs locally on **CPU / CUDA / Apple-MPS**, serves the Jev / TypeSafe HTTP
contract, and has **no vendor lock-in**.

## What Haruspex is (and is not)

Haruspex **reuses** the published decision encoder
[`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide)
(Apache-2.0, ~494M params, DeBERTa-family, non-autoregressive). It does **not** train a
decision model from scratch — that space is mature, and a redundant checkpoint helps
no one.

The contribution is the layer around the reused model:

1. **A typed decision contract** (boolean / choice / score) over `classify_text`.
2. **Security specialization** — a small LoRA adapter trained on your decisions
   (an example offensive+defensive data pack is included).
3. **Selective calibration** — split-conformal abstention: accept a verdict only when
   its confidence clears a threshold fitted to hold the error *among accepted items* at
   your target risk; otherwise abstain. A raw confidence is not a risk control; this is.

The artifact you ship is a **30 MB LoRA adapter + a calibrator**, reattached to the
reused base at load time — not a 2 GB copy.

## Measured (RTX 3060, 12 GB, WSL2)

**Inference.** Base GLiNER2.5-Decide, no training, on 34 security decisions:
boolean 0.62 · choice 0.55 · score 0.20 · overall **0.47**. ~2.0 GB VRAM,
28 ms median/decision.

**Held-out gain** (real train≠eval; a *partial* 250-step LoRA on ~2k public mmdm
decision rows, 240 held out):

| | accuracy | Brier | ECE |
|---|---|---|---|
| base (untrained) | 0.358 | 0.274 | 0.131 |
| + LoRA (partial) | **0.433** | 0.211 | **0.057** |

Modest but real: specialization generalizes and roughly halves calibration error. The
selective layer at target risk 0.1 then accepts only the most-confident ~2.5 % with
**zero accepted errors** — correct for a weak partial model; higher coverage needs a
fuller train.

**Security held-out — with enough data, specialization helps.** `data/security.jsonl` is
157 clean rows. On a stratified 3-way split (train 102 / cal 22 / test 33), with
hyperparameters selected on **cal** (never on test):

| on the 33-row held-out test | accuracy | ECE | Brier |
|---|---|---|---|
| base zero-shot | 0.61 | 0.13 | 0.22 |
| LoRA, tuned (5 epochs, LoRA rank 16) | **0.88** | 0.10 | 0.08 |

The calibrator is fit on the held-out **cal** set (not train), so its guarantee holds
out-of-sample: at target risk 0.2 it accepts 97 % of the test set with accepted error
0.09. The winning recipe (`--epochs 5`, with the default LoRA rank 16, calibrator on
`*_cal.jsonl`) beat more epochs (8), higher rank (32), and retraining on train+cal — so
for this pack size, train-only plus a cal-fit calibrator is the ceiling. Caveat: 33 test rows carry ~±8 %
noise and the recipe was picked on a 22-row cal set, so treat 0.88 as "best cleanly
measured here," not a population estimate; grow the pack for a tighter number. This
reverses an earlier 97-row result (training *hurt*: too little data + a 20-row test + a
train-fit calibrator); fixing all three flips it.

**Training cost — the base is DeBERTa-v3-large, whose disentangled attention is
memory-heavy** (take note before you scale up):

- Short rows (the security pack): LoRA peaks ~3.8 GB — comfortable.
- Real-scale rows (long, many-option states): **batch 4 OOMs a 12 GB card** — a single
  many-option row can need ~6 GB. The 3060 trains them only after dropping rows with
  > 8 options or > 700 chars and using **batch 2 + grad-accum + gradient checkpointing +
  `max_len` 192** → peak 2.7 GB, ~1.4 s/step.
- A comfortable full-scale run (all rows, full length, several epochs) wants a **24 GB
  GPU**, or the slow checkpointed path (~2 h / 2 epochs) on the 3060.

## Install

```sh
pip install -e .                 # inference (brings gliner2 -> torch/transformers)
pip install -e ".[train]"        # + LoRA / full fine-tune
pip install -e ".[serve]"        # + Starlette/uvicorn for the HTTP server
```

## Quickstart

```python
from haruspex import Haruspex
m = Haruspex.load(device="cuda")          # the reused base, no training
print(m.decide({"type": "boolean",
                "state": "Target: WordPress 6.2, PHP 8.1, nginx",
                "question": "Is a WordPress xmlrpc.php template relevant?",
                "criteria": {"true": "WordPress target", "false": "not WordPress"}}))
```

Specialize, calibrate, evaluate, serve:

```sh
python scripts/split_data.py data/security.jsonl --test-frac 0.2 --cal-frac 0.15   # train / cal / held-out test
python -m haruspex.train     --train data/security_train.jsonl --out runs/sec --epochs 5
# real-scale / long rows on a 12 GB GPU: add memory flags
# python -m haruspex.train   --train data/security_train.jsonl --out runs/sec \
#     --batch 2 --grad-accum 4 --grad-checkpoint --max-len 192
python -m haruspex.calibrate --model runs/sec --data data/security_cal.jsonl --target-risk 0.2
python -m haruspex.eval      --model runs/sec --data data/security_test.jsonl  # held-out: Accuracy/ECE/Brier + coverage/risk
python -m haruspex.serve     --model runs/sec --port 3000               # POST /v1/systemone · /v1/evaluate
```

Device auto-selects (CUDA → Apple-MPS → CPU); override with `--device`.

## Data

JSONL decision rows — format in [`data/schema.md`](data/schema.md). One row maps to one
GLiNER2 classification task (labels = the candidates / levels / yes-no). Bootstrap labels
from a heuristic, real outcomes, or one-shot distillation from a larger model.

## Security decision dataset

`data/security.jsonl` is the curated dataset — 157 clean, labeled decisions spanning both
sides of security work, across all three decision types (boolean / choice / score) and
14 categories. `data/security_seed.jsonl` is a smaller worked example. The model itself
is general; point it at your own decisions too.

- **Offensive:** template/probe relevance, "is this scanner hit a real finding?",
  verdict discipline (vulnerable / not-vulnerable / inconclusive), finding severity,
  test-class prioritization, CVE applicability, chain-worthiness of a low finding.
- **Defensive:** alert triage (true vs false positive), alert severity, detection-rule
  applicability to a log source, incident routing, CVE remediation urgency, phishing,
  telemetry selection.

Every row carries `side` / `category` / `id` tags. The dataset is gated clean, and a
stratified held-out split keeps a test set the model never trains on:

```sh
python scripts/validate_data.py data/security.jsonl --build   # schema + dupes + balance; exits non-zero on error
python scripts/split_data.py    data/security.jsonl --test-frac 0.2 --cal-frac 0.15   # train / cal / held-out test
```

Use `*_cal.jsonl` to fit the calibrator and keep `*_test.jsonl` for the final number —
fitting a selective threshold on train gives a false guarantee out-of-sample.

## Serving (Jev / TypeSafe HTTP contract)

Self-hosted and **keyless** (the key only authenticates a remote gateway, never your
own model):

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

Each answer carries the verdict plus `accept` / `abstain` when a calibrator is loaded.

## Publish to Hugging Face

The artifact is the adapter dir (`haruspex.json` manifest + the LoRA adapter) plus
`calibrator.json`, with a generated model card:

```sh
pip install -e ".[data]" && huggingface-cli login     # or export HF_TOKEN=...
python scripts/publish_hf.py --model runs/sec --repo <you>/haruspex
```

Load it back with `Haruspex.load("<you>/haruspex", device="cuda")`.

## Status

Early. The toolkit (specialize → calibrate → eval → serve) runs end-to-end, and a real
train→held-out-eval cycle on a 12 GB RTX 3060 shows a genuine (if partial) gain on
general data — see [Measured](#measured-rtx-3060-12-gb-wsl2). On the 157-row security pack,
tuned specialization **helps** (base 0.61 → LoRA 0.88 on the held-out split, recipe
selected on cal; calibration guarantee holds); no trained checkpoint is published yet
pending a larger, more robust evaluation. Known limits, all noted there or in code:
a 12 GB card trains only capped/filtered real-scale data (DeBERTa-v3-large is
memory-heavy); full-mode reload waits on a gliner2 save fix; prediction *sets* (beyond
abstention) wait on a gliner2 per-label-score fix. Contributions and independent
evaluations welcome — run `python -m unittest discover tests` before a PR.

## License

[Apache-2.0](LICENSE).

## Acknowledgements

The decision-model idea comes from TypeSafe's **Jev** ("System One") and the open
reproductions (**OpenJev**, **Von**, **Kev**, **NanoJev**, **Laya**). The base encoder is
**GLiNER2.5-Decide** by fastino (built on GLiNER2 / DeBERTa). Haruspex reuses that work;
its own contribution is the typed contract, the security specialization, and the
selective-calibration layer.

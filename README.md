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
python -m haruspex.train     --train data/security_seed.jsonl --out runs/sec
# real-scale / long rows on a 12 GB GPU: add memory flags
# python -m haruspex.train   --train data/train.jsonl --out runs/sec \
#     --batch 2 --grad-accum 4 --grad-checkpoint --max-len 192
python -m haruspex.calibrate --model runs/sec --data data/val.jsonl --target-risk 0.1
python -m haruspex.eval      --model runs/sec --data data/val.jsonl   # Accuracy/ECE/Brier/latency + coverage/risk
python -m haruspex.serve     --model runs/sec --port 3000             # POST /v1/systemone · /v1/evaluate
```

Device auto-selects (CUDA → Apple-MPS → CPU); override with `--device`.

## Data

JSONL decision rows — format in [`data/schema.md`](data/schema.md). One row maps to one
GLiNER2 classification task (labels = the candidates / levels / yes-no). Bootstrap labels
from a heuristic, real outcomes, or one-shot distillation from a larger model.

## Example domain pack: security operations

`data/security_seed.jsonl` is a worked example — labeled decisions for both sides of
security work. The model itself is general; point it at your own decisions.

- **Offensive:** template/probe relevance, "is this scanner hit a real finding?",
  finding severity, verdict discipline (vulnerable / not-vulnerable / inconclusive).
- **Defensive:** alert triage (true vs false positive), alert severity, detection-rule
  applicability, incident routing, CVE applicability + remediation urgency.

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
train→held-out-eval cycle on a 12 GB RTX 3060 shows a genuine (if partial) gain — see
[Measured](#measured-rtx-3060-12-gb-wsl2). Known limits, all noted there or in code:
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

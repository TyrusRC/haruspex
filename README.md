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
460 clean rows. On a stratified 3-way split (train 299 / cal 70 / test 91), with
hyperparameters selected on **cal** (never on test):

| on the 91-row held-out test | accuracy | ECE | Brier |
|---|---|---|---|
| base GLiNER2.5-Decide (zero-shot) | 0.60 | 0.12 | 0.23 |
| Llama-3.1-8B-Instruct (prompted) | 0.70 | — | — |
| **Haruspex** LoRA, tuned (8 epochs, rank 16) | **0.88** | 0.12 | 0.12 |

The calibrator is fit on the held-out **cal** set (not train), so its guarantee holds
out-of-sample: at target risk 0.2 the adapter accepts the whole test set at accepted
error 0.12. The headline: a ~494M specialized adapter **beats a prompted 8B general model
(0.88 vs 0.70) at ~1/16 the size** and comparable latency, and lifts the base encoder from
0.60. Larger sets need more epochs — at this size 8 epochs beat 5 (which underfit); rank
16 beat rank 32. Per-category, the adapter leads on severity/triage/routing/verdict; the
8B still edges it on CVE-version-range reasoning (knowledge-heavy). Caveat: 91 test rows
still carry noise and the recipe was chosen on a 70-row cal set, so treat 0.88 as "best
cleanly measured here," not a population estimate. See the comparison card in
[`docs/benchmark.md`](docs/benchmark.md).

*False-positive reduction (security goal).* The pack grew to **665 rows** (hard negatives +
agent-authored judgment cases). Two changes drove false positives toward zero on a 120-row
held-out split:

1. **Hard negatives** (patched/out-of-range CVEs, benign-but-noisy alerts, non-chainable
   lows) lifted the actionable ("yes") class to **precision 0.97 / recall 0.94**.
2. **CVE-applicability routing** (`haruspex/cve.py`). The residual false positives were
   *confident* (P≈1.0) and confined to `off_cve_applicable` — fine-grained CVE version/OS
   logic a 0.5B encoder won't learn and an abstention threshold can't catch. Those questions
   now route to a deterministic resolver that answers only when confident and otherwise
   **defers** (abstain → external KEV/NVD lookup). On the held-out set it answered CVE
   applicability at **100 % accuracy (0 wrong)**, deferring the rest — so the model emits
   **no** CVE false positives. `CveResolver` is pluggable (inject praetor's `lookup_cve`);
   the safety property is defer-on-uncertainty, so a coverage gap causes an abstention, never
   a false positive (unit-tested in `tests/test_cve.py`).

Net on the 120-row test: selective coverage 0.96, accepted accuracy 0.90, boolean precision
0.97, CVE false positives 0. The model is reserved for judgment (triage/severity/routing/
verdict), where it leads even a prompted 8B; the residual soft spot is 4-way ordinal scoring
(`off_severity`), an ordinal-accuracy limit, not a false-positive one.

*On building the pack:* the rows are hand-authored and Claude-authored (Claude acting as
a strong teacher that labels its own scenarios). Distillation from a small local teacher
(Llama-3.1-8B) was tried and **discarded** — its labels on security judgment were too
noisy and *lowered* held-out accuracy; curation fixes label distribution, not per-row
correctness. For this task, a strong teacher's label quality is what matters.

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
python -m haruspex.train     --train data/security_train.jsonl --out runs/sec --epochs 8 --seed 0
# training is seeded (reproducible; retry a bad run with a new --seed) and ends with a
# sanity-gate: a fixed discrimination probe that RAISES if the checkpoint collapsed
# (every output ~0.5) instead of shipping a degenerate model. Override with --no-sanity-gate.
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

`data/security.jsonl` is the curated dataset — 665 clean, labeled decisions spanning both
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
- `GET /v1/models` — model list (lists the served `domains`).

```
POST /v1/evaluate  (or /v1/systemone)
{ "model":"haruspex", "state": <str|obj>, "domain": "web"|"mobile"|...,
  "questions": { "<name>": { "type":"boolean"|"noul"|"choice"|"score",
                             "instructions":"...",
                             "criteria": {true,false} | {opt:desc} | [low,high] } } }
```

Each answer carries the verdict plus `accept` / `abstain` when a calibrator is loaded.
The response echoes the `domain` that answered.

**One endpoint, any domain (the router).** Serve several per-domain adapters behind the
same endpoint and let the request's `domain` field pick the adapter:

```sh
haruspex-serve --adapters web=runs/web,mobile=runs/mobile --port 3000
# a request with "domain":"mobile" is answered by the mobile adapter; "web" by the web
# model; an unknown or missing domain falls back to --default-domain (web) — fail-open,
# never an error. A single --model server ignores the field and answers from its one model.
```

The shared base is loaded once *per domain* (gliner2 does not yet expose single-base
adapter switching), so the router's win over running one server per domain is the single
endpoint and routing — a mixed-domain caller (e.g. praetor) hits one URL. Each tool sends
its own domain: a mobile scanner sends `"domain":"mobile"`, web/code tools send `"web"`.

## Publish to Hugging Face

The artifact is the adapter dir (`haruspex.json` manifest + the LoRA adapter) plus
`calibrator.json`, with a generated model card:

```sh
pip install -e ".[data]" && huggingface-cli login     # or export HF_TOKEN=...
python scripts/publish_hf.py --model runs/sec --repo <you>/haruspex
```

Load it back with `Haruspex.load("<you>/haruspex", device="cuda")`.

## Integrating Haruspex into a scanner (the useful flow)

Haruspex is a **native, standalone** decision layer — point any tool at `/v1/evaluate`
(keyless, loopback) or call `Haruspex.decide()` in-process. It does **not** depend on
praetor.

**Where it earns its place — finding triage / adjudication**, run as a **fail-open**
pass *after* a tool has produced (and ideally verified) candidates:

- the tool asks, per candidate, "is this a real, reportable finding?" with the
  **verified evidence** in the state, plus a severity score;
- Haruspex **annotates and re-ranks** — raises confidence on confirmed bugs, sinks
  noise (missing headers, version banners, `OPTIONS`/`TRACE`, empty dir-listings),
  withholds "confirmed" on a mere indicator (an error/reflection is a *lead*, not a
  finding);
- **it never removes a candidate.** A "no"/abstain means "not proven yet — keep
  hunting", never "drop it". Haruspex may block an *unproven report*, never a *lead*.
  A missed finding is the cardinal harm, so Haruspex is wired only at the report/triage
  stage — **never** at lead/probe selection or any truncation gate.

**Do not** use it to prioritize *which* targets/classes to probe: boolean "worth
testing?" saturates when everything is relevant, and the relevance signal is
phrasing-sensitive. Triage and severity are the reliable, discriminating heads.

Measured on a mixed probe set (real bugs vs noise vs safe code vs mere indicators) the
triage head separates cleanly (real → `yes`≈1.0; noise/safe/indicator → `no`≈0.0). A
live pentest A/B on an all-vulnerable, zero-noise lab API showed **no** differentiation
(nothing to rank or suppress) and a small latency cost — Haruspex's value is **conditional
on noise/false-positives/irrelevance being present**, i.e. a real scanner's raw output,
which is exactly where it plugs in.

## Per-domain adapters

One ~0.5B model cannot hold every security domain at once — merging a small mobile pack
into the web/SOC set destabilized training (it collapsed or ignored the new domain). The
architecture is therefore **one shared base + a LoRA adapter per domain**, served either
behind one routed endpoint (`--adapters web=…,mobile=…`, picked by the request `domain`) or
one endpoint per domain.

`data/mobile.jsonl` (203 rows, MASVS-aligned Android/iOS) trains a **dedicated mobile
adapter**: on its held-out split it scores **accuracy 0.98, recall 1.00** (no missed
findings) and cleanly separates real findings (hardcoded secret, debuggable, exported
component, ATS-off, WebView bridge → ~1.0) from hardened ones (Keystore, `exported=false`,
EncryptedSharedPreferences → ~0.0) — concepts the shared model could not learn.

Each tool names its domain: a mobile scanner (e.g. centurion) sends `"domain":"mobile"`,
web/code tools (assay, mantis, praetor) send `"web"`; a routed server picks the adapter and
a single-adapter server ignores the hint. (Note: the post-train sanity-gate's probe is
web-specific — pass `--gate-domain mobile` or `--no-sanity-gate` when training a non-web
adapter.)

## Status

Early. The toolkit (specialize → calibrate → eval → serve) runs end-to-end, and a real
train→held-out-eval cycle on a 12 GB RTX 3060 shows a genuine (if partial) gain on
general data — see [Measured](#measured-rtx-3060-12-gb-wsl2). On the 460-row security pack,
tuned specialization **helps** (base 0.60 → LoRA 0.88 on a 91-row held-out split, beating a prompted 8B at 0.70, recipe
selected on cal; calibration guarantee holds); no trained checkpoint is published yet
pending the operator's go. Known limits, all noted there or in code:
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

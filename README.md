# Haruspex

> A small, local, calibrated decision layer for security operations. It returns typed
> verdicts, and it abstains when it isn't sure.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Haruspex reads a situation and a bounded set of options and returns a typed answer:

- **boolean**: yes/no plus `P(yes)`
- **choice**: one option from a set
- **score**: a level on an ordered rubric

Every verdict carries the full probability distribution over its options and, once
calibrated, a prediction set: the options the true answer is among, at a chosen error rate.

It runs locally on CPU, CUDA or Apple MPS, and serves a simple HTTP decision API.

## How it works

Haruspex reuses the published decision encoder
[`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide) (Apache-2.0,
non-autoregressive). It does not train a model from scratch. It adds four things:

1. **A typed decision contract** (boolean / choice / score) over the encoder.
2. **Domain specialization.** Each security domain gets a small LoRA adapter, which is
   reattached to the shared base at load time.
3. **Deterministic routing.** Questions that are lookups rather than judgments, such as
   CVE applicability and cloud scanner findings, go to rule-based resolvers. A resolver
   answers only when the evidence pins the answer down. Otherwise it defers, so a gap in
   its rules causes an abstention, not a wrong verdict.
4. **One evidence format.** Defender, Sentinel, Elastic ECS and Windows Security/Sysmon
   records are normalised into one schema before the model reads them, so a new log
   source needs a mapper, not a new model.
5. **Selective calibration.** A verdict is accepted only when its confidence clears a
   threshold set for its decision type. That threshold is certified on a held-out
   calibration split with Learn-then-Test (an exact binomial test, Bonferroni-corrected
   over a fixed grid). With probability at least 1−δ, the error among accepted answers
   stays within your target risk. When the calibration data can't certify any threshold,
   Haruspex abstains on that decision type.
6. **Novelty gate.** An input unlike anything the adapter was trained on abstains and is
   marked for escalation, instead of getting a confident guess.

## Install

```sh
pip install -e .              # inference
pip install -e ".[train]"     # + LoRA training
pip install -e ".[serve]"     # + HTTP server
```

## Quickstart

```python
from haruspex import Haruspex

m = Haruspex.load()   # base encoder; pass an adapter dir or HF repo id to specialize
print(m.decide({"type": "boolean",
                "state": "Target: WordPress 6.2, PHP 8.1, nginx",
                "question": "Is a WordPress xmlrpc.php template relevant?",
                "criteria": {"true": "WordPress target", "false": "not WordPress"}}))
```

Device auto-selects (CUDA → MPS → CPU); override with `device=` or `--device`.

## Train, calibrate, evaluate, serve

```sh
# 1. split: train / dev / cal / test
python scripts/split_data.py data/security.jsonl --test-frac 0.2 --cal-frac 0.15 --dev-frac 0.15

# 2. train a LoRA adapter; choose epochs / rank / seed by eval on *_dev only
python -m haruspex.train --train data/security_train.jsonl --out runs/sec --epochs 8 --seed 0

# 3. fit the calibrator on the cal split (temperature on dev; threshold + prediction sets on cal)
python -m haruspex.calibrate --model runs/sec --data data/security_cal.jsonl \
    --target-risk 0.2 --delta 0.1 --temp-data data/security_dev.jsonl

# 3b. fit the novelty gate (abstain on unfamiliar inputs)
python -m haruspex.ood --model runs/sec --train data/security_train.jsonl \
    --cal data/security_cal.jsonl --alpha 0.01

# 4. report the final number once, on test
python -m haruspex.eval --model runs/sec --data data/security_test.jsonl

# 5. serve
python -m haruspex.serve --model runs/sec --port 3000
```

Training ends with a sanity gate: a fixed discrimination probe that refuses to save a
collapsed checkpoint (one where every output is about 0.5). Retry with another `--seed`
if it fires. The default learning rate is deliberately low; higher rates collapse on some
seeds. On smaller GPUs, add `--batch 2 --grad-accum 4 --grad-checkpoint --max-len 192`.

### Seed ensembles

Confident mistakes are the main failure of a single adapter. Train the same recipe with a
few `--seed` values and list them in one manifest; they share one base model, and their
disagreement lowers confidence where a single adapter would be sure and wrong:

```json
{"base": "fastino/GLiNER2.5-Decide", "adapters": ["m0", "m1", "m2"]}
```

(`m0`…`m2` are the `final/` directories of the seed runs.) Calibrate the ensemble directory
like any other adapter. Each verdict then also reports `agreement` across members.

## Data

Decision rows are JSONL, one decision per line; the format is in
[`data/schema.md`](data/schema.md). Validate a file before training:

```sh
python scripts/validate_data.py data/security.jsonl --build
```

Included packs: `security.jsonl` (offensive and defensive: scanner adjudication, triage,
severity, routing, CVE applicability, phishing, detection rules), plus `mobile`, `cloud`,
`network` and `llm`.

Converters for public datasets (their output is not committed):

| script | source | decision |
|---|---|---|
| `convert_guide.py` | Microsoft GUIDE (Defender XDR incidents) | incident triage grade |
| `convert_otrf.py` | OTRF Security-Datasets (Windows Security / Sysmon) | ATT&CK tactic of attacker events |
| `convert_nvd.py` | NVD (after `fetch_nvd.py`) | CVSS v3 severity from a description |
| `convert_owasp_benchmark.py` | OWASP Benchmark (GPL-2.0) | external test only, never trained on |

Splits are content-hashed: a row's split depends only on the row itself, so adding data
never moves a training row into test. Each split has one job:

| split | used for |
|---|---|
| `train` | fitting the adapter |
| `dev` | choosing hyperparameters and temperature |
| `cal` | fitting the calibrator, and nothing else |
| `test` | the final reported number |

## Log sources

A `state` may be raw JSON from a supported source; it is normalised and rendered before
the model reads it (`haruspex/evidence.py`):

- Microsoft Defender XDR alerts (Graph security `alerts_v2`)
- Microsoft Sentinel `SecurityAlert` rows
- Elastic Common Schema documents
- Windows Security and Sysmon events, flat or `evtx_dump`-style nested
- a list of any of these, read as one event window

Other JSON is passed through as-is; the novelty gate then decides whether the adapter
should answer at all. Long states are read in overlapping windows, and a verdict stands
only when every window agrees.

## HTTP API

```
POST /v1/evaluate        (also /v1/systemone; type "noul" is accepted as boolean)
{ "state": <str|obj>, "domain": "web"|"mobile"|...,
  "questions": { "<name>": { "type": "boolean"|"choice"|"score",
                             "instructions": "...",
                             "criteria": {"true":..,"false":..} | {opt: desc} | [levels] } } }
```

Each answer carries its verdict, the full `probabilities`, `accept` / `abstain` and a
`prediction_set` (when a calibrator is loaded), and a `source`. All questions in one
request are scored as a single batch.

| source | meaning |
|---|---|
| `model` | the encoder answered |
| `cve_lookup` | the CVE resolver answered |
| `cloud_rule` | the cloud resolver answered |
| `model_declined` | abstained; route the question to an external lookup |

Other endpoints are `GET /health`, `GET /v1/models` and `GET /metrics`.

Limits: the body is capped at 512 KiB (checked before parsing), `state` at 64 K
characters, and a request at 64 questions. Malformed input returns `400`; a request over
any limit returns `413`.

Optional settings:

- `HARUSPEX_API_KEY` requires a bearer key, compared in constant time.
- `HARUSPEX_AUDIT_LOG` writes one JSONL line per request; the raw `state` is never logged.
- `--ssl-keyfile` / `--ssl-certfile` serve HTTPS.

## Agents (MCP)

`haruspex-mcp` (`pip install -e ".[mcp]"`) is a stdio MCP server with `decide` and
`list_domains` tools, sharing the HTTP server's routing and limits. Every result carries
`escalate` and an `action`: when Haruspex abstains, the agent is told to hand the question
to a stronger model or a human.

```json
{"mcpServers": {"haruspex": {"command": "/path/to/.venv/bin/haruspex-mcp",
  "args": ["--adapters", "web=/abs/runs/sec,soc=/abs/runs/soc"]}}}
```

## Per-domain adapters

Each domain gets its own adapter on the shared base, served behind one endpoint. The
request's `domain` field picks the adapter. When `domain` is omitted, the domain is
inferred from the request text; otherwise the default adapter answers.

```sh
haruspex-serve --adapters web=runs/sec,mobile=runs/mobile,cloud=runs/cloud --port 3000
haruspex-serve --adapters web=TyrusRC/haruspex,mobile=TyrusRC/haruspex#mobile  # from the Hub
```

`repo#subfolder` loads one adapter from a multi-domain repository. A `general` adapter
trained on every domain together is the fallback when no domain fits.

## Benchmarking

`scripts/bench.py` runs Haruspex and any OpenAI-compatible model on the same rows. The
other model sees the same option descriptions, decodes greedily with output constrained
to the valid labels, and can take few-shot examples drawn from the train split. The
script reports accuracy with 95 % intervals, per-category accuracy, and an exact paired
McNemar test. It also runs the cascade — Haruspex answers what it accepts, the LLM gets
only the abstentions — and reports how many LLM calls that leaves.

```sh
python scripts/bench.py --data data/security_test.jsonl --model runs/sec \
    --llm-url http://127.0.0.1:8088 --shots 3 --shots-from data/security_train.jsonl
```

Results are recorded in [`docs/benchmark.md`](docs/benchmark.md).

## Integration guidance

The intended shape is a cascade: Haruspex as the fast, calibrated first pass, and a general
LLM or a human for whatever it declines. Use Haruspex after a tool has produced candidates, to triage and rank them. For example,
for each finding, ask "is this real and reportable?" and "how severe?". Treat a `no` or
an abstention as "not proven yet", never as "drop it". Don't use it to gate which targets
get tested.

## Limitations

- It decides bounded questions; open-ended judgment belongs to a general model.
- Recalibrate on your own data before relying on the risk guarantee in a new domain.
- Full fine-tuning is disabled until gliner2 can reload its full-mode checkpoints.
- Prediction-set coverage is marginal over the calibration mix, not guaranteed per input.
- Ensembles run their members one after another, so latency grows with their size.
- SOC triage and Windows event logs are covered; other DFIR artefacts (memory, disk,
  timelines) are not yet.
- The novelty gate compares surface text, not meaning: a familiar format with new content
  may pass it.
- A decision that needs facts from two distant parts of a long log at once is beyond the
  windowed reader; escalate those.

## Contributing

Run `python -m unittest discover tests` before opening a pull request.

## License

[Apache-2.0](LICENSE).

## Acknowledgements

The decision-model framing follows TypeSafe's Jev ("System One") and its open
reproductions. The base encoder is GLiNER2.5-Decide by fastino, built on GLiNER2 and
DeBERTa.

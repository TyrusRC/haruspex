"""Generate a Hugging Face model card (README.md) for a specialized Haruspex model."""

from __future__ import annotations

import json
from pathlib import Path

_TEMPLATE = """---
license: apache-2.0
library_name: gliner2
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - selective-prediction
  - security
  - pentest
  - soc
base_model: {base}
---

# {repo}

**Haruspex** is a small, local **decision layer** for security operations. It reads a
situation and returns a *calibrated* typed verdict — **boolean** (yes/no + probability),
**choice** (pick one option), **score** (rate on a rubric) — and **abstains** when it is
not confident enough for your risk budget.

Haruspex **reuses** the published decision encoder
[`fastino/GLiNER2.5-Decide`]({base_url}) (Apache-2.0, non-autoregressive, CPU/GPU) and
specializes it on security decisions. It does not train a decision model from scratch.

- Base: `{base}`.
- Added: a typed decision contract, security specialization data, and a
  split-conformal **selective-calibration** layer (accept/abstain against a risk budget).
- Serves the TypeSafe `POST /v1/systemone` and native `/v1/evaluate` contracts.

## Intended use (security, both sides)

- **Offensive (pentest / red-team):** is this template/probe relevant to the target?
  is this scanner hit a real finding? finding severity; verdict discipline
  (vulnerable / not-vulnerable / inconclusive); attack-path next step.
- **Defensive (SOC / blue-team):** alert triage (true vs false positive), alert
  severity/priority, detection-rule applicability, incident routing, CVE applicability.

It is a classifier for bounded decisions — not a chatbot, not an exploit generator.

## Use

```python
from haruspex import Haruspex
m = Haruspex.load("{repo}", device="cuda")   # or "mps" / "cpu"
print(m.decide({{"type":"boolean","state":"Target: WordPress 6.2, PHP 8.1",
                 "question":"Is a WordPress xmlrpc.php template relevant?",
                 "criteria":{{"true":"WordPress target","false":"not WordPress"}}}}))
```

Or serve the decision API (keyless, self-hosted):

```sh
pip install "haruspex[serve] @ git+https://github.com/TyrusRC/haruspex"
haruspex-serve --model {repo} --port 3000      # POST /v1/systemone  ·  /v1/evaluate
```

## Calibration{cal}

A split-conformal selective layer gates each verdict: it accepts when the top-1
confidence clears a threshold fitted to hold the error AMONG accepted items at a target
risk, and abstains otherwise. Fit it with `python -m haruspex.calibrate`; evaluate
coverage/risk with `python -m haruspex.eval`.

## Limitations

- A specialist for the decision types above; novel free-form judgments belong to a
  general LLM.
- Calibrated on the training mix — recalibrate for a very different domain.
- Prediction SETS (not just abstention) await a GLiNER2 fix for per-label scores.

Apache-2.0. Repo + recipe: https://github.com/TyrusRC/haruspex
"""


def _base_model(model_dir: Path) -> str:
    cfg = model_dir / "config.json"
    if cfg.exists():
        try:
            d = json.loads(cfg.read_text())
            return d.get("_name_or_path") or d.get("base_model") or "fastino/GLiNER2.5-Decide"
        except (ValueError, OSError):
            pass
    return "fastino/GLiNER2.5-Decide"


def build_card(model_dir: str | Path, repo: str) -> str:
    model_dir = Path(model_dir)
    base = _base_model(model_dir)
    cal_file = model_dir / "calibrator.json"
    cal = ""
    if cal_file.exists():
        try:
            c = json.loads(cal_file.read_text())
            cal = (f"\n\nFitted calibrator: accept threshold `{c.get('threshold')}`, "
                   f"target risk `{c.get('target_risk')}`, coverage `{c.get('coverage')}` "
                   f"(n={c.get('n')}).")
        except (ValueError, OSError):
            pass
    return _TEMPLATE.format(repo=repo, base=base,
                            base_url="https://huggingface.co/fastino/GLiNER2.5-Decide",
                            cal=cal)


def write_card(model_dir: str | Path, repo: str) -> Path:
    out = Path(model_dir) / "README.md"
    out.write_text(build_card(model_dir, repo), encoding="utf-8")
    return out

"""Generate a Hugging Face model card (README.md) for a trained Haruspex model."""

from __future__ import annotations

import json
from pathlib import Path

_TEMPLATE = """---
license: apache-2.0
library_name: haruspex
pipeline_tag: text-classification
tags:
  - decision-model
  - jev
  - calibration
  - security
  - pentest
  - soc
base_model: {base}
---

# {repo}

**Haruspex** is a small, open-weight **decision model**: it reads a situation and
returns *calibrated* typed verdicts — **boolean** (yes/no + probability),
**choice** (pick one option), **score** (rate on a rubric) — instead of free text.
It is optimized for **security operations, offensive and defensive**.

- Backbone: `{base}` ({btype}).
- Scoring: per-candidate `Linear(h,1)` head → temperature + isotonic calibration.
- Serves the TypeSafe `POST /v1/systemone` contract and native `/v1/evaluate`.

## Intended use (security, both sides)

- **Offensive (pentest / red-team):** is this template/probe relevant to the target?
  is this scanner hit a real finding? finding severity; verdict discipline
  (vulnerable / not-vulnerable / inconclusive); attack-path next step.
- **Defensive (SOC / blue-team):** alert triage (true vs false positive), alert
  severity/priority, detection-rule applicability to a log source, incident routing,
  CVE applicability + remediation urgency.

It is a classifier for bounded decisions — not a chatbot, not an exploit generator.

## Use

```python
from haruspex import HaruspexModel
m = HaruspexModel.load("{repo}", device="cuda")   # or "mps" / "cpu"
print(m.decide({{"type":"boolean","state":"OpenSSH 9.6 on Ubuntu 22.04",
                 "question":"Affected by CVE-2024-6387 (OpenSSH <9.6)?"}}))
```

Or serve the decision API (keyless, self-hosted):

```sh
pip install "haruspex[serve] @ git+https://github.com/TyrusRC/haruspex"
haruspex-serve --model {repo} --port 3000      # POST /v1/systemone  ·  /v1/evaluate
```

Loading needs the `haruspex` package (the custom scoring head lives in `head.pt` +
`meta.json` beside the backbone weights) — a plain `AutoModel` load gives only the
encoder, not the decision head.

## Calibration

Trained with NLL + Brier, then post-hoc **temperature** scaling on a held-out split
and **isotonic** recalibration of boolean P(yes). Report Accuracy + Brier + ECE
(`python -m haruspex.eval`).

## Limitations

- A specialist for the decision types above; novel free-form judgments belong to a
  general LLM.
- Calibrated on the training mix — recalibrate for a very different domain.

Apache-2.0. Repo + recipe: https://github.com/TyrusRC/haruspex
"""


def build_card(model_dir: str | Path, repo: str) -> str:
    meta = json.loads((Path(model_dir) / "meta.json").read_text())
    return _TEMPLATE.format(repo=repo, base=meta.get("backbone", "unknown"),
                            btype=meta.get("type", "encoder"))


def write_card(model_dir: str | Path, repo: str) -> Path:
    out = Path(model_dir) / "README.md"
    out.write_text(build_card(model_dir, repo), encoding="utf-8")
    return out

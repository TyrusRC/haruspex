"""Generate a Hugging Face model card (README.md) for a specialized Haruspex model."""

from __future__ import annotations

import json
from pathlib import Path

_TEMPLATE = """---
license: apache-2.0
library_name: gliner2
pipeline_tag: text-classification
language:
  - en
tags:
  - decision-model
  - calibration
  - selective-prediction
  - conformal-prediction
  - abstention
  - risk-aware
  - security
  - cybersecurity
  - security-operations
  - soc
  - pentest
  - red-team
  - blue-team
  - gliner2
base_model: {base}
metrics:
  - accuracy
{model_index}---

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


def _eval_blocks(model_dir: Path, repo: str):
    """Return (model_index_yaml, eval_section_md) from an optional eval.json."""
    ef = model_dir / "eval.json"
    if not ef.exists():
        return "", ""
    try:
        e = json.loads(ef.read_text())
    except (ValueError, OSError):
        return "", ""
    name = repo.split("/")[-1]
    ds = e.get("dataset", "security held-out")
    n = e.get("test_n")
    acc = e.get("accuracy")
    mlines = [f"            value: {acc}", "            name: accuracy"]
    mi = (
        "model-index:\n"
        f"  - name: {name}\n"
        "    results:\n"
        "      - task:\n"
        "          type: text-classification\n"
        "          name: Security decision (boolean / choice / score)\n"
        "        dataset:\n"
        "          type: custom\n"
        f"          name: {ds}" + (f" ({n} rows)" if n else "") + "\n"
        "        metrics:\n"
        "          - type: accuracy\n" + "\n".join(mlines) + "\n"
    )
    extra = "".join(
        f"          - type: {k}\n            value: {e[k]}\n            name: {k.upper()}\n"
        for k in ("ece", "brier") if k in e
    )
    mi += extra
    sec = (f"\n\n## Evaluation\n\nOn a {n}-row held-out split of the security pack "
           f"(recipe selected on a separate calibration split, never on test): "
           f"**accuracy {acc}**")
    if "ece" in e:
        sec += f", ECE {e['ece']}"
    if "brier" in e:
        sec += f", Brier {e['brier']}"
    sec += (". The adapter beats a prompted general 8B model on the same set at a small "
            "fraction of the parameters; see the repo's `docs/benchmark.md` comparison card.")
    return mi, sec


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
    model_index, eval_section = _eval_blocks(model_dir, repo)
    return _TEMPLATE.format(repo=repo, base=base, model_index=model_index,
                            base_url="https://huggingface.co/fastino/GLiNER2.5-Decide",
                            cal=cal + eval_section)


def write_card(model_dir: str | Path, repo: str) -> Path:
    out = Path(model_dir) / "README.md"
    out.write_text(build_card(model_dir, repo), encoding="utf-8")
    return out

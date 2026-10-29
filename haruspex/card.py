"""Generate a Hugging Face model card (README.md) for a specialized Haruspex model."""

from __future__ import annotations

import json
from pathlib import Path

_TEMPLATE = """---
license: apache-2.0
library_name: gliner2
pipeline_tag: text-classification
inference: false
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
- Added: a typed decision contract, security specialization data, a certified
  **selective-calibration** layer (accept/abstain against a risk budget), conformal
  **prediction sets**, and a **novelty gate** that abstains on unfamiliar input.
- Serves the TypeSafe `POST /v1/systemone` and native `/v1/evaluate` contracts.

{index}## Intended use (security, both sides)

- **Offensive (pentest / red-team):** is this template/probe relevant to the target?
  is this scanner hit a real finding? finding severity; verdict discipline
  (vulnerable / not-vulnerable / inconclusive); attack-path next step.
- **Defensive (SOC / blue-team):** alert and incident triage (true / benign / false
  positive), alert severity/priority, detection-rule applicability, incident routing, CVE
  applicability, ATT&CK tactic of Windows/Sysmon events. Raw Defender, Sentinel, Elastic
  ECS, Windows, CloudTrail, Suricata, Zeek and Okta JSON is normalised before the model
  reads it (adapters are trained on the Microsoft/Windows sources only).

It is a classifier for bounded decisions — not a chatbot, not an exploit generator.

## Use

```python
from haruspex import Haruspex
m = Haruspex.load({load_args}, device="cuda")   # or "mps" / "cpu"
print(m.decide({{"type":"boolean","state":"Target: WordPress 6.2, PHP 8.1",
                 "question":"Is a WordPress xmlrpc.php template relevant?",
                 "criteria":{{"true":"WordPress target","false":"not WordPress"}}}}))
```

Or serve the decision API (keyless, self-hosted):

```sh
pip install "haruspex[serve] @ git+https://github.com/TyrusRC/haruspex"
haruspex-serve {serve_args} --port 3000      # POST /v1/systemone  ·  /v1/evaluate
```

## Calibration{cal}

Each verdict carries the full probability distribution over its options (temperature-
scaled). A Learn-then-Test selective layer accepts a verdict only when its confidence
clears a threshold certified on a held-out calibration split, so that with probability
1 - delta the error among accepted verdicts stays within the target risk; otherwise it
abstains with `"route": "escalate"`. A conformal prediction set lists the options the
true answer is among, and a novelty gate abstains on inputs unlike the training data.
Fit with `python -m haruspex.calibrate` and `python -m haruspex.ood`; evaluate with
`python -m haruspex.eval`.{eval}

## Limitations

- A specialist for the decision types above; novel free-form judgments belong to a
  general LLM.
- Calibrated on this domain's held-out split; recalibrate before relying on the
  guarantee on your own data.
- Prediction-set coverage is marginal over the calibration mix, not per input.
- A calibrator fitted on a mix of domains holds its risk target on average over that
  mix, not on each domain: check each test set's accepted accuracy under Evaluation,
  and route a domain with its own adapter to that adapter.

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
    """Return (model_index_yaml, eval_section_md) from every eval*.json in the dir
    (`python -m haruspex.eval --out`), one result per held-out test set."""
    evals = []
    for ef in sorted(model_dir.glob("eval*.json")):
        try:
            evals.append(json.loads(ef.read_text()))
        except (ValueError, OSError):
            continue
    if not evals:
        return "", ""
    results, lines = [], []
    for e in evals:
        ds = e.get("dataset", "held-out test split")
        n = e.get("test_n")
        label = ds + (f" ({n} rows)" if n else "")
        metrics = "".join(f"          - type: {k}\n            value: {e[k]}\n            name: {k}\n"
                          for k in ("accuracy", "ece", "brier") if k in e)
        results.append("      - task:\n          type: text-classification\n"
                       "          name: Security decision (boolean / choice / score)\n"
                       f"        dataset:\n          type: custom\n          name: {label}\n"
                       f"        metrics:\n{metrics}")
        line = f"- **{label}**: accuracy {e.get('accuracy')}"
        if "ece" in e:
            line += f", ECE {e['ece']}"
        if "brier" in e:
            line += f", Brier {e['brier']}"
        sel = e.get("selective") or {}
        if sel:
            line += (f"; with the calibrator it answers {sel.get('coverage')} of rows at "
                     f"accuracy {sel.get('accepted_accuracy')} and abstains on the rest")
        lines.append(line + ".")
    mi = f"model-index:\n  - name: {repo.split('/')[-1]}\n    results:\n" + "".join(results)
    sec = ("\n\n## Evaluation\n\nHeld-out test sets, never used for training, model "
           "selection or calibration:\n\n" + "\n".join(lines) +
           "\n\nComparisons against general LLMs are in the repo's `docs/benchmark.md`.")
    return mi, sec


def _index_section(repo: str, index: dict) -> str:
    """Root-card table of every adapter in the repo, from each dir's eval*.json."""
    rows = []
    for sub, d in index.items():
        for ef in sorted(Path(d).glob("eval*.json")):
            try:
                e = json.loads(ef.read_text())
            except (ValueError, OSError):
                continue
            sel = e.get("selective") or {}
            ans = (f"{round(sel['coverage'] * 100)} % at {sel['accepted_accuracy']}"
                   if sel.get("coverage") is not None else "-")
            load = f"`{repo}`" if sub in ("", "web") else f'`{repo}#{sub}`'
            rows.append(f"| {load} | {e.get('dataset', '')} | {e.get('test_n', '')} | "
                        f"{e.get('accuracy')} | {ans} |")
    if not rows:
        return ""
    return ("## Adapters in this repo\n\nEach is a LoRA adapter on the same base, with its own "
            "calibrator and novelty gate. Held-out test results (answers = share accepted by the "
            "calibrator, and accuracy on those):\n\n| load as | test set | n | accuracy | answers |\n"
            "|---|---|---|---|---|\n" + "\n".join(rows) +
            "\n\nAgainst Llama-3.1-8B-Instruct (3-shot, constrained output) on the same test "
            "sets: no significant difference on the security pack, where the cascade (Haruspex "
            "first, the LLM only on abstentions) matches the 8B with one LLM call in six; on "
            "Microsoft GUIDE incident grading Haruspex scores 0.565 against the 8B's 0.272. "
            "Out-of-distribution the larger LLM is the more robust detector: on independent "
            "prompt-injection sets it holds 0.82–0.93 across two sources, and a 0.5B fine-tuned "
            "on one of them overfits that benchmark rather than learning the task, so no "
            "prompt-injection specialist is shipped. "
            "Details: the repo's `docs/benchmark.md`.\n\n")


def build_card(model_dir: str | Path, repo: str, subfolder: str = "",
               index: dict | None = None) -> str:
    model_dir = Path(model_dir)
    base = _base_model(model_dir)
    cal_file = model_dir / "calibrator.json"
    cal = ""
    if cal_file.exists():
        try:
            c = json.loads(cal_file.read_text())
            thr = ({t: f.get("threshold") for t, f in c["by_type"].items()}
                   if c.get("by_type") else c.get("threshold"))
            cal = (f"\n\nFitted calibrator: accept threshold `{thr}`, target risk "
                   f"`{c.get('target_risk')}` at delta `{c.get('delta')}`, coverage on the "
                   f"calibration split `{c.get('coverage')}` (n={c.get('n')}), temperature "
                   f"`{round(c.get('temperature', 1.0), 3)}`.")
        except (ValueError, OSError):
            pass
    model_index, eval_section = _eval_blocks(model_dir, repo)
    load_args = f'"{repo}", subfolder="{subfolder}"' if subfolder else f'"{repo}"'
    serve_args = f"--adapters {subfolder}={repo}#{subfolder}" if subfolder else f"--model {repo}"
    title = f"{repo} ({subfolder})" if subfolder else repo
    return _TEMPLATE.format(repo=title, base=base, model_index=model_index,
                            index=_index_section(repo, index) if index else "",
                            load_args=load_args, serve_args=serve_args,
                            base_url="https://huggingface.co/fastino/GLiNER2.5-Decide",
                            cal=cal, eval=eval_section)


def write_card(model_dir: str | Path, repo: str, subfolder: str = "",
               index: dict | None = None) -> Path:
    out = Path(model_dir) / "README.md"
    out.write_text(build_card(model_dir, repo, subfolder, index), encoding="utf-8")
    return out

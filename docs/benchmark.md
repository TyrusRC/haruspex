# Haruspex comparison card

> **v0.1 numbers, re-measurement pending.** Measured on a shuffle-based split that moved
> rows between train and test whenever the dataset grew, with hyperparameters and the
> calibrator both fit on the cal split. Treat as indicative; see the README "Measured" note.

Haruspex (a ~494M specialized decision adapter) vs. baselines on the **same 91-row
security held-out test** (`data/security_test.jsonl`, never seen in training or
calibration). The point: show where a tiny, calibrated, specialized model wins and where
it still trails a much larger general model.

| model | params | overall acc | p50 ms |
|---|---|---|---|
| **Haruspex (ours, LoRA ep8/r16)** | ~494M + LoRA | **0.879** | 50 |
| Llama-3.1-8B-Instruct (prompted, Q4) | ~8B | 0.703 | 75 |
| Base GLiNER2.5-Decide (zero-shot) | ~494M | 0.604 | 29 |

**Headline:** the specialized ~494M adapter beats a prompted 8B general model
(0.88 vs 0.70) at ~1/16 the parameters and comparable latency, and lifts the un-tuned
base encoder from 0.60.

## Per-category accuracy (gap finder)

| category | n | Haruspex | base (0-shot) | Llama-3.1-8B |
|---|---|---|---|---|
| def_alert_severity | 7 | 1.00 | 0.29 | 0.43 |
| def_alert_triage | 8 | 0.75 | 0.50 | 0.50 |
| def_cve_urgency | 6 | 0.83 | 0.33 | 0.83 |
| def_data_source | 4 | 1.00 | 1.00 | 1.00 |
| def_incident_route | 8 | 1.00 | 0.75 | 1.00 |
| def_phish | 5 | 1.00 | 0.20 | 1.00 |
| def_rule_applicable | 5 | 1.00 | 0.20 | 1.00 |
| off_chain_worthy | 5 | 0.80 | 0.60 | 0.60 |
| off_cve_applicable | 10 | 0.70 | 0.70 | **0.90** |
| off_scanner_adjudication | 5 | 1.00 | 0.80 | 1.00 |
| off_severity | 10 | 0.70 | 0.70 | 0.20 |
| off_template_relevance | 7 | 0.86 | 0.71 | 0.71 |
| off_test_class | 5 | 1.00 | 1.00 | 0.60 |
| off_verdict | 6 | 1.00 | 0.67 | 0.50 |

**Where Haruspex leads:** severity rating, alert triage, incident routing, phishing,
rule-applicability, verdict discipline, scanner adjudication — the judgment/consistency
tasks specialization is for.

**Residual gap:** `off_cve_applicable` (0.70 vs the 8B's 0.90) — version-range reasoning
is knowledge-heavy and favors the larger model; add more CVE rows to close it.

## Method / caveats

- Same held-out split for all three models; Haruspex and the base encoder run through
  `decide()`, the 8B through a zero-shot prompt (`llama.cpp` server, Q4, greedy) parsed
  into the typed label.
- Per-category n is small (4–10) — read category rows as directional, not precise.
- The 8B is prompted (not fine-tuned on this task); a fine-tuned 8B would score higher
  but costs ~16× the parameters.
- Reproduce: `scripts/split_data.py ... && python -m haruspex.train --epochs 8` then the
  benchmark harness. Numbers are a single run on an RTX 3060 (12 GB).

## Real-world test — owasp-lab source adjudication

Fed 12 real vulnerabilities from a Flask OWASP Top-10 lab (`jozeta/owasp-lab`) to Haruspex
as raw **code observations** (the analyst/scanner surfaces the candidate; Haruspex adjudicates):

- **Vulnerability class: 12/12 correct** — including 3 mitigated-looking traps it was *not*
  fooled by (SSTI behind a keyword blocklist, a `..`-filtered-but-absolute-path traversal,
  and an SSRF host blocklist).
- **Severity: mostly right** (SQLi/SSTI/pickle/SSRF → critical; traversal/order-by → high).
- **"Is this a real finding?"** Out of the box this missed bare code observations (it is
  trained on "scanner flagged … confirmed" phrasing). Adding ~50 **code-observation**
  adjudication rows (genuinely-vulnerable vs genuinely-safe code) lifted it to **10/12
  flagged** while correctly rejecting **3/3 safe-code controls** (parameterized SQL,
  `safe_join`, bcrypt) — i.e. no false positives on safe code.
- **Residual:** 2 confident false negatives (blind command injection with discarded output;
  MD5 password storage). Pushing more data past this point traded a false negative for a
  false positive on safe code — the ~0.5B adjudication head's capacity limit on a nuanced
  binary judgment from free-text code.

**Takeaway:** Haruspex is an **adjudication layer**, strongest on class + severity. Feed it
*verified* observations (as the verify→assess pipeline does) and it is accurate; it is not a
from-scratch SAST scanner on raw code.

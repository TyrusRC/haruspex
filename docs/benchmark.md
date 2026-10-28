# Haruspex comparison card (v0.3)

Haruspex adapters vs. Llama-3.1-8B-Instruct (Q4, llama.cpp, greedy, output constrained to
the valid labels, 3 worked examples drawn from the train split) on held-out test splits.
Each test split was run once, after model selection on dev and calibration on cal.
Reproduce with `scripts/bench.py` (see the README).

## Results

| test set | n | Haruspex | Llama-3.1-8B 3-shot | paired McNemar p | cascade | LLM calls in cascade |
|---|---|---|---|---|---|---|
| security pack | 188 | 0.878 [0.823, 0.917] | 0.904 [0.854, 0.939] | 0.47 | 0.904 | 16 % |
| Microsoft GUIDE incidents (3-way grade) | 1500 | **0.565** [0.540, 0.590] | 0.272 [0.250, 0.295] | 8e-54 | 0.367 | 82 % |

Median latency per decision on one RTX 3060: Haruspex 34–43 ms, the 8B 170–203 ms.

- **Security pack:** no significant difference between the 8B and Haruspex. The cascade
  (Haruspex answers what its calibrator accepts, the LLM gets only the abstentions)
  matches the 8B exactly while sending it one call in six.
- **GUIDE:** Haruspex roughly doubles the 8B, which scores below chance (0.333) on SOC
  incident grading. Here the cascade *hurts*: the 8B is worse than Haruspex on the cases
  Haruspex declines, so abstentions on this task belong with a human or a much stronger
  model.

## Selective accuracy (calibrator on)

| adapter | test set | answers | accuracy when answering |
|---|---|---|---|
| web | security pack | 84 % | 0.931 |
| soc | GUIDE | 18 % | 0.872 |
| mobile / cloud / network / llm | own packs | 100 % | 0.939 / 0.871 / 0.971 / 0.985 |
| general (pooled) | security pack | 88 % | 0.888 |
| general (pooled) | GUIDE | 32 % | 0.706 (above its 0.2 risk target: pooled calibration holds on the mix, not per domain) |

Per-category accuracy on the security pack (n per category is small; treat as a gap finder):

| category | n | Haruspex | 8B 3-shot |
|---|---|---|---|
| def_alert_severity | 9 | 0.67 | 0.56 |
| def_alert_triage | 11 | 0.91 | 1.00 |
| def_cve_urgency | 7 | 0.57 | 1.00 |
| def_data_source | 5 | 1.00 | 1.00 |
| def_incident_route | 8 | 0.75 | 1.00 |
| def_phish | 7 | 0.86 | 1.00 |
| def_rule_applicable | 5 | 1.00 | 1.00 |
| off_chain_worthy | 10 | 0.80 | 1.00 |
| off_cve_applicable | 15 | 0.87 | 0.93 |
| off_scanner_adjudication | 69 | 0.96 | 0.90 |
| off_severity | 17 | 0.76 | 0.71 |
| off_template_relevance | 8 | 0.88 | 1.00 |
| off_test_class | 9 | 0.89 | 1.00 |
| off_verdict | 8 | 1.00 | 0.88 |

## Caveats

- The security pack is this project's own data; GUIDE (Microsoft, analyst-graded) is the
  independent one. GUIDE's ceiling with tenant-independent fields is about 0.80, since
  about 20 % of incidents share a summary with differently graded ones.
- One 8B model and one prompt format; larger LLMs were not run.
- OTRF Windows-log test (35 rows, 3 tactics) is too small to report as a benchmark.

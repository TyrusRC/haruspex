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

## Independent datasets (never trained on)

A score on a split of our own data can hide overfitting, so each adapter is also run on a
public dataset it never saw in training. These are the honest numbers.

| adapter | independent set | n | accuracy | notes |
|---|---|---|---|---|
| `llm` (zero-shot transfer) | [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections) | 116 | 0.67 | adapter never trained on the prompt-injection task; real signal above chance, but accepted risk ~0.33 (calibration fit elsewhere does not hold here) |
| `llm-v2` (fine-tuned on deepset-train) | `deepset/prompt-injections` | 116 | 0.966 | **overfits deepset** — see below; the same adapter is 0.59 on an independent PI source, so this is not a general capability |
| `llm-v2` (same adapter) | [`xTRam1/safe-guard-prompt-injection`](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) | 2060 | **0.59** | below the 0.684 majority baseline; the 8B scores 0.925 here — fine-tuning on one PI benchmark did not transfer to another |
| `web` (generalist) | `deepset/prompt-injections` | 116 | 0.41 | correctly abstains on 76 % — it defers rather than guess outside its domain |
| `code` | [OWASP Benchmark](https://owasp.org/www-project-benchmark/) | 600 | **0.50** | at chance — see below |
| `code` | PrimeVul (real C/C++) | 303 | **0.50** | at chance |

**Code-level vulnerability detection does not work at this model scale.** A code adapter
trained on NIST Juliet reaches 0.71 on a Juliet split but collapses to chance on both
independent code sets (OWASP 0.50, PrimeVul 0.50); training longer only reduces the Juliet
overfit (0.71 → 0.54) without moving the independent score. Llama-3.1-8B also scores ~0.53
on OWASP, so static taint analysis from a code snippet is hard for this class of model, not
just for Haruspex. **No code adapter is published**; the converters
(`convert_juliet.py`, `convert_owasp_benchmark.py`, `convert_primevul.py`) remain so the
result is reproducible and so a future, larger backend can be measured the same way.

The lesson shapes what *is* shipped: an adapter earns a place only when it holds up on
independent data (like `llm`), and the selective calibrator refuses to accept answers it
cannot stand behind (the code adapter's calibrator set no threshold — coverage 0).

### Head-to-head vs a larger LLM

**In-distribution (security triage), the cascade matches the 8B at ~6× less LLM compute.**
On the 188-row security test set:

| approach | accuracy | LLM calls | latency (p50) |
|---|---|---|---|
| Haruspex `web` alone | 0.878 | 0 | 34 ms |
| Llama-3.1-8B-Instruct (Q4, 3-shot) | 0.904 | 188/188 | 166 ms |
| **Haruspex → 8B cascade** | **0.904** | **30/188 (16 %)** | 34 ms (local) / 166 ms (escalated) |

Haruspex alone is statistically tied with the 8B (McNemar p = 0.47). The cascade — Haruspex
first, escalate only the calibrated-abstention cases — reaches the **8B's exact accuracy**
(p = 1.0 vs the 8B) while sending **16 %** of items to the LLM. The other 84 % are decided
locally at 34 ms. That is the defensible "beat a model that uses more resource" result:
equal accuracy at roughly one-sixth the LLM compute, because the selective calibrator here
is honest (accepted accuracy 0.93 at 0.84 coverage, ECE 0.035).

**The larger LLM is the more robust detector across sources; fine-tuning the 0.5B wins only on
the benchmark it trained on.** The `llm` adapter had never trained on prompt-injection — its
training questions are "Is this a real, reportable security finding?" and the like — so its 0.67
on deepset was zero-shot transfer, and the 8B beat it. Fine-tuning the 0.5B on deepset's own
492-row train split (`llm-v2`) does raise its deepset-test score to 0.966, above the 8B. But a
second, independent PI set from a different source shows that gain does **not** generalize:

| model | params | deepset-test (n=116) | safe-guard-PI (n=400, balanced) | latency |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct (Q4, few-shot) | ~8 B | 0.819 | **0.925** | 165 ms |
| Haruspex `llm` (zero-shot transfer) | ~0.5 B | 0.672 | — | 40 ms |
| Haruspex `llm-v2` (fine-tuned on deepset-train) | ~0.5 B | **0.966** | 0.68 | 40 ms |

Both numbers are paired against the same live 8B: on deepset-test `llm-v2` wins (McNemar
p = 7.6e-5), on [`xTRam1/safe-guard-prompt-injection`](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection)
the 8B wins decisively (p = 1.3e-15). On the full 2060-row safe-guard test split `llm-v2` scores
only **0.591 — below the 0.684 majority-class baseline** — with ECE 0.41. So `llm-v2` did not
learn general prompt-injection detection; it learned deepset's distribution. The 8B holds 0.82–0.93
across both sources; the fine-tuned specialist beats it on its own training distribution and
collapses on a different one.

**The honest conclusion.** This is the same overfitting the code adapter showed: training a 0.5B
on one benchmark's train split can beat a 16×-larger general model *on that benchmark's test
split*, but it is not a real-world capability — it does not transfer to another source of the
same task. For prompt injection specifically, **the larger general LLM is the better production
detector** because it generalizes across sources. A single-source fine-tune should never be
reported as "beats the 8B" without a second independent set; here the second set reverses the
result. `llm-v2` is therefore **not published** (same policy as the code adapter). A genuine
specialist would need training across *multiple* PI sources and validation on a *held-out* source
it never saw — a real multi-source data effort, not a single-benchmark fine-tune.

The earlier cascade finding stands and is consistent: routing low-confidence cases to the 8B
cannot rescue a task the small model has not genuinely learned, because its confidence does not
separate right from wrong off-distribution (mean 0.992 correct vs 0.993 wrong; novelty-p 0.022 vs
0.022). The cascade remains the right tool only where calibration genuinely holds — in-distribution
security triage, where it matches the 8B while calling it on ~16 % of items.

`runs/llm-v2` is reproducible from `scripts/convert_deepset_pi.py` (pointed at each split) plus
`haruspex.train`; the two eval sets are `deepset/prompt-injections` and
`xTRam1/safe-guard-prompt-injection`, neither overlapping the training text.

## Caveats

- The security pack is this project's own data; GUIDE (Microsoft, analyst-graded) is the
  independent one. GUIDE's ceiling with tenant-independent fields is about 0.80, since
  about 20 % of incidents share a summary with differently graded ones.
- Selective-calibration risk guarantees are in-distribution; on a shifted input the real
  risk can exceed the target (the `llm` adapter: 0.2 target, ~0.33 on deepset). The novelty
  gate catches inputs unlike training data, but not a same-shaped harder distribution.
- One 8B model and one prompt format; larger LLMs were not run.
- OTRF Windows-log test (35 rows, 3 tactics) is too small to report as a benchmark.

## Publishing results on the Hugging Face Hub

Per the [eval-results spec](https://huggingface.co/docs/hub/eval-results), scores live in
the model repo under `.eval_results/*.yaml` and link to a **benchmark dataset** that holds
the eval splits and a root `eval.yaml`. `scripts/publish_eval_results.py` writes both from
`runs/<domain>/eval.json`:

```bash
python scripts/publish_eval_results.py --runs runs \
    --dataset-dir ~/datasets/haruspex-bench \
    --eval-results runs/sec-v2-lr1e-4-s2/.eval_results
```

Then, after review, upload by hand (publishing a public dataset is licence-sensitive):

```bash
hf upload-large-folder TyrusRC/haruspex-bench ~/datasets/haruspex-bench --repo-type dataset
hf upload TyrusRC/haruspex runs/sec-v2-lr1e-4-s2/.eval_results .eval_results
```

Three honest caveats about what will render:

- **Leaderboard aggregation needs an allow-listed benchmark.** The dataset's `eval.yaml`
  is validated at push, but HF only aggregates its leaderboard once the dataset is on
  their beta allow-list — you have to ask them.
- **`evaluation_framework` is an HF-maintained enum** (e.g. `inspect-ai`, `math-arena`);
  the generated file uses `custom` as a placeholder, which you must replace with an
  allow-listed id (or get one added) before the dataset validates as a benchmark.
- **A "verified" badge needs the eval to run in HF Jobs with inspect-ai.** Our harness is
  a plain Python one, so these are community (unverified) scores until the eval is ported.
- **OWASP Benchmark rows (GPL) are never shipped** in the dataset; that score is reported
  in prose against the upstream dataset only.

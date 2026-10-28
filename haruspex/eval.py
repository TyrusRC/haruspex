"""Evaluate Haruspex: Accuracy, ECE, Brier, latency, and selective coverage/risk.

Metrics judge capability-per-cost, not a leaderboard. ECE/Brier are computed on
the top-1 confidence vs correctness (confidence calibration). With a calibrator,
the report also shows accepted coverage and the error rate AMONG accepted items
(the number the risk budget controls). `--compare-url` runs the same rows against
any /v1/evaluate or /v1/systemone server for a head-to-head.

  python -m haruspex.eval --model runs/sec --data data/security_seed.jsonl
  python -m haruspex.eval --model runs/sec --data bench.jsonl --compare-url http://127.0.0.1:3001
"""

from __future__ import annotations

import json

import argparse
import math
import time

from .data import gold_label, load_rows


def _ece(conf_correct, bins: int = 10, adaptive: bool = False) -> float:
    """Expected calibration error. `adaptive` uses equal-mass bins, which are less
    biased than equal-width ones when most confidences pile up near 1.0 (and n is small)."""
    n = len(conf_correct)
    if n == 0:
        return 0.0
    if adaptive:
        s = sorted(conf_correct)
        groups = [s[n * b // bins: n * (b + 1) // bins] for b in range(bins)]
    else:
        edges = [i / bins for i in range(bins + 1)]
        groups = [[(c, ok) for c, ok in conf_correct
                   if (c > edges[b] or (b == 0 and c >= edges[b])) and c <= edges[b + 1]]
                  for b in range(bins)]
    ece = 0.0
    for g in groups:
        if g:
            acc = sum(1 for _, ok in g if ok) / len(g)
            ece += (len(g) / n) * abs(acc - sum(c for c, _ in g) / len(g))
    return ece


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n — honest at small n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _mcnemar_p(a_ok, b_ok) -> float:
    """Exact two-sided McNemar p-value for paired correctness: only rows where the two
    models disagree carry evidence about which is better."""
    b = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    c = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _pred_label(verdict: dict):
    return {"boolean": verdict.get("label"), "choice": verdict.get("choice"),
            "score": verdict.get("level")}[verdict["type"]]


def score_rows(answer_fn, rows, per_row: list | None = None) -> dict:
    """Metrics over labeled rows. `per_row`, if given, collects each row's correctness
    (for a paired comparison). Answers from the deterministic resolvers are reported
    under `by_source`, apart from the model's; the selective numbers are the model's,
    since that is what the calibrator gates."""
    correct = n = 0
    brier_sum = 0.0
    cc = []            # (confidence, correct) for ECE
    accepted_ok = accepted = model_n = 0
    by_source: dict[str, list] = {}
    abs_err = []       # score rows: |predicted level - gold level|
    lats = []
    for row in rows:
        gold = gold_label(row)
        t0 = time.perf_counter()
        v = answer_fn(row)
        lats.append((time.perf_counter() - t0) * 1000)
        if gold is None:
            continue
        ok = _pred_label(v) == gold
        conf = float(v.get("confidence") or 0.0)
        correct += int(ok)
        brier_sum += (conf - (1.0 if ok else 0.0)) ** 2   # binary Brier on top-1
        cc.append((conf, ok))
        by_source.setdefault(v.get("source", "model"), []).append(ok)
        if per_row is not None:
            per_row.append(ok)
        if row.get("type") == "score":
            levels = [str(x) for x in row.get("levels") or []]
            pred = v.get("level")
            if pred in levels:
                abs_err.append(abs(levels.index(pred) - levels.index(gold)))
        if v.get("source", "model") == "model":
            model_n += 1
            if v.get("abstain") is False:   # calibrator present and accepted
                accepted += 1
                accepted_ok += int(ok)
        n += 1
    lats.sort()
    lo, hi = _wilson(correct, n)
    out = {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "accuracy_ci95": [round(lo, 4), round(hi, 4)],
        "brier": round(brier_sum / n, 4) if n else 0.0,
        "ece": round(_ece(cc), 4),
        "ece_adaptive": round(_ece(cc, adaptive=True), 4),
        "p50_ms": round(lats[len(lats) // 2], 1) if lats else 0.0,
        "by_source": {s: {"n": len(oks), "accuracy": round(sum(oks) / len(oks), 4)}
                      for s, oks in sorted(by_source.items())},
    }
    if abs_err:
        out["score_mae"] = round(sum(abs_err) / len(abs_err), 4)
    if accepted:
        out["selective"] = {"coverage": round(accepted / model_n, 4),
                            "accepted_accuracy": round(accepted_ok / accepted, 4),
                            "accepted_risk": round(1 - accepted_ok / accepted, 4)}
    return out


def _http_answer_fn(url: str, key: str = "", model: str = "haruspex"):
    import httpx

    def fn(row: dict) -> dict:
        qn = "q"
        crit = row.get("criteria") or row.get("candidates") or row.get("levels")
        q = {qn: {"type": row.get("type", "boolean"),
                  "instructions": row.get("question", ""), "criteria": crit}}
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = httpx.post(url.rstrip("/") + "/v1/evaluate", headers=headers, timeout=60,
                       json={"model": model, "state": row.get("state", ""), "questions": q})
        r.raise_for_status()
        return r.json().get("answers", {}).get(qn, {})

    return fn


def main():
    ap = argparse.ArgumentParser(description="Evaluate Haruspex")
    ap.add_argument("--model", default="", help="model id/dir (default base)")
    ap.add_argument("--data", required=True, help="labeled JSONL")
    ap.add_argument("--device", default="")
    ap.add_argument("--compare-url", default="", help="an /v1/evaluate endpoint to benchmark against")
    ap.add_argument("--compare-key", default="")
    ap.add_argument("--out", default="", help="also write the metrics as JSON (e.g. <model>/eval.json, "
                                              "which the model card reads)")
    ap.add_argument("--name", default="", help="dataset name recorded in --out")
    args = ap.parse_args()

    rows = load_rows(args.data)

    from .config import pick_device
    from .model import Haruspex
    hx = Haruspex.load(args.model, device=pick_device(args.device))
    ours: list = []
    metrics = score_rows(hx.decide, rows, ours)
    print("haruspex:", metrics)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps({**metrics, "test_n": metrics.get("n"),
                                              "dataset": args.name or Path(args.data).stem},
                                             indent=1), encoding="utf-8")
    if args.compare_url:
        theirs: list = []
        print("compare :", score_rows(_http_answer_fn(args.compare_url, args.compare_key), rows, theirs))
        print(f"mcnemar : p={_mcnemar_p(ours, theirs):.4g} (paired, exact; p < 0.05 = a real difference)")


if __name__ == "__main__":
    main()

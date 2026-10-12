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

import argparse
import time

from .data import gold_label, load_rows


def _ece(conf_correct, bins: int = 10) -> float:
    n = len(conf_correct)
    if n == 0:
        return 0.0
    edges = [i / bins for i in range(bins + 1)]
    ece = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = [(c, ok) for c, ok in conf_correct
                  if (c > lo or (b == 0 and c >= lo)) and c <= hi]
        if not in_bin:
            continue
        acc = sum(1 for _, ok in in_bin if ok) / len(in_bin)
        avg_conf = sum(c for c, _ in in_bin) / len(in_bin)
        ece += (len(in_bin) / n) * abs(acc - avg_conf)
    return ece


def _pred_label(verdict: dict):
    return {"boolean": verdict.get("label"), "choice": verdict.get("choice"),
            "score": verdict.get("level")}[verdict["type"]]


def score_rows(answer_fn, rows) -> dict:
    correct = n = 0
    brier_sum = 0.0
    cc = []            # (confidence, correct) for ECE
    accepted_ok = accepted = 0
    lats = []
    for row in rows:
        gold = gold_label(row)
        t0 = time.perf_counter()
        v = answer_fn(row)
        lats.append((time.perf_counter() - t0) * 1000)
        if gold is None:
            continue
        ok = _pred_label(v) == gold
        conf = float(v.get("confidence", 0.0))
        correct += int(ok)
        brier_sum += (conf - (1.0 if ok else 0.0)) ** 2   # binary Brier on top-1
        cc.append((conf, ok))
        if v.get("abstain") is False:   # calibrator present and accepted
            accepted += 1
            accepted_ok += int(ok)
        n += 1
    lats.sort()
    out = {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "brier": round(brier_sum / n, 4) if n else 0.0,
        "ece": round(_ece(cc), 4),
        "p50_ms": round(lats[len(lats) // 2], 1) if lats else 0.0,
    }
    if accepted:
        out["selective"] = {"coverage": round(accepted / n, 4),
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
    args = ap.parse_args()

    rows = load_rows(args.data)

    from .config import pick_device
    from .model import Haruspex
    hx = Haruspex.load(args.model, device=pick_device(args.device))
    print("haruspex:", score_rows(hx.decide, rows))
    if args.compare_url:
        print("compare :", score_rows(_http_answer_fn(args.compare_url, args.compare_key), rows))


if __name__ == "__main__":
    main()

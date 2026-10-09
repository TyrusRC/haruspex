"""Evaluate Haruspex: Accuracy, Brier, ECE, and P50 latency on a labeled JSONL.

The point is to PROVE a win vs OpenJev on the axes that matter — domain accuracy,
calibration (Brier + ECE, which OpenJev does not report), and latency/footprint.
`--compare-url` runs the same rows against any /v1/evaluate endpoint (hosted Jev,
a local OpenJev server) for a head-to-head.

  python -m haruspex.eval --model runs/haruspex-v0 --data data/seed.jsonl
  python -m haruspex.eval --model runs/haruspex-v0 --data bench.jsonl \
      --compare-url http://127.0.0.1:3001   # an OpenJev /v1/evaluate server
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .data import normalize


def _answer_metrics(row: dict, ans: dict) -> tuple[bool, float, float, float]:
    """(correct, brier, confidence, is_correct_for_ece) for one answered row."""
    ex = normalize(row)
    if ex.type == "boolean":
        p = float(ans.get("probability", 0.0))
        y = ex.target[0]
        pred = p >= 0.5
        correct = pred == (y >= 0.5)
        brier = (p - y) ** 2
        conf = max(p, 1 - p)
        return correct, brier, conf, correct
    probs = ans.get("probabilities", {}) or {}
    if ex.type == "choice":
        vec = [float(probs.get(n, 0.0)) for n in ex.names]
    else:  # score -> probabilities keyed "0".."n"
        vec = [float(probs.get(str(i), 0.0)) for i in range(len(ex.names))]
    if not vec or sum(vec) == 0:
        return False, 1.0, 0.0, False
    pred = max(range(len(vec)), key=lambda i: vec[i])
    gold = max(range(len(ex.target)), key=lambda i: ex.target[i])
    correct = pred == gold
    brier = sum((vec[i] - ex.target[i]) ** 2 for i in range(len(vec)))
    return correct, brier, vec[pred], correct


def _ece(conf_correct: list[tuple[float, bool]], bins: int = 10) -> float:
    n = len(conf_correct)
    if n == 0:
        return 0.0
    edges = [i / bins for i in range(bins + 1)]
    ece = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = [(c, ok) for c, ok in conf_correct if (c > lo or (b == 0 and c >= lo)) and c <= hi]
        if not in_bin:
            continue
        acc = sum(1 for _, ok in in_bin if ok) / len(in_bin)
        avg_conf = sum(c for c, _ in in_bin) / len(in_bin)
        ece += (len(in_bin) / n) * abs(acc - avg_conf)
    return ece


def _score_rows(answer_fn, rows) -> dict:
    correct = n = 0
    brier_sum = 0.0
    cc: list[tuple[float, bool]] = []
    lats: list[float] = []
    for row in rows:
        t0 = time.perf_counter()
        ans = answer_fn(row)
        lats.append((time.perf_counter() - t0) * 1000)
        ok, brier, conf, ok_ece = _answer_metrics(row, ans)
        correct += int(ok)
        brier_sum += brier
        cc.append((conf, ok_ece))
        n += 1
    lats.sort()
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "brier": round(brier_sum / n, 4) if n else 0.0,
        "ece": round(_ece(cc), 4),
        "p50_ms": round(lats[len(lats) // 2], 1) if lats else 0.0,
    }


def _http_answer_fn(url: str, key: str = "", model: str = "openjev"):
    import httpx

    def fn(row: dict) -> dict:
        qn = "q"
        q = {qn: {"type": row.get("type", "boolean"),
                  "instructions": row.get("question", ""),
                  "criteria": row.get("criteria") or row.get("candidates") or row.get("levels")}}
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = httpx.post(url.rstrip("/") + "/v1/evaluate", headers=headers, timeout=60,
                       json={"model": model, "state": row.get("state", ""), "questions": q})
        r.raise_for_status()
        return r.json().get("answers", {}).get(qn, {})

    return fn


def main():
    ap = argparse.ArgumentParser(description="Evaluate Haruspex (and compare to OpenJev)")
    ap.add_argument("--model", required=True, help="trained model dir")
    ap.add_argument("--data", required=True, help="labeled JSONL")
    ap.add_argument("--device", default="")
    ap.add_argument("--compare-url", default="", help="an /v1/evaluate endpoint to benchmark against")
    ap.add_argument("--compare-key", default="")
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in Path(args.data).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]

    from .config import pick_device
    from .model import HaruspexModel
    model = HaruspexModel.load(args.model, device=pick_device(args.device))

    print("haruspex:", _score_rows(model.decide, rows))
    if args.compare_url:
        fn = _http_answer_fn(args.compare_url, args.compare_key)
        print("compare :", _score_rows(fn, rows))


if __name__ == "__main__":
    main()

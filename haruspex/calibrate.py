"""Selective calibration — abstain when the model is not sure enough.

GLiNER2.5-Decide returns a top label and a confidence. A raw confidence is not a
risk control: 0.52 and 0.95 both "decide", but one should not. The calibrator
turns confidences into an operator risk budget.

Split-conformal selective prediction (distribution-free): on a held-out
calibration set of (confidence, correct) pairs, pick the LOWEST accept threshold
`t` whose empirical error among accepted items is <= `target_risk`, breaking ties
toward higher coverage. At inference: accept when confidence >= t, else ABSTAIN.
The accepted-set error is then controlled at `target_risk` on exchangeable data.

NOTE (ceiling + upgrade path): this is a selective-prediction guarantee on the
top-1 prediction, not a full conformal prediction SET ("the answer is one of
{a,b}"). Prediction sets need per-label scores, which GLiNER2's top_k/multi_label
schema path does not yet return (processor IndexError). When that is fixed, add
LAC/APS set construction here; the selective layer stays as the abstention gate.
"""

from __future__ import annotations

import json
from pathlib import Path


class SelectiveCalibrator:
    def __init__(self, threshold: float, target_risk: float,
                 achieved_risk: float = 0.0, coverage: float = 0.0, n: int = 0):
        self.threshold = threshold
        self.target_risk = target_risk
        self.achieved_risk = achieved_risk
        self.coverage = coverage
        self.n = n

    @classmethod
    def fit(cls, pairs, target_risk: float = 0.1) -> "SelectiveCalibrator":
        """pairs: iterable of (confidence, correct_bool). Returns a calibrator
        that accepts the most items while holding accepted error <= target_risk.
        Falls back to the most conservative threshold if no threshold meets it."""
        data = [(float(c), bool(ok)) for c, ok in pairs]
        n = len(data)
        if n == 0:
            return cls(1.0, target_risk, 0.0, 0.0, 0)
        # Candidate thresholds = the observed confidences (accept conf >= t).
        best = None  # (coverage, -threshold) maximized
        chosen = None
        for t in sorted({c for c, _ in data}):
            acc = [ok for c, ok in data if c >= t]
            if not acc:
                continue
            risk = 1.0 - sum(acc) / len(acc)
            if risk <= target_risk:
                cov = len(acc) / n
                key = (cov, -t)
                if best is None or key > best:
                    best = key
                    chosen = (t, risk, cov, len(acc))
        if chosen is None:
            # No threshold hits the budget — accept only the single most confident.
            top = max(data, key=lambda cr: cr[0])
            acc = [ok for c, ok in data if c >= top[0]]
            chosen = (top[0], 1.0 - sum(acc) / len(acc), len(acc) / n, len(acc))
        t, risk, cov, _ = chosen
        return cls(round(t, 6), target_risk, round(risk, 6), round(cov, 6), n)

    def apply(self, confidence: float) -> dict:
        """Gate a confidence: accept or abstain against the fitted threshold."""
        accept = float(confidence) >= self.threshold
        return {"accept": accept, "abstain": not accept,
                "accept_threshold": self.threshold, "target_risk": self.target_risk}

    # ---- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {"threshold": self.threshold, "target_risk": self.target_risk,
                "achieved_risk": self.achieved_risk, "coverage": self.coverage, "n": self.n}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SelectiveCalibrator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["threshold"], d["target_risk"],
                   d.get("achieved_risk", 0.0), d.get("coverage", 0.0), d.get("n", 0))


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Fit a selective calibrator on labeled rows")
    ap.add_argument("--model", default="", help="model id/dir (default base)")
    ap.add_argument("--data", required=True, help="labeled JSONL (held-out calibration split)")
    ap.add_argument("--out", default="", help="write calibrator.json here (default beside model)")
    ap.add_argument("--target-risk", type=float, default=0.1)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    from .config import pick_device, resolve_model
    from .data import gold_label, load_rows
    from .model import Haruspex

    model_id = resolve_model(args.model)
    hx = Haruspex.load(model_id, device=pick_device(args.device))
    pairs = []
    for row in load_rows(args.data):
        gold = gold_label(row)
        if gold is None:
            continue
        v = hx.decide(row)
        pred = {"boolean": v.get("label"), "choice": v.get("choice"),
                "score": v.get("level")}[v["type"]]
        pairs.append((v["confidence"], pred == gold))
    cal = SelectiveCalibrator.fit(pairs, target_risk=args.target_risk)
    out = args.out or str(Path(model_id) / "calibrator.json")
    cal.save(out)
    print(f"[calibrate] {cal.to_dict()} -> {out}")


if __name__ == "__main__":
    main()

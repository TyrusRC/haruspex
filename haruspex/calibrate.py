"""Selective calibration — abstain when the model is not sure enough.

GLiNER2.5-Decide returns a top label and a confidence. A raw confidence is not a
risk control: 0.52 and 0.95 both "decide", but one should not. The calibrator
turns confidences into an operator risk budget.

Learn-then-Test (Angelopoulos et al. 2021) over a FIXED grid of accept thresholds:
for each threshold t, test H0 "risk among items with confidence >= t exceeds
`target_risk`" with an exact binomial p-value on a held-out calibration set, and
Bonferroni-correct over the grid. Every rejected H0 is a certified threshold; the
lowest one (most coverage) is chosen. Guarantee, on exchangeable data:
P(risk among accepted > target_risk) <= delta. If no threshold certifies, the
calibrator FAILS CLOSED and abstains on everything — too little evidence is not a
licence to guess.

Thresholds are fitted per decision type: a 2-way softmax never drops below 0.5 while
a k-way one reaches 1/k, so one shared cutoff means something different per type. A
type with no calibration rows abstains.

Temperature scaling (Guo et al. 2017) of the boolean P(yes) is optional and fitted on
a SEPARATE split (`--temp-data`): fitting it on the calibration split would make the
threshold's data dependent and void the guarantee.

NOTE (ceilings + upgrade paths):
- Bonferroni over 14 grid points is conservative; fixed-sequence testing on a finer
  grid buys coverage once calibration sets are larger.
- Temperature scaling covers boolean only: classify_text returns just the top label's
  confidence, which pins the full distribution for 2 labels but not for k > 2.
- This is selective prediction on the top-1 label, not a conformal prediction SET.
  Sets need per-label scores, which gliner2's classify_text path does not return.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# Fixed before seeing any data (the guarantee needs that); 0.0 = accept everything.
GRID = (0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(Binomial(n, p) <= k), in log space so large n cannot overflow."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                                 + i * lp + (n - i) * lq) for i in range(k + 1)))


def _fit_threshold(data, target_risk: float, delta: float) -> dict:
    """Lowest grid threshold whose risk <= target_risk is certified at level delta."""
    n = len(data)
    for t in GRID:   # ascending: the first certified threshold has the most coverage
        acc = [ok for c, ok in data if c >= t]
        if not acc:
            continue
        errors = len(acc) - sum(acc)
        if _binom_cdf(errors, len(acc), target_risk) <= delta / len(GRID):
            return {"threshold": t, "achieved_risk": round(errors / len(acc), 6),
                    "coverage": round(len(acc) / n, 6), "n": n}
    return {"threshold": None, "achieved_risk": 0.0, "coverage": 0.0, "n": n}


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def fit_temperature(pairs) -> float:
    """pairs: (p_yes, is_yes) for boolean rows. The T minimising NLL of sigmoid(z/T)."""
    data = [(_logit(float(p)), bool(y)) for p, y in pairs]
    if not data:
        return 1.0

    def nll(T):
        return -sum(math.log(max(_sigmoid(z / T) if y else 1 - _sigmoid(z / T), 1e-12))
                    for z, y in data)

    return min((math.exp(x / 50) for x in range(-150, 151)), key=nll)   # T in [0.05, 20]


class SelectiveCalibrator:
    def __init__(self, threshold: float | None, target_risk: float,
                 achieved_risk: float = 0.0, coverage: float = 0.0, n: int = 0,
                 delta: float = 0.1, by_type: dict | None = None, temperature: float = 1.0):
        # `threshold` gates every type when `by_type` is None (legacy single-threshold
        # files); None means "accept nothing".
        self.threshold = threshold
        self.target_risk = target_risk
        self.achieved_risk = achieved_risk
        self.coverage = coverage
        self.n = n
        self.delta = delta
        self.by_type = by_type
        self.temperature = temperature

    @classmethod
    def fit(cls, pairs, target_risk: float = 0.1, delta: float = 0.1) -> "SelectiveCalibrator":
        """pairs: (confidence, correct) or (confidence, correct, type). Typed pairs get
        one threshold per type; untyped ones a single threshold."""
        data = [tuple(p) for p in pairs]
        if data and len(data[0]) == 3:
            groups: dict[str, list] = {}
            for c, ok, t in data:
                groups.setdefault(t, []).append((float(c), bool(ok)))
            by_type = {t: _fit_threshold(g, target_risk, delta) for t, g in groups.items()}
            n = len(data)
            cov = sum(f["coverage"] * f["n"] for f in by_type.values()) / n
            return cls(None, target_risk, coverage=round(cov, 6), n=n, delta=delta, by_type=by_type)
        f = _fit_threshold([(float(c), bool(ok)) for c, ok in data], target_risk, delta)
        return cls(f["threshold"], target_risk, f["achieved_risk"], f["coverage"], f["n"], delta)

    def scale(self, p_yes: float) -> float:
        """Temperature-scaled boolean P(yes)."""
        return p_yes if self.temperature == 1.0 else _sigmoid(_logit(p_yes) / self.temperature)

    def apply(self, confidence: float, dtype: str | None = None) -> dict:
        """Gate a confidence for a decision type: accept or abstain."""
        if self.by_type is None:
            thr = self.threshold
        else:
            thr = (self.by_type.get(dtype) or {}).get("threshold")
        accept = thr is not None and float(confidence) >= thr
        return {"accept": accept, "abstain": not accept,
                "accept_threshold": thr, "target_risk": self.target_risk}

    # ---- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        d = {"threshold": self.threshold, "target_risk": self.target_risk,
             "achieved_risk": self.achieved_risk, "coverage": self.coverage, "n": self.n,
             "delta": self.delta, "temperature": self.temperature}
        if self.by_type is not None:
            d["by_type"] = self.by_type
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SelectiveCalibrator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["threshold"], d["target_risk"], d.get("achieved_risk", 0.0),
                   d.get("coverage", 0.0), d.get("n", 0), d.get("delta", 0.1),
                   d.get("by_type"), d.get("temperature", 1.0))


def _model_rows(hx, path):
    """(verdict, gold) for rows the MODEL answered — resolver answers bypass the gate."""
    from .data import gold_label, load_rows
    for row in load_rows(path):
        gold = gold_label(row)
        if gold is None:
            continue
        v = hx.decide(row)
        if v.get("source") == "model":
            yield v, gold


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Fit a selective calibrator on labeled rows")
    ap.add_argument("--model", default="", help="model id/dir (default base)")
    ap.add_argument("--data", required=True,
                    help="labeled JSONL (held-out calibration split, never used for tuning)")
    ap.add_argument("--temp-data", default="",
                    help="optional labeled JSONL to fit boolean temperature on (a split "
                         "other than --data, e.g. the dev split)")
    ap.add_argument("--out", default="", help="write calibrator.json here (default beside model)")
    ap.add_argument("--target-risk", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.1,
                    help="allowed probability that accepted risk exceeds --target-risk")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    from .config import pick_device, resolve_model
    from .eval import _pred_label
    from .model import Haruspex

    model_id = resolve_model(args.model)
    hx = Haruspex.load(model_id, device=pick_device(args.device))
    hx.calibrator = None   # fit on raw model output, not on a previous calibrator's
    T = 1.0
    if args.temp_data:
        T = fit_temperature((v["probability"], g == "yes")
                            for v, g in _model_rows(hx, args.temp_data) if v["type"] == "boolean")
    scaler = SelectiveCalibrator(None, args.target_risk, temperature=T)
    pairs = []
    for v, gold in _model_rows(hx, args.data):
        conf = v["confidence"]
        if v["type"] == "boolean":
            p = scaler.scale(v["probability"])
            conf = max(p, 1 - p)
        pairs.append((conf, _pred_label(v) == gold, v["type"]))
    cal = SelectiveCalibrator.fit(pairs, target_risk=args.target_risk, delta=args.delta)
    cal.temperature = T
    out = args.out or str(Path(model_id) / "calibrator.json")
    cal.save(out)
    print(f"[calibrate] {json.dumps(cal.to_dict())} -> {out}")
    for t, f in (cal.by_type or {}).items():
        if f["threshold"] is None:
            print(f"[calibrate] WARNING: {t}: {f['n']} rows cannot certify risk "
                  f"<= {args.target_risk} at delta {args.delta}; {t} questions will abstain")


if __name__ == "__main__":
    main()

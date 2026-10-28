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
type with no calibration rows abstains. `--pooled` fits one threshold instead, when the
calibration split is too small to certify each type.

Temperature scaling (Guo et al. 2017) of the per-label logits, for every decision type,
is optional and fitted on a SEPARATE split (`--temp-data`): fitting it on the
calibration split would make the threshold's data dependent and void the guarantee.

Prediction sets (split conformal, LAC score 1 - p(true label); Sadinle et al. 2019) are
fitted on the same calibration split: P(true label in set) >= 1 - set_alpha, marginally,
on exchangeable data. Where the top-1 gate abstains, the set still says "it is one of
these", which is often enough to act on.

NOTE (ceilings + upgrade paths):
- Bonferroni over 14 grid points is conservative; fixed-sequence testing on a finer
  grid buys coverage once calibration sets are larger.
- One temperature shared across decision types; per-type temperatures need a larger
  temperature split.
- Set coverage is marginal over the calibration mix, not conditional per input.
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


def fit_temperature(pairs) -> float:
    """pairs: (member logits [{label: logit}, ...], gold label). The T minimising the NLL
    of the gold label under the members' mean softmax(z / T)."""
    from .scorer import ensemble_probs
    data = [(m, g) for m, g in pairs if m and all(x is not None and g in x for x in m)]
    if not data:
        return 1.0

    def nll(T):
        return -sum(math.log(max(ensemble_probs(m, T)[g], 1e-12)) for m, g in data)

    return min((math.exp(x / 50) for x in range(-150, 151)), key=nll)   # T in [0.05, 20]


def _fit_set(true_probs, alpha: float) -> float | None:
    """Split-conformal LAC threshold: q-hat over scores 1 - p(true). None when the
    calibration rows are too few to certify 1 - alpha coverage."""
    scores = sorted(1.0 - float(p) for p in true_probs)
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return None if n == 0 or k > n else round(scores[k - 1], 6)


class SelectiveCalibrator:
    def __init__(self, threshold: float | None, target_risk: float,
                 achieved_risk: float = 0.0, coverage: float = 0.0, n: int = 0,
                 delta: float = 0.1, by_type: dict | None = None, temperature: float = 1.0,
                 temperature_scope: str = "all", set_q: float | None = None,
                 set_alpha: float | None = None):
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
        # Files written before full logits existed fitted T on boolean P(yes) only.
        self.temperature_scope = temperature_scope
        self.set_q = set_q          # pooled prediction-set threshold (by_type holds per-type)
        self.set_alpha = set_alpha

    def temperature_for(self, dtype: str) -> float:
        return self.temperature if self.temperature_scope == "all" or dtype == "boolean" else 1.0

    def prediction_set(self, probs: dict, dtype: str | None = None) -> list | None:
        """Labels whose probability clears the conformal threshold, most likely first;
        None when no set was fitted for this type."""
        q = self.set_q if self.by_type is None else (self.by_type.get(dtype) or {}).get("set_q")
        if q is None:
            return None
        keep = [k for k, p in probs.items() if 1.0 - p <= q + 1e-12]
        return sorted(keep or [max(probs, key=probs.get)], key=lambda k: -probs[k])

    def fit_sets(self, true_probs, alpha: float) -> None:
        """true_probs: p(true label) per cal row, or (p, type) pairs for per-type sets."""
        data = list(true_probs)
        self.set_alpha = alpha
        if data and isinstance(data[0], tuple):
            groups: dict = {}
            for p, t in data:
                groups.setdefault(t, []).append(p)
            self.by_type = self.by_type if self.by_type is not None else {}
            for t, g in groups.items():
                self.by_type.setdefault(t, {})["set_q"] = _fit_set(g, alpha)
        else:
            self.set_q = _fit_set(data, alpha)

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
             "delta": self.delta, "temperature": self.temperature,
             "temperature_scope": self.temperature_scope, "set_q": self.set_q,
             "set_alpha": self.set_alpha}
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
                   d.get("by_type"), d.get("temperature", 1.0),
                   d.get("temperature_scope", "boolean"), d.get("set_q"), d.get("set_alpha"))


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
                    help="optional labeled JSONL to fit the logit temperature on (a split "
                         "other than --data, e.g. the dev split)")
    ap.add_argument("--set-alpha", type=float, default=0.1,
                    help="prediction sets miss the true label at most this often")
    ap.add_argument("--out", default="", help="write calibrator.json here (default beside model)")
    ap.add_argument("--target-risk", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.1,
                    help="allowed probability that accepted risk exceeds --target-risk")
    ap.add_argument("--pooled", action="store_true",
                    help="one threshold across all decision types: the guarantee then holds "
                         "on average over the type mix, not per type. Use it when the cal "
                         "split is too small to certify each type on its own")
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
        from .data import gold_label, load_rows
        pairs = []
        for row in load_rows(args.temp_data):
            gold = gold_label(row)
            if gold is not None and hx.decide(row).get("source") == "model":
                pairs.append((hx.member_logits(row), gold))
        T = fit_temperature(pairs)
    # Apply the fitted temperature while reading the cal split; no threshold yet.
    hx.calibrator = SelectiveCalibrator(None, args.target_risk, temperature=T)
    pairs, true_probs = [], []
    for v, gold in _model_rows(hx, args.data):
        ok = _pred_label(v) == gold
        pairs.append((v["confidence"], ok) if args.pooled else (v["confidence"], ok, v["type"]))
        p_true = (v.get("probabilities") or {}).get(gold, 0.0)
        true_probs.append(p_true if args.pooled else (p_true, v["type"]))
    cal = SelectiveCalibrator.fit(pairs, target_risk=args.target_risk, delta=args.delta)
    cal.temperature = T
    cal.fit_sets(true_probs, args.set_alpha)
    out = args.out or str(Path(model_id) / "calibrator.json")
    cal.save(out)
    print(f"[calibrate] {json.dumps(cal.to_dict())} -> {out}")
    for t, f in (cal.by_type or {}).items():
        if f["threshold"] is None:
            print(f"[calibrate] WARNING: {t}: {f['n']} rows cannot certify risk "
                  f"<= {args.target_risk} at delta {args.delta}; {t} questions will abstain")


if __name__ == "__main__":
    main()

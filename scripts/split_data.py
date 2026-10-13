"""Deterministic, stratified train/test split for a decision dataset.

Stratifies by (side, category, type) so the held-out set covers every category,
and is seeded so the split is reproducible. The test set is never trained on — it
is the only honest source of a generalization number.

  python scripts/split_data.py data/security.jsonl --test-frac 0.2
  # -> data/security_train.jsonl + data/security_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haruspex.data import load_rows  # noqa: E402


def split(rows, test_frac: float, seed: int, cal_frac: float = 0.0):
    """Stratified split by (side, category, type). With cal_frac > 0 it is a
    three-way train/cal/test split; the calibrator is fit on cal and test stays
    clean (fitting a selective threshold on train gives a false guarantee)."""
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r.get("side", "?"), r.get("category", "?"), r.get("type", "?"))].append(r)
    rng = random.Random(seed)
    train, cal, test = [], [], []
    for key in sorted(groups):
        g = groups[key][:]
        rng.shuffle(g)
        n = len(g)
        n_test = max(1, round(n * test_frac)) if n >= 2 else 0
        n_cal = max(1, round(n * cal_frac)) if (cal_frac > 0 and n - n_test >= 2) else 0
        test.extend(g[:n_test])
        cal.extend(g[n_test:n_test + n_cal])
        train.extend(g[n_test + n_cal:])
    for part in (train, cal, test):
        rng.shuffle(part)
    return train, cal, test


def main():
    ap = argparse.ArgumentParser(description="Stratified train/(cal)/test split")
    ap.add_argument("path")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--cal-frac", type=float, default=0.0,
                    help="held-out calibration fraction (>0 writes a *_cal.jsonl)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.path)
    train, cal, test = split(rows, args.test_frac, args.seed, args.cal_frac)
    stem = Path(args.path).with_suffix("")
    parts = [(f"{stem}_train.jsonl", train), (f"{stem}_test.jsonl", test)]
    if cal:
        parts.append((f"{stem}_cal.jsonl", cal))
    for name, part in parts:
        with open(name, "w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tail = f" + cal {len(cal)}" if cal else ""
    print(f"[split] {len(rows)} -> train {len(train)}{tail} + test {len(test)}")


if __name__ == "__main__":
    main()
